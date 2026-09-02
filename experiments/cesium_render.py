#!/usr/bin/env python3
"""Render a Cesium ion scene at a given camera pose, for scene matching.

Produces, for one pose, the two things a scene matcher needs:
  <out>.png   the rendered view
  <out>.json  the exact camera used, plus a sparse depth grid sampled
              with scene.pickPositionWorldCoordinates (ECEF -> lat/lon/
              height), so every sampled pixel carries a 3D point and PnP
              becomes possible against the photograph.

The camera convention matches skyfix.py: heading in degrees true, pitch
positive UP, roll positive right-wing-down, and `fov` the HORIZONTAL
field of view, so a render can be dropped in beside a photograph
without re-deriving anything.

NETWORK: Cesium ion is a streaming service and this container's egress
policy denies cesium.com, api.cesium.com and assets.ion.cesium.com (the
proxy answers 403 to CONNECT). Like fetch_osm_landmarks.py, this script
is meant to run on a machine with normal network access; commit the
resulting PNG/JSON pairs and everything downstream works offline.

    pip install playwright && playwright install chromium
    export CESIUM_ION_TOKEN=...
    python3 cesium_render.py --lat 37.01992 --lon 27.44426 --height 9 \
        --heading 167.09 --fov 73.7 --out out/cesium/BD9

Asset ids: 2275207 = Google Photorealistic 3D Tiles, 1 = Cesium World
Terrain, 96188 = Cesium OSM Buildings. --asset may be repeated; tilesets
are added in order over the terrain.
"""

import argparse
import base64
import json
import os
import sys

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>html,body,#c{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#000}</style>
</head><body><div id="c"></div><script>
window.CESIUM_BASE_URL =
  "https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/";
window.__ready = false; window.__error = null; window.__depth = null;
(async () => {
  try {
    Cesium.Ion.defaultAccessToken = "__TOKEN__";
    const viewer = new Cesium.Viewer("c", {
      // a bare canvas: every widget is chrome that would pollute the render
      animation:false, baseLayerPicker:false, fullscreenButton:false,
      geocoder:false, homeButton:false, infoBox:false, sceneModePicker:false,
      selectionIndicator:false, timeline:false, navigationHelpButton:false,
      creditContainer: document.createElement("div"),
      terrain: __TERRAIN__,
      contextOptions: { webgl: { preserveDrawingBuffer: true } },
    });
    window.__viewer = viewer;
    const s = viewer.scene;
    s.globe.depthTestAgainstTerrain = true;
    s.skyAtmosphere.show = __ATMOSPHERE__;
    s.fog.enabled = false;
    s.globe.enableLighting = false;

    for (const id of __ASSETS__) {
      const ts = await Cesium.Cesium3DTileset.fromIonAssetId(id);
      s.primitives.add(ts);
    }

    // the camera: heading/pitch/roll in the local ENU frame, pitch
    // positive UP to match skyfix, and the HORIZONTAL fov pinned by
    // setting the frustum's aspect-derived vertical angle
    const pos = Cesium.Cartesian3.fromDegrees(__LON__, __LAT__, __HEIGHT__);
    viewer.camera.setView({
      destination: pos,
      orientation: {
        heading: Cesium.Math.toRadians(__HEADING__),
        pitch: Cesium.Math.toRadians(__PITCH__),
        roll: Cesium.Math.toRadians(__ROLL__),
      },
    });
    const w = s.canvas.clientWidth, h = s.canvas.clientHeight;
    const fovx = Cesium.Math.toRadians(__FOV__);
    // Cesium's PerspectiveFrustum.fov is the field of view in the WIDER
    // direction, so for a landscape canvas it already is the horizontal
    // one; set it explicitly and assert the aspect
    viewer.camera.frustum.fov = fovx;
    viewer.camera.frustum.aspectRatio = w / h;

    // wait until the tiles actually stop loading, not merely a fixed
    // delay: a screenshot of a half-streamed tileset is worthless
    let quiet = 0;
    const t0 = Date.now();
    await new Promise((resolve) => {
      const tick = s.postRender.addEventListener(() => {
        const pending = s.globe.tilesLoaded ? 0 : 1;
        let tiles = 0;
        for (let i = 0; i < s.primitives.length; i++) {
          const p = s.primitives.get(i);
          if (p && p.tilesLoaded === false) tiles++;
        }
        if (pending === 0 && tiles === 0) quiet++; else quiet = 0;
        if (quiet > 60 || Date.now() - t0 > __TIMEOUT__) {
          tick(); resolve();
        }
      });
    });

    // sparse depth: pick the world position under a grid of pixels so
    // each sampled pixel carries a 3D point for PnP
    const N = __DEPTH_N__, depth = [];
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x = ((i + 0.5) / N) * w, y = ((j + 0.5) / N) * h;
        const c = s.pickPositionWorldCoordinates(new Cesium.Cartesian2(x, y));
        if (!c) continue;
        const g = Cesium.Cartographic.fromCartesian(c);
        if (!g) continue;
        depth.push([x, y,
                    Cesium.Math.toDegrees(g.latitude),
                    Cesium.Math.toDegrees(g.longitude),
                    g.height,
                    Cesium.Cartesian3.distance(c, pos)]);
      }
    }
    window.__depth = depth;
    window.__ready = true;
  } catch (e) {
    window.__error = String(e && e.stack ? e.stack : e);
    window.__ready = true;
  }
})();
</script></body></html>
"""


def _default_chromium():
    """The image ships a chromium under PLAYWRIGHT_BROWSERS_PATH whose
    build number rarely matches the pip-installed playwright's pin, and
    re-downloading is blocked; point at it directly when it is there."""
    root = os.environ.get('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
    for rel in ('chromium/chrome-linux/chrome',
                'chromium-1194/chrome-linux/chrome'):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    import glob as _g
    for p in sorted(_g.glob(os.path.join(root, 'chromium*/chrome-linux/chrome'))):
        return p
    return None


def build_html(a, token):
    terrain = ('Cesium.Terrain.fromWorldTerrain()' if a.world_terrain
               else 'undefined')
    return (HTML
            .replace('__VER__', a.cesium_version)
            .replace('__TOKEN__', token)
            .replace('__TERRAIN__', terrain)
            .replace('__ATMOSPHERE__', 'true' if a.atmosphere else 'false')
            .replace('__ASSETS__', json.dumps([int(x) for x in a.asset]))
            .replace('__LAT__', repr(a.lat)).replace('__LON__', repr(a.lon))
            .replace('__HEIGHT__', repr(a.height))
            .replace('__HEADING__', repr(a.heading))
            .replace('__PITCH__', repr(a.pitch)).replace('__ROLL__', repr(a.roll))
            .replace('__FOV__', repr(a.fov))
            .replace('__TIMEOUT__', repr(int(a.tile_timeout * 1000)))
            .replace('__DEPTH_N__', repr(a.depth_grid)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--height', type=float, required=True,
                    help='metres above the WGS84 ellipsoid, NOT above ground')
    ap.add_argument('--heading', type=float, required=True, help='deg true')
    ap.add_argument('--pitch', type=float, default=0.0, help='deg, +up')
    ap.add_argument('--roll', type=float, default=0.0)
    ap.add_argument('--fov', type=float, default=73.7, help='HORIZONTAL, deg')
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height-px', type=int, default=1200)
    ap.add_argument('--asset', action='append', default=[],
                    help='ion asset id; repeatable. 2275207 = Google '
                         'Photorealistic 3D Tiles, 96188 = OSM Buildings')
    ap.add_argument('--world-terrain', action='store_true',
                    help='drape over Cesium World Terrain (asset 1)')
    ap.add_argument('--atmosphere', action='store_true',
                    help='render the sky atmosphere; off by default so the '
                         'sky/terrain boundary stays a hard edge for the '
                         'skyline extractor')
    ap.add_argument('--cesium-version', default='1.120')
    ap.add_argument('--tile-timeout', type=float, default=120.0)
    ap.add_argument('--depth-grid', type=int, default=64,
                    help='NxN pixel grid to sample world positions at')
    ap.add_argument('--token', default=os.environ.get('CESIUM_ION_TOKEN'))
    ap.add_argument('--out', required=True, help='path stem: writes .png/.json')
    ap.add_argument('--keep-html', action='store_true')
    ap.add_argument('--chromium', default=_default_chromium(),
                    help='chromium executable; defaults to the one already '
                         'in PLAYWRIGHT_BROWSERS_PATH when playwright\'s own '
                         'pinned build is absent')
    a = ap.parse_args()

    if not a.token:
        sys.exit('no ion token: pass --token or set CESIUM_ION_TOKEN')
    if not a.asset and not a.world_terrain:
        a.asset = [2275207]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit('pip install playwright && playwright install chromium')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    html = build_html(a, a.token)
    if a.keep_html:
        with open(a.out + '.html', 'w') as f:
            f.write(html)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=a.chromium, args=[
            # a real GL context: Cesium will not render on SwiftShader's
            # default blocklist without these
            '--use-gl=angle', '--use-angle=swiftshader',
            '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
        ])
        page = browser.new_page(
            viewport={'width': a.width, 'height': a.height_px},
            device_scale_factor=1)
        errs = []
        page.on('pageerror', lambda e: errs.append(str(e)))
        page.set_content(html, wait_until='load')
        page.wait_for_function('window.__ready === true',
                               timeout=int((a.tile_timeout + 90) * 1000))
        err = page.evaluate('window.__error')
        if err:
            browser.close()
            sys.exit('cesium failed:\n' + err)
        depth = page.evaluate('window.__depth') or []
        page.locator('canvas').first.screenshot(path=a.out + '.png')
        browser.close()

    meta = {
        'lat': a.lat, 'lon': a.lon, 'height_m': a.height,
        'heading_deg': a.heading, 'pitch_deg': a.pitch, 'roll_deg': a.roll,
        'fov_deg': a.fov, 'width': a.width, 'height_px': a.height_px,
        'assets': a.asset, 'world_terrain': a.world_terrain,
        'atmosphere': a.atmosphere, 'cesium_version': a.cesium_version,
        'depth_grid': a.depth_grid,
        'depth_cols': ['px', 'py', 'lat', 'lon', 'height_m', 'range_m'],
        'depth': depth,
        'page_errors': errs,
    }
    with open(a.out + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote %s.png (%dx%d) and %s.json (%d depth samples)'
          % (a.out, a.width, a.height_px, a.out, len(depth)))


if __name__ == '__main__':
    main()
