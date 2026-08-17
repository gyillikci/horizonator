#!/usr/bin/env python3
"""Curate a folder of Theodolite (iPhone) sightings into a dataset.

The field procedure produces, per sighting, either one plain photo or a
PAIR: the same frame with and without Theodolite's HUD burned in
(attitude, GPS, altitude, lens zoom). The HUD carries the numbers the
instrument needs; the clean frame is what the skyline extractor should
see — an overlay crosshair sitting on the horizon is a fake edge.

This tool does the mechanical half of curation:

  inventory   every image, HEIC included, with its EXIF (time, GPS,
              heading, 35mm-equivalent focal length, lens, size)
  classify    HUD-overlaid vs clean, from the extra high-frequency
              text/graticule structure the HUD adds
  pair        clean <-> HUD frames of one sighting, matched by capture
              time and by actually looking the same (thumbnail
              correlation), so bursts don't cross-pair
  qc          sharpness, exposure, and — the decisive one — whether the
              sea horizon detector (E4q-2) finds a horizon, with the
              pitch/roll it yields; that is what makes a sighting
              usable for auto-levelled fixing
  crop        the HUD text bands, written out for transcription

The remaining half is reading the HUD numbers. No OCR is installed
here, and HUD text is small, so the flow is: this tool writes
out/theodolite/crops/<id>_hud_{top,bottom}.png, those get read, and the
values go into a transcription JSON keyed by sighting id:

    {"IMG_0123": {"azimuth_deg": 213.4, "pitch_deg": -1.2,
                  "roll_deg": 0.3, "lat": 37.0312, "lon": 27.4188,
                  "alt_m": 12.0, "zoom": 2.0}}

Feed it back with --transcript to get validated per-sighting metadata
(GPS cross-checked against EXIF, FOV cross-checked against the zoom
factor) plus ready-to-run skyfix command lines.

Usage:
    python3 theodolite_curate.py DIR [--out OUTDIR] [--horizon]
    python3 theodolite_curate.py DIR --transcript hud.json
"""

import os
import re
import sys
import json
import glob
import argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ExifTags

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff')
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
SENSOR_35MM_W = 36.0        # full-frame width, for the FOV conversion


def _rat(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        try:
            return float(v[0]) / float(v[1])
        except Exception:
            return None


def read_meta(path):
    """EXIF fields that matter, normalised. iPhone writes
    FocalLengthIn35mmFilm per captured lens, which is what the FOV
    comes from; Theodolite may also write GPSImgDirection."""
    out = dict(path=path, name=os.path.basename(path))
    with Image.open(path) as im:
        out['width'], out['height'] = im.size
        exif = im.getexif()
        if not exif:
            return out
        def g(tag, src=exif):
            return src.get(TAGS.get(tag, -1))
        out['make'] = str(g('Make') or '').strip()
        out['model'] = str(g('Model') or '').strip()
        ifd = exif.get_ifd(0x8769) if hasattr(exif, 'get_ifd') else {}
        def ge(tag):
            return ifd.get(TAGS.get(tag, -1))
        dt = ge('DateTimeOriginal') or g('DateTime')
        if dt:
            try:
                out['time'] = datetime.strptime(str(dt),
                                                '%Y:%m:%d %H:%M:%S')
                sub = ge('SubsecTimeOriginal')
                out['t'] = out['time'].timestamp() + (
                    float('0.' + str(sub)) if sub else 0.0)
                out['time'] = out['time'].isoformat()
            except ValueError:
                pass
        f35 = ge('FocalLengthIn35mmFilm')
        if f35:
            out['focal35_mm'] = float(f35)
            out['fov_deg'] = float(np.degrees(
                2 * np.arctan(SENSOR_35MM_W / (2 * float(f35)))))
        fl = _rat(ge('FocalLength'))
        if fl:
            out['focal_mm'] = fl
        lens = ge('LensModel')
        if lens:
            out['lens'] = str(lens)
        zoom = _rat(ge('DigitalZoomRatio'))
        if zoom:
            out['zoom'] = zoom
        out['sub'] = str(ge('SubsecTimeOriginal') or '')
        out['dt_raw'] = str(dt or '')
        sw = str(g('Software') or '')
        out['software'] = sw
        out['theodolite'] = 'Theodolite' in sw
        out.update(theodolite_meta(exif, ifd))
        gps = exif.get_ifd(0x8825) if hasattr(exif, 'get_ifd') else {}
        if gps:
            def gg(tag):
                return gps.get(GPSTAGS.get(tag, -1))
            for key, ref_tag, tag in (('lat', 'GPSLatitudeRef',
                                       'GPSLatitude'),
                                      ('lon', 'GPSLongitudeRef',
                                       'GPSLongitude')):
                v = gg(tag)
                if v:
                    d, m, s = [_rat(x) for x in v]
                    val = d + m / 60.0 + s / 3600.0
                    if str(gg(ref_tag) or '').upper() in ('S', 'W'):
                        val = -val
                    out[key] = val
            alt = _rat(gg('GPSAltitude'))
            if alt is not None:
                if gg('GPSAltitudeRef') in (1, b'\x01'):
                    alt = -alt
                out['alt_m'] = alt
            hd = _rat(gg('GPSImgDirection'))
            if hd is not None:
                out['heading_deg'] = hd
                out['heading_ref'] = str(gg('GPSImgDirectionRef') or '')
    return out


def theodolite_meta(exif, ifd):
    """Theodolite's own record, which makes every heuristic in this
    file a fallback rather than the method.

    The app writes the sighting into EXIF twice: a human line in
    ImageDescription ('vert_angle_deg=-0.4 / horiz_angle_deg=-0.2')
    and the full XML in the MakerNote, which also carries the
    ACCURACIES iOS reported at the moment of capture — GPS horizontal
    and vertical in meters, and the compass azimuth accuracy in
    degrees. That last one matters more than it looks: it is routinely
    +-10 to +-20 deg, which is the width the heading prior deserves,
    not the single GPSImgDirection number.

    The two saved versions of one sighting (HUD screen capture and
    clean camera frame) carry IDENTICAL angles and timestamps, which
    is what pairs them exactly."""
    out = {}
    desc = exif.get(TAGS.get('ImageDescription', -1))
    if desc:
        m = re.search(r'vert_angle_deg=([-\d.]+)\s*/\s*'
                      r'horiz_angle_deg=([-\d.]+)', str(desc))
        if m:
            out['pitch_deg'] = float(m.group(1))
            out['roll_deg'] = float(m.group(2))
    note = ifd.get(TAGS.get('MakerNote', -1))
    if note:
        s = (note.decode('utf-8', 'ignore') if isinstance(note, bytes)
             else str(note))
        if '<theodolite>' in s:
            for tag, key in (('vert_angle_deg', 'pitch_deg'),
                             ('horiz_angle_deg', 'roll_deg'),
                             ('gps_horz_m', 'gps_horz_m'),
                             ('gps_vert_m', 'gps_vert_m'),
                             ('azimuth_deg', 'azimuth_acc_deg')):
                m = re.search(rf'<{tag}>([-\d.]+)</{tag}>', s)
                if m:
                    out[key] = float(m.group(1))
    return out


def _gray(path, max_w=640):
    with Image.open(path) as im:
        im = im.convert('L')
        if im.width > max_w:
            im = im.resize((max_w, max(1, round(im.height * max_w
                                                / im.width))))
        return np.asarray(im, float) / 255.0


def hud_score(g):
    """How much burned-in overlay the frame carries.

    The HUD adds small high-contrast glyphs in the top and bottom
    bands and a graticule through the middle — structure that survives
    where a natural scene's does not: count strong local gradients in
    the border bands, normalised by the frame's own texture so a busy
    coastline does not read as a HUD."""
    H, W = g.shape
    gx = np.abs(np.diff(g, axis=1))
    band = max(4, int(0.14 * H))
    top, bot = gx[:band], gx[-band:]
    mid = gx[band:-band] if H > 3 * band else gx
    strong = lambda a: float((a > 0.25).mean())
    base = strong(mid) + 1e-4
    return float((strong(top) + strong(bot)) / (2 * base))


def thumb(path, n=32, keep=0.70):
    """Normalised thumbnail of the frame CENTER only. The HUD lives in
    the top and bottom bands, so including them would make a sighting's
    two frames look unlike each other — exactly backwards."""
    g = _gray(path, max_w=256)
    H = g.shape[0]
    m = int(H * (1 - keep) / 2)
    g = g[m:H - m] if H - 2 * m > 8 else g
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8))
                   .resize((n, n)), float) / 255.0
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-9)


def sharpness(g):
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def _name_dist(a, b):
    """Distance between the trailing numbers of two filenames
    (IMG_0123 vs IMG_0124 -> 1); large when they share no numbering."""
    import re
    na = re.findall(r'(\d+)', a)
    nb = re.findall(r'(\d+)', b)
    if not na or not nb:
        return 1e6
    return abs(int(na[-1]) - int(nb[-1]))


def _gps_dist(a, b):
    if a.get('lat') is None or b.get('lat') is None:
        return None
    import math
    return math.hypot((a['lat'] - b['lat']) * 111132.0,
                      (a['lon'] - b['lon']) * 111320.0
                      * math.cos(math.radians(a['lat'])))


def pair_up(items, dt_max=25.0, sim_min=0.80, gps_max=60.0):
    """Match each HUD frame to the clean frame of the same sighting:
    closest in capture time AND visually the same scene. Both tests are
    needed — a pan produces frames seconds apart that look different,
    a tripod burst produces frames that look identical minutes apart."""
    hud = [i for i in items if i['is_hud']]
    raw = [i for i in items if not i['is_hud']]
    used, pairs = set(), []
    for h in sorted(hud, key=lambda x: x.get('t', 0)):
        best, best_key = None, None
        for r in raw:
            if r['name'] in used:
                continue
            sim = float((h['_thumb'] * r['_thumb']).sum())
            timed = bool(h.get('t') and r.get('t'))
            dt = abs(h['t'] - r['t']) if timed else None
            gd = _gps_dist(h, r)
            if timed:
                # A HUD capture and its clean photo show DIFFERENT
                # fields (the app was zoomed; the photo is 4:3), so
                # appearance cannot pair them — capture time and
                # position can.
                if dt > dt_max or (gd is not None and gd > gps_max):
                    continue
            else:
                # no capture time (EXIF stripped in transfer): fall back
                # to appearance alone, with a stricter threshold, and
                # break ties by filename distance since cameras number
                # a sighting's frames consecutively
                if sim < max(sim_min, 0.90):
                    continue
            key = ((dt, -sim) if timed
                   else (1e6, _name_dist(h['name'], r['name'])))
            if best is None or key < best_key:
                best, best_key = r, key
        if best is not None:
            used.add(best['name'])
            pairs.append((h, best))
        else:
            pairs.append((h, None))
    singles = [r for r in raw if r['name'] not in used]
    return pairs, singles


def horizon_check(path, fov_deg, z_m):
    """Does the sea-horizon detector find a horizon, and what attitude
    does it give? This is the decisive curation criterion: a sighting
    whose horizon is detectable can be auto-levelled, which is what
    buys the tight elevation window."""
    import extract
    import skyline as S
    with Image.open(path) as im:
        im = im.convert('RGB')
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)))
        rgb = np.asarray(im, float) / 255.0
    W = rgb.shape[1]
    f_px = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    dip = S.horizon_dip_rad(max(z_m, 1.0))
    est = extract.sea_horizon_attitude_radon(rgb, f_px, dip)
    if est is None:
        return None
    return {k: (v if isinstance(v, str) else float(v))
            for k, v in est.items()}


def save_hud_crops(path, out_dir, sid, frac=0.16):
    """Write the HUD text bands full-width at native resolution — they
    are what gets read to recover azimuth/pitch/roll/GPS/zoom."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    with Image.open(path) as im:
        im = im.convert('RGB')
        W, H = im.size
        b = int(frac * H)
        for tag, box in (('top', (0, 0, W, b)),
                         ('bottom', (0, H - b, W, H))):
            crop = im.crop(box)
            if crop.width > 1600:
                crop = crop.resize((1600, round(crop.height * 1600
                                                / crop.width)))
            p = os.path.join(out_dir, f'{sid}_hud_{tag}.png')
            crop.save(p)
            paths[tag] = p
    return paths


def curate(src, out, do_horizon, hud_thresh=1.6, z_default=10.0,
           keep_unpaired=False):
    files = sorted(f for f in glob.glob(os.path.join(src, '**', '*'),
                                        recursive=True)
                   if f.lower().endswith(EXTS))
    if not files:
        print(f'no images under {src}')
        return None
    items = []
    for f in files:
        try:
            m = read_meta(f)
            g = _gray(f)
            m['hud_score'] = hud_score(g)
            ar = m['width'] / max(m['height'], 1)
            m['aspect'] = float(ar)
            # Theodolite writes a SCREEN capture: the phone's own
            # aspect (2622x1206 = 2.17 on this device), never the
            # camera's 4:3. That is a far harder discriminator than
            # texture — the app's widgets are translucent and sit all
            # over the frame, so the border-band score is marginal and
            # mislabels pan frames as clean.
            m['screenish'] = bool(ar >= 1.90 or ar <= 0.53)
            m['is_hud'] = m['screenish'] or m['hud_score'] >= hud_thresh
            m['sharpness'] = sharpness(g)
            m['mean_level'] = float(g.mean())
            m['_thumb'] = thumb(f)
            items.append(m)
        except Exception as e:
            print(f'  ! {os.path.basename(f)}: {e}')
    print(f'{len(items)} images: '
          f'{sum(1 for i in items if i["is_hud"])} look HUD-overlaid, '
          f'{sum(1 for i in items if not i["is_hud"])} clean')

    # ---- exact grouping on Theodolite's own record. The app stamps
    # both saved versions of one sighting with the same inclinometer
    # angles and the same timestamp down to the subsecond, so pairing
    # is a dictionary lookup, not a similarity problem. Images without
    # that record are not sightings at all (ordinary camera roll
    # photos travelling in the same folder) and are set aside.
    theo = [i for i in items if i.get('theodolite')
            and i.get('pitch_deg') is not None and i.get('dt_raw')]
    not_theo = [i for i in items if i not in theo]
    groups = {}
    for i in theo:
        key = (i['pitch_deg'], i['roll_deg'], i['dt_raw'], i['sub'])
        groups.setdefault(key, []).append(i)
    pairs, singles, lone = [], [], []
    for key, v in groups.items():
        v.sort(key=lambda x: -(x['width'] * x['height']))
        screen = [x for x in v if x.get('screenish')]
        clean = [x for x in v if not x.get('screenish')]
        if screen and clean:
            pairs.append((screen[0], clean[0]))
        elif clean:
            (singles if keep_unpaired else lone).append(clean[0])
        elif keep_unpaired:
            pairs.append((screen[0], None))
        else:
            lone.append(screen[0])
    print(f'  EXIF: {len(theo)} Theodolite frames, {len(not_theo)} '
          f'ordinary photos set aside; {sum(1 for p in pairs if p[1])} '
          f'exact HUD+original pairs'
          + (f', {len(lone)} unpaired frames dropped'
             if lone else ''))
    if False:
        pairs, singles = pair_up(items)
    crop_dir = os.path.join(out, 'crops')
    sightings = []
    for h, r in pairs:
        sid = os.path.splitext((r or h)['name'])[0]
        s = dict(id=sid, hud=h['name'], raw=r['name'] if r else None,
                 kind='pair' if r else 'hud-only')
        s['exif'] = {k: v for k, v in (r or h).items()
                     if not k.startswith('_') and k != 'path'}
        # attitude comes from Theodolite's record, not from reading
        # the HUD: pitch/roll from the inclinometer, heading from
        # GPSImgDirection (true), each with its reported accuracy
        e = s['exif']
        s['attitude'] = dict(
            heading_deg=e.get('heading_deg'),
            pitch_deg=e.get('pitch_deg'), roll_deg=e.get('roll_deg'),
            azimuth_acc_deg=e.get('azimuth_acc_deg'),
            gps_horz_m=e.get('gps_horz_m'), zoom=e.get('zoom'),
            fov_deg=e.get('fov_deg'))
        s['hud_crops'] = save_hud_crops(h['path'], crop_dir, sid)
        sightings.append((s, (r or h)))
    for r in singles:
        sid = os.path.splitext(r['name'])[0]
        e = {k: v for k, v in r.items()
             if not k.startswith('_') and k != 'path'}
        sightings.append((dict(
            id=sid, hud=None, raw=r['name'], kind='original-only',
            exif=e,
            attitude=dict(heading_deg=e.get('heading_deg'),
                          pitch_deg=e.get('pitch_deg'),
                          roll_deg=e.get('roll_deg'),
                          azimuth_acc_deg=e.get('azimuth_acc_deg'),
                          gps_horz_m=e.get('gps_horz_m'),
                          zoom=e.get('zoom'),
                          fov_deg=e.get('fov_deg'))), r))

    out_rows = []
    for s, src_item in sightings:
        e = s['exif']
        fov = e.get('fov_deg')
        if do_horizon and fov:
            s['horizon'] = horizon_check(src_item['path'], fov,
                                         e.get('alt_m') or z_default)
        flags = []
        if s['kind'] == 'hud-only':
            flags.append('no-clean-frame')       # HUD sits on the skyline
        a = s.get('attitude') or {}
        if a.get('azimuth_acc_deg') and a['azimuth_acc_deg'] > 15:
            flags.append(f"compass+-{a['azimuth_acc_deg']:.0f}deg")
        if not fov:
            flags.append('no-focal-exif')
        if e.get('lat') is None:
            flags.append('no-gps')
        if src_item.get('sharpness', 1) < 1e-4:
            flags.append('soft')
        if do_horizon and fov and not s.get('horizon'):
            flags.append('no-sea-horizon')
        s['flags'] = flags
        s['usable'] = (s['kind'] != 'hud-only' and bool(fov)
                       and 'soft' not in flags)
        out_rows.append(s)

    os.makedirs(out, exist_ok=True)
    idx = os.path.join(out, 'index.json')
    with open(idx, 'w') as f:
        json.dump(dict(source=os.path.abspath(src), n=len(out_rows),
                       sightings=out_rows), f, indent=1, default=str)

    print(f'\n{len(out_rows)} sightings '
          f'({sum(1 for s in out_rows if s["kind"] == "pair")} paired, '
          f'{sum(1 for s in out_rows if s["kind"] == "raw-only")} raw-only, '
          f'{sum(1 for s in out_rows if s["kind"] == "hud-only")} HUD-only)')
    for s in out_rows:
        e = s['exif']
        loc = (f"{e['lat']:.5f},{e['lon']:.5f}" if e.get('lat')
               else 'no-gps')
        hz = ''
        if s.get('horizon'):
            hz = (f"  horizon pitch {s['horizon']['pitch_deg']:+.2f} "
                  f"roll {s['horizon']['roll_deg']:+.2f}")
        print(f"  {s['id']:24s} {s['kind']:9s} "
              f"fov {e.get('fov_deg', float('nan')):5.1f}  {loc}{hz}"
              + (f"  [{' '.join(s['flags'])}]" if s['flags'] else ''))
    print(f'\nindex: {idx}')
    print(f'HUD crops for transcription: {crop_dir}')
    return out_rows


def apply_transcript(out, tpath, z_default=10.0):
    """Merge transcribed HUD values into the index, cross-check them
    against EXIF, and emit skyfix invocations."""
    idx = os.path.join(out, 'index.json')
    with open(idx) as f:
        data = json.load(f)
    with open(tpath) as f:
        tr = json.load(f)
    cmds = []
    for s in data['sightings']:
        t = tr.get(s['id'])
        if not t:
            continue
        s['hud'] = t
        e = s['exif']
        checks = []
        if e.get('lat') is not None and t.get('lat') is not None:
            dm = np.hypot((t['lat'] - e['lat']) * 111132.0,
                          (t['lon'] - e['lon']) * 111320.0
                          * np.cos(np.radians(e['lat'])))
            checks.append(f'gps_delta_m={dm:.0f}')
            if dm > 50:
                s.setdefault('flags', []).append('gps-mismatch')
        if t.get('zoom') and e.get('fov_deg'):
            # a zoom factor should already be reflected in the EXIF
            # 35mm-equivalent; if it is not, the FOV is wrong by that
            # factor and every elevation angle scales with it
            implied = 2 * np.degrees(np.arctan(
                np.tan(np.radians(e['fov_deg']) / 2) / t['zoom']))
            checks.append(f'fov_exif={e["fov_deg"]:.1f} '
                          f'fov_if_zoom_uncorrected={implied:.1f}')
        s['checks'] = checks
        if s.get('usable') and e.get('lat') is not None:
            hdg = t.get('azimuth_deg')
            cmd = (f"python3 skyfix.py {s['raw']} "
                   f"--center {e['lat']:.6f},{e['lon']:.6f} "
                   f"--fov {e.get('fov_deg', 0):.2f} "
                   + (f"--heading {hdg:.1f} " if hdg is not None else '')
                   + f"--z {t.get('alt_m', e.get('alt_m', z_default)):.1f} "
                   f"--auto-level --box 6000")
            cmds.append(cmd)
            s['skyfix_cmd'] = cmd
    with open(idx, 'w') as f:
        json.dump(data, f, indent=1, default=str)
    print(f'merged {sum(1 for s in data["sightings"] if s.get("hud"))} '
          f'transcriptions into {idx}\n')
    for c in cmds:
        print('  ' + c)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='folder of Theodolite photos')
    ap.add_argument('--out', default=os.path.join(HERE, 'out',
                                                  'theodolite'))
    ap.add_argument('--horizon', action='store_true',
                    help='run the sea-horizon detector on each clean '
                         'frame (slower, but it decides usability)')
    ap.add_argument('--transcript', help='JSON of read HUD values')
    ap.add_argument('--hud-thresh', type=float, default=1.6)
    ap.add_argument('--include-unpaired', action='store_true',
                    help='also keep frames saved in only one version. '
                         'Off by default: a sighting without both '
                         'versions gives either no clean image to '
                         'extract from or no cross-check, and the set '
                         'has plenty of complete pairs')
    a = ap.parse_args()
    if a.transcript:
        apply_transcript(a.out, a.transcript)
    else:
        curate(a.src, a.out, a.horizon, a.hud_thresh,
               keep_unpaired=a.include_unpaired)


if __name__ == '__main__':
    main()
