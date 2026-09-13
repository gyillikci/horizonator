# Running the skyline-matching work on your own machine

The container this campaign ran in is an ordinary Ubuntu box with three
things layered on it: Python packages, a handful of pretrained weights,
and a few gigabytes of elevation tiles. Only the first is small enough
to describe in a requirements file; the other two have to be fetched.

    git clone <horizonator> && git clone <celestial-navigation>   # siblings
    cd horizonator/experiments
    ./setup_local.sh --all

That script is idempotent and prints what it skips. Read on for what it
is doing and where it can bite.

## What the container actually is

| | |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.11 |
| compiler | gcc 13 (`cc`), used to build `fastmarch.so` on first import |
| GPU | none — torch runs on 4 CPU threads, and every timing in the study doc is CPU |

Package versions the results were produced on:

| package | version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.1 |
| OpenCV | 5.0.0 |
| torch / torchvision | 2.13.0 / 0.28.0 |
| timm | 1.0.28 |
| rasterio | 1.4.4 |

`scikit-image` and `scikit-learn` are **not** installed and nothing in
the pipeline needs them. Torch is only there for the two segmentation
models; the solver itself is numpy.

## The two things that are not in git

**Pretrained weights, ~600 MB.** `eWaSR` (the maritime water/sky/obstacle
segmenter, Apache 2.0, MaSTr1325 weights) and `MobileSAM` (Apache 2.0,
used by the sea-line detector). The eWaSR clone needs two small patches
against current dependencies — `timm` moved `to_2tuple`, and the
ImageNet backbone download is dead weight because the checkpoint
overwrites it — both applied by the setup script and documented at the
top of `ewasr_bridge.py`.

These used to be resolved through a hard-coded session scratch path,
which made the tree unrunnable anywhere else. That is now
`experiments/assets.py`, which honours **`$HORIZONATOR_ASSETS`**, then
`~/.horizonator/assets`, then the legacy path if it happens to exist.
Export the variable in your shell profile and every script finds them.

**Elevation tiles, ~3.5 GB for everything this campaign touched.** The
stores are plain SRTM1-shaped `.hgt` directories under `~/.horizonator`:

| store | what it is | size |
|---|---|---|
| `DEMs_SRTM1_WM` | the working store most results use | 322 MB |
| `DEMs_COP30` | Copernicus GLO-30 DSM, built by `copernicus_to_hgt.py` | 198 MB |
| `DEMs_SRTM3` | 3-arcsecond, the horizonator default | 462 MB |
| `WorldCover` | ESA land cover, for the canopy raster | 238 MB |

`fetch_dems.py` pulls SRTM from the AWS `elevation-tiles-prod` mirror;
`copernicus_to_hgt.py` pulls GLO-30 COGs from `copernicus-dem-30m` on S3
and converts them. Both buckets are public and need no credentials.
Only fetch the boxes you work in — the defaults cover the Aegean and
Marmara.

## Things that work better on your machine than they did here

The container's egress policy is restrictive, and several dead ends in
the study doc are that policy rather than a real limit:

- **Overpass is blocked**, so `fetch_osm_landmarks.py` never ran and
  `experiments/data/` holds no landmark databases. On your machine it
  runs; commit the JSON and the night/day landmark channels work
  offline afterwards.
- **Cesium ion is blocked** (403 to CONNECT on `cesium.com`,
  `api.cesium.com`, `assets.ion.cesium.com`), so `cesium_render.py`
  reaches "Cesium is not defined" and no further. Everything up to the
  library fetch is verified; with a token and normal network it should
  run as written.
- **Bing / Virtual Earth is blocked** too. AWS open data is not, which
  is why the Copernicus route exists.
- **The GL renderer does not work here.** `skyline.GlSkyline` needs a
  real GL context and fails under this container's xvfb with
  `horizonator returned NULL`. `panorama.py` depends on it. The
  ray-marching path (`CMarcher`, `dem_scene.py`) is pure C/numpy and is
  what every result in the study doc actually used, so this is a
  missing extra rather than a missing dependency.
- **Playwright's pinned browser build is absent** and re-downloading is
  blocked, so `cesium_render.py` takes a `--chromium` path and defaults
  to whatever is under `PLAYWRIGHT_BROWSERS_PATH`. On your machine
  `playwright install chromium` just works and the default is fine.

## Sanity check

The last step of the setup script ray-casts the Bodrum viewpoint and
prints the elevation range and median subject distance. If that prints,
the native marcher, the DEM store and the geometry conventions all
agree. After that:

    python3 skyfix.py --help
    python3 e5bd_prescreen.py PHOTO LAT LON Z HEADING FOV --sweep-heading 20

The pre-screen is the cheapest way to tell whether a new frame is worth
solving, and `--sweep-heading` exists because a phone compass was 13
degrees wrong on one frame and 0.9 degrees on another (E5bf, E5bh).

## Repository layout

The two repositories must be **siblings** — `e5l_samsea.py` reaches
`../../celestial-navigation/MaSTr1325`, and the field photographs live
in `celestial-navigation/peakfinder/`.

    parent/
      horizonator/            solver, experiments, study doc
      celestial-navigation/   photographs, MaSTr1325
