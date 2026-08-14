# Skyline-matching experiments

Companion experiments to [`doc/skyline-matching-study.md`](../doc/skyline-matching-study.md):
localize a camera with known intrinsics and orientation, but unknown position,
by matching its observed skyline against skylines rendered from a DEM. The
scenario is an observer at sea (viewer height a few meters, known), searching
a 1 km × 1 km box, looking at island/mainland skylines. Test area: the
Bodrum–Kos region of the Aegean.

## Files

- `skyline.py` — shared machinery: DEM loading, a GL-renderer skyline backend
  (the horizonator), an independent NumPy ray-marching backend with the same
  curvature+refraction model, the robust matching cost, and the
  coarse-to-fine position solver.
- `fetch_dems.py` — downloads the 4 SRTM tiles for the test area from the AWS
  `elevation-tiles-prod` mirror and decimates them to 3".
- `e0_validate.py` — E0: physics validation of the (curvature-patched)
  renderer for a sea-level observer. 16 checks.
- `e1_closed_loop.py` — E1: synthetic closed-loop localization at two sites
  (a strait with terrain all around; open water with land in one narrow
  sector), 20 ground-truth positions each, with FOV / noise / heading-bias
  configs. Writes `out/e1_results.json` and per-site `.npz`.
- `e1_plots.py` — figures from the E1 outputs.

## Setup

Build the horizonator Python module (see the top-level README for the
dependency list; only the Python module is needed, not the FLTK GUI):

    make horizonator$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

Runtime needs `numpy` (< 2 if the module was built against numpy 1.x
headers), and `matplotlib` + `scipy` for the plots. Fetch the DEMs:

    python3 fetch_dems.py

The GL renderer needs a display; on a headless box run everything under
`xvfb-run` (Mesa/llvmpipe renders a 45 km-radius 3" scene in ~0.35 s):

    xvfb-run -a python3 e0_validate.py
    xvfb-run -a python3 e1_closed_loop.py
    python3 e1_plots.py

`HORIZONATOR_DEMS` overrides the DEM directory (default
`~/.horizonator/DEMs_SRTM3`).

## Notes

- The AWS skadi tiles contain ocean **bathymetry** (negative elevations).
  Both the horizonator (`dem.c`) and `skyline.py` clamp negative elevations
  to sea level, so the sea renders at z=0 as the skyline model requires.
- Skylines are extracted from the range image: per azimuth column, the
  topmost pixel with a valid range. At the default 0.1°/pixel this
  quantizes elevation angles to ±0.87 mrad, which is the current accuracy
  floor (see the E1 results in the study doc).
- Azimuth bins with no terrain (open sea out to the far clip plane) are
  filled with the analytic sea-horizon dip `sqrt(2h/Reff)` so that
  land-vs-sea disagreements between observation and candidate still carry
  cost.
