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

## Native ray-marcher (`fastmarch.c`)

The on-device prototype (e.g. Raspberry Pi CM5): one C/OpenMP ray per
azimuth over a DEM mosaic, same curvature model as the patched shader,
division- and trig-free inner loop. `skyline.CMarcher` compiles it on first
use (`cc -O3 -march=native -fopenmp`; use `-mcpu=native` on ARM) and wraps
it via ctypes. Measured on the 4-vCPU build container: 13.8 ms/skyline
single-thread, 4.4 ms on 4 threads (3600 azimuths, 40 km range at 90 m
steps) — ~100x the NumPy marcher, agreeing with it to 0.02 mrad RMS.
`e3_scale.py` uses it to search a 100 km x 100 km box.

## Automatic front-end and the `skyfix` CLI

- `extract.py` — automatic skyline extraction (local-linear-continuation
  boundary detector: first sustained deviation from the sky, robust to
  graded/hazy skies and to crisp sea horizons below faint distant ridges)
  plus EXIF utilities (FOV from FocalLengthIn35mmFilm, GPS direction and
  altitude).
- `skyfix.py` — the end-to-end CLI: photo(s) in, position fix + covariance
  out. Takes one photo or several from the SAME position (the
  recommended field procedure is a telephoto pan; per-photo
  `--fov/--heading/--pitch/--roll` as comma lists, `x` = unknown
  heading), fused into one joint fix with per-photo precision weights
  (`--px-err`, `--sigma-dem`; see study doc E4i). The heading-shift
  search is FFT-accelerated — bit-identical robust optimum, ~100×
  faster, so a full-circle heading-unknown solve costs the same ~5 s
  as one with a compass prior. Includes margin-based fix rejection: when the second-best coarse
  basin is within `--min-margin` (default 0.15) of the best, the
  landscape is ambiguous. Any failed trust check — ambiguous basins,
  minimum railed on the box boundary, residual the DEM cannot explain
  (`--max-rms`), or too little skyline relief (`--min-relief`) — makes
  the result `status: "inconclusive"` (exit code 2, reasons listed), a
  first-class no-fix like a navigator refusing a doubtful sight. The
  same checks in `SkyNav.take_fix` keep such fixes out of the factor
  graph, and the NMEA GGA quality field drops to 6 (estimated/DR)
  until the next accepted fix. `python3 skyfix.py IMG --center LAT,LON [--pitch P --roll R ...]`;
  FOV/heading/altitude default from EXIF. Run it on your own coastal
  photos: keep original files (EXIF intact), supply pitch/roll from an
  IMU app to ~0.5 deg, use `--box` for your dead-reckoning uncertainty.
- `e4c_synth.py` — end-to-end validation on photo-realistic composites
  with real EXIF (written via piexif): 23–239 m error at the four
  sea-observer cases in a 5 km box, ~11 s per fix, with IMU-prior
  attitude and the wide ±10 mrad offset window (the large end of that
  range is wide-window range wander, which auto-levelling below
  removes).
- Sea-horizon auto-levelling (`skyfix --auto-level`,
  `extract.sea_horizon_attitude`): estimates pitch/roll from the visible
  sea horizon — its dip below level is exactly known from the camera
  height, making the horizon line a drift-free attitude reference better
  than an IMU. On success the co-estimated elevation-offset window
  tightens from ±10 to ±2 mrad, which sharpens range observability
  (position σ roughly halves on the E4c sea cases, `e4g_autolevel.py`).
  Guards: RANSAC with a nothing-below-the-line veto plus a photometric
  water check that rejects "false horizons" (straight hazy ridges); when
  no sea horizon is accepted, skyfix falls back to `--pitch`/`--roll`.

## Fusion (E5) and hardware benchmark

- `skyline_factor.py` — skyline fixes as GTSAM factors: a unary Pose2
  `CustomFactor` carrying the fix's anisotropic Laplace covariance
  (with an empirical calibration, see the docstring), works with the
  stock `pip install gtsam` wheel.
- `e5_fusion.py` — a simulated 2 h coastal passage (Bodrum–Kos strait):
  dead reckoning with +3% log and 1.5° compass bias vs the factor graph
  fusing a skyline fix every 15 min (each solved over a 2 km box around
  the current DR estimate, 0.6 s/fix). DR alone: 416 m mean / 787 m
  final; fused: **46 m mean / 9 m final**.
- `bench_cm5.sh` — turnkey benchmark for target hardware (CM5): fetches
  DEMs, compiles the marcher, times skyline synthesis and full fixes.
  Reference (4-core x86): 13.5/3.8 ms per skyline (1/all threads),
  0.38 s per 1 km-box fix, ~9 s per 100 km-box coarse stage.
- `skyline_fix.py` in the celestial-navigation repo (same branch) bridges
  skyfix JSON output into that toolkit (LatLonGeodetic + Circle).

## Live instrument loop (`skynav.py`)

The deployable version of E5: `SkyNav` fuses odometry legs and skyline
fixes incrementally with iSAM2 (0.1 ms per odometry update, ~0.7 s per
fix on the build box) and emits the fused position as NMEA 0183
$GPGGA/$GPRMC sentences — feed them to OpenCPN/a chartplotter and the
skyline-derived position displays like any GPS. `e5b_live.py` replays
the E5 passage as a stream: live (causal) fused error 78 m mean / 10 m
final vs the batch smoother's 31 m / 7 m. Note the compass-vs-Pose2
angle convention (theta = pi/2 - heading), converted at the API
boundary — feeding compass headings straight into Pose2 diverges.

## Scene panorama (`panorama.py`)

Render a colorized 360° scene panorama from any point:

    xvfb-run -a python3 panorama.py LAT LON [Z] -o pano.png

Equirectangular at 0.05°/px with an azimuth ruler (center = North);
E4c scene coloring plus hillshade, elevation tint and haze. Z defaults
to the DEM height + 3 m. This is a visualization of exactly what the
matcher compares observations against.

## Cross-pollination (E5c)

Three imports from the parallel study branch (celestial-navigation
`claude/iphone-celestial-sighting-imu-ctwbnf`): `SkyNav` estimates the
compass bias as a graph variable, measured directly by each skyline
fix's co-estimated azimuth shift (`e5c_bias.py`: recovers +1.32° of a
true +1.50° and improves the live loop); `skyfix --pitch-sigma` sizes
the elevation-offset window for the pitch source (~0.3 braced IMU,
~1.5 for uncalibrated AR-app pitch, which was field-measured to carry
a +1.5° platform-dependent bias); and the recorded warning that rms is
not comparable across different extractions — multi-hypothesis
selection must use margin/separation, never raw residual.

## SRTM1 full resolution (E4p)

The AWS skadi tiles are 1-arcsecond; `fetch_dems.py` decimates them to
3″ for the default `DEMs_SRTM3`. Keep the originals in
`~/.horizonator/DEMs_SRTM1` and pass `--dem` to use them — `CMarcher`
scales its march step to the posting automatically (90 m → 40 m).
Decimation was costing 2.18 mrad rms of skyline detail; on the sea
cases full resolution halves position error (e.g. 136→71 m) at ~2.5×
solve time. On ambiguous alpine terrain it is a wash — the limiter
there is the landscape, not the DEM.
