# Real-time boat cruise over Google Photorealistic 3D Tiles

A boat at 2.5 m above mean sea level, sailing CesiumJS's Google 3D Tiles at 60 fps
(SSE 16, 1280x720 on the RTX PRO 2000), with the skyline solver running live on its
frames. The open sea is a globe raised by the EGM96 geoid separation, as in
`../cesium_render/` (Unreal's Cesium scene has no flat sea surface to stand on).

| file | what it does |
|---|---|
| `cruise.html` | the page: boat, autopilot, HUD, live fix map |
| `cruise.mjs` | opens it in Chrome with the Cesium ion token in memory; bench, record, live |
| `live_solver.py` | local server: SegFormer + skyfix on each frame it is sent |
| `solve_cruise.py` | the same solve over recorded frames; `boat_cmd()` is the boat's skyfix |
| `plan_routes.py` | water-only routes with land in view, from COP30 (`routes/*.json`) |
| `collect.sh` | sails every route headless, recording a pose-tagged frame every 15 s |
| `analyze_cruise.py` | bins the fix errors by region, ridge range, relief, compass, levelling |
| `pf_track.py` | particle filter on the track: dead reckoning + skyfix's skyline likelihood |
| `hybrid.py` | the particle filter handing over to a GTSAM iSAM2 factor graph once it has found the boat |
| `look_planner.py` | the camera yaw (a pan head) that sees the most usable land, from COP30 (`look` in `routes/*.json`) |
| `dem_range.py` | render against COP30 skyline, binned by the ridge's range |
| `seg_eval.py` | SegFormer's skyline against the render's exact one, by input size |
| `res_compare.py`, `look_compare.py` | single-frame solves side by side (resolutions; looking vs dead ahead) |

## Drive it and watch the fixes

```bash
export CESIUM_ION_TOKEN=$(grep '^CESIUM_ION_TOKEN=' /home/user/workspace/NavionSIM/sim/env.local | cut -d= -f2-)
/home/user/workspace/dinov3/.venv/bin/python live_solver.py &          # SegFormer + skyfix server
node cruise.mjs --live http://127.0.0.1:8765 --live-every 10           # a window: drive with the keys
node cruise.mjs --route routes/gokova.json --live http://127.0.0.1:8765   # or let the autopilot sail
```

Keys: W/S speed, A/D rudder, Q/E look down/up, Space stop, `[` `]` detail.

With `--pf` on both (`live_solver.py --pf`, `cruise.mjs ... --pf`) the server tracks with
the particle filter instead of solving each frame alone: the map shows its track (blue)
and the particle cloud.
The boat tells the solver what a boat would know: a compass (5 deg noise), an IMU
(pitch and roll, 0.3 deg; it vets the sea line) and a dead-reckoned position from its
last accepted fix (1 km off before the first). The map at the top right shows the true
track (white) and every fix (green accepted, amber refused). Frames and fixes are kept
in `out/live/<time>/`.

The boat's solve is the campaign pipeline (`campaign_frames.json` pipeline_flags)
without the land observer and land-only search (a boat is on the water), with no crest
offset and straight rays (a render has neither canopy bias nor refraction), the
levelling's tolerance at 2.5 px (a render's horizon wavers 2-3 px), the IMU vetting the
sea line, and a 5 mrad relief gate (a flat skyline far off does not pin a boat).

## Results (2026-10-25)

305 frames from seven Aegean routes (8 km each, 12 m/s): 300 solved, 196 accepted,
145 of them within 300 m, 11 more than 1 km off. The error follows the range of the
ridges in view (median 40-50 m under 5 km, 85 m at 5-8 km, 170 m at 8-12 km, ~400 m
beyond 12 km) and the skyline's relief (1.5 km median below 5 mrad std). Every wrong
accept was a far (15-30 km), low-relief view; the 5 mrad relief gate (now in the boat's
flags) cuts them from 11 to 2 for 10 good fixes: 176 accepted, 135 within 300 m.

`--band` (solve_cruise.py, live_solver.py) runs SegFormer on a third of the frame around
the horizon the IMU predicts, grown when a mountain reaches its top or a shore fills its
bottom, on SegFormer's 5 px grid. It is 1.8x faster (0.051 against 0.092 s a frame on the
RTX PRO 2000) but SegFormer, missing the rest of the frame, draws the skyline ~2 px higher
and the sea line 0.5-1.5 px further from the true horizon, whatever the band's height (a
third, a half, two thirds), and the fixes suffer: 145 accepted, 109 within 300 m, 9 over
1 km off. Off by default; SegFormer is ~1 % of a frame's time next to skyfix.

## Particle filter (sequential Monte Carlo) on the track

`pf_track.py` is the terrain-aided-navigation particle filter (Gustafsson et al. 2002;
Nordlund's marginalized form) with the camera's skyline in place of a radar altimeter:
1000 particles move by the dead-reckoned leg (2 % speed, 5 deg heading error, 20 m of
current) and are weighted by exp(-lam * cost), skyfix's cost at each particle with
lam = 20 / (3 mrad)^2, the statistics of skyfix's own compass prior. Frames whose sea line
is refused or whose relief is under 5 mrad only predict. 1.1-1.8 s a frame on 8 cores.

| route | particle filter: median, frames < 300 m | single-frame skyfix: accepted, < 300 m |
|---|---|---|
| bodrum | 62 m, 43/44 | 36, 20 |
| gokova | 85 m, 43/44 | 32, 24 |
| datca | 81 m, 34/44 | 34, 21 |
| kusadasi | 87 m, 37/44 | 28, 23 |
| cesme | 100 m, 42/43 | 20, 16 |
| ayvalik | 29 m, 41/43 | 30, 26 |
| edremit | 359 m, 16/43 | 6, 4 |

256 of 305 frames within 300 m against 134; its 95 % ellipse held the truth on 293 of 305.
Edremit's first and last legs see a flat coast 15-31 km off (relief under 5 mrad): no updates.

With a compass that errs as a real one does (a deviation N(0, 5 deg) drifting 0.1 deg a frame,
plus 2 deg of noise, the legs dead-reckoned off the same compass), over all 305 frames:

| filter | median | p90 | < 300 m | 95 % ellipse holds the truth |
|---|---|---|---|---|
| as above | 130 m | 407 m | 249 | 263 |
| `--dense-init` (likelihood on a 75 m grid seeds the cloud) | 128 m | 406 m | 251 | 265 |
| `--bias-state` (particles carry the compass bias) | 96 m | 482 m | 260 | 292 |
| `--soft-relief` (flat skylines update, weighted) | 148 m | 906 m | 221 | 222 |
| `--dense-init --bias-state` (the live default) | **77 m** | 412 m | **262** | **294** |

The bias is learnt to 1.2 deg (median). Soft-relief updates are systematically wrong on a
flat skyline (Edremit 923 m): off. The waterline is no substitute at a boat's 2.5 m: 500 m
along the line of sight moves the predicted water edge 0.00 mrad (median; p90 0.18), against
SegFormer's 1.75 mrad on the sea line (`waterline_boat.py`).

Google's terms require attribution (on screen and in every recorded frame's JSON) and
limit caching and derived datasets.

## A known departure point

`pf_track.py --known-start 50` seeds the cloud as N(departure, 50 m) (the harbour, the last
GNSS fix) instead of the 6 km box about a prior 1 km off: every hill is where the chart says
from the first frame on. Biased compass, `--bias-state`, all 305 frames:

| start | median | p90 | < 300 m | 95 % ellipse holds the truth |
|---|---|---|---|---|
| 6 km box, `--dense-init` | 77 m | 412 m | 262 | 294 |
| known to 50 m | **38 m** | **89 m** | **305** | **300** |

Edremit, whose flat far coast never lets the box converge, goes from 303 m to 55 m.

## The particle filter hands over to a factor graph (GTSAM)

`hybrid.py`: once the cloud is one tight blob (major sigma under 120 m on three updates
running) the posterior is near Gaussian, and an iSAM2 graph carries on: positions and the
compass bias as variables; the dead-reckoned legs turned back by the bias, the bias's random
walk, and per frame the skyline as a Gaussian factor (skyfix's cost on a 13 x 13 grid about
the prediction, times the prediction, moment-matched, the prediction divided out again;
Huber-robust) plus the heading shift that fits there as a measurement of the bias. Three
frames that fit badly or land far outside the prediction send it back to the particle filter.

| from a known start | median | p90 | < 300 m | in 95 % | cost evaluations a frame | s a frame |
|---|---|---|---|---|---|---|
| particle filter | 38 m | 89 m | 305 | 300 | 1000 | 2.6 |
| hybrid (filtered) | **29 m** | **83 m** | **305** | **301** | **~150** | **0.4** |
| hybrid (smoothed) | 34 m | 87 m | | | | |

Smoothing does not help: the skyline's errors are the DEM's, the same ridges' from frame to
frame (Bodrum's last third, Gokova's: ~100 m off in the particle filter too), so later
frames' bias runs back along the dead-reckoned chain.

From the 6 km box the hybrid is the particle filter until the hand-over, and no better after
it: median 90 m against the particle filter's 77 m (same 262 of 305 within 300 m). A cloud
still converging is tighter than its error (Kusadasi handed over 186 m off at a sigma under
120 m), and widening the hand-over's covariance trades one route for another (x1.5: Bodrum
41 m, Kusadasi 124 m; x4: 78 m, 101 m). The skyline factors' kernel never mattered (Huber
and plain Gaussian agree to the metre: the Mahalanobis gate leaves no outliers for it), nor
did loosening the legs by the particle filter's 15 m roughening. Where the skyline says
nothing (Gokova's last 12 frames: the cost within 2.6 / lam over +-400 m, rms 2.7-3 mrad
at 15 mrad of relief) both dead-reckon. So: the particle filter while the start is
uncertain, the graph from a known start (or a GNSS fix) on.

## Where COP30 stops describing the world

`dem_range.py`: the render's exact skyline against COP30's at the same pose, per column,
binned by the range of the ridge COP30 puts there (77 poses; signed: + COP30 lower):

| ridge range | 1280 px: median, p90 mrad | 2560 px: median, p90 mrad | signed median (2560) |
|---|---|---|---|
| 1-2 km | 1.22, 10.4 | 1.10, 10.4 | -0.93 |
| 2-3 km | 2.22, 5.8 | 2.06, 5.6 | -1.88 |
| 3-5 km | 1.33, 3.9 | 1.05, 3.7 | -0.99 |
| 5-8 km | 0.79, 2.3 | 0.49, 2.1 | -0.40 |
| 8-12 km | 1.26, 2.7 | 0.87, 2.0 | -0.83 |
| 12-20 km | 0.98, 2.1 | 0.61, 1.3 | -0.59 |
| 20-30 km | 1.07, 2.2 | 0.53, 1.3 | -0.48 |
| 30-60 km | 0.93, 2.2 | 0.60, 1.6 | -0.51 |

Beyond 5 km the difference halves with the resolution (it was the pixel); under 5 km it
does not: there the DEM is the limit (30 m posts against near detail, COP30 standing
0.9-1.9 mrad high). The offset common to all ranges is skyfix's elevation offset beta.


## Camera resolution and SegFormer's input size

`seg_eval.py`: SegFormer-B2's skyline against the render's exact one on the same 77 poses
rendered at 1280x640 and 2560x1280 (70 deg), by the size SegFormer's input is scaled to
(its short side; native = the frame's own). Errors in mrad, so the two cameras compare:

| SegFormer input | 1280 px camera: mean mrad, >= 2 px, s | 2560 px camera: mean mrad, >= 2 px, s |
|---|---|---|
| 512 (the default) | 0.93, 17.7 %, 0.17 | 0.94, 48.4 %, 0.22 |
| 768 | 0.81, 12.0 %, 0.30 | 0.71, 36.8 %, 0.32 |
| 1024 | **0.62**, 5.6 %, 0.59 | 0.52, 22.3 %, 0.53 |
| native (640 / 1280) | 0.76, 10.0 %, 0.23 | **0.37**, 10.7 %, 0.85 |

The skyline's error is SegFormer's input's, not the camera's: at the default 512 both
cameras draw it to 0.93 mrad. A sharper camera pays only with SegFormer run larger (2560 px
at its native 1280: 0.37 mrad, 2.5x better, 0.85 s a frame on the RTX PRO 2000); a flip
average of the logits changed nothing (the share of columns a pixel off moved under 0.4
points, 1280 px). Note `extract.load_image` downsizes
frames wider than 1600 px, skyfix's too: a 2560 px camera needs `--max-width` (skyfix,
solve_cruise.py) to be one.

What the sharper skylines do to the fixes (`res_compare.py`, the same 77 poses, the boat's
skyfix; the levelling's tolerance scaled with the frame, 2.5 px at 1280, 5 px at 2560):

| camera, SegFormer input | accepted | < 300 m | > 1 km | every fix, median | paired with 1280/512: median, better/worse |
|---|---|---|---|---|---|
| 1280 px, 512 | 47 | 34 | 1 | 251 m | |
| 1280 px, 1024 | 50 | 33 | 3 | 221 m | +0 m, 21/21 |
| 2560 px, 512 | 44 | 30 | 1 | 188 m | +5 m, 15/22 |
| 2560 px, native 1280 | 49 | 30 | 1 | 156 m | +16 m, 16/25 |

Nothing that matters moves. The sharper input also levels better (pitch |p90| 0.31-0.35 deg
-> 0.13-0.14 deg) and fits COP30 better at the true pose (0.84 against 1.06 mrad beyond
12 km), but even the render's exact skyline stands 0.75-1.8 mrad off COP30 there: the DEM's
error, not the segmenter's 0.4-0.9 mrad, sets a fix's. A higher resolution camera buys
nothing until the terrain model is better than 30 m posts.

## Looking around (a pan head)

`look_planner.py` gives each route a camera yaw every 150 m (`look` in `routes/*.json`):
from where the boat roughly is, COP30's skyline over the field of view for yaws every 15
deg, refused when flat (relief under 5 mrad), bearing-only (1/r spread under 0.03 per km),
mostly a shore nearer than skyfix's 1 km, or without the true sea horizon across half the
frame (the levelling's), scored relief x land / max(range, 3 km), dead ahead winning
near-ties. The score predicts the fix: over the 304 dead-ahead frames, its top third had 97
accepted (median 75 m), its bottom third 18 (290 m). `cruise.html` turns the camera to the
nearest look point's yaw at 30 deg/s (hull pitch and roll mixed into the camera's as it
turns); `cruise.mjs --look` sails with it, `live_solver.py --pf --look` answers each frame
with the next yaw from the filter's estimate and `cruise.mjs --look-live` turns to it.

The seven routes sailed again looking where it said (`LOOK=1 ./collect.sh` ->
out/aegean_look; |yaw| median 30-105 deg by route), single-frame skyfix:

| route | dead ahead: accepted, < 300 m, median | looking around: accepted, < 300 m, median |
|---|---|---|
| ayvalik | 30/43, 26, 72 m | 39/44, 29, 68 m |
| bodrum | 36/44, 20, 276 m | 41/44, 38, 75 m |
| cesme | 20/43, 16, 127 m | 37/44, 25, 164 m |
| datca | 34/44, 21, 174 m | 42/44, 41, 69 m |
| edremit | 6/43, 4, 256 m | 29/43, 8, 430 m |
| gokova | 32/44, 24, 71 m | 43/44, 43, 40 m |
| kusadasi | 28/44, 23, 139 m | 43/44, 38, 117 m |
| **all** | **186/305, 134, 136 m** | **274/307, 222, 83 m** |

But 10 accepted fixes are over 1 km off (3 dead ahead): the looks the planner chose sit on
skyfix's own gates -- Ayvalik's last six frames look astern at 6 mrad of relief, Edremit's
at Kaz Dagi's massif with 17-19 mrad of relief but a 1/r spread of 0.031-0.037 (the gate is
0.03): one range, bearing only.

The trackers on the same recordings (biased compass, `--bias-state`):

| tracker | dead ahead: median, p90, < 300 m | looking around: median, p90, < 300 m |
|---|---|---|
| particle filter, known start | 38 m, 89 m, 305/305 | 51 m, 134 m, 307/307 |
| hybrid, known start | 29 m, 83 m, 305/305 | 49 m, 125 m, 307/307 |
| particle filter, 6 km box (`--dense-init`) | 77 m, 412 m, 262/305 | 80 m, 298 m, 276/307 |

Looking around helps a filter that has not found the boat yet (the box start's tail) and
hurts one that has. The planner's ridges are nearer (median 7.8 km in view against 20 km
dead ahead; under 5 km in 25 % of the frames against 10 %): more parallax, which is what
pins a single frame, but near terrain is where COP30 errs most (above) and where a roll
error trades against position. On Bodrum's sideways frames the levelled roll is 0.2 deg off;
the filter's likelihood (rolls searched at -0.5, 0, +0.5 deg) puts its minimum 100-200 m east
of the truth where skyfix (0.25 deg steps) stays within 20-60 m. Finer rolls in the filter
(`--rolls -0.5,-0.25,0,0.25,0.5`) do not settle it: route medians move 20-30 m either way
(looking: Bodrum 85 -> 53 m, Kusadasi 54 -> 34 m; dead ahead: Bodrum 33 -> 65 m), about the
spread between runs of a route.
