# Skyline / scene matching for camera localization — a study

*Goal: estimate the **position** of a camera whose **intrinsics and orientation
(extrinsics) are known**, by matching the observed skyline (island / mainland
silhouette) against synthetic skylines rendered from a DEM with the
horizonator. Initial search region: a 1 km × 1 km box. Primary scenario: an
observer at sea (height above sea level known to within a few meters), so the
unknown state is essentially 2D — latitude and longitude.*

This document is a study: it analyzes the horizonator codebase as the
candidate rendering engine, reviews the (sparse) literature on skyline-based
geo-localization and the neighboring literature on terrain-aided navigation
and structural/topological matching, and proposes a concrete matching +
optimization pipeline and an experiment plan.

---

## 1. Problem statement

- **Known:** camera intrinsic calibration (focal length, principal point,
  distortion) and absolute orientation (heading/pitch/roll from compass + IMU,
  or from celestial observation). Observer height above sea level `h` known
  approximately (deck/mast height + tide uncertainty of a couple of meters).
- **Unknown:** observer position `(lat, lon)`, constrained to a search box,
  initially 1 km × 1 km (later: grow to the whole coastal strip reachable by
  dead reckoning).
- **Observation:** one or more camera frames containing the skyline of an
  island or mainland coast — the curve separating sky from terrain.
- **Map:** SRTM (or better) DEM. Missing DEM cells are treated as elevation 0
  = open ocean, which the horizonator already does by convention.

### 1.1 Why known orientation changes the problem

Almost all published skyline-matching work searches over orientation (mostly
yaw, sometimes full rotation) *given* a known or hypothesized position —
because their use case is photo annotation or coarse geo-localization. Our
problem is the transpose: orientation is known, position is not. This matters:

- With absolute heading known, the azimuth of every recognizable skyline
  feature is a **bearing to a known landmark**. Dense skyline matching then
  becomes a continuous generalization of the classic navigator's *resection /
  compass bearing fix* (three bearings → position). This is also exactly the
  "terrestrial navigation" mode of the companion `celestial-navigation`
  toolkit, with the skyline supplying hundreds of simultaneous bearings
  instead of three hand-taken ones.
- The residual between observed and predicted skyline is dominated by
  **parallax**: a feature at range `d` shifts in azimuth by `Δ⊥/d` radians
  when the observer moves `Δ⊥` perpendicular to the line of sight. Features at
  different ranges shift by different amounts — that differential shift is
  what makes the position observable in *both* axes even from a single view.

### 1.2 Observability back-of-envelope (sea-level observer)

Using an effective Earth radius `R_eff = R/(1−k)`, `k ≈ 0.13` (standard
atmospheric refraction), `R_eff ≈ 7320 km`:

| Observer height `h` | Distance to sea horizon `√(2·R_eff·h)` |
|---|---|
| 2 m | 5.4 km |
| 5 m | 8.6 km |
| 10 m | 12.1 km |
| 20 m | 17.1 km |

Terrain of height `H` is visible from sea level out to roughly
`√(2·R_eff·h) + √(2·R_eff·H)` — a 500 m island remains skyline-visible from
~90 km away. So the relevant feature ranges are 5–50 km.

Curvature + refraction drop of a target at distance `d` is `d²/(2·R_eff)`:

| `d` | vertical drop | angular effect `d/(2·R_eff)` |
|---|---|---|
| 5 km | 1.7 m | 0.34 mrad |
| 10 km | 6.8 m | 0.68 mrad |
| 20 km | 27 m | 1.37 mrad |
| 40 km | 109 m | 2.73 mrad |

Sensitivities, for skyline extraction good to ~0.5–1 mrad (sub-pixel on a
0.1°/pixel render):

- **Azimuth channel (strong):** a feature at `d = 10 km` moves 1 mrad in
  azimuth per 10 m of cross-track observer motion. With heading known to
  σ ≈ 2 mrad (a good compass/IMU), a single feature bearing already pins the
  cross-line-of-sight coordinate to ~`d·σ` ≈ 20 m at 10 km. Two features with
  well-separated bearings give a full 2D fix. A 1 km × 1 km box therefore
  spans on the order of ±100 mrad of azimuth parallax at 10 km — a huge,
  easily detectable signal.
- **Elevation channel (weak):** the elevation angle of a peak of height `H` at
  range `d` is `θ ≈ (H−h)/d − d/(2·R_eff)`, so `∂θ/∂d ≈ −(H−h)/d²`. For
  `H = 500 m`, `d = 20 km`: ~1 mrad per 800 m of range change. Elevation alone
  localizes poorly along the line of sight; it mainly helps reject gross
  mismatches and disambiguate which ridge is which.
- **Heading error is the dominant systematic:** a heading bias of `ε` shifts
  the *whole* skyline in azimuth and aliases into a cross-track position error
  `≈ d·ε` toward whatever the mean feature range is. If the compass is only
  good to 1° (17 mrad), position from a 10 km skyline is biased by ~170 m.
  Mitigation: (a) estimate a single azimuth-offset nuisance parameter jointly
  with position (cheap — see §5), or (b) anchor absolute azimuth
  astronomically (Sun/Polaris azimuth), which the `celestial-navigation`
  toolkit already computes.

Conclusion: within a 1 km box the problem is well-posed with meaningful
terrain in view; expected accuracy is tens of meters, limited by heading
accuracy, DEM quality, and refraction modeling — not by the search algorithm.

---

## 2. The horizonator as the synthetic-skyline engine

### 2.1 What the code gives us today

The horizonator (C library + Python wrapper, `horizonator.h`,
`horizonator-pywrap.c`) renders SRTM terrain with OpenGL and is structured
exactly the way a matching loop wants:

- **Slow constructor, fast renders.** `horizonator_init()` loads DEM tiles
  around a center point once (`render_radius_cells`, default 1000 cells ≈
  90 km at 3″) and builds the triangle mesh / VBOs. Subsequent
  `horizonator_move()` + `horizonator_redraw()` calls only update uniforms
  (`viewer_cell_i/j`, `viewer_z`) and redraw — moving the viewer inside the
  loaded area costs one GPU pass, no re-meshing. A 1 km × 1 km candidate grid
  sits trivially inside one loaded DEM block, so per-candidate cost ≈ one
  offscreen render.
- **The projection is already the matching representation.** The render is
  equirectangular: pixel x is azimuth (linear, `az_deg0..az_deg1`), pixel y is
  elevation angle, with equal angular resolution in both axes
  (`vertex.glsl`). A synthetic **skyline is a 1D function θ(az)** and falls
  out of the render directly.
- **The range image makes skyline extraction trivial and depth-aware.**
  `horizonator_render_offscreen()` / Python `render()` return a float range
  image where sky pixels have range < 0. The skyline is, per column, the
  topmost pixel with range ≥ 0 — and we simultaneously get the **range to
  every skyline pixel**, which §5 uses to weight the cost function by
  parallax sensitivity `1/d`.
- **Sea is handled by convention.** Missing DEM files are treated as
  elevation 0 (open ocean), so coastal/offshore rendering works without
  special-casing.
- **`horizonator_pick()`/`horizonator_unproject()`** map a rendered pixel back
  to (lat, lon) — useful for diagnostics ("which ridge produced this skyline
  segment") and for peak-labeling debug overlays.

### 2.2 Gaps that matter for the sea-level scenario

1. **No Earth curvature or refraction.** The vertex shader works in the
   viewer's tangent plane; the in-code comment (`vertex.glsl`) itself measures
   ~31 m of vertical error at 20 km (≈1.5 mrad — several skyline pixels). For
   a mountain-top viewer this is tolerable; for a sea-level viewer looking at
   coastlines 10–50 km away it is first-order: it raises distant terrain,
   *fails to hide* terrain that should be below the sea horizon, and gets the
   sea-horizon distance itself wrong. **Fix is one line in the shader:**
   `enh.z -= dot(en,en) / (2.0 * R_eff);` with `R_eff = R/(1−k)`, `k ≈ 0.13`
   — the standard surveyor's curvature-plus-refraction correction. (Doing it
   in the shader keeps the CPU-side mesh untouched.)
2. **The Python API does not expose viewer height.** `py_horizonator_render()`
   calls `horizonator_move(&ctx, NULL, lat, lon)`, and the NULL makes the C
   code auto-select `z = max(4 surrounding DEM cells) + 1 m`. At sea that
   yields z = 1 m regardless of the actual bridge/deck height. Needed: a
   `z=` keyword on `render()` (and constructor) plumbed into the existing
   `viewer_z` in/out parameter of `horizonator_move()` — the C API already
   supports it.
3. **Pitch/roll are hard-wired to zero.** The renderer always looks out
   parallel to the horizontal plane. That is fine — with known extrinsics the
   *observed* image should be resampled into the same az/el equirectangular
   frame (each camera pixel maps to a ray, each ray to (az, el) via the known
   rotation), rather than making the renderer match the camera. This also
   makes the comparison camera-agnostic and handles lens distortion once.
4. **Headless operation.** Offscreen rendering still goes through GLUT, so a
   display (or Xvfb/EGL work) is needed on servers. For batch evaluation on a
   GPU-less box, an alternative is a small CPU ray-marching skyline generator
   over the same loaded DEM (for a 1D skyline we only need ~1 ray per azimuth
   column, marched with the curvature term — this is cheap and
   embarrassingly parallel, and several papers below do exactly this instead
   of full rendering).
5. **3″ SRTM (~90 m posting, ~10 m vertical RMS).** At 10–20 km this is
   ≈0.5–1 mrad of skyline noise — the same order as our extraction accuracy,
   so it sets the accuracy floor. 1″ SRTM is supported (`SRTM1`) but renders
   9× more triangles; for skyline-only evaluation the ray-marcher route makes
   1″ affordable. Note SRTM is a C-band surface model: forest canopy biases
   coastal skylines by up to tree height. Where available, prefer
   1″ Copernicus GLO-30, which the horizonator can read if converted to the
   same `.hgt` layout.

### 2.3 Cost of an exhaustive 1 km × 1 km sweep

With the observed skyline resampled to e.g. 0.05°/bin, a candidate evaluation
is: move viewer → render (or ray-march) → extract θ_syn(az) → compare against
θ_obs(az). At 25 m grid pitch a 1 km box is 41 × 41 = 1 681 candidates; at
50 m it is 441. Even full OpenGL renders at a few ms each put the exhaustive
sweep in the seconds range; a 1D ray-marcher makes it interactive. **The
search problem at 1 km scale is computationally easy — the interesting
questions are the cost function's shape (basin of attraction, coastal
degeneracies) and robustness (clouds, sea clutter, DEM error), which is where
the literature below is most useful.** The optimization machinery of §5
matters when the box grows to 10–100 km (dead-reckoning uncertainty after a
day), where exhaustive dense search stops being free.

---

## 3. Literature: skyline / horizon matching for geo-localization

*(Survey of the direct literature — see §4 for the neighboring fields the
optimization ideas come from.)*

<!-- FILLED FROM LITERATURE AGENT A -->

---

## 4. Literature: search & optimization techniques for the position sweep

<!-- FILLED FROM LITERATURE AGENT B -->

---

## 5. Proposed pipeline

<!-- FILLED AFTER LITERATURE SECTIONS -->

---

## 6. Experiment plan (1 km × 1 km, observer at sea)

<!-- FILLED AFTER LITERATURE SECTIONS -->
