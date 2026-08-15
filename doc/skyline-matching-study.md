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
   in the shader keeps the CPU-side mesh untouched.) **Implemented:**
   `vertex.glsl` and the CPU-side `horizonator_project()` now apply this
   correction, with `k = HORIZONATOR_REFRACTION_K = 0.13` shared via
   `horizonator.h`; validated by experiment E0 below.
2. **The Python API does not expose viewer height.** `py_horizonator_render()`
   calls `horizonator_move(&ctx, NULL, lat, lon)`, and the NULL makes the C
   code auto-select `z = max(4 surrounding DEM cells) + 1 m`. At sea that
   yields z = 1 m regardless of the actual bridge/deck height. Needed: a
   `z=` keyword on `render()` (and constructor) plumbed into the existing
   `viewer_z` in/out parameter of `horizonator_move()` — the C API already
   supports it. **Implemented:** both the constructor and `render()` now
   accept `z=` (default: auto-select as before).
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

The user's instinct is right: the direct literature is small — a few dozen
papers over 30 years, in five clusters. No paper treats *exactly* "known full
rotation, unknown position, 1 km² box"; the closest templates are the
planetary-rover work (§3.2) and the maritime archipelago work (§3.4).

### 3.1 Classical robotics origins (1990s)

- **Talluri & Aggarwal**, "Position estimation for an autonomous mobile robot
  in an outdoor environment" (IEEE TRA 8(5), 1992). Earliest "skyline as a
  position fingerprint": geometric constraints from image-horizon features
  vs DEM horizon, hypothesize-and-test, heading approximately known.
- **Stein & Medioni**, "Map-based localization using the panoramic horizon"
  (IEEE TRA 11(6), 1995). 360° horizon curve, polygonally approximated into
  "super-segments" used as index keys into a precomputed database of
  DEM-synthesized horizons over a position grid — the ancestor of all later
  indexing approaches.
- **Cozman & Krotkov (+ Guestrin), Viper** — ["Automatic mountain detection
  and pose estimation for teleoperation of lunar
  rovers"](https://ieeexplore.ieee.org/document/619329) (ICRA 1997);
  ["Outdoor Visual Position Estimation for Planetary
  Rovers"](https://link.springer.com/article/10.1023/A:1008966317408)
  (Autonomous Robots 9(2), 2000). Skyline as a **1D elevation-angle function
  of azimuth** matched against map-rendered horizons; probabilistic posterior
  over an exhaustive position grid; orientation approximately known from the
  pan/tilt mount. Localized the Apollo 17 site to a few hundred meters over
  multi-km² areas. *The closest classical analog to our problem.*
- **Naval, Mukunoki, Minoh & Ikeda**, ["Estimating Camera Position and
  Orientation from Geographical Map and Mountain
  Image"](https://www.researchgate.net/publication/2475290_Estimating_Camera_Position_and_Orientation_from_Geographical_Map_and_Mountain_Image)
  (1997). Neural-net skyline extraction + nonlinear optimization for pose.
- **Behringer** (IEEE VR 1999) — the *inverse* problem (known GPS position,
  imprecise orientation): predicts the DEM horizon and matches to correct
  heading/attitude to <10 mrad. Foundational for AR annotation.

### 3.2 Planetary-rover horizon localization

- **[Chiodini et al., "Mars rovers localization by matching local horizon to
  surface DEMs"](https://ieeexplore.ieee.org/document/7999600/)**
  (MetroAeroSpace 2017). Skyline from rover panoramas (sky mask → 1D curve);
  **skylines simulated on a template grid of candidate positions** over a
  HiRISE DEM; azimuth alignment as a cheap circular-correlation shift;
  exhaustive position grid over hundreds of meters to km, error of order the
  DEM grid. *Almost exactly the target problem and the recommended
  baseline architecture.*
- Carle et al., "Long-range rover localization by matching LIDAR scans to
  orbital elevation maps" (JFR 2010) — same geometry, lidar sensor.

### 3.3 Photo geo-localization in mountains (the ETH/Brno thread)

- **[Baboud et al., "Automatic photo-to-terrain alignment for the annotation
  of mountain pictures"](https://dl.acm.org/doi/10.1109/CVPR.2011.5995727)**
  (CVPR 2011). Known GPS + FOV, **full rotation searched** over SO(3):
  spherical cross-correlation of edge-orientation fields for pruning, then a
  robust silhouette-edge metric. 86% correct alignment, <0.2° error. The
  reference for silhouette matching *quality*; inverse of our unknowns.
- **[Baatz, Saurer, Köser, Pollefeys, "Large Scale Visual Geo-Localization of
  Images in Mountainous
  Terrain"](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_37)**
  (ECCV 2012) and the journal version **[Saurer et al., "Image Based
  Geo-localization in the
  Alps"](https://link.springer.com/article/10.1007/s11263-015-0830-0)** (IJCV
  116(3), 2016). Sky segmentation → 360° skyline; skyline cut into ~10°
  segments encoded as quantized **"contour words"**; inverted-file
  bag-of-words retrieval voting jointly for location *and* azimuth, then
  ICP-style refinement. ~88% of 200+ queries localized over **all of
  Switzerland (40 000 km²)**. Dataset (CH1) released
  ([project page](https://cvg.ethz.ch/research/mountain_res)).
- **[Tzeng et al., "User-Driven Geolocation of Untagged Desert Imagery Using
  DEMs"](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W07/papers/Tzeng_User-Driven_Geolocation_of_2013_CVPR_paper.pdf)**
  (CVPRW 2013). No metadata at all; **concavity features** on skylines
  (stable under unknown FOV). Shows which features survive when calibration
  is unknown — by contrast, our known intrinsics+orientation admit plain 1D
  correlation.
- **[Chen et al., "Camera geolocation from mountain
  images"](https://ieeexplore.ieee.org/document/7266746)** (FUSION 2015).
  Explicitly handles **degraded skylines** (haze, occlusion); matches skyline
  *and interior ridge structure* by vector cross-correlation on a uniform
  geospatial grid.
- **[GeoPose3K](https://cphoto.fit.vutbr.cz/geoPose3K/)** (Brejcha & Čadík,
  IVC 2017): 3 000+ precisely-posed mountain photos with rendered
  depth/normals — the benchmark of this niche. Survey: Brejcha & Čadík,
  ["State-of-the-art in visual
  geo-localization"](https://link.springer.com/article/10.1007/s10044-017-0611-1)
  (PAA 2017).
- Azimuth-only variant: [Nagy, PFG 2020](https://link.springer.com/content/pdf/10.1007/s41064-020-00093-1.pdf)
  (known position, azimuth from skyline correlation) — useful as our
  heading-bias co-estimation step, run in reverse.

### 3.4 Maritime / sea-level observers — the closest cluster to our scenario

- **[Grelsson, Robinson, Felsberg & Khan, "GPS-level accurate camera
  localization with
  HorizonNet"](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.21929)**
  (Journal of Field Robotics 37(6), 2020). USV in the Swedish archipelago,
  360° panorama, sea-level observer viewing island silhouettes. One CNN
  estimates the approximate horizon (→ pitch/roll, image levelled), a second
  extracts the pixel-wise water/land/sky boundary; matching against
  DEM+island-geography horizons over a candidate-position grid with 1D
  correlation, plus temporal filtering along the track. **~10 m ("GPS-level")
  accuracy.** *The single most relevant modern paper for our scenario.*
- **[Naus & Wąż, "Precision in Determining Ship Position using the Method of
  Comparing an Omnidirectional Map to a Visual Shoreline
  Image"](https://www.cambridge.org/core/journals/journal-of-navigation/article/precision-in-determining-ship-position-using-the-method-of-comparing-an-omnidirectional-map-to-a-visual-shoreline-image/034E262CC0968993E439F2F50B33C6FD)**
  (Journal of Navigation 69(2), 2016). Spherical catadioptric shoreline image
  correlated against a synthetic view generated from the **Electronic
  Navigational Chart** near the dead-reckoning position; accuracy assessed
  against DGPS. Also one of the few works modeling refraction/tide effects.
- **[Foucher et al., "Deep Visual-Geolocalization in Maritime Coastal
  Environment"](https://hal.science/hal-05138313v1/document)** (IEEE OCEANS
  2025). Deep horizon extraction + horizon correlation for USVs with a
  **limited-FOV camera** (incl. thermal); quantifies accuracy degradation as
  FOV shrinks — directly relevant to a non-panoramic camera.
- Radar analog: "GPS-less Coastal Navigation using Marine Radar" (IFAC 2016)
  — coastline matching with the same geometry, different sensor.

### 3.5 Aerial and urban variants

- **[Dumble & Gibbens, "Efficient Terrain-Aided Visual Horizon Based Attitude
  Estimation and
  Localization"](https://link.springer.com/article/10.1007/s10846-014-0043-8)**
  (JIRS 78(2), 2015). Aircraft horizon profile vs **pre-generated reference
  profiles stored on a grid** — the precompute-side architecture of §5.
- **[SKYLINE2GPS](https://inria.hal.science/inria-00523997)** (Ramalingam,
  Bouaziz, Sturm, Brand, IROS 2010). Upward fisheye skyline, graph-cut sky
  segmentation (works day/night/rain), matched to coarse 3D city models;
  meter-level along urban trajectories. Demonstrates fine positional
  sensitivity of skylines in a confined search area.
- UAV DEM-matching context: [heightmap-gradient TAN with clustered particle
  filter, arXiv 2510.01348](https://arxiv.org/pdf/2510.01348) (2025).

### 3.6 Deep-learning era

- **PeakLens** (Fedorov, Frajberg, Fraternali et al., 2016–2017;
  [peaklens.com](https://peaklens.com/)) — mobile-grade CNN pixel-wise
  skyline extraction (~94% accuracy, 9 MB model, ~270 ms on a 2015 phone) +
  alignment to SRTM-rendered panoramas. State of practice for the
  segmentation front-end.
- **[CrossLocate](https://openaccess.thecvf.com/content/WACV2022/html/Tomesek_CrossLocate_Cross-Modal_Large-Scale_Visual_Geo-Localization_in_Natural_Environments_Using_Rendered_WACV_2022_paper.html)**
  (Tomešek, Čadík, Brejcha, WACV 2022;
  [code](https://github.com/JanTomesek/CrossLocate)). Learned cross-modal
  retrieval of photos against DEM-rendered views (semantics, silhouettes,
  **depth — which wins**) over all of Switzerland. Best open large-scale
  baseline.
- **[CMLocate](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12883)**
  (IET IP 2023) and **[Lan, Tang & Guo,
  2024](https://link.springer.com/article/10.1007/s11042-024-19189-6)**
  (horizon retrieval + hierarchical coarse-to-fine search: **95.8% success,
  ~40 m error over 183 km²**) — concrete modern accuracy-vs-area datapoints.
- Segmentation comparisons: [Ahmad et al., arXiv
  1805.08105](https://arxiv.org/pdf/1805.08105); shallow-learning skyline
  extraction [arXiv 2107.10997](https://arxiv.org/pdf/2107.10997);
  [LandscapeAR, ECCV 2020](https://github.com/brejchajan/LandscapeAR) (code).

### 3.7 Synthesis

- **Representation converges by regime.** For levelled/known-orientation and
  especially sea-level observers, everyone uses the **1D elevation-vs-azimuth
  horizon function with (circular) cross-correlation** — Cozman/Krotkov,
  Chiodini, Dumble & Gibbens, Grelsson, Naus & Wąż. Contour-word/embedding
  retrieval only pays at ≥10³ km² scales with unknown orientation/FOV. Edge
  maps + chamfer are for full-rotation search. Our known rotation collapses
  the problem to the 1D representation — the simplest regime.
- **Search converges too:** exhaustive/coarse-to-fine position grids with
  cheap 1D correlation up to ~10²–10³ km²; retrieval indexes beyond that;
  filters once a temporal sequence exists. Branch-and-bound is notably
  *absent* from this niche (an opportunity — §4.2).
- **Accuracy datapoints:** ~10 m (maritime panorama, Grelsson), ~40 m
  (183 km², Lan 2024), ~43 m (desert 360°, VISAPP 2020), few hundred m
  (Apollo 17, 1990s tooling). Tens of meters in a 1 km box is realistic.
- **Gaps we must handle ourselves:** (a) positional sensitivity for
  sea-level observers of *distant* terrain is barely quantified (the §1.2
  analysis fills this); (b) low-relief coastlines are degenerate along the
  viewing ray — interior ridges/coastline occlusion cues mitigate (Chen
  2015); (c) refraction/tide are unmodelled outside the navigation-journal
  thread — our §2.2 curvature patch and height nuisance handle this.

---

## 4. Literature: search & optimization techniques for the position sweep

The per-candidate cost (synthesize a skyline, compare to the observation) is
the expensive primitive; the cost surface over position is smooth-ish but
multimodal. Seven threads of literature bear on how to spend those
evaluations. Note first that the *exact* structure — grid search of
DEM-synthesized skylines against an observed panoramic horizon — already
exists in the planetary-rover literature: Stein & Medioni, "Map-based
localization using the panoramic horizon" (IEEE Trans. Robotics & Automation,
1995); Cozman & Krotkov's Viper for lunar/Mars rovers (ICRA 1997, Autonomous
Robots 2000); [Chiodini et al., "Mars rovers localization by matching local
horizon to surface DEMs"](https://ieeexplore.ieee.org/document/7999600/)
(MetroAeroSpace 2017). Those papers define the pipeline; the threads below
are about doing the search well.

### 4.1 Terrain-aided navigation (TERCOM / SITAN / point-mass filters)

The closest structural analog: match a measured 1D terrain profile against a
stored DEM to fix position.

- Golden, "Terrain contour matching (TERCOM)" (SPIE 1980) — batch MAD/MSD
  correlation over a search grid; robust to large initial error.
- Hostetler & Andreas, "Nonlinear Kalman filtering techniques for
  terrain-aided navigation" (IEEE TAC 1983) — SITAN: recursive EKF on local
  terrain slope; fast but false-fixes when initial uncertainty is large.
- [Bergman, Ljung & Gustafsson, "Point-mass filter and Cramér–Rao bound for
  terrain-aided navigation"](https://ieeexplore.ieee.org/document/650690/)
  (CDC 1997) and Bergman's 1999 [PhD
  thesis](https://www.rt.isy.liu.se/research/reports/Ph.D.Thesis/PhD579.pdf) —
  full Bayesian grid filter over position, achieving the CRLB; ancestor of
  particle-filter TAN ([Gustafsson et al. 2002](https://www.irisa.fr/aspi/legland/ref/gustafsson02a.pdf)).
- [Zhao et al., IEEE Sensors J. 2015](https://www.researchgate.net/publication/273394609_A_Novel_Terrain-Aided_Navigation_Algorithm_Combined_With_the_TERCOM_Algorithm_and_Particle_Filter)
  — hybrid: batch acquisition + recursive tracking (the TERPROM two-phase
  design).

**Transferable lesson:** the *acquisition-then-track* architecture — batch
correlation to find the basin, recursive/local refinement inside it. TERCOM's
profile is spatial (along-track) while ours is angular (azimuth sweep from a
point), so a single skyline is one very informative point-mass-filter
measurement update rather than a trajectory.

### 4.2 Branch-and-bound over pose space

- [Breuel, "Implementation techniques for geometric branch-and-bound matching
  methods"](https://www.sciencedirect.com/science/article/abs/pii/S1077314203000262)
  (CVIU 2003) — subdivide transformation space, bound the best achievable
  cost per cell, prune; globally optimal and adaptive.
- [Yang et al., "Go-ICP"](https://arxiv.org/abs/1605.03344) (ICCV 2013 /
  TPAMI 2016) — BnB over motion space with derived error bounds, local
  refinement nested inside to tighten the incumbent.

**Fit:** strong, because a rigorous per-cell bound exists for our cost. A
skyline point at range `r` moves at most `δ/r` radians in azimuth when the
observer moves `δ` — and the range image gives `r` per azimuth bin for free,
so each rendered candidate yields a Lipschitz bound over its neighborhood.
This certifies a no-miss search of the box with far fewer renders than a
dense grid. Bounds go loose only in azimuths with very close foreground
terrain (small `r_min`) — handled with robust per-bin costs.

### 4.3 Coarse-to-fine / multi-resolution grid search

- [Borgefors, "Hierarchical Chamfer Matching"](https://ieeexplore.ieee.org/document/9107/)
  (TPAMI 1988) — the canonical coarse-to-fine curve matcher.
- [Lewis, "Fast Normalized Cross-Correlation"](https://scribblethink.org/Work/nvisionInterface/nip.html)
  (1995) — sum-table NCC machinery.

**Fit:** the workhorse baseline and what the rover papers actually do. Coarse
grid spacing can be *chosen from the same parallax bound as §4.2* (nearest
skyline range sets the safe spacing), which turns naive coarse-to-fine into a
non-adaptive BnB with no missed-basin risk. For a 1 km box: ~100 m spacing
(121 renders) → keep top-k basins → 25 m → continuous local refinement.

### 4.4 Particle filters / Monte Carlo localization

- [Dellaert, Fox, Burgard & Thrun, "Monte Carlo Localization"](https://www.ri.cmu.edu/pub_files/pub1/dellaert_frank_1999_2/dellaert_frank_1999_2.pdf)
  (ICRA 1999); [Thrun et al., "Robust MCL"](https://www.sciencedirect.com/science/article/pii/S0004370201000698/pdf)
  (AIJ 2001) — including the standard treatment of expensive ray-cast beam
  models vs precomputed *likelihood fields*.
- Fox, "KLD-sampling" (IJRR 2003) — adaptive sample counts.

**Fit:** for a single photo, MCL degenerates to randomized grid search. It
becomes the right frame once there are *sequences* (a drifting/underway
vessel taking frames over time): the posterior stays multimodal until skyline
evidence disambiguates. The **likelihood-field trick transfers regardless**:
precompute per-azimuth horizon-angle maps over the box once, then candidate
evaluation is a lookup, not a render (see §5).

### 4.5 Surrogate / Bayesian optimization (expensive black-box)

- Jones, Perttunen & Stuckman, "Lipschitzian optimization without the
  Lipschitz constant" (JOTA 1993) — **DIRECT**: deterministic global search
  on a box, derivative-free, zero tuning.
- Jones, Schonlau & Welch, "Efficient Global Optimization of Expensive
  Black-Box Functions" (JGO 1998) — **EGO**: Gaussian-process surrogate +
  expected improvement; the canonical Bayesian-optimization reference.
- Hansen & Ostermeier (Evol. Comp. 2001) — CMA-ES; poor fit here (needs
  10³–10⁴ evaluations).

**Fit:** a 2–3D box with an expensive smooth-ish multimodal objective is the
textbook BO/DIRECT regime — tens-to-low-hundreds of evaluations, and GP
lengthscales can be set from the parallax physics. But it ignores the strong
problem structure (bounds, precomputation) the other threads exploit, and GP
stationarity is violated at occlusion "cliffs." Best as a model-free
fallback, or for the height/heading nuisance dimensions.

### 4.6 Topological / structural matching (the "topological search" thread)

This is the closest match to the intuition of searching *structure* rather
than metric values:

- **Persistence on the skyline itself.** 0-dimensional persistent homology of
  the 1D curve θ(az) ranks skyline peaks by persistence (prominence),
  giving a noise-robust, scale-free set of "significant peaks" — far more
  stable than thresholded local maxima. (PH used as a localization
  fingerprint: [IEEE IV 2021 for LiDAR](https://www.researchgate.net/publication/355836716_Persistent_Homology_in_LiDAR-Based_Ego-Vehicle_Localization);
  as terrain morphology descriptor: [RSE 2020](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301863).)
  Note that *topographic prominence*, beloved of mountaineers, **is** the
  0-D persistence of the elevation function — the concepts align exactly.
- **Qualitative localization by cyclic order of landmarks.**
  [Qualitative place signatures of visible landmarks](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2348736)
  (IJGIS 2024): a position's signature is the cyclic azimuth order (+
  qualitative angle classes) of visible landmarks; the map partitions into
  cells of constant signature and retrieval is cyclic-sequence matching.
  Related: qualitative angle-order navigation (Levitt & Lawton lineage;
  [Mor & Indelman 2020/2023](https://arxiv.org/pdf/2302.08735)). Cozman &
  Krotkov's rover work already used peak azimuth ordering + elevation angles
  as the match primitive.
- **Ridge graphs / surface networks / contour trees.** Pfaltz "surface
  networks" (1976); [survey of terrain topology structures](https://doi.org/10.3390/encyclopedia5030098)
  (2025); [Reeb-graph metrics](https://arxiv.org/pdf/2110.05631); merge-tree
  distances (Beketayev et al. 2014) — machinery for comparing the observed
  skyline's merge tree against candidates', robust to smooth deformation.

**Fit:** within a 1 km box the cyclic order of distant peaks changes only
when the observer crosses an aspect-graph boundary (an occlusion event or an
azimuth-order swap), so the box partitions into a handful of
constant-signature cells. That makes topology a superb **gating/pruning
stage** — reject cells whose peak signature contradicts the observation, at
near-zero cost — and a poor *standalone* localizer (it yields a cell, not a
point). This is, we believe, the productive reading of "optimization
resembling topological search": use topology to prune, metric matching to
refine. At larger search scales (10–100 km) the signature index becomes the
retrieval mechanism, as in Baatz et al.'s contour-word approach (§3).

### 4.7 1D signal matching specifics (the inner-loop cost)

- **Circular/FFT correlation** over azimuth scores all heading offsets at
  once — with known orientation it is not needed for search, but is a cheap
  way to co-estimate the compass-bias nuisance parameter
  ([Nagy, PFG 2020](https://link.springer.com/article/10.1007/s41064-020-00093-1)
  does azimuth refinement by skyline matching).
- **Banded dynamic time warping** absorbs small residual warps from
  calibration/DEM/refraction error that pointwise L2 punishes
  (used in the rover VIPER line; [Grelsson et al., JFR 2016](https://www.researchgate.net/publication/283967541_Highly_Accurate_Attitude_Estimation_via_Horizon_Detection)).
- **Directional chamfer / distance-transform costs**
  ([Liu et al., FDCM, CVPR 2010](https://pure.johnshopkins.edu/en/publications/fast-directional-chamfer-matching/);
  [SKYLINE2GPS, IROS 2010](https://inria.hal.science/inria-00523997)):
  rasterize the *observed* skyline once into a (azimuth × elevation) distance
  transform; each synthetic candidate is then scored by O(n) lookups. The
  render becomes the only expensive step.
- **Information weighting:** flat sea-horizon spans carry no positional
  information — weight azimuth bins by local skyline variance (and by
  parallax sensitivity `1/r`, §5).

---

## 5. Proposed pipeline

Combining §2 (what the horizonator provides), §3 (what the field converged
on) and §4 (how to search): a Chiodini/Grelsson-style grid pipeline, with a
topological gate and a parallax-bound-derived grid, using the horizonator as
the synthesis engine.

**Stage 0 — observation front-end (once per frame).**
Undistort the image with the known intrinsics; map every pixel to a ray and
every ray through the known rotation to absolute (azimuth, elevation).
Extract the sky/terrain boundary (gradient + dynamic-programming seam for a
first implementation; a PeakLens/HorizonNet-style CNN when robustness to
haze/sun/sea-spray matters). Resample to a uniform azimuth grid, e.g.
0.05°/bin, giving **θ_obs(az) over the camera's azimuth span**, with a
per-bin confidence. This levelling step is also where nonzero pitch/roll is
absorbed, so the horizonator's always-level rendering is matched by
construction.

**Stage 1 — synthesis engine (once per search area).**
`horizonator_init()` centered on the box, radius large enough to include all
visible terrain (~50–90 km at sea). Apply the two §2.2 patches (curvature
term in `vertex.glsl`; `viewer_z` plumbed through the Python `render()`).
Per candidate: `render()` → skyline θ_syn(az) = per-column topmost valid
range pixel, plus per-bin range r(az). Optionally precompute **per-azimuth
horizon-angle maps** over the box (the MCL "likelihood field" trick, §4.4) so
repeated queries in the same box become lookups; or swap in a 1D CPU
ray-marcher with the same curvature model for headless batch runs.

**Stage 2 — per-candidate cost.**
Rasterize θ_obs once into a directional distance transform (§4.7). Candidate
cost = Σ over azimuth bins of a robust (Huber/truncated) distance between
θ_syn and the DT, with per-bin weights `w(az) = confidence × local skyline
variance × 1/r(az)`: flat sea-horizon spans get ~zero weight, near/steep
terrain gets more. Allow a **single global azimuth offset ψ** (compass bias)
via 1D FFT correlation — one cheap extra dimension estimated in closed form
per candidate. Optionally a ±1° banded DTW pass for residual warp.

**Stage 3 — search over the box.**
1. *Topological gate:* compute persistence-ranked peaks of θ_obs (=
   prominence of skyline maxima); on a very coarse candidate set (~9–25
   renders), compute each candidate's peak signature (cyclic azimuth order of
   persistent peaks). Discard regions whose signature cannot match. In a
   1 km box this typically leaves one or two contiguous cells.
2. *Coarse grid:* spacing set by the parallax bound — the skyline moves ≤
   δ/r_min radians for a δ move, so choose spacing where the predicted shift
   stays within the cost function's basin (~100 m when nearest skyline
   terrain is ≥5 km). 1 km box → ≤121 renders.
3. *Refinement:* keep top-k basins, descend to 25 m grid, then fit a local
   quadratic (or Gauss–Newton on the per-bin residuals) for a sub-grid
   minimum, jointly with ψ (and h if the observer height is uncertain beyond
   ~2 m).
4. *(When the box grows to dead-reckoning scale, 10–100 km):* replace step 2
   with branch-and-bound using the same per-cell parallax bounds (§4.2) —
   novel in this niche and cheap to add — or the Baatz-style signature index
   for retrieval at even larger scales.

**Stage 4 — fix and uncertainty.**
Laplace approximation at the minimum (inverse Hessian of the cost) → a
position covariance ellipse. The ellipse's long axis will point along the
mean viewing ray when only distant low terrain is visible — an honest report
of the coastal degeneracy. Publish the result as a *terrestrial fix*
compatible with the `celestial-navigation` toolkit (it is exactly a dense
resection), and — with multiple frames from a moving vessel — as a
measurement factor in a GTSAM factor graph fused with dead reckoning and
celestial sights, or as the measurement update of a point-mass/particle
filter (§4.1, §4.4).

**Failure modes to design for:** clouds truncating ridge tops (robust cost +
confidence weights), sun glare and sea clutter at the horizon (front-end),
vessel roll smearing during exposure (IMU timestamping), DEM canopy bias
(prefer Copernicus GLO-30), and the all-sea-horizon case (detect: total
skyline variance below threshold → report "no fix possible," like a
navigator would).

---

## 6. Experiment plan (1 km × 1 km, observer at sea)

**E0 — Renderer validation.** After the curvature patch: render from a
sea-level viewpoint toward a known coast; check (a) the sea horizon dips by
`√(2h/R_eff)` and sits at the right distance, (b) terrain beyond the
geometric limit is hidden, (c) skyline elevation angles of surveyed peaks
match ephemeris-grade computation to <0.5 mrad. This also produces the
first regression tests for the patch.

> **E0 results** (`experiments/e0_validate.py`, Bodrum–Kos test area,
> SRTM 3″ from the AWS skadi mirror — note those tiles contain bathymetry,
> clamped to 0 like `dem.c` does): **16/16 checks pass.** Sea-horizon dip
> matches `√(2h/R_eff)` to <0.15 mrad (GL, quantization-limited) and
> <0.01 mrad (ray-marcher) for h = 2–20 m; horizon distance matches
> `√(2·R_eff·h)` exactly at the marcher's step resolution. The GL skyline
> agrees with an independent NumPy ray-marcher to 1.8 mrad RMS (0.9 mrad
> median) over a full 360° terrain panorama — quantization (0.87 mrad at
> 0.1°/px) plus mesh-vs-bilinear differences. On distant terrain where
> curvature matters (>1 mrad effect), the GL render matches the
> curved-Earth model 2.7× better than the flat-Earth model, confirming the
> patch does what it should.

**E1 — Synthetic closed loop (render-vs-render).** Choose 2–3 real sites
with different character — e.g. an archipelago (many islands, near+far), a
single mountainous island seen broadside at 20–40 km, and a low featureless
coast (the hard case). Ground truth: render θ_obs at a random position in
the box (viewer h = 2–20 m). Recover position with the §5 pipeline. Map the
**full cost surface over the box** (this is cheap and is the most
informative artifact: basin width, multimodality, coastal degeneracy made
visible). Metrics: CEP50/CEP95, ellipse orientation vs coast geometry.

> **E1 results** (`experiments/e1_closed_loop.py`, 20 ground-truth
> positions per site, viewer z = 5 m, 0.1°/px panoramas, coarse-to-fine
> search on a 25 m lattice with quadratic sub-grid refinement; ~0.27 s per
> candidate render on software GL, all solves served from a precomputed
> 41×41 lattice):
>
> *Site A — mid-strait between the Bodrum peninsula and Kos (land in 80% of
> azimuths, mean land range 14.5 km):*
>
> | config | CEP50 | CEP95 | max |
> |---|---|---|---|
> | clean, 360° | **7.2 m** | 26.6 m | 29.8 m |
> | clean, 90° FOV | 10.2 m | 58.0 m | 80.7 m |
> | 1 mrad noise, 360° | 7.1 m | 27.1 m | 30.1 m |
> | 0.2° heading bias, 360° | 23.4 m | 45.5 m | 48.4 m |
>
> *Site B — open water SW of Kos (land in 24% of azimuths, in one NNE–ENE
> sector, mean land range 23.8 km):*
>
> | config | CEP50 | CEP95 | max |
> |---|---|---|---|
> | clean, 360° | 13.6 m | 46.3 m | 46.5 m |
> | clean, 90° FOV | 22.2 m | 50.1 m | 56.7 m |
> | 1 mrad noise, 360° | 15.7 m | 46.1 m | 51.6 m |
> | 0.2° heading bias, 360° | 68.2 m | 85.9 m | 90.5 m |
>
> Every §1.2 prediction checks out quantitatively: GPS-class accuracy in
> the strong-geometry case; 1 mrad of random skyline noise is almost
> invisible (hundreds of azimuth bins average it away); the heading-bias
> error matches `d·ε` (14.5 km × 3.5 mrad ≈ 50 m worst-case at site A,
> 23.8 km × 3.5 mrad ≈ 83 m at site B — measured 48 m and 90 m max);
> and the site-B cost surface shows exactly the predicted along-ray
> degeneracy, a valley elongated along the mean line of sight toward the
> single visible coast (see `experiments/out/e1_B-offshore.png`). The cost
> surfaces are unimodal over the whole 1 km box at both sites — a single
> coarse-to-fine descent suffices here, and the topological gate of §5
> only becomes necessary at larger search scales.

**E2 — Noise ablations on E1.** Inject, one at a time and combined: heading
bias 0.1–2° (expect bias ≈ d·ε, confirming §1.2), skyline extraction noise
0.5–2 mrad, observer height error ±2 m, refraction coefficient k ∈
[0.10, 0.20], tide ±2 m, DEM degradation (SRTM3 vs SRTM1 vs Copernicus;
add canopy-height noise), cloud truncation of the top n% of ridges, and FOV
reduction 360° → 90° → 40° (the Foucher 2025 question). Output:
accuracy-vs-nuisance curves; identifies the error budget's dominant terms.

> **E2 results** (`experiments/e2_ablations.py`, figure
> `experiments/out/e2_ablations.png`; CEP50/CEP95 in meters, 20 trials per
> point; site A baseline 7/27 m, site B 14/46 m). The error budget, ranked:
>
> 1. **Heading bias dominates, and the mitigation kills it.** Naive solves
>    scale linearly with the bias: at 1°, 99 m (A) / 345 m (B); at 2°,
>    195 m / 587 m. Site B tracks the predicted `d·ε` (24 km × ε) almost
>    exactly; site A sits ~2.5× *below* its prediction — with land on
>    opposite bearings, a rigid azimuth shift makes contradictory demands
>    that partially cancel, so two-sided geometry is intrinsically more
>    bias-tolerant. Co-estimating a single azimuth-offset nuisance
>    (`skyline.cost_azshift`, §5 stage 2) **fully restores the baseline at
>    every bias level up to 2°** — 8/26 m (A), 14/46 m (B). Compass quality
>    stops mattering; the mitigation should simply always be on.
> 2. **Cloud truncation is the main environmental risk.** Truncating the
>    top 10% of the land skyline is free (7/17 m at A); 25% is mild
>    (10/65 m); at 50% the solve breaks down at site A (35 m CEP50 but
>    756 m CEP95 — the surviving low skyline is ambiguous). A per-bin
>    confidence weight from the sky-segmentation front-end (§5 stage 0)
>    is the right defense, plus refusing a fix when too little skyline
>    variance survives.
> 3. **Random skyline noise is a non-issue**: flat to 2 mrad, and only
>    4 mrad (2× the pixel quantization) shows at all — hundreds of azimuth
>    bins average it away.
> 4. **Height/tide mismatch of ±2 m is a non-issue** (within noise of the
>    baseline at both sites): the skyline barely moves with observer
>    height compared to its motion with horizontal position.
> 5. **Refraction-coefficient mismatch k′ ∈ [0.10, 0.20] is negligible**
>    (≤0.2 mrad at 40 km, no measurable CEP change), justifying the
>    fixed k = 0.13 in the renderer. (Anomalous refraction/mirage
>    conditions remain out of scope.)
> 6. **DEM source mismatch is small**: observations rendered from 1″ SRTM
>    matched against the 3″ lattice cost ~1 m at both sites (8/30 m at A,
>    15/33 m at B) — encouraging, though both are the same SRTM family;
>    an independent DEM (Copernicus GLO-30) remains future work.
> 7. **FOV**: site A degrades gracefully (10 m CEP50 even at 40°, though
>    CEP95 reaches 94 m as some solves get fragile); site B, whose land
>    all sits in one ~90° sector, loses accuracy once the FOV clips that
>    sector (36/155 m at 40°). Matches Foucher et al.'s limited-FOV
>    findings qualitatively.

**E3 — Search-strategy comparison.** On the E1/E2 cost surfaces, compare:
dense 25 m grid (reference), coarse-to-fine (§5), DIRECT, GP-based BO, and
BnB with parallax bounds. Metrics: renders-to-solution, miss rate of the
global basin. At 1 km this mostly documents that coarse-to-fine is enough;
rerun at a 20 km box to see the ranking change and the topological gate's
pruning factor.

> **E3 results — the box scaled ×100** (`experiments/e3_scale.py`, figure
> `experiments/out/e3_scale.png`). A **100 km × 100 km** box
> (Dodecanese/SE Aegean, 90% sea), searched with the native C ray-marcher
> (below) and a sea-masked hierarchical scheme: 2 km coarse grid over the
> 2 336 at-sea candidates → non-max-suppressed top-15 seeds → 500/125/25 m
> refinement + sub-grid quadratic; GL-rendered observations at 15 random
> at-sea positions, z = 5 m.
>
> - **15/15 trials succeed, CEP50 15.8 m, worst 61 m** — the same
>   accuracy class as the 1 km box, now over 10 000 km².
> - **The true basin was the single lowest-cost coarse candidate in every
>   trial** (L0 rank 0/2336), with the best basin beating the runner-up
>   by ≥29% in cost. In this region the skyline is globally
>   discriminative: there is no confusable stretch of coastline, even for
>   the trial with land in only 7% of its azimuths. The feared
>   multimodality did not materialize at this scale — the top-K/NMS
>   machinery, the §4.2 branch-and-bound and the §4.6 topological gate
>   remain insurance rather than necessity here. (Regions with repetitive
>   coastal topography may still need them; that is a site-dependent
>   question the L0 margin statistic now measures directly.)
> - **Cost: 2 870 skyline evaluations ≈ 10.6 s per cold fix** on the
>   4-core build machine (8.7 s coarse + 1.9 s refine). A dense 25 m grid
>   over the same box would need 16 M evaluations (~20 h) — the hierarchy
>   is a ~5 600× reduction at zero measured miss rate.
>
> Amendment to the E3 plan: at 100 km the cost landscape proved clean
> enough that the planned DIRECT/BO/BnB comparison is moot in this region;
> the informative remaining comparison is L0 grid pitch (2 km was safe;
> how coarse can it go before rank-0 breaks) and repeating in a
> low-relief, repetitive-coastline region.

**E3b — On-device (Raspberry Pi CM5) sizing.** Prototyped in
`experiments/fastmarch.c` + `skyline.CMarcher`: a C/OpenMP ray-marcher
(one ray per azimuth over a float32 DEM mosaic, bilinear sampling, the
same curvature model as the patched shader, division/trig-free inner
loop), agreeing with the NumPy reference to 0.02 mrad RMS. Measured on the
4-core x86 build container: **13.8 ms/skyline single-thread,
4.4 ms on 4 threads** (3600 azimuths × 443 steps to 40 km) — ~100× the
NumPy marcher, ~60× the software-GL render. Projected to the CM5's 4×
Cortex-A76 (×2.3 per-core factor, NEON via `-mcpu=native`):
~10 ms/skyline. Per fix:
>
> | scenario | evaluations | CM5 time |
> |---|---|---|
> | 1 km box, coarse-to-fine | ~106 | **~1 s** |
> | 100 km box, hierarchical | ~2 870 | **~25 s** |
> | tracking mode (precomputed lattice around the last fix) | cost lookups only | **≪1 s, ~1 Hz** |
>
> Plus a sky-segmentation front-end (ms for a classic gradient/seam
> method, ~100–300 ms for a small CNN) and a one-time DEM-mosaic load
> (~50 MB for 3°×3° at 3″). The full OpenGL renderer is not needed on the
> device (and its `#version 420` shader exceeds the Pi GPU's GLES 3.1
> anyway) — it remains the ground-truth/validation tool.

**E4 — Real imagery.** Phone or camera photos from a ferry/boat position
with GPS + compass logged (or harbor webcams with known mounts as a poor
man's version): run the full pipeline, compare to GPS. Expect the front-end
(sky segmentation over water, haze) to be the pain point, per the
literature.

> **E4 first attempt** (`experiments/e4_real.py`, figure
> `experiments/out/e4_real.png`): a real phone photo from the Gulf of
> Akbük coast (2026-08-10), with a Theodolite-app frame supplying attitude
> (heading 037° true as a prior, altitude 62 ft = 18.9 m); GPS was used
> only to center a **25 km² (5 km × 5 km) search box**. Because the
> original image file was not available to the solver, the skyline was
> **hand-digitized** (~30 points, ±6 mrad) — an important caveat on the
> numbers.
>
> - **Result: 320–770 m from the GPS position**, depending on the assumed
>   camera FOV (solved as an outer parameter, 62–78°; best fit near
>   62–66°, 324 m at 66°). Solve time: 29 s for the full FOV sweep
>   (native marcher, 4 cores).
> - **Lesson 1 — near-field DEM occlusion.** At the true position the
>   predicted skyline was poisoned (25 mrad RMS) by SRTM smearing the
>   observer's own coastal bluff into blocking terrain at 150–500 m
>   range; a camera at a bluff edge sees over its local ground, but the
>   DEM does not know that. Masking ranges < 1 km restored the fit to
>   digitization-noise level (6 mrad). For land-based observers this mask
>   (or an explicit camera-above-local-terrain model) is essential; pure
>   sea observers don't have the problem.
> - **Lesson 2 — FOV must be known, not searched.** With a single-sided
>   ~65° view sector, position trades almost linearly against azimuth
>   scale: **~130 m of position slide per degree of FOV error**, and
>   ±6 mrad digitization noise cannot break the tie between FOV
>   hypotheses. The original file's EXIF focal length (or a one-time
>   calibration) removes this entirely — with intrinsics truly known, as
>   the study assumes, this axis vanishes.
> - **Lesson 3 — the heading prior was ~4° off** (the raw photo was
>   framed differently from the Theodolite frame); the azimuth-shift
>   co-estimation absorbed it, as designed (E2).
>
> With automatic pixel-level skyline extraction (±0.5–1 mrad) and EXIF
> intrinsics, the E2 site-B numbers (~15–50 m) are the expectation for
> this geometry. The script accepts `--csv` with properly extracted
> skyline points to rerun against the same box.
>
> **E4b — second photo and a joint solve** (`experiments/e4b_dual.py`).
> A second frame from the same spot (iPhone 17 Pro 0.5×, heading 352°
> true; the site is the south shore of Lake Bafa, so the "horizon" is
> the far lakeshore and the elevation datum was co-estimated rather than
> anchored to a sea-horizon dip) enabled a two-photo joint solve with a
> proper pinhole model (106° FOV from the 13 mm-equiv lens spec — at
> that FOV the linear pixel→angle mapping is invalid). Outcome: the
> joint fix **degraded** to 3.4 km, and the diagnosis is the useful
> part: cross-checking the two hand-digitized skylines against each
> other showed the ultrawide digitization to be internally inconsistent
> by ~2× in elevation scale (the same conical hill reads 124 mrad in
> the 0.5× frame vs 56 mrad in the 1.0× frame and ~52 mrad in the DEM),
> an error neither the elevation-offset nor the heading nuisance can
> absorb. Visual (eyeball) digitization breaks down on an ultrawide's
> strongly nonlinear projection; the 1.0× photo alone, whose
> digitization is DEM-consistent, keeps the ~320 m result. Conclusion
> unchanged and sharpened: pixel-level extraction from the original
> files with EXIF intrinsics is the mandatory front-end — hand
> digitization is only usable for narrow-FOV frames, and never for
> wide-angle ones.
>
> **E4c — the automatic front-end, built and validated end-to-end**
> (`experiments/extract.py`, `experiments/skyfix.py`,
> `experiments/e4c_synth.py`). Since no usable external photo source is
> reachable under this environment's egress policy (Wikimedia blocked,
> GeoPose3K blocked, open-S3 YFCC images are 500 px with EXIF stripped),
> the pipeline was validated on photo-realistic composites with real
> EXIF: GL-rendered ground truth + sky gradient, range-dependent haze,
> sea/terrain coloring, camera pitch/roll, sensor noise, and
> FocalLengthIn35mmFilm / GPSImgDirection / GPSAltitude written into the
> JPEGs. Results over a 5 km box with a randomly offset center,
> ~11 s/fix: **40–62 m at all four sea-observer cases (0.7–0.9 mrad
> extraction residual, zero recovered heading offset)**, 83 m at one
> land case; one land case fails through the 1 km near-field mask
> (its view contains genuine terrain at 100–800 m) — soft near-field
> weighting is the open item for land-based observers.
>
> Three transferable findings from getting there: (1) **extraction
> method matters more than anything downstream** — global sky-color
> models fail on graded/hazy skies (extrapolation drift reads as
> terrain), and pure edge detectors lock onto a crisp sea horizon below
> a faint distant ridge; the working formulation is *local linear
> continuation* (predict each row from the rows just above; boundary =
> first sustained deviation), which handles both failure modes, each of
> which was actually hit during validation. (2) **The elevation
> nuisance must stay tight** (±10 mrad residual around a pitch prior
> good to ~0.5°): freeing it wide discards the absolute-elevation
> information that pins range, and accuracy collapses even with perfect
> extraction. (3) The pitch/roll priors an IMU provides are not
> optional garnish — they are what makes a single-frame fix
> well-conditioned. `skyfix.py` is ready to run on real originals:
> `python3 skyfix.py IMG --center LAT,LON --pitch P --roll R`.
>
> **E4d — real photos at last: the CH1 alpine benchmark** (the
> Baatz/Saurer ECCV'12 query set, pushed to `celestial-navigation:main`;
> `experiments/e4d_ch1.py`). Calibrated focal lengths, ground-truth
> positions, hand-made sky masks — but *no attitude information*: unknown
> heading, pitch and roll, i.e. deliberately outside this project's
> known-extrinsics design envelope. Results on the trial subset (5 km
> boxes, solving with the ground-truth masks to isolate matching from
> extraction): with a ±6° pitch window, photos whose pitch happens to be
> small localize to ~50 m; opening the search to full rotation (pitch
> ±20°, roll ±3°, heading free) fixes some failures (2.4 km → 336 m,
> 2.9 km → 257 m) but makes others slide kilometers away *with excellent
> residuals* (5–6 mrad). The diagnosis is structural, not a bug: with a
> completely free rotation, 1D skyline matching over a 5 km alpine box is
> under-constrained — wrong positions fit well. This cleanly delineates
> the method's domain: **attitude priors (IMU pitch/roll to ~0.5°, a
> compass heading prior) are what make single-frame skyline fixes
> well-posed**; without them one needs the discriminative
> contour-descriptor + verification machinery of Baatz et al. — a
> different algorithm, not a parameter change. The maritime instrument
> this study targets sits firmly on the well-posed side.
>
> Full 12-photo run (moderate ±6° pitch window, 5 km boxes, ~95 s/photo):
> **median 2.4 km with the automatic extractor, 2.1 km with the
> ground-truth masks** — the two agree closely on nearly every photo,
> which is the one genuinely positive finding: on real alpine imagery
> the automatic extractor is *not* the bottleneck (its errors track the
> hand-made masks within ~20%); the failures are matching-side, caused
> by the missing attitude priors. Successes when pitch happens small:
> 45–430 m. n=12 confirms the trial verdict at scale.

**E5 — Integration.** Wrap the fix + covariance as (a) a terrestrial-fix
input to `celestial-navigation` (it already has a terrestrial/bearing
mode), and (b) a custom GTSAM unary factor on pose, enabling fusion of
skyline fixes, celestial sights, and dead reckoning along a track —
at which point the §4.4 particle/point-mass machinery becomes the natural
sequential estimator.

> **E5 results** (`experiments/skyline_factor.py`, `e5_fusion.py`,
> `celestial-navigation:skyline_fix.py`; figure
> `experiments/out/e5_fusion.png`). Implemented both integrations:
> (a) `skyline_fix.py` in the celestial-navigation repo converts skyfix
> JSON into toolkit objects (a `LatLonGeodetic` estimated position and a
> `Circle` of position); (b) a GTSAM `CustomFactor` carries the fix and
> its anisotropic covariance into a Pose2 factor graph. Demo: a 2-hour
> simulated passage through the Bodrum–Kos strait with +3% log bias and
> 1.5° compass bias, a skyline fix every 15 min solved over a 2 km box
> around the *current dead-reckoning estimate* (0.6 s/fix, native
> marcher). **Dead reckoning alone: 416 m mean, 787 m final. Fused:
> 46 m mean, 9 m final** — intermittent skyline fixes bound DR drift
> indefinitely, closing the loop the study set out to close. Two
> modeling lessons from getting the graph right: odometry noise must
> price in unmodeled log/compass biases (an over-trusted odometry chain
> outvotes even 15 m fixes — the wrong answer becomes the graph
> optimum), and the Laplace covariance heuristic is ~10× pessimistic
> against measured accuracy and needs an empirical calibration factor
> (documented in `laplace_cov`).

**Implementation order in this repo:** (1) `vertex.glsl` curvature patch +
`viewer_z` in the Python API (small, self-contained); (2) skyline extraction
from the range image + 1D cost module in Python; (3) E0/E1 scripts; (4) the
search loop. Steps 2–4 are pure Python on top of the existing renderer.

**Status: steps 1–4 and experiments E0/E1/E2/E3 are implemented** — see
`experiments/` (`skyline.py`, `fastmarch.c`, `e0_validate.py`,
`e1_closed_loop.py`, `e2_ablations.py`, `e3_scale.py`, the plot scripts,
`fetch_dems.py` and the README there), with figures in `experiments/out/`.
Remaining: an independent DEM family (Copernicus GLO-30), canopy-height
noise, an L0-pitch sweep and a repetitive-coastline region for E3, and
E4–E5. Two amendments from the results: azimuth-offset co-estimation
should be part of the default cost (E2), and the native ray-marcher —
not the GL renderer — is the production solver engine (E3b).
