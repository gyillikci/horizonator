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
>
> **E5b — the live instrument loop** (`experiments/skynav.py`,
> `e5b_live.py`). The deployable form: iSAM2 fuses each odometry leg
> (0.1 ms/update) and each skyline fix (~0.7 s solve+fuse) as they
> arrive, and emits the fused position as NMEA 0183 $GPGGA/$GPRMC
> sentences a chartplotter consumes like GPS. Replaying the E5 passage
> as a stream: **78 m mean / 10 m final** causally (vs the batch
> smoother's 31 m / 7 m — the honest filtering-vs-smoothing gap). One
> convention trap worth recording: compass headings (CW from north)
> fed directly into Pose2.theta (CCW from east) made incremental iSAM2
> diverge by tens of km while batch LM silently absorbed the same bug
> by rotating the whole chain — the conversion now happens at the
> SkyNav API boundary, and fixing it improved the batch result too
> (46→31 m mean).
>
> **Margin-based fix rejection** (follow-up to the perturbation study of
> the CH1 failures): `skyfix` and `SkyNav.take_fix` now compute the
> E3-style basin margin (relative cost gap between the two best
> non-max-suppressed coarse basins) and refuse the fix below a threshold
> (default 0.15; genuine fixes measured ≥0.29 in E3). Validated: the
> healthy synthetic case passes at margin 40.8; the E4c land-case
> failure — previously thought unfilterable — is caught at margin 0.07
> and correctly reported as NO FIX; the E5b live-loop results are
> unchanged (all sea fixes pass). Stable single-impostor failures with a
> genuinely large margin remain the residual risk, gated in practice by
> the factor graph's dead-reckoning consistency.
>
> **E4e — the gates audited against all 203 CH1 photos**
> (`experiments/e4e_gate_audit.py`, `out/e4e_audit.csv`; GT-mask
> observations, full heading search, 5 km boxes). Confusion matrix:
> TRUE-ACCEPT 29, FALSE-ACCEPT 29, CAUGHT 134, OVER-CAUTIOUS 11. Read
> two ways: the gates correctly convert **82% of wrong solves (134/163)
> into honest no-fixes** — but among the 58 solves that pass all four
> gates, **half are still wrong** (median 1.25 km), and raising the
> margin threshold does not separate them (false share stays ~50% up to
> margin 1.0, where availability collapses to 5 fixes). The impostors'
> margins (median 0.41) overlap the genuine fixes' — from the inside
> they are indistinguishable, because their fits are genuinely good.
> **Conclusion, stated precisely: internal quality gates alone cannot
> guarantee "genuine convergence or no convergence" in the
> attitude-free regime.** They make failures mostly-detected, not
> impossible. The guarantee-shaped claim survives only inside the
> design envelope — attitude priors (where E1–E3's 35+ solves produced
> zero impostors and ≥0.29 margins for every true basin) — and, when
> underway, the factor graph's dead-reckoning consistency as the
> external cross-check. This is why the instrument carries an IMU.
>
> **E4f — the attitude-prior A/B on the same 203 photos**
> (`experiments/e4f_ab_audit.py`, `out/e4f_audit.csv`). Identical to
> E4e except the attitude is instrumented: per photo, a reference
> attitude is solved once at the ground truth, corrupted with realistic
> instrument noise (σ 1° heading, 0.5° pitch), and the position solve
> searches only ±2° heading / ±1° pitch around that prior. Confusion
> matrix: TRUE-ACCEPT 62, FALSE-ACCEPT 40, CAUGHT 91, OVER-CAUTIOUS 10.
> Against E4e: **availability doubles** (58→102 accepted, 29→62 of them
> correct), median error over all 203 solves **halves** (1.9 km→791 m;
> <500 m solves 20%→35%). At the default margin threshold the
> false-accept share falls only 50%→39% — but the *composition*
> changes: half the E4f false-accepts (21/40) lie at 500–1000 m,
> adjacent-cell picks on the 250 m coarse grid straddling the 500 m
> correctness line, not distant impostors. More importantly, **the
> margin statistic now separates genuine from impostor** (median 0.96
> for true accepts vs 0.30 for false — in E4e they overlapped): raising
> `--min-margin` trades availability for integrity along a usable
> curve — at 0.5: 56 accepted, 20% wrong, 3 beyond 1.5 km; at 1.0: 30
> accepted, 7% wrong, none beyond 1.5 km. So the attitude prior does
> two things the free search cannot: it stops wrong basins from
> rotating into good fits (restoring meaning to the margin), and it
> halves the raw position error. What it does not do, on this
> mountain-terrain dataset with its repetitive ridgelines, is make a
> *single* photo unconditionally trustworthy — high-confidence
> single-fix operation needs the tightened margin threshold, and full
> integrity still comes from fusing successive fixes against dead
> reckoning (E5). The sea-observer envelope (E1–E3, E4c), with its
> anchor to the flat horizon, remains the benign regime.
>
> **E4g — sea-horizon auto-levelling** (`extract.sea_horizon_attitude`,
> `skyfix --auto-level`, `e4g_autolevel.py`). The Grelsson-style
> levelling stage in closed form: for a known camera height the sea
> horizon sits at the exactly known dip √(2h/Reff) below level, so the
> horizon line in the image — v = tan(−dip−pitch)·hypot(u,f) − roll·u,
> linear in (pitch, roll) — is a drift-free attitude reference. Fit by
> 2-point RANSAC with two guards: a one-sided veto (boundary points
> below a candidate line are impossible for a true sea horizon, which
> kills lines fitted to elevated ridges while real sea is visible
> lower), and a photometric water check (the band below a genuine
> horizon is water, darker than the sky above it — a straight distant
> ridge under haze, the navigator's "false horizon", fails it).
> Implementing this exposed and fixed a systematic extractor bias: the
> sustained-deviation trigger read crisp boundaries up to ~7 px early
> (its forward window already contained them) — invisible under the
> ±10 mrad offset window, fatal under ±2 mrad. A/B on the E4c sea
> cases, same composites and boxes (A: perfect IMU prior, ±10 mrad
> offset; B: **no attitude prior at all**, sea-horizon levelling,
> ±2 mrad): where the horizon is accepted (2/4 cases) attitude is
> recovered to ≤0.08° in both pitch and roll and the tight window
> pays exactly as E2 predicted — strait3 matches the prior run's 26 m
> with position σ nearly halved (76,78)→(44,42) m; offshore beats it
> outright, 337 m→70 m with σ halved, because the wide offset window
> had let the solve wander in range. Where no sea horizon is in view
> (strait1) or only a false one (strait2, correctly rejected by the
> water check) the estimator declines and skyfix falls back to the
> priors. In deployment both are used together — IMU prior as the
> fallback, horizon levelling as the upgrade whenever open water is
> visible, which for the study's at-sea observer is nearly always.
> The extractor fix also refreshes the E4c baseline itself
> (`out/e4c_results.json`): sea cases 23–239 m under the wide window
> (the high end is the range wander), and the bafa land case now
> reports INCONCLUSIVE through the gates instead of a wrong fix.
>
> **E4h — second instrumented Bafa pair** (`e4h_bafa2.py`). Two photos
> seconds apart from the E4a bluff, now with full Theodolite attitude
> (headings 037/014 true, pitch +5.3/+5.8, roll −0.9/0.0, z 18.3 m
> from GPS altitude; 1.0× lens confirmed by the level-line position the
> pitch implies). The photos again arrived file-stripped, so the
> skylines were hand-digitized from the rendered images — coarser than
> E4a's session (~±8 mrad structure error vs the ~2 mrad the geometry
> needs). Solves (absolute and waterline-referenced, single and joint)
> land 0.5–2.3 km from the GPS with basin margins 0.01–0.33 — mostly
> BELOW the 0.15 gate, several boundary-railed: skyfix would return
> INCONCLUSIVE rather than a confident wrong fix, which is the system
> working as specified on input it cannot trust. Residuals sit at
> 7–9 mrad (digitization noise) against the 1–3 mrad the automatic
> extractor achieves on real files (E4c/E4d). The blocker is purely
> input fidelity: with the two originals pushed to a repo (EXIF
> intact, full resolution) this pair becomes the definitive
> in-envelope real-photo test.
>
> Two more pairs from the same spot extended the set: C = 8.0×
> telephoto of the conical hill (10.3° FOV, heading 011, pitch +0.2)
> and D = 2.0× looking ESE down the lake (39.7° FOV, heading 100) —
> nearly 90° of bearing spread. The telephoto proves the fidelity
> argument from the other side: at 10.3° FOV the same hand digitization
> is worth ~1 mrad instead of ~8, and photo C alone lands **250 m**
> from the GPS (stable at 250–540 m across a ±20% FOV sweep), vs
> 1.5–2.3 km for every wide shot. Joint solves (A+B, C+D, all four)
> stay at 1.4–2.3 km: the wide photos' correlated digitization biases
> are of similar residual magnitude to C's (5–8 mrad rms), so an
> unweighted cost sum drags the fix away from the telephoto's answer.
> Margins remain 0.01–0.28 throughout — the gates would still declare
> these inconclusive, correctly. Field guidance that follows: when
> photographing for a skyline fix, a telephoto pan (several narrow-FOV
> frames swept across the terrain) beats one wide frame — angular
> resolution per digitized point is the whole budget.
>
> **E4i — multi-photo skyfix and the FFT heading search**
> (`skyfix.py`, `e4i_multi.py`). Two upgrades productize the E4h
> lessons. (1) skyfix now takes N photos from one position — the
> telephoto-pan procedure — and fuses them into one joint, gated fix
> with per-photo weights 1/σ², σ² = (px_err·FOV/width)² + σ_DEM²: the
> extraction term scales with FOV, the DEM/model term (~1.5 mrad) is
> FOV-independent and floors the weight. The synthetic pan test showed
> why the floor matters: with a clean full-res extractor the DEM term
> dominates, weights equalize, and azimuth COVERAGE decides (wide frame
> 17 m; three 10° tele slivers 241 m and correctly inconclusive at
> margin 0.03; combined 52 m) — the tele-pan advantage belongs to the
> high-extraction-noise regime (coarse input, hand digitization, E4h),
> reachable via `--px-err`. (2) The heading-shift search is now
> FFT-accelerated: the weighted quadratic cost with its offset
> minimized in closed form is three cross-correlations, computed for
> all 3600 lags at once; the exact Huber cost then runs only on the
> top-12 candidate lags, so the optimum is bit-identical to the
> exhaustive search (verified over 6 randomized cases) at ~100× speed.
> A full-circle (heading-unknown) solve now costs the same ~5 s as a
> ±6° one — the E4e/E4f audit regime drops from hours to minutes, and
> a CM5 heading-free fix becomes interactive.
>
> **E4j — coastline stadimetry: a negative result**
> (`e4j_coastline.py`). Tested whether matching the visible waterline's
> depression curve δ(az) ≈ −z/d − d/2Reff as a second channel (the
> "dip short of the horizon" stadimeter; Grelsson's water/land
> boundary) rounds out the range-elongated basins. It does not, at our
> observer heights: with 1 mrad curve noise, range sensitivity
> σ_d ≈ (d²/z)·σ_δ is ~1.8 km for a 3 km shore at z=5 m — weaker than
> what the skyline already provides. Across the E4c sea sites the
> joint cost left errors unchanged, roughly halved the relative basin
> margins (the extra term's noise floor inflates the denominator), and
> inflated the Laplace σ_worst by ~40% (55→79 m at strait1); the
> offshore site has no shore in stadimetric reach at all. Conclusion:
> do NOT build the water/land segmenter for ranging. The stadimetric
> channel only becomes competitive for a high observer with near
> shores (z ≳ 30 m, d ≲ 1 km — sensitivity ∝ z/d²), and Grelsson's
> real benefit from the water/land boundary is land-vs-sea
> classification per azimuth, which our cost already captures through
> the analytic sea-horizon fill of open-water bins.
>
> **E5c — imports from the parallel study branch**
> (celestial-navigation `claude/iphone-celestial-sighting-imu-ctwbnf`,
> an independent 94-commit study that converged on the same terrain
> problem; its findings agree with ours everywhere the two overlap).
> Three lessons imported. (1) *Biases belong in the graph, not the
> sigmas*: `SkyNav` now estimates the compass bias as a shared GTSAM
> variable. The pure heading-factor formulation is correct in batch
> but iSAM2 will not excite the stiff whole-chain-rotation mode it
> needs (verified frozen at +0.06° of a true +1.50°); the working
> observability comes from physics we already had — each accepted
> skyline fix's co-estimated azimuth shift IS a direct compass-bias
> measurement (the E4h mechanism), added as a factor on the bias.
> A/B on the E5 passage (`e5c_bias.py`): recovered bias +1.32° of
> +1.50°, fused error 78.5→69.4 m mean, 6.6→2.8 m final — and the
> instrument now calibrates its own compass underway. (2) *App pitch
> is not trustworthy*: the parallel branch measured a +1.50°±0.17°,
> platform-dependent pitch bias in a real AR theodolite app —
> 2.6× our whole offset window. `skyfix --pitch-sigma` (default 0.29°
> ≡ the historical ±10 mrad) sizes the window as ±2σ; use ~1.5 for
> uncalibrated app pitch; auto-levelling supersedes it. (3) *rms is
> anti-predictive across extractions* (their CH1 sweep: spearman
> +0.32 — a worse boundary scores a LOWER residual): any future
> multi-hypothesis extraction must select on a dimensionless quantity
> (basin margin / separation), never the raw residual. Recorded here
> so it cannot be re-learned the hard way.
>
> **E4k — the trust-gate operating curve, tuned**
> (`e4k_gate_curve.py`, `out/e4k_curve.png`). With the FFT solver the
> full-CH1 audits are cheap enough to treat `--min-margin` as a chosen
> operating point instead of a first guess. Sweeping the threshold over
> the recorded E4e/E4f audits: **with attitude priors** the trade is
> clean and monotone — 0.3 → 37% availability at 26% wrong, 0.5 → 28%
> at 20%, 0.7 → 24% at 14%, 1.0 → 15% at 7%, and **1.5 → 7%
> availability (14 photos) at 0% wrong, none beyond 500 m**.
> **Attitude-free**, no threshold below ~1.0 cleans the accept set
> (wrong share stays 38–56% until availability collapses to 2%) — the
> E4e conclusion, now as a curve. Cross-validation: the parallel
> branch's independent pipeline (different extractor, different
> solver) chose separation ≥ 1.5 and kept **the same 14 photos'
> worth** (7%, 93% within 1 km) — two blind implementations agreeing
> on both the threshold value and the yield. Operating guidance now in
> the tools: fused/underway (factor-graph cross-check present) keep
> `--min-margin` 0.15–0.3 for availability; a STANDALONE single-photo
> fix that must be trusted on its own merits 0.7–1.0 with attitude
> priors; margin ≥ 1.5 is the "certain enough to act on" tier.
>
> **E4l — multi-hypothesis extraction: the headroom is real, no
> selector claims it** (`e4l_multihyp.py`, `e4l_consensus.py`,
> `out/e4l_multihyp.csv`; real extractor, 4 parameterizations, all 203
> CH1 photos, E4f attitude-prior regime). The oracle confirms the
> parallel branch's finding on our pipeline: the right boundary is
> often in the set (43% → 52% within 1 km, median 1250 → 791 m). But
> no selector tried claims it: margin-argmax is a wash (switches on
> 159 photos, helps 36, hurts 34 — unlike rms it is not
> anti-predictive, merely uninformative across hypotheses), and the
> consensus medoid scores 44%. Position AGREEMENT among hypotheses is
> not an integrity signal either — at k=4/4 agreeing within 500 m the
> accepted set is still 63% wrong, because the four hypotheses are one
> detector family and share failure modes: they agree on the same
> wrong edge. Correlated ensembles cannot vote their way to truth.
> Two usable residues: agreement works as a TRIM on the high-trust
> tier (k≥3 AND margin≥1.0: 7% availability at 0% wrong, vs 5% wrong
> for margin alone at 11%), and max-of-N selection inflates margins,
> so gate thresholds must be recalibrated under selection. The open
> road to the oracle's 2× runs through genuinely diverse detector
> families (e.g. this extractor + a DP seam + an SVM boundary), whose
> failures could actually decorrelate — that is the prerequisite, not
> a better scalar.
>
> **E4m — the diverse family, built and tested: failures are
> scene-correlated, and the ensemble road is closed**
> (`e4m_diverse.py`, `out/e4m_diverse.csv`). We built the genuinely
> different detector the E4l conclusion called for — the Ahmad et al.
> IJCNN'21 shallow architecture (16×16 linear patch classifier as one
> correlation + a DP seam), trained by us on the even half of CH1's
> masks (their published CH1-trained weights are non-commercial-only
> and were used solely as an uncommitted research reference), evaluated
> on the held-out odd half. It works differently (21.5 px median vs
> our 0.8, right on half the photos, catastrophic on others) — and the
> decorrelation hypothesis is REFUTED anyway: given our default
> extraction produced a wrong fix, the seam family's fix is also wrong
> **92%** of the time — the same rate as our own same-family variants
> (92–97%). Cross-family position agreement within 500 m still admits
> 67% wrong fixes (same-family: 72%), and agreement+margin matches
> margin alone at every threshold. The reason is the real finding:
> the dominant failure driver is not the detector — it is the SCENE.
> Hazy, low-relief, ambiguous landscapes defeat every extractor at
> once, and when extraction succeeds, an impostor basin is a property
> of the TERRAIN, so every correct extraction lands in the same wrong
> place. Landscape ambiguity is common-mode across any ensemble; no
> vote among extractors can gate it. The margin measures it directly,
> which is why the E4k operating curve — not a better front-end
> ensemble — is the actual ceiling for single-photo trust, and why
> integrity beyond it must come from the fusion layer (E5) or from
> more photos with real baseline/bearing diversity (E4i), not more
> extractions of the same photo.
>
> **E4n — clouds vs the extractor and the gates** (`e4n_clouds.py`,
> `out/e4n_results.json`; E4c sea composites + painted stratus decks,
> cumulus fields, and haze, 27 runs through the full CLI). The
> environmental risk E2 flagged, finally measured — and the system's
> failure mode is the right one. **Overcast (stratus clipping the
> ridges): 9/9 INCONCLUSIVE**, residuals 20–47 mrad — the extractor
> traces the cloud base, the DEM cannot explain it, and the rms gate
> slams shut; zero wrong fixes at any severity. **Haze: benign** —
> fixes stay good (45–155 m) through a 50% contrast wash (the
> local-linear-continuation design goal, confirmed), degrading to
> honest no-fixes near 75%. **Scattered cumulus is the marginal
> case**: usually margin-collapse → inconclusive, but 2 of 9 runs
> produced ~670–700 m fixes that passed the default gates — both with
> margins 0.40–0.64, i.e. both caught by the standalone tier
> (`--min-margin` 0.7) from E4k. Net: 25/27 correct behavior (good
> fix or honest refusal); clouds cost AVAILABILITY, not integrity,
> and the tier system covers the residue. A learned front-end is
> therefore a coverage upgrade (fixing under overcast by masking
> cloud, not tracing it), not a safety requirement.
>
> **E4o — Copernicus GLO-30 as an independent DEM family**
> (`e4o_glo30.py`; tiles auto-fetched from the AWS open-data bucket and
> converted to SRTM3-shaped .hgt under `~/.horizonator/DEMs_GLO30`).
> Until now synthesis and solving shared one DEM, cancelling its
> systematics; deployment gets no such favor. Measured cross-family
> skyline discrepancy (SRTM3 vs GLO-30, six coastal viewpoints, ~18k
> terrain azimuths): **bimodal** — 0.2–0.6 mrad rms where the visible
> terrain is distant, 2.6–3.3 mrad where near coastal terrain is in
> view (the same height error subtends more angle up close, plus
> DSM-vs-DEM canopy differences); overall rms 2.37 mrad, median |d|
> 0.45. The guessed `--sigma-dem` 1.5 mrad thus sits correctly between
> the regimes and is now measurement-backed. Cross-DEM solves (SRTM-
> rendered composites solved against GLO-30): all three sea cases stay
> `ok` at 57–437 m vs 21–346 m same-DEM — mild degradation, no gate
> trips. The instrument can carry either DEM family.
>
> **E4p — SRTM1 at full resolution: the free upgrade**
> (`e4p_srtm1.py`; `CMarcher` now scales its march step to the DEM
> posting, 90 m at 3″ → 40 m at 1″, so `--dem ~/.horizonator/
> DEMs_SRTM1` just works). The pipeline had always downloaded
> 1-arcsecond skadi tiles and DECIMATED them to 3″. Measured cost of
> that decimation: **2.18 mrad rms of skyline detail** (0.3–0.8 mrad
> far terrain, 2.5–3.2 near) — almost identical to E4o's 2.37 mrad
> "cross-family" number, i.e. much of what was booked as SRTM↔GLO-30
> disagreement was self-inflicted resolution loss; the true
> family-vs-family gap is correspondingly smaller. Where it lands:
> the sea-observer solves (E4c cases) improve outright — strait1
> 136→71 m, offshore 267→106 m, strait2 20→17 m — at 2–3× solve time
> (4–5 s→9–13 s, still interactive). On the hard alpine CH1 subset
> (n=20, instrumented regime) it is a wash (median 1752→1591 m,
> margins 0.20→0.30): there the limiter is landscape ambiguity, not
> DEM posting. Deployment guidance: carry SRTM1 for the operating
> area; it costs only disk and a factor ~2.5 in solve time.
>
> **E4r — segmenter-assisted extraction under cloud: negative, twice**
> (`e4r_cloudmask.py`, `ewasr_bridge.py`; the pretrained Apache-2.0
> eWaSR maritime segmenter vs the E4n cloud matrix, paired A/B). Both
> formulations failed informatively. Masking cloud-base columns is a
> no-op: the surviving columns still carry the deck-bottom's wrong
> flat geometry, and the information a mask could protect was never
> extracted (the top-down extractor stops at the cloud). Replacing
> extraction with the segmentation's own topmost-land boundary
> recovers solves under stratus but WRONG ones (1.9–3.8 km), while
> costing 5–7 mrad of boundary precision on clear scenes (127→253 m,
> 13→179 m; tally 8 good/2 wrong plain vs 5 good/3 wrong seg) — the
> E4m lesson from a new direction: segmentation-grade boundaries are
> not matching-grade boundaries. E4n's conclusion stands: overcast
> costs availability, honestly. The segmenter's justified roles are
> diagnostic — naming WHY a scene is inconclusive ("overcast
> detected", from its sky-over-boundary signature), sea-span selection
> for auto-levelling, false-horizon second opinions — and those are
> what `ewasr_bridge` ships for. Revisit recovery-under-cloud only
> with a fine-resolution model validated on real maritime imagery.
>
> **Literature dig II — four veins, two assayed on the spot.**
> (1) *The founding lineage*: Cozman & Krotkov's Viper (CMU, 1995–97,
> already in §3.1) turns out to carry the field's first quantified
> field accuracies — 100–400 m (2.5–6.5× DEM resolution) across
> Pittsburgh, the Atacama, the Rockies and a simulated Apollo 17
> site — numbers our CH1 accepted-set results finally improve on with
> the same class of instrument, thirty years later. (2) *Country-scale retrieval exists off the shelf*:
> CrossLocate (Tomešek & Čadík, WACV 2022; code on GitHub) learns
> cross-modal embeddings between photos and 10.7M rendered
> silhouette/depth/semantic views covering the entire Alps at 1M
> locations — the exact architecture for the "somewhere on the
> Turkish coast" search scale our dense solver cannot reach; our
> fastmarch renderer can generate the equivalent coastal database.
> (3) *Refraction dip empirics* (van der Werf 2016; the SDSU
> refraction corpus): dip variability is driven by the air–sea
> temperature difference, and corrections from ΔT + wind cut dip
> error by ~1/3 and outliers by ~2/3 — directly applicable to
> auto-level: an optional air–sea ΔT input should correct the assumed
> dip, and large |ΔT| should widen the ±2 mrad window. The cheapest
> unclaimed accuracy in the tight-β mode. (4) *Haze ranging*
> (Koschmieder): assayed immediately, both ways. On our synthetic
> composites (self-consistent haze) per-azimuth contrast recovers
> range with corr 0.75, median |err| 0.75 km over 4–14 km — but on 15
> real CH1 photos only 2 carry the signal (the two genuinely hazy
> ones; clear air has no gradient and albedo variation swamps the
> cue). Verdict: a conditional auxiliary for hazy maritime days,
> never a dependable channel — recorded so the synthetic best case
> doesn't seduce anyone later.
>
> **E4t — the whole-coast fix: learned retrieval has no job here**
> (`e4t_coastwide.py`). CrossLocate exists because mountain photos at
> country scale need learned retrieval to prune candidates before
> verification. The coastal regime does not: candidates are only sea
> cells with land in view — a natural prune no mountain photo gets —
> and the FFT cost makes each candidate ~6 ms. One composite photo
> with a compass heading and NO position prior, searched against the
> entire southeast Aegean (36–38°N, 26–28.5°E, 222 × 222 km, ~50× the
> E3 box, 34,466 candidates on a 1 km grid): fix within **10 m** of
> truth, coast-scale basin margin **1.14** (20 km NMS), **197 s** on
> four x86 cores — CM5-projected ~10–15 min, and embarrassingly
> parallel. The weakest geometry confirms it: the offshore case
> (land in one narrow sector) still resolves coast-wide to 247 m with
> margin **6.24** — open water in most directions is itself a
> discriminative signature, since few places on the coast have land
> in exactly that sector at those elevations. The "somewhere on this
> coast" scenario is thus solved by brute force within the existing
> solver; CrossLocate-class retrieval
> remains the right architecture only for inland/mountain operation,
> where the sea mask does not exist.
>
> **E5e/E5f + rigid pan — the redundancy suite** (three veins worked
> in one pass). (1) *Rigid-pan fusion* (`skyfix --rigid-pan`): a pan's
> relative headings are gyro-accurate, so all frames share ONE compass
> offset instead of independent per-photo shifts — on the E4i
> synthetic pan, 70→**15 m**. (2) *The echo sounder* (`e5e_depth.py`):
> the skadi tiles' discarded negatives ARE a bathymetric grid;
> `depth_factor` fuses NMEA-DPT-class readings, fully decorrelated
> from the camera. Clear weather it adds little (the skyline
> dominates); in the no-camera fog regime it bounds final drift
> **611→170 m**. (3) *The night watch* (`e5f_lights.py`): identified
> charted lights (flash characteristics make them self-identifying)
> enter as bearing factors through the shared compass-bias variable —
> the passage run at night, skyline blind: DR 590 m final →
> **lights 62 m**, full night suite (lights+depth) **50 m**, and the
> lights recover the compass bias to +1.54° of the true +1.50° in the
> dark. The instrument is now a 24-hour, all-weather *suite*: skyline
> by day, lights by night, depth always, dead reckoning between —
> every channel through one graph with one self-calibrating compass.
> Real light positions/characteristics come from any List of Lights
> or OSM seamark data at deployment; the synthetic stand-ins here
> only prove the estimator.
>
> **E5d — the C++ route** (`e5d_export.py` here;
> `examples/SkylineNavExample.cpp` + `examples/Data/
> skyline_nav_stream.csv` in the gyillikci/gtsam fork, same branch).
> The fusion layer ported to C++ as an alternative route to the same
> results: `SkylineFixFactor` (unary Pose2 fix with anisotropic
> covariance) and `HeadingBiasFactor` (heading through the shared
> bias variable), consuming the exported sensor stream with no
> Python, DEM, or solver in the loop. Parity against the python-gtsam
> batch reference is exact to ~10 significant digits: final pose
> (15786.30955, −4020.638423, θ −0.354814401), recovered compass bias
> +1.315748 deg, mean error 31.394351 m, final 2.800441 m — plus the
> C++-side marginals (σ 14.9/13.0 m). Built against the fork (GTSAM
> 4.3, boost-free config). The embedded integration path is now:
> front-end produces the stream rows, this graph consumes them.

> **E4u — the CH1 full set in 20-photo batches, plus the DEM/near-field
> A/B** (`experiments/e4u_ch1_batches.py`, outputs `out/e4u_*`). The
> instrumented regime (E4f priors, same seed, FFT solver) over all 203
> CH1 photos, reported per 20-photo batch. Baseline (SRTM3, hard 1 km
> clip): median 1031 m, 70/203 within 500 m, confusion TA 63 / FA 50 /
> CAUGHT 83 / OC 7; batch medians swing 354 m (best terrain) to
> 1960 m (repetitive foothills), and the margin tiers hold their
> ordering at full scale but soften from the 14-photo E4k calibration:
> accepted fixes are 56% correct at margin ≥0.15, 81% at ≥0.7, 88% at
> ≥1.5, 100% at ≥3.0 — the residual standalone-tier false accepts are
> "wrong ridge, same shape" basins on high-relief alpine terrain
> (30–66 mrad relief), all within ~2 km. Two A/Bs against it: (1)
> full-set SRTM1 (hard clip unchanged) — median 791 m, 75/203 hits,
> FA 50→37 with CAUGHT 83→91 at similar availability, tiers 63/86/86%:
> full resolution sharpens the cost landscape more than it moves the
> optimum, so its real value inland is catching wrong basins (the E4p
> "wash" verdict was about accuracy only). (2) The sea configuration
> (SRTM1 + soft near-field with the C0_NOINFO coverage charge) on the
> first 60 photos: hits 6→3, median 1677→2186 m, availability
> collapsed to 54/60 CAUGHT — suppressing sub-kilometer terrain
> discards an alpine viewpoint's own foreground ridges. The soft ramp
> is confirmed regime-specific: right at sea (where near terrain
> cannot exist and fake near-field basins are the threat), wrong on
> land. Config selection between the two regimes is a one-bit input
> (are we afloat?), not a tuning problem.

> **E5g — the night channel without an oracle**
> (`experiments/lights.py`, `lightscan.py`, `e5g_lightscan.py`). E5f
> assumed something told the camera WHICH charted light it saw; E5g
> removes that assumption. `lights.py` is the light database layer: an
> Overpass fetcher for OSM seamark data (position + flash character +
> range/height; blocked from this container's egress — run user-side
> and commit the JSON), a chart-notation parser, and a matcher that
> identifies a classified character within a radius gate.
> `lightscan.py` is the front end: MAD-thresholded point tracking
> across night frames (a percentile threshold sits in the noise tail
> and spawns junk tracks — first bug found), and a chart-like
> character classifier — binary lit/dark wave, sub-half-second gap
> closing (a wave occluding the light for a frame must not split a
> flash — second bug), unbiased-autocorrelation cycle length (the
> full period aligns every cycle; a group's intra-flash spacing only
> partially aligns), flashes-per-cycle, duty-cycle pattern class, and
> uniform-train harmonic reduction (Fl.2.5s sampled at 5 Hz first
> reads Fl(2)5s; equal edge spacing reduces it to the fundamental —
> the reduction lives in the classifier, which can see uniformity,
> NOT in the matcher, where it would falsely equate Fl(3)15s with
> Fl.5s — third bug, caught by the decoy test). End-to-end on
> synthetic 40 s / 5 Hz night video with sensor noise and 6% wave
> dropouts: Fl(3)W.15s, Fl.W.5s, Iso.W.4s all classified and uniquely
> identified; an uncharted decoy (fishing light, Fl.2.5s) correctly
> classified and REJECTED (no chart entry — a track that does not
> match exactly one charted light is never used). The E5f passage
> re-run with identification in the loop: 54 of 59 sightings
> identified and used, 5 conservatively rejected, mean 154 m / final
> 148 m vs the oracle's 138 / 62 m, compass bias recovered +1.64 vs
> +1.54 deg — full autonomy costs ~10% in accuracy and nothing in
> integrity. Remaining for a live run: the regional OSM extract and
> real dark-frame video (the tracker's MAD threshold and the
> classifier were tuned on the synthetic sensor model only).

> **E5h — wind turbines as a charted daytime landmark channel**
> (`experiments/turbines.py`, `blade_flicker_hz` in `lightscan.py`,
> `e5h_turbines.py`). The user's observation: a scene WITH wind
> turbines is information in itself. Turbines are surveyed (OSM
> power=generator + generator:source=wind; fetcher mirrors the lights
> one, run user-side), ~100 m tall, ridge-crowning, and dense on the
> Aegean coast — lighthouses of the daytime, identified not by flash
> character but by (a) blade-pass glint periodicity in video
> (0.5–1 Hz; detector requires the autocorrelation to decorrelate
> before re-peaking, else slow illumination drift false-fires — bug
> found) and (b) the CONSTELLATION of bearings to a farm. Three
> results on two synthetic farms (8-turbine ridge line + 3-turbine
> cape cluster): (1) presence filter — candidates that could not see
> turbines are impossible when the scene shows them (10% of a 24 km
> box pruned in this two-farm geometry; against a mostly turbine-free
> coastline the prune is the point); (2) constellation fix — a 1-D
> Hough over the unknown compass offset solves correspondence AND
> compass bias together: 11 anonymous bearings -> best cell at 0 m
> error, bias recovered +1.61 vs true +1.50 deg; (3) the passage:
> turbine bearings flow through the same shared-bias factor as the
> night lights (nothing new graph-side). Day without skyline: DR
> 382/590 m -> +turbines 262/386 m. With skyline: mean unchanged
> (69 vs 71 m) but bias estimate improves (+1.59 vs +1.32, true
> +1.50) — the channel's value is availability and compass
> observability, not beating a healthy skyline fix. The graph
> DIVERGED on first wiring (mean 2700 km): an evenly spaced turbine
> line admits a shifted, picket-fence correspondence that aligns n−1
> bearings, and one wrong Hough alignment poisons everything —
> defence is acceptance gates (>=3 inliers, offset within ±4 deg,
> post-alignment rms <= half tolerance) before any bearing enters;
> with them, 159 bearings fused cleanly. Next integration: turbine
> presence as a candidate prune in the E4t whole-coast search.

> **E5i — pylon rows and comm masts join the landmark web**
> (`experiments/landmarks.py`, `e5i_landmarks.py`; class-aware
> matching added to `turbines.py`). Two more charted point classes:
> transmission towers (OSM power=tower — picket-fence rows over
> ridges and straits, handled by the E5h Hough + gates) and
> communication masts (man_made=mast/communications_tower — isolated
> summit points). The constellation machinery is now class-aware:
> detector classes are coarse ('turbine' by blade flicker vs 'static'
> for everything else) and assignment forbids pairs where exactly one
> side is a turbine. Results: (1) mixed 20-bearing scene (8 turbines
> + 10 pylons + 2 masts, unknown correspondence and bias) -> 0 m cell
> error, bias +1.44 vs +1.50 true; in this rich geometry classless
> matching was already unambiguous — the class gate is a no-harm
> constraint here and matters for sparse scenes. (2) The night
> crossover: masts carry red aviation obstruction lights, so
> `as_light_entries` exports them into the E5g light DB (generic
> Fl.R character; two masts sharing it are disambiguated by the
> DR-predicted bearing, gate 5 deg). Night passage: sea lights only
> 147 m mean -> +mast lights 104 m mean, identifications 50 -> 65.
> (3) Day passage, whole web (21 points, three classes) vs turbines
> only: mean 317 -> 253 m, final 425 -> 96 m, 278 bearings fused.
> Also fixed en route: the synthetic flash generator gave a 1.5 s
> aviation light a 53% duty cycle (reads as isophase) — flash width
> now scales with period. The daytime landmark web and the night
> light set now share one database format, one Hough, one factor,
> and one compass-bias variable.

> **E5j — literature dig III: trained models for the landmark
> channels** (`experiments/e5j_openvocab.py`, out/e5j*). Hunted
> specifically for downloadable weights; egress decides what is
> usable here. LANDED AND ASSAYED: YOLOE (yoloe-11s-seg, Ultralytics,
> AGPL-3.0 — flag for commercial use) — open-vocabulary detection
> with text prompts; chosen over YOLO-World v2 purely by egress
> (YOLO-World's text encoder is OpenAI CLIP on a blocked Azure host;
> YOLOE's MobileCLIP TorchScript lives on GitHub release assets,
> which are reachable — detector weights for both are too).
> Zero-shot on 60 CH1 alpine photos, prompts {electricity pylon,
> transmission tower, communication antenna mast, wind turbine,
> lighthouse}: 15/60 photos fire, confidences 0.05-0.73; visual
> audit shows the strong fires are real near-field vertical
> infrastructure (with class confusion: a roadside pole reads as
> transmission tower 0.73), the weak ones are posts/fences, and an
> ACTUAL summit antenna on the skyline went undetected — at 1024 px
> a landmark 5-15 km out subtends a few pixels, below single-frame
> detection. CONCLUSION: single-frame open-vocab detection is a
> proposal generator for near/mid range only; distance landmarks are
> identified temporally (blade flicker, flash character — resolution-
> independent while the glint subtends a pixel) and geometrically
> (constellation + gates), which is exactly how E5g-i are built.
> Telephoto/pan frames (already in the instrument for skyline work)
> are the other lever. USER-SIDE LIST (blocked hosts): TTPLA aerial
> tower/line dataset + YOLACT weights (Google Drive, license
> unspecified — fine-tune material); PLD-UAV (PLDU/PLDM) wire
> segmentation datasets; GroundingDINO swint_ogc.pth IS on GitHub
> releases (694 MB, reachable) as a heavier open-vocab backup if
> YOLOE proposals prove too weak. Literature without public weights
> but validating our design: BLDCNet (buoy-light detection + flash-
> pattern classification on frame sequences, 96-98%) and a
> multilabel navigation-mark light video classifier (~99%, 9 light
> types) — both do temporally what our numpy classify_trace does;
> no weights, so ours stays. OpenCellID (CC BY-SA) is crowd-
> triangulated cell positions, hundreds of meters off — NOT
> surveyed mast coordinates; OSM man_made=mast remains the bearing-
> landmark source, OpenCellID only a coverage cross-check.

> **E4q — the auto-leveller against measured horizons, and the sign
> error it exposed** (`experiments/e4q_imu_horizon.py`, out/e4q_*).
> MaSTr1325 arrived (1325 real maritime images, each with an IMU mask
> whose boundary is the inertially measured horizon, plus a
> sea/sky/obstacle segmentation), so the +-2 mrad post-levelling
> window could finally be checked against something other than the
> solver's own geometry. First, what the dataset is: shot from a USV
> in and around a marina, so in the MEDIAN image the horizon is fully
> occluded by land — 0% of columns have sky directly above water, and
> only 104 of 1030 usable images show an open horizon over half their
> width (295 `old_*` images carry no IMU and are excluded). That makes
> it two tests. **A. Veto correctness**: the levelling stage should
> accept open-horizon scenes and refuse occluded ones. It did the
> opposite — 2% availability on open horizons, 19% false accepts on
> occluded ones. **The cause was a sign error in the physics.** The
> photometric water check assumed open water is DARKER than the sky by
> >= 0.30; measured on this data, a true sea horizon has a median
> sky-minus-water step of 0.031 (p90 0.185) while land boundaries step
> 0.375 (p10 0.244). Near the horizon the sea reflects the sky at
> grazing incidence (Fresnel reflectivity -> 1), so a genuine horizon
> is nearly CONTINUOUS in brightness and a big step means land. The
> check now tests |step| <= max_step (default 0.20, `--max-step` on
> skyfix): availability 2% -> 17%, false accepts 19% -> 7%. The old
> rule had been rejecting 98% of real horizons and admitting the false
> ones — invisible for eleven experiments because every prior test was
> synthetic. **B. Accuracy** on open-horizon accepts: the fitted line
> sits within a median 2.9 px of the IMU line at image center (p90
> 6.9 px), pitch error median +0.41 deg, roll +0.75 deg, edge
> elevation error median 21.8 mrad. Caveat that dominates the mrad
> numbers: at 512x384 with the assumed 65 deg FOV one pixel IS
> 2.5 mrad, so +-2 mrad is sub-pixel on this imagery — the pixel-space
> result is the honest measurement, and MaSTr1325's downscaled frames
> cannot validate a 2 mrad claim. The real limiter found: on
> open-horizon scenes the terrain seam detector tracks the true
> horizon (>70% of columns within 3 px) in only 10 of 28 audited
> images, and when it does, levelling accepts 8/10. So the levelling
> front end needs a horizon-specific low-contrast line detector, not
> the mountain seam finder — that is the next concrete piece of work.
> Two knock-ons: the E4c synthetic composites paint the sea 0.17-0.30
> darker than sky (unphysical at the horizon, which is why they never
> exposed the sign error) and now need `--max-step 0.35`; and E4g's
> four synthetic cases currently yield no sea-horizon fit at all,
> refused by the GEOMETRIC gates — verified by disabling the
> photometric check entirely and seeing the same refusals, so this is
> not the E4q change — the harness now records the refusal instead of
> crashing, and the cause is an open item.

> **E4q-2 — the horizon-specific detector** (`horizon_candidates` and
> `sea_horizon_attitude_radon` in `extract.py`, `skyfix
> --level-detector`). E4q identified the front end as the limiter, so
> this replaces it. The insight is that a sea horizon is defined by
> COHERENCE, not contrast: the step is tiny (median 0.03) but the line
> is perfectly straight across the whole frame. So the vertical
> brightness derivative is normalised per column by its own robust
> scale — making a faint step in a dim scene count as much as a strong
> one in a bright scene — and summed along every near-horizontal line,
> a Radon transform restricted to the slopes camera roll can produce.
> Candidates are refined column-wise, fitted to the same physical
> model, and passed through the same two defences (nothing below the
> line; |brightness step| <= max_step). One bug worth recording: the
> IRLS fit must be initialised from the CANDIDATE line, not from
> level — starting level leaves every point tens of pixels outside the
> tolerance so the fit never starts, which showed up as 0% acceptance.
> Scored against the HAND-ANNOTATED water/sky boundary over all 1030
> usable images:
>
> |                           | seam front end | radon front end |
> |---------------------------|----------------|-----------------|
> | availability, open scenes | 17% (18/104)   | **32% (33/104)** |
> | row offset, median        | 2.9 px         | **2.1 px**      |
> | roll error, median        | +0.33 deg      | **+0.04 deg**   |
> | edge elevation err, median| 18.6 mrad      | **7.6 mrad**    |
> | within 10 mrad            | 22%            | **70%**         |
> | within 2 mrad             | 0%             | **12%**         |
>
> Availability doubles and error drops 2.4x; the roll bias essentially
> vanishes. The cost: accepts on segmentation-"occluded" scenes rise
> 7% -> 27% — but those accepts sit a median 17 mrad from the horizon
> (against seam's 51 mrad), i.e. the detector is levelling off distant
> low shoreline lying close to the true horizon rather than inventing
> a line. Whether to spend that availability is a policy choice; the
> harmful-error tail is what to watch, not the label. Radon is now the
> skyfix default (`--level-detector seam` restores the old path).
> REFERENCE CORRECTION made here: MaSTr1325's IMU mask is an
> approximate onboard prior, not calibration truth — over the 104
> open-horizon images it agrees with the drawn boundary to a median
> 0.2 px in row but differs by a median 0.42 deg in ROLL (p90 1.15),
> the same order as the estimator's own error, so accuracy is scored
> against the hand annotation and the IMU mask is kept only as a
> coarse cross-check. (An early single-image spot check suggested
> 1.7 deg; the full-set measurement corrects that.) Still open: 12%
> within 2 mrad means the window remains unvalidated at this image
> scale, where one pixel IS 2.5 mrad — it needs native-resolution
> imagery, not more of this dataset.

> **E4v — the instrument on its own field data** (`experiments/
> theodolite_curate.py`, `e4v_theodolite.py`; 83 curated pairs from
> five sites, July–August 2026, iPhone 17 Pro + Theodolite). The first
> photographs taken FOR this pipeline: clean frame, app-recorded
> attitude with the accuracies iOS reported (compass median ±10 deg,
> worst ±41; GPS median 4.7 m), GPS kept for scoring only.
>
> Single narrow-FOV sightings do not localise. Six of eight prime
> sightings solve, median error 914 m, best 396 m; five pass the
> gates but only two of those land within 500 m, and the worst false
> accept carries margin 3.67 at 1695 m error. The errors are large
> along AND across the line of sight, because the compass offset is
> co-estimated and frees the across-sight axis too — a 10–20 deg
> slice of distant terrain is close to a bearing measurement. Two
> explanations were tested and refuted: relief is adequate (4.5–8.8
> mrad, well above the 1.5 gate), and the tight ±2 mrad window is not
> at fault — substituting the phone's own inclinometer with the wide
> window makes every case worse (460→1099, 396→1066, 675→2646 m) and
> all four then fail the gates, so the auto-leveller beats the
> device's inclinometer here.
>
> Fusing a pan fixes it. The Marmara site holds a real pan: 20 frames
> from one spot (<25 m) inside ten minutes, spanning 75 deg. Six
> telephoto frames of it fuse to **168 m** — against a 914 m median
> for the same kind of frame solved singly — with the reported sigma
> (389/367 m) finally honest rather than the 4.4x optimism the single
> fixes showed. RIGID-pan, one shared compass offset, gives 451 m and
> a joint rms of 3.00 mrad against 0.37–1.75 for the frames singly:
> its premise is that the RELATIVE headings are gyro-accurate and
> only their common offset unknown, which holds for a sweep taken in
> seconds but not for handheld sightings spread over ten minutes,
> each carrying an independent compass reading. So the capture
> procedure decides the fusion mode, and for this procedure the
> per-frame free offset is correct.
>
> Two calibrations for the record: the Laplace covariance is a median
> 4.4x optimistic on single field fixes (its shape is right — on
> KWHC9160 the reported north:east ratio matched the actual error
> ratio to 1%), and the auto-leveller differs from the Theodolite
> inclinometer by a median 0.60 deg in pitch and roll, consistent
> with the 7.6 mrad it measured against MaSTr1325 annotations. Open:
> the basin margin measures how distinct the winning basin is, not
> whether the geometry can support a position, which is why it waves
> through single-frame bearings; a conditioning term from the
> covariance ellipse is the fix.

> **E4y — depth layers: the information the matcher throws away.**
> Raised from the field photographs: a coastal scene is layered — an
> island in front, a hazy coast behind — and the matcher keeps only
> the topmost sky boundary. Marching each ray past the near-field mask
> and grouping the running maxima by range gaps shows what is being
> discarded. At OREJ1026's azimuth the DEM has three visible layers:
> 1.0 km at 0.5 mrad, an island at 6.2 km at 7.5 mrad, and the far
> coast at 41.1 km at 17.9 mrad. Only the last one reaches the cost.
>
> The near layers are where the position information lives. A 500 m
> lateral move swings the 41 km coast by 12 mrad but the 6.2 km island
> by 81 — seven times more — and a move ALONG the sight line changes
> the island's angular width while leaving the far ridge essentially
> unchanged. That is precisely the direction every single-frame field
> fix failed in (along-sight errors of +307 m and similar), and it
> explains why a single narrow frame behaved like a bearing: its only
> observable was the layer least sensitive to position.
>
> Building it needs three pieces. The DEM side is a marcher that
> returns every visible crest per azimuth instead of the maximum
> (prototyped, ~30 lines). The image side is depth-layer extraction,
> for which atmospheric perspective is the natural cue — farther
> layers are lighter, bluer and lower in contrast, which is how the
> haze assay (E4s) measured range in the first place. The cost then
> weights layers by what each constrains: near layers pin position,
> far layers pin heading. Note also what does NOT work: reading range
> from the depression angle of an island's waterline, the classical
> vertical sextant angle, is useless at these eye heights — at 2.5 m
> the horizon is 5.7 km away, so anything further has its waterline
> hidden. Angular width, not depression, is the near-layer range
> observable.

> **Horizonator vs PeakFinder — what each is for.** PeakFinder is the
> better panorama: a polished commercial renderer with named peaks in
> ten locales, sun and moon tracks, and a telescope mode, embeddable by
> URL, iframe or canvas (gyillikci/PeakFinder-API). Its own terms are
> 'all rights reserved', its terrain lives on its servers, and its
> Javascript surface — loadViewpoint, azimut, altitude, fieldofview,
> elevationOffset, plus events for viewpoint/sun/moon/poi — displays a
> view and returns strings about it. Nothing in that surface returns
> the rendered geometry.
>
> That is the whole difference for this project, and it is not about
> quality. Fixing a position means evaluating a hypothesis: elevation
> and RANGE per azimuth at a candidate position, 625 of them per fix on
> a 6 km box, then again at every layer once the near ones matter.
> Measured here, our marcher produces a full 3600-azimuth skyline in
> 6.5 ms on 3-arcsecond data and 21.2 ms on 1-arcsecond — 4.1 s and
> 13.2 s per fix, offline, with the range array that E4y needs for the
> depth layers. A display API cannot answer that question at any speed:
> it renders pixels of a viewpoint, over the network, for a human to
> look at.
>
> So they are complementary rather than competing. PeakFinder is the
> reference a navigator can check a viewpoint against by eye, and the
> URL format takes our sighting parameters directly
> (?lat=&lng=&ele=&azi=&fov=), which is how the E5a panels are made
> comparable. Horizonator (LGPL, and its own README points at
> PeakFinder as the more polished tool) is the building block the
> solver runs inside. Note also the licence asymmetry: nothing of
> PeakFinder's can be vendored, while the horizonator can be modified
> and shipped.

> **Lit dig IV — the whole horizon-navigation landscape** (search
> pass over every sense of the term; repos probed from here where
> reachable). The field splits into five families, and knowing which
> family a paper belongs to saves misreading its claims:
>
> 1. TERRESTRIAL SKYLINE GEOLOCALIZATION — our core family, already
> surveyed (sec. 3, CrossLocate, E4t). New finds: two US patents,
> 11678140 and 12101693 'Localization by using skyline data' (plus the
> older 9165217/9292766 on DEM photo geolocation) — an IP-awareness
> note for any commercial turn, not an obstacle to research use.
> 2. HORIZON ATTITUDE (UAV/USV) — Ettinger-to-Grelsson line, still
> active: U-Net horizon detectors with sub-2-degree rms attitude
> (2022-24 surveys), thermal-image variants, fisheye Hough methods.
> Our radon detector plus inclinometer cross-check sits squarely in
> this family's accuracy envelope, measured against hand annotation.
> 3. PLANETARY ROVER SKYLINE LOCALIZATION — the most direct cousins,
> GPS-denied by construction: the 2017 Mars-rover horizon-to-DEM
> matcher (already cited), Mercator (2016, 6 m from panoramic
> horizon), and NEW: ALPER (Acta Astronautica 2025) with a Skyline
> Matching mode among three complementary absolute localizers, and
> AAS 25-358 (2025) fusing celestial with horizon matching for lunar
> surface ops — the same skyline+celestial pairing this project's
> parallel branch explored. No public code found for either.
> 4. SPACECRAFT LIMB OPNAV ('horizon-based optical navigation',
> Christian et al.) — different physics (planet limb as an ellipse,
> position from a conic fit) but a mature covariance literature: the
> Christian-Robinson non-iterative solver, the 2023 EW-TLS short-arc
> variant, and a 2021 tutorial. Their treatment of measurement
> covariance under partial arcs is the polished version of what our
> jackknife is approximating.
> 5. MARITIME COASTAL VISUAL GEO-LOCALIZATION — deep retrieval against
> rendered coastlines (the Lyon/Brest 2025 line, already cited), plus
> 2025 JFR work on featureless GNSS-denied maritime aerial nav.
>
> Front-end news: YUNet (arXiv 2502.12449, 2025) — YOLOv11+UNet
> skyline detector, IoU 0.9858, 1.36 px mean error — has its code at
> github.com/kuazhangxiaoai/SkylineDet-YOLOv11Seg (verified reachable
> from this container; AGPL-3.0, ultralytics fork). It trains on
> GeoPose3K and VALIDATES ON CH1 — the exact datasets on hand here —
> but ships no pretrained weights, so a head-to-head against our
> seam/radon extractors requires a training run (GeoPose3K is on the
> user-side download list already). SkylineDet joins eWaSR as the only
> reachable, license-clean model families in the space.

> **SkylineDet/YUNet: the training recipe, and the benchmark that
> justifies it** (clone inspected; extractor benchmark run here).
> The repo is an ultralytics fork; training is a two-stage recipe in
> `ultralytics/cfg/skyseg.yaml`: (1) pretrain the yolo11-skyseg model
> on Skyfinder sky masks, (2) fine-tune on GeoPose3K, validating on
> CH1 (`GeoPose3K.yaml`: path/train/images + parallel labels/, one
> class 'sky'; entry `models/yolo/skyseg/train.py`, defaults 50
> epochs, batch 16, imgsz 640, single_cls, mask_ratio 1). No
> pretrained weights ship, and both stages need a GPU: 50 epochs over
> ~3k GeoPose3K images is hours on a consumer GPU and out of reach of
> this container's 4 CPU cores. The container CAN prepare everything
> (configs with our paths, CH1 masks already in the right form) so
> the user-side run is one command once GeoPose3K lands.
>
> Why bother is now measured, not assumed: on CH1 ground truth our
> seam detector scores a mean 22.4 px per-image boundary error
> (median 3.6 — the mean is carried by a failing tail) and the E4m
> learned template 82.8 px on its held-out half, against YUNet's
> published 1.36 px mean on the same dataset. If that number
> reproduces, the front end stops being the accuracy limiter at
> CH1-style scales; the caveat is that CH1 is also their validation
> set, so the honest test after training is CH1 PLUS our Theodolite
> frames, which no one trained or validated on.

> **E5k — MobileSAM closes the extraction problem; the island frame
> then measures what remains** (`e5k_mobilesam.py`, `e5b --sam
> --edges`). MobileSAM (Apache-2.0, 40 MB in-repo weights, CPU ~2 s a
> frame) replaces the fragile nearness threshold for the near layer,
> with one design correction along the way: prompting from 'dark
> pixels above the horizon' segmented the hazy FAR COAST, because the
> island body sits below the drawn horizon line — the working scheme
> seeds SAM from the sparse E5b detections (fragile as a contour,
> correct as points). Result on the island frame: column coverage 12%
> -> 64% (the island's true span) and a 0.0 px contour spread across
> prompt seeds, on all four scene types tried — the threshold
> sensitivity that swung the fix 559 -> 3250 m is gone.
>
> With extraction solved, the remaining failures are attributable and
> were measured one by one on OREJ1026. (1) Full-contour height
> matching, offset locked: 3162 m at rms 3.39 mrad — the ~7 mrad DEM
> height bias over the island (E4x) dominates once the whole flank is
> matched; the earlier 559 m via the sparse top-only contour rode
> partly on luck. (2) Width-and-bearing matching (height-free): the
> DEM's 2-15 km band at truth spans 10.1 deg against 7.36 observed —
> part model (neighbouring features merged into the run), part FRAME
> TRUNCATION, the DEM run reaching 184 deg where the photo ends at
> 181. (3) With the window clipped to the frame for both sides, the
> cost collapses to 0.044 but the margin to 0.13 at 2761 m: the
> island fills 7.4 of the frame's 10.3 deg, both edges sit near the
> frame limits, and the truncated width barely changes with position
> — the observable SATURATES. So the near-layer channel is
> information-rich in principle (81 vs 12 mrad per 500 m, E4y) but
> needs (a) the feature FULLY framed with sky/sea margins — a 20-41
> deg frame, not a 10 deg tele filled edge to edge — and (b) a model
> that carries small-island heights honestly. Field-procedure
> consequence: photograph islands with margins around them; the only
> island frame in this set does not satisfy that, through no fault of
> the sighting procedure at the time.

> **E5l — MobileSAM finds the sea line; the hybrid keeps the
> refinement** (`e5l_samsea.py`, `skyfix --level-detector sam`,
> `extract.sea_horizon_attitude_radon(extra_candidates=...)`).
> Prompting SAM low in the frame and taking the water mask's upper
> boundary, scored against MaSTr1325's hand annotation over the
> open-horizon subset, beats both existing detectors on BOTH axes:
> 74% availability at 1.0 px median row offset (p90 3.6) against
> radon's 32% at 2.1 px and seam's 17% at 2.9 px; on jointly-accepted
> frames SAM is 0.9 px to radon's 2.0. The union covers 79% (SAM-only
> 49 images, radon-only 5). One sampling lesson recorded: the first
> 25 images gave 40% at 6.1 px — a hard marina sequence — so subset
> numbers on this dataset mislead in either direction.
>
> The field frames then showed the catch: SAM alone mis-tilts. On
> KWHC9160 its line agrees in row but differs 1.6 deg in ROLL (mask
> blockiness that 512 px scoring cannot see), and the fix degrades
> 436 -> 1367 m. So the integration is a proposal hybrid: the SAM
> line enters the radon detector as an extra candidate, and the
> column-wise sub-pixel refinement and gates decide as always. Where
> radon already worked the hybrid converges to the identical answer
> (436 m, same attitude); where the only line in view is a shoreline
> (APST5638) the gates refuse the proposal — correctly. Availability
> gains land in MaSTr-like scenes; nothing regresses. The pattern is
> now uniform across three tools (eWaSR, the learned template, SAM):
> segmentation proposes, geometry refines and gates decide.

> **E5m — what actually breaks the fix: the perfect-compass bound**
> (`e5m_blame.py`). The standing suspicion, repeated throughout E4v-x,
> was that the phone compass dominates single-frame error. Measured,
> it does not. Per prime sighting the reference heading was recovered
> at the GPS truth (offset floating +-15 deg in a 400 m box) and the
> full 6 km solve re-run with heading clamped to +-0.5 deg:
>
> |          | compass | perfect heading |         |
> |----------|--------:|----------------:|---------|
> | MYQR7719 |   396 m |           387 m | -2%     |
> | KWHC9160 |   460 m |           379 m | -18%    |
> | PQBC6867 |   675 m |           407 m | -40%    |
> | EWAC7374 |  1154 m |          1065 m | -8%     |
> | INKX2521 |  1695 m |          1189 m | -30%    |
> | SRYK4301 |  3024 m |          3091 m | +2%     |
> | median   |   914 m |           736 m | **-19%**|
>
> A perfect compass buys 19% at the median, and nothing at all on the
> worst frame. The device heading was a median 4.2 deg off — inside
> its own +-10 claim. Yet the six-frame pan, with the SAME compass
> errors, reaches 168 m. So what the pan buys is not compass
> correction: it is ANGULAR COVERAGE (75 deg of terrain against a
> 10-20 deg slice) and the partial cancellation of independent DEM
> errors across frames. Corrected ranking of what breaks a
> single-frame fix on this coast: (1) DEM-versus-world model error —
> the floor a perfect compass cannot touch (736 m median; SRYK4301
> immune to heading entirely); (2) angular coverage; (3) heading;
> (4) roll; (5) extraction, now largely solved. Investment
> consequence: a better compass is nearly worthless here — wide
> capture and better terrain handling are where the meters are.

> **E5n — reducing the DEM's share** (GLO-30 full-res store fetched
> from the reachable Copernicus S3 bucket; A/B on the field frames).
> The fit at the GPS truth improves with GLO-30 on some frames
> (PQBC6867 3.08 -> 1.65 mrad, fix 690 -> 551 m) and ties or slightly
> loses on others (median across seven frames 2.65 -> 1.65 mrad,
> carried by one frame): a second DEM family is worth having but not
> as a replacement — as a CONSENSUS instrument, where agreement
> raises confidence and disagreement flags model error per azimuth.
> Notably KWHC9160's margin rises 1.83 -> 3.18 on GLO-30 at equal
> error, again confidence rather than accuracy.
>
> The island bias is NOT an SRTM defect: GLO-30 carries the same
> profile (8.1 vs 7.8 mrad against a much lower observed top). Both
> are surface models, so shared canopy is one suspect — but the size
> of the residual (~7 mrad = ~43 m at 6.2 km) and OREJ1026's failed
> auto-level (sea horizon hidden behind the island) point at a
> simpler culprit for that frame: the PITCH PRIOR. Without a horizon
> the attitude falls back to the phone inclinometer (+-0.5 deg =
> +-9 mrad), and a constant pitch error shifts the whole observed
> profile — indistinguishable from a DEM height bias in a single
> frame, and fatal to the locked-offset foreground matching that E5b
> relies on. Foreground absolute-height work therefore requires
> frames where the horizon IS visible beside the feature.
>
> The menu, ranked by measured or expected return: (1) SRTM1 over
> SRTM3 — done, integrity gain measured (E4u); (2) dual-DEM
> consensus SRTM1+GLO-30 — data in place, cheap, flags model error;
> (3) canopy correction — ESA WorldCover 10 m landcover is on a
> reachable S3 bucket, so a per-class height offset (a poor man's
> FABDEM, which itself is CC BY-NC-SA) is implementable here;
> (4) local calibration survey — at a known point, measure per-azimuth
> DEM-vs-observed offsets once and apply them as a correction field
> for the operating area, the practical navigator's move; (5)
> coastline registration — the waterline from SAM against the OSM
> coastline vector to correct azimuth registration per frame.

> **E5o — the consensus term, built and measured** (`skyfix.py
> --dem2`, six-frame A/B against the field truth, SRTM1 vs GLO-30).
> Two candidate mechanisms were implemented and the data kept one.
> (a) Disagreement WEIGHTING — a Lorentzian per-azimuth down-weight
> where the families disagree, suppressed weight charged `C0_NOINFO`
> — LOSES: median 563 -> 849 m over the six frames (APST5638
> 162 -> 545, PQBC6867 690 -> 1371), because disagreement azimuths
> carry discriminative relief along with the model error, and
> removing them costs more position information than the error they
> contain. It survives as the study-only flag `--dem2-weight`.
> (b) The SPLIT STATISTIC — cross-solve the fix on the second family
> alone, report the separation as `dem_split_m` — is the missing
> error predictor this toolkit has been hunting since E4v:
>
> | frame    | error (SRTM1) | dem_split | basin margin |
> |----------|--------------|-----------|--------------|
> | APST5638 |   162 m |  136 m | 0.52 |
> | MYQR7719 |   396 m |  148 m | 1.45 |
> | KWHC9160 |   436 m |  101 m | 1.83 |
> | PQBC6867 |   690 m |  735 m | 0.59 |
> | EWAC7374 |  1181 m |  229 m | 0.25 |
> | SRYK4301 |  3066 m | 5286 m | 1.19 |
>
> Rank correlation with the actual error: split +0.77, margin
> -0.14 on the same frames. The decisive case is SRYK4301: the
> margin is a CONFIDENT 1.19 while the fix is 3 km off — the one
> failure mode a single-model statistic cannot see, because the DEM
> is wrong in a self-consistent way. The split sees it, and the
> autopsy explains why: the SRTM1 tile carries phantom terrain over
> open water north of Foça (raw-SRTM sea noise, 13-20 m of false
> height out to ~2 km offshore), which sustains a flat false ridge
> across az 350-352 deg where GLO-30 — water-masked — correctly
> shows the distant coast. Angularly the wedge is small (p90
> disagreement 1.6 mrad) but positionally it is worth kilometers,
> which is exactly what a cross-solve integrates and a per-azimuth
> weight does not. Interface: `--dem2 <store>` computes the
> statistic, `--max-dem-split <m>` gates on it ("DEM families
> disagree"), and the E5n menu's item (2) is hereby measured: the
> second family's value is as a WITNESS, not a judge — it should
> never touch the cost, only the confidence. (`e5o_visual.py`
> draws the scatter and the per-azimuth disagreement panels.)

> **E5p — the landcover correction, applied and calibrated**
> (`canopy.py`: ESA WorldCover 10 m classes from the public S3
> bucket, CC BY 4.0 — the licence-clean poor man's FABDEM — drive a
> per-class offset on the DEM store; six-frame batch against the
> field truth). Building the stores first widened E5o's autopsy:
> essentially 100% of water pixels in the SRTM1 Aegean/Marmara
> tiles carry nonzero height (GLO-30: 1-3%) — raw-SRTM sea noise is
> not a Foça anomaly, it is the whole coastline. The calibration,
> medians over the six frames:
>
> | variant                    | median error |
> |----------------------------|--------------|
> | raw SRTM1                  | 563 m |
> | water mask only            | **521 m** |
> | water mask + tree -8 m     | 637 m |
> | water mask + tree -15 m    | 619 m |
>
> Verdict in two halves. The WATER MASK is kept: it never hurts,
> helps modestly wherever it can (APST 162 -> 153, PQBC 690 -> 613,
> MYQR 396 -> 390) and visibly heals the confidence statistics
> (MYQR margin 1.45 -> 3.44 at half the rms). The CANOPY CONSTANT
> is rejected by measurement: it swings single frames wildly in
> both directions (PQBC 690 -> 365 at -15 m, but APST 162 -> 541)
> and loses at the median — C-band penetrates partway into the
> canopy and canopy height varies, so a constant is the wrong
> model; per-pixel canopy height (FABDEM's actual move) is the only
> version worth revisiting. `canopy.py` defaults are now water mask
> only, and `DEMs_SRTM1_WM` is the operational store.
>
> Two honest corrections to E5o's story. (1) Erasing the phantom
> offshore plateau does NOT rescue SRYK4301 (3066 -> 3066 m): the
> sector profiles now agree between families and the fix does not
> move, so that frame's 3 km failure is deeper than the water
> noise — the 10.3 deg window is simply ill-conditioned against the
> remaining model error. (2) Re-running the consensus split on the
> water-masked pair shows the split was never the phantom ridge's
> doing either: SRYK's split barely moves (5286 -> 5091 m) and
> still towers over the clean frames (81-525 m), so the GATE
> survives intact — but the fine-grained rank correlation among the
> well-solved frames washes out (0.77 -> ~0.2 over the full set,
> APST now 525 m of split at 153 m of error). The right reading:
> `dem_split_m` is a FAILURE DETECTOR, not a proportional error
> estimator — a model-perturbation jackknife whose large values
> mark ill-conditioned frames, to be thresholded (`--max-dem-split`
> ~1 km) rather than regressed. (`e5p_visual.py` draws the
> per-frame bars and the healed SRYK sector.)

> **E5q — the lost boat** (`e5q_voyage.py`; nine-stop cruise through
> the Izmir gulf, Mytilene strait and Chios strait). The end-to-end
> scenario the study builds toward: a boat with NO position prior
> takes a 360-deg skyline panorama at each stop and searches the
> whole 155 x 104 km region (1.5 km sea-only grid, refined to 60 m).
> Honesty rules: the panorama is rendered from GLO-30 and matched
> against water-masked SRTM1 (independent families — the E5o-measured
> model error, not an inverse crime), each stop draws a compass bias
> (sigma 2.5 deg), an elevation bias (1.5 mrad) and correlated 1-mrad
> extraction noise, and between stops a log+compass dead-reckoning
> track (5% / 2 deg) is fused with the fix by inverse covariance,
> falling back to DR when the margin gate refuses.
>
> Result: all nine blind fixes land — median 61 m, worst 238 m —
> while DR alone drifts to 1-2 km within a leg or two. No stop was
> vetoed; margins ran 0.45-9.06. Fusion neither helped nor hurt
> (median 69 m): with fixes this good its only job is bridging
> vetoes that never came.
>
> The number to hold against the field data: the SAME matching
> machinery scored 914 m median on real single frames (E4v) and
> 61 m here. The gap is the measured price of narrow angular
> coverage plus real extraction against a synthetic 360-deg
> panorama with 1-mrad noise — which is E5m's ranking read in the
> other direction: give the instrument wide coverage and a clean
> silhouette, and two independent DEM families disagreeing with the
> world only costs tens of meters. A camera that sweeps the full
> horizon (or a pan stitched with per-frame offsets, E4v) is worth
> more than every other improvement in this study combined.

> **E5r — per-peak height residuals: the vertical-angle check as a
> diagnostic** (`e5r_peaks.py`; every solved E4v frame, 34 usable of
> 56 after refusing garbage extractions at |offset| > 100 mrad,
> 81 matched crests). The classical navigator's move — compare a
> peak's apparent height above the sea to its charted height — is
> already inside the profile cost (E5b measured its worth: locking
> the offset took the foreground fix 2850 -> 559 m), so re-solving
> with it would double-count pixels. What it CAN do post-fix is
> decompose the residual: a global offset (median residual —
> pitch/refraction/uniform-bias mixture, inseparable in one frame
> per E5n) and per-crest height residuals in METERS via each
> crest's range.
>
> Measured, aggregated: the central mass sits at median -1.2 m,
> IQR -24..+3 m — on well-fixed frames the DEM's silhouette-crest
> heights are good to ~10 m. Per site: Maltepe +0.6 m (31 crests,
> essentially perfect), Foça -3..-7 m (photo slightly BELOW the
> model — the expected sign if the DSM carries canopy the rock
> silhouette lacks, but small), and the two inland Bergama frames
> -395 m — not a height error but the position error of bad fixes
> leaking into the height channel, exactly the ambiguity the
> decomposition cannot break on a single frame.
>
> As an error predictor on frames with >=2 crests (n=21, where the
> statistic is non-degenerate — with one crest the median-offset
> removal absorbs the signal by construction): rank correlation
> with actual error is +0.60 for |median dh| and +0.50 for the
> crest-residual spread, against +0.34 for the basin margin on the
> same frames. So the vertical-angle check earns its keep in the
> same role as dem_split (E5o): a WITNESS that flags a suspect fix,
> not a judge that moves it.
>
> Wired into the instrument as `skyfix.py --max-peak-dh` (the
> residuals are always reported as `peak_dh_m` / `peak_dh_mad_m` /
> `n_peaks` when computable). The gate fires on either measured
> failure signature: |median| over the limit (a shared height offset
> across the crests), or the crest SPREAD over 3x the limit — the
> EURK8793 re-solve showed why the second is needed: a wrong fix
> whose crest residuals straddle zero (median +4 m) while
> scattering 476 m. At the measured separation (30/90 m) the gate
> rejects five of the eight >1.4 km failures on the field set with
> ZERO false positives (every fix under 1 km measured |median|
> <= 16 m, spread <= 39 m); the three it passes are bearing-type
> misses that do not touch the height channel and remain
> margin/jackknife business. Validated live: KWHC9160 passes
> (+1 m, 1 crest), EURK8793 is refused by the witness alone.

> **E5s — why an AR skyline overlay cannot sit on the photograph
> from metadata alone** (`e5s_pfoto.py`; two PeakFinder-labeled
> iPhone frames, Kumlubük, seconds apart, same spot). The app crops
> the 4:3 sensor to screen aspect, so the intrinsics are exact
> (f_px = H * 24/36 = 2688); the file carries GPS and a compass
> heading and NOTHING about pitch/roll. Fitting attitude against
> the extracted silhouette at the fixed GPS position decomposes the
> misfit: (1) PITCH dominates — +24 vs +168 mrad between the two
> shots (the photographer recomposed; a level-camera assumption
> misses by up to 9.6 deg of ridge placement); (2) compass — raw
> headings differed by 8.0 deg between the shots, yet the fitted
> TRUE azimuths agree to 1.2 deg (220.3 / 221.5), i.e. the compass
> wandered, the camera barely did (E5m's 4.2-deg median compass
> error, live); (3) roll 0-2 deg. Two contamination effects worth
> recording: the seam extractor climbs the app's white label pill
> and leader line (masked by a wide rolling-median upward-outlier
> rule: terrain has no 20-mrad plateau a quarter-frame wide), and
> in haze the strongest edge mid-frame is PEAKFINDER'S OWN DRAWN
> TRACE, so an extractor can inherit the overlay's error — ink on
> the photograph becomes evidence. After the attitude fit our curve
> sits on the true ridge where the app's overlay visibly does not:
> the overlay's misfit is sensor-pose error, not terrain-model
> error, and pixels beat sensors whenever both are on offer.

> **E5t — five blind solves on PeakFinder field frames** (Kumlubük;
> search engine as-is: 6 km box, water-masked SRTM1, --auto-level,
> peak witness armed; GPS for scoring only). Errors 291 / 647 / 680
> / 1592 / 3902 m — and the GATES WERE RIGHT 5/5: the two accepted
> fixes were the two best (291, 647), all three failures refused
> with correct reasons (margin 0.03 / box boundary + rms 12.1 /
> margin 0.03 + rms 29.1). Zero false accepts, again.
>
> The shared root cause of the failures: the sea-horizon auto-level
> never engaged on ANY frame (attitude_source=prior on all five) —
> in a bay the waterline is terrain-backed, not sky-backed, and the
> E4q-2 photometric continuity rule correctly refuses it as a sky
> horizon. Pitch then falls back to the zero prior and el_offset
> saturates at the +-10 mrad bound on the pitched-up frames — the
> telltale is IN the output.
>
> The obvious fix was tried and MEASURED TO LOSE: rerunning the
> three failures with --pitch-sigma 6 (wide pitch freedom in the
> joint solve) improved the fit (rms 12.1 -> 5.3) and worsened the
> position (680 -> 3230 m; 3902 -> 4382) — the study's recurring
> lesson again: unanchored nuisance freedom eats position
> information (E5b, E4v heading-window, E5o weighting). Attitude
> from pixels works at a KNOWN position (E5s); as a free parameter
> of the position search it collapses.
>
> What the code should gain instead (proposed): (1) a TERRAIN-BACKED
> WATERLINE level — a second acceptance class for the horizon
> detector (water-to-land step instead of water-to-sky) that anchors
> roll and approximate pitch in bays, where these five frames all
> had a usable waterline; (2) the E5s overlay-ink/graphics mask
> (wide rolling-median upward-outlier rule) in observation(), which
> also catches power lines and drawn AR traces; (3) same-station
> pan fusion for phone frames taken seconds apart at different
> headings (three of the five form such a pan; E4v measured pan
> fusion at 168 m vs 914 m single-frame).

> **E5u — the waterline anchor: built, gated, and honestly parked.**
> The E5t proposal (1) implemented end to end: multi-row SAM prompts
> (beach frames are not MaSTr's USV geometry), upward region growth
> past glare bands, the seam boundary as a ceiling (a waterline is
> below the silhouette by definition), waterline-physics gates for
> segmentation-proposed seeds (soft haze edge -> threshold 1.5 and
> frac 0.15; boats legitimately float below -> below-veto 8%), and
> a roll sanity clamp after the one unclamped acceptance measured
> roll +14 deg off a glare-band edge. Measured on the five Kumlubük
> frames: availability 0/5, fixes identical to the E5t baseline,
> zero poisonings. Each physical obstacle is now named: glare bands
> split SAM's water region, haze softens the far waterline below
> the edge detector's floor, boats sit on the line, the near-coast
> limb curves it. The chain stays in as safe infrastructure; its
> missing piece is a TRAINED water segmenter as the mask source
> (eWaSR is already in hand) instead of prompt-grown SAM masks.

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
