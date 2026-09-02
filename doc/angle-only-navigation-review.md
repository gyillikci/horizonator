# Angle-only navigation: literature review, and what it says about this instrument

Written 2026-09-01, after the E5ba / E5bb false accepts on the Bodrum
frames. The question behind it: our instrument measures nothing but
angles, so what does the angle-only navigation literature already know
about problems shaped like ours, and which of its results are
actionable here?

Sources were reached through search only — the network egress proxy in
this session blocks arxiv, springer, mdpi and wikipedia to direct
fetch, so the summaries below rest on search abstracts and on standard
results in the field. Every claim that carries a number for *our*
instrument was measured here and is marked as such.

---

## 0. What kind of angle-only problem this is

A skyline fix is an angle-only measurement in the strict sense: for
each azimuth the photograph gives one elevation angle to the terrain
crest, and nothing else. No range, no time series, no baseline. In the
taxonomy of the literature we sit in the least-favourable corner:

| axis | favourable case | ours |
|---|---|---|
| observer motion | maneuvering observer | single static station |
| landmark identity | known, discrete, labelled | a continuous unlabelled curve |
| measurement type | line of sight (2 angles per landmark) | elevation only, azimuth implied by the pixel column |
| attitude | known from a star tracker / gyro | unknown, co-estimated |
| target model | exact ephemeris / survey | a 30 m DEM with a known height deficit |

Everything below should be read against that table. Most of the
angle-only literature buys its observability with observer motion,
which we do not have; the sub-literature that does not (surveying
resection, P3P, horizon-based OPNAV) buys it with an exact target
model, which we also do not have. That is the honest position.

---

## 1. Bearings-only target motion analysis: the observability core

The founding result is Nardone and Aidala's observability criteria for
bearings-only TMA, obtained by solving a third-order nonlinear
differential equation for the observability condition. The operative
consequence, repeated in every later treatment: **the lack of a direct
range measurement makes the problem instantaneously unobservable.**
Range enters only through the change in the bearing geometry produced
by observer motion.

Two refinements matter to us more than the headline:

- **Motion is necessary but not sufficient.** The maneuver literature
  is explicit that for certain observer maneuvers the estimate stays
  unobservable *even when the bearing rate is nonzero* — which the
  authors flag as the source of common misconceptions about acceptable
  own-ship motion. Movement that looks informative can carry no
  information. This is the same failure mode as our clamped nuisance
  parameters: a number that looks like evidence but is an artifact of
  the parameterisation.
- **Higher-order motion.** The cooperative-estimation work states the
  requirement sharply: the observer needs higher-order motion than the
  target. For a static target (terrain) a constant-velocity observer
  suffices; a static observer never does.

In-orbit angles-only rendezvous sharpens the counting: **three
non-coplanar (or four coplanar) measurement epochs with maneuvers
between them** are needed to resolve full relative position and
velocity, and navigation performance then depends directly on the
trajectory flown. Modern versions replace the heuristic S-maneuver on
the line of sight with an explicit Fisher-information cost, choosing
the single impulse that maximises observability under an orbital-energy
constraint.

**Read across to us:** we take one photograph from one point. Under
this literature that is the unobservable case, and the only reason we
get a position at all is that the "target" is a *known-shape extended
body* rather than a point. All our information comes from the shape
constraint, none from geometry change. That is worth stating plainly
because it sets the ceiling on what a single frame can ever do.

## 2. Static angle-only resection: the danger circle and the danger cylinder

The surveying literature solved our actual problem — angles from one
unknown station to known points — three centuries ago, and it also
found the degeneracy.

**Snellius–Pothenot / three-point resection.** Occupy an unknown point,
measure horizontal angles to three known control points. The classical
result: **if the unknown point lies on the circle through the three
control points, the solution is indeterminate** — there are infinitely
many positions consistent with the observed angles, because a chord
subtends equal angles from every point of its circle. This is the
"danger circle"; near it the solution is merely weak rather than
impossible, and surveyors are taught to avoid the neighbourhood, not
just the curve. Collinear or near-collinear control points are the
other failure.

**P3P / space resection.** The photogrammetric version, with a
calibrated camera, has the same structure with more surfaces. The pose
is ambiguous when the projection centre is coplanar with the reference
points, and the full set of critical configurations is known: the
**danger cylinder**, the **Euclidean horopter** (a twisted cubic
through the camera centre and the control points) and its degenerate
forms. Recent work maps the companion surface of the danger cylinder
and how the solution count changes across it. The practical statement
in that literature is the one we need: the instability is a property of
the *configuration*, not of the image quality.

**Read across to us — this is the most actionable item in the review.**
Our gate quartet is entirely *a posteriori*: it looks at the fit that
came back. The resection literature says the dangerous configurations
are computable **before any photograph is taken**, from the control
geometry alone. We have the control geometry: it is the DEM. A map of
resection conditioning over a coastal area, computed from the DEM by
itself, would tell an operator where this instrument can work and where
it cannot, and it would have flagged the Bodrum marina without ever
seeing the photograph. We do not have this and should.

## 3. Angles-only orbit determination: short arcs and ridge estimation

Angles-only initial orbit determination (IOD) is the oldest
mature angle-only estimator: Laplace (1780), Gauss (1857), and
Gooding's iterative method, now the most widely used. Comparative
studies find Gauss best over short intervals, and Gauss and Double-R
robust across interval length.

The finding that transfers is the failure mode. **Accuracy collapses on
short arcs** because the geometric constraint is weak, and observation
error then produces large errors specifically in the *estimated
ranges* — the angles stay fitted, the ranges go wrong. Work on the
Gooding algorithm treats this explicitly as an **ill-conditioned
problem** and applies **ridge estimation** to it.

**Read across to us:** this is precisely the E5t lesson the campaign
keeps re-deriving — "a better fit is not a better position" — arrived
at independently by the astrodynamics community with an ill-conditioning
diagnosis attached. And their remedy is instructive. They regularise
explicitly, with a stated ridge parameter, so the conditioning is
visible in the output. We instead *clamp*: the elevation offset and the
heading trade are confined to hard bands. E5at and E5bb both showed
what that costs — a clamped nuisance parameter compares two basins
under the same artificial constraint and manufactures a margin.
Replacing the hard bands with an explicit prior/ridge term and
reporting the regularised condition number would put the same
information in the open.

## 4. Horizon-based optical navigation

The closest relative in spacecraft navigation is horizon-based OPNAV:
observe the limb of a planet, infer position. Christian's tutorial and
the Cholesky-factorisation work give analytically exact, non-iterative
solutions for every combination of unknown position, attitude and
target shape, by transforming the triaxial ellipsoid into a unit sphere
so the limb projection constraint collapses; the accompanying
subpixel limb localisation uses Zernike moments. Short-arc horizon-based
OPNAV by total least squares, and vision-based LEO navigation using
stars plus the horizon as an alternative PNT source, extend it.

**What transfers:** the framing (position from a silhouette), the
insistence that limb *localisation* accuracy is a first-class subproblem
worth its own subpixel estimator, and the star-plus-horizon
architecture — attitude from one source, position from the horizon.

**What does not:** the entire method rests on the target being an
analytic quadric known to metres. Our target is a DEM whose crest
height deficit we have measured at 9–12 m and only partly explained
(canopy accounts for about half). A closed-form solution against an
exact ellipsoid has no analogue when the shape model is the dominant
error term. The star-plus-horizon split, though, is exactly the
architecture our own results argue for — see §7.1.

## 5. Geometry, CRLB and GDOP for angle-only fixes

For static angle-only localisation the accuracy is characterised by the
**Cramér–Rao lower bound**, the inverse of the Fisher information
matrix, and reported through scalar summaries — GDOP, the concentration
ellipse, circular error probable. Bishop and co-workers' optimality
analysis of sensor–target geometries characterises which configurations
minimise the CRLB, and a useful counter-intuitive result from the AOA
work: **uniform angular separation of sensors around the target is
suboptimal in general**, and optimal configurations are not unique even
when all sensors are equidistant. Recursive CRLB formulations then let
optimal geometry be built up incrementally.

**Measured read across to us (n=170 unique solves in
`experiments/out/*/*.json`, ellipses with anisotropy > 1.15):**

- Our reported 1σ major axis is optimistic. **Median actual error /
  reported σ_major = 1.73 over all solves, 1.53 over accepted ones**
  (BD9 was 2.7). The bound is not a bound.
- The *orientation* of our covariance carries essentially no
  information about where the error actually lies: the median angle
  between the reported major axis and the true error direction is
  **42°**, against 45° for a random orientation, with 67/170 inside
  30° where chance alone gives about 57.

The CRLB literature explains both. A CRLB is a bound on *noise-driven*
error under a correct model. Our covariance is computed from extraction
noise and a DEM sigma floor, so it describes the estimator's response to
photon and segmentation noise. Our dominant error is **model error** —
the DEM's crest deficit, its missing canopy and buildings, its 30 m
posts — which is not noise, is spatially correlated, and has no reason
to align with the measurement geometry. That is why the magnitude is
too small and the direction is uninformative. Any honest σ needs a
model-error term propagated to position, not just a noise term.

## 6. The maritime classic: the vertical sextant angle

Coastal navigation has an angle-only ranging method that is exactly our
elevation measurement: the **vertical sextant angle**. Measure the
angle subtended by a charted object of known height — a lighthouse — and
read the distance off a table. One object gives a distance line of
position; two give a fix. Its documented error budget is a
point-for-point match to ours:

- **Datum error.** Chart heights are referenced to MHWS, so the water
  is normally below datum and an uncorrected reading puts the object
  *closer* than it is. Our analogue is the crest deficit: the DEM's
  reference surface is systematically below the real skyline, which is
  why `--crest-dh` exists and why it is regional (9 near Marmaris, ~12
  on the Theodolite set) rather than universal.
- **Refraction** is called out as unpredictable near the horizon and
  severe for narrow angles at long range — the same term that forces
  our dip anomaly correction and the `--dt-air-sea` flag.
- **Height of eye** is dismissed as below sextant precision. Ours is
  not: `z` enters the dip directly and we carry it explicitly.
- **A minimum angle.** The rule of thumb is that the corrected angle
  should exceed 20 arcminutes (≈5.8 mrad) — below that the range is not
  trustworthy. **We have no such rule and we should.** A crest at 3.4 km
  standing 50 m above the eye subtends 14.7 mrad, which passes; but the
  relevant angle for us is not the crest height, it is the *relief*
  across the scene, and BD9's was 11.1 mrad at the true position.

## 7. What is actually useful for us

### 7.1 Our elevation offset throws away the only range information we have — and the data shows it

This is the review's main finding, and it comes straight from §6.

The elevation angle to a crest of known height *is* a range
measurement: el ≈ (h − z)/r − r/2R. Given h from the DEM, each
column's absolute elevation angle is a vertical sextant angle and
therefore a pseudo-range. But we co-estimate a free elevation offset
β to absorb attitude error. β subtracts a constant from every elevation
angle, which removes exactly the common-mode component that carries
scale. **With β free we are not doing angle-only ranging at all — we
are doing differential angle-only navigation, matching shape and
discarding scale.**

The prediction is that the tighter the β band, the more range
information survives, and the better the position. Measured across the
campaign's solves, grouped by the attitude source that sets the band:

| attitude source | β band | n | median position error | β pinned at the band edge |
|---|---|---|---|---|
| `prior` | ±10 mrad | 111 | **847 m** | 53/111 |
| `radon` | ±2 mrad | 38 | **347 m** | 29/38 |
| `waterline` | ±5 mrad | 25 | **244 m** | 0/25 |

The direction is what the mechanism predicts and the spread is a factor
of 3.5. The clamping column adds the second half of the story: the
`waterline` group is best not because its band is tightest but because
its level is *right* — not one of its 25 solves needed to push β to the
edge — whereas `radon` spends 29 of 38 solves fighting its own band.

**Tested, and it failed.** The controlled experiment called for below
was run on the Milas apron frame (E5bd in the study doc): same frame,
same columns, pitch supplied at its truth-measured value and the beta
band squeezed from +-10 to +-3.8 mrad. The fit became the best of the
campaign (rms 1.36 mrad, margin 5.78) and the position went from 301 m
to **902 m wrong**. Tightening the level constraint does not, on its
own, recover position. The table below therefore records a correlation
whose mechanism is not established, and the paragraph that follows
explains why it could never have been more than that.

**This comparison is confounded and must not be read as a controlled
experiment**: frames that yield a waterline are near-coastal frames with
favourable geometry, and frames that fall back to `prior` are the hard
ones, so scene difficulty and attitude quality vary together. The clean
test is the same frame solved with the band tightened, and it has not
been run. But it is consistent with the one clean data point we have:
the campaign's best single fix, **42 m**, came on a frame where a real
measured pitch was supplied.

**Action:** treat the level as a first-class measurement with its own
error budget rather than as a nuisance to be absorbed, and run the
controlled test — one frame, β band swept, position vs band width.
Christian's star-plus-horizon architecture is the same idea: get
attitude from an independent source so the horizon can be spent on
position.

### 7.2 Build a DOP map from the DEM, before the photograph

From §2 and §5. The degeneracies of angle-only resection are properties
of the control geometry, and our control geometry is the DEM alone. For
any candidate position we can ray-cast the crest set and compute the
Fisher information of the elevation profile with respect to (north,
east) — no photograph needed. That gives a per-cell GDOP for a coastal
area, and it subsumes the near-field pre-screen E5ba proposed as a
fifth gate: instead of a hand-set threshold on median subject range, a
principled scalar that already accounts for azimuth spread, relief and
range distribution together. It also inherits the danger-circle
warning: check for near-degeneracy explicitly rather than hoping the
basin margin notices.

### 7.3 Replace the hard bands with an explicit ridge, and report the conditioning

From §3. Both false accepts in this campaign involved a nuisance
parameter at a band edge (E5at heading; E5bb's masked variant at −5.0
of ±6). The IOD community treats the identical situation as
ill-conditioning and regularises it openly. A ridge term with a stated
weight, plus the regularised condition number in the JSON, converts an
invisible artifact into a reported number.

### 7.4 Give σ a model-error term

From §5, measured: σ is 1.5–1.7× optimistic in the median and its
orientation is uninformative. Propagating a DEM crest-error term (we
have measured it: 9–12 m regional bias, ~18 m spread, of which canopy
explains about half) through the same Jacobian would at least put the
magnitude in the right decade, and would automatically inflate σ in
exactly the near-field scenes where a metre of DEM error is worth the
most angular error.

### 7.5 Adopt the sextant's minimum-angle rule

From §6. The mariner refuses a vertical sextant angle below ~20
arcminutes. Our equivalent — a minimum relief, or better, a minimum
angular *signal-to-model-error* ratio across the scene — is a one-line
gate with three centuries of operational precedent behind it. BD9 had
11.1 mrad of relief against a DEM whose crest error is ~10 m, which at
3.4 km is 2.9 mrad: a signal-to-model-error ratio under 4.

### 7.6 The one structural upgrade: take a baseline

From §1, and the only item here that changes the observability class
rather than the estimate. Every result in bearings-only navigation says
a single static station is the unobservable case and that motion or
multiple stations fixes it. We are on a boat. Two skyline fixes from
points 500 m apart, with the baseline known from log and compass to a
few percent, is a *different problem* — one where range enters
geometrically instead of only through the height model, and where the
β degeneracy of §7.1 no longer aligns with a position shift because the
two stations see the same crests at different (h/r). This is the
highest-value experiment the review suggests, and the campaign already
has the frames to try it: several of the Bodrum and Marmaris sets were
shot from different points along the same shore within minutes.

---

## Sources

Reached by search; abstracts and summaries only, as noted above.

- Nardone & Aidala, *Observability Criteria for Bearings-Only Target Motion Analysis* — https://www.semanticscholar.org/paper/Observability-Criteria-for-Bearings-Only-Target-Nardone-Aidala/3e40ae620c1a3619efb1c84a4fb228727b5f28d4
- *Observability Criteria for Angles-Only Navigation* — https://www.researchgate.net/publication/224594864_Observability_Criteria_for_Angles-Only_Navigation
- *Observability Criteria and Unobservable Maneuvers for In-Orbit Bearings-Only Navigation*, J. Guid. Control Dyn. — https://dx.doi.org/10.2514/1.62476
- *Analytic Optimal Observability Maneuvers for In-Orbit Bearings-Only Rendezvous*, J. Guid. Control Dyn. — https://arc.aiaa.org/doi/10.2514/1.G000612
- *One-step rendezvous guidance for improving observability in bearings-only navigation* — https://www.sciencedirect.com/science/article/abs/pii/S0273117720305421
- *Computational Guidance & Navigation for Bearings-Only Rendezvous (GUIBEAR)* — https://www.researchgate.net/publication/353224415
- *Bearings-Only Tracking: Observer Maneuver Recommendation* — https://www.academia.edu/83956493/Bearings_Only_Tracking_Observer_Maneuver_Recommendation
- *Position resection and intersection* / *Snellius–Pothenot problem* (danger circle) — https://en.wikipedia.org/wiki/Position_resection_and_intersection , https://en.wikipedia.org/wiki/Snellius%E2%80%93Pothenot_problem
- *Simple Solution to the Three Point Resection Problem*, J. Surv. Eng. — https://ascelibrary.org/doi/abs/10.1061/(ASCE)SU.1943-5428.0000104
- *Companion Surface of Danger Cylinder and its Role in Solution Variation of P3P* — https://arxiv.org/pdf/1906.08598
- Ding, Yang, Larsson & Olsson, *Revisiting the P3P Problem*, CVPR 2023 — https://openaccess.thecvf.com/content/CVPR2023/papers/Ding_Revisiting_the_P3P_Problem_CVPR_2023_paper.pdf
- *Snapshot of Algebraic Vision* (critical configurations, horopter) — https://arxiv.org/pdf/2210.11443
- *Addressing the ill-conditioned problem in initial orbit determination via the Gooding algorithm*, Astrodynamics — https://link.springer.com/article/10.1007/s42064-024-0251-3
- *Short-Arc Angles-Only Initial Orbit Determination for LEO* (thesis) — https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=4834&context=theses
- *A Comprehensive Comparison Between Angles-only Initial Orbit Determination Techniques* — https://www.academia.edu/71965222
- Christian, *A Tutorial on Horizon-Based Optical Navigation and Attitude Determination With Space Imaging Systems* — https://www.semanticscholar.org/paper/c23529d2984c8342ec6f1bc4341a0c64bc7d834b
- *Accurate Planetary Limb Localization for Image-Based Spacecraft Navigation*, J. Spacecraft & Rockets — https://arc.aiaa.org/doi/10.2514/1.A33692
- *Short-Arc Horizon-Based Optical Navigation by Total Least-Squares Estimation*, Aerospace — https://www.mdpi.com/2226-4310/10/4/371
- *Vision-based navigation in low Earth orbit — using the stars and horizon as an alternative PNT* — https://www.sciencedirect.com/science/article/abs/pii/S027311772300087X
- Bishop et al., *Optimality analysis of sensor-target localization geometries*, Automatica — https://www.sciencedirect.com/science/article/abs/pii/S0005109809005500
- *Optimal angular sensor separation for AOA localization*, Signal Processing — https://www.sciencedirect.com/science/article/abs/pii/S0165168407003891
- *Geometric dilution of precision for bearing-only passive location in three-dimensional space* — https://www.researchgate.net/publication/273835186
- *Sensor Networks for Optimal Target Localization with Bearings-Only Measurements in Constrained 3D Scenarios*, Sensors — https://doi.org/10.3390/s130810386
- *Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments* — https://arxiv.org/html/2602.11116
- *Vertical sextant angles*, Splash Maritime — http://www.splashmaritime.com.au/Marops/data/less/Nav/Vsa.pdf
- *Distance by vertical sextant angle*, Nautical Science — https://maritimesa.org/nautical-science-grade-10/2020/12/09/distance-by-vertical-sextant-angle/
- *Mathematics of vertical sextant angles* — https://sailingissues.com/vier/mathproof3.html
- *A New Method of Improving the Azimuth in Mountainous Terrain by Skyline Matching*, PFG — https://link.springer.com/article/10.1007/s41064-020-00093-1
- *Large Scale Visual Geo-Localization of Images in Mountainous Terrain* — https://www.researchgate.net/publication/262359169
- *Camera Geolocation From Mountain Images* — https://c4i.gmu.edu/~pcosta/F15/data/fileserver/file/472116/filename/Paper_1570111401.pdf
- *Automatic photo-to-terrain alignment for the annotation of mountain pictures* — https://www.researchgate.net/publication/224254978
- *Horizon-based navigation*, US patent 11,580,690 — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11580690
- *AUSLUN: A Fixed-Hover UAV–USV System for GNSS-Denied Maritime Search and Navigation* — https://arxiv.org/pdf/2606.29875
