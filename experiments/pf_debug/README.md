# Skyfix Eye: look at a solve through the camera

A debug viewer, nothing more: it shows what the camera saw and what the
solver thought it was looking at, step by step. PeakFinder draws the
scene from each pose; nothing flows back into the solve.

    cd experiments
    python3 skyfix.py PHOTO --center LAT,LON ... --view-json run.view.json
    python3 pf_debug/serve.py run.view.json        # opens http://127.0.0.1:8765

A ready-made example (the Boğaz frame, hand-drawn skyline, the 54 m
configuration) is included:

    python3 pf_debug/serve.py pf_debug/example_bosphorus.view.json

## What you see

| layer | meaning |
|---|---|
| PeakFinder render | the terrain PeakFinder draws from this step's position, heading, pitch and field of view, at the solver's camera height |
| photo, levelled | the photograph with its roll removed and scaled to the same field of view |
| yellow line | the skyline the solver extracted from the photo, placed with this step's heading offset and β |
| cyan line | the skyline **our** DEM predicts from this step's position; the solver minimises the gap between yellow and cyan |
| magenta line | the sea line that set pitch and roll (`skyfix --level-npy`), photo side like yellow; dotted when the levelling fit refused it and the priors were used |
| white dashed line | the sea horizon the levelling assumed, `-dip(z)`, world side like cyan. At β = 0 the magenta line sits on it by construction; a shoreline closer than the horizon lies below it |

The steps are the camera's own claim (box centre, compass heading,
pitch prior), the best basins of the coarse grid scan (up to five, worst
to best, with the 4-cell suppression distance the ambiguity test uses),
the two refinements around basin #1, and the final fix. The grid scan
is exhaustive, so there is no search path to replay: view files from
before the basins showed the running best in raster order instead
(always the south-west corner first, then along the south row), which
read like a path and was not one. The cost map marks the basins as
ranked rings and draws a line only from #1 through the refinements;
the side panel lists them with their cost over #1. Two basins of nearly
equal cost far apart are what an "ambiguous landscape" refusal means.

With `skyfix --land-observer H` each step carries its own camera height
(DEM ground there + H), which the page also hands to PeakFinder; with
`--land-only R` the cells outside the search (no land within R) are
drawn dark on the cost map. With `--waterline-band LO,HI` the β band is
one-sided; the gauge, the inner map's dashed box and the bound chips use
`[LO, HI]`.

A red chip means a nuisance parameter (heading offset or β) is sitting
on its search bound at that step. That is the clamping signature behind
the campaign's false accepts (E5at, E5bf, E5bi): the fit looks
acceptable only because the parameter was not allowed to go where it
wanted.

**Variables & constraints** (top of the side panel) states the problem
being solved, `min J(n, e, ψ, β)` subject to its bounds, and for the
current step shows each decision variable as a gauge inside its bound
(red when it sits on it), the inner heading × β landscape with the
feasible box dashed, and where the optimum would go if that box were
three times wider. A red "binding" chip means the bound, not the data,
chose the answer. Below it: the parameters held fixed (pitch prior,
roll, field of view, camera height, near-field cut), and under *Solve*
the acceptance checks, which run after the optimum is found and are
not constraints of the search.

Keys: ←/→ step, space play, L unlocks PeakFinder so you can drag around
(L again snaps back to the solver's pose). "open on peakfinder.com"
opens the same view on the website.

## Two things to check the first time

- **Overlay alignment.** If the cyan line and PeakFinder's terrain
  disagree by the same amount on *every* step, it is a projection
  mismatch, not a solver result: try "PF fov is: vertical", or nudge
  shift/scale under *Overlay alignment* (saved per browser). Small,
  pose-dependent differences between cyan and PeakFinder are real: they
  are two different DEMs and refraction models.

  Two such mismatches are measured and handled in the page (headless
  Chrome, PeakFinder's own coordinate grid, fov 60 and 73.7°, altitude
  0 / 1.69 / 5°, canvases 1200×600 to 1200×1200):
  - PeakFinder draws elevation *e* at `centre − f·tan(1.2·e + altitude)`,
    *f* from the horizontal fov. Its altitude has the opposite sign of
    this page's pitch and lives in a space stretched ~1.2× vertically, so
    the page sends `−1.2 · pitch`. The 1.2 stretch about the centre has
    no API switch and remains: a few px for terrain a degree or two off
    the centre line. The peakfinder.com link still sends the raw pitch
    (not measured on the website).
  - PeakFinder sizes its drawing buffer only on a window resize. It kept
    the size the stage had at init, before the stage took the photo's
    aspect, and the browser stretched that render up to 1.6× vertically.
    `layout()` now fires a resize when the buffer and the stage differ.
- **Camera height.** PeakFinder stands the viewer on its own terrain at
  that point; the page lifts it by `z − ground` so the eye sits at the
  solver's height above sea level. The status line at the bottom left
  shows both numbers.

PeakFinder's module loads from peakfinder.com in the browser (its API
licence allows embedding; see the PeakFinder-API repository). Without
internet the page still works, minus the render.
