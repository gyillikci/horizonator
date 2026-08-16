#!/usr/bin/env python3
"""Build the campaign dossier: one HTML page presenting the whole
skyline-navigation study (E0..E5d) from the committed results, for
handing to a collaborator as a single link. Writes doc/dossier.html."""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DOC = os.path.join(HERE, '..', 'doc')


def b64(path):
    p = os.path.join(OUT, path)
    if not os.path.exists(p):
        return ''
    return base64.b64encode(open(p, 'rb').read()).decode()


e4k = json.load(open(os.path.join(OUT, 'e4k_curve.json')))
e5d = json.load(open(os.path.join(OUT, 'e5d_reference.json')))
prior = e4k['attitude priors (E4f)']


def tier_rows():
    rows = []
    for q in prior:
        if q.get('n'):
            rows.append(
                f"<tr><td>≥ {q['thresh']:g}</td>"
                f"<td>{q['n']} ({100 * q['avail']:.0f}%)</td>"
                f"<td>{100 * q['false_rate']:.0f}%</td>"
                f"<td>{q['far']}</td>"
                f"<td>{q['median_acc_err']:.0f} m</td></tr>")
    return '\n'.join(rows)


HTML = f"""<title>Skyline Fix Dossier</title>
<style>
:root {{
  --paper:#f4f6f8; --card:#fdfefe; --ink:#16232e; --mute:#5b6a76;
  --sea:#1c5cab; --sea-soft:#dbe7f4; --mark:#d95926; --rule:#c9d3db;
  --good:#0b7a3e; --warn:#a86a00; --bad:#b3372f;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#0f1b26; --card:#152534; --ink:#e8edf2; --mute:#93a4b2;
  --sea:#5598e7; --sea-soft:#1c3552; --mark:#e87b4f; --rule:#2c4055;
  --good:#4cc37e; --warn:#e0a437; --bad:#e07a72;
}} }}
:root[data-theme="dark"] {{
  --paper:#0f1b26; --card:#152534; --ink:#e8edf2; --mute:#93a4b2;
  --sea:#5598e7; --sea-soft:#1c3552; --mark:#e87b4f; --rule:#2c4055;
  --good:#4cc37e; --warn:#e0a437; --bad:#e07a72;
}}
body {{ background:var(--paper); color:var(--ink); margin:0;
  font:16px/1.62 Charter,Georgia,'Times New Roman',serif; }}
.wrap {{ max-width:880px; margin:0 auto; padding:34px 22px 80px; }}
.disp, h1,h2,h3, th, .eyebrow, .tile span {{
  font-family:'Avenir Next Condensed','Arial Narrow','Helvetica Neue',
  Arial,sans-serif; }}
.mono, td.n, .num {{ font-family:ui-monospace,'SF Mono',Menlo,Consolas,
  monospace; font-variant-numeric:tabular-nums; }}
/* ---- chart title block */
.tblock {{ border:2px solid var(--ink); padding:26px 28px 20px;
  background:var(--card); }}
.tblock .eyebrow {{ letter-spacing:.22em; font-size:12px;
  text-transform:uppercase; color:var(--mute); }}
h1 {{ margin:6px 0 2px; font-size:44px; line-height:1.05;
  letter-spacing:.01em; text-transform:uppercase; font-weight:700;
  text-wrap:balance; }}
.tblock .sub {{ color:var(--mute); font-style:italic; margin:4px 0 14px; }}
.tiles {{ display:flex; flex-wrap:wrap; gap:0; border-top:1px solid
  var(--rule); }}
.tile {{ flex:1; min-width:130px; padding:12px 14px 6px;
  border-right:1px solid var(--rule); }}
.tile:last-child {{ border-right:0 }}
.tile b {{ font-size:25px; display:block; color:var(--sea);
  font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums; }}
.tile span {{ font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--mute); }}
/* ---- azimuth-ruler divider */
.ruler {{ height:26px; margin:36px 0 8px;
  background:
   repeating-linear-gradient(90deg, var(--rule) 0 1px, transparent 1px 12px),
   repeating-linear-gradient(90deg, var(--ink) 0 1.5px, transparent 1.5px 60px);
  background-size:100% 8px, 100% 16px;
  background-position:bottom, bottom; background-repeat:repeat-x; }}
h2 {{ font-size:23px; text-transform:uppercase; letter-spacing:.05em;
  margin:2px 0 12px; }}
h2 .no {{ color:var(--mark); margin-right:10px; }}
h3 {{ font-size:16px; letter-spacing:.04em; text-transform:uppercase;
  margin:20px 0 6px; }}
p {{ max-width:66ch; margin:10px 0; }}
.mute {{ color:var(--mute); }}
table {{ border-collapse:collapse; margin:14px 0; width:100%;
  font-size:14.5px; }}
th {{ text-align:left; font-size:12px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--mute); font-weight:600;
  border-bottom:1.5px solid var(--ink); padding:6px 12px 6px 0; }}
td {{ border-bottom:1px solid var(--rule); padding:7px 12px 7px 0;
  vertical-align:top; }}
td.n {{ white-space:nowrap; }}
.scroll {{ overflow-x:auto; }}
.panel {{ background:var(--card); border:1px solid var(--rule);
  border-left:4px solid var(--sea); padding:6px 20px 10px; margin:18px 0; }}
.panel.neg {{ border-left-color:var(--bad); }}
.panel.field {{ border-left-color:var(--good); }}
img.fig {{ max-width:100%; border:1px solid var(--rule);
  background:#fcfcfb; margin:8px 0; }}
.figrow {{ display:flex; gap:10px; flex-wrap:wrap; }}
.figrow img {{ flex:1; min-width:250px; max-width:100%; }}
.cap {{ font-size:13px; color:var(--mute); font-style:italic;
  margin:2px 0 14px; }}
.ok {{ color:var(--good); font-weight:600; }}
.no2 {{ color:var(--bad); font-weight:600; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:14px;
  background:var(--sea-soft); padding:1px 5px; border-radius:4px; }}
footer {{ margin-top:46px; border-top:2px solid var(--ink);
  padding-top:10px; font-size:13px; color:var(--mute); }}
</style>
<div class="wrap">
<div class="tblock">
 <div class="eyebrow">GPS-denied coastal navigation · survey of works
 E0–E5d · Aegean test area 36–38°N 26–28°E</div>
 <h1>Skyline Fix Dossier</h1>
 <div class="sub">Position from a photographed horizon, matched against
 a digital elevation model — studied, gated, fused, ported, and given a
 face.</div>
 <div class="tiles">
  <div class="tile"><b>15.8 m</b><span>median fix · 100 km box
   (synthetic, E3)</span></div>
  <div class="tile"><b>0 %</b><span>false accepts at margin ≥ 1.5
   (203 real photos)</span></div>
  <div class="tile"><b>~5 s</b><span>full-circle solve after the FFT
   (was ~60 s)</span></div>
  <div class="tile"><b>2.8 m</b><span>fused final error, 2 h passage
   (E5c/E5d)</span></div>
  <div class="tile"><b>+1.32°</b><span>compass bias self-recovered
   (true +1.50°)</span></div>
 </div>
</div>

<div class="ruler"></div>
<h2><span class="no">01</span>The instrument in one paragraph</h2>
<p>A camera with known attitude photographs the horizon. The skyline —
elevation angle versus azimuth — is extracted and matched against
skylines synthesized from a DEM over a grid of candidate positions
(native C ray-marcher, curvature and refraction corrected, ~4 ms per
skyline; heading search FFT-accelerated with a bit-identical robust
optimum). The best basin is a position fix with an anisotropic
covariance; four trust gates (basin margin, boundary, residual, relief)
make <em>inconclusive</em> a first-class outcome. Fixes fuse with dead
reckoning in a factor graph that also estimates the compass bias, and
the result leaves the box as NMEA a chartplotter reads like GPS.</p>

<div class="ruler"></div>
<h2><span class="no">02</span>Accuracy, by regime</h2>
<div class="scroll"><table>
<tr><th>Regime</th><th>Evidence</th><th>Error</th></tr>
<tr><td>Synthetic closed loop, 1 km box</td><td class="mute">E1, 40
 truths × configs</td><td class="n">7–14 m CEP50</td></tr>
<tr><td>100 km × 100 km box, sea-masked</td><td class="mute">E3,
 15/15 solves, margins ≥ 0.29</td><td class="n">15.8 m median</td></tr>
<tr><td>Photo-realistic composites, EXIF pipeline</td><td
 class="mute">E4c sea cases</td><td class="n">23–239 m</td></tr>
<tr><td>+ sea-horizon auto-levelling</td><td class="mute">E4g (no IMU
 at all)</td><td class="n">26–70 m, σ halved</td></tr>
<tr><td>Real photos, curated masks, attitude priors</td><td
 class="mute">E4f, 203 photos</td><td class="n">250 m median
 (accepted)</td></tr>
<tr><td>Real field photos, hand-digitized</td><td class="mute">E4h,
 telephoto frame</td><td class="n">250 m (wide frames
 1.5–2.3 km)</td></tr>
<tr><td>Fused passage, live iSAM2</td><td class="mute">E5b/E5c, 2 h
 simulated</td><td class="n">69 m mean · 2.8 m final</td></tr>
</table></div>
<p class="cap">Input angular fidelity is the whole budget: the same
solver spans 15 m to 2 km depending only on how well the skyline is
read.</p>

<div class="ruler"></div>
<h2><span class="no">03</span>Trust: the operating curve</h2>
<p>The basin margin — best-vs-second-basin cost separation — prices
every fix. With attitude priors the availability/integrity trade is
clean; attitude-free, no usable threshold exists. Independently
cross-validated: a parallel study with a different extractor and
solver chose the same 1.5 threshold and kept the same 14 photos.</p>
<img class="fig" alt="operating curve"
 src="data:image/png;base64,{b64('e4k_curve.png')}">
<div class="scroll"><table>
<tr><th>Margin</th><th>Accepted</th><th>Wrong (≥500 m)</th>
<th>&gt;1.5 km</th><th>Median err</th></tr>
{tier_rows()}
</table></div>
<p><span class="ok">Tier guidance:</span> ≥ 0.15 fused-only (graph
cross-checks) · ≥ 0.7 standalone · <b>≥ 1.5 act-on-it</b>.</p>
<p>Weather (E4n, 27 degraded runs): overcast → 9/9 honest
inconclusives (the DEM cannot explain a cloud base); haze benign to a
50 % wash; scattered cumulus marginal — its 2 escapes fall inside the
0.7 tier. <b>Clouds cost availability, not integrity.</b> Cross-DEM
(E4o): SRTM↔Copernicus GLO-30 skyline discrepancy 0.45 mrad median
(2–3 mrad with near terrain); solves survive the swap.</p>

<div class="ruler"></div>
<h2><span class="no">04</span>What did not work — kept on the record</h2>
<div class="panel neg">
<p><b>Coastline stadimetry (E4j).</b> Matching the waterline's
depression adds no range information below ~30 m observer height —
σ<sub>d</sub> ≈ (d²/z)·σ<sub>δ</sub> is kilometer-scale for
kilometer-scale shores. The water/land segmenter should not be built
for ranging.</p>
<p><b>Ensemble self-selection (E4l/E4m).</b> Across extraction
hypotheses, rms is <em>anti</em>-predictive and margin is a wash; even
a genuinely different detector family agrees mostly where everything
already agrees. Landscape ambiguity is common-mode — the ceiling for
single-photo trust is the operating curve above, not a better
front-end ensemble.</p>
<p><b>Pure heading-bias factors under iSAM2.</b> Batch-correct, but
incremental solving leaves the whole-chain-rotation mode frozen; the
fix was physical — each skyline fix's azimuth shift measures the bias
directly.</p>
</div>

<div class="ruler"></div>
<h2><span class="no">05</span>Field procedure</h2>
<div class="panel field">
<p>Keep originals (EXIF intact). Shoot a <b>telephoto pan</b> — several
narrow-FOV frames swept across the terrain — plus one wide frame for
azimuth coverage; narrow FOV multiplies angular precision ~8×. Stamp
attitude per frame (Theodolite-class app), but treat app pitch as
±1.5° until terrain-calibrated, and check its declination handling —
a 7° heading bias was caught in the field photos by the DEM itself.
Prefer open water in frame: the sea horizon auto-levels pitch/roll to
&lt;0.1° with no IMU. Then <code>skyfix.py IMG1 IMG2 … --center
LAT,LON --auto-level</code>, and read the tier before trusting the
dot.</p>
</div>

<div class="ruler"></div>
<h2><span class="no">06</span>Fusion, ported</h2>
<p>Fix + covariance enter GTSAM as a unary Pose2 factor; heading
factors run through a shared compass-bias variable, made observable by
each fix's azimuth shift. The C++ port in the gtsam fork replays the
exported sensor stream with no Python, DEM, or solver in the loop:</p>
<div class="scroll"><table>
<tr><th>Quantity</th><th>Python (wheel)</th><th>C++ (fork)</th></tr>
<tr><td>Mean error</td><td class="n">{e5d['mean_err_m']:.6f} m</td>
 <td class="n">31.394351 m</td></tr>
<tr><td>Final error</td><td class="n">{e5d['final_err_m']:.7f} m</td>
 <td class="n">2.8004408 m</td></tr>
<tr><td>Compass bias</td><td class="n">{e5d['bias_deg']:+.7f}°</td>
 <td class="n">+1.3157484°</td></tr>
</table></div>
<p class="cap">Agreement to ~10 significant digits — one estimator,
two implementations.</p>

<div class="ruler"></div>
<h2><span class="no">07</span>The human interface</h2>
<p>SkyFix Station (<code>station.py</code>): a browser HMI any phone
on the network can open — photo in, tiered fix on a self-contained
DEM-hillshade chart; the live passage with its σ and self-calibrating
compass; scene panoramas. NMEA 0183 on <code>tcp/10110</code> makes
the whole stack look like a GPS receiver to OpenCPN — quality drops to
6 (estimated) whenever a fix attempt is inconclusive.</p>
<div class="figrow">
<img class="fig" alt="fix tab"
 src="data:image/png;base64,{{IMG_FIX}}">
<img class="fig" alt="underway tab"
 src="data:image/png;base64,{{IMG_UW}}">
</div>
<p class="cap">Left: a photo sight solved to an ACT-ON-IT fix with its
3σ ellipse. Right: the passage underway — 8 fixes fused, σ 20 m,
compass bias recovered live.</p>

<footer>Repos: gyillikci/horizonator (study + experiments + station) ·
gyillikci/celestial-navigation (CH1 benchmark, bridge, parallel study)
· gyillikci/gtsam (C++ factors) — branch
<span class="mono">claude/horizonator-skyline-matching-s2l1t9</span>.
Full experiment log: <span class="mono">doc/skyline-matching-study.md
</span> §6, E0–E5d.</footer>
</div>
"""

# station screenshots live in the session scratchpad; fall back to empty
SCRATCH = ('/tmp/claude-0/-home-user/792503f9-74c5-5111-83ca-eeeda63e83'
           '8d/scratchpad')


def b64f(path):
    return (base64.b64encode(open(path, 'rb').read()).decode()
            if os.path.exists(path) else '')


html = HTML.replace('{IMG_FIX}', b64f(os.path.join(
    SCRATCH, 'station_fix_crop.png')))
html = html.replace('{IMG_UW}', b64f(os.path.join(
    SCRATCH, 'station_underway.png')))
dst = os.path.join(DOC, 'dossier.html')
open(dst, 'w').write(html)
print('wrote', dst, f'{os.path.getsize(dst)/1e6:.1f} MB')
