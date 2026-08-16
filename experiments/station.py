#!/usr/bin/env python3
"""SkyFix Station: the human interface over the skyline-navigation stack.

A single-file local web app (Flask, stdlib otherwise) that wraps the
tools built in this directory and serves a browser UI any phone or
laptop on the network can open — on a CM5 this IS the instrument face.

  FIX       drop/choose a photo, enter attitude (or trust EXIF /
            sea-horizon auto-levelling), solve; the fix is drawn on a
            DEM-hillshade basemap (self-contained: no tile servers) with
            its error ellipse and the TRUST TIER verdict — including a
            proud, explicit INCONCLUSIVE with the gates' reasons.
  UNDERWAY  the live SkyNav loop on the simulated E5 passage: track,
            current sigma, live compass-bias estimate, and the NMEA 0183
            feed also served on TCP :10110 for OpenCPN / chartplotters.
  PANORAMA  tap-a-point "what would I see from here" scene render.

Run:   python3 station.py [--port 8990] [--demo]
       (--demo pre-computes one fix and starts the passage replay so
        every tab has content immediately — used for screenshots)
"""

import argparse
import base64
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
PY = sys.executable
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))

app = Flask('skyfix-station')
STATE = dict(fix=None, passage=None, pano=None, nmea_tail=[])
NMEA_CLIENTS = []


# ---------------------------------------------------------------- basemap
def hillshade_png(lat, lon, km=6.0, px=640):
    """Self-contained basemap: DEM hillshade + sea fill, ENU-linear."""
    cm = S.CMarcher(DIR3, (lat - .2, lat + .2), (lon - .25, lon + .25))
    mlat, mlon = S.meters_per_degree(lat)
    half = km * 500.0
    n = np.linspace(half, -half, px)[:, None]
    e = np.linspace(-half, half, px)[None, :]
    la = lat + n / mlat
    lo = lon + e / mlon
    y = np.clip(((cm.lat_nw - la) / cm.dpp).astype(int), 0,
                cm.mosaic.shape[0] - 1)
    x = np.clip(((lo - cm.lon_nw) / cm.dpp).astype(int), 0,
                cm.mosaic.shape[1] - 1)
    h = cm.mosaic[y, x]
    gx = np.gradient(h, axis=1)
    gy = np.gradient(h, axis=0)
    shade = np.clip(0.72 + (gx - gy) * 0.012, 0.35, 1.0)
    img = np.zeros((px, px, 3))
    land = h > 0.5
    t = np.clip(h / 700.0, 0, 1)[..., None]
    land_c = (np.array([0.72, 0.70, 0.62]) * (1 - t)
              + np.array([0.55, 0.50, 0.44]) * t) * shade[..., None]
    sea_c = np.array([0.78, 0.85, 0.91])
    img[:] = np.where(land[..., None], land_c, sea_c[None, None])
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- fix
def tier(margin, status):
    if status != 'ok':
        return ('INCONCLUSIVE', '#d03b3b')
    if margin >= 1.5:
        return ('ACT-ON-IT (margin ≥1.5)', '#0ca30c')
    if margin >= 0.7:
        return ('STANDALONE (margin ≥0.7)', '#1baf7a')
    return ('FUSED-ONLY (margin ≥0.15)', '#fab219')


def run_fix(params):
    args = [PY, os.path.join(HERE, 'skyfix.py'), params['photo'],
            '--center', params['center'], '--box', str(params.get(
                'box', 5000)),
            '--dmin', str(params.get('dmin', 1000)),
            '--out', os.path.join(OUT, 'station_fix')]
    for k in ('fov', 'heading', 'pitch', 'roll', 'z'):
        if params.get(k) not in (None, ''):
            args += [f'--{k}', str(params[k])]
    if params.get('auto_level'):
        args += ['--auto-level']
    p = subprocess.run(args, capture_output=True, text=True, cwd=HERE)
    t = p.stdout
    j = json.loads(t[t.index('{'):t.rindex('}') + 1])
    lat_c, lon_c = (float(v) for v in params['center'].split(','))
    png = ''
    fp = os.path.join(OUT, 'station_fix.png')
    if os.path.exists(fp):
        png = base64.b64encode(open(fp, 'rb').read()).decode()
    tname, tcolor = tier(j['basin_margin'], j['status'])
    return dict(result=j, tier=tname, tier_color=tcolor,
                center=[lat_c, lon_c],
                basemap=hillshade_png(lat_c, lon_c), diag_png=png)


@app.route('/fix', methods=['POST'])
def fix_route():
    STATE['fix'] = run_fix(request.get_json())
    return jsonify(STATE['fix'])


# ---------------------------------------------------------------- underway
def nmea_broadcast(line):
    STATE['nmea_tail'] = (STATE['nmea_tail'] + [line])[-8:]
    dead = []
    for c in NMEA_CLIENTS:
        try:
            c.sendall((line + '\r\n').encode())
        except OSError:
            dead.append(c)
    for c in dead:
        NMEA_CLIENTS.remove(c)


def nmea_server(port=10110):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(4)
    while True:
        c, _ = srv.accept()
        NMEA_CLIENTS.append(c)


def passage_thread(leg_seconds=1.2):
    """Replay the E5 passage through the live SkyNav loop."""
    from skynav import SkyNav, AZ
    LAT0, LON0, Z = 36.95, 27.25, 5.0
    DT, N, SPEED = 300.0, 24, 3.0
    LOG_BIAS, CB = 1.03, np.radians(1.5)
    rng = np.random.default_rng(20260818)
    mlat, mlon = S.meters_per_degree(LAT0)
    cm = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))
    hdg_true = lambda k: np.radians(80.0 if k < N // 2 else 110.0)
    truth = [np.array([-5000.0, -2200.0])]
    for k in range(N):
        h = hdg_true(k)
        truth.append(truth[-1] + SPEED * DT
                     * np.array([np.sin(h), np.cos(h)]))
    nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
                 lon_range=(26.6, 27.9), start_pos=tuple(truth[0]),
                 start_heading=hdg_true(0),
                 start_sigma=(50.0, 50.0, np.radians(3)))
    ps = dict(lat0=LAT0, lon0=LON0, track=[], truth=[], fixes=[],
              sigma=0.0, bias=0.0, k=0, n=N, running=True,
              basemap=hillshade_png(LAT0, LON0, km=16.0))
    STATE['passage'] = ps
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = hdg_true(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        if (k + 1) % 3 == 0:
            el, _ = cm.skyline(LAT0 + truth[k + 1][1] / mlat,
                               LON0 + truth[k + 1][0] / mlon, Z, AZ)
            el = el + rng.normal(0, 1e-3, el.size)
            el = np.roll(el, int(round(np.degrees(CB) / 0.1)))
            fix, cov, margin, ok, _ = nav.take_fix(el)
            if ok:
                ps['fixes'].append([float(fix[0]), float(fix[1])])
        lat, lon, cov2 = nav.current()
        e = (lon - LON0) * mlon
        n2 = (lat - LAT0) * mlat
        ps['track'].append([float(e), float(n2)])
        ps['truth'].append([float(truth[k + 1][0]),
                            float(truth[k + 1][1])])
        ps['sigma'] = float(np.sqrt(np.trace(cov2)))
        ps['bias'] = nav.compass_bias_deg() or 0.0
        ps['k'] = k + 1
        tsim = (k + 1) * 300
        hms = f'{12 + tsim // 3600:02d}{tsim % 3600 // 60:02d}' \
              f'{tsim % 60:02d}.00'
        nmea_broadcast(nav.nmea_gga(hms))
        nmea_broadcast(nav.nmea_rmc(hms, '100826', 5.8,
                                    float(np.degrees(hdg_true(k)))))
        time.sleep(leg_seconds)
    ps['running'] = False


@app.route('/passage/start', methods=['POST'])
def passage_start():
    if not (STATE['passage'] and STATE['passage']['running']):
        threading.Thread(target=passage_thread, daemon=True).start()
    return jsonify(ok=True)


@app.route('/state')
def state():
    return jsonify(fix=STATE['fix'], passage=STATE['passage'],
                   nmea=STATE['nmea_tail'], pano=STATE['pano'])


# ---------------------------------------------------------------- panorama
@app.route('/pano', methods=['POST'])
def pano_route():
    p = request.get_json()
    dst = os.path.join(OUT, 'station_pano.png')
    subprocess.run(['xvfb-run', '-a', PY,
                    os.path.join(HERE, 'panorama.py'),
                    str(p['lat']), str(p['lon']), '-o', dst],
                   cwd=HERE, capture_output=True)
    STATE['pano'] = dict(
        lat=p['lat'], lon=p['lon'],
        png=base64.b64encode(open(dst, 'rb').read()).decode())
    return jsonify(STATE['pano'])


# ---------------------------------------------------------------- page
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>SkyFix Station</title><style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--blue:#2a78d6;
--orange:#eb6834;}
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#f9f9f7;
color:var(--ink)}
header{background:#0d366b;color:#fff;padding:10px 18px;display:flex;
align-items:baseline;gap:14px}
header h1{font-size:17px;margin:0;font-weight:600}
header span{opacity:.75;font-size:12px}
nav{display:flex;gap:4px;padding:8px 14px;background:var(--surface);
border-bottom:1px solid #e2e1dd}
nav button{border:1px solid #d5d4cf;background:#fff;border-radius:7px;
padding:7px 16px;font-size:13px;cursor:pointer}
nav button.on{background:var(--blue);color:#fff;border-color:var(--blue)}
main{padding:16px;max-width:1180px;margin:0 auto}
.card{background:var(--surface);border:1px solid #e2e1dd;border-radius:10px;
padding:14px;margin-bottom:14px}
.row{display:flex;gap:14px;flex-wrap:wrap}
.col{flex:1;min-width:330px}
label{display:block;font-size:11px;color:var(--ink2);margin:7px 0 2px}
input,select{width:95%;padding:6px;border:1px solid #d5d4cf;
border-radius:6px;font-size:13px}
button.go{margin-top:12px;background:var(--blue);color:#fff;border:0;
border-radius:7px;padding:9px 22px;font-size:14px;cursor:pointer}
.tier{display:inline-block;padding:5px 14px;border-radius:16px;color:#fff;
font-weight:600;font-size:13px}
.kv{font-size:12.5px;color:var(--ink2)} .kv b{color:var(--ink)}
canvas{border:1px solid #e2e1dd;border-radius:8px;max-width:100%}
.nmea{font:11.5px/1.5 ui-monospace,monospace;background:#0d366b;
color:#b7d3f6;border-radius:8px;padding:10px;white-space:pre;
overflow-x:auto}
img.diag{max-width:100%;border:1px solid #e2e1dd;border-radius:8px}
.stat{display:inline-block;margin-right:22px}
.stat b{font-size:21px}.stat span{font-size:11px;color:var(--ink2);
display:block}
h3{margin:4px 0 10px;font-size:14px}
</style></head><body>
<header><h1>SkyFix Station</h1><span>DEM-skyline navigation
&nbsp;·&nbsp; NMEA on tcp/10110</span></header>
<nav><button id=tb_fix onclick="tab('fix')">Fix</button>
<button id=tb_uw onclick="tab('uw')">Underway</button>
<button id=tb_pn onclick="tab('pn')">Panorama</button></nav>
<main>
<div id=tab_fix>
 <div class=row><div class=col style="max-width:340px"><div class=card>
  <h3>Photo sight</h3>
  <label>photo path</label><input id=f_photo value="out/synth/strait2.jpg">
  <label>box center lat,lon (DR position)</label>
  <input id=f_center value="36.9631,27.2371">
  <label>heading °T (blank = EXIF / full search)</label><input id=f_heading>
  <label>pitch ° / roll °</label>
  <div style="display:flex;gap:6px"><input id=f_pitch value="-0.5">
  <input id=f_roll value="0.5"></div>
  <label>FOV ° (blank = EXIF)</label><input id=f_fov>
  <label><input type=checkbox id=f_auto style="width:auto"> sea-horizon
  auto-level (±2 mrad window)</label>
  <button class=go onclick="doFix()">Solve fix</button>
 </div></div>
 <div class=col><div class=card><h3>Chart</h3>
  <div id=f_tier></div><div id=f_stats class=kv style="margin:8px 0"></div>
  <canvas id=f_map width=640 height=640></canvas></div></div>
 </div>
 <div class=card><h3>Diagnostics — extraction &amp; skyline match</h3>
 <img id=f_diag class=diag></div>
</div>
<div id=tab_uw style="display:none">
 <div class=card><h3>Live passage — iSAM2 fusion, compass bias estimated
 underway</h3>
 <div id=u_stats></div>
 <button class=go onclick="fetch('/passage/start',{method:'POST'})">
 Start demo passage</button></div>
 <div class=row>
 <div class=col><div class=card><canvas id=u_map width=640 height=640>
 </canvas></div></div>
 <div class=col><div class=card><h3>NMEA 0183 → chartplotter</h3>
 <div class=nmea id=u_nmea>(waiting for sentences…)</div>
 <p class=kv>Also served on <b>tcp/10110</b> — point OpenCPN at this
 host and the skyline-derived position displays like a GPS; quality
 drops to 6 (estimated) whenever a fix attempt is INCONCLUSIVE.</p>
 </div></div></div>
</div>
<div id=tab_pn style="display:none">
 <div class=card><h3>What would I see from here?</h3>
 <label>lat / lon</label>
 <div style="display:flex;gap:6px;max-width:340px">
 <input id=p_lat value="37.476847"><input id=p_lon value="27.414204">
 </div>
 <button class=go onclick="doPano()">Render panorama</button></div>
 <div class=card><img id=p_img class=diag></div>
</div>
</main><script>
let cur='fix';
function tab(t){cur=t;
 for(const [k,n] of [['fix','fix'],['uw','uw'],['pn','pn']]){
  document.getElementById('tab_'+k).style.display=k===t?'':'none';
  document.getElementById('tb_'+n).className=k===t?'on':'';}}
tab('fix');
function drawMap(cv,bm,km,items){const c=cv.getContext('2d');
 const img=new Image();img.onload=()=>{c.drawImage(img,0,0,cv.width,
 cv.height);const s=cv.width/(km*1000);const T=(e,n)=>[cv.width/2+e*s,
 cv.height/2-n*s];
 for(const it of items){
  if(it.t==='track'||it.t==='truth'){c.beginPath();
   it.pts.forEach((p,i)=>{const q=T(p[0]-it.o[0],p[1]-it.o[1]);
   i?c.lineTo(...q):c.moveTo(...q)});c.strokeStyle=it.c;
   c.lineWidth=it.w||2;c.setLineDash(it.d||[]);c.stroke();
   c.setLineDash([])}
  if(it.t==='pt'){const q=T(it.e,it.n);c.beginPath();
   c.arc(q[0],q[1],7,0,7);c.fillStyle=it.c;c.fill();
   c.lineWidth=2.5;c.strokeStyle='#fff';c.stroke();
   if(it.l){c.fillStyle='#0b0b0b';c.font='12px sans-serif';
   c.fillText(it.l,q[0]+10,q[1]-8)}}
  if(it.t==='ell'){const q=T(it.e,it.n);c.beginPath();
   c.ellipse(q[0],q[1],Math.max(it.se*s,4),Math.max(it.sn*s,4),0,0,7);
   c.strokeStyle=it.c;c.lineWidth=1.6;c.stroke()}}};
 img.src='data:image/png;base64,'+bm;}
async function doFix(){
 document.getElementById('f_tier').innerHTML='solving…';
 const b={photo:v('f_photo'),center:v('f_center'),heading:v('f_heading'),
  pitch:v('f_pitch'),roll:v('f_roll'),fov:v('f_fov'),
  auto_level:document.getElementById('f_auto').checked};
 const r=await(await fetch('/fix',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}
  )).json();renderFix(r);}
function v(id){return document.getElementById(id).value}
function renderFix(r){if(!r)return;const j=r.result;
 document.getElementById('f_tier').innerHTML=
  `<span class=tier style="background:${r.tier_color}">${r.tier}</span>`+
  (j.reasons.length?`<div class=kv style="margin-top:6px;color:#d03b3b">`+
   j.reasons.map(x=>'· '+x).join('<br>')+`</div>`:'');
 document.getElementById('f_stats').innerHTML=
  `fix <b>${j.lat.toFixed(5)}, ${j.lon.toFixed(5)}</b> · σ
  <b>${j.sigma_n_m.toFixed(0)}/${j.sigma_e_m.toFixed(0)} m</b> · margin
  <b>${j.basin_margin.toFixed(2)}</b> · rms
  <b>${j.rms_mrad.toFixed(1)} mrad</b> · relief
  <b>${j.relief_mrad.toFixed(1)} mrad</b>`;
 drawMap(document.getElementById('f_map'),r.basemap,6,[
  {t:'pt',e:0,n:0,c:'#52514e',l:'DR center'},
  {t:'pt',e:j.de_m,n:j.dn_m,c:'#eb6834',l:'FIX'},
  {t:'ell',e:j.de_m,n:j.dn_m,se:j.sigma_e_m*3,sn:j.sigma_n_m*3,
   c:'#eb6834'}]);
 if(r.diag_png)document.getElementById('f_diag').src=
  'data:image/png;base64,'+r.diag_png;}
async function doPano(){const r=await(await fetch('/pano',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(
 {lat:+v('p_lat'),lon:+v('p_lon')})})).json();
 document.getElementById('p_img').src='data:image/png;base64,'+r.png;}
async function poll(){const s=await(await fetch('/state')).json();
 if(s.fix&&!document.getElementById('f_diag').src)renderFix(s.fix);
 if(s.pano&&!document.getElementById('p_img').src)
  document.getElementById('p_img').src='data:image/png;base64,'+s.pano.png;
 if(s.passage){const p=s.passage;
  document.getElementById('u_stats').innerHTML=
   `<span class=stat><b>${p.k}/${p.n}</b><span>legs (5 min each)</span>
   </span><span class=stat><b>${p.sigma.toFixed(0)} m</b><span>current σ
   </span></span><span class=stat><b>${p.bias>=0?'+':''}${p.bias.toFixed(2)}°
   </b><span>compass bias (true +1.50°)</span></span>
   <span class=stat><b>${p.fixes.length}</b><span>skyline fixes fused
   </span></span>`;
  drawMap(document.getElementById('u_map'),p.basemap,16,[
   {t:'truth',pts:p.truth,o:[0,0],c:'#52514e',w:1.5,d:[5,4]},
   {t:'track',pts:p.track,o:[0,0],c:'#2a78d6',w:2.5},
   ...p.fixes.map(f=>({t:'pt',e:f[0],n:f[1],c:'#eb6834'}))]);
  document.getElementById('u_nmea').textContent=
   (s.nmea||[]).join('\\n')||'(waiting…)';}
 setTimeout(poll,900);}poll();
</script></body></html>"""


@app.route('/')
def index():
    return PAGE


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8990)
    ap.add_argument('--demo', action='store_true',
                    help='pre-run one fix and start the passage replay')
    a = ap.parse_args()
    threading.Thread(target=nmea_server, daemon=True).start()
    if a.demo:
        def prefill():
            STATE['fix'] = run_fix(dict(
                photo='out/synth/strait2.jpg', center='36.9631,27.2371',
                pitch=-0.5, roll=0.5))
            passage_thread(leg_seconds=0.7)
        threading.Thread(target=prefill, daemon=True).start()
    print(f'SkyFix Station on http://0.0.0.0:{a.port}  '
          f'(NMEA on tcp/10110)')
    app.run(host='0.0.0.0', port=a.port, threaded=True)
