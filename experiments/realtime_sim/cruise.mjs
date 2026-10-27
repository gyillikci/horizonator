// Serve cruise.html and open it in Chrome with the Cesium ion token injected in memory.
//   CESIUM_ION_TOKEN=... node cruise.mjs                  interactive window (drive with the keys)
//   CESIUM_ION_TOKEN=... node cruise.mjs --bench 120 --speed 10 [--sse 16] [--size 1280x720]
//     headless: sail a scripted route for N seconds, print frame statistics every 10 s
//   ... --route routes/NAME.json [--look]   sail a planned route; --look turns the camera as planned
//   ... --live URL --pf [--look-live]        live_solver.py --pf [--look] tracks; --look-live turns where it says
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const a = process.argv.slice(2);
const opt = (k, d) => (a.includes(k) ? a[a.indexOf(k) + 1] : d);
const token = process.env.CESIUM_ION_TOKEN;
if (!token) { console.error('set CESIUM_ION_TOKEN'); process.exit(2); }
const bench = a.includes('--bench') ? +opt('--bench') : (a.includes('--route') ? 3600 : 0);
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const start = { token, lat: +opt('--lat', 37.01999), lon: +opt('--lon', 27.34782), n: +opt('--n', 33.424),
                heading: +opt('--heading', 110), h: +opt('--h', 2.5), sse: +opt('--sse', 16), fov: +opt('--fov', 70) };
const routeFile = opt('--route', null);
let prefix = 'cruise_';
if (routeFile) {
  const r = JSON.parse(readFileSync(routeFile, 'utf8'));
  Object.assign(start, { lat: r.lat, lon: r.lon, heading: r.heading, n: r.n, route: r.waypoints });
  if (a.includes('--look')) start.look = r.look;      // the camera yaws of look_planner.py
  prefix = 'cruise_' + r.name + '_';
}
if (bench || routeFile) start.bench = { speed: +opt('--speed', 10), weave: +opt('--weave', 4) };
if (a.includes('--wobble')) start.wobble = { pitch: +opt('--wobble', 1), roll: +opt('--wobble', 1) * 2 };
const recDir = opt('--record', null), every = +opt('--every', 15), minLoaded = +opt('--wait-loaded', 3);

const page = readFileSync(join(HERE, 'cruise.html'));
const server = createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html' }); r.end(page); }).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const profile = mkdtempSync(join(tmpdir(), 'cruise-'));
const flags = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--no-first-run', '--remote-debugging-port=0',
               '--user-data-dir=' + profile, `--window-size=${W},${H}`];
const chrome = spawn('google-chrome', bench ? ['--headless=new', ...flags, 'about:blank'] : [...flags, 'about:blank'],
                     { stdio: 'ignore', env: { ...process.env, __EGL_VENDOR_LIBRARY_FILENAMES: '/usr/share/glvnd/egl_vendor.d/10_nvidia.json' } });
const done = (c) => { try { chrome.kill('SIGKILL'); } catch (e) {} try { rmSync(profile, { recursive: true, force: true }); } catch (e) {} process.exit(c); };
process.on('SIGINT', () => done(130)); process.on('SIGTERM', () => done(143));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 100 && !target; i++) {
  await sleep(200);
  try { const port = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
        target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch (e) {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let id = 0; const pend = new Map();
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (e) => { const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 300)); return r.result?.result?.value; };
await send('Page.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
for (let i = 0; i < 60; i++) { await sleep(500); if (await ev('typeof window.start === "function"')) break; }
console.log('page ready', JSON.stringify(await ev(`start(${JSON.stringify(start)})`)));
// ---- --live URL: every N s send the frame to live_solver.py with what a boat would know (a
// compass, an IMU, a dead-reckoned position from the last accepted fix) and draw the answer
const live = opt('--live', null), liveEvery = +opt('--live-every', 10), usePF = a.includes('--pf'), lookLive = a.includes('--look-live');
const session = 'cruise_' + Date.now();
let lastShot = null;                // the pose at the last frame sent (for the dead-reckoned leg)
// the compass, as a real one errs: a deviation drifting 0.1 deg a frame plus 2 deg of noise;
// the dead-reckoned legs come off the same compass
let cbias = null;
const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
const nav = { fix: null, fixTruth: null };
let nLive = 0, inflight = 0;
async function liveShot() {
  if (inflight >= (usePF ? 1 : 2)) return;        // the filter's steps go in order
  const g = await ev('grab()'), p = g.pose;
  const tag = prefix + 'live_' + String(nLive++).padStart(3, '0');
  const mlat = 111132, mlon = 111320 * Math.cos(p.lat * Math.PI / 180);
  if (cbias === null) cbias = gauss() * 5;
  cbias += gauss() * 0.1;
  const cerr = cbias + gauss() * 2;
  let prior;
  if (nav.fix) {      // the log and compass since the last fix: 2 % speed error, a 5 deg heading error
    const dn = (p.lat - nav.fixTruth.lat) * mlat, de = (p.lon - nav.fixTruth.lon) * mlon;
    const a = gauss() * 5 * Math.PI / 180, sc = 1 + gauss() * 0.02;
    prior = [nav.fix.lat + sc * (dn * Math.cos(a) - de * Math.sin(a)) / mlat, nav.fix.lon + sc * (dn * Math.sin(a) + de * Math.cos(a)) / mlon];
  } else {            // no fix yet: a prior 1 km off in a random direction
    const th = Math.random() * 2 * Math.PI;
    prior = [p.lat + 1000 * Math.cos(th) / mlat, p.lon + 1000 * Math.sin(th) / mlon];
  }
  const q = { tag, png: g.png, prior, compass_deg: (p.heading_deg + cerr + 360) % 360, compass_err_deg: cerr,
              imu: [p.pitch_deg + gauss() * 0.3, p.roll_deg + gauss() * 0.3], z_msl: p.z_msl, hfov_deg: p.hfov_deg,
              truth: { lat: p.lat, lon: p.lon }, W: p.width, H: p.height, session, yaw_deg: p.yaw_deg };
  if (usePF && lastShot) {          // the leg since the last frame, as the log and compass give it
    const dn = (p.lat - lastShot.lat) * mlat, de = (p.lon - lastShot.lon) * mlon, L = Math.hypot(dn, de);
    const ae = cerr * Math.PI / 180, sc = 1 + gauss() * 0.02;
    q.dr = [sc * (dn * Math.cos(ae) - de * Math.sin(ae)), sc * (dn * Math.sin(ae) + de * Math.cos(ae)),
            Math.sqrt((0.02 * L) ** 2 + (L * 2 * Math.PI / 180) ** 2 + 400)];
  }
  lastShot = { lat: p.lat, lon: p.lon };
  inflight++;
  fetch(live + '/solve', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(q) })
    .then((r) => r.json()).then(async (f) => {
      inflight--;
      if (f.status === 'ok') { nav.fix = { lat: f.lat, lon: f.lon }; nav.fixTruth = { lat: p.lat, lon: p.lon }; }
      if (lookLive && f.look_yaw !== undefined) await ev(`setLook(${f.look_yaw})`);   // the solver says where to look next
      await ev(`addFix(${JSON.stringify({ ...f, truthAt: { lat: p.lat, lon: p.lon } })})`);
      console.log(`${tag}  ${f.status.padEnd(12)}  ${f.err_m !== undefined ? f.err_m.toFixed(0).padStart(5) + ' m' : '      -'}  (${(f.seg_s + f.solve_s).toFixed(1)} s)` +
                  (f.pf ? `  sigma ${f.pf.sigma_major.toFixed(0)} m  bias ${f.pf.compass_bias_deg.toFixed(1)} (true ${cerr.toFixed(1)})  ` +
                          `${f.update ? 'update' : 'predict only: ' + f.why}` : ''));
    }).catch((e) => { inflight--; console.log(tag, 'live solve failed:', e.message); });
}
const t0 = Date.now();
let nextLive = live ? 5 : Infinity;
if (!bench) {
  console.log('drive in the Chrome window; Ctrl-C here to quit');
  for (;;) { await sleep(500); if (live && (Date.now() - t0) / 1000 >= nextLive) { await liveShot(); nextLive += liveEvery; } }
}
let nRec = 0;
const { mkdirSync, writeFileSync } = await import('node:fs');
if (recDir) mkdirSync(recDir, { recursive: true });
async function record() {
  // wait (up to minLoaded s) for the tile queue to drain, then take the frame on screen
  for (let i = 0; i < minLoaded * 5; i++) { if ((await ev('grab().pose.pending')) === 0) break; await sleep(200); }
  const g = await ev('grab()');
  const tag = prefix + String(nRec++).padStart(3, '0');
  writeFileSync(join(recDir, tag + '_photo.png'), Buffer.from(g.png.split(',')[1], 'base64'));
  writeFileSync(join(recDir, tag + '.json'), JSON.stringify({ tag, t_s: (Date.now() - t0) / 1000, sse: start.sse, ...g.pose,
    source: 'Google Photorealistic 3D Tiles via Cesium ion asset 2275207 (attribution: Google)' }, null, 1));
}
let nextRec = every;
while ((Date.now() - t0) / 1000 < bench) {
  if (live && (Date.now() - t0) / 1000 >= nextLive) { await liveShot(); nextLive += liveEvery; }
  if (routeFile && await ev('isDone()')) { console.log('route done after', ((Date.now() - t0) / 1000).toFixed(0), 's,', nRec, 'frames'); break; }
  if (recDir || live) {
    await sleep(live && !recDir ? 500 : 1000);
    if (recDir && (Date.now() - t0) / 1000 >= nextRec) { await record(); nextRec += every; }
    continue;
  }
  await sleep(10000);
  const s = await ev('take()');
  console.log(`t ${((Date.now() - t0) / 1000).toFixed(0).padStart(4)} s  ${(1000 / s.mean_ms).toFixed(1).padStart(5)} fps  frame ms: p50 ${s.p50.toFixed(1)} p95 ${s.p95.toFixed(1)} p99 ${s.p99.toFixed(1)} max ${s.max.toFixed(0)}  ` +
              `sailed ${s.dist_m.toFixed(0)} m  tiles ${s.tiles}  pending ${s.pending}  ${s.mem_mb.toFixed(0)} MB  ${s.loaded ? 'loaded' : 'streaming'}`);
}
if (live) { for (let i = 0; i < 60 && inflight > 0; i++) await sleep(1000); }
if (a.includes('--shot')) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  (await import('node:fs')).writeFileSync(opt('--shot'), Buffer.from(r.result.data, 'base64'));
}
done(0);
