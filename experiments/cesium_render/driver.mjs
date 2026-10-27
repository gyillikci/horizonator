// Render poses over Google Photorealistic 3D Tiles in headless Chrome.
//
//   CESIUM_ION_TOKEN=... node driver.mjs POSES.json OUTDIR [--gl swiftshader|egl|vulkan] [--workers N]
//
// POSES.json: [{tag, lat, lon, h_ellipsoid, heading_deg, pitch_deg, roll_deg,
//               hfov_deg, geoid_n, width, height, sse?}, ...]
// Writes OUTDIR/<tag>_photo.png, <tag>_seg.png and <tag>.json (the pose, load
// status, level of detail, timing). Poses already rendered are skipped, so a
// run can resume. The token is read from the environment and handed to the
// page in memory.
//
// --workers N browsers (default 3) take poses from one queue. A worker is one
// busy JavaScript thread (the tileset is traversed every frame), so workers
// scale with cores until the GPU's memory runs out: a heavy view holds 1.1 GB
// of tiles, and a 1600x3479 frame's buffers come on top. On the 8 GB GPU the
// 29 sea-level poses took 283 s with three workers; with four, 251 s once
// and 485 s the next time, when four portrait frames met at 7.7 GB and took
// 2-4 min each. Three workers at SSE 4, 2-2.5 GB of tiles each, drew under
// one frame a second.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const [posesPath, outDir] = args;
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const gl = opt('--gl', 'swiftshader');
const nWorkers = Math.max(1, +opt('--workers', 3));
const token = process.env.CESIUM_ION_TOKEN;
if (!posesPath || !outDir || !token) {
  console.error('usage: CESIUM_ION_TOKEN=... node driver.mjs POSES.json OUTDIR [--gl swiftshader|egl|vulkan] [--workers N]');
  process.exit(2);
}
const poses = JSON.parse(readFileSync(posesPath, 'utf8'));
mkdirSync(outDir, { recursive: true });

const page = readFileSync(join(HERE, 'index.html'));
const server = createServer((req, res) => {
  if (req.url === '/' || req.url.startsWith('/index.html')) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(page);
  } else { res.writeHead(404); res.end(); }
}).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const url = `http://127.0.0.1:${server.address().port}/`;

const W = Math.max(...poses.map((p) => p.width)), H = Math.max(...poses.map((p) => p.height));
const glFlags = { swiftshader: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
                  egl: ['--use-gl=angle', '--use-angle=gl-egl'],
                  vulkan: ['--use-angle=vulkan', '--enable-features=Vulkan'] }[gl];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// every browser this run started: an interrupted run (timeout, Ctrl-C) must
// not leave them rendering, and their throwaway profiles hold tile caches
const browsers = [];
function cleanup() {
  for (const b of browsers.splice(0)) {
    try { b.chrome.kill('SIGKILL'); } catch (e) {}
    try { rmSync(b.profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch (e) {}
  }
}
process.on('exit', cleanup);
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => process.exit(130));

async function openWorker() {
  const profile = mkdtempSync(join(tmpdir(), 'cesr-'));
  // port 0: Chrome picks a free one and writes it to the profile
  const chrome = spawn('google-chrome', ['--headless=new', '--no-first-run', '--remote-debugging-port=0',
    '--user-data-dir=' + profile, '--ignore-gpu-blocklist', '--hide-scrollbars',
    `--window-size=${W},${H}`, ...glFlags, 'about:blank'], { stdio: 'ignore' });
  browsers.push({ chrome, profile });
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await sleep(200);
    try {
      const port = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
      target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page');
    } catch (e) {}
  }
  if (!target) throw new Error('Chrome did not start');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r, { once: true }));
  let id = 0; const pending = new Map(); const logs = [];
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') logs.push('ERR ' + m.params.args.map((a) => a.value || a.description).join(' '));
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  async function evaluate(expr) {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  }
  await send('Page.enable'); await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });
  for (let i = 0; i < 60; i++) { await sleep(500); if (await evaluate('typeof window.init === "function"')) break; }
  const info = await evaluate(`init(${JSON.stringify({ token, width: W, height: H })})`);
  return { evaluate, info, logs };
}

if (args.includes('--water')) {
  // the nearest point within R m where the tiles' surface is at sea level (or
  // there is no tile: open sea), on rings 10 m apart, 24 azimuths each
  const { evaluate } = await openWorker();
  const R = +opt('--water');
  for (const [k, p] of poses.entries()) {
    const mlat = 111132.0, mlon = 111320.0 * Math.cos(p.lat * Math.PI / 180);
    let found = null;
    for (let r = 0; r <= R && !found; r += 10) {
      const pts = [];
      for (let a = 0; a < 360; a += (r ? 15 : 360)) {
        pts.push({ lat: p.lat + r * Math.cos(a * Math.PI / 180) / mlat, lon: p.lon + r * Math.sin(a * Math.PI / 180) / mlon, r, az: a });
      }
      const h = await evaluate(`sampleHeights(${JSON.stringify(pts)})`);
      const wet = pts.map((q, i) => ({ ...q, h_msl: h[i] === null ? null : h[i] - p.geoid_n }))
        .filter((q) => q.h_msl === null || q.h_msl <= 0.5);
      if (wet.length) {
        // of the wet points on this ring, the one nearest the view direction
        const d = (q) => Math.abs(((q.az - p.heading_deg + 540) % 360) - 180);
        found = wet.sort((x, y) => d(x) - d(y))[0];
      }
    }
    writeFileSync(join(outDir, p.tag + '_water.json'), JSON.stringify({ pose: p, water: found, search_m: R }, null, 1));
    console.log(`${k + 1}/${poses.length} ${p.tag} ` + (found ? `water at ${found.r} m, az ${found.az}` : `no water within ${R} m`));
  }
  server.close();
  process.exit(0);
}

if (args.includes('--sample')) {
  // the tiles' surface height at the camera and on rings around it: where is
  // the ground in this model, and is the camera above it?
  const { evaluate } = await openWorker();
  for (const [k, p] of poses.entries()) {
    const pts = [{ lat: p.lat, lon: p.lon, r: 0, az: 0 }];
    const mlat = 111132.0, mlon = 111320.0 * Math.cos(p.lat * Math.PI / 180);
    for (const r of [30, 100, 300]) for (let a = 0; a < 360; a += 45) {
      pts.push({ lat: p.lat + r * Math.cos(a * Math.PI / 180) / mlat, lon: p.lon + r * Math.sin(a * Math.PI / 180) / mlon, r, az: a });
    }
    let h;
    try { h = await evaluate(`sampleHeights(${JSON.stringify(pts)})`); }
    catch (e) { console.log(`${p.tag} FAILED ${e.message.split('\n')[0]}`); continue; }
    const res = pts.map((q, i) => ({ ...q, h_ellipsoid: h[i], h_msl: h[i] === null ? null : h[i] - p.geoid_n }));
    writeFileSync(join(outDir, p.tag + '_ground.json'), JSON.stringify({ pose: p, points: res }, null, 1));
    const c = res[0];
    console.log(`${k + 1}/${poses.length} ${p.tag} ground at camera ${c.h_msl === null ? 'none' : c.h_msl.toFixed(1) + ' m MSL'}, camera ${p.z_msl} m MSL`);
  }
  server.close();
  process.exit(0);
}

const todo = poses.filter((p) => !existsSync(join(outDir, p.tag + '.json')));
console.log(`${poses.length - todo.length} of ${poses.length} poses already rendered, ${todo.length} to go`);
const workers = await Promise.all(Array.from({ length: Math.min(nWorkers, todo.length) }, openWorker));
if (workers.length) console.log(`${workers.length} workers ready, Cesium ${workers[0].info.cesium}, gl ${gl}`);
const t0 = Date.now();
// frames over 4 Mpx (the 1600x3479 portraits) render at most two at a time:
// their buffers on top of heavy views' tiles are what filled the GPU when
// four met
const BIG = 4e6, BIG_AT_ONCE = 2;
const pending = [...todo], tries = new Map();
let bigBusy = 0, done = 0;
async function renderOne(wk, wi, p) {
  const base = join(outDir, p.tag);
  const n = (tries.get(p.tag) || 0) + 1; tries.set(p.tag, n);
  const full = (b) => b[0] === p.width && b[1] === p.height;
  const tp = Date.now();
  let r;
  try {
    const buf = await wk.evaluate(`sizeCanvas(${p.width}, ${p.height})`);
    if (full(buf)) r = await wk.evaluate(`renderPose(${JSON.stringify(p)})`);
    const got = r ? r.buffer : buf;
    if (!full(got)) {
      // WebGL fell back to a smaller drawing buffer: short of GPU memory
      if (n >= 3) throw new Error(`drawing buffer ${got.join('x')}, not ${p.width}x${p.height}, after ${n} tries`);
      console.log(`[w${wi}] ${p.tag} got a ${got.join('x')} drawing buffer, will retry`);
      await sleep(5000);
      pending.push(p);
      return;
    }
  } catch (e) { console.log(`${++done}/${todo.length} [w${wi}] ${p.tag} FAILED ${e.message.split('\n')[0]}`); return; }
  for (const kind of ['photo', 'seg']) writeFileSync(`${base}_${kind}.png`, Buffer.from(r[kind].split(',')[1], 'base64'));
  const meta = { pose: p, camera_used: r.cam, loaded: r.loaded, settle_s: r.settle_s, wall_s: (Date.now() - tp) / 1000,
                 canvas: r.canvas, buffer: r.buffer, lod: r.lod, attempts: n,
                 source: 'Google Photorealistic 3D Tiles via Cesium ion asset 2275207 (attribution: Google)',
                 cesium: wk.info.cesium, gl };
  writeFileSync(base + '.json', JSON.stringify(meta, null, 1));
  console.log(`${++done}/${todo.length} [w${wi}] ${p.tag} loaded=${r.loaded} settle ${r.settle_s.toFixed(1)} s, ` +
              `${r.lod.tiles} tiles ${r.lod.tile_mb.toFixed(0)} MB sse ${r.lod.sse_used}, buffer ${r.buffer}`);
}
async function work(wk, wi) {
  while (pending.length) {
    const i = pending.findIndex((p) => bigBusy < BIG_AT_ONCE || p.width * p.height <= BIG);
    if (i < 0) { await sleep(500); continue; }
    const p = pending.splice(i, 1)[0];
    const big = p.width * p.height > BIG;
    if (big) bigBusy++;
    try { await renderOne(wk, wi, p); } finally { if (big) bigBusy--; }
  }
}
await Promise.all(workers.map(work));
if (todo.length) console.log(`rendered ${todo.length} poses in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
for (const [wi, wk] of workers.entries()) if (wk.logs.length) console.log(`page errors, worker ${wi}:\n  ` + wk.logs.slice(0, 10).join('\n  '));
server.close();
process.exit(0);
