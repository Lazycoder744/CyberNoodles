/* worker_sim.mjs — run worker.js in Node with a shim for the worker globals.
 *
 * This is the closest thing to a browser smoke test available here.  Unlike
 * web/test/parity.mjs (pure JS stages) it exercises the *wiring*: the order
 * importScripts() loads config.js / onnxruntime-web / lib/*, the postMessage
 * protocol, fetching the model over HTTP from the page's own origin, the real
 * onnxruntime-web session, and the ArrayBuffer transfer of the finished BSOR.
 *
 * It loads worker.js and the lib/ files in the *main* realm with
 * vm.runInThisContext(), so the typed arrays the lib code makes and the ones
 * onnxruntime-web expects are from the same realm.  (A vm.createContext()
 * sandbox would be more faithful but breaks: ORT's browser build dynamic-imports
 * its wasm glue, which a contextified vm cannot do.)
 *
 * The rollout is clamped (PROBE_SECONDS) so the smoke test stays quick; the
 * unclamped, full-length rollout is verified by the ORT end-to-end harness.
 *
 *   node web/test/worker_sim.mjs
 */
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import http from 'http';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const REPO = path.resolve(HERE, '..', '..');
const WEB = path.join(REPO, 'web');
const ORT_DIST = process.env.ORT_DIST || '/tmp/opencode/orttest/node_modules/onnxruntime-web/dist';
const PROBE_SECONDS = Number(process.env.PROBE_SECONDS || 30);

// ---------------------------------------------------------------------------
// a static server for web/ (so the worker's fetch() has a real origin)
// ---------------------------------------------------------------------------
const MIME = { '.js': 'text/javascript', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  const full = path.join(WEB, rel);
  if (!full.startsWith(WEB) || !fs.existsSync(full) || !fs.statSync(full).isFile()) {
    res.writeHead(404); res.end('not found'); return;
  }
  const buf = fs.readFileSync(full);
  res.writeHead(200, { 'Content-Type': MIME[path.extname(full)] || 'application/octet-stream', 'Content-Length': buf.length });
  res.end(buf);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const PORT = server.address().port;
const ORIGIN = `http://127.0.0.1:${PORT}`;
console.log(`static origin ${ORIGIN}`);

// ---------------------------------------------------------------------------
// worker global shim
// ---------------------------------------------------------------------------
const scriptOrder = [];
globalThis.self = globalThis;

let postToHost = () => {};
globalThis.postMessage = (m) => postToHost(m);

// onnxruntime-web is imported here (main realm, Node build); the worker's
// importScripts() call for it is satisfied by this instance.
const ortEntry = process.env.ORT_MODULE || path.join(ORT_DIST, 'ort.node.min.mjs');
const ort = await import(ortEntry);
globalThis.ort = ort;

globalThis.importScripts = function (...urls) {
  for (const u of urls) {
    scriptOrder.push(u);
    if (/^https?:/.test(u)) {
      if (!u.includes('onnxruntime-web')) throw new Error('unexpected remote script ' + u);
      globalThis.ort = ort;  // already imported above
      continue;
    }
    const file = path.join(WEB, u);
    vm.runInThisContext(fs.readFileSync(file, 'utf8'), { filename: file });
    // In the browser CN.config.ortBase is the pinned jsDelivr dist URL; in Node
    // the wasm backend has to load from disk, so pin it to the npm copy.
    if (file.endsWith('config.js')) CN.config.ortBase = ORT_DIST + '/';
  }
};

vm.runInThisContext(fs.readFileSync(path.join(WEB, 'worker.js'), 'utf8'), { filename: 'worker.js' });

// The worker assigned self.onmessage; drive it from here.
const onmessage = globalThis.onmessage;

// ---------------------------------------------------------------------------
// protocol helpers
// ---------------------------------------------------------------------------
let nextId = 1;
const pending = new Map();
postToHost = (m) => {
  if (m.type === 'progress') return;
  const p = pending.get(m.id);
  if (p) { pending.delete(m.id); p(m); }
};
const call = (msg) => new Promise((resolve) => {
  const id = nextId++;
  pending.set(id, resolve);
  onmessage({ data: Object.assign({ id }, msg) });
});

// ---------------------------------------------------------------------------
console.log('--- init ---');
const ready = await call({ type: 'init', modelUrl: [
  `${ORIGIN}/models/cybernoodles_v2_fp32_part0.onnx`,
  `${ORIGIN}/models/cybernoodles_v2_fp32_part1.onnx`,
] });
if (ready.type === 'error') { console.log('FAIL init: ' + ready.message); process.exit(1); }
console.log(`ready: ${(ready.modelBytes / 1048576).toFixed(1)} MiB inputs=[${ready.inputs}]`);

// guard the ordering bug: config.js must define CN.config.ortScript *before*
// the runtime script is requested.
const iCfg = scriptOrder.findIndex((u) => u.endsWith('lib/config.js'));
const iOrt = scriptOrder.findIndex((u) => u.includes('onnxruntime-web'));
const orderOk = iCfg >= 0 && iOrt > iCfg;
console.log(`importScripts order: config.js@${iCfg} runtime@${iOrt} -> ${orderOk ? 'OK' : 'FAIL'}`);
console.log(`  ${scriptOrder.join('\n  ')}`);
if (!orderOk) process.exit(1);

// clamp the rollout for the smoke test (the unclamped path is covered by the
// ORT end-to-end harness)
const realRollout = CN.model.rollout;
CN.model.rollout = (o) => realRollout(Object.assign({}, o, { maxSeconds: PROBE_SECONDS }));

console.log('--- lookup ---');
const map = await call({ type: 'lookup', bsr: '1a8d4' });
if (map.type === 'error') { console.log('FAIL lookup: ' + map.message); process.exit(1); }
console.log(`map: ${map.meta.songName} — ${map.meta.mapper}; ${map.versions.length} version(s)`);
for (const d of map.versions[0].diffs) console.log(`  ${d.difficulty}: ${d.notes} notes, njs ${d.njs}`);

console.log('--- run ---');
const v = map.versions[0];
const t0 = Date.now();
const res = await call({
  type: 'run', bsr: '1a8d4', hash: v.hash, downloadURL: v.downloadURL,
  apiJson: map.api, difficulty: 'ExpertPlus',
});
if (res.type === 'error') { console.log('FAIL run: ' + res.message + '\n' + (res.stack || '')); process.exit(1); }
console.log(`run in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
console.log(`  accuracy ${res.accuracy.pct.toFixed(4)}% (${res.accuracy.score}/${res.accuracy.max}) good=${res.accuracy.good} miss=${res.accuracy.miss}`);
console.log(`  ${res.frames} frames, ${res.events} events, ${res.walls} walls, ${(res.meta.bytes / 1048576).toFixed(2)} MiB`);
console.log(`  meta ${JSON.stringify(res.meta)}`);

// ---------------------------------------------------------------------------
// decode the transferred BSOR and score it independently
// ---------------------------------------------------------------------------
const { readNotes, accuracyFromNotes } = await import('./bsorread.mjs');
const bytes = new Uint8Array(res.bsor);
console.log(`  bsor magic 0x${new DataView(res.bsor).getInt32(0, true).toString(16)} (expect 0x442d3d69)`);
const notes = readNotes(bytes);
const acc = accuracyFromNotes(notes, CN.avscore);
console.log(`  decoded from bytes: ${acc.pct.toFixed(4)}% (${acc.score}/${acc.max}) good=${acc.good} miss=${acc.miss}`);
const same = Math.abs(acc.pct - res.accuracy.pct) < 1e-9;
console.log(same ? '  OK: reported score == score decoded from the written file'
  : '  FAIL: reported score != score decoded from the written file');

// ---------------------------------------------------------------------------
// cross-check the truncated-map score against the Python/JS parity reference
// (js_ref_dump.py used MAXS=30.0, the same horizon as PROBE_SECONDS above)
// ---------------------------------------------------------------------------
const ref = JSON.parse(fs.readFileSync('/tmp/opencode/jsref/events.json', 'utf8'));
const refAcc = CN.avscore.accuracy(ref.events, ref.walls);
const refMatch = Math.abs(refAcc.pct - acc.pct) < 1e-9;
console.log(`  reference (Python, same horizon): ${refAcc.pct.toFixed(4)}% -> ${refMatch ? 'OK' : 'DIFFERS'}`);
if (refAcc.score !== acc.score) {
  console.log(`  (ref ${refAcc.score} vs worker ${acc.score}; map hash/MAPDIR must match js_ref_dump.py)`);
}

fs.writeFileSync('/tmp/opencode/worker_sim.bsor', bytes);
console.log('  wrote /tmp/opencode/worker_sim.bsor');

server.close();
process.exit(same && orderOk ? 0 : 1);
