/* ort_e2e.mjs — real onnxruntime-web end-to-end check of the browser pipeline.
 *
 * web/test/parity.mjs checks every JS stage against Python reference *data*, but
 * deliberately stops short of inference.  This script closes that gap: it loads
 * the same lib/* the Web Worker loads, creates a real onnxruntime-web session
 * from the exported ONNX, runs the two-pass rollout, reconstructs/scores the
 * cuts, and — optionally — compares the result with a .bsor written by the
 * Python pipeline.
 *
 * onnxruntime-web is not vendored here (it is ~99 MB and the page loads it from
 * a pinned jsDelivr URL), so point ORT_MODULE at an installed copy:
 *
 *   npm i onnxruntime-web@1.22.0
 *   ORT_MODULE=$PWD/node_modules/onnxruntime-web/dist/ort.node.min.mjs \
 *     node web/test/ort_e2e.mjs \
 *       --map maps_cache/b5a1424cad898181d429cb231b83296a3e67d58f \
 *       --diff ExpertPlus \
 *       --shipped replays/41a23_v2b_step305000.bsor
 *
 * Options:
 *   --map <dir>        map directory (info.dat + difficulty files)   [required]
 *   --diff <name>      difficulty                                     [ExpertPlus]
 *   --model <file>     ONNX checkpoint  [replays/cybernoodles_v2b_best.onnx]
 *   --shipped <file>   .bsor to compare against (decoded, not byte-compared)
 *   --seconds <n>      clamp the rollout to n seconds   (default 30; 0 = full)
 *   --full             shorthand for --seconds 0
 */
import fs from 'fs';
import path from 'path';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const REPO = path.resolve(HERE, '..', '..');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const mapDir = arg('map', null);
const diff = arg('diff', 'ExpertPlus');
const modelPath = arg('model', path.join(REPO, 'replays/cybernoodles_v2b_best.onnx'));
const shippedPath = arg('shipped', null);
const seconds = process.argv.includes('--full') ? 0 : Number(arg('seconds', 30));
const ortModule = process.env.ORT_MODULE
  || '/tmp/opencode/orttest/node_modules/onnxruntime-web/dist/ort.node.min.mjs';

if (!mapDir) {
  console.error('usage: node web/test/ort_e2e.mjs --map <mapDir> [--diff ExpertPlus] '
    + '[--shipped <bsor>] [--model <onnx>] [--full]');
  process.exit(2);
}

// ---- load the lib modules exactly as worker.js does ------------------------
for (const f of ['util.js', 'config.js', 'zip.js', 'map.js', 'encode.js',
  'model.js', 'score.js', 'avscore.js', 'bsor.js']) {
  await import(path.join(REPO, 'web', 'lib', f));
}
const CN = globalThis.CN;

const ort = await import(ortModule);
// CN.config.ortBase is the pinned jsDelivr URL; in Node the backend must load
// from disk instead.
CN.config.ortBase = path.dirname(fs.realpathSync(ortModule)) + '/';

const files = new Map();
for (const n of fs.readdirSync(mapDir)) {
  const f = path.join(mapDir, n);
  if (fs.statSync(f).isFile()) files.set(n.toLowerCase(), new Uint8Array(fs.readFileSync(f)));
}
const md = CN.map.parseMap(files, 'local', diff, 'Standard', null);
const fps = CN.encode.dynamicFps(md);
const nNotes = md.notes.length / 5;
console.log(`map  ${md.songName} — ${md.mapper}`);
console.log(`     ${diff}, ${nNotes} notes, ${md.duration.toFixed(1)} s, ${fps} fps, `
  + `${md.bombs.length / 3} bombs, ${md.walls.length / 6} walls, ${md.chains.length / 8} chains`);

const modelBytes = new Uint8Array(fs.readFileSync(modelPath));
const t0 = Date.now();
const sess = await CN.model.createSession(ort, modelBytes, (l) => console.log('  ' + l));
console.log(`     session ${((Date.now() - t0) / 1000).toFixed(1)} s, `
  + `${(modelBytes.length / 1048576).toFixed(1)} MiB, inputs [${sess.inputNames}]`);

const t1 = Date.now();
const traj = await CN.model.rollout({
  ort, sess, md, fps,
  overlap: CN.config.overlap, batch: CN.config.batch, histFrames: CN.config.histFrames,
  maxSeconds: seconds > 0 ? seconds : null,
  onProgress: (l) => console.log('  ' + l),
});
console.log(`     rollout ${traj.length / 22} frames in ${((Date.now() - t1) / 1000).toFixed(1)} s`);

// ---- reconstruct + score, exactly as the worker does ----------------------
const det = CN.score.reconstruct(traj, md);
const events = CN.score.noteEvents(det);
for (const b of CN.score.bombEvents(traj, md)) {
  events.push({ t: b.t, line: b.line, layer: b.layer, sabertype: 0, cutdir: 0, event: 3 });
}
events.sort((a, b) => a.t - b.t);
const walls = CN.score.detectWallHits(traj, md);
const acc = CN.avscore.accuracy(events, walls);
const horizon = seconds > 0 ? `${seconds} s` : 'full';
console.log(`score (JS/ORT, ${horizon}): ${acc.pct.toFixed(4)}% (${acc.score}/${acc.max}) `
  + `good=${acc.good} miss=${acc.miss} walls=${walls.length}`);

let ok = true;
if (shippedPath) {
  const { readNotes, accuracyFromNotes } = await import(path.join(HERE, 'bsorread.mjs'));
  const bytes = new Uint8Array(fs.readFileSync(shippedPath));
  const notes = readNotes(bytes);
  const ref = accuracyFromNotes(notes, CN.avscore);
  const d = Math.abs(ref.pct - acc.pct);
  console.log(`score (shipped .bsor) : ${ref.pct.toFixed(4)}% (${ref.score}/${ref.max}) `
    + `good=${ref.good} miss=${ref.miss}`);
  console.log(`delta                 : ${d.toFixed(4)} pt`);
  // Only meaningful at full length: the truncated horizon scores a different set
  // of notes, so require an exact match only for --full runs.
  if (horizon === 'full') ok = acc.score === ref.score;
  else ok = true;
}

process.exit(ok ? 0 : 1);
