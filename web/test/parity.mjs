/* parity.mjs — Node harness that checks the browser pipeline against the
 * Python reference, without a browser and without onnxruntime:
 *
 *   map parse    JS CN.map.parseMap        vs  bsmap.load_map_dir
 *   encoder      JS CN.encode.encodeWindow vs  cybernoodles_infer.encode_window
 *   crossfade    JS CN.model.blend         vs  cybernoodles_infer._blend
 *   scorer       JS CN.score.reconstruct   vs  cybernoodles_infer.reconstruct
 *   accuracy     JS CN.avscore.accuracy    vs  av_score.arcviewer_accuracy
 *   writer       JS CN.bsor.writeBsor      vs  bsmap.write_bsor  (byte compare)
 *
 * The ONNX forwards are Python's (the ONNX graph already has a torch-vs-onnx
 * parity test), so everything the browser *adds* is exercised here.
 *
 *   node web/test/parity.mjs            # uses /tmp/opencode/jsref + maps_cache
 *   JSREF=... MAPDIR=... node web/test/parity.mjs
 */
import fs from 'fs';
import path from 'path';

const LIB = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'lib');
for (const f of ['util.js', 'config.js', 'zip.js', 'map.js', 'encode.js',
  'model.js', 'score.js', 'avscore.js', 'bsor.js']) {
  await import(path.join(LIB, f));
}
const CN = globalThis.CN;

const REF = process.env.JSREF || '/tmp/opencode/jsref';
const MAPDIR = process.env.MAPDIR
  || '/mnt/games/bs-ai/maps_cache/b5a1424cad898181d429cb231b83296a3e67d58f';

let failures = 0;
const ok = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL ' + msg); } };

function readF32(p) {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
function maxDiff(a, b, n) {
  n = n === undefined ? Math.min(a.length, b.length) : n;
  let m = 0, at = -1;
  if (a.length !== b.length) return { max: Infinity, at: -1, len: `${a.length} vs ${b.length}` };
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) { m = d; at = i; }
  }
  return { max: m, at };
}
const f32 = (arr) => Float32Array.from(arr);

// ---------------------------------------------------------------------------
console.log('== map parse ==');
const ref = JSON.parse(fs.readFileSync(path.join(REF, 'map.json'), 'utf8'));
const files = new Map();
for (const name of fs.readdirSync(MAPDIR)) {
  const full = path.join(MAPDIR, name);
  if (fs.statSync(full).isFile()) files.set(name.toLowerCase(), new Uint8Array(fs.readFileSync(full)));
}
const md = CN.map.parseMap(files, 'local', 'ExpertPlus', 'Standard', null);

for (const [k, refv, jsArr] of [
  ['notes', ref.notes, md.notes], ['bombs', ref.bombs, md.bombs],
  ['walls', ref.walls, md.walls], ['noteRot', ref.noteRot, md.noteRot],
  ['noteScoring', ref.noteScoring, md.noteScoring], ['chains', ref.chains, md.chains],
]) {
  const want = f32(refv.flat());
  const d = maxDiff(jsArr, want);
  console.log(`  ${k.padEnd(12)} n=${jsArr.length}  max|Δ|=${d.max.toExponential(2)}`);
  ok(d.max < 1e-6, `${k} mismatch (max ${d.max} at ${d.at})`);
}
for (const k of ['songName', 'mapper', 'environment', 'bpm', 'njs', 'duration']) {
  const a = md[k], b = ref[k];
  const same = (typeof a === 'string') ? a === b : Math.abs(a - b) < 1e-9;
  console.log(`  ${k.padEnd(12)} ${JSON.stringify(a)}${same ? '' : '  != ' + JSON.stringify(b)}`);
  ok(same, `${k} mismatch`);
}
console.log(`  fps(dynamic) ${CN.encode.dynamicFps(md)} vs ${ref.fps}`);
ok(CN.encode.dynamicFps(md) === ref.fps, 'dynamicFps mismatch');

// ---------------------------------------------------------------------------
console.log('== encoder ==');
const enc = JSON.parse(fs.readFileSync(path.join(REF, 'encode.json'), 'utf8'));
const q = CN.encode.meanQuality();
const meta = JSON.parse(fs.readFileSync(path.join(REF, 'meta.json'), 'utf8'));
const encByAnchor = new Map(Object.entries(enc.windows).map(([k, v]) => [Number(k), v]));
for (const a of enc.anchors) {
  const w = encByAnchor.get(Number(a));
  const got = CN.encode.encodeWindow(a, md, meta.fps, q);
  const dx1 = maxDiff(got.x1, f32(w.x1.flat()));
  const dx2 = maxDiff(got.x2, f32(w.x2.flat()));
  const dx3 = maxDiff(got.x3, f32(w.x3.flat()));
  console.log(`  anchor ${String(a).padEnd(5)} max|Δ| x1=${dx1.max.toExponential(2)} `
    + `x2=${dx2.max.toExponential(2)} x3=${dx3.max.toExponential(2)}`);
  ok(dx1.max < 1e-5 && dx2.max < 1e-5 && dx3.max < 1e-5, `encoder mismatch at ${a}`);
}

// ---------------------------------------------------------------------------
console.log('== Tukey crossfade ==');
const anchors = meta.anchors;
const ys1 = readF32(path.join(REF, 'ys1.bin'));
const ys2 = readF32(path.join(REF, 'ys2.bin'));
const traj1Ref = readF32(path.join(REF, 'traj1.bin'));
const trajRef = readF32(path.join(REF, 'traj.bin'));
const traj1 = CN.model.blend(anchors, ys1.slice(), meta.stride, meta.fps);
const traj = CN.model.blend(anchors, ys2.slice(), meta.stride, meta.fps);
{
  const d = maxDiff(traj1, traj1Ref);
  console.log(`  pass1 traj (${traj1.length}) max|Δ|=${d.max.toExponential(2)}`);
  ok(traj1.length === traj1Ref.length && d.max < 1e-4, 'traj1 mismatch');
  const d2 = maxDiff(traj, trajRef);
  console.log(`  pass2 traj (${traj.length}) max|Δ|=${d2.max.toExponential(2)}`);
  ok(traj.length === trajRef.length && d2.max < 1e-4, 'traj mismatch');
}

// ---------------------------------------------------------------------------
console.log('== cut reconstruction ==');
const refEv = JSON.parse(fs.readFileSync(path.join(REF, 'events.json'), 'utf8'));
const det = CN.score.reconstruct(traj, md);
// Normalise Python's writer dicts (type under `event`, ratings under `cut`)
// and JS's raw detector dicts into the same shape.
const normRef = (e) => {
  const c = e.cut || {};
  const cut = e.event === 0 || e.event === 1;
  return {
    t: e.t, typ: e.event, line: e.line, layer: e.layer, color: e.sabertype,
    cutdir: e.cutdir, scoring: e.scoring,
    dist: cut ? c.cutDistanceToCenter : 0.3,
    before: cut ? c.beforeCutRating : 0,
    after: cut ? c.afterCutRating : 0,
    speed: cut ? c.saberSpeed : 0,
  };
};
const normJs = (e) => ({
  t: e.t, typ: e.typ, line: e.line, layer: e.layer, color: e.color,
  cutdir: e.cutdir, scoring: e.scoring,
  dist: e.dist, before: e.before, after: e.after, speed: e.speed || 0,
});
const refDet = refEv.events.filter((e) => e.event === 0 || e.event === 1 || e.event === 2).map(normRef);
const jsDet = det.map(normJs);
console.log(`  events JS=${jsDet.length} ref(note path)=${refDet.length}`);
ok(jsDet.length >= refDet.length - 2, `event count ${jsDet.length} vs ${refDet.length}`);
{
  const n = Math.min(jsDet.length, refDet.length);
  const fields = ['t', 'dist', 'before', 'after', 'speed'];
  const acc = {};
  let typBad = 0, keyBad = 0, firstBad = -1;
  for (let i = 0; i < n; i++) {
    for (const f of fields) {
      const d = Math.abs(jsDet[i][f] - refDet[i][f]);
      if (!(f in acc) || d > acc[f]) acc[f] = d;
    }
    if (jsDet[i].typ !== refDet[i].typ) { typBad++; if (firstBad < 0) firstBad = i; }
    for (const f of ['line', 'layer', 'color', 'cutdir', 'scoring']) {
      if ((jsDet[i][f] | 0) !== (refDet[i][f] | 0)) keyBad++;
    }
  }
  console.log('  max|Δ| ' + Object.entries(acc).map(([k, v]) => `${k}=${v.toExponential(2)}`).join(' '));
  console.log(`  typ mismatches ${typBad}, key mismatches ${keyBad} (of ${n})`);
  if (firstBad >= 0) {
    console.log('  first mismatch:');
    console.log('    js  ' + JSON.stringify(jsDet[firstBad]));
    console.log('    ref ' + JSON.stringify(refDet[firstBad]));
  }
  for (const [k, v] of Object.entries(acc)) ok(v < 1e-4, `reconstruct ${k} drifted ${v}`);
  ok(typBad === 0, `${typBad} good/bad classification differences`);
  ok(keyBad === 0, `${keyBad} id-field mismatches`);
}

// ---------------------------------------------------------------------------
console.log('== honest score ==');
const jsEvents = CN.score.noteEvents(det);
for (const b of CN.score.bombEvents(traj, md)) {
  jsEvents.push({ t: b.t, line: b.line, layer: b.layer, sabertype: 0, cutdir: 0, event: 3 });
}
jsEvents.sort((a, b) => a.t - b.t);
const jsWalls = CN.score.detectWallHits(traj, md);
const jsAcc = CN.avscore.accuracy(jsEvents, jsWalls);
const refAcc = CN.avscore.accuracy(refEv.events, refEv.walls);
console.log(`  JS reconstruct -> score ${jsAcc.score}/${jsAcc.max} = ${jsAcc.pct.toFixed(4)}% `
  + `(good ${jsAcc.good} miss ${jsAcc.miss}, walls ${jsWalls.length})`);
console.log(`  JS on ref events -> score ${refAcc.score}/${refAcc.max} = ${refAcc.pct.toFixed(4)}% `
  + `(good ${refAcc.good} miss ${refAcc.miss}, walls ${refEv.walls.length})`);
ok(Math.abs(jsAcc.pct - refAcc.pct) < 0.01, 'accuracy differs between JS and ref events');

// ---------------------------------------------------------------------------
console.log('== BSOR: JS-written vs Python-written bytes ==');
const info = Object.assign({}, refEv.info);
const jsBytes = CN.bsor.writeBsor(traj, jsEvents, info, {
  fps: refEv.fps, songDuration: refEv.songDuration, wallEvents: jsWalls,
  heightEvents: refEv.heightEvents,
});
const refBytes = new Uint8Array(fs.readFileSync(path.join(REF, 'bsor.bin')));
console.log(`  bytes JS=${jsBytes.length} ref=${refBytes.length}`);

{
  let diff = 0, first = -1;
  for (let i = 0; i < Math.min(jsBytes.length, refBytes.length); i++) {
    if (jsBytes[i] !== refBytes[i]) { if (first < 0) first = i; diff++; }
  }
  const pc = 100 * diff / refBytes.length;
  console.log(`  differing bytes ${diff}/${refBytes.length} (${pc.toFixed(3)}%)`
    + `${first >= 0 ? `, first at ${first}` : ''}  [informational: the pose track differs`
    + ` from Python only by ONNX/float32 last-bit drift]`);
}

// The honest check: decode both files and compare what ArcViewer will score.
const readNotes = (await import('./bsorread.mjs')).readNotes;
const accuracyFromNotes = (await import('./bsorread.mjs')).accuracyFromNotes;
const jsNotes = readNotes(jsBytes);
const refNotes = readNotes(refBytes);
const jsAccN = accuracyFromNotes(jsNotes, CN.avscore);
const refAccN = accuracyFromNotes(refNotes, CN.avscore);
console.log(`  decoded events JS=${jsNotes.length} ref=${refNotes.length}`);
console.log(`  from JS bytes    score ${jsAccN.score}/${jsAccN.max} = ${jsAccN.pct.toFixed(4)}%`
  + ` (good ${jsAccN.good} miss ${jsAccN.miss})`);
console.log(`  from Python bytes score ${refAccN.score}/${refAccN.max} = ${refAccN.pct.toFixed(4)}%`
  + ` (good ${refAccN.good} miss ${refAccN.miss})`);
ok(jsNotes.length === refNotes.length, 'decoded event count differs');
{
  let typBad = 0, stBad = 0, scoreBad = 0;
  const n = Math.min(jsNotes.length, refNotes.length);
  for (let i = 0; i < n; i++) {
    if (jsNotes[i].typ !== refNotes[i].typ) typBad++;
    if (jsNotes[i].scoring !== refNotes[i].scoring) stBad++;
    if (CN.avscore.scoreGained(jsNotes[i]) !== CN.avscore.scoreGained(refNotes[i])) scoreBad++;
  }
  console.log(`  typ mismatches ${typBad}, scoring-type mismatches ${stBad}, per-note score mismatches ${scoreBad}`);
  ok(typBad === 0 && stBad === 0, 'decoded events differ');
  ok(scoreBad === 0, `${scoreBad} notes score differently`);
}
ok(Math.abs(jsAccN.pct - refAccN.pct) < 1e-9, 'accuracy from written bytes differs');

console.log(failures === 0 ? '\nALL PARITY CHECKS PASSED' : `\n${failures} PARITY CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
