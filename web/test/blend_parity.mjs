/* blend_parity.mjs — check lib/model.js blend() against the Python reference.
 *
 *   python ../bsopencode/gen_blend_ref.py --onnx <model.onnx> \
 *          --map /mnt/games/bs-ai/maps_cache/<hash> --out /tmp/opencode/blendref
 *   node test/blend_parity.mjs [REFDIR]
 *
 * Covers the v2 head delta anchoring: the re-based head must match Python's
 * _blend exactly, so the browser replay and the honest scorer see one pipeline.
 */
import fs from 'fs';
import path from 'path';

const LIB = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'lib');
for (const f of ['util.js', 'config.js', 'model.js']) await import(path.join(LIB, f));
const CN = globalThis.CN;

const REF = process.argv[2] || '/tmp/opencode/blendref';
const f32 = (p) => {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const meta = JSON.parse(fs.readFileSync(path.join(REF, 'meta.json'), 'utf8'));
let bad = 0;
for (const [name, ysName, trajName] of [
  ['pass1', 'ys1.bin', 'traj1.bin'],
  ['pass2', 'ys2.bin', 'traj.bin'],
]) {
  const ys = f32(path.join(REF, ysName));
  const want = f32(path.join(REF, trajName));
  const got = CN.model.blend(meta.anchors, ys.slice(), meta.stride, meta.fps);
  let max = 0, at = -1;
  const n = Math.min(got.length, want.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > max) { max = d; at = i; }
  }
  const ok = got.length === want.length && max < 1e-4;
  console.log(`  ${name}: ${got.length} vs ${want.length} floats, max|d| ${max.toExponential(2)} at ${at} ${ok ? 'OK' : 'FAIL'}`);
  if (!ok) bad++;
  // Head channels are columns 1..3 of each 22-float frame; report them alone so
  // the anchoring fix is visible in the output.
  let hmax = 0;
  for (let f = 0; f + 22 <= n; f += 22) {
    for (let c = 1; c <= 3; c++) hmax = Math.max(hmax, Math.abs(got[f + c] - want[f + c]));
  }
  console.log(`    head max|d| ${hmax.toExponential(2)}`);
}
process.exit(bad ? 1 : 0);
