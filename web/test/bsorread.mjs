/* bsorread.mjs — test-only BSOR reader, a port of av_score.read_notes.
 * Used to prove that the replay bytes the browser writes score identically to
 * the ones the Python pipeline writes.
 */
const ST_NORMAL = 3;

export function readNotes(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = 0;
  const i32 = () => { const v = dv.getInt32(p, true); p += 4; return v; };
  const u8 = () => dv.getUint8(p++);
  const f32 = () => { const v = dv.getFloat32(p, true); p += 4; return v; };
  const bool = () => !!dv.getUint8(p++);
  const str = () => { const n = i32(); p += n; };
  const dec = new TextDecoder();

  i32(); u8();                       // magic, file version
  const notes = [];
  for (;;) {
    if (p >= bytes.length) break;
    const m = u8();
    if (m === 0) {
      for (let i = 0; i < 13; i++) str();
      i32(); str(); str(); str();
      f32(); bool(); f32(); f32(); f32(); f32();
    } else if (m === 1) {
      const n = i32();
      for (let i = 0; i < n; i++) { f32(); i32(); for (let k = 0; k < 21; k++) f32(); }
    } else if (m === 2) {
      const n = i32();
      for (let i = 0; i < n; i++) {
        const nid = i32(); const et = f32(); f32(); const typ = i32();
        const st = (nid >= 0 && nid < 100000) ? Math.floor(nid / 10000) : ST_NORMAL;
        const ev = { nid, t: et, typ, before: 0, after: 0, dist: 0, scoring: st };
        if (typ === 0 || typ === 1) {
          bool(); bool(); bool(); bool(); f32();
          for (let k = 0; k < 3; k++) f32();
          i32(); f32(); f32();
          for (let k = 0; k < 3; k++) f32();
          for (let k = 0; k < 3; k++) f32();
          ev.dist = f32(); f32();
          ev.before = f32(); ev.after = f32();
        }
        notes.push(ev);
      }
    } else if (m === 3) {
      const n = i32();
      for (let i = 0; i < n; i++) {
        i32(); f32(); const t = f32(); f32();
        notes.push({ nid: -1, t, typ: 2, before: 0, after: 0, dist: 0, wall: true });
      }
    } else if (m === 4) {
      const n = i32();
      for (let i = 0; i < n; i++) { f32(); f32(); }
    } else if (m === 5) {
      // NB: must not be `p += 24 * i32()` — the compound assignment reads `p`
      // before i32() advances it, discarding the advance.
      const np = i32();
      p += 24 * np;
    } else if (m === 6) {
      for (let i = 0; i < 2; i++) for (let k = 0; k < 7; k++) f32();
    } else if (m === 7) {
      const n = i32();
      for (let i = 0; i < n; i++) { str(); str(); str(); }
    } else break;
  }
  notes.sort((a, b) => a.t - b.t);
  void dec;
  return notes;
}

/** ArcViewer accuracy, using the shared score model from lib/avscore.js. */
export function accuracyFromNotes(notes, AV) {
  const MULT = AV.MULT, HITS = AV.HITS;
  let maxMult = 0, maxProg = 0, combo = 0, mult = 0, prog = 0, misses = 0;
  let cur = 0, mx = 0, goods = 0;
  for (const ev of notes) {
    const wall = !!ev.wall;
    if (maxMult < MULT.length - 1 && !wall) {
      maxProg += 1;
      if (maxProg >= HITS[maxMult]) { maxMult += 1; maxProg = 0; }
    }
    mx += AV.maxEventScore(ev.scoring) * MULT[maxMult];
    const bad = ev.typ === 1 || ev.typ === 2 || ev.typ === 3;
    if (bad) {
      combo = 0;
      if (mult > 0) mult -= 1;
      prog = 0; misses += 1;
    } else {
      goods += 1; combo += 1;
      if (mult < MULT.length - 1) {
        prog += 1;
        if (prog >= HITS[mult]) { mult += 1; prog = 0; }
      }
      cur += AV.scoreGained(ev) * MULT[mult];
    }
  }
  return { pct: mx ? 100.0 * cur / mx : 100.0, score: cur, max: mx, good: goods, miss: misses };
}
