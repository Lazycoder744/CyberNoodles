/* encode.js — map-window encoder, a verbatim port of
 * cybernoodles_infer.encode_window / gen_policy.encode_window.
 *
 * A "window" is 50 frames of pose the model predicts, conditioned on the notes
 * that fall in (anchor, anchor + 1.5 s].  Everything is window-relative: the
 * frame times are 0, 1/fps, ... and each note slot stores t - anchor.  (That
 * window-relative design is why the head channel is window-context decoration
 * and why the writer's Tukey crossfade invents a little head speed; see
 * FINDINGS_pass5.md.)
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});

  const FRAME_COUNT_OUTPUT = 50;
  const NOTE_COUNT = 50;
  const NOTE_LENGTH = 31;
  const WALL_COUNT = 50;
  const WALL_LENGTH = 6;
  const MAX_TIME_AHEAD = 1.5;
  const DEFAULT_FPS = 90.0;
  const QUIET_STAT = 4;
  const SINCE_LAST_CAP = 10.0;
  const EPS = 1e-6;

  /** Dataset mean of the x2 quality columns (cols 1:5). Baked-in constant. */
  function meanQuality() {
    return new Float32Array([0.6660554, 0.6134027, 0.7156775, 0.0230407]);
  }

  /** Per-map rollout fps from NJS (matches gen_policy.dynamic_fps). */
  function dynamicFps(md, base, hi, knee, kneeFps, njsLo, njsHi) {
    base = base === undefined ? 90.0 : base;
    hi = hi === undefined ? 140.0 : hi;
    knee = knee === undefined ? 20.0 : knee;
    kneeFps = kneeFps === undefined ? 110.0 : kneeFps;
    njsLo = njsLo === undefined ? 18.0 : njsLo;
    njsHi = njsHi === undefined ? 21.0 : njsHi;
    const njs = Number((md && md.njs) || 18.0) || 18.0;
    let f;
    if (njs <= knee) {
      f = base + (kneeFps - base) * (njs - njsLo) / Math.max(1e-6, knee - njsLo);
      f = Math.max(base, Math.min(kneeFps, f));
    } else {
      f = kneeFps + (hi - kneeFps) * (njs - knee) / Math.max(1e-6, njsHi - knee);
      f = Math.max(kneeFps, Math.min(hi, f));
    }
    return CN.util.pyRound(f / 5.0) * 5;
  }

  /** Seconds since the most recent note/bomb/chain at `anchor` (capped). */
  function sinceLastEvent(anchor, md) {
    let best = -Infinity;
    const scan = (arr, stride) => {
      if (!arr || !arr.length) return;
      for (let i = 0; i < arr.length; i += stride) {
        const t = arr[i];
        if (t <= anchor + 1e-6 && t > best) best = t;
      }
    };
    scan(md.notes, 5);
    scan(md.bombs, 3);
    scan(md.chains, 8);
    if (best === -Infinity) return SINCE_LAST_CAP;
    return Math.min(anchor - best, SINCE_LAST_CAP);
  }

  /**
   * Soft "note smoothing": each filled slot's channels 1,2,4 become a
   * time-distance-weighted average over the other filled slots of the same
   * window.  x2 is a flat (N, 50, 31) Float32Array; modified in place.
   */
  function smoothNotes(x2, N) {
    const NC = NOTE_COUNT, NL = NOTE_LENGTH;
    const t = new Float64Array(N * NC), ti = new Float64Array(N * NC);
    const zt = new Float64Array(N * NC), zi = new Float64Array(N * NC);
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < NC; i++) {
        const b = (n * NC + i) * NL;
        t[n * NC + i] = x2[b];
        ti[n * NC + i] = x2[b + 4];
        zt[n * NC + i] = x2[b] !== 0 ? 1 : 0;
        zi[n * NC + i] = x2[b + 4] !== 0 ? 1 : 0;
      }
    }
    const w = new Float64Array(NC * NC);
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < NC; i++) {
        const base = n * NC;
        for (let j = 0; j < NC; j++) {
          let v = 1.0 - 25.0 * Math.abs(t[base + i] - t[base + j]);
          if (v < 0) v = 0;
          w[i * NC + j] = v * zt[base + i] * zi[base + j];
        }
      }
      for (let i = 0; i < NC; i++) {
        let d = EPS;
        for (let j = 0; j < NC; j++) d += w[i * NC + j];
        for (const ch of [1, 2, 4]) {
          let s = 0;
          for (let j = 0; j < NC; j++) s += w[i * NC + j] * x2[(n * NC + j) * NL + ch];
          x2[(n * NC + i) * NL + ch] = s / d;
        }
      }
    }
  }

  /** Encode one anchor's window.  Returns {x1: Float32Array(56),
   *  x2: Float32Array(1550), x3: Float32Array(300)}. */
  function encodeWindow(anchor, md, fps, qualityMean) {
    anchor = Number(anchor);
    const q = qualityMean || meanQuality();

    const x1 = new Float32Array(FRAME_COUNT_OUTPUT + 6);
    for (let i = 0; i < FRAME_COUNT_OUTPUT; i++) x1[i] = i / fps;
    x1[FRAME_COUNT_OUTPUT + 0] = md.njs / 30.0;
    x1[FRAME_COUNT_OUTPUT + 1] = 18.0 / 30.0;
    x1[FRAME_COUNT_OUTPUT + 2] = 1.5;
    x1[FRAME_COUNT_OUTPUT + QUIET_STAT] = sinceLastEvent(anchor, md) / SINCE_LAST_CAP;

    const x2 = new Float32Array(NOTE_COUNT * NOTE_LENGTH);
    const elems = [];
    const nn = md.notes.length / 5;
    for (let idx = 0; idx < nn; idx++) {
      const t = md.notes[idx * 5];
      if (t > anchor + 1e-6 && t <= anchor + MAX_TIME_AHEAD) elems.push([t, 0, idx]);
    }
    const nb = md.bombs.length / 3;
    if (nb) {
      for (let idx = 0; idx < nb; idx++) {
        const t = md.bombs[idx * 3];
        if (t > anchor + 1e-6 && t <= anchor + MAX_TIME_AHEAD) elems.push([t, 1, idx]);
      }
    }
    const nch = md.chains.length / 8;
    if (nch) {
      for (let idx = 0; idx < nch; idx++) {
        const t = md.chains[idx * 8];
        if (t > anchor + 1e-6 && t <= anchor + MAX_TIME_AHEAD) elems.push([t, 2, idx]);
      }
    }
    elems.sort((a, b) => a[0] - b[0]);

    const rots = md.noteRot;
    const lim = Math.min(elems.length, NOTE_COUNT);
    for (let i = 0; i < lim; i++) {
      const kind = elems[i][1], idx = elems[i][2];
      const o = i * NOTE_LENGTH;
      if (kind === 0) {
        const t = md.notes[idx * 5], line = md.notes[idx * 5 + 1];
        const layer = md.notes[idx * 5 + 2], color = md.notes[idx * 5 + 3];
        const cd = md.notes[idx * 5 + 4];
        x2[o] = t - anchor;
        x2[o + 1] = q[0]; x2[o + 2] = q[1]; x2[o + 3] = q[2]; x2[o + 4] = q[3];
        const wxy = CN.map.noteWorld(line, layer);
        const xq = wxy[0] / 2.0, yq = wxy[1] / 2.0;
        const rot = rots ? rots[idx] / 360.0 : 0.0;
        if (color === 0) {
          x2[o + 5] = xq; x2[o + 6] = yq; x2[o + 7] = rot;
          x2[o + 8 + cd] = 1.0;
        } else {
          x2[o + 18] = xq; x2[o + 19] = yq; x2[o + 20] = rot;
          x2[o + 21 + cd] = 1.0;
        }
      } else if (kind === 1) {
        const t = md.bombs[idx * 3], line = md.bombs[idx * 3 + 1], layer = md.bombs[idx * 3 + 2];
        x2[o] = t - anchor;
        x2[o + 1] = q[0]; x2[o + 2] = q[1]; x2[o + 3] = q[2]; x2[o + 4] = q[3];
        const wxy = CN.map.noteWorld(line, layer);
        const xq = wxy[0] / 2.0, yq = wxy[1] / 2.0;
        x2[o + 5] = xq; x2[o + 6] = yq; x2[o + 7] = 0.0;
        x2[o + 8 + 9] = 1.0;
        x2[o + 18] = xq; x2[o + 19] = yq; x2[o + 20] = 0.0;
        x2[o + 21 + 9] = 1.0;
      } else {
        const b = idx * 8;
        const t = md.chains[b], color = md.chains[b + 3], cd = md.chains[b + 4];
        const rot = md.chains[b + 5], wx = md.chains[b + 6], wy = md.chains[b + 7];
        x2[o] = t - anchor;
        x2[o + 1] = q[0]; x2[o + 2] = q[1]; x2[o + 3] = q[2]; x2[o + 4] = q[3];
        const xq = wx / 2.0, yq = wy / 2.0, rq = rot / 360.0;
        if (color === 0) {
          x2[o + 5] = xq; x2[o + 6] = yq; x2[o + 7] = rq;
          x2[o + 8 + cd] = 1.0;
        } else {
          x2[o + 18] = xq; x2[o + 19] = yq; x2[o + 20] = rq;
          x2[o + 21 + cd] = 1.0;
        }
      }
    }

    const x3 = new Float32Array(WALL_COUNT * WALL_LENGTH);
    const nw = md.walls.length / 6;
    if (nw) {
      let ii = 0;
      for (let idx = 0; idx < nw && ii < WALL_COUNT; idx++) {
        const b = idx * 6;
        const t = md.walls[b], dur = md.walls[b + 3];
        if (!(t <= anchor + MAX_TIME_AHEAD && (t + dur) >= anchor)) continue;
        const line = md.walls[b + 1], width = md.walls[b + 2];
        const height = md.walls[b + 4], layer = md.walls[b + 5];
        const wx = (line - 1.5) * 0.6 + (width - 1.0) * 0.3;
        const wy = (0.85 - 0.2625) + 0.525 * layer + 0.2625 * height;
        const o = ii * WALL_LENGTH;
        x3[o + 0] = Math.max(0.0, t - anchor);
        x3[o + 1] = (t - anchor > 0 || t + dur - anchor < 0) ? 0.0 : 1.0;
        x3[o + 2] = wx / 2.0; x3[o + 3] = wy / 2.0;
        x3[o + 4] = width * 0.3; x3[o + 5] = height * 0.2625;
        ii++;
      }
    }

    smoothNotes(x2, 1);
    return { x1, x2, x3 };
  }

  CN.encode = {
    FRAME_COUNT_OUTPUT, NOTE_COUNT, NOTE_LENGTH, WALL_COUNT, WALL_LENGTH,
    MAX_TIME_AHEAD, DEFAULT_FPS, QUIET_STAT, SINCE_LAST_CAP,
    meanQuality, dynamicFps, sinceLastEvent, smoothNotes, encodeWindow,
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
