/* score.js — honest cut reconstruction and event detection, a port of
 * cybernoodles_infer.reconstruct / note_events / bomb_events / detect_wall_hits
 * (which are themselves ports of saber_eval / game_score / writer_game).
 *
 * Nothing here is cosmetic: the returned events are exactly what the BSOR
 * contains and exactly what ArcViewer will score, so the accuracy the app
 * reports is the accuracy of the file you watch.
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});
  const U = CN.util;

  const L_BLADE = 1.27;      // pose blade length for distance tests
  const BLADE_CONTACT = 1.0; // blade length used for the cut reconstruction
  const Z0 = 1.00;
  const BOMB_R = 0.25;

  /** 180/pi rounded to float32 — numpy's float32 degrees() multiplier. */
  const DEG_F32 = Math.fround(180.0 / Math.PI);

  // cut direction -> required world swing angle (deg); 8/9 = any ("dot")
  const CUT_ANGLE = {
    0: 90.0, 1: -90.0, 2: 180.0, 3: 0.0, 4: 135.0,
    5: 45.0, 6: -135.0, 7: -45.0, 8: null, 9: null,
  };

  function searchLeft(a, v) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; } return lo; }
  function searchRight(a, v) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; } return lo; }
  function pyMod(x, m) { return ((x % m) + m) % m; }

  // ---------------------------------------------------------------------------
  // float32 helpers — numpy computes all of this in float32; JS doubles would
  // drift, and arccos() amplifies the drift near 0 deg, so the swing ratings
  // must reproduce numpy's rounding exactly.
  // ---------------------------------------------------------------------------
  const R32 = Math.fround;

  /** numpy np.linalg.norm of a 4-vector (quaternion). */
  function norm4f32(a, o) {
    const p0 = R32(a[o] * a[o]), p1 = R32(a[o + 1] * a[o + 1]);
    const p2 = R32(a[o + 2] * a[o + 2]), p3 = R32(a[o + 3] * a[o + 3]);
    return R32(Math.sqrt(R32(R32(R32(p0 + p1) + p2) + p3)));
  }

  /** Tips of both blades for every frame. motion: (F,21) flat. */
  function bladeTips(motion, F, length) {
    const tl = new Float32Array(F * 3), tr = new Float32Array(F * 3);
    for (let i = 0; i < F; i++) {
      const b = i * 21;
      for (let s = 0; s < 2; s++) {
        const ho = b + (s === 0 ? 7 : 14);
        const qo = b + (s === 0 ? 10 : 17);
        const inv = R32(1.0 / Math.max(norm4f32(motion, qo), 1e-9));
        const qx = R32(motion[qo] * inv), qy = R32(motion[qo + 1] * inv);
        const qz = R32(motion[qo + 2] * inv), qw = R32(motion[qo + 3] * inv);
        const o = i * 3;
        const out = s === 0 ? tl : tr;
        // Euler-Rodrigues column 2 of R(q) @ [0,0,length]
        out[o] = motion[ho] + R32(length * R32(2.0 * R32(R32(qx * qz) + R32(qw * qy))));
        out[o + 1] = motion[ho + 1] + R32(length * R32(2.0 * R32(R32(qy * qz) - R32(qw * qx))));
        out[o + 2] = motion[ho + 2] + R32(length * R32(1.0 - R32(2.0 * R32(R32(qx * qx) + R32(qy * qy)))));
      }
    }
    return [tl, tr];
  }

  /** Per-frame blade normal + per-frame blade-angle delta (degrees). */
  function saberRateArrays(hilt, tip, F) {
    const normal = new Float32Array(F * 3);
    const seg = new Float32Array(F);
    const bd = new Float32Array(F * 3);
    for (let i = 0; i < F; i++) {
      const o = i * 3;
      const dx = R32(tip[o] - hilt[o]), dy = R32(tip[o + 1] - hilt[o + 1]);
      const dz = R32(tip[o + 2] - hilt[o + 2]);
      const p0 = R32(dx * dx), p1 = R32(dy * dy), p2 = R32(dz * dz);
      const n = R32(Math.max(R32(Math.sqrt(R32(R32(p0 + p1) + p2))), 1e-9));
      bd[o] = R32(dx / n); bd[o + 1] = R32(dy / n); bd[o + 2] = R32(dz / n);
    }
    if (F > 1) {
      for (let i = 1; i < F; i++) {
        const o = i * 3, p = (i - 1) * 3;
        // mid point of the previous blade
        const mx = R32(R32(hilt[p] + tip[p]) * 0.5), my = R32(R32(hilt[p + 1] + tip[p + 1]) * 0.5);
        const mz = R32(R32(hilt[p + 2] + tip[p + 2]) * 0.5);
        // numpy crosses the *unnormalised* blade with (mid - hilt)
        const ax = R32(tip[o] - hilt[o]), ay = R32(tip[o + 1] - hilt[o + 1]);
        const az = R32(tip[o + 2] - hilt[o + 2]);
        const bx = R32(mx - hilt[o]), by = R32(my - hilt[o + 1]), bz = R32(mz - hilt[o + 2]);
        const cx = R32(R32(ay * bz) - R32(az * by));
        const cy = R32(R32(az * bx) - R32(ax * bz));
        const cz = R32(R32(ax * by) - R32(ay * bx));
        const q0 = R32(cx * cx), q1 = R32(cy * cy), q2 = R32(cz * cz);
        const cn = R32(Math.max(R32(Math.sqrt(R32(R32(q0 + q1) + q2))), 1e-9));
        normal[o] = R32(cx / cn); normal[o + 1] = R32(cy / cn); normal[o + 2] = R32(cz / cn);
        let cs = dot3f32(bd, o, bd, p);
        cs = Math.max(-1, Math.min(1, cs));
        seg[i] = R32(R32(Math.acos(cs)) * DEG_F32);
      }
    }
    return [normal, seg];
  }

  /** Dotted product of two 3-vectors, accumulated in float32 like numpy. */
  function dot3f32(a, ai, b, bi) {
    const p0 = R32(a[ai] * b[bi]);
    const p1 = R32(a[ai + 1] * b[bi + 1]);
    const p2 = R32(a[ai + 2] * b[bi + 2]);
    return R32(R32(p0 + p1) + p2);
  }

  /** Point-to-segment distance. */
  function segmentDistance(hand, ho, tip, to, px, py, pz) {
    const ax = tip[to] - hand[ho], ay = tip[to + 1] - hand[ho + 1], az = tip[to + 2] - hand[ho + 2];
    let den = ax * ax + ay * ay + az * az;
    if (den < 1e-9) den = 1e-9;
    const qx = px - hand[ho], qy = py - hand[ho + 1], qz = pz - hand[ho + 2];
    let u = (qx * ax + qy * ay + qz * az) / den;
    u = u < 0 ? 0 : (u > 1 ? 1 : u);
    const ex = hand[ho] + u * ax - px, ey = hand[ho + 1] + u * ay - py, ez = hand[ho + 2] + u * az - pz;
    return Math.sqrt(ex * ex + ey * ey + ez * ez);
  }

  function angIdx(a, ai, b, bi) {
    // numpy: _ang = degrees(arccos(clip((u*v).sum(-1), -1, 1))) on float32
    // arrays, so every step rounds to float32 before the float64 comparison.
    const d = dot3f32(a, ai, b, bi);
    const c = Math.max(-1, Math.min(1, d));
    return Math.fround(Math.fround(Math.acos(c)) * DEG_F32);
  }

  /** np.float32 arithmetic for one swing-loop term (see _swing_before). */
  function swingTerm(s, aw, k) {
    if (aw < 75) return Math.fround(s / k);
    const t = Math.fround(s * Math.fround(90.0 - aw));
    return Math.fround(Math.fround(t / 15.0) / k);
  }

  /** Slab test: does each blade segment [hil[i], tip[i]] touch the axis-aligned
   *  box (center[i] +/- half)?  Returns a Uint8Array over the window. */
  function segBox(hil, tip, c, n, half) {
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      let tmin = 0.0, tmax = 1.0, ok = true;
      for (let a = 0; a < 3; a++) {
        const d = tip[o + a] - hil[o + a];
        const inv = 1.0 / (Math.abs(d) < 1e-9 ? 1e-9 : d);
        const t1 = (c[o + a] - half[a] - hil[o + a]) * inv;
        const t2 = (c[o + a] + half[a] - hil[o + a]) * inv;
        const lo = Math.min(t1, t2), hi = Math.max(t1, t2);
        if (lo > tmin) tmin = lo;
        if (hi < tmax) tmax = hi;
        if (tmin > tmax) { ok = false; break; }
      }
      out[i] = ok ? 1 : 0;
    }
    return out;
  }

  function swingBefore(normal, seg, times, i, overrideDeg, cap) {
    const k = 100.0;
    let r = (overrideDeg === null || overrideDeg === undefined ? seg[i] : overrideDeg) / k;
    let prevT = times[i];
    for (let j = i - 1; j >= 0; j--) {
      if (times[i] - prevT >= 0.4) break;
      const aw = angIdx(normal, i * 3, normal, j * 3);
      if (aw >= 90) break;
      prevT = times[j];
      // numpy accumulates the float32 sum into a float32 result
      // (python float + np.float32 -> np.float32), so round each step.
      r = Math.fround(Math.fround(r) + swingTerm(seg[j], aw, k));
      if (cap && r > 1.0) return 1.0;
    }
    return r;
  }

  function swingAfter(normal, seg, times, i, afterErrorDeg, cap) {
    const k = 60.0;
    let r = afterErrorDeg / k;
    for (let j = i + 1; j < times.length; j++) {
      if (times[j] - times[i] > 0.4) break;
      const aw = angIdx(normal, i * 3, normal, j * 3);
      if (aw >= 90) break;
      r = Math.fround(Math.fround(r) + swingTerm(seg[j], aw, k));
      if (cap && r > 1.0) return 1.0;
    }
    return r;
  }

  /**
   * Full cut reconstruction.
   * @param {Float32Array} traj (F,22)
   * @param {CN.map.MapData} md
   * @returns {Array} deterministic events (one per note/chain link)
   */
  function reconstruct(traj, md, opts) {
    opts = opts || {};
    const half = opts.half || [0.55, 0.40, 0.70];
    const halfDot = opts.halfDot || half;
    const search = opts.search === undefined ? 0.16 : opts.search;
    const z0 = opts.z0 === undefined ? Z0 : opts.z0;
    const bladeLen = opts.bladeLen === undefined ? BLADE_CONTACT : opts.bladeLen;
    const capRatings = opts.capRatings === undefined ? true : opts.capRatings;

    const F = traj.length / 22;
    const tarr = new Float64Array(F);
    const motion = new Float32Array(F * 21);
    for (let i = 0; i < F; i++) {
      tarr[i] = traj[i * 22];
      for (let c = 0; c < 21; c++) motion[i * 21 + c] = traj[i * 22 + 1 + c];
    }
    const tips = bladeTips(motion, F, bladeLen);
    const tl = tips[0], tr = tips[1];
    const hilL = sub(motion, F, 7), hilR = sub(motion, F, 14);
    const nmL = saberRateArrays(hilL, tl, F);
    const nmR = saberRateArrays(hilR, tr, F);
    const njs = Number(md.njs);

    // ---- target list: notes then chain links, sorted by time ----
    const nsc = md.noteScoring.length ? md.noteScoring : null;
    const nin = md.notes.length / 5;
    const targets = [];
    for (let n = 0; n < nin; n++) {
      const b = n * 5;
      const wxy = CN.map.noteWorld(md.notes[b + 1], md.notes[b + 2]);
      targets.push({
        t: md.notes[b], line: md.notes[b + 1] | 0, layer: md.notes[b + 2] | 0,
        color: md.notes[b + 3] | 0, cd: md.notes[b + 4] | 0,
        wx: wxy[0], wy: wxy[1], scoring: nsc ? nsc[n] : CN.map.NOTE_SCORE_TYPE_NORMAL,
      });
    }
    const nch = md.chains.length / 8;
    for (let n = 0; n < nch; n++) {
      const b = n * 8;
      targets.push({
        t: md.chains[b], line: md.chains[b + 1] | 0, layer: md.chains[b + 2] | 0,
        color: md.chains[b + 3] | 0, cd: md.chains[b + 4] | 0,
        wx: md.chains[b + 6], wy: md.chains[b + 7],
        scoring: CN.map.NOTE_SCORE_TYPE_CHAIN_LINK,
      });
    }
    targets.sort((a, b) => a.t - b.t);

    const miss = (g) => ({
      t: g.t, typ: 2, before: 0.0, after: 0.0, dist: 0.3, scoring: g.scoring,
      line: g.line, layer: g.layer, color: g.color, cutdir: g.cd,
    });

    const evs = [];
    for (const g of targets) {
      const tn = g.t;
      const left = g.color === 0;
      let lo = searchLeft(tarr, tn - search);
      let hi = searchRight(tarr, tn + search);
      lo = Math.max(1, lo); hi = Math.min(F, hi);
      const n = hi - lo;
      if (n < 3) { evs.push(miss(g)); continue; }

      const c = new Float32Array(n * 3);
      for (let f = 0; f < n; f++) {
        c[f * 3] = g.wx; c[f * 3 + 1] = g.wy; c[f * 3 + 2] = z0 + njs * (tn - tarr[lo + f]);
      }
      const hil = left ? hilL : hilR;
      const tip = left ? tl : tr;
      const halfb = g.cd >= 8 ? halfDot : half;
      // Python slices the window out (hil[lo:hi]); mirror that exactly.
      const hilW = new Float32Array(n * 3), tipW = new Float32Array(n * 3);
      for (let f = 0; f < n; f++) {
        for (let a = 0; a < 3; a++) {
          hilW[f * 3 + a] = hil[(lo + f) * 3 + a];
          tipW[f * 3 + a] = tip[(lo + f) * 3 + a];
        }
      }
      const hit = segBox(hilW, tipW, c, n, halfb);
      let any = false;
      for (let f = 0; f < n; f++) if (hit[f]) { any = true; break; }
      if (!any) { evs.push(miss(g)); continue; }

      // nearest in-box frame
      let fbest = -1, dbest = Infinity;
      for (let f = 0; f < n; f++) {
        if (!hit[f]) continue;
        const o = f * 3;
        const ax = tipW[o] - hilW[o], ay = tipW[o + 1] - hilW[o + 1], az = tipW[o + 2] - hilW[o + 2];
        let den = ax * ax + ay * ay + az * az; if (den < 1e-9) den = 1e-9;
        const qx = c[o] - hilW[o], qy = c[o + 1] - hilW[o + 1], qz = c[o + 2] - hilW[o + 2];
        let u = (qx * ax + qy * ay + qz * az) / den;
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
        const ex = hilW[o] + u * ax - c[o], ey = hilW[o + 1] + u * ay - c[o + 1], ez = hilW[o + 2] + u * az - c[o + 2];
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez);
        if (d < dbest) { dbest = d; fbest = f; }
      }
      const fr = lo + fbest;
      const normal = left ? nmL[0] : nmR[0];
      const seg = left ? nmL[1] : nmR[1];
      const fo = fbest * 3;
      const hlt = fr * 3;
      const dist = Math.abs((c[fo] - hil[hlt]) * normal[fr * 3]
        + (c[fo + 1] - hil[hlt + 1]) * normal[fr * 3 + 1]
        + (c[fo + 2] - hil[hlt + 2]) * normal[fr * 3 + 2]);

      const tipv = left ? tl : tr;
      const hiltv = left ? hilL : hilR;
      let speed = 0.0, ang = null;
      if (fr > 0 && fr < F - 1) {
        const dtf = tarr[fr + 1] - tarr[fr - 1];
        if (dtf > 1e-6) {
          const vx = (tipv[(fr + 1) * 3] - tipv[(fr - 1) * 3]) / dtf;
          const vy = (tipv[(fr + 1) * 3 + 1] - tipv[(fr - 1) * 3 + 1]) / dtf;
          const vz = (tipv[(fr + 1) * 3 + 2] - tipv[(fr - 1) * 3 + 2]) / dtf;
          speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
          const req = CUT_ANGLE[g.cd];
          if (req !== null && req !== undefined) {
            const a = Math.atan2(vy, vx) * 180.0 / Math.PI;
            ang = Math.abs(pyMod(a - req + 180.0, 360.0) - 180.0);
          }
        }
      }
      const speedOk = speed >= 2.0;
      const bo = fr * 3;
      const sbl = [tipv[bo] - hiltv[bo], tipv[bo + 1] - hiltv[bo + 1], tipv[bo + 2] - hiltv[bo + 2]];
      const snbl = Math.max(Math.hypot(sbl[0], sbl[1], sbl[2]), 1e-9);
      const sdir = [sbl[0] / snbl, sbl[1] / snbl, sbl[2] / snbl];
      const cutNormal = [normal[bo], normal[bo + 1], normal[bo + 2]];
      const cutPoint = [c[fo], c[fo + 1], c[fo + 2]];

      let before = 0.0, after = 0.0;
      if (fr > 0 && fr < F - 1) {        // plane normal = cross(blade normal, forward) = [n1, -n0, 0]
        let pnx = normal[bo + 1];
        let pny = -normal[bo];
        let pnz = 0.0;
        const pnlen = Math.max(Math.hypot(pnx, pny, pnz), 1e-9);
        pnx /= pnlen; pny /= pnlen; pnz /= pnlen;
        const pix = g.wx, piy = g.wy, piz = z0 + njs * (tn - tarr[fr]);
        const side = new Float64Array(n);
        for (let j = 0; j < n; j++) {
          side[j] = (tipv[(lo + j) * 3] - pix) * pnx
            + (tipv[(lo + j) * 3 + 1] - piy) * pny
            + (tipv[(lo + j) * 3 + 2] - piz) * pnz;
        }
        let cr = -1;
        for (let j = 1; j < n; j++) {
          if (side[j - 1] === 0 || side[j] === 0 || side[j - 1] * side[j] < 0) { cr = j; break; }
        }
        if (cr < 0) cr = Math.max(1, fbest);
        const jf = lo + cr, jp = Math.max(lo, jf - 1);
        const d0 = side[cr - 1], d1 = side[cr];
        const wq = Math.abs(d1 - d0) < 1e-9 ? 0.0 : d0 / (d0 - d1);
        const jfo = jf * 3, jpo = jp * 3;
        // numpy: tipp is float32, w is a python float, and float32 (weak)
        // promotion keeps cut_tip / cut_hil / bl in float32.
        const cutTip = [
          R32(tipv[jpo] + R32(wq * R32(tipv[jfo] - tipv[jpo]))),
          R32(tipv[jpo + 1] + R32(wq * R32(tipv[jfo + 1] - tipv[jpo + 1]))),
          R32(tipv[jpo + 2] + R32(wq * R32(tipv[jfo + 2] - tipv[jpo + 2]))),
        ];
        const cutHil = [
          R32(R32(hiltv[jpo] + hiltv[jfo]) * 0.5),
          R32(R32(hiltv[jpo + 1] + hiltv[jfo + 1]) * 0.5),
          R32(R32(hiltv[jpo + 2] + hiltv[jfo + 2]) * 0.5),
        ];
        const bl = new Float32Array([
          R32(cutTip[0] - cutHil[0]), R32(cutTip[1] - cutHil[1]), R32(cutTip[2] - cutHil[2]),
        ]);
        const av = new Float32Array([
          R32(tipv[jpo] - hiltv[jpo]), R32(tipv[jpo + 1] - hiltv[jpo + 1]),
          R32(tipv[jpo + 2] - hiltv[jpo + 2]),
        ]);
        const bv = new Float32Array([
          R32(tipv[jfo] - hiltv[jfo]), R32(tipv[jfo + 1] - hiltv[jfo + 1]),
          R32(tipv[jfo + 2] - hiltv[jfo + 2]),
        ]);
        // _ang(u, v) = degrees(arccos(clip((u*v).sum(-1), -1, 1))) on float32,
        // with no normalisation — reproduce that literally.
        const cb = Math.max(-1, Math.min(1, dot3f32(bl, 0, av, 0)));
        const ca = Math.max(-1, Math.min(1, dot3f32(bl, 0, bv, 0)));
        const bErr = R32(R32(Math.acos(cb)) * DEG_F32);
        const aErr = R32(R32(Math.acos(ca)) * DEG_F32);
        before = swingBefore(normal, seg, tarr, jf, bErr, capRatings);
        after = swingAfter(normal, seg, tarr, jf, aErr, capRatings);
      }

      evs.push({
        t: tn, typ: speedOk ? 0 : 1, before, after, dist, speed, ang,
        speed_ok: speedOk, dir_ok: true, sdir, cut_normal: cutNormal,
        cut_point: cutPoint, scoring: g.scoring, line: g.line, layer: g.layer,
        color: g.color, cutdir: g.cd,
      });
    }
    evs.sort((a, b) => a.t - b.t);
    return evs;
  }

  function sub(src, F, off) {
    const out = new Float32Array(F * 3);
    for (let i = 0; i < F; i++) {
      out[i * 3] = src[i * 21 + off];
      out[i * 3 + 1] = src[i * 21 + off + 1];
      out[i * 3 + 2] = src[i * 21 + off + 2];
    }
    return out;
  }

  /** Event dicts for the BSOR notes section (good/bad/miss + cut payload). */
  function noteEvents(det) {
    const evs = [];
    for (const d of det) {
      const ev = {
        t: d.t, line: d.line | 0, layer: d.layer | 0,
        sabertype: (d.color | 0) === 0 ? 0 : 1,
        cutdir: d.cutdir | 0,
        scoring: d.scoring === undefined ? CN.map.NOTE_SCORE_TYPE_NORMAL : d.scoring,
        event: d.typ,
      };
      if (d.typ === 0 || d.typ === 1) {
        ev.cut = {
          speedOK: d.speed_ok === undefined ? true : !!d.speed_ok,
          directionOk: d.dir_ok === undefined ? true : !!d.dir_ok,
          saberTypeOk: true, wasCutTooSoon: false,
          saberSpeed: d.speed === undefined ? 20.0 : d.speed,
          saberDirection: d.sdir || [0.0, 0.0, 1.0],
          saberType: ev.sabertype, timeDeviation: 0.0, cutDeviation: 0.0,
          cutPoint: d.cut_point || [0.0, 0.0, 0.0],
          cutNormal: d.cut_normal || [0.0, 0.0, 1.0],
          cutDistanceToCenter: d.dist, cutAngle: 90.0,
          beforeCutRating: d.before, afterCutRating: d.after,
        };
      }
      evs.push(ev);
    }
    return evs;
  }

  /** Closest approach of either blade to each bomb. */
  function bombDistances(traj, md, z0) {
    z0 = z0 === undefined ? Z0 : z0;
    const nb = md.bombs.length / 3;
    const out = new Float64Array(nb);
    if (!nb) return out;
    const F = traj.length / 22;
    const tarr = new Float64Array(F);
    const motion = new Float32Array(F * 21);
    for (let i = 0; i < F; i++) {
      tarr[i] = traj[i * 22];
      for (let c = 0; c < 21; c++) motion[i * 21 + c] = traj[i * 22 + 1 + c];
    }
    const tips = bladeTips(motion, F, L_BLADE);
    const tl = tips[0], tr = tips[1];
    const njs = Number(md.njs);
    for (let b = 0; b < nb; b++) {
      const tb = md.bombs[b * 3];
      const line = md.bombs[b * 3 + 1], layer = md.bombs[b * 3 + 2];
      let f = 0, best = Infinity;
      for (let i = 0; i < F; i++) { const d = Math.abs(tarr[i] - tb); if (d < best) { best = d; f = i; } }
      const z = z0 + njs * (tb - tarr[f]);
      const wxy = CN.map.noteWorld(line, layer);
      const dl = segmentDistance(motion, f * 21 + 7, tl, f * 3, wxy[0], wxy[1], z);
      const dr = segmentDistance(motion, f * 21 + 14, tr, f * 3, wxy[0], wxy[1], z);
      out[b] = Math.min(dl, dr);
    }
    return out;
  }

  function bombEvents(traj, md, radius, z0) {
    radius = radius === undefined ? BOMB_R : radius;
    const ds = bombDistances(traj, md, z0);
    const out = [];
    for (let i = 0; i < ds.length; i++) {
      if (ds[i] < radius) {
        out.push({ t: md.bombs[i * 3], typ: 3, line: md.bombs[i * 3 + 1] | 0, layer: md.bombs[i * 3 + 2] | 0 });
      }
    }
    return out;
  }

  /** Honest wall collisions (head inside the wall's x/y rectangle while active). */
  function detectWallHits(traj, md) {
    const F = traj.length / 22;
    const ts = new Float64Array(F);
    const hx = new Float64Array(F), hy = new Float64Array(F);
    for (let i = 0; i < F; i++) { ts[i] = traj[i * 22]; hx[i] = traj[i * 22 + 1]; hy[i] = traj[i * 22 + 2]; }
    const hits = [];
    const nw = md.walls.length / 6;
    for (let i = 0; i < nw; i++) {
      const b = i * 6;
      const t = md.walls[b], line = md.walls[b + 1], width = md.walls[b + 2];
      const dur = md.walls[b + 3], height = md.walls[b + 4], layer = md.walls[b + 5];
      const wx = (line - 1.5) * 0.6 + (width - 1.0) * 0.3;
      const wy = (0.85 - 0.2625) + 0.525 * layer + 0.2625 * height;
      const hw = width * 0.3, hh = height * 0.2625;
      const lo = searchLeft(ts, t), hi = searchRight(ts, t + dur);
      if (hi <= lo) continue;
      for (let k = lo; k < hi; k++) {
        if (Math.abs(hx[k] - wx) <= hw && Math.abs(hy[k] - wy) <= hh) {
          hits.push({ id: i, energy: 0.5, time: t, spawnTime: Math.max(0.0, t - 1.2) });
          break;
        }
      }
    }
    return hits;
  }

  CN.score = {
    L_BLADE, BLADE_CONTACT, Z0, BOMB_R, CUT_ANGLE,
    bladeTips, saberRateArrays, segmentDistance, segBox,
    swingBefore, swingAfter, reconstruct, noteEvents,
    bombDistances, bombEvents, detectWallHits, angIdx, dot3f32,
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
