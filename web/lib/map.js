/* map.js — Beat Saber map parsing, a faithful port of bsmap.py.
 *
 * Input is the set of files from a BeatSaver zip (see zip.js) plus the map's
 * BeatSaver API JSON, which is authoritative for BPM / song name / njs and is
 * also what the difficulty dropdown is built from.
 *
 * Output arrays are in *song seconds* and use the pipeline's world frame:
 *   x = (lineIndex - 1.5) * 0.6   -> -0.9, -0.3, 0.3, 0.9
 *   y = 0.85 + 0.525 * lineLayer  -> 0.85, 1.375, 1.9   (note_world uses
 *                                    0.85/1.4/1.9, matching the shards)
 *
 *   notes        Float32Array, stride 5: [t, line, layer, color(0=R,1=B), cutdir]
 *   bombs        Float32Array, stride 3: [t, line, layer]
 *   walls        Float32Array, stride 6: [t, line, width, duration, height, layer]
 *   noteRot      Float32Array(n): swing angle in degrees (deg/360 fed to the net)
 *   noteScoring  Int32Array(n): 3 normal / 6 chain head
 *   chains       Float32Array, stride 8: [t, line, layer, color, 8, rot, wx, wy]
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});
  const U = CN.util;

  const NOTE_SCORE_TYPE_NORMAL = 3;
  const NOTE_SCORE_TYPE_CHAIN_HEAD = 6;
  const NOTE_SCORE_TYPE_ARC_HEAD = 4;
  const NOTE_SCORE_TYPE_ARC_TAIL = 5;

  const LAYER_Y = [0.85, 1.4, 1.9];

  const DIFF_RANK = {
    easy: 1, normal: 3, hard: 5, expert: 7, expertplus: 9,
    'expert+': 9, easyplus: 2, hardplus: 6, expertplusplus: 10,
  };

  // World-space angle of the required swing direction (atan2(vy, vx), degrees).
  const CUT_BASE = {
    0: 90.0, 1: -90.0, 2: 180.0, 3: 0.0, 4: 135.0,
    5: 45.0, 6: -135.0, 7: -45.0, 8: 0.0, 9: 0.0,
  };

  function layerY(layer) { return LAYER_Y[Math.max(0, Math.min(2, U.i(layer)))]; }
  function lineX(line) { return (Number(line) - 1.5) * 0.6; }
  function noteWorld(line, layer) { return [lineX(line), layerY(layer)]; }

  /** Cut direction -> arrow rotation (deg).  Swing angle == rotation - 90. */
  function rotationAngle(cd) {
    cd = U.i(cd);
    if (cd >= 1000 && cd <= 1360) return 1000 - cd;
    if (cd >= 2000 && cd <= 2360) return 2000 - cd;
    const m = { 0: -180.0, 1: 0.0, 2: -90.0, 3: 90.0, 4: -135.0, 5: 135.0, 6: -45.0, 7: 45.0 };
    return (cd in m) ? m[cd] : 0.0;
  }

  function signedAngle(fx, fy, tx, ty) {
    const nf = Math.hypot(fx, fy) || 1e-12;
    const nt = Math.hypot(tx, ty) || 1e-12;
    let c = (fx * tx + fy * ty) / (nf * nt);
    c = Math.max(-1, Math.min(1, c));
    const ang = Math.acos(c) * 180.0 / Math.PI;
    return (fx * ty - fy * tx) >= 0 ? ang : -ang;
  }

  /** Cut direction -> required swing angle in world XY degrees. */
  function cutAngle(cd) {
    cd = U.i(cd);
    if (cd >= 1000 && cd <= 1360) return cd - 1000;
    if (cd >= 2000 && cd <= 2360) return cd - 2000;
    return (cd in CUT_BASE) ? CUT_BASE[cd] : 0.0;
  }

  function normDiff(s) {
    return String(s == null ? '' : s).trim().toLowerCase().replace(/ /g, '').replace(/_/g, '');
  }

  // -------------------------------------------------------------------------
  // difficulty-set selection (v2 uses _difficultyBeatmapSets, v3/v4 use
  // difficultyBeatmaps)
  // -------------------------------------------------------------------------
  function pickSetV2(v2, want) {
    const out = [];
    for (const s of (v2 || [])) {
      const ch = String(s._beatmapCharacteristicName || 'Standard').toLowerCase();
      const d = {};
      for (const bm of (s._difficultyBeatmaps || [])) {
        const dn = normDiff(bm._difficulty);
        d[dn] = [
          bm._beatmapFilename,
          Number(bm._noteJumpMovementSpeed || 16) || 16,
          bm._difficultyRank != null ? bm._difficultyRank : (DIFF_RANK[dn] || 1),
        ];
      }
      out.push([ch, d]);
    }
    const wantN = normDiff(want);
    for (const [ch, d] of out) if (ch.indexOf('standard') >= 0 && wantN in d) return d[wantN];
    for (const [ch, d] of out) if (ch.indexOf('onehanded') >= 0 && wantN in d) return d[wantN];
    for (const [ch, d] of out) {
      if ((ch === 'standard' || ch === '') && Object.keys(d).length) {
        const items = Object.entries(d).sort((a, b) => b[1][2] - a[1][2]);
        return items[0][1];
      }
    }
    return null;
  }

  function pickSetV3(v3, want) {
    const wantN = normDiff(want);
    const sets = Array.isArray(v3) ? v3 : [v3];
    let std = null, fallback = null;
    for (const s of sets) {
      if (!s || typeof s !== 'object' || !Array.isArray(s.difficultyBeatmaps)) continue;
      const ch = String(s.characteristic || 'Standard').toLowerCase();
      for (const bm of s.difficultyBeatmaps) {
        const dn = normDiff(bm.difficulty);
        const entry = [
          bm.beatmapDataFilename,
          Number(bm.noteJumpMovementSpeed || 16) || 16,
          bm.difficultyRank != null ? bm.difficultyRank : (DIFF_RANK[dn] || 1),
        ];
        if (ch.indexOf('standard') >= 0 && dn === wantN) return entry;
        if (ch.indexOf('standard') >= 0 && std === null) std = entry;
        if (fallback === null) fallback = entry;
      }
    }
    return std || fallback;
  }

  /** Beat -> seconds, honouring the map's BPM-change timeline. */
  function makeBeatToSec(dj, bpm, isV2) {
    let raw;
    if (isV2) {
      raw = [];
      for (const e of (dj._events || [])) {
        if (U.i(e._type != null ? e._type : -1) === 100) {
          raw.push([Number(e._time || 0.0), Number(e._value != null ? e._value : bpm)]);
        }
      }
    } else {
      raw = [];
      for (const e of (dj.bpmEvents || [])) {
        raw.push([Number(e.b || 0.0), Number(e.m != null ? e.m : bpm)]);
      }
    }
    raw.sort((a, b) => a[0] - b[0]);
    let events = [[0.0, Number(bpm)]];
    for (const [beat, m] of raw) {
      if (beat <= 1e-9) events[0] = [0.0, m];
      else events.push([beat, m]);
    }
    events.sort((a, b) => a[0] - b[0]);
    const dedup = [];
    for (const [beat, m] of events) {
      if (dedup.length && Math.abs(beat - dedup[dedup.length - 1][0]) < 1e-9) dedup[dedup.length - 1] = [beat, m];
      else dedup.push([beat, m]);
    }
    events = dedup;
    return function conv(beat) {
      beat = Number(beat) || 0;
      let t = 0.0, prev = 0.0, cur = events[0][1];
      for (const [b, m] of events) {
        if (beat <= b + 1e-9) return t + (beat - prev) * 60.0 / Math.max(cur, 1e-9);
        t += (b - prev) * 60.0 / Math.max(cur, 1e-9);
        prev = b; cur = m;
      }
      return t + (beat - prev) * 60.0 / Math.max(cur, 1e-9);
    };
  }

  // -------------------------------------------------------------------------
  // note/bomb/wall/chain extraction
  // -------------------------------------------------------------------------
  function parseDiff(dj, beatToSec, isV2) {
    const noteRows = [], bombs = [], walls = [], chains = [];

    if (isV2) {
      for (const n of (dj._notes || [])) {
        const t = beatToSec(n._time || 0);
        const li = U.i(n._lineIndex || 0), ll = U.i(n._lineLayer || 0);
        const ty = U.i(n._type || 0), cd = U.i(n._cutDirection != null ? n._cutDirection : 1);
        if (ty === 3) bombs.push([t, li, ll]);
        else {
          const color = ty === 0 ? 0 : 1;
          noteRows.push([t, li, ll, color, cd, cutAngle(cd), NOTE_SCORE_TYPE_NORMAL]);
        }
      }
      for (const w of (dj._obstacles || [])) {
        const b = Number(w._time || 0.0);
        const t = beatToSec(b);
        const dur = beatToSec(b + Number(w._duration || 0.0)) - t;
        walls.push([t, U.i(w._lineIndex || 0), Number(w._width != null ? w._width : 1),
          dur, Number(w._height != null ? w._height : 3), U.i(w._lineLayer || 0)]);
      }
    } else {
      for (const n of (dj.colorNotes || [])) {
        const t = beatToSec(n.b || 0);
        const cd = U.i(n.d != null ? n.d : 1);
        const rot = cutAngle(cd) + Number(n.a || 0.0);
        noteRows.push([t, U.i(n.x || 0), U.i(n.y || 0), U.i(n.c || 0), cd, rot, NOTE_SCORE_TYPE_NORMAL]);
      }
      for (const n of (dj.bombNotes || [])) {
        bombs.push([beatToSec(n.b || 0), U.i(n.x || 0), U.i(n.y || 0)]);
      }
      for (const w of (dj.obstacles || [])) {
        const b = Number(w.b || 0.0);
        const t = beatToSec(b);
        const dur = beatToSec(b + Number(w.d || 0.0)) - t;
        walls.push([t, U.i(w.x || 0), Number(w.w != null ? w.w : 1),
          dur, Number(w.h != null ? w.h : 3), U.i(w.y || 0)]);
      }

      // ---- chains (burst sliders) ----
      for (const sl of (dj.burstSliders || [])) {
        const tb = beatToSec(sl.b || 0);
        const tt = beatToSec(sl.tb != null ? sl.tb : (sl.b || 0));
        const x = U.i(sl.x || 0), y = U.i(sl.y || 0);
        const color = U.i(sl.c || 0);
        const sc = U.i(sl.sc || 1) || 1;
        const squish = Number(sl.s != null ? sl.s : 1.0) || 1.0;
        const tx = (sl.tx != null) ? sl.tx : x;
        const ty = (sl.ty != null) ? sl.ty : y;

        let hrow = -1;
        for (let i = 0; i < noteRows.length; i++) {
          const r = noteRows[i];
          if ((r[6] === NOTE_SCORE_TYPE_NORMAL || r[6] === NOTE_SCORE_TYPE_ARC_HEAD
            || r[6] === NOTE_SCORE_TYPE_ARC_TAIL)
            && Math.abs(r[0] - tb) < 1e-3 && r[1] === x && r[2] === y && r[3] === color) {
            hrow = i; break;
          }
        }
        let headRot;
        if (hrow >= 0) {
          const r = noteRows[hrow];
          noteRows[hrow] = [r[0], r[1], r[2], r[3], r[4], r[5], NOTE_SCORE_TYPE_CHAIN_HEAD];
          headRot = r[5];
        } else {
          headRot = cutAngle(U.i(sl.d != null ? sl.d : 8));
        }
        if (sc <= 1) continue;

        const hx = lineX(x), hy = layerY(y);
        const tailx = lineX(tx), taily = layerY(U.i(ty));
        const p2x = tailx - hx, p2y = taily - hy;
        const mag = Math.hypot(p2x, p2y);
        const f = headRot * Math.PI / 180.0;
        const p1x = 0.5 * mag * Math.cos(f), p1y = 0.5 * mag * Math.sin(f);
        for (let i = 1; i < sc; i++) {
          const frac = i / (sc - 1);
          const ti = tb + (tt - tb) * frac;
          const tpar = frac * squish;
          const nt = 1.0 - tpar;
          const posx = 2.0 * nt * tpar * p1x + tpar * tpar * p2x;
          const posy = 2.0 * nt * tpar * p1y + tpar * tpar * p2y;
          const tanx = 2.0 * nt * p1x + 2.0 * tpar * (p2x - p1x);
          const tany = 2.0 * nt * p1y + 2.0 * tpar * (p2y - p1y);
          const rot = signedAngle(0.0, -1.0, tanx, tany);
          chains.push([ti, x, y, color, 8, rot, hx + posx, hy + posy]);
        }
      }
    }

    const byT = (a, b) => a[0] - b[0];
    noteRows.sort(byT); bombs.sort(byT); walls.sort(byT); chains.sort(byT);

    const nn = noteRows.length;
    const notes = new Float32Array(nn * 5);
    const noteRot = new Float32Array(nn);
    const noteScoring = new Int32Array(nn);
    for (let i = 0; i < nn; i++) {
      const r = noteRows[i];
      notes[i * 5] = r[0]; notes[i * 5 + 1] = r[1]; notes[i * 5 + 2] = r[2];
      notes[i * 5 + 3] = r[3]; notes[i * 5 + 4] = r[4];
      noteRot[i] = r[5];
      noteScoring[i] = r[6];
    }
    const bm = new Float32Array(bombs.length * 3);
    for (let i = 0; i < bombs.length; i++) { bm[i * 3] = bombs[i][0]; bm[i * 3 + 1] = bombs[i][1]; bm[i * 3 + 2] = bombs[i][2]; }
    const wl = new Float32Array(walls.length * 6);
    for (let i = 0; i < walls.length; i++) for (let c = 0; c < 6; c++) wl[i * 6 + c] = walls[i][c];
    const ch = new Float32Array(chains.length * 8);
    for (let i = 0; i < chains.length; i++) for (let c = 0; c < 8; c++) ch[i * 8 + c] = chains[i][c];

    return { notes, bombs: bm, walls: wl, noteRot, noteScoring, chains: ch };
  }

  // -------------------------------------------------------------------------
  // MapData
  // -------------------------------------------------------------------------
  class MapData {
    constructor(hash, notes, bombs, walls, stats, songName, mapper, env,
      difficulty, mode, bpm, njs, noteRot, noteScoring, chains) {
      this.hash = hash;
      this.notes = notes;
      this.bombs = bombs;
      this.walls = walls;
      this.noteRot = noteRot;
      this.noteScoring = noteScoring;
      this.chains = chains;
      this.stats = stats;
      this.songName = songName;
      this.mapper = mapper;
      this.environment = env;
      this.difficulty = difficulty;
      this.mode = mode;
      this.bpm = bpm;
      this.njs = njs;
    }

    get duration() {
      let t = 0.0;
      const n = this.notes.length / 5;
      const w = this.walls.length / 6;
      const c = this.chains.length / 8;
      for (let i = 0; i < n; i++) t = Math.max(t, this.notes[i * 5]);
      for (let i = 0; i < w; i++) t = Math.max(t, this.walls[i * 6] + this.walls[i * 6 + 3]);
      for (let i = 0; i < c; i++) t = Math.max(t, this.chains[i * 8]);
      return t;
    }
  }

  // -------------------------------------------------------------------------
  // file lookup inside a virtual map directory (Map<path, Uint8Array>)
  // -------------------------------------------------------------------------
  function findKey(files, name) {
    const lname = String(name).replace(/\\/g, '/').toLowerCase();
    if (files.has(lname)) return lname;
    const base = lname.split('/').pop();
    for (const k of files.keys()) {
      if (k.split('/').pop() === base) return k;
    }
    return null;
  }

  function readText(files, name) {
    const k = findKey(files, name);
    if (k === null) return null;
    return U.decodeUtf8(files.get(k));
  }

  /**
   * Parse a map from its extracted files.
   * @param {Map<string,Uint8Array>|object} files  path -> bytes (lowercased keys)
   * @param {string} hash   song hash (40 hex chars, lowercase)
   * @param {string} difficulty  e.g. 'ExpertPlus'
   * @param {string} mode   'Standard'
   * @param {object|null} apiJson  the BeatSaver response for this version
   */
  function parseMap(files, hash, difficulty, mode, apiJson) {
    if (!(files instanceof Map)) {
      const m = new Map();
      for (const [k, v] of Object.entries(files || {})) m.set(k.toLowerCase(), v);
      files = m;
    }

    const infoText = readText(files, 'info.dat');
    if (infoText === null) throw new Error('map has no info.dat');
    const info = JSON.parse(infoText);
    const ver = String(info.version || '2.0.0');

    const meta = (apiJson && apiJson.metadata) || {};
    const bpm = Number(meta.bpm || info._beatsPerMinute || info.beatsPerMinute || 120) || 120;
    const songName = meta.songName || info._songName || info.songName || 'Unknown';
    const mapper = meta.levelAuthorName || info._levelAuthorName || info.levelAuthorName
      || info._songAuthorName || 'Unknown';
    const env = info._environmentName || info.environmentName || 'DefaultEnvironment';

    const wantN = normDiff(difficulty);
    let dfname = null, njs = 16.0;
    const picked = ver.startsWith('2')
      ? pickSetV2(info._difficultyBeatmapSets, difficulty)
      : pickSetV3(info.difficultyBeatmaps, difficulty);
    if (picked) { dfname = picked[0]; njs = picked[1]; }

    // Fall back to the API's diff list if info.dat did not resolve one.
    if (dfname === null && apiJson) {
      for (const v of (apiJson.versions || [])) {
        for (const d of (v.diffs || [])) {
          if (['standard', 'lawsaber'].indexOf(normDiff(d.characteristic)) >= 0
            && normDiff(d.difficulty) === wantN) {
            njs = Number(d.njs || njs);
            break;
          }
        }
      }
    }

    let parsed = {
      notes: new Float32Array(0), bombs: new Float32Array(0), walls: new Float32Array(0),
      noteRot: new Float32Array(0), noteScoring: new Int32Array(0), chains: new Float32Array(0),
    };

    if (dfname) {
      const base = String(dfname).replace(/\\/g, '/').split('/').pop();
      let text = readText(files, base);
      if (text === null) text = readText(files, dfname);
      if (text !== null) {
        const dj = JSON.parse(text);
        const isV2 = ('_notes' in dj) && !('colorNotes' in dj);
        parsed = parseDiff(dj, makeBeatToSec(dj, bpm, isV2), isV2);
      }
    }

    const stats = new Float32Array([njs, 18.0, 1.5, 0.0, 0.0]);
    return new MapData(hash, parsed.notes, parsed.bombs, parsed.walls, stats,
      songName, mapper, env, difficulty, mode, bpm, Number(njs),
      parsed.noteRot, parsed.noteScoring, parsed.chains);
  }

  /** Standard-characteristic difficulties from a BeatSaver API response. */
  function listDifficulties(apiJson) {
    const ver = ((apiJson || {}).versions || [])[0] || {};
    const out = [];
    for (const d of (ver.diffs || [])) {
      if (normDiff(d.characteristic) !== 'standard') continue;
      out.push({
        difficulty: d.difficulty, njs: d.njs, notes: d.notes, bombs: d.bombs,
        obstacles: d.obstacles, nps: d.nps, length: d.length,
        stars: d.stars, blStars: d.blStars, maxScore: d.maxScore,
        environment: d.environment,
      });
    }
    const order = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'];
    out.sort((a, b) => order.indexOf(a.difficulty) - order.indexOf(b.difficulty));
    return out;
  }

  CN.map = {
    MapData, parseMap, listDifficulties,
    noteWorld, cutAngle, rotationAngle, signedAngle, normDiff,
    layerY, lineX, makeBeatToSec, parseDiff, pickSetV2, pickSetV3,
    NOTE_SCORE_TYPE_NORMAL, NOTE_SCORE_TYPE_CHAIN_HEAD,
    NOTE_SCORE_TYPE_ARC_HEAD, NOTE_SCORE_TYPE_ARC_TAIL,
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
