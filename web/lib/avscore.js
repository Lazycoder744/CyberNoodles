/* avscore.js — ArcViewer's score model, a port of av_score.py.
 *
 * This is the honest score: it is computed from the same cut ratings that go
 * into the BSOR, using ArcViewer's own per-note max scores and combo
 * multipliers, so the number the app shows is the number you will see when you
 * watch the replay.
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});
  const U = CN.util;

  const MULT = [1, 2, 4, 8];
  const HITS = [2, 4, 8, 255];

  const ST_NO_SCORE = 2;
  const ST_NORMAL = 3;
  const ST_ARC_HEAD = 4;
  const ST_ARC_TAIL = 5;
  const ST_CHAIN_HEAD = 6;
  const ST_CHAIN_LINK = 7;

  // ScoringType -> (maxCenterDistanceCut, minBefore, maxBefore, minAfter,
  //                 maxAfter, fixedCut)
  const SCORE_DEF = {
    0: [15, 0, 70, 0, 30, 0],
    3: [15, 0, 70, 0, 30, 0],
    4: [15, 0, 70, 30, 30, 0],
    5: [15, 70, 70, 0, 30, 0],
    6: [15, 0, 70, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 20],
    8: [15, 70, 70, 30, 30, 0],
    9: [15, 70, 70, 30, 30, 0],
    10: [0, 0, 0, 0, 0, 20],
    11: [15, 0, 70, 30, 30, 0],
    12: [15, 70, 70, 30, 30, 0],
  };

  function maxEventScore(st) {
    if (st === ST_CHAIN_HEAD || st === 9 || st === 11 || st === 12) return 85;
    if (st === ST_CHAIN_LINK || st === 10) return 20;
    return 115;
  }

  /** ArcViewer ScoringEvent.CalculateNoteScore for a good hit. */
  function scoreGained(ev) {
    if (ev.typ !== 0) return 0;
    const st = ev.scoring === undefined ? ST_NORMAL : ev.scoring;
    if (st === ST_CHAIN_LINK || st === 10) return 20;
    const d = SCORE_DEF[st] || SCORE_DEF[ST_NORMAL];
    const maxc = d[0], minb = d[1], maxb = d[2], mina = d[3], maxa = d[4], fixed = d[5];
    const pre = Math.max(minb, Math.min(maxb, U.pyRound(U.clamp01(U.f32(ev.before)) * maxb)));
    const post = Math.max(mina, Math.min(maxa, U.pyRound(U.clamp01(U.f32(ev.after)) * maxa)));
    const acc = fixed > 0 ? fixed
      : U.pyRound(maxc * U.clamp01(1 - U.f32(ev.dist) / 0.3));
    return pre + post + acc;
  }

  /** The scoring-type ArcViewer recovers from a BSOR note-ID prefix. */
  function scoringFromId(stype, line, layer, sabertype, cutdir) {
    const nid = stype * 10000 + line * 1000 + layer * 100 + sabertype * 10 + cutdir;
    return (nid >= 0 && nid < 100000) ? Math.floor(nid / 10000) : ST_NORMAL;
  }

  /**
   * @param {Array} events  note/bomb events (writer dicts: {t,event,scoring,line,layer,sabertype,cutdir,cut})
   * @param {Array} walls   wall hit dicts ({time,...})
   * @returns {{pct:number, score:number, max:number, good:number, miss:number}}
   */
  function accuracy(events, walls, maxEvent) {
    maxEvent = maxEvent === undefined ? 115 : maxEvent;
    const list = [];
    for (const ev of events) {
      const stype = ev.scoring === undefined ? ST_NORMAL : ev.scoring;
      const st = scoringFromId(stype, ev.line | 0, ev.layer | 0, ev.sabertype | 0, ev.cutdir | 0);
      const c = ev.cut || {};
      list.push({
        t: ev.t, typ: ev.event, scoring: st, wall: false,
        before: ev.event === 0 || ev.event === 1 ? (c.beforeCutRating || 0) : 0,
        after: ev.event === 0 || ev.event === 1 ? (c.afterCutRating || 0) : 0,
        dist: ev.event === 0 || ev.event === 1 ? (c.cutDistanceToCenter || 0) : 0,
      });
    }
    for (const w of (walls || [])) {
      list.push({ t: w.time, typ: 2, scoring: ST_NORMAL, wall: true, before: 0, after: 0, dist: 0 });
    }
    list.sort((a, b) => a.t - b.t);   // stable: notes before walls at equal t

    let maxMult = 0, maxProg = 0, combo = 0, mult = 0, prog = 0, misses = 0;
    let cur = 0, mx = 0, goods = 0;
    for (const ev of list) {
      const st = ev.scoring;
      const wall = ev.wall;
      if (maxMult < MULT.length - 1 && !wall) {
        maxProg += 1;
        if (maxProg >= HITS[maxMult]) { maxMult += 1; maxProg = 0; }
      }
      mx += maxEventScore(st) * MULT[maxMult];
      const bad = ev.typ === 1 || ev.typ === 2 || ev.typ === 3;
      if (bad) {
        combo = 0;
        if (mult > 0) mult -= 1;
        prog = 0; misses += 1;
      } else {
        goods += 1;
        combo += 1;
        if (mult < MULT.length - 1) {
          prog += 1;
          if (prog >= HITS[mult]) { mult += 1; prog = 0; }
        }
        cur += scoreGained(ev) * MULT[mult];
      }
    }
    const pct = mx ? 100.0 * cur / mx : 100.0;
    return { pct, score: cur, max: mx, good: goods, miss: misses };
  }

  CN.avscore = { MULT, HITS, maxEventScore, scoreGained, scoringFromId, accuracy };
})(typeof globalThis !== 'undefined' ? globalThis : self);
