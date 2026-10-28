/* bsor.js — BSOR writer, a byte-for-byte port of bsmap.write_bsor.
 *
 * Layout (magic 0x442d3d69, file version 1): info, frames, notes, walls,
 * heights, pauses, controller offsets, user data.  It was taken from the
 * `bsor` package's own decoder, so replays round-trip through
 * `bsor.Bsor.make_bsor` and load in ArcViewer / BeatLeader.
 *
 * Replay identity is the CyberNoodles bot account (see identity.py).
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});
  const U = CN.util;

  const BSOR_MAGIC = 0x442D3D69;

  const NOTE_EVENT_GOOD = 0;
  const NOTE_EVENT_BAD = 1;
  const NOTE_EVENT_MISS = 2;
  const NOTE_EVENT_BOMB = 3;

  const IDENTITY = {
    playerId: '345479',
    playerName: 'CyberNoodles [BOT]',
    platform: 'oculus',
    trackingSystem: 'Oculus',
    hmd: 'Oculus Rift S',
    controller: 'Oculus Touch',
  };

  function makeWriter() {
    let cap = 1 << 20;
    let buf = new Uint8Array(cap);
    let n = 0;
    let dv = new DataView(buf.buffer);
    const ensure = (k) => {
      if (n + k <= cap) return;
      let nc = cap;
      while (nc < n + k) nc *= 2;
      const nb = new Uint8Array(nc);
      nb.set(buf.subarray(0, n));
      buf = nb; cap = nc; dv = new DataView(buf.buffer);
    };
    return {
      u8(v) { ensure(1); buf[n++] = v & 0xff; },
      i32(v) { ensure(4); dv.setInt32(n, v | 0, true); n += 4; },
      f32(v) { ensure(4); dv.setFloat32(n, v, true); n += 4; },
      bool(v) { ensure(1); buf[n++] = v ? 1 : 0; },
      str(s) {
        const b = U.encodeUtf8(s);
        ensure(4); dv.setInt32(n, b.length, true); n += 4;
        ensure(b.length); buf.set(b, n); n += b.length;
      },
      f3(v) { this.f32(v[0]); this.f32(v[1]); this.f32(v[2]); },
      out() { return buf.slice(0, n); },
    };
  }

  /**
   * @param {Float32Array} frames flat (F,22): [t, head pos3 quat4, L, R]
   * @param {Array} noteEvents  writer dicts (see score.noteEvents)
   * @param {object} info  metadata (songHash/songName/mapper/difficulty/mode/...)
   * @param {object} opts  {fps, songDuration, wallEvents, heightEvents}
   * @returns {Uint8Array}
   */
  function writeBsor(frames, noteEvents, info, opts) {
    opts = opts || {};
    const fps = opts.fps || 90;
    const songDuration = opts.songDuration || 0.0;
    const wallEvents = opts.wallEvents || [];
    const heightEvents = opts.heightEvents || null;
    const F = frames.length / 22;

    const w = makeWriter();
    w.i32(BSOR_MAGIC);
    w.u8(1);                                   // file version

    // ---- info section (magic 0) ----
    w.u8(0);
    w.str(info.version || '1.37.0_1');
    w.str(info.gameVersion || '1.37.0');
    w.str(info.timestamp || '1700000000');
    w.str(info.playerId || IDENTITY.playerId);
    w.str(info.playerName || IDENTITY.playerName);
    w.str(info.platform || IDENTITY.platform);
    w.str(info.trackingSystem || IDENTITY.trackingSystem);
    w.str(info.hmd || IDENTITY.hmd);
    w.str(info.controller || IDENTITY.controller);
    w.str(info.songHash || '');
    w.str(info.songName || 'Unknown');
    w.str(info.mapper || 'Unknown');
    w.str(info.difficulty || 'ExpertPlus');
    w.i32(info.score || 0);
    w.str(info.mode || 'SoloStandard');
    w.str(info.environment || info._env || 'DefaultEnvironment');
    w.str(info.modifiers || '');
    w.f32(info.jumpDistance === undefined ? 18.0 : info.jumpDistance);
    w.bool(!!info.leftHanded);
    w.f32(info.height === undefined ? 1.5 : info.height);
    w.f32(0.0);                                // startTime
    w.f32(songDuration);                       // failTime
    w.f32(0.0);                                // speed

    // ---- frames section (magic 1) ----
    w.u8(1);
    w.i32(F);
    for (let i = 0; i < F; i++) {
      const b = i * 22;
      w.f32(frames[b]);
      w.i32(fps);
      for (let k = 1; k <= 21; k++) w.f32(frames[b + k]);
    }

    // ---- notes section (magic 2) ----
    w.u8(2);
    w.i32(noteEvents.length);
    for (const ev of noteEvents) {
      const stype = ev.scoring === undefined ? CN.map.NOTE_SCORE_TYPE_NORMAL : ev.scoring;
      const nid = stype * 10000 + ev.line * 1000 + ev.layer * 100
        + (ev.sabertype || 0) * 10 + (ev.cutdir || 0);
      w.i32(nid);
      w.f32(ev.t);
      w.f32(Math.max(0.0, ev.t - 1.2));
      const et = ev.event || 0;
      w.i32(et);
      if (et === NOTE_EVENT_GOOD || et === NOTE_EVENT_BAD) {
        const c = ev.cut || {};
        w.bool(c.speedOK === undefined ? true : c.speedOK);
        w.bool(c.directionOk === undefined ? true : c.directionOk);
        w.bool(c.saberTypeOk === undefined ? true : c.saberTypeOk);
        w.bool(c.wasCutTooSoon === undefined ? false : c.wasCutTooSoon);
        w.f32(c.saberSpeed === undefined ? 10.0 : c.saberSpeed);
        w.f3(c.saberDirection || [0.0, 0.0, 1.0]);
        w.i32(c.saberType || 0);
        w.f32(c.timeDeviation || 0.0);
        w.f32(c.cutDeviation || 0.0);
        w.f3(c.cutPoint || [0.0, 0.0, 0.0]);
        w.f3(c.cutNormal || [0.0, 0.0, 1.0]);
        w.f32(c.cutDistanceToCenter || 0.0);
        w.f32(c.cutAngle === undefined ? 90.0 : c.cutAngle);
        w.f32(c.beforeCutRating === undefined ? 0.9 : c.beforeCutRating);
        w.f32(c.afterCutRating === undefined ? 0.9 : c.afterCutRating);
      }
    }

    // ---- walls (magic 3) ----
    w.u8(3);
    w.i32(wallEvents.length);
    for (const x of wallEvents) {
      w.i32(x.id || 0);
      w.f32(x.energy === undefined ? 0.0 : x.energy);
      w.f32(x.time === undefined ? 0.0 : x.time);
      w.f32(x.spawnTime === undefined ? Math.max(0.0, (x.time || 0.0) - 1.2) : x.spawnTime);
    }

    // ---- heights (magic 4) ----
    const heights = heightEvents || [];
    w.u8(4);
    w.i32(heights.length);
    for (const h of heights) {
      w.f32(h.height === undefined ? 1.5 : h.height);
      w.f32(h.time === undefined ? 0.0 : h.time);
    }

    // ---- pauses (magic 5) ----
    w.u8(5);
    w.i32(0);

    // ---- controller offsets (magic 6): two VRObjects, NO count prefix ----
    w.u8(6);
    for (let i = 0; i < 2; i++) for (let k = 0; k < 7; k++) w.f32(0.0);

    // ---- user data (magic 7) ----
    w.u8(7);
    w.i32(0);

    return w.out();
  }

  CN.bsor = { BSOR_MAGIC, IDENTITY, writeBsor, NOTE_EVENT_GOOD, NOTE_EVENT_BAD, NOTE_EVENT_MISS, NOTE_EVENT_BOMB };
})(typeof globalThis !== 'undefined' ? globalThis : self);
