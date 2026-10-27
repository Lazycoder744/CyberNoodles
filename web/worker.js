/* worker.js — the whole CyberNoodles pipeline, running client-side.
 *
 * This is a classic worker (importScripts), so it can pull in onnxruntime-web
 * and the lib/ modules without a bundler and without COOP/COEP headers.
 *
 * Message protocol (postMessage):
 *   <- {type:'init'}                                   load model + create session
 *   -> {type:'ready', modelBytes}
 *   -> {type:'error', message}
 *
 *   <- {type:'lookup', bsr}
 *   -> {type:'map', meta, versions:[{hash, downloadURL, diffs:[...]}]}
 *
 *   <- {type:'run', bsr, hash, downloadURL, difficulty}
 *   -> {type:'progress', phase, done, total, detail}
 *   -> {type:'result', bsor:ArrayBuffer, accuracy, frames, events, meta}
 *
 * Nothing here talks to a server beyond BeatSaver (CORS `*`) and whatever host
 * you point config.modelUrl at.
 */
'use strict';

/* global CN, importScripts, self */

const CN = (self.CN = self.CN || {});

self.onmessage = (ev) => {
  const msg = ev.data || {};
  const job = { id: msg.id };
  const post = (m) => self.postMessage(Object.assign({ id: job.id }, m));

  if (msg.type === 'init') {
    handleInit(msg, post).catch((e) => post({ type: 'error', message: String(e && e.message || e) }));
  } else if (msg.type === 'lookup') {
    handleLookup(msg, post).catch((e) => post({ type: 'error', message: String(e && e.message || e) }));
  } else if (msg.type === 'run') {
    handleRun(msg, post).catch((e) => post({ type: 'error', message: String(e && e.message || e), stack: e && e.stack }));
  }
};

function loadScripts() {
  if (self.__cnLoaded) return;
  // config.js first: it defines CN.config.ortScript.
  importScripts('lib/config.js');
  importScripts(CN.config.ortScript);
  for (const f of ['lib/util.js', 'lib/zip.js', 'lib/map.js',
    'lib/encode.js', 'lib/model.js', 'lib/score.js', 'lib/avscore.js', 'lib/bsor.js']) {
    importScripts(f);
  }
  self.__cnLoaded = true;
}

let SESSION = null;
let MODEL_BYTES = 0;
let INPUTS = null;
let OUTPUTS = null;

async function handleInit(msg, post) {
  loadScripts();
  if (SESSION) {
    post({
      type: 'ready', modelBytes: MODEL_BYTES, inputs: INPUTS, outputs: OUTPUTS,
      cached: true, ep: SESSION.ep, threads: SESSION.threads,
    });
    return;
  }
  if (msg.modelUrl) CN.config.modelUrl = msg.modelUrl;
  if (msg.ep) CN.config.executionProvider = String(msg.ep).toLowerCase();

  post({ type: 'progress', phase: 'fetch-model', done: 0, total: 0, detail: CN.config.modelUrl });
  const bytes = await CN.model.fetchBytes(CN.config.modelUrl, (got, total) => {
    post({ type: 'progress', phase: 'fetch-model', done: got, total });
  });
  MODEL_BYTES = bytes.length;
  post({ type: 'progress', phase: 'create-session', done: 0, total: 0, detail: `${(bytes.length / 1048576).toFixed(1)} MiB` });
  SESSION = await CN.model.createSession(self.ort, bytes,
    (line) => post({ type: 'progress', phase: 'create-session', done: 0, total: 0, detail: line }));
  INPUTS = SESSION.inputNames;
  OUTPUTS = SESSION.outputNames;
  post({
    type: 'ready', modelBytes: MODEL_BYTES, inputs: INPUTS, outputs: OUTPUTS,
    ep: SESSION.ep, threads: SESSION.threads,
  });
}

async function fetchJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (r.status === 404) throw new Error(`BeatSaver has no map with that ID (${url})`);
  if (!r.ok) throw new Error(`BeatSaver lookup failed: HTTP ${r.status}`);
  return r.json();
}

function standardDiffs(version) {
  return (version.diffs || []).filter((d) => String(d.characteristic || '').toLowerCase() === 'standard');
}

async function handleLookup(msg, post) {
  loadScripts();
  const bsr = String(msg.bsr || '').trim().toLowerCase();
  if (!bsr) throw new Error('empty map ID');
  const api = await fetchJson(CN.config.beatSaverApi + encodeURIComponent(bsr));
  const versions = (api.versions || []).map((v) => ({
    hash: v.hash, downloadURL: v.downloadURL, state: v.state,
    diffs: standardDiffs(v),
  })).filter((v) => v.diffs.length);
  if (!versions.length) throw new Error('that map has no Standard-characteristic difficulties');
  const meta = api.metadata || {};
  post({
    type: 'map',
    meta: {
      bsr, id: api.id, name: api.name, songName: meta.songName, songAuthor: meta.songAuthorName,
      mapper: meta.levelAuthorName, bpm: meta.bpm, duration: meta.duration,
      uploader: (api.uploader || {}).name,
    },
    api,
    versions,
  });
}

async function handleRun(msg, post) {
  loadScripts();
  if (!SESSION) throw new Error('model not loaded');

  const difficulty = msg.difficulty || 'ExpertPlus';
  let downloadURL = msg.downloadURL;
  let hash = msg.hash;

  // If the caller did not pass a resolved version, look it up now.
  if (!downloadURL) {
    const api = await fetchJson(CN.config.beatSaverApi + encodeURIComponent(String(msg.bsr).toLowerCase()));
    let best = null;
    for (const v of (api.versions || [])) {
      const ok = standardDiffs(v).some((d) => d.difficulty === difficulty);
      if (ok) { best = v; break; }
    }
    if (!best) best = (api.versions || [])[0];
    if (!best) throw new Error('no downloadable version for that map');
    downloadURL = best.downloadURL;
    hash = best.hash;
  }
  if (!hash) throw new Error('missing song hash');

  post({ type: 'progress', phase: 'download-map', done: 0, total: 0, detail: 'BeatSaver CDN' });
  const zipResp = await fetch(downloadURL);
  if (!zipResp.ok) throw new Error(`map download failed: HTTP ${zipResp.status}`);
  const zipBuf = await zipResp.arrayBuffer();
  post({ type: 'progress', phase: 'parsing-map', done: 0, total: 0, detail: `${(zipBuf.byteLength / 1048576).toFixed(1)} MiB zip` });

  const files = await CN.zip.readZip(zipBuf);
  const apiJson = msg.apiJson || null;
  const md = CN.map.parseMap(files, hash, difficulty, 'Standard', apiJson);
  const nNotes = md.notes.length / 5;
  if (!nNotes) throw new Error('the chosen difficulty has no notes (wrong characteristic?)');

  const fps = CN.encode.dynamicFps(md);
  post({
    type: 'progress', phase: 'rollout', done: 0, total: 0,
    detail: `${nNotes} notes, ${md.duration.toFixed(1)} s, ${fps} fps`,
  });

  const traj = await CN.model.rollout({
    ort: self.ort, sess: SESSION, md, fps,
    overlap: CN.config.overlap, batch: CN.config.batch, histFrames: CN.config.histFrames,
    onProgress: (line) => post({ type: 'progress', phase: 'rollout', done: 0, total: 0, detail: line }),
  });
  if (!traj.length) throw new Error('rollout produced no frames');

  post({ type: 'progress', phase: 'scoring', done: 0, total: 0, detail: 'reconstructing cuts' });
  const det = CN.score.reconstruct(traj, md);
  const events = CN.score.noteEvents(det);
  for (const b of CN.score.bombEvents(traj, md)) {
    events.push({ t: b.t, line: b.line, layer: b.layer, sabertype: 0, cutdir: 0, event: 3 });
  }
  events.sort((a, b) => a.t - b.t);
  const walls = CN.score.detectWallHits(traj, md);
  const acc = CN.avscore.accuracy(events, walls);

  const info = {
    version: '0.5.1', gameVersion: '1.37.0',
    timestamp: String(Math.floor(Date.now() / 1000)),
    playerId: CN.bsor.IDENTITY.playerId, playerName: CN.bsor.IDENTITY.playerName,
    platform: CN.bsor.IDENTITY.platform, trackingSystem: CN.bsor.IDENTITY.trackingSystem,
    hmd: CN.bsor.IDENTITY.hmd, controller: CN.bsor.IDENTITY.controller,
    songHash: String(hash).toUpperCase(), songName: md.songName, mapper: md.mapper,
    difficulty, score: Math.round(acc.score), mode: 'Standard',
    environment: md.environment, modifiers: '', jumpDistance: 18.0,
    leftHanded: false, height: 1.5,
  };

  const bytes = CN.bsor.writeBsor(traj, events, info, {
    fps: Math.round(fps), songDuration: md.duration, wallEvents: walls,
    heightEvents: [{ height: 1.5, time: 0.0 }],
  });
  // writeBsor() returns an exactly-sized Uint8Array (byteOffset 0), so its
  // buffer can be transferred as-is; slice() here would be a second full
  // copy of a multi-MB file on every run.
  const out = (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength)
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

  self.postMessage({
    id: msg.id,
    type: 'result',
    bsor: out,
    accuracy: { pct: acc.pct, score: acc.score, max: acc.max, good: acc.good, miss: acc.miss },
    frames: traj.length / 22,
    events: events.length,
    walls: walls.length,
    meta: {
      bsr: msg.bsr, hash: String(hash).toUpperCase(), difficulty,
      songName: md.songName, mapper: md.mapper, bpm: md.bpm,
      njs: md.njs, fps, notes: nNotes, bombs: md.bombs.length / 3,
      walls: md.walls.length / 6, chains: md.chains.length / 8,
      duration: md.duration, bytes: out.byteLength,
    },
  }, [out]);
}
