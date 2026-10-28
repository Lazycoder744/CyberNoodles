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
 *   <- {type:'cancel'}                                 abort the job in flight
 *   -> {type:'cancelled'}                              (echoes the job's id)
 *
 *   <- {type:'unload'}                                 release the session (model switch)
 *   -> {type:'unloaded'}
 *   <- {type:'dispose'}                                page is closing: abort, then release
 *   -> {type:'log', line, cls}                         diagnostics for the page log
 *
 * Nothing here talks to a server beyond BeatSaver (CORS `*`) and whatever host
 * you point config.modelUrl at.
 */
'use strict';

/* global CN, importScripts, self */

const CN = (self.CN = self.CN || {});

let MODEL_LATENT = 0;    // style-latent width read from the .onnx.json sidecar
let CANCELLED = false;   // set by a 'cancel' message; cleared when the job settles
let IN_FLIGHT = false;   // one job at a time (the page serializes them)
let DISPOSE = false;     // the page is closing: free the session once nothing is running

self.onmessage = (ev) => {
  const msg = ev.data || {};
  const job = { id: msg.id };
  const post = (m) => self.postMessage(Object.assign({ id: job.id }, m));

  if (msg.type === 'cancel') {
    CANCELLED = true;
    // The next batch run rejects, so a rollout stops at its next await. If
    // nothing is running, confirm straight away — otherwise the page would
    // wait for a reply that would never come.
    if (!IN_FLIGHT) {
      CANCELLED = false;
      post({ type: 'cancelled' });
    }
    return;
  }

  if (msg.type === 'dispose') {
    // The tab is going away. Stop at the next batch and hand the GPU session
    // back cleanly instead of letting Chrome tear it down mid-inference.
    DISPOSE = true;
    if (IN_FLIGHT) CANCELLED = true;
    else releaseSession();
    return;
  }

  IN_FLIGHT = true;
  let p;
  if (msg.type === 'init') {
    p = handleInit(msg, post);
  } else if (msg.type === 'unload') {
    p = handleUnload(msg, post);
  } else if (msg.type === 'lookup') {
    p = handleLookup(msg, post);
  } else if (msg.type === 'run') {
    p = handleRun(msg, post);
  } else {
    IN_FLIGHT = false;
    return;
  }

  p.then(
    () => {
      IN_FLIGHT = false;
      CANCELLED = false;
      if (DISPOSE) releaseSession();
    },
    (e) => {
      const cancelled = CANCELLED;
      IN_FLIGHT = false;
      CANCELLED = false;
      if (DISPOSE) releaseSession();
      if (cancelled) {
        post({ type: 'cancelled' });
      } else if (msg.type === 'run') {
        post({ type: 'error', message: String(e && e.message || e), stack: e && e.stack });
      } else {
        post({ type: 'error', message: String(e && e.message || e) });
      }
    },
  );
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
let SESSION_KEY = '';
let MODEL_BYTES = 0;
let INPUTS = null;
let OUTPUTS = null;

async function releaseSession() {
  const s = SESSION;
  SESSION = null;
  SESSION_KEY = '';
  INPUTS = null;
  OUTPUTS = null;
  MODEL_LATENT = 0;
  if (s && s.raw && typeof s.raw.release === 'function') {
    try { await s.raw.release(); } catch (e) { /* already gone */ }
  }
}

async function handleUnload(msg, post) {
  await releaseSession();
  post({ type: 'unloaded' });
}

async function handleInit(msg, post) {
  loadScripts();
  const url = msg.modelUrl || CN.config.modelUrl;
  const fp16 = !!msg.fp16;
  const key = url + '|' + (fp16 ? 'fp16' : 'fp32');
  if (SESSION && SESSION_KEY === key) {
    post({
      type: 'ready', modelBytes: MODEL_BYTES, inputs: INPUTS, outputs: OUTPUTS,
      cached: true, ep: SESSION.ep, threads: SESSION.threads, fp16,
      latent: MODEL_LATENT,
    });
    return;
  }
  if (SESSION) await releaseSession();   // a different model was asked for
  CN.config.modelUrl = url;
  if (msg.ep) CN.config.executionProvider = String(msg.ep).toLowerCase();
  if (fp16) await CN.model.assertFp16Ready(CN.config.executionProvider);   // before the download

  const parts = Array.isArray(CN.config.modelUrl)
    ? CN.config.modelUrl.length : 1;
  post({ type: 'progress', phase: 'fetch-model', done: 0, total: 0,
         detail: parts > 1 ? `${CN.config.modelUrl[0]} (+${parts - 1} part)` : CN.config.modelUrl });
  const bytes = await CN.model.fetchBytes(CN.config.modelUrl, (got, total, info) => {
    post({ type: 'progress', phase: 'fetch-model', done: got, total,
           detail: info && info.parts > 1 ? `part ${info.part}/${info.parts}` : '' });
  });
  // The sidecar describes the style latent (0 for v1 checkpoints).  Optional:
  // an old model without it simply has no temperature knob.
  MODEL_LATENT = 0;
  try {
    const metaUrl = (Array.isArray(CN.config.modelUrl)
      ? CN.config.modelUrl[0] : CN.config.modelUrl) + '.json';
    const mr = await fetch(metaUrl, { cache: 'no-store' });
    if (mr.ok) {
      const meta = await mr.json();
      MODEL_LATENT = Number(meta && meta.latent_dim) || 0;
    }
  } catch (e) { /* no sidecar: fine */ }
  MODEL_BYTES = bytes.length;
  post({ type: 'progress', phase: 'create-session', done: 0, total: 0, detail: `${(bytes.length / 1048576).toFixed(1)} MiB` });
  const session = await CN.model.createSession(self.ort, bytes,
    (line) => post({ type: 'progress', phase: 'create-session', done: 0, total: 0, detail: line }),
    { fp16, log: (line, cls) => post({ type: 'log', line, cls }) });
  SESSION = session;
  SESSION_KEY = key;

  // Cancellation hook: make the next batch reject so a cancelled rollout
  // stops at its next await instead of grinding through every remaining
  // window.  This wraps the session facade only — the runtime, its inputs
  // and its outputs are untouched.
  const rawRun = SESSION.run;
  SESSION.run = (feeds) => (CANCELLED
    ? Promise.reject(new Error('cancelled'))
    : rawRun.call(SESSION, feeds));

  INPUTS = SESSION.inputNames;
  OUTPUTS = SESSION.outputNames;
  post({
    type: 'ready', modelBytes: MODEL_BYTES, inputs: INPUTS, outputs: OUTPUTS,
    ep: SESSION.ep, threads: SESSION.threads, fp16, latent: MODEL_LATENT,
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
    latent: MODEL_LATENT, temp: Number(msg.temp) || 0, seed: Number(msg.seed) || 0,
    onProgress: (line) => post({ type: 'progress', phase: 'rollout', done: 0, total: 0, detail: line }),
  });
  if (!traj.length) throw new Error('rollout produced no frames');
  if (CANCELLED) {
    post({ type: 'cancelled' });
    return;
  }

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
  if (CANCELLED) {
    post({ type: 'cancelled' });
    return;
  }

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
