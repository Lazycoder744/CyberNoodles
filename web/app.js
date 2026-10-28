/* app.js — the page side of the CyberNoodles web app.
 *
 * All the work happens in worker.js; this file only talks to it and renders
 * progress.  Nothing is uploaded anywhere: the map comes from BeatSaver and
 * the replay is written in the browser.
 */
'use strict';

/* global CN */

const el = (id) => document.getElementById(id);

const ui = {
  bsr: el('bsr'),
  lookup: el('lookup'),
  diff: el('diff'),
  run: el('run'),
  status: el('status'),
  log: el('log'),
  result: el('result'),
  modelState: el('model-state'),
  bar: el('bar'),
  barFill: el('bar-fill'),
  modelRow: el('model-row'),
  modelPath: el('model-path'),
  loadModel: el('load-model'),
  cancel: el('cancel'),
  precision: el('precision'),
  temp: el('temp'),
  tempVal: el('temp-val'),
};

let worker = null;
let ready = false;
let pending = new Map();
let nextId = 1;
let lastLookup = null;
let lastResult = null;
let lastUrl = null;

let bootModelUrl = null;
let bootEp = '';
let precision = 'fp32';   // 'fp16' | 'fp32', the dropdown at the top
let running = false;   // a generation is in flight (cancel visible, unload guarded)

function beforeUnload(e) {
  e.preventDefault();
  e.returnValue = '';
}

function guardUnload(on) {
  // While inference runs, confirm before the tab closes. Tearing a WebGPU
  // worker down mid-inference can crash Chrome's GPU process, so an
  // accidental close is worth one confirmation dialog.
  if (on) window.addEventListener('beforeunload', beforeUnload);
  else window.removeEventListener('beforeunload', beforeUnload);
}

const MODEL_SIZE = { fp16: '≈67 MiB', fp32: '≈134 MiB' };

// ---- v2 style temperature ------------------------------------------------
// z is a 16-d latent learned from human replay history (a beta-VAE style
// latent).  temp=0 is the deterministic policy; temp>0 samples one coherent
// style for the whole replay, so the same map produces different but
// human-plausible runs.  The knob is inert on models without a style latent.
function savedTemp() {
  try {
    const v = parseFloat(localStorage.getItem('cn.temp'));
    return Number.isFinite(v) ? Math.max(0, Math.min(2, v)) : 0;
  } catch { return 0; }
}

function saveTemp(v) {
  try { localStorage.setItem('cn.temp', String(v)); } catch { /* storage unavailable */ }
}

function applyTemp() {
  const v = parseFloat(ui.temp.value) || 0;
  if (ui.tempVal) ui.tempVal.textContent = v.toFixed(1);
  saveTemp(v);
}

function savedPrecision() {
  try {
    const v = localStorage.getItem('cn.precision');
    return v === 'fp16' || v === 'fp32' ? v : null;
  } catch { return null; }
}

function savePrecision(v) {
  try { localStorage.setItem('cn.precision', v); } catch { /* storage unavailable */ }
}

// The dropdown stays locked while a model loads or a replay runs, and when
// ?model= points the page at a custom file.
function setPrecisionEnabled(on) {
  ui.precision.disabled = bootModelUrl ? true : !on;
}

function applyPrecision() {
  if (!bootModelUrl) CN.config.modelUrl = CN.config.modelUrls[precision];
  ui.modelPath.textContent = Array.isArray(CN.config.modelUrl)
    ? CN.config.modelUrl.join(' + ')
    : CN.config.modelUrl;
  ui.loadModel.textContent = bootModelUrl ? 'Download model' : `Download model (${MODEL_SIZE[precision]})`;
}

function log(line, cls) {
  const div = document.createElement('div');
  div.className = 'log-line' + (cls ? ' ' + cls : '');
  div.textContent = line;
  ui.log.appendChild(div);
  ui.log.scrollTop = ui.log.scrollHeight;
}

function setStatus(text, cls) {
  ui.status.textContent = text || '';
  ui.status.className = 'status' + (cls ? ' ' + cls : '');
}

function call(type, payload) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage(Object.assign({ type, id }, payload || {}));
  });
}

function onMessage(ev) {
  const m = ev.data || {};
  if (m.type === 'progress') {
    renderProgress(m);
    return;
  }
  if (m.type === 'log') {
    log(m.line, m.cls);
    return;
  }
  if (m.type === 'cancelled') {
    // The worker rejected the in-flight job; surface it through the same
    // promise so run()/loadModel() can reset the UI in one place.
    const p = pending.get(m.id);
    if (p) {
      pending.delete(m.id);
      p.reject(new Error('cancelled'));
    }
    return;
  }
  const p = pending.get(m.id);
  if (!p) return;
  if (m.type === 'error') {
    pending.delete(m.id);
    p.reject(new Error(m.message));
    return;
  }
  pending.delete(m.id);
  p.resolve(m);
}

const PHASE_LABEL = {
  'fetch-model': 'Downloading model',
  'create-session': 'Building inference session',
  'download-map': 'Downloading map',
  'parsing-map': 'Parsing map',
  rollout: 'Running the model',
  scoring: 'Scoring',
};

function renderProgress(m) {
  const label = PHASE_LABEL[m.phase] || m.phase;
  if (m.total > 0) {
    const pc = 100 * m.done / m.total;
    ui.bar.hidden = false;
    ui.barFill.style.width = pc.toFixed(1) + '%';
    setStatus(`${label}… ${pc.toFixed(0)}% (${(m.done / 1048576).toFixed(1)} / ${(m.total / 1048576).toFixed(1)} MiB)`);
  } else {
    ui.bar.hidden = false;
    ui.barFill.style.width = '100%';
    ui.barFill.classList.add('indeterminate');
    setStatus(`${label}…${m.detail ? ' ' + m.detail : ''}`);
  }
}

function endProgress() {
  ui.bar.hidden = true;
  ui.barFill.classList.remove('indeterminate');
  ui.barFill.style.width = '0%';
}

function boot() {
  const params = new URLSearchParams(location.search);
  bootModelUrl = params.get('model');
  const forced = (params.get('precision') || '').toLowerCase();
  precision = (forced === 'fp16' || forced === 'fp32') ? forced : (savedPrecision() || 'fp32');
  if (bootModelUrl) CN.config.modelUrl = bootModelUrl;
  ui.precision.value = precision;
  setPrecisionEnabled(true);
  let ep = (params.get('ep') || '').toLowerCase();
  if (ep && ['auto', 'webgpu', 'wasm'].indexOf(ep) < 0) {
    log(`ignoring ?ep=${ep} (expected auto, webgpu or wasm)`, 'hint');
    ep = '';
  }
  bootEp = ep;

  if (!window.Worker) {
    setStatus('This browser has no Web Worker support.', 'bad');
    return;
  }

  worker = new Worker('worker.js');
  worker.onmessage = onMessage;
  worker.onerror = (e) => {
    log('worker error: ' + (e.message || e), 'bad');
    setStatus('Worker crashed: ' + (e.message || 'unknown'), 'bad');
  };

  // Best-effort teardown on leave: the worker aborts at the next batch and
  // releases the GPU session instead of dying with work in flight. Harmless
  // when it loses the race with a hard tab close. A page that goes into the
  // back/forward cache (persisted) is coming back, so it keeps its session.
  window.addEventListener('pagehide', (e) => {
    if (e.persisted || !worker) return;
    try { worker.postMessage({ type: 'dispose' }); } catch {}
  });

  // Nothing is fetched until the user asks for it.
  ui.temp.value = String(savedTemp());
  applyTemp();
  applyPrecision();
  setStatus('Load the model to begin.');
  log(`model: ${CN.config.modelUrl} (${precision.toUpperCase()}; download starts on your click)`);
  log(`onnxruntime-web ${CN.config.ortVersion} from ${CN.config.ortBase}`);
  log(`execution provider: ${bootEp || 'auto (WebGPU if available)'}`);
}

async function loadModel() {
  if (!worker) return;
  ui.loadModel.disabled = true;
  ui.loadModel.textContent = 'Loading…';
  setPrecisionEnabled(false);
  setStatus('Loading the model…');
  try {
    const r = await call('init', {
      modelUrl: CN.config.modelUrl, fp16: precision === 'fp16', ep: bootEp || null,
    });
    ready = true;
    endProgress();
    const webgpu = r.ep === 'webgpu';
    const backend = webgpu
      ? 'WebGPU'
      : `WASM, ${r.threads} thread${r.threads === 1 ? '' : 's'}`;
    ui.modelState.textContent = `${CN.config.modelUrl} · ${backend}`;
    setStatus(`Model ready (${backend}${webgpu ? '' : ', legacy fallback'}). Enter a BeatSaver ID.`, webgpu ? 'ok' : 'warn');
    log(`model loaded (${precision.toUpperCase()}): ${(r.modelBytes / 1048576).toFixed(1)} MiB; inputs [${r.inputs.join(', ')}]`);
    log(`backend: ${backend}`);
    if (!webgpu) {
      log('Legacy WASM fallback: inference is roughly 30× slower, and last-bit '
        + 'float32 drift can slightly change the score. Use a WebGPU-capable browser '
        + 'for the intended result.', 'hint');
    }
    ui.modelRow.hidden = true;
    ui.lookup.disabled = false;
    ui.run.disabled = !lastLookup;
    setPrecisionEnabled(true);
    ui.bsr.focus();
  } catch (e) {
    endProgress();
    if (e.message === 'cancelled') {
      setStatus('Model load cancelled.', 'warn');
      log('model load cancelled', 'hint');
    } else {
      setStatus('Could not load the model: ' + e.message, 'bad');
      log('model load failed: ' + e.message, 'bad');
      if (/fetch/i.test(e.message)) {
        log('If the file is not in web/models/, add ?model=<url> to the page URL '
          + '(a CORS-enabled host such as a Hugging Face resolve/main link works).', 'hint');
      }
    }
    setPrecisionEnabled(true);
    ui.loadModel.disabled = false;
    ui.loadModel.textContent = 'Retry download';
  }
}

async function switchPrecision() {
  const next = ui.precision.value;
  if (next === precision) return;
  if (running) { ui.precision.value = precision; return; }
  precision = next;
  savePrecision(next);
  if (ready) {
    // Different weights: let go of the loaded session, then offer the new download.
    setPrecisionEnabled(false);
    ui.lookup.disabled = true;
    ui.run.disabled = true;
    try { await call('unload'); } catch (e) { log('unload failed: ' + e.message, 'bad'); }
    ready = false;
    ui.modelState.textContent = '…';
    ui.modelRow.hidden = false;
    setPrecisionEnabled(true);
  }
  applyPrecision();
  ui.loadModel.disabled = false;
  setStatus('Load the model to begin.');
  log(`switched to ${next.toUpperCase()}: ${CN.config.modelUrl}`);
}

async function lookup() {
  const bsr = ui.bsr.value.trim();
  ui.lookup.disabled = true;
  ui.run.disabled = true;
  ui.result.hidden = true;
  lastLookup = null;
  try {
    setStatus('Looking up ' + bsr + ' on BeatSaver…');
    const r = await call('lookup', { bsr });
    lastLookup = r;
    ui.diff.innerHTML = '';
    const v = r.versions[0];
    for (const d of v.diffs) {
      const o = document.createElement('option');
      o.value = d.difficulty;
      const bits = [`${d.notes} notes`];
      if (d.length) bits.push(`${Math.floor(d.length / 60)}:${String(Math.round(d.length % 60)).padStart(2, '0')}`);
      if (d.njs) bits.push(`NJS ${d.njs}`);
      if (d.nps) bits.push(`${d.nps.toFixed(1)} nps`);
      o.textContent = `${d.difficulty} · ${bits.join(' · ')}`;
      ui.diff.appendChild(o);
    }
    const prefer = ['ExpertPlus', 'Expert', 'Hard', 'Normal', 'Easy'];
    for (const p of prefer) {
      if (v.diffs.some((d) => d.difficulty === p)) { ui.diff.value = p; break; }
    }
    ui.diff.disabled = false;
    ui.run.disabled = false;
    setStatus(`${r.meta.songName} · ${r.meta.songAuthor} (mapped by ${r.meta.mapper})`, 'ok');
    log(`found ${r.meta.songName} [${r.meta.bpm} BPM], ${v.diffs.length} Standard difficulties`);
    ui.run.focus();
  } catch (e) {
    setStatus(e.message, 'bad');
    log('lookup failed: ' + e.message, 'bad');
  } finally {
    ui.lookup.disabled = false;
  }
}

async function run() {
  if (!ready || !lastLookup) return;
  ui.run.disabled = true;
  ui.lookup.disabled = true;
  ui.result.hidden = true;
  ui.cancel.hidden = false;
  ui.cancel.disabled = false;
  running = true;
  setPrecisionEnabled(false);
  guardUnload(true);
  const difficulty = ui.diff.value;
  try {
    log(`generating ${difficulty} replay…`);
    const temp = parseFloat(ui.temp.value) || 0;
    // One style sample per run: a fresh seed gives a different (still
    // human-trained) variation each time, and the seed is logged so a run can
    // be reproduced with ?seed= on the next version of this page.
    const seed = (Math.random() * 0x7fffffff) | 0;
    if (temp > 0) log(`style temperature ${temp.toFixed(1)} (seed ${seed})`);
    const r = await call('run', {
      bsr: lastLookup.meta.bsr,
      hash: lastLookup.versions[0].hash,
      downloadURL: lastLookup.versions[0].downloadURL,
      apiJson: lastLookup.api,
      difficulty,
      temp,
      seed,
    });
    lastResult = r;
    renderResult(r);
  } catch (e) {
    if (e.message === 'cancelled') {
      setStatus('Cancelled — the session stays loaded, generate again any time.', 'warn');
      log('generation cancelled', 'hint');
    } else {
      setStatus('Generation failed: ' + e.message, 'bad');
      log('generation failed: ' + e.message, 'bad');
    }
  } finally {
    running = false;
    setPrecisionEnabled(true);
    guardUnload(false);
    endProgress();
    ui.cancel.hidden = true;
    ui.run.disabled = false;
    ui.lookup.disabled = false;
  }
}

function cancelRun() {
  if (!running || !worker) return;
  ui.cancel.disabled = true;
  setStatus('Cancelling…');
  try { worker.postMessage({ type: 'cancel' }); } catch {}
}

function safeName(s) {
  return String(s || 'replay').replace(/[^\w.\-]+/g, '_').slice(0, 60);
}

function codeEl(text) {
  const c = document.createElement('code');
  c.textContent = text;
  return c;
}

function renderResult(r) {
  const m = r.meta;
  const acc = r.accuracy;
  const cls = acc.pct >= 95 ? 'ok' : (acc.pct >= 85 ? 'warn' : 'bad');

  // Regenerating replaces the old result; drop its blob URL so the bytes
  // are not pinned in memory for the rest of the session.
  if (lastUrl) {
    URL.revokeObjectURL(lastUrl);
    lastUrl = null;
  }
  const blob = new Blob([r.bsor], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  lastUrl = url;
  const fname = `${safeName(m.songName)}-${safeName(m.difficulty)}.bsor`;

  ui.result.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'score ' + cls;
  head.textContent = `${acc.pct.toFixed(2)}%`;
  ui.result.appendChild(head);

  const sub = document.createElement('div');
  sub.className = 'score-sub';
  sub.textContent = `${acc.score.toLocaleString()} / ${acc.max.toLocaleString()} points `
    + `· ${acc.good} hits · ${acc.miss} misses`;
  ui.result.appendChild(sub);

  const table = document.createElement('dl');
  const rows = [
    ['Song', `${m.songName} · ${m.mapper}`],
    ['Difficulty', `${m.difficulty} · ${m.notes} notes · NJS ${m.njs}`],
    ['Map', `${m.bsr} (${m.hash.slice(0, 8)})`],
    ['Replay', `${r.frames} frames @ ${m.fps} fps · ${r.events} events · ${(r.bytes / 1048576).toFixed(2)} MiB`],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    table.appendChild(dt); table.appendChild(dd);
  }
  ui.result.appendChild(table);

  const actions = document.createElement('div');
  actions.className = 'actions';
  const dl = document.createElement('a');
  dl.href = url;
  dl.download = fname;
  dl.className = 'btn primary';
  dl.textContent = 'Download .bsor';
  actions.appendChild(dl);

  const copy = document.createElement('button');
  copy.className = 'btn';
  copy.textContent = 'Copy file name';
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(fname);
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy file name'; }, 1400);
    } catch { /* clipboard unavailable */ }
  };
  actions.appendChild(copy);
  ui.result.appendChild(actions);

  const hint = document.createElement('p');
  hint.className = 'hint';
  hint.append('ArcViewer loads the replay from a URL, so the folder you serve it '
    + 'from must send CORS headers. Serve it with ');
  hint.appendChild(codeEl('python3 bsopencode/serve_replays.py <folder> 8765'));
  hint.append(' or ');
  hint.appendChild(codeEl('npx http-server --cors -p 8765'));
  hint.append(', then open ');
  const av = document.createElement('a');
  av.href = `https://allpoland.github.io/ArcViewer/?replayURL=http://127.0.0.1:8765/${encodeURIComponent(fname)}&noProxy=true`;
  av.target = '_blank';
  av.rel = 'noopener';
  av.textContent = 'ArcViewer';
  hint.appendChild(av);
  hint.append(' with ');
  hint.appendChild(codeEl(`?replayURL=http://127.0.0.1:8765/${fname}&noProxy=true`));
  hint.append('.');
  ui.result.appendChild(hint);

  ui.result.hidden = false;
  setStatus('Done.', 'ok');
  log(`score ${acc.pct.toFixed(4)}% (${acc.score}/${acc.max}); wrote ${(r.bytes / 1048576).toFixed(2)} MiB`);
}

function main() {
  ui.lookup.disabled = true;
  ui.run.disabled = true;
  ui.diff.disabled = true;
  ui.lookup.addEventListener('click', lookup);
  ui.run.addEventListener('click', run);
  ui.loadModel.addEventListener('click', loadModel);
  ui.precision.addEventListener('change', switchPrecision);
  ui.temp.addEventListener('input', applyTemp);
  ui.cancel.addEventListener('click', cancelRun);
  ui.bsr.addEventListener('keydown', (e) => { if (e.key === 'Enter') lookup(); });
  boot();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main);
else main();
