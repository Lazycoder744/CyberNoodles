/* model.js — ONNX rollout + Tukey overlap-add, a port of
 * cybernoodles_infer.OnnxRunner / rollout / _blend / _tukey.
 *
 * Two passes, both over every 50-frame window of the map:
 *   pass 1  forward with an empty history      -> first trajectory
 *   pass 2  forward with the model's own 100-frame causal history taken from
 *           pass 1's trajectory (never ground truth)
 * then a Tukey crossfade stitches the overlapping windows into one 50 Hz-ish
 * uniform-dt pose track (traj[:, 0] = t, traj[:, 1:22] = 21-dim pose).
 *
 * `ort` is the global from onnxruntime-web; the caller supplies it so this file
 * has no hard dependency on how the runtime is loaded.
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});

  const FRAME_COUNT_OUTPUT = 50;
  const POSE = 21;
  const HIST = 100;
  const QUAT_COLS = [[3, 7], [10, 14], [17, 21]];
  const TRAJ_QUAT_COLS = [[4, 8], [11, 15], [18, 22]];

  function tukey(n, seg) {
    const w = new Float32Array(n).fill(1.0);
    if (seg <= 0 || seg >= n) return w;
    const ramp = new Float32Array(seg);
    for (let i = 0; i < seg; i++) ramp[i] = 0.5 * (1.0 - Math.cos(Math.PI * i / seg));
    for (let i = 0; i < seg; i++) w[i] = ramp[i];
    for (let i = 0; i < seg; i++) w[n - seg + i] = ramp[seg - 1 - i];
    return w;
  }

  function normalizeRows(a, off, n, stride, cols) {
    for (const [c0, c1] of cols) {
      for (let i = 0; i < n; i++) {
        let s = 0;
        const b = off + i * stride;
        for (let c = c0; c < c1; c++) s += a[b + c] * a[b + c];
        const inv = 1.0 / Math.max(Math.sqrt(s), 1e-9);
        for (let c = c0; c < c1; c++) a[b + c] *= inv;
      }
    }
  }

  function fixQuatSigns(a, off, n, stride, cols) {
    for (const [c0, c1] of cols) {
      for (let i = 1; i < n; i++) {
        let d = 0;
        const b = off + i * stride, pb = off + (i - 1) * stride;
        for (let c = c0; c < c1; c++) d += a[pb + c] * a[b + c];
        if (d < 0) for (let c = c0; c < c1; c++) a[b + c] = -a[b + c];
      }
    }
  }

  /**
   * Tukey overlap-add of K windows of 50x21 poses.
   * @returns {Float32Array} flat (F, 22): [t, head pos3 quat4, left, right]
   */
  function headBaseline(ys, K, stride, fps) {
    // Mirrors cybernoodles_infer._head_baseline: each window's frame-0 head,
    // low-passed across windows (~1.4 s).  The reference never depends on the
    // running blend, so the head cannot accumulate drift (the old running-blend
    // anchor integrated a per-window head slope and drove the head through the
    // floor), while fast window-to-window baseline disagreement -- the "head
    // swims 2-3x too fast" artifact -- is still removed.
    const kk = Math.max(3, (Math.round((1.4 * fps) / Math.max(1, stride)) | 1));
    const raw = new Float64Array(K * 3);
    for (let k = 0; k < K; k++) {
      const yoff = k * FRAME_COUNT_OUTPUT * POSE;
      raw[k * 3] = ys[yoff]; raw[k * 3 + 1] = ys[yoff + 1]; raw[k * 3 + 2] = ys[yoff + 2];
    }
    const out = new Float64Array(K * 3);
    if (K <= 1 || kk <= 1) { out.set(raw); return out; }
    const pad = kk >> 1;
    for (let k = 0; k < K; k++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = 0; j < kk; j++) {
          const idx = Math.min(K - 1, Math.max(0, k - pad + j));
          s += raw[idx * 3 + c];
        }
        out[k * 3 + c] = s / kk;
      }
    }
    return out;
  }

  function blend(anchors, ys, stride, fps) {
    const K = anchors.length;
    if (!K) return new Float32Array(0);

    // Rows are *frames*, not windows: K windows hold K*50 of them.  Passing K
    // here (the old bug) normalized only the first 48 frames -- i.e. window 0 --
    // and left every later window's quaternions non-unit, which biases every
    // crossfaded rotation slightly and made this file disagree with
    // cybernoodles_infer._blend.
    normalizeRows(ys, 0, K * FRAME_COUNT_OUTPUT, POSE, QUAT_COLS);
    for (let k = 0; k < K; k++) fixQuatSigns(ys, k * FRAME_COUNT_OUTPUT * POSE, FRAME_COUNT_OUTPUT, POSE, QUAT_COLS);

    const overlap = FRAME_COUNT_OUTPUT - stride;
    const w = tukey(FRAME_COUNT_OUTPUT, overlap);
    const nF = (K - 1) * stride + FRAME_COUNT_OUTPUT;
    const acc = new Float64Array(nF * POSE);
    const wsum = new Float64Array(nF);
    const has = new Uint8Array(nF);
    const base = new Float64Array(POSE);
    const hbase = headBaseline(ys, K, stride, fps);

    for (let k = 0; k < K; k++) {
      const yoff = k * FRAME_COUNT_OUTPUT * POSE;
      const idx0 = k * stride;
      if (has[idx0] && wsum[idx0] > 1e-6) {
        const invw = 1.0 / wsum[idx0];
        for (let c = 0; c < POSE; c++) base[c] = acc[idx0 * POSE + c] * invw;
        for (const [c0, c1] of QUAT_COLS) {
          let d = 0;
          for (let c = c0; c < c1; c++) d += ys[yoff + c] * base[c];
          if (d < 0) {
            for (let i = 0; i < FRAME_COUNT_OUTPUT; i++) {
              const b = yoff + i * POSE;
              for (let c = c0; c < c1; c++) ys[b + c] = -ys[b + c];
            }
          }
        }
      }
      // Head baseline anchoring: re-base this window's head onto the smoothed
      // reference.  Within-window head motion is untouched; only the
      // window-to-window offset is pinned.
      const dh0 = hbase[k * 3] - ys[yoff + 0];
      const dh1 = hbase[k * 3 + 1] - ys[yoff + 1];
      const dh2 = hbase[k * 3 + 2] - ys[yoff + 2];
      for (let i = 0; i < FRAME_COUNT_OUTPUT; i++) {
        const b = yoff + i * POSE;
        ys[b] += dh0; ys[b + 1] += dh1; ys[b + 2] += dh2;
      }
      for (let i = 0; i < FRAME_COUNT_OUTPUT; i++) {
        const idx = k * stride + i;
        has[idx] = 1;
        const wi = w[i];
        const yo = yoff + i * POSE, ao = idx * POSE;
        for (let c = 0; c < POSE; c++) acc[ao + c] += wi * ys[yo + c];
        wsum[idx] += wi;
      }
    }

    let lo = -1, hi = -1;
    for (let i = 0; i < nF; i++) if (wsum[i] > 0.5) { if (lo < 0) lo = i; hi = i; }
    if (lo < 0) return new Float32Array(0);

    const F = hi + 1;
    const traj = new Float32Array(F * 22);
    for (let i = 0; i < F; i++) {
      if (has[i] && wsum[i] > 0.5) {
        const inv = 1.0 / wsum[i];
        for (let c = 0; c < POSE; c++) traj[i * 22 + 1 + c] = acc[i * POSE + c] * inv;
      }
      traj[i * 22] = i / fps;
    }
    for (let i = lo - 1; i >= 0; i--) for (let c = 0; c < POSE; c++) traj[i * 22 + 1 + c] = traj[(i + 1) * 22 + 1 + c];
    for (let i = hi + 1; i < F; i++) for (let c = 0; c < POSE; c++) traj[i * 22 + 1 + c] = traj[(i - 1) * 22 + 1 + c];

    normalizeRows(traj, 0, F, 22, TRAJ_QUAT_COLS);
    fixQuatSigns(traj, 0, F, 22, TRAJ_QUAT_COLS);
    return traj;
  }

  // -------------------------------------------------------------------------
  // ONNX session
  // -------------------------------------------------------------------------
  /**
   * Look at the WebGPU adapter without creating a device.
   * @returns {Promise<{ok:boolean, reason?:string, f16?:boolean, desc?:string}>}
   */
  async function probeWebGPU() {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return { ok: false, reason: 'navigator.gpu is not exposed (WebGPU is off in this browser, or not supported on this GPU/session)' };
    }
    let adapter = null;
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch (e) {
      return { ok: false, reason: 'requestAdapter failed: ' + (e && e.message || e) };
    }
    if (!adapter) return { ok: false, reason: 'the browser returned no WebGPU adapter' };
    const info = adapter.info || {};
    const desc = [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || 'unknown adapter';
    // A software adapter (SwiftShader) "works" but is slower than WASM and
    // far more likely to fall over, so it does not count.
    if (adapter.isFallbackAdapter || info.isFallbackAdapter) {
      return { ok: false, reason: `only a software adapter is available (${desc})`, desc };
    }
    const f16 = !!(adapter.features && adapter.features.has && adapter.features.has('shader-f16'));
    return { ok: true, f16, desc };
  }

  /** FP16 needs a real WebGPU adapter with shader-f16; fail early (before any download) if not. */
  async function assertFp16Ready(want) {
    if (String(want || 'auto').toLowerCase() === 'wasm') {
      throw new Error('FP16 runs on WebGPU only. Pick FP32 to use WASM.');
    }
    const probe = await probeWebGPU();
    if (!probe.ok) throw new Error('FP16 needs WebGPU, and it is not usable here: ' + probe.reason + '. Pick FP32.');
    if (!probe.f16) throw new Error('This GPU does not support shader-f16, which FP16 needs. Pick FP32.');
    return probe;
  }

  function wrapSession(sess, ep, threads) {
    // Thin facade so the worker/UI can report which backend actually runs.
    return {
      ep,
      threads,
      inputNames: sess.inputNames,
      outputNames: sess.outputNames,
      inputMetadata: sess.inputMetadata || [],
      run: (feeds) => sess.run(feeds),
      raw: sess,
    };
  }

  /**
   * @param {object} opts  {fp16: boolean, log: (line, cls) => void}
   *   fp16 models run on WebGPU only: the WASM (CPU) backend has no usable
   *   FP16 kernels, so asking for FP16 without WebGPU is an error, not a
   *   silent fallback.
   */
  async function createSession(ort, modelBytes, onProgress, opts) {
    const report = onProgress || (() => {});
    const o = opts || {};
    const say = o.log || (() => {});
    ort.env.wasm.wasmPaths = CN.config.ortBase;
    ort.env.logLevel = 'warning';

    // Threaded WASM needs SharedArrayBuffer, which browsers only expose on
    // cross-origin-isolated pages (COOP+COEP).  GitHub Pages cannot send those
    // headers, so ordinary visitors get 1 thread; a self-hosted copy that sets
    // them gets up to 8.
    const cores = Number(typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;
    const isolated = typeof globalThis.crossOriginIsolated === 'boolean'
      && globalThis.crossOriginIsolated;
    const threads = isolated ? Math.max(1, Math.min(8, cores)) : 1;
    ort.env.wasm.numThreads = threads;

    const want = String(CN.config.executionProvider || 'auto').toLowerCase();
    const probe = want === 'wasm' ? { ok: false, reason: 'WASM was forced with ?ep=wasm' } : await probeWebGPU();
    if (probe.ok) say(`WebGPU adapter: ${probe.desc}${probe.f16 ? ' (shader-f16)' : ' (no shader-f16)'}`);
    else say('WebGPU not used: ' + probe.reason, 'hint');

    if (o.fp16) {
      await assertFp16Ready(want);
      try {
        report('creating a WebGPU FP16 session');
        const sess = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all',
        });
        return wrapSession(sess, 'webgpu', threads);
      } catch (e) {
        throw new Error('The WebGPU FP16 session could not be built: ' + (e && e.message || e) + '. Pick FP32.');
      }
    }

    let ep = 'wasm';
    if (want === 'webgpu') ep = 'webgpu';
    else if (want !== 'wasm' && probe.ok) ep = 'webgpu';

    if (ep === 'webgpu') {
      try {
        report('WebGPU adapter found; creating a WebGPU session');
        const sess = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all',
        });
        return wrapSession(sess, 'webgpu', threads);
      } catch (e) {
        // A page can expose navigator.gpu but still fail to build the session
        // (driver or kernel coverage).  WASM is kept as a legacy fallback for
        // those browsers; it is far slower and can change scores slightly.
        say('WebGPU session failed: ' + (e && e.message || e), 'bad');
        report('WebGPU session failed; using the legacy WASM fallback');
      }
    }
    report(`using the legacy WASM fallback (${threads} thread${threads === 1 ? '' : 's'}) — slower, may change scores`);
    const sess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    return wrapSession(sess, 'wasm', threads);
  }

  /** Fetch model bytes with a progress callback.
   *
   * `url` may be an array of parts: GitHub blocks files over 100 MiB, so the
   * v2 fp32 model ships as two same-origin parts and is joined in memory here.
   * The ONNX runtime only ever sees the complete byte array.
   */
  async function fetchBytes(url, onProgress) {
    const list = Array.isArray(url) ? url : [url];
    if (list.length === 1) return fetchOne(list[0], onProgress);
    const parts = [];
    let done = 0;
    for (let i = 0; i < list.length; i++) {
      const buf = await fetchOne(list[i], (got) => {
        if (onProgress) onProgress(done + got, 0, { part: i + 1, parts: list.length });
      });
      parts.push(buf);
      done += buf.length;
    }
    const out = new Uint8Array(done);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  async function fetchOne(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status} (${url})`);
    const total = Number(resp.headers.get('content-length') || 0);
    if (!resp.body || !resp.body.getReader) return new Uint8Array(await resp.arrayBuffer());
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (onProgress) onProgress(got, total);
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  /**
   * Run the model over a list of encoded windows, in batches.
   * @param {Array<{x1:Float32Array,x2:Float32Array,x3:Float32Array}>} windows
   * @param {object|null} hist  {hist, htime, hmask} for all K windows, or null
   * @returns {Float32Array} (K, 50, 21) poses
   */
  async function runWindows(ort, sess, windows, hist, batch, onBatch, z) {
    const K = windows.length;
    const out = new Float32Array(K * FRAME_COUNT_OUTPUT * POSE);
    const inNames = sess.inputNames;
    const hasHist = inNames.indexOf('hist') >= 0;
    const hasZ = inNames.indexOf('z') >= 0 && z && z.length > 0;

    for (let s = 0; s < K; s += batch) {
      const e = Math.min(K, s + batch);
      const B = e - s;
      const x1 = new Float32Array(B * 56);
      const x2 = new Float32Array(B * (50 * 31));
      const x3 = new Float32Array(B * (50 * 6));
      for (let i = 0; i < B; i++) {
        x1.set(windows[s + i].x1, i * 56);
        x2.set(windows[s + i].x2, i * (50 * 31));
        x3.set(windows[s + i].x3, i * (50 * 6));
      }
      const feeds = {
        x1: new ort.Tensor('float32', x1, [B, 56]),
        x2: new ort.Tensor('float32', x2, [B, 50, 31]),
        x3: new ort.Tensor('float32', x3, [B, 50, 6]),
      };
      if (hasHist) {
        if (hist) {
          feeds.hist = new ort.Tensor('float32', hist.hist.subarray(s * HIST * POSE, e * HIST * POSE), [B, HIST, POSE]);
          feeds.htime = new ort.Tensor('float32', hist.htime.subarray(s * HIST, e * HIST), [B, HIST]);
          feeds.hmask = new ort.Tensor('bool', hist.hmask.subarray(s * HIST, e * HIST), [B, HIST]);
        } else {
          feeds.hist = new ort.Tensor('float32', new Float32Array(B * HIST * POSE), [B, HIST, POSE]);
          feeds.htime = new ort.Tensor('float32', new Float32Array(B * HIST), [B, HIST]);
          feeds.hmask = new ort.Tensor('bool', new Uint8Array(B * HIST), [B, HIST]);
        }
      }
      if (hasZ) {
        const D = z.length;
        const zb = new Float32Array(B * D);
        for (let i = 0; i < B; i++) zb.set(z, i * D);
        feeds.z = new ort.Tensor('float32', zb, [B, D]);
      }
      const res = await sess.run(feeds);
      const pose = res.pose || res[sess.outputNames[0]];
      out.set(pose.data, s * FRAME_COUNT_OUTPUT * POSE);
      if (onBatch) onBatch(e, K);
    }
    return out;
  }

  /** Deterministic seeded standard normal (mulberry32 + Box-Muller). */
  function sampleNormal(dim, scale, seed) {
    let a = (seed | 0) >>> 0;
    const rnd = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const z = new Float32Array(dim);
    for (let i = 0; i < dim; i += 2) {
      const u = Math.max(rnd(), 1e-7), v = rnd();
      const r = Math.sqrt(-2 * Math.log(u));
      z[i] = r * Math.cos(2 * Math.PI * v) * scale;
      if (i + 1 < dim) z[i + 1] = r * Math.sin(2 * Math.PI * v) * scale;
    }
    return z;
  }

  /**
   * Full two-pass rollout.
   * @param {object} o  {ort, sess, md, fps, overlap, batch, histFrames,
   *                     latent, temp, seed, z, onProgress}
   */
  async function rollout(o) {
    const ort = o.ort, sess = o.sess, md = o.md;
    const fps = o.fps;
    const overlap = Math.max(0, Math.min(o.overlap, FRAME_COUNT_OUTPUT - 1));
    const batch = o.batch || 24;
    const H = o.histFrames || HIST;
    const q = CN.encode.meanQuality();
    const report = o.onProgress || (() => {});

    // One style for the whole replay: per-window z would make neighbouring
    // windows disagree on the hands, exactly like the head-seam problem.
    let zUse = null;
    if (sess.inputNames.indexOf('z') >= 0) {
      // The style-latent width normally comes from the .onnx.json sidecar
      // (worker.js passes `latent`).  Fall back to the graph's own z input
      // shape: otherwise a missing/failed sidecar leaves `z` unfed, ORT
      // throws "input 'z' is missing in 'feeds'", and the page fails outright.
      let lat = (o.latent | 0) || (o.z && o.z.length) || 0;
      if (!lat) {
        const metas = sess.inputMetadata || (sess.raw && sess.raw.inputMetadata) || [];
        const zm = metas.find ? metas.find((m) => m.name === 'z') : null;
        const d = zm && zm.shape && zm.shape[zm.shape.length - 1];
        if (typeof d === 'number' && d > 0) lat = d;
      }
      if (lat > 0) {
        zUse = o.z ? Float32Array.from(o.z)
          : sampleNormal(lat, Number(o.temp) || 0, Number(o.seed) || 0);
      }
    }

    let dur = md.duration + 1.0;
    if (o.maxSeconds) dur = Math.min(dur, o.maxSeconds);
    const stride = FRAME_COUNT_OUTPUT - overlap;
    const step = stride / fps;
    const anchors = [];
    for (let k = 0; ; k++) {
      const a = k * step;
      if (!(a < Math.max(0, dur))) break;
      anchors.push(a);
    }
    if (!anchors.length) return new Float32Array(0);

    const encodeAll = () => anchors.map((a) => CN.encode.encodeWindow(a, md, fps, q));

    // ---- pass 1 ----
    report('pass 1/2: first trajectory');
    const win1 = encodeAll();
    const ys1 = await runWindows(ort, sess, win1, null, batch,
      (done, total) => report(`pass 1/2: ${done}/${total} windows`), zUse);
    const traj1 = blend(anchors, ys1, stride, fps);

    if (sess.inputNames.indexOf('hist') < 0) return traj1;

    // ---- pass 2: history from pass 1's own trajectory ----
    report('pass 2/2: history-conditioned trajectory');
    const K = anchors.length;
    const hist = new Float32Array(K * H * POSE);
    const htime = new Float32Array(K * H);
    const hmask = new Uint8Array(K * H);
    const F1 = traj1.length / 22;
    for (let k = 0; k < K; k++) {
      const a = k * stride;
      for (let j = 0; j < H; j++) {
        const idx = a - H + j;
        if (idx >= 0 && idx < F1) {
          const dst = (k * H + j) * POSE;
          const src = idx * 22 + 1;
          for (let c = 0; c < POSE; c++) hist[dst + c] = traj1[src + c];
          htime[k * H + j] = (idx - a) / fps;
          hmask[k * H + j] = 1;
        }
      }
    }
    const win2 = encodeAll();
    const ys2 = await runWindows(ort, sess, win2, { hist, htime, hmask }, batch,
      (done, total) => report(`pass 2/2: ${done}/${total} windows`), zUse);
    return blend(anchors, ys2, stride, fps);
  }

  CN.model = {
    FRAME_COUNT_OUTPUT, POSE, HIST, tukey, blend,
    createSession, probeWebGPU, assertFp16Ready, fetchBytes, runWindows, rollout,
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
