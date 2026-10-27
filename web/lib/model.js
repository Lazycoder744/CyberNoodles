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
  function blend(anchors, ys, stride, fps) {
    const K = anchors.length;
    if (!K) return new Float32Array(0);

    normalizeRows(ys, 0, K, POSE, QUAT_COLS);
    for (let k = 0; k < K; k++) fixQuatSigns(ys, k * FRAME_COUNT_OUTPUT * POSE, FRAME_COUNT_OUTPUT, POSE, QUAT_COLS);

    const overlap = FRAME_COUNT_OUTPUT - stride;
    const w = tukey(FRAME_COUNT_OUTPUT, overlap);
    const nF = (K - 1) * stride + FRAME_COUNT_OUTPUT;
    const acc = new Float64Array(nF * POSE);
    const wsum = new Float64Array(nF);
    const has = new Uint8Array(nF);
    const base = new Float64Array(POSE);

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
  async function webgpuAvailable() {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      return !!adapter;
    } catch (e) {
      return false;
    }
  }

  function wrapSession(sess, ep, threads) {
    // Thin facade so the worker/UI can report which backend actually runs.
    return {
      ep,
      threads,
      inputNames: sess.inputNames,
      outputNames: sess.outputNames,
      run: (feeds) => sess.run(feeds),
      raw: sess,
    };
  }

  async function createSession(ort, modelBytes, onProgress) {
    const report = onProgress || (() => {});
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
    let ep = 'wasm';
    if (want === 'webgpu') ep = 'webgpu';
    else if (want !== 'wasm' && await webgpuAvailable()) ep = 'webgpu';

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
        report('WebGPU session failed (' + e.message + '); using the legacy WASM fallback');
      }
    }
    report(`using the legacy WASM fallback (${threads} thread${threads === 1 ? '' : 's'}) — slower, may change scores`);
    const sess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    return wrapSession(sess, 'wasm', threads);
  }

  /** Fetch model bytes with a progress callback. */
  async function fetchBytes(url, onProgress) {
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
  async function runWindows(ort, sess, windows, hist, batch, onBatch) {
    const K = windows.length;
    const out = new Float32Array(K * FRAME_COUNT_OUTPUT * POSE);
    const inNames = sess.inputNames;
    const hasHist = inNames.indexOf('hist') >= 0;

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
      const res = await sess.run(feeds);
      const pose = res.pose || res[sess.outputNames[0]];
      out.set(pose.data, s * FRAME_COUNT_OUTPUT * POSE);
      if (onBatch) onBatch(e, K);
    }
    return out;
  }

  /**
   * Full two-pass rollout.
   * @param {object} o  {ort, sess, md, fps, overlap, batch, histFrames, onProgress}
   * @returns {Promise<Float32Array>} traj (F, 22)
   */
  async function rollout(o) {
    const ort = o.ort, sess = o.sess, md = o.md;
    const fps = o.fps;
    const overlap = Math.max(0, Math.min(o.overlap, FRAME_COUNT_OUTPUT - 1));
    const batch = o.batch || 24;
    const H = o.histFrames || HIST;
    const q = CN.encode.meanQuality();
    const report = o.onProgress || (() => {});

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
      (done, total) => report(`pass 1/2: ${done}/${total} windows`));
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
      (done, total) => report(`pass 2/2: ${done}/${total} windows`));
    return blend(anchors, ys2, stride, fps);
  }

  CN.model = {
    FRAME_COUNT_OUTPUT, POSE, HIST, tukey, blend,
    createSession, fetchBytes, runWindows, rollout,
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
