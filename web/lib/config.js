/* config.js — where the app gets its model and runtime from.
 *
 * Everything here is client-side.  The ONNX model is fetched by the Web Worker
 * (with a progress readout) and, by default, lives next to the page so it is
 * same-origin and needs no CORS at all.  Point `modelUrl` at any CORS-enabled
 * host if you would rather keep the ~95 MB fp32 file out of the git repo (a
 * Hugging Face `resolve/main/<file>` URL works: it answers with
 * `access-control-allow-origin: *`).
 *
 * onnxruntime-web is loaded from a pinned jsDelivr build.  Its `wasmPaths` is
 * the same directory, so the .wasm backend is fetched from the same pinned
 * version.  To go fully offline, vendor `dist/ort.all.min.js` plus the
 * `ort-wasm-simd-threaded*.wasm` files from the same version and set
 * `ortBase` to that local directory.
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});

  const ORT_VERSION = '1.22.0';
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

  CN.config = {
    // CyberNoodles v2 (12-block trunk + style latent), exported by
    // bsopencode/export_onnx.py and export_onnx_fp16.py.  GitHub rejects any
    // file over 100 MiB, and the fp32 model is ~134 MiB, so it ships as two
    // same-origin parts that worker.js joins in memory.  Both URLs are
    // same-origin by default.
    modelUrl: [
      'models/cybernoodles_v2_fp32_part0.onnx',
      'models/cybernoodles_v2_fp32_part1.onnx',
    ],

    // The two modes behind the FP16 / FP32 dropdown.  FP16 is a native
    // half-precision export (same weights, fp32 inputs/outputs) and runs on
    // WebGPU only; it is roughly half the download.
    modelUrls: {
      fp16: 'models/cybernoodles_v2_fp16.onnx',
      fp32: [
        'models/cybernoodles_v2_fp32_part0.onnx',
        'models/cybernoodles_v2_fp32_part1.onnx',
      ],
    },

    // onnxruntime-web
    ortVersion: ORT_VERSION,
    ortBase: ORT_BASE,
    ortScript: ORT_BASE + 'ort.all.min.js',

    // 'auto' prefers WebGPU and keeps WASM as a legacy fallback (much slower,
    // and its float32 path can change scores slightly).  Force one with ?ep=wasm
    // or ?ep=webgpu; ?ep=wasm is intended for debugging/verification only.
    executionProvider: 'auto',

    // rollout settings — must match the verified Python pipeline
    overlap: 25,
    histFrames: 100,
    batch: 24,

    // BeatSaver
    beatSaverApi: 'https://api.beatsaver.com/maps/id/',
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
