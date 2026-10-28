/* util.js — small helpers shared by the CyberNoodles browser pipeline.
 *
 * These exist mainly to keep the JavaScript numerically identical to the
 * Python reference (bsmap.py / cybernoodles_infer.py / av_score.py).  The two
 * that matter are `pyRound`, because Python 3's round() is round-half-to-even
 * (JavaScript's Math.round is round-half-up, which changes scores), and `f32`,
 * because every value that goes through a BSOR float field is IEEE single
 * precision, and ArcViewer scores the *read-back* values.
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});

  const f32buf = new Float32Array(1);

  const U = {
    /** Python 3 round(): round half to even. */
    pyRound(x) {
      if (!isFinite(x)) return 0;
      const f = Math.floor(x);
      const d = x - f;
      if (d > 0.5) return f + 1;
      if (d < 0.5) return f;
      return (f % 2 === 0) ? f : f + 1;
    },

    /** Round x to IEEE single precision (what a BSOR float field stores). */
    f32(x) {
      f32buf[0] = x;
      return f32buf[0];
    },

    clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); },

    encodeUtf8(s) { return new TextEncoder().encode(s == null ? '' : String(s)); },

    decodeUtf8(b) { return new TextDecoder('utf-8').decode(b); },

    /** Python int(): truncate toward zero. */
    i(v) { return Math.trunc(Number(v) || 0); },

    /** Python float(). */
    f(v) { return Number(v); },
  };

  CN.util = U;
})(typeof globalThis !== 'undefined' ? globalThis : self);
