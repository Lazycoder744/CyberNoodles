# CyberNoodles — browser app

Paste a BeatSaver ID, pick a Standard difficulty, and this app runs the
CyberNoodles policy network **in your browser** and writes a real `.bsor`
replay you can watch in ArcViewer or upload to BeatLeader.

Nothing is uploaded, and there is no inference server: the map is fetched from
BeatSaver, the model is fetched from wherever `CN.config.modelUrl` points, and
the rollout, cut reconstruction, scoring and BSOR writing all happen locally.

```
index.html      page shell + script tags
app.js          UI, talks to the worker, renders progress/result
worker.js       the whole pipeline (classic worker, importScripts)
style.css       styling
lib/util.js     pyRound (round-half-even) and IEEE float32 helpers
lib/config.js   model URL, onnxruntime-web version, rollout settings
lib/zip.js      dependency-free ZIP reader (DecompressionStream)
lib/map.js      Beat Saber map parsing            <- port of bsmap.py
lib/encode.js   map-window encoder                <- port of encode_window
lib/model.js    2-pass ONNX rollout + Tukey blend <- port of cybernoodles_infer
lib/score.js    cut reconstruction / events       <- port of reconstruct
lib/avscore.js  ArcViewer score model             <- port of av_score.py
lib/bsor.js     BSOR writer                       <- port of bsmap.write_bsor
test/parity.mjs Node harness: checks every JS stage against Python
test/bsorread.mjs test-only BSOR reader
test/ort_e2e.mjs  + real onnxruntime-web inference (needs the npm package)
test/worker_sim.mjs runs worker.js in Node: protocol, model fetch, BSOR transfer
```

## Running it locally

A worker plus `importScripts` needs a real origin, so serve the directory —
opening `index.html` from `file://` will not work:

```bash
python3 -m http.server 8000 --directory web
# then open http://127.0.0.1:8000/
```

## Execution providers

The page picks **WebGPU** when the browser exposes an adapter and otherwise runs
the **legacy WASM fallback** (it also retries on WASM if a WebGPU session cannot
be built). Add `?ep=wasm` or `?ep=webgpu` to force one; the active backend is
shown in the page log. The page warns when WASM is active: it is roughly 30×
slower and its float32 path can slightly change the score. `?ep=wasm` is meant
for debugging and verification. WebNN is intentionally not offered — see
Runtime notes for the measurement.

## Watching the replay

The page writes the `.bsor` and offers it as a download. ArcViewer takes a
*hosted URL* (`?replayURL=`) and fetches it cross-origin, so the folder you serve
must send `Access-Control-Allow-Origin`. Plain `python3 -m http.server` does
**not**, and the fetch is silently blocked.

```bash
python3 bsopencode/serve_replays.py <folder with the .bsor> 8765   # CORS + Range
# or: cd <folder> && npx http-server --cors -p 8765

# then, with the printed filename:
# https://allpoland.github.io/ArcViewer/?replayURL=http://127.0.0.1:8765/<file>.bsor&noProxy=true
```

`http://127.0.0.1` is a *potentially trustworthy* origin, so an HTTPS page
(ArcViewer) is allowed to fetch it — that is what makes the loopback trick work
from a Pages deployment.

## The model

The shipped model is `web/models/cybernoodles_final_r5_275000.onnx` (≈91 MiB,
step 275000), committed with the site so the default same-origin URL needs no
CORS. `web/models/` and `*.onnx` are gitignored by the repository, so the file
was force-added; if you replace it, stage the new file with `git add -f`. The
cost is 91 MiB in every clone and every Pages deploy.

To use a different model without changing the repository, open the page with:

```
index.html?model=https://huggingface.co/<user>/<repo>/resolve/main/cybernoodles_final_r5_275000.onnx
```

A Hugging Face `resolve/main` link echoes the browser's origin and works, as
does jsDelivr. GitHub *release assets* are not CORS-open — the final redirect to
`release-assets.githubusercontent.com` carries no `access-control-allow-origin`,
so a release URL will not work here.

## Publishing

`.github/workflows/pages.yml` stages the runtime files from `web/` into
`_site` (dropping the local `web_v2/` reference copy, `test/` and `files.zip`)
and uploads that as the Pages artifact on every push that touches `web/` or the
workflow.  In the repository settings set **Settings → Pages → Build and
deployment → Source: GitHub Actions** once; the branch-based option would only
let you serve `/` or `/docs`.

## Runtime notes

* **onnxruntime-web is pinned to 1.22.0** and loaded from jsDelivr
  (`ort.all.min.js`), with `wasmPaths` pointing at the same pinned `dist/`
  directory so the `.wasm` backend matches the JS. Later versions restructured
  the dist layout.
* **WebGPU by default, legacy WASM fallback.** `?ep=webgpu` and `?ep=wasm` force
  an execution provider; the default `auto` creates a WebGPU session when
  `navigator.gpu` yields an adapter and falls back to WASM if that session
  cannot be built. The active backend (`WebGPU`, or `WASM, N threads`) is shown
  in the page log, and WASM carries a warning: it is roughly 30× slower and can
  slightly change the score. On this dev machine the adapter is an RTX 4050 via
  Vulkan.
* **WebNN is not used.** onnxruntime-web 1.22 ships a `webnn` EP, but it needs
  `navigator.ml`, which Chrome 155 on Linux does not expose even with
  `--enable-features=WebNN`; ORT logs `WebNN is not supported in current
  environment` and silently runs on WASM. Browsers that do expose WebNN also
  expose WebGPU, so it would add a third backend without changing the default.
* **WASM threading.** Threaded WASM needs `SharedArrayBuffer`, which browsers
  only expose to cross-origin-isolated pages (COOP+COEP). GitHub Pages cannot
  send those headers, so visitors get `ort.env.wasm.numThreads = 1`; a
  self-hosted copy that sets them gets `min(8, hardwareConcurrency)`.
* **Speed.** Two forward passes per 50-frame window, stride 25. Measured
  headlessly (batch 24, real encoded windows):
  - one batch: **125 ms WebGPU**, 1,382 ms WASM 8 threads, 4,128 ms WASM
    1 thread;
  - a 30 s slice end-to-end: **1.3 s vs 37.2 s** WASM 1 thread (28.5×);
  - the full 218 s, 1,216-note map: **8.5 s rollout** (2,333 frames/s) on
    WebGPU, versus roughly 3.5 minutes single-threaded on WASM.
  Very dense maps scale linearly with note count.
* **WebGPU numerics.** On the same 30 s slice WebGPU and WASM produce the same
  frame count and the same good/miss (130/1086). Hand positions differ at
  p50 1.1 mm / p90 5.0 mm / p99 67 mm, with one smooth 30-frame bump peaking at
  0.21 m; speed distributions are identical (p99 10.5 vs 11.2 m/s). On the full
  map WebGPU scores 98.9859% against the shipped Python replay's 98.9395%, with
  the same 1,216 hits and 0 misses. 97 of the 1,216 notes changed their per-note
  score, by 64 base points in total before the combo multiplier; no hit or miss
  changed. That is the same last-bit float32 drift, not a different trajectory —
  and the page always reports the score of the file it writes.
* **Standard characteristic only** (no 360/90/lightshow difficulties).

## Is the replay honest?

Yes — the number the page shows is ArcViewer's own score model
(`lib/avscore.js`, a port of `bsopencode/av_score.py`) applied to the same cut
ratings that get written into the BSOR. There is no cosmetic score field: the
`score` in the replay metadata is the same value the page displays.

The graded model is the one that passed the honest gate
(`validate_ranked.py`, mean 95.74 / median 90.22 over the ten held-out ranked
maps); the exported ONNX tracks the PyTorch checkpoint to 7.2e-06.

## Parity tests

`web/lib/*` is a line-by-line port of the Python pipeline, so it is checked
against it in Node rather than by eye. `js_ref_dump.py` runs the Python side and
writes reference data; the harness then compares each stage.

```bash
# 1. Python reference (writes /tmp/opencode/jsref)
python3 bsopencode/js_ref_dump.py \
    /mnt/games/bs-ai/maps_cache/b5a1424cad898181d429cb231b83296a3e67d58f ExpertPlus

# 2. Every JS stage against it (no browser, no ONNX runtime needed)
node web/test/parity.mjs
```

Last run, on *Death From Above* [ExpertPlus]:

| stage | result |
| --- | --- |
| map parse (notes/bombs/walls/rot/scoring/chains + metadata) | bit-exact, max abs diff `0` |
| encoder `x1` / `x3`, and `x2` | exact / `2.4e-07` |
| Tukey crossfade, pass 1 / pass 2 | `2.4e-07` / `1.2e-07` |
| cut reconstruction: good/bad classification, note-id fields | `0` mismatches of 1216 |
| cut reconstruction: `dist`/`speed`/`before` | `1.3e-07` / `2.2e-05` / `2.9e-06` |
| **ArcViewer accuracy** | **identical** (JS `111776/1111475` = Python) |
| BSOR: decoded events, scoring types, per-note score | `0` mismatches |

A byte-for-byte comparison of the two BSOR files differs in ~8% of bytes: the
pose floats come out of two different ONNX runtimes, so last-bit float32 drift
is unavoidable and meaningless. The harness therefore compares what ArcViewer
actually consumes (decoded events + the score they produce), which is equal.

The two things that are *not* self-evident from a port and were worth getting
exactly right:

* Python 3's `round()` is round-half-to-even, and the score model depends on it
  (`util.pyRound`).
* numpy does the blade/angle geometry in **float32**, and `arccos` amplifies the
  difference near 0°, which is large enough to flip a before/after rating. The
  JS reproduces numpy's float32 rounding step by step (`lib/score.js`).

### Full end-to-end with real ONNX inference

`parity.mjs` deliberately stops short of inference, so `ort_e2e.mjs` closes the
gap: it loads the same `lib/*`, creates a real `onnxruntime-web` session from the
exported ONNX, runs both passes, reconstructs and scores the cuts, and can
compare the result with a `.bsor` written by the Python pipeline.

```bash
npm i onnxruntime-web@1.22.0
ORT_MODULE=$PWD/node_modules/onnxruntime-web/dist/ort.node.min.mjs \
  node web/test/ort_e2e.mjs \
    --map maps_cache/b5a1424cad898181d429cb231b83296a3e67d58f \
    --diff ExpertPlus \
    --shipped replays/replays_final_r5_275000/DeathFromAbove.bsor \
    --full
```

Full-length results against the shipped Python replays (real ORT, WASM backend):

| map | browser pipeline | shipped Python `.bsor` | delta |
| --- | --- | --- | --- |
| Death From Above | 98.9395% (`1099688/1111475`) | 98.9395% (`1099688/1111475`) | **0** |
| Mine All Day | 96.3891% (`1237168/1283515`) | 96.3891% (`1237168/1283515`) | **0** |
| Xeon | 92.4304% (`2205939/2386595`) | 92.4334% (`2206011/2386595`) | 0.0030 pt |
| Unwelcome School | 86.1897% (`1211720/1405875`) | 86.1897% (`1211720/1405875`) | **0** |
| Elderly Care Brothel | 28.1479% (`366721/1302835`) | 28.1485% (`366728/1302835`) | 0.0005 pt |

Three of the five agree exactly; the other two differ by at most 0.003 pt and
still have exactly the same good/miss counts. Where the trajectory is
byte-identical (`parity.mjs` feeds Python's own trajectory through the JS
scorer) the scores are exactly equal. The page always displays the score
produced by its own BSOR, so the number a visitor sees and the number ArcViewer
computes from that file are the same.

Cost on this machine (CPU/WASM, batch 24, single-threaded): a 218 s, 1216-note
map is 789 windows × 2 passes = **208 s**, ~640 MB peak.

### Worker smoke test

`node web/test/worker_sim.mjs` runs `worker.js` itself in Node (main-realm
`vm.runInThisContext`, a shim for `self`/`importScripts`/`postMessage`) with a
static server standing in for the page's origin. It checks the wiring that the
other two harnesses cannot see:

* `importScripts()` loads `lib/config.js` **before** `CN.config.ortScript`
  (a real bug found this way: the runtime URL used to be read while `CN.config`
  was still undefined),
* the model is fetched over HTTP from the page's own origin (91.1 MiB) and a
  real session is created with `inputs=[x1,x2,x3,hist,htime,hmask]`,
* BeatSaver lookup → CDN zip download → map parse → rollout → score,
* the finished BSOR is transferred and **re-decoded**: the score read back out of
  the bytes equals the score the worker reported (`10.0565%`, the same as the
  Python reference at that truncated horizon).
