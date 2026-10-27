# CyberNoodles
Making an small Open-source model that can play Beat Saber!

## Browser app

The client-side replay generator lives in [`web/`](web/README.md). It looks up a
map on BeatSaver, runs the ONNX policy in a Web Worker (WebGPU by default, with a
legacy WASM fallback), and downloads a `.bsor` replay — nothing is uploaded.

It is deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml):

https://lazycoder744.github.io/CyberNoodles/
