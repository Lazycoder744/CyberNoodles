# CyberNoodles
Making an small Open-source model that can play Beat Saber!

## Browser app

The client-side replay generator lives in [`web/`](web/README.md). It looks up a
map on BeatSaver, runs the model in a Web Worker (WebGPU by default, with a
legacy WASM fallback), and downloads a `.bsor` replay.

Visit the site at:

https://lazycoder744.github.io/CyberNoodles/

# Update on website

As far as I can tell, the website inference is utterly broken right now. I will attempt to fix it through the coming days, as I also train the next version of the beta model. Please be patient in the meantime! Thank you!

There is a potential for the Python inference scripts to be released here also, once the weights are updated.
