# bsor_tools

Rust BSOR parser/writer and audit utilities for the CyberNoodles replay pipeline.

## Build

```powershell
cargo build --release
```

Binary:

- Windows: `rust/bsor_tools/target/release/bsor_tools.exe`

## CLI

```powershell
bsor_tools dump-json <replay.bsor>
bsor_tools dump-dataset-json <replay.bsor>
bsor_tools validate <replay.bsor>
bsor_tools audit --replay-dir <dir> --check both
```

Write from JSON:

```powershell
Get-Content replay.json | bsor_tools write-json --output out.bsor
```

## Python integration

The repo uses `cybernoodles.bsor_bridge` to call the binary when needed.

Relevant environment variables:

- `CYBERNOODLES_BSOR_BACKEND=auto|python|rust`
- `CYBERNOODLES_BSOR_WRITE_BACKEND=python|rust`
- `CYBERNOODLES_BSOR_VALIDATE_BACKEND=auto|python|rust`
- `CYBERNOODLES_BSOR_TOOLS_BIN=<absolute path to bsor_tools>`

`auto` keeps the existing Python path as the default and falls back to Rust when the `bsor` package rejects a replay variant.
