# wt-channel

Round-trips [`witwin.channel`](e:/Code/witwin-platform/channel) solver scenes to and
from the witwin-studio editor, re-runs both the **deterministic** and **Monte Carlo**
radiomap solvers, and visualizes the resulting **2D radiomap** (coverage / path-gain
field).

Built on the frozen core base layer (`witwin_server.platform_bridge`): geometry,
transform, scalar material, and structure metadata are shared; this plugin adds the
channel-specific surface — scene config, ITU material presets, endpoints (Tx / Rx /
ReceiverGrid), antenna arrays (custom / ULA / UPA), the two solver families, and the
radiomap result component.

## Layout

```
wt_channel/
  witwin.manifest.json   plugin manifest (id witwin.channel; engines/contributes)
  __init__.py            registers components + the "channel" adapter
  components/            ChannelSettings, MaterialPreset, endpoints, result
  adapter/               ChannelAdapter + settings/material/endpoint maps + solve
  library_items.py       drag-in "Channel Scene" starter + endpoint / array items
  examples/              demo_scene factory (for live load verification)
  tests/                 per-phase round-trip suites (CPU) + live-solve gate (CUDA)
```

## Conventions

Units everywhere: meters, radians, Hz, S/m. Rotation quaternions are `[w, x, y, z]`.
The platform packages (`witwin.channel` / `witwin.core`) are **not** manifest
`dependencies` — they are local `PYTHONPATH` packages; the adapter lazy-imports them
and degrades if absent. Round-trip is CPU-only (`device='cpu'`, native-python tuples);
live solves need CUDA RayD/OptiX.

## Tests

Run from this directory in an environment where `witwin_server` (with
`platform_bridge`) and `witwin.channel` are importable:

```
python -m pytest tests
```

Round-trip suites run on CPU. The live-solve suite (C3) is skipped unless CUDA is
available.
