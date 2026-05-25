# wt-channel

Round-trips [`witwin.channel`](e:/Code/witwin-platform/channel) solver scenes to and
from the witwin-studio editor, re-runs both the **deterministic** and **Monte Carlo**
radiomap solvers, and visualizes the resulting **2D radiomap** (coverage / path-gain
field).

Built on the frozen core base layer (`witwin_server.platform_bridge`): geometry,
transform, scalar material, and structure metadata are shared; this plugin adds the
channel-specific surface as **four core components**:

- **Tx** / **Rx** — transmitter / point receiver endpoints; the antenna (pattern +
  single / ULA / UPA / custom array + polarization) is defined inline on each. Each draws
  a viewport marker gizmo (green Tx, red Rx).
- **RadioMap** — **subclasses the core ``Field`` component** (2D), so it inherits the
  viewport border/handle gizmo and the 2D plane visualizer for free. It adds the scene
  config (device / noise; frequency is the Field field), the solver config (deterministic
  + Monte Carlo) and a **Solve** button, auto-finds the transmitter, and writes the solved
  metric back into the field's plane data — the radiomap shows as the colored 2D plane
  inside its border. The Field region (``min_bound`` / ``max_bound`` / ``plane`` /
  ``resolution``) is the receiver grid, in the object's local frame (grid follows the
  Transform).
- **PathSolver** — point-to-point path solve (CIR / CFR / taps + viewport rays); carries
  its own scene config and auto-finds the Tx/Rx.

Plus a `MaterialPreset` companion for ITU structure materials. There is no separate
scene-config object: frequency / device / noise ride the RadioMap (preferred) and
PathSolver, and are discovered from them on export.

## Layout

```
wt_channel/
  witwin.manifest.json   plugin manifest (id witwin.channel; engines/contributes)
  __init__.py            registers components + the "channel" adapter
  components/            endpoints (Tx/Rx), radiomap, pathsolver, materials
  adapter/               ChannelAdapter + material/endpoint/array maps + solve/path
  library_items.py       drag-in Channel Scene (Demo) + Tx / Rx / RadioMap / Path Solver
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
