# wt-channel — follow-ups (Phase C5) & verification notes

## Live-solve verification status (CUDA, conda `witwin2`)

Round-trip + config building are fully tested on CPU. The live GPU solves are gated
per-solver (each probes its own solver and skips only if *that* one can't run here).
**All three now run and pass** — the full suite is `42 passed, 0 skipped`:

- **Monte Carlo radiomap — VERIFIED LIVE** (`test_c3_solve.py::
  test_live_montecarlo_matches_direct`).
- **Deterministic radiomap — VERIFIED LIVE** (`test_c3_solve.py::
  test_live_deterministic_matches_direct`).
- **Point-to-point path solve — VERIFIED LIVE** (`test_c4_path.py::
  test_live_path_solve`).

Each runs end-to-end through the adapter (`to_studio` -> `SolveRunner` /
`PathSolveRunner` -> extractor) and matches a direct platform solve. The earlier
`optixPipelineCreate(multipath)` compile failure (which had blocked the deterministic +
path pipeline in this env) is resolved; `tests/repro_optix_multipath.py` reproduces all
three solvers OK. The per-solver gates remain in `conftest.py` so the suite still
degrades gracefully (skip, not error) on a machine where the pipeline can't compile.

Still unverified (no running server this session): the live editor load via
`load_platform_scene` with factory `wt_channel.examples.demo_scene:build_scene`, and
the actual viewport rendering of the radiomap textured plane / ray poly-lines.

## Requested base-layer change (do NOT make from this plugin)

Core ships `witwin_server/components/transmitter.py` and `receiver.py`
(`@component(name="Transmitter"/"Receiver")`), always loaded and referenced by core's
`scene.py`, `builtin_entity_types.py`, `library.py`, and the frontend Hierarchy. Per
the master plan §8 these should migrate to (or be superseded by) wt-channel. Until
that core migration happens, this plugin registers its endpoint components as
**`ChannelTx` / `ChannelRx`** (display names `Tx` / `Rx`) to avoid silently overwriting
the core components in the global registry. When core removes its versions, consider
registering these under `Transmitter` / `Receiver`.

## Component model (post-C5 simplification)

The surface was collapsed to four core components — **Tx**, **Rx**, **RadioMap**,
**PathSolver** (+ the `MaterialPreset` companion). Antennas are inline on Tx/Rx (the old
`AntennaArraySpec` / `ChannelDefaultArray` and the standalone ULA/UPA library items are
gone; scene-default `tx_array`/`rx_array` are baked into each endpoint on import). The
old `ChannelSettings` object is removed — scene config rides RadioMap (preferred) and
PathSolver, discovered on export. `ReceiverGrid` + `ChannelResult` folded into RadioMap.

## Deferred polish

- **Per-mechanism component-map layers** — `RadioMapResult.components`
  (los / reflection / diffraction) as toggleable layers in `RadioMap`.
- **Best-Tx association map** — multi-Tx `cell_association` / `tx_association_map`
  view + a multi-Tx transmitter list on `RadioMap` (currently a single Tx, blank =
  auto-find; the platform `solve` already accepts a Tx list).
- **Viewport plane overlay** — the 2D radiomap as a textured plane aligned to the
  RadioMap grid bounds (master §7.1), in addition to the Results-panel / figure.
- **Sionna / Mitsuba interop** — pass-through of `Scene.from_sionna` /
  `Scene.load_mitsuba` / `to_sionna`.
- **Grid receiver polarization / array** — the RadioMap grid currently uses the default
  receiver polarization + array (the per-grid polarization/orientation/array authoring
  that the old `ReceiverGrid` exposed was dropped in the simplification).
- **Reusable antenna-array assets** — antennas are authored inline per endpoint;
  promoting arrays to shareable library assets is a follow-up.
- **Full deterministic / Monte Carlo tuning surface** — only the curated tuning fields
  are exposed (resolution_wavelength, solver_mode, memory_profile,
  reflection_field_backend); the remaining advanced tuning fields can be surfaced as
  needed.
