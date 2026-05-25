# wt-channel — follow-ups (Phase C5) & verification notes

## Needs user verification (environment-blocked)

Round-trip + config building are fully tested on CPU (35 passing tests). The **live
GPU solves** are written but **skipped** in the dev environment and must be confirmed
on a working CUDA RayD/OptiX setup:

- `tests/test_c3_solve.py::test_live_both_solvers_match_direct` — deterministic +
  Monte Carlo radiomaps via the adapter vs a direct platform run.
- `tests/test_c4_path.py::test_live_path_solve` — point-to-point path solve + ray
  poly-line construction.

In conda env `witwin2` (CUDA present) the OptiX pipeline fails to compile
(`optixPipelineCreate` COMPILE ERROR), which also breaks a *direct*
`witwin.channel.deterministic.solve` / `path.solve` — i.e. it is an OptiX/driver
environment issue, not a plugin bug. The `solver_ready` fixture skips rather than
faking a pass. Also unverified (no running server this session): the live editor load
via `load_platform_scene` with factory `wt_channel.examples.demo_scene:build_scene`,
and the actual viewport rendering of the radiomap textured plane / ray poly-lines.

## Requested base-layer change (do NOT make from this plugin)

Core ships `witwin_server/components/transmitter.py` and `receiver.py`
(`@component(name="Transmitter"/"Receiver")`), always loaded and referenced by core's
`scene.py`, `builtin_entity_types.py`, `library.py`, and the frontend Hierarchy. Per
the master plan §8 these should migrate to (or be superseded by) wt-channel. Until
that core migration happens, this plugin names its endpoint components
**`ChannelTransmitter` / `ChannelReceiver`** to avoid silently overwriting the core
components in the global registry. When core removes its versions, consider renaming
these back to `Transmitter` / `Receiver`.

## Deferred polish (plan §5 C5)

- **Per-mechanism component-map layers** — `RadioMapResult.components`
  (los / reflection / diffraction) as toggleable layers in `ChannelResult`.
- **Best-Tx association map** — multi-Tx `cell_association` / `tx_association_map`
  view + a multi-Tx transmitter list on `ChannelResult` (currently single Tx by name;
  the platform `solve` already accepts a Tx list).
- **Viewport plane overlay** — the 2D radiomap as a textured plane aligned to the
  `ReceiverGrid` bounds (master §7.1), in addition to the Results-panel / figure.
- **Sionna / Mitsuba interop** — pass-through of `Scene.from_sionna` /
  `Scene.load_mitsuba` / `to_sionna`.
- **Reusable antenna-array assets** — the array spec is currently nested per endpoint
  (+ scene-default objects); promoting arrays to shareable library assets is a follow-up.
- **Full deterministic / Monte Carlo tuning surface** — only the curated tuning fields
  are exposed (resolution_wavelength, solver_mode, memory_profile,
  reflection_field_backend); the remaining advanced tuning fields can be surfaced as
  needed.
