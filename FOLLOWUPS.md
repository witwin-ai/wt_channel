# wt-channel — follow-ups (Phase C5) & verification notes

## Live-solve verification status (CUDA, conda `witwin2`)

Round-trip + config building are fully tested on CPU. The live GPU solves are gated
per-solver (each probes its own solver and skips only if *that* one can't run here):

- **Monte Carlo radiomap — VERIFIED LIVE.** `test_c3_solve.py::
  test_live_montecarlo_matches_direct` runs end-to-end through the adapter
  (`to_studio` -> `SolveRunner` -> `RadioMapExtractor`) and matches a direct
  `montecarlo.solve`; it PASSES in `witwin2`.
- **Deterministic radiomap + point-to-point path solve — environment-blocked.**
  `test_c3_solve.py::test_live_deterministic_matches_direct` and
  `test_c4_path.py::test_live_path_solve` are SKIPPED: both use the platform's
  deterministic-family OptiX pipeline ("multipath"), which **fails to compile** in
  `witwin2` (`optixPipelineCreate(multipath)` COMPILE ERROR) — a *direct*
  `deterministic.solve` / `path.solve` fails identically (mostly reproducible; one
  rare path success observed), so it is an OptiX/driver issue, not a plugin bug. Monte
  Carlo uses a different pipeline and is unaffected. Re-run on a setup where the
  multipath pipeline compiles to confirm deterministic radiomaps, path CIR/CFR/taps,
  and the viewport ray poly-lines.

Also unverified (no running server this session): the live editor load via
`load_platform_scene` with factory `wt_channel.examples.demo_scene:build_scene`, and
the actual viewport rendering of the radiomap textured plane / ray poly-lines.

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
