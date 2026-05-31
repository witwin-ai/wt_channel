"""PathSolver — point-to-point path solve (CIR/CFR/taps) + viewport ray drawing.

A standalone solver object: it carries the scene config (frequency / device)
and the path config, auto-finds the transmitter and point receiver in the
scene (or uses the named ones), solves the discrete Tx<->Rx propagation paths
(``path.solve``), keeps the ``PathResult`` server-side, plots the CIR / CFR /
taps, and (when ``draw_paths``) draws every ray as ONE bulk ``ray_batch``
RenderSpec — every segment of every path packed into one ``LineSegments``
draw call (RB1-RB5). Live solve needs CUDA RayD/OptiX.
"""
from witwin_server import Notifications, SceneObject, api
from witwin_server.core.components import (
    Component,
    bool_field,
    button,
    component,
    component_field,
    figure,
    float_field,
    int_field,
    string_field,
    vector2_field,
)
from witwin_server.core.components import RenderSpec, ray_batch_geometry, ray_batch_material

from ..adapter.path_solve import PathLineBuilder, PathPlots, PathSolveRunner
from .category import CATEGORY

_SCENE = "Scene"
_TAPS = {"field_name": "plot_mode", "operator": "eq", "value": "taps"}
_GAIN = {"field_name": "color_mode", "operator": "eq", "value": "gain"}
TX_COMPONENT = "ChannelTx"
RX_COMPONENT = "ChannelRx"

# Fixed name of the hidden anchor SceneObject this component owns; one anchor
# per PathSolver instance, looked up via tag because slug ids are scene-scoped.
_ANCHOR_TAG_PREFIX = "wt_channel:ray_batch:"


@component(name="PathSolver", display_name="PathSolver", category=CATEGORY)
class PathSolverComponent(Component):
    """Discrete Tx<->Rx path solve + CIR/CFR/taps plots + bulk ray_batch viewport."""

    _result = None        # runtime PathResult (never serialized)
    _anchor_id = None     # id of this component's hidden ray-batch anchor object

    # --- scene config ---
    frequency = float_field(3.5e9, min=0.0, group=_SCENE,
                            description="Scene carrier frequency in Hz (re-evaluates ITU materials).")
    device = string_field("cuda", options=["cuda", "cpu"], enum_toggle=True, group=_SCENE,
                          description="Solve device. Live solve needs CUDA RayD.")
    use_noise_power = bool_field(False, group=_SCENE, title="Noise")
    noise_power = float_field(1e-13, min=0.0, group=_SCENE, show_if="use_noise_power",
                             description="Linear noise power (carried with the scene config).")

    # --- endpoints (empty = auto-find the first one in the scene) ---
    transmitter = component_field("ChannelTx", title="Solve",
                                  description="Transmitter object (empty = first Tx).")
    receiver = component_field("ChannelRx",
                              description="Point Receiver object (empty = first Rx; not a RadioMap).")

    # --- path config ---
    num_samples = int_field(10000, min=1, description="Probe ray count.")
    max_bounces = int_field(2, min=0, description="Reflection depth.")
    max_diffraction_order = int_field(1, min=0, description="UTD diffraction order.")
    use_max_num_paths = bool_field(False, title="Limit paths")
    max_num_paths = int_field(50, min=1, show_if="use_max_num_paths")
    synthetic_array = bool_field(True)

    # --- visualization ---
    draw_paths = bool_field(True, description="Draw the bulk ray batch in the viewport.")
    color_mode = string_field("type", options=["type", "gain"], enum_toggle=True, title="Color",
                              description=("type: LoS / reflection / diffraction "
                                           "tint; gain: per-path power → colormap."))
    gain_colormap = string_field("viridis", options=["viridis"], show_if=_GAIN,
                                 description="Colormap for the gain channel (RB6).")
    gain_db_range = vector2_field([-120.0, -40.0], show_if=_GAIN,
                                  description="dB clamp range [min, max] for the gain colormap.")
    plot_mode = string_field("cir", options=["cir", "cfr", "taps"], enum_toggle=True, title="Plot")
    bandwidth = float_field(100e6, min=1.0, show_if=_TAPS, description="Tap synthesis bandwidth (Hz).")
    num_taps = int_field(64, min=1, show_if=_TAPS)
    path_figure = figure(title="Channel Response")

    @button(display_name="Solve Paths")
    async def solve(self):
        from ..adapter.channel_adapter import ChannelAdapter
        from ..adapter.offload import run_off_loop

        scene = self._owner.scene
        transmitter = self._resolve("transmitter", TX_COMPONENT)
        if transmitter is None:
            Notifications.error("Channel", "No transmitter found (add a Tx, or assign one).")
            return "Transmitter not found"
        receiver = self._resolve("receiver", RX_COMPONENT)
        if receiver is None:
            Notifications.error("Channel", "No point Receiver found (add an Rx, or assign one).")
            return "Receiver not found"
        if not ChannelAdapter.has_structures(scene):
            Notifications.error("Channel", "Path solve needs at least one structure/mesh "
                                "(e.g. a ground plane or building). Add geometry to the scene.")
            return "No structures in scene"
        # Run off the loop so the GPU solve doesn't block the asyncio loop while
        # broadcasts are queued; one try-boundary so any failure surfaces as a
        # toast (the dispatcher otherwise logs to the Console only). Crucially,
        # we also do plot/batch *extraction* on the same worker thread — the
        # PathResult's tensors are DrJit cuda.ad and are pinned to the JIT scope
        # of the thread that produced them; touching .cir() / .taps() / .cfr()
        # or to_torch_view(...) from any other thread trips DrJit's "scope X
        # depends on scope Y" assertion.
        draw = bool(self.draw_paths)
        try:
            plot_data, batch = await run_off_loop(
                self._solve_extract, scene, transmitter, receiver, draw)
            self._publish_plot(plot_data)
            drawn = self._publish_draw(scene, batch) if draw else 0
        except Exception as exc:  # noqa: BLE001 - GPU/render boundary: surface, don't swallow
            Notifications.error("Channel", f"path solve failed: {exc}")
            return f"Path solve failed: {exc}"
        Notifications.success("Channel", f"Path solve {transmitter} -> {receiver}: {drawn} rays")
        return f"Path solve complete ({drawn} rays)"

    @button(display_name="Update Plot")
    async def update_plot(self):
        from ..adapter.offload import run_off_loop

        if self._result is None:
            Notifications.warning("Channel", "Solve paths first.")
            return "Solve first"
        try:
            # Extract on the solve worker thread — see solve() for why.
            data = await run_off_loop(self._extract_plot_data)
            self._publish_plot(data)
        except Exception as exc:  # noqa: BLE001 - surface as a toast
            Notifications.error("Channel", f"plot failed: {exc}")
            return f"Plot failed: {exc}"
        Notifications.success("Channel", f"{self.plot_mode} plot updated")
        return "Plot updated"

    @button(display_name="Recolor / Redraw")
    async def draw(self):
        """Re-emit the ray batch with the current color_mode / gain settings.

        Costs nothing on the WebSocket: the binary asset is content-addressed,
        so an unchanged batch dedupes to the same asset id and the frontend
        cache hits. Use this after toggling color_mode or gain_db_range.
        """
        from ..adapter.offload import run_off_loop

        if self._result is None:
            Notifications.warning("Channel", "Solve paths first.")
            return "Solve first"
        try:
            # Extract on the solve worker thread — see solve() for why.
            batch = await run_off_loop(self._extract_batch)
            drawn = self._publish_draw(self._owner.scene, batch)
        except Exception as exc:  # noqa: BLE001 - surface as a toast
            Notifications.error("Channel", f"draw failed: {exc}")
            return f"Draw failed: {exc}"
        Notifications.success("Channel", f"Drew {drawn} rays")
        return f"Drew {drawn} rays"

    # --- internals (no notifications; shared by the buttons + solve) ----------
    #
    # DrJit thread-affinity rule for this component: the platform's PathResult
    # holds cuda.ad tensors whose JIT scope belongs to the thread that produced
    # them. Anything that reads them (`.cir()`, `.taps()`, `.cfr()`, or
    # `to_torch_view(...)`, which returns a live DLPack view, not a copy) must
    # run on that same thread, otherwise DrJit raises
    # ``jitc_assemble(): variable rX (scope A) depends on rY (scope B)``.
    # The wt_channel-solve ThreadPoolExecutor in offload.py is a single-worker
    # pool exactly so the same thread services every solve and every extraction.
    # So: ``_extract_*`` methods touch the result (worker thread only),
    # ``_publish_*`` methods touch the scene / figure / viewport API (main
    # thread only), and the sync ``_plot`` / ``_draw`` wrappers chain them
    # together for unit tests that use a torch-only synthetic PathResult where
    # thread affinity is moot.

    def _solve_extract(self, scene, transmitter, receiver, draw):
        """Worker-thread: run the solve AND extract everything DrJit-backed.

        Returns ``(plot_data, batch_or_None)`` as pure numpy so the caller can
        publish on the main asyncio thread without touching DrJit.
        """
        self._result = PathSolveRunner.run(
            scene, self, transmitter=transmitter, receiver=receiver)
        plot_data = self._extract_plot_data()
        batch = self._extract_batch() if draw else None
        return plot_data, batch

    def _extract_plot_data(self):
        """Worker-thread: build a pure-numpy PlotData from ``self._result``."""
        mode = str(self.plot_mode)
        if mode == "cfr":
            return PathPlots.cfr(self._result)
        if mode == "taps":
            return PathPlots.taps(self._result, bandwidth=float(self.bandwidth),
                                  num_taps=int(self.num_taps))
        return PathPlots.cir(self._result)

    def _extract_batch(self):
        """Worker-thread: build the numpy RayBatch from ``self._result``."""
        db_min, db_max = self._gain_range()
        return PathLineBuilder.build_batch(
            self._result,
            gain_db_range=(db_min, db_max),
            colormap=str(self.gain_colormap),
        )

    def _publish_plot(self, data):
        """Main-thread: update the figure from a pure-numpy PlotData."""
        fig = self.path_figure.clear()
        # Pass the ndarrays directly: Figure._to_list has a fast path for them.
        # list(np.ndarray) would unwrap to a list of numpy scalars, which JSON
        # can't encode and which used to silently break the broadcast set.
        (fig.stem if data.kind == "stem" else fig.line)(data.x, data.y)
        fig.title(data.title)

    def _publish_draw(self, scene, batch) -> int:
        """Main-thread: push ``batch`` as one ray_batch RenderSpec on a hidden anchor."""
        anchor = self._ensure_anchor(scene)
        if batch is None or batch.empty:
            api.viewport.clear_render_spec(anchor.id)
            return 0
        # Register the bulk arrays as a content-addressed binary asset; the
        # frontend fetches it once and caches by content hash. Re-solving the
        # same batch (or just flipping color_mode without changing gain bounds)
        # dedupes to the same asset.
        meta = api.viewport.register_ray_batch(
            batch.segments,
            colors_type=batch.colors_type,
            colors_gain=batch.colors_gain,
            types=batch.types,
            scene_id=scene.scene_id,
        )
        spec = RenderSpec(
            geometry=ray_batch_geometry(
                segment_count=batch.segment_count,
                binary=meta,
                color_channel=str(self.color_mode),
                bounds=meta.get("bounds"),
            ),
            material=ray_batch_material(additive=(str(self.color_mode) == "gain")),
        )
        api.viewport.set_render_spec(anchor.id, spec.to_dict())
        return batch.segment_count

    # Sync convenience wrappers — only safe for callers that hold the
    # PathResult on the thread that produced it (e.g. unit tests that build a
    # torch-only synthetic result). The async buttons MUST go through
    # _extract_* + _publish_* via run_off_loop instead.

    def _plot(self):
        self._publish_plot(self._extract_plot_data())

    def _draw(self, scene) -> int:
        return self._publish_draw(scene, self._extract_batch())

    def _ensure_anchor(self, scene):
        """Find or create this component's hidden ray-batch anchor SceneObject.

        The anchor is ``system=True`` so it never appears in Hierarchy nor in
        the .wtscene file; one anchor per PathSolver instance, looked up by a
        per-component tag to survive scene reloads that re-instantiate this
        Python component (``self._anchor_id`` is in-memory only).
        """
        owner_id = self._owner.id if self._owner is not None else "0"
        tag = _ANCHOR_TAG_PREFIX + str(owner_id)
        if self._anchor_id is not None:
            obj = scene.get_object(self._anchor_id)
            if obj is not None and obj.tag == tag:
                return obj
        for obj in scene.objects.values():
            if obj.tag == tag:
                self._anchor_id = obj.id
                return obj
        anchor = SceneObject(
            name="Channel Path Rays",
            mesh_type="Empty",
            tag=tag,
            system=True,
            disable_transform=True,
            allow_add_components=False,
        )
        scene.add_object(anchor)
        self._anchor_id = anchor.id
        return anchor

    def _gain_range(self) -> tuple:
        rng = self.gain_db_range
        if rng is None:
            return (-120.0, -40.0)
        try:
            return (float(rng[0]), float(rng[1]))
        except Exception:  # noqa: BLE001 - vector2 may be None pre-init
            return (-120.0, -40.0)

    def _resolve(self, field_name, component_name):
        # Use the referenced endpoint object if assigned; else the first matching one.
        ref = self.get_referenced_component(field_name)
        if ref is not None and ref._owner is not None:
            return ref._owner.name
        for obj in self._owner.scene.objects.values():
            if obj.get_component(component_name) is not None:
                return obj.name
        return None
