"""PathSolver — point-to-point path solve (CIR/CFR/taps) + viewport ray drawing.

A standalone solver object: it carries the scene config (frequency / device) and the
path config, auto-finds the transmitter and point receiver in the scene (or uses the
named ones), solves the discrete Tx<->Rx propagation paths (``path.solve``), keeps the
``PathResult`` server-side, plots the CIR / CFR / taps, and (when ``draw_paths``) draws
each ray as a declarative poly-line render spec colored by interaction type
(LoS / reflection / diffraction). Live solve needs CUDA RayD/OptiX.
"""
from witwin_server import Notifications, SceneObject, api
from witwin_server.components import (
    Component,
    bool_field,
    button,
    component,
    component_field,
    figure,
    float_field,
    int_field,
    string_field,
)
from witwin_server.components import RenderSpec, lines_geometry, pbr_material

from ..adapter.path_solve import PathLineBuilder, PathPlots, PathSolveRunner
from .category import CATEGORY

_SCENE = "Scene"
_TAPS = {"field_name": "plot_mode", "operator": "eq", "value": "taps"}
TX_COMPONENT = "ChannelTx"
RX_COMPONENT = "ChannelRx"


@component(name="PathSolver", display_name="PathSolver", category=CATEGORY)
class PathSolverComponent(Component):
    """Discrete Tx<->Rx path solve + CIR/CFR/taps plots + viewport ray poly-lines."""

    _result = None        # runtime PathResult (never serialized)
    _path_ids = None      # ids of the viewport ray-path objects this component owns

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
    draw_paths = bool_field(True, description="Draw ray poly-lines in the viewport.")
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
        # toast (the dispatcher otherwise logs to the Console only).
        try:
            self._result = await run_off_loop(PathSolveRunner.run, scene, self,
                                              transmitter=transmitter, receiver=receiver)
            self._plot()
            drawn = self._draw(scene) if bool(self.draw_paths) else 0
        except Exception as exc:  # noqa: BLE001 - GPU/render boundary: surface, don't swallow
            Notifications.error("Channel", f"path solve failed: {exc}")
            return f"Path solve failed: {exc}"
        Notifications.success("Channel", f"Path solve {transmitter} -> {receiver}: {drawn} paths")
        return f"Path solve complete ({drawn} paths)"

    @button(display_name="Update Plot")
    def update_plot(self):
        if self._result is None:
            Notifications.warning("Channel", "Solve paths first.")
            return "Solve first"
        try:
            self._plot()
        except Exception as exc:  # noqa: BLE001 - surface as a toast
            Notifications.error("Channel", f"plot failed: {exc}")
            return f"Plot failed: {exc}"
        Notifications.success("Channel", f"{self.plot_mode} plot updated")
        return "Plot updated"

    @button(display_name="Draw Paths")
    def draw(self):
        if self._result is None:
            Notifications.warning("Channel", "Solve paths first.")
            return "Solve first"
        try:
            drawn = self._draw(self._owner.scene)
        except Exception as exc:  # noqa: BLE001 - surface as a toast
            Notifications.error("Channel", f"draw failed: {exc}")
            return f"Draw failed: {exc}"
        Notifications.success("Channel", f"Drew {drawn} ray paths")
        return f"Drew {drawn} paths"

    # --- internals (no notifications; shared by the buttons + solve) ----------

    def _plot(self):
        mode = str(self.plot_mode)
        if mode == "cfr":
            data = PathPlots.cfr(self._result)
        elif mode == "taps":
            data = PathPlots.taps(self._result, bandwidth=float(self.bandwidth),
                                  num_taps=int(self.num_taps))
        else:
            data = PathPlots.cir(self._result)
        fig = self.path_figure.clear()
        # Pass the ndarrays directly: Figure._to_list has a fast path for them.
        # list(np.ndarray) would unwrap to a list of numpy scalars, which JSON
        # can't encode and which used to silently break the broadcast set.
        (fig.stem if data.kind == "stem" else fig.line)(data.x, data.y)
        fig.title(data.title)

    def _draw(self, scene) -> int:
        self._clear_paths(scene)
        ids = []
        for i, line in enumerate(PathLineBuilder.build(self._result)):
            obj = SceneObject(name=f"Channel Path {i}", mesh_type="Empty")
            scene.add_object(obj)
            spec = RenderSpec(lines_geometry(line.positions), pbr_material(color=line.color))
            api.viewport.set_render_spec(obj.id, spec.to_dict())
            ids.append(obj.id)
        self._path_ids = ids
        return len(ids)

    def _clear_paths(self, scene):
        # Remove the ray-path objects from a previous solve so they don't pile up.
        for object_id in (self._path_ids or []):
            api.viewport.clear_render_spec(object_id)
            if scene.get_object(object_id) is not None:
                scene.remove_object(object_id)
        self._path_ids = []

    def _resolve(self, field_name, component_name):
        # Use the referenced endpoint object if assigned; else the first matching one.
        ref = self.get_referenced_component(field_name)
        if ref is not None and ref._owner is not None:
            return ref._owner.name
        for obj in self._owner.scene.objects.values():
            if obj.get_component(component_name) is not None:
                return obj.name
        return None
