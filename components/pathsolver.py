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

    # --- endpoints (blank = auto-find in the scene) ---
    transmitter = string_field("", title="Solve", description="Transmitter name (blank = first Tx).")
    receiver = string_field("", description="Point Receiver name (blank = first Rx; not a RadioMap).")

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
    def solve(self):
        from ..adapter.channel_adapter import ChannelAdapter

        scene = self._owner.scene
        transmitter = self._resolve(scene, self.transmitter, TX_COMPONENT)
        if transmitter is None:
            Notifications.error("Channel", "No transmitter found (add a Tx, or set its name).")
            return "Transmitter not found"
        receiver = self._resolve(scene, self.receiver, RX_COMPONENT)
        if receiver is None:
            Notifications.error("Channel", "No point Receiver found (add an Rx, or set its name).")
            return "Receiver not found"
        if not ChannelAdapter.has_structures(scene):
            Notifications.error("Channel", "Path solve needs at least one structure/mesh "
                                "(e.g. a ground plane or building). Add geometry to the scene.")
            return "No structures in scene"
        try:
            self._result = PathSolveRunner.run(scene, self, transmitter=transmitter,
                                               receiver=receiver)
        except Exception as exc:  # noqa: BLE001 - GPU solver boundary: notify, don't crash the button
            Notifications.error("Channel", f"path solve failed: {exc}")
            return f"Path solve failed: {exc}"
        self.update_plot()
        if bool(self.draw_paths):
            self.draw()
        Notifications.success("Channel", "Path solve complete")
        return "Path solve complete"

    @button(display_name="Update Plot")
    def update_plot(self):
        if self._result is None:
            return "Solve first"
        mode = str(self.plot_mode)
        if mode == "cfr":
            data = PathPlots.cfr(self._result)
        elif mode == "taps":
            data = PathPlots.taps(self._result, bandwidth=float(self.bandwidth),
                                  num_taps=int(self.num_taps))
        else:
            data = PathPlots.cir(self._result)
        fig = self.path_figure.clear()
        (fig.stem if data.kind == "stem" else fig.line)(list(data.x), list(data.y))
        fig.title(data.title)
        return "Plot updated"

    @button(display_name="Draw Paths")
    def draw(self):
        if self._result is None:
            return "Solve first"
        scene = self._owner.scene
        self._clear_paths(scene)
        ids = []
        for i, line in enumerate(PathLineBuilder.build(self._result)):
            obj = SceneObject(name=f"Channel Path {i}", mesh_type="Empty")
            scene.add_object(obj)
            spec = RenderSpec(lines_geometry(line.positions), pbr_material(color=line.color))
            api.viewport.set_render_spec(obj.id, spec.to_dict())
            ids.append(obj.id)
        self._path_ids = ids
        return f"Drew {len(ids)} paths"

    def _clear_paths(self, scene):
        # Remove the ray-path objects from a previous solve so they don't pile up.
        for object_id in (self._path_ids or []):
            api.viewport.clear_render_spec(object_id)
            if scene.get_object(object_id) is not None:
                scene.remove_object(object_id)
        self._path_ids = []

    @staticmethod
    def _resolve(scene, field_value, component_name):
        # Use the named endpoint if set + valid; else the first matching object in the scene.
        name = str(field_value).strip()
        if name:
            obj = scene.get_object_by_name(name)
            if obj is not None and obj.get_component(component_name) is not None:
                return name
            return None
        for obj in scene.objects.values():
            if obj.get_component(component_name) is not None:
                return obj.name
        return None
