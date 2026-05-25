"""PathSolve — point-to-point path solve (CIR/CFR/taps) + viewport ray drawing.

Lives on the ``Channel Settings`` object. Solves discrete Tx<->Rx propagation paths
(``path.solve``), keeps the ``PathResult`` server-side, plots the CIR / CFR / taps in
its figure, and (when ``draw_paths``) draws each ray as a declarative poly-line render
spec in the viewport, colored by interaction type (LoS / reflection / diffraction).
``bandwidth`` feeds the discrete-tap synthesis only. Live solve needs CUDA RayD/OptiX.
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
from .settings import CATEGORY


@component(name="PathSolve", category=CATEGORY)
class PathSolveComponent(Component):
    """Discrete Tx<->Rx path solve + CIR/CFR/taps plots + viewport ray poly-lines."""

    _result = None        # runtime PathResult (never serialized)
    _path_ids = None      # ids of the viewport ray-path objects this component owns

    # --- endpoints ---
    transmitter = string_field("tx", title="Path Solve", description="Transmitter name.")
    receiver = string_field("rx", description="Point Receiver name (not a grid).")

    # --- path config ---
    num_samples = int_field(10000, min=1, description="Probe ray count.")
    max_bounces = int_field(2, min=0, description="Reflection depth.")
    max_diffraction_order = int_field(1, min=0, description="UTD diffraction order.")
    use_max_num_paths = bool_field(False, title="Limit paths")
    max_num_paths = int_field(50, min=1, show_if="use_max_num_paths")
    synthetic_array = bool_field(True)

    # --- visualization ---
    draw_paths = bool_field(True, description="Draw ray poly-lines in the viewport.")
    plot_mode = string_field("cir", options=["cir", "cfr", "taps"], enum_toggle=True,
                             title="Plot")
    bandwidth = float_field(100e6, min=1.0, show_if={"field_name": "plot_mode", "operator": "eq",
                                                     "value": "taps"},
                            description="Tap synthesis bandwidth (Hz).")
    num_taps = int_field(64, min=1, show_if={"field_name": "plot_mode", "operator": "eq",
                                             "value": "taps"})
    path_figure = figure(title="Channel Response")

    @button(display_name="Solve Paths")
    def solve(self):
        scene = self._owner.scene
        tx_obj = scene.get_object_by_name(str(self.transmitter))
        if tx_obj is None or tx_obj.get_component("ChannelTransmitter") is None:
            Notifications.error("Channel", f"Transmitter '{self.transmitter}' not found")
            return "Transmitter not found"
        rx_obj = scene.get_object_by_name(str(self.receiver))
        if rx_obj is None or rx_obj.get_component("ChannelReceiver") is None:
            Notifications.error(
                "Channel",
                f"Point Receiver '{self.receiver}' not found "
                "(path solve needs a Receiver, not a ReceiverGrid)")
            return "Receiver not found"
        try:
            self._result = PathSolveRunner.run(scene, self)
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
