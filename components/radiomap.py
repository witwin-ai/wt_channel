"""RadioMap — a 2D field that is its own receiver grid, solver, and result viewer.

A ``RadioMap`` object *is* the receiver grid: the ``Field`` group defines the sampling
plane (axis + height + tangential bounds + resolution), and on export the object becomes
a ``ReceiverGrid`` named after itself. It also carries the scene config (frequency /
device / noise) and the full solver config (deterministic + Monte Carlo + edge policy),
so pressing **Solve** rebuilds the platform scene, auto-finds the transmitter, runs the
chosen solver against this grid, and shows the chosen metric as a 2D heatmap (and,
optionally, in the versioned Results panel). A metric / dB / value-range change
re-renders without re-solving. Live solve needs CUDA RayD/OptiX.
"""
from witwin_server import Notifications
from witwin_server.components import (
    Component,
    bool_field,
    button,
    component,
    figure,
    float_field,
    int_field,
    string_field,
    vector4_field,
)
from witwin_server.platform_bridge import FieldViz

from ..adapter.common import num, vec
from ..adapter.solve import RadioMapExtractor, SolveRunner
from .category import CATEGORY

_DET = {"field_name": "solver", "operator": "eq", "value": "deterministic"}
_MC = {"field_name": "solver", "operator": "eq", "value": "montecarlo"}
_STRAT = {"field_name": "det_quadrature_mode", "operator": "eq", "value": "stratified_fixed_n"}
_GRID_SHAPE = {"field_name": "grid_mode", "operator": "eq", "value": "grid_shape"}
_CELL_SIZE = {"field_name": "grid_mode", "operator": "eq", "value": "cell_size"}
_SCENE = "Scene"
_DET_TUNING = "Deterministic Tuning"
_MC_INTEGRATOR = "Monte Carlo Integrator"
_MC_FILTER = "Monte Carlo Filtering"
_MC_TUNING = "Monte Carlo Tuning"
_EDGE = "Edge Policy"
TX_COMPONENT = "ChannelTx"


@component(name="RadioMap", display_name="RadioMap", category=CATEGORY)
class RadioMapComponent(Component):
    """2D radiomap: receiver grid + scene/solver config + solve + field visualization."""

    _result = None   # runtime RadioMapResult (never serialized)
    _bounds = None   # tangential extent [min0, max0, min1, max1] of the solved grid

    # --- scene config ---
    frequency = float_field(3.5e9, min=0.0, group=_SCENE,
                            description="Scene carrier frequency in Hz (re-evaluates ITU materials).")
    device = string_field("cuda", options=["cuda", "cpu"], enum_toggle=True, group=_SCENE,
                          description="Solve device. Live solve needs CUDA RayD.")
    use_noise_power = bool_field(False, group=_SCENE, title="Noise")
    noise_power = float_field(1e-13, min=0.0, group=_SCENE, show_if="use_noise_power",
                             description="Linear noise power for the deterministic SINR metric.")

    # --- field / receiver grid ---
    axis = string_field("z", options=["x", "y", "z"], enum_toggle=True, title="Field",
                        description="Plane normal axis.")
    height = float_field(1.5, description="Plane offset along the axis (m).")
    bounds = vector4_field([-10.0, 10.0, -10.0, 10.0],
                           description="Tangential extents [min0, max0, min1, max1].")
    grid_mode = string_field("grid_shape", options=["grid_shape", "cell_size"], enum_toggle=True,
                             description="Discretize by cell counts or by cell size.")
    grid_nx = int_field(64, min=1, show_if=_GRID_SHAPE, horizontal_group="Grid shape")
    grid_ny = int_field(64, min=1, show_if=_GRID_SHAPE, horizontal_group="Grid shape")
    cell_dx = float_field(0.5, min=1e-6, show_if=_CELL_SIZE, horizontal_group="Cell size")
    cell_dy = float_field(0.5, min=1e-6, show_if=_CELL_SIZE, horizontal_group="Cell size")

    # --- solver selection + transmitter ---
    solver = string_field("montecarlo", options=["deterministic", "montecarlo"],
                          enum_toggle=True, title="Solve",
                          description="Monte Carlo is the robust default; deterministic needs its OptiX pipeline to compile.")
    transmitter = string_field("", description="Transmitter name (blank = first Tx in the scene).")

    # --- deterministic config ---
    det_metric = string_field("path_gain", options=["path_gain", "rss", "sinr"], show_if=_DET,
                              title="Deterministic")
    det_num_samples = int_field(10000, min=1, show_if=_DET, description="Reflection ray count.")
    det_max_bounces = int_field(2, min=0, show_if=_DET, description="Reflection depth.")
    det_max_diffraction_order = int_field(1, min=0, show_if=_DET, description="UTD diffraction order.")
    det_quadrature_mode = string_field("center", options=["center", "stratified_fixed_n"], show_if=_DET)
    det_samples_per_cell = int_field(4, min=1, show_if=_STRAT)
    det_shadow_boundary_correction = bool_field(False, show_if=_DET,
                                                description="Requires center quadrature.")
    det_synthetic_array = bool_field(True, show_if=_DET, description="Far-field synthetic array.")
    det_resolution_wavelength = float_field(0.125, min=1e-6, group=_DET_TUNING, show_if=_DET,
                                            description="cell_size = this x wavelength.")
    det_solver_mode = string_field("accuracy", options=["accuracy", "fast_approximate"],
                                   group=_DET_TUNING, show_if=_DET)
    det_memory_profile = string_field("default", options=["default", "memory_safe"],
                                      group=_DET_TUNING, show_if=_DET)
    det_reflection_field_backend = string_field("native", options=["drjit", "native"],
                                                group=_DET_TUNING, show_if=_DET)

    # --- monte carlo config ---
    mc_num_samples = int_field(10000, min=1, show_if=_MC, title="Monte Carlo")
    mc_max_bounces = int_field(2, min=0, show_if=_MC)
    mc_max_diffraction_order = int_field(1, min=0, max=3, show_if=_MC,
                                         description="0-3 (basic integrator clamps to 0/1).")
    mc_synthetic_array = bool_field(True, show_if=_MC)
    mc_integrator = string_field("basic", options=["basic", "bdpt"], enum_toggle=True,
                                 group=_MC_INTEGRATOR, show_if=_MC)
    mc_samples_per_tx = int_field(65536, min=1, group=_MC_INTEGRATOR, show_if=_MC,
                                  description="Rays per transmitter.")
    mc_seed = int_field(0, min=0, group=_MC_INTEGRATOR, show_if=_MC)
    mc_bdpt_diffraction_sampling = string_field("sobol", options=["hash", "sobol"],
                                                group=_MC_INTEGRATOR, show_if=_MC)
    mc_use_rr_depth = bool_field(False, group=_MC_INTEGRATOR, show_if=_MC, title="Russian roulette")
    mc_rr_depth = int_field(0, min=0, group=_MC_INTEGRATOR, show_if="mc_use_rr_depth")
    mc_use_rr_prob = bool_field(False, group=_MC_INTEGRATOR, show_if=_MC)
    mc_rr_prob = float_field(0.95, min=1e-6, max=1.0, group=_MC_INTEGRATOR, show_if="mc_use_rr_prob")
    mc_use_stop_threshold = bool_field(False, group=_MC_INTEGRATOR, show_if=_MC)
    mc_stop_threshold = float_field(0.0, min=0.0, group=_MC_INTEGRATOR, show_if="mc_use_stop_threshold")
    mc_use_filtering = bool_field(False, group=_MC_FILTER, show_if=_MC,
                                  description="Differentiable power-domain filtering.")
    mc_filter_method = string_field("gaussian", options=["gaussian", "bilateral"],
                                    group=_MC_FILTER, show_if="mc_use_filtering")
    mc_filter_radius = int_field(1, min=0, group=_MC_FILTER, show_if="mc_use_filtering")
    mc_filter_sigma = float_field(1.0, min=1e-6, group=_MC_FILTER, show_if="mc_use_filtering")
    mc_filter_blend = float_field(1.0, min=0.0, max=1.0, group=_MC_FILTER, show_if="mc_use_filtering")
    mc_resolution_wavelength = float_field(0.125, min=1e-6, group=_MC_TUNING, show_if=_MC)
    mc_solver_mode = string_field("accuracy", options=["accuracy", "fast_approximate"],
                                  group=_MC_TUNING, show_if=_MC)
    mc_memory_profile = string_field("default", options=["default", "memory_safe"],
                                     group=_MC_TUNING, show_if=_MC)
    mc_reflection_field_backend = string_field("native", options=["drjit", "native"],
                                               group=_MC_TUNING, show_if=_MC)

    # --- shared edge policy ---
    ep_vertical_ratio = float_field(0.7, min=0.0, max=1.0, group=_EDGE)
    ep_edge_selection_mode = string_field("all_edges", options=["all_edges", "vertical_only"],
                                          group=_EDGE)
    ep_edge_diffraction = string_field("default", options=["default", "on", "off"], group=_EDGE,
                                       description="Boundary-edge diffraction (default lets the solver decide).")

    # --- result visualization ---
    metric = string_field("path_gain", options=["path_gain", "rss", "sinr"], enum_toggle=True,
                          title="Radiomap")
    tx_index = int_field(0, min=0, description="Transmitter index (multi-Tx results).")
    log_scale = bool_field(True, description="Show the map in dB (10*log10).")
    auto_range = bool_field(True, title="Value range",
                            description="Autoscale the color range; uncheck to window it (no re-solve).")
    range_min = float_field(-120.0, hide_if="auto_range", horizontal_group="Range",
                            description="Min displayed value (dB if log scale).")
    range_max = float_field(-40.0, hide_if="auto_range", horizontal_group="Range",
                            description="Max displayed value (dB if log scale).")
    send_to_results = bool_field(True, description="Also publish to the versioned Results panel.")
    result_figure = figure(title="Radiomap")

    @button(display_name="Solve")
    def solve(self):
        scene = self._owner.scene
        transmitter = self._resolve_transmitter(scene)
        if transmitter is None:
            Notifications.error("Channel", "No transmitter found (add a Tx, or set its name).")
            return "Transmitter not found"
        self._bounds = vec(self.bounds)
        try:
            self._result = SolveRunner.run(scene, self, transmitter=transmitter,
                                           receiver=self._owner.name)
        except Exception as exc:  # noqa: BLE001 - GPU solver boundary: notify, don't crash the button
            Notifications.error("Channel", f"{self.solver} solve failed: {exc}")
            return f"Solve failed: {exc}"
        self.update_map()
        Notifications.success("Channel", f"{str(self.solver)} radiomap solved")
        return "Solve complete"

    @button(display_name="Update Map")
    def update_map(self):
        if self._result is None:
            return "Solve first"
        auto = bool(self.auto_range)
        img = RadioMapExtractor.extract(
            self._result, metric=str(self.metric), tx_index=int(self.tx_index),
            bounds=self._bounds, log_scale=bool(self.log_scale),
            vmin=None if auto else float(num(self.range_min)),
            vmax=None if auto else float(num(self.range_max)))
        self.result_figure.clear().imshow(img.data).title(img.label)
        if bool(self.send_to_results):
            FieldViz.show_2d(img.data, bounds=((img.extent[0], img.extent[1]),
                                               (img.extent[2], img.extent[3])), title=img.label)
        return "Map updated"

    def _resolve_transmitter(self, scene):
        # Use the named transmitter if set + valid; else the first Tx object in the scene.
        name = str(self.transmitter).strip()
        if name:
            obj = scene.get_object_by_name(name)
            if obj is not None and obj.get_component(TX_COMPONENT) is not None:
                return name
            return None
        for obj in scene.objects.values():
            if obj.get_component(TX_COMPONENT) is not None:
                return obj.name
        return None
