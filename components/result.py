"""ChannelResult — solver selection, config, the Solve button, and 2D radiomap viz.

Lives on the ``Channel Settings`` object. It carries both solver families' curated
config (deterministic + Monte Carlo, with advanced foldouts) plus a shared edge
policy, runs the chosen ``solve`` on the rebuilt platform scene, keeps the
``RadioMapResult`` server-side (runtime state, never persisted), and shows the chosen
metric map in its figure (and, optionally, the versioned Results panel via the shared
``FieldViz``). A metric selector + dB toggle re-render without re-solving.

Live solve needs CUDA RayD/OptiX (the round-trip / config tests run on CPU).
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
)
from witwin_server.platform_bridge import FieldViz

from ..adapter.common import vec
from ..adapter.solve import RadioMapExtractor, SolveRunner
from .settings import CATEGORY

_DET = {"field_name": "solver", "operator": "eq", "value": "deterministic"}
_MC = {"field_name": "solver", "operator": "eq", "value": "montecarlo"}
_STRAT = {"field_name": "det_quadrature_mode", "operator": "eq", "value": "stratified_fixed_n"}
_DET_TUNING = "Deterministic Tuning"
_MC_INTEGRATOR = "Monte Carlo Integrator"
_MC_FILTER = "Monte Carlo Filtering"
_MC_TUNING = "Monte Carlo Tuning"
_EDGE = "Edge Policy"


@component(name="ChannelResult", category=CATEGORY)
class ChannelResultComponent(Component):
    """Radiomap solve (deterministic / Monte Carlo) + 2D field visualization."""

    _result = None   # runtime RadioMapResult (never serialized)
    _bounds = None   # tangential extent [min0, max0, min1, max1] of the solved grid

    # --- solver selection + endpoints ---
    solver = string_field("deterministic", options=["deterministic", "montecarlo"],
                          enum_toggle=True, title="Solve")
    transmitter = string_field("tx", description="Transmitter endpoint name.")
    receiver_grid = string_field("rmap", description="ReceiverGrid endpoint name.")

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
    send_to_results = bool_field(True, description="Also publish to the versioned Results panel.")
    result_figure = figure(title="Radiomap")

    @button(display_name="Solve")
    def solve(self):
        scene = self._owner.scene
        grid_obj = scene.get_object_by_name(str(self.receiver_grid))
        if grid_obj is None or grid_obj.get_component("ReceiverGrid") is None:
            Notifications.error("Channel", f"ReceiverGrid '{self.receiver_grid}' not found")
            return "ReceiverGrid not found"
        self._bounds = vec(grid_obj.get_component("ReceiverGrid").bounds)
        self._result = SolveRunner.run(scene, self)
        self.update_map()
        Notifications.success("Channel", f"{str(self.solver)} radiomap solved")
        return "Solve complete"

    @button(display_name="Update Map")
    def update_map(self):
        if self._result is None:
            return "Solve first"
        img = RadioMapExtractor.extract(
            self._result, metric=str(self.metric), tx_index=int(self.tx_index),
            bounds=self._bounds, log_scale=bool(self.log_scale))
        self.result_figure.clear().imshow(img.data).title(img.label)
        if bool(self.send_to_results):
            FieldViz.show_2d(img.data, bounds=((img.extent[0], img.extent[1]),
                                               (img.extent[2], img.extent[3])), title=img.label)
        return "Map updated"
