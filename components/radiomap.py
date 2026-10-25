"""RadioMap — a 2D ``Field`` that also solves and displays a channel radiomap.

``RadioMap`` **subclasses the core ``Field`` component**, so it inherits the editor's
field surface for free: the viewport border / handle gizmo (``FieldGizmo``), the 2D
plane visualizer (``FieldVisualizer``, matched via ``inheritsFrom('Field')``), and the
region fields (``min_bound`` / ``max_bound`` / ``plane`` / ``resolution`` / ``colormap`` /
``display_range``). On top of that it adds the scene config (device / noise — frequency
is the inherited ``Field`` field), the solver config (deterministic + Monte Carlo + edge
policy), and a **Solve** button. The region *is* the receiver grid (in the object's local
frame, so the grid follows the Transform), and the solved metric is written back into the
field's own plane data — the radiomap shows as the colored 2D plane inside its border.

Live solve needs CUDA RayD/OptiX (the round-trip / config tests run on CPU).
"""
import asyncio

import numpy as np

from witwin_server import Notifications
from witwin_server.api import api
from witwin_server.core.components import (
    bool_field,
    button,
    component,
    component_field,
    define_group,
    float_field,
    int_field,
    string_field,
    tab_group,
    vector3_field,
)
from witwin_server.core.components.field_region import FREQUENCY_UNITS, FieldComponent
from witwin_server.features.platform.bridge import FieldViz

from ..adapter.common import json_value
from ..adapter.solve import RadioMapExtractor
from .category import CATEGORY

_DET = {"field_name": "solver", "operator": "eq", "value": "deterministic"}
_MC = {"field_name": "solver", "operator": "eq", "value": "montecarlo"}
_STRAT = {"field_name": "det_quadrature_mode", "operator": "eq", "value": "stratified_fixed_n"}
_SCENE = "Channel"
# dB cells at or below this received nothing (the map floors 0 at -300 dB).
_NO_COVERAGE_DB = -250.0
_GRID = "Grid"
_DET_TUNING = "Deterministic Tuning"
_MC_INTEGRATOR = "Monte Carlo Integrator"
_MC_FILTER = "Monte Carlo Filtering"
_MC_TUNING = "Monte Carlo Tuning"
_EDGE = "Edge Policy"
TX_COMPONENT = "ChannelTx"

# plane -> (normal axis index, tangential0 index, tangential1 index).
PLANE_AXES = {"XY": (2, 0, 1), "XZ": (1, 0, 2), "YZ": (0, 1, 2)}


@component(name="RadioMap", display_name="RadioMap", category=CATEGORY)
class RadioMapComponent(FieldComponent):
    """A 2D ``Field`` that solves a channel radiomap and shows it as its plane."""

    define_group(tab_group(_GRID, "Resolution", "PPW", display_name="Grid", sync_field="grid_mode"))

    _result = None   # runtime RadioMapResult (never serialized)
    _solver_result_handle = ""
    _solver_run_id = ""

    # --- lock the inherited Field to a 2D XZ-plane solvable grid ---
    dimension = string_field("2D", options=["2D", "3D"], hidden=True)
    plane = string_field("XZ", options=["XY", "XZ", "YZ"], hidden=True)  # locked horizontal (Y up)
    render_mode = string_field("Slices", options=["None", "Slices", "Volume", "SDF", "Vector"],
                               hidden=True)
    frequency_dependent_resolution = bool_field(False, hidden=True)
    frequency = float_field(3.5e9, min=1e3, max=300e9, step=0.1, units=FREQUENCY_UNITS,
                            default_unit="GHz", group="Settings",
                            description="Scene carrier frequency (also re-evaluates ITU materials).")
    # A flat 20 m square on the horizontal XZ plane (Y is the normal/up axis).
    min_bound = vector3_field([-10.0, 0.0, -10.0], group="Settings", description="Min corner (local).")
    max_bound = vector3_field([10.0, 0.0, 10.0], group="Settings", description="Max corner (local).")
    grid_mode = string_field("Resolution", options=["Resolution", "PPW"], hidden=True)
    resolution = vector3_field([256.0, 1.0, 256.0], group=f"{_GRID}.Resolution",
                               description="Grid resolution per axis (tangential axes used).")
    points_per_wavelength = float_field(4.0, min=1e-6, group=f"{_GRID}.PPW", title="PPW",
                                        description="Receiver-grid points per wavelength.")

    # --- scene config (frequency is the inherited Field field) ---
    device = string_field("cuda", options=["cuda", "cpu"], enum_toggle=True, group=_SCENE,
                          description="Solve device. Live solve needs CUDA RayD.")
    use_noise_power = bool_field(False, group=_SCENE, title="Channel")
    noise_power = float_field(1e-13, min=0.0, group=_SCENE, show_if="use_noise_power",
                             description="Linear noise power for the deterministic SINR metric.")

    # --- solver selection + transmitter ---
    solver = string_field("montecarlo", options=["deterministic", "montecarlo"],
                          enum_toggle=True, group=_SCENE, title="Solve",
                          description="Monte Carlo is the robust default; deterministic needs its OptiX pipeline.")
    transmitter = component_field("ChannelTx", group=_SCENE,
                                  description="Transmitter object (empty = first Tx in the scene).")
    metric = string_field("path_gain", options=["path_gain", "rss", "sinr"], enum_toggle=True,
                          group=_SCENE, description="Radiomap metric to display.")
    tx_index = int_field(0, min=0, group=_SCENE, description="Transmitter index (multi-Tx results).")
    log_scale = bool_field(True, group=_SCENE, description="Show the map in dB (10*log10).")
    send_to_results = bool_field(False, group=_SCENE,
                                 description="Also publish to the versioned Results panel.")

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
    mc_accumulation_backend = string_field("auto",
                                           options=["auto", "native_monte_carlo",
                                                    "rayd_reflection_accumulation"],
                                           group=_MC_INTEGRATOR, show_if=_MC,
                                           description="Basic integrator accumulation backend.")
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
    ep_edge_diffraction = string_field("on", options=["default", "on", "off"], group=_EDGE,
                                       description="Boundary-edge diffraction. Use default to let the solver decide.")

    @button(display_name="Solve", group=_SCENE, operation="solver", cancellable=True,
            progress_surface="component")
    async def solve(self):
        transmitter = self._resolve_transmitter(self._owner.scene)
        if transmitter is None:
            Notifications.error("Channel", "No transmitter found (add a Tx, or assign one).")
            return "Transmitter not found"
        return await self.solve_for(transmitter)

    async def solve_for(self, transmitter: str):
        """Solve this map for the named Tx object (the Solve button and the Agent tool)."""
        from ..adapter.channel_adapter import ChannelAdapter

        scene = self._owner.scene
        if not ChannelAdapter.has_structures(scene):
            Notifications.error("Channel", "Radiomap solve needs at least one structure/mesh "
                                "(e.g. a ground plane or building). Add geometry to the scene.")
            return "No structures in scene"
        try:
            run = await asyncio.to_thread(
                api.solvers.solve,
                "witwin.channel.radiomap",
                scene=scene,
                config=self._solver_config(transmitter, self._owner.name),
            )
            if run.status != "succeeded":
                message = (run.error or {}).get("message") or "radiomap solve failed"
                Notifications.error("Channel", message)
                return f"Solve failed: {message}"
            handle = (run.outputs or {}).get("resultHandle")
            if not handle:
                Notifications.error("Channel", "Solver returned no result handle")
                return "Solve failed: no result handle"
            self._solver_result_handle = str(handle)
            self._solver_run_id = run.run_id
            self._result = None
            self.update_map()
        except Exception as exc:  # noqa: BLE001 - GPU solver boundary: notify, don't crash the button
            Notifications.error("Channel", f"{self.solver} solve failed: {exc}")
            return f"Solve failed: {exc}"
        await asyncio.sleep(0)  # flush the field-update broadcasts on the (free) loop
        Notifications.success("Channel", f"{str(self.solver)} radiomap solved")
        return "Solve complete"

    @button(display_name="Update Map", group=_SCENE)
    def update_map(self):
        if self._solver_result_handle:
            payload = self._query_result("extract_map", {
                "metric": str(self.metric),
                "tx_index": int(self.tx_index),
                "bounds": [0.0, 1.0, 0.0, 1.0],
                "log_scale": bool(self.log_scale),
            })
            data = np.asarray(payload["data"], dtype=np.float32)
            label = payload.get("label") or str(self.metric)
        elif self._result is None:
            return "Solve first"
        else:
            img = RadioMapExtractor.extract(self._result, metric=str(self.metric),
                                            tx_index=int(self.tx_index), bounds=[0.0, 1.0, 0.0, 1.0],
                                            log_scale=bool(self.log_scale))
            data = img.data
            label = img.label
        self.set_volume_data(self._to_plane_volume(self._display_values(data)))  # normalizes + sets resolution
        self.set_field("render_mode", "Slices", force=True)
        self.export_volume_data()                              # caches -> volume resource -> plane shows
        if bool(self.send_to_results):
            FieldViz.show_2d(data, title=label)
        return "Map updated"

    def _display_values(self, data2d: np.ndarray) -> np.ndarray:
        """The map as the plane should color it.

        The plane normalizes min..max. In dB, cells that received nothing sit at the
        -300 dB floor (log of 0) and would squeeze every covered cell into the top
        colors, so the window is the covered cells' 1st percentile..max and uncovered
        cells take the lowest color.
        """
        data = np.asarray(data2d, dtype=np.float32)
        if not bool(self.log_scale):
            return data
        covered = np.isfinite(data) & (data > _NO_COVERAGE_DB)
        if not covered.any():
            return data
        low = np.float32(np.percentile(data[covered], 1.0))
        return np.where(covered, np.clip(data, low, None), low).astype(np.float32)

    def _to_plane_volume(self, data2d: np.ndarray) -> np.ndarray:
        """Embed the solved map into a thin world-axis volume aligned to the field region.

        The platform radiomap tensor is ``(n_bounds1, n_bounds0)`` (rows = second
        tangential / ``v`` axis, cols = first tangential / ``u`` axis), so ``data2d[iv, iu]``
        is the value at tangential cell ``(u=iu, v=iv)``. The Field volume is indexed by
        world axes ``(X, Y, Z)``; ``t0``/``t1`` are the plane's world tangential-axis
        indices (``u``/``v``) and ``normal`` is its thin axis. Transposing ``data2d`` lines
        the ``u`` axis up with ``t0`` and ``v`` with ``t1`` so the map matches the scene.
        """
        normal, t0, t1 = PLANE_AXES[str(self.plane)]
        n_v, n_u = int(data2d.shape[0]), int(data2d.shape[1])
        shape = [1, 1, 1]
        shape[t0], shape[t1] = n_u, n_v
        vol = np.zeros(shape, dtype=np.float32)
        np.moveaxis(vol, (t0, t1, normal), (0, 1, 2))[:, :, 0] = np.asarray(data2d, dtype=np.float32).T
        return vol

    def _resolve_transmitter(self, scene):
        # Use the referenced Tx object if set; else the first Tx object in the scene.
        ref = self.get_referenced_component("transmitter")
        if ref is not None and ref._owner is not None:
            return ref._owner.name
        for obj in scene.objects.values():
            if obj.get_component(TX_COMPONENT) is not None:
                return obj.name
        return None

    def _query_result(self, op: str, params: dict):
        response = api.solvers.query(
            "witwin.channel.radiomap",
            self._solver_result_handle,
            op,
            params,
            run_id=self._solver_run_id or None,
        )
        return response.get("data") or {}

    def _solver_config(self, transmitter: str, receiver: str) -> dict:
        return {
            "mode": "radiomap",
            "transmitter": transmitter,
            "receiver": receiver,
            "component": self._component_config(),
        }

    def _component_config(self) -> dict:
        keys = (
            "frequency", "use_noise_power", "noise_power", "device", "solver",
            "grid_mode", "points_per_wavelength",
            "det_metric", "det_num_samples", "det_max_bounces", "det_max_diffraction_order",
            "det_quadrature_mode", "det_samples_per_cell", "det_shadow_boundary_correction",
            "det_synthetic_array", "det_resolution_wavelength", "det_solver_mode",
            "det_memory_profile", "det_reflection_field_backend",
            "mc_num_samples", "mc_max_bounces", "mc_max_diffraction_order",
            "mc_synthetic_array", "mc_integrator", "mc_samples_per_tx", "mc_seed",
            "mc_bdpt_diffraction_sampling", "mc_accumulation_backend",
            "mc_use_rr_depth", "mc_rr_depth",
            "mc_use_rr_prob", "mc_rr_prob", "mc_use_stop_threshold", "mc_stop_threshold",
            "mc_use_filtering", "mc_filter_method", "mc_filter_radius", "mc_filter_sigma",
            "mc_filter_blend", "mc_resolution_wavelength", "mc_solver_mode",
            "mc_memory_profile", "mc_reflection_field_backend",
            "ep_vertical_ratio", "ep_edge_selection_mode", "ep_edge_diffraction",
        )
        return {key: json_value(getattr(self, key)) for key in keys}
