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
import time
import uuid

import numpy as np

from witwin_server.api import (
    Notifications,
    api,
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
    FREQUENCY_UNITS,
    FieldComponent,
    VOLUME_FLOAT32_FORMAT,
    FieldViz,
)

from ..adapter.common import json_value
from ..adapter.live import LiveTxScheduler
from ..adapter.solve import RadioMapExtractor
from .category import CATEGORY

_DET = {"field_name": "solver", "operator": "eq", "value": "deterministic"}
_MC = {"field_name": "solver", "operator": "eq", "value": "montecarlo"}
_STRAT = {"field_name": "det_quadrature_mode", "operator": "eq", "value": "stratified_fixed_n"}
_LIVE_DET = {"field_name": "live_preview", "operator": "eq", "value": "deterministic"}
_LIVE_MC = {"field_name": "live_preview", "operator": "eq", "value": "montecarlo"}
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
# RadioMap fields that change the receiver grid the solver host has cached.
_GRID_FIELDS = frozenset({"min_bound", "max_bound", "resolution", "grid_mode",
                          "points_per_wavelength", "plane", "dimension", "frequency"})
_SCENE_EVENTS = ("object_updated", "component_updated", "object_added", "object_removed",
                 "component_added", "component_removed")

# plane -> (normal axis index, tangential0 index, tangential1 index).
PLANE_AXES = {"XY": (2, 0, 1), "XZ": (1, 0, 2), "YZ": (0, 1, 2)}


# A "CUDA error" leaves the solver host's CUDA context broken; the host exits after
# reporting it (solver_host._exit_after_reply) and the solve is tried once more in a
# fresh host, after this pause for the old one to exit.
_CUDA_RETRY_DELAY_S = 1.5


def _cuda_error(message: str) -> bool:
    return "cuda error" in str(message).lower()


def watch_scene_radiomaps(scene, **_) -> None:
    """Arm live Tx drag for every RadioMap of a Scene that was just loaded.

    A map loaded from disk shows at once (see ``_export_map``); dragging its Tx did
    nothing until a solve in the current run (Eiffel Tower). Called on
    ``scene_loaded`` and for the Scenes already open when the plugin activates.
    """
    for obj in list(getattr(scene, "objects", {}).values()):
        comp = obj.get_component("RadioMap")
        if comp is not None:
            comp._watch_scene(scene)


@component(name="RadioMap", display_name="RadioMap", category=CATEGORY)
class RadioMapComponent(FieldComponent):
    """A 2D ``Field`` that solves a channel radiomap and shows it as its plane."""

    define_group(tab_group(_GRID, "Resolution", "PPW", display_name="Grid", sync_field="grid_mode"))

    _result = None   # runtime RadioMapResult (never serialized)
    _solver_result_handle = ""
    _solver_run_id = ""
    # Live Tx drag (runtime only): the scheduler, the scene subscribed to, the event
    # loop the solves run on, the solver host's cached-scene key and the live run.
    _live = None
    _live_scene = None
    _live_loop = None
    _live_tx_name = ""
    _live_run_id = ""
    _scene_key = ""
    _explicit_solving = False
    _applying_map = False
    _ground_cache = None  # (scene_key, world vertices) for the below-ground Tx check
    # Whether the next published map is a full-grid result (kept on disk for the next
    # start) or a live preview / coarse pass (memory only), and the files kept (oldest
    # first): the saved Scene may still point at the previous one until its autosave.
    _persist_map = False
    _persisted_assets = ()

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
    live_update = bool_field(True, group=_SCENE, title="Live",
                             description="After a solve, follow a dragged Tx: Monte Carlo "
                                         "previews, then one solve with this solver once it stops.")
    live_preview = string_field("deterministic", options=["deterministic", "montecarlo"],
                                enum_toggle=True, group=_SCENE, show_if="live_update",
                                description="Previews while dragging: coarse deterministic "
                                            "(shows diffraction) or Monte Carlo basic.")
    live_preview_resolution = int_field(64, min=8, group=_SCENE, show_if=_LIVE_DET,
                                        description="Cells per side of a deterministic preview.")
    live_preview_bounces = int_field(1, min=1, group=_SCENE, show_if=_LIVE_DET,
                                     description="Reflections in a deterministic preview "
                                                 "(UCSD 64x64: 1 -> ~0.4 s, 2 -> ~1.0 s).")
    live_preview_samples = int_field(262144, min=1024, group=_SCENE, show_if=_LIVE_MC,
                                     description="Monte Carlo basic rays per preview.")
    live_settle_s = float_field(1.5, min=0.0, group=_SCENE, show_if="live_update",
                                description="Seconds the Tx must be still before the full solve.")
    live_refine_coarse = int_field(128, min=0, group=_SCENE, show_if="live_update",
                                   description="Cells per side of a quick first refine pass "
                                               "(0: refine on the full grid only).")
    live_status = string_field("", group=_SCENE, show_if="live_update", readonly=True,
                               description="What the live update is doing: previewing, "
                                           "refining with this solver, or done.")
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
        self._live_prepare(transmitter)
        self._explicit_solving = True
        try:
            return await self._solve_explicit(transmitter)
        finally:
            self._explicit_solving = False

    async def _solve_explicit(self, transmitter: str):
        from ..adapter.channel_adapter import ChannelAdapter

        scene = self._owner.scene
        if not ChannelAdapter.has_structures(scene):
            Notifications.error("Channel", "Radiomap solve needs at least one structure/mesh "
                                "(e.g. a ground plane or building). Add geometry to the scene.")
            return "No structures in scene"
        tx = scene.get_object_by_name(transmitter)
        if tx is not None and self._below_ground(scene, tx.get_component("Transform").position):
            Notifications.error("Channel", "Radio map not solved: the Tx is below the ground. Move it up.")
            return "Not solved: the Tx is below the ground"
        try:
            # The full Scene goes out and seeds the solver host's cache for live previews.
            scene_key = uuid.uuid4().hex
            config = self._solver_config(transmitter, self._owner.name)
            config["scene_key"] = scene_key
            self._scene_key = ""
            for attempt in range(2):
                run = await asyncio.to_thread(
                    api.solvers.solve,
                    "witwin.channel.radiomap",
                    scene=scene,
                    config=config,
                )
                if run.status == "succeeded" or attempt or not _cuda_error((run.error or {}).get("message")):
                    break
                await asyncio.sleep(_CUDA_RETRY_DELAY_S)
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
            self._scene_key = scene_key
            self._result = None
            self._persist_map = True
            self.update_map()
        except Exception as exc:  # noqa: BLE001 - GPU solver boundary: notify, don't crash the button
            Notifications.error("Channel", f"{self.solver} solve failed: {exc}")
            return f"Solve failed: {exc}"
        await asyncio.sleep(0)  # flush the field-update broadcasts on the (free) loop
        return "Solve complete"

    # --- live Tx drag ----------------------------------------------------------

    def _live_prepare(self, transmitter: str) -> None:
        """Before an explicit solve: drop pending live work and watch this scene."""
        self._live_tx_name = transmitter
        self._live_loop = asyncio.get_running_loop()
        if self._live is not None:
            self._live.reset()
        self._watch_scene(self._owner.scene)

    def _watch_scene(self, scene) -> None:
        if self._live_scene is scene:
            return
        if self._live_scene is not None:
            for event in _SCENE_EVENTS:
                self._live_scene.off(event, self._on_scene_event)
        for event in _SCENE_EVENTS:
            scene.on(event, self._on_scene_event)
        self._live_scene = scene

    def _on_scene_event(self, scene, object_id=None, object=None, update_type=None,
                        component_name=None, field_name=None, **_):
        if self._owner is None or self._owner.scene is not scene:
            return
        obj = object if object is not None else (scene.get_object(object_id) if object_id else None)
        if not self._live_tx_name and obj is not None and obj.get_component(TX_COMPONENT) is not None:
            self._live_tx_name = self._resolve_transmitter(scene) or ""
        is_tx = obj is not None and obj.name == self._live_tx_name
        if is_tx and (update_type == "transform" or component_name == "Transform"):
            if update_type == "transform":
                self._tx_moved()
            return
        if obj is self._owner and component_name is not None and (
                self._applying_map or field_name not in _GRID_FIELDS):
            return  # this map's own display state (the solved plane data)
        self._scene_key = ""  # any other edit: the solver host's cached scene is stale

    def _tx_moved(self) -> None:
        shows_map = bool(self._solver_result_handle or (self._values.get("volume") or {}).get("assetId"))
        if not (bool(self.live_update) and shows_map) or self._explicit_solving:
            return
        if self._live_loop is None:  # no solve in this run yet: a map loaded from disk
            try:
                self._live_loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        if self._live is None:
            self._live = LiveTxScheduler(self._live_preview, self._live_refine, self._live_cancel)
        self._live.settle_s = float(self.live_settle_s)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._live_loop:
            self._live.moved()
        elif self._live_loop is not None:
            self._live_loop.call_soon_threadsafe(self._live.moved)

    def _set_live_status(self, text: str) -> None:
        self.set_field("live_status", text, force=True)

    async def _live_preview(self) -> None:
        config = self._solver_config(self._live_tx_name, self._owner.name)
        if str(self.live_preview) == "montecarlo":
            self._set_live_status("Previewing (Monte Carlo); the full solve runs once the Tx stops")
            config["component"].update(solver="montecarlo", mc_integrator="basic",
                                       mc_samples_per_tx=int(self.live_preview_samples))
        else:
            # A coarse deterministic grid keeps diffraction visible while dragging (UCSD
            # 64x64, one reflection: ~0.4 s, 78 % of shadow cells lit vs ~12 % for Monte
            # Carlo basic at 256k rays); the refine restores the full grid and bounces.
            n_u, n_v = self._scaled_grid(int(self.live_preview_resolution))
            self._set_live_status(f"Previewing (deterministic, {n_u}x{n_v}); "
                                  "the full solve runs once the Tx stops")
            config["component"].update(solver="deterministic",
                                       det_max_bounces=int(self.live_preview_bounces),
                                       det_max_diffraction_order=1)
            config["grid_shape"] = [n_u, n_v]
        await self._live_solve(config, None)

    async def _live_refine(self, is_current) -> None:
        # The last preview lands a moment after the drag and looks final, so say when the
        # full solve starts and when its map is on screen.
        # A coarse pass first (UCSD deterministic: 128^2 in ~1.6 s) puts a map with every
        # propagation effect on screen quickly; the RadioMap's own grid follows (~6.5 s).
        solver = str(self.solver)
        started = time.perf_counter()
        for shape in self._refine_passes():
            n_u, n_v = shape or self._grid_shape()
            self._set_live_status(f"Refining ({solver}, {n_u}x{n_v})...")
            config = self._solver_config(self._live_tx_name, self._owner.name)
            if shape is not None:
                config["grid_shape"] = [int(n_u), int(n_v)]
            outcome = await self._live_solve(config, is_current)
            if outcome == "skipped":
                return  # Live Status already says why
            if outcome != "applied":
                self._set_live_status("Tx moved: refining again once it stops")
                return
        seconds = time.perf_counter() - started
        self._set_live_status(f"Done: {solver}, {seconds:.1f} s")

    def _grid_shape(self):
        """The RadioMap's own receiver grid ``(n_u, n_v)``."""
        from ..adapter.endpoint_map import EndpointMap

        return tuple(int(v) for v in EndpointMap.radiomap_to_platform(self._owner).shape)

    def _scaled_grid(self, cells: int):
        """The RadioMap's grid scaled so its longer side has ``cells`` cells."""
        n_u, n_v = self._grid_shape()
        scale = min(1.0, cells / max(n_u, n_v))
        return max(1, round(n_u * scale)), max(1, round(n_v * scale))

    def _refine_passes(self):
        """Grid overrides of the refine passes; ``None`` is the RadioMap's own grid."""
        n_u, n_v = self._grid_shape()
        coarse = int(self.live_refine_coarse)
        if coarse <= 0 or max(n_u, n_v) <= coarse:
            return [None]
        return [self._scaled_grid(coarse), None]

    def _live_cancel(self) -> None:
        if self._live_run_id:
            api.solvers.cancel(self._live_run_id, "Tx moved")

    def _below_ground(self, scene, position) -> bool:
        """Whether the Tx sits under the geometry's ground at its (x, z).

        A Tx dragged under the terrain has nothing to solve, and the deterministic
        refine there ran the GPU out of memory (University of Michigan map).
        """
        from ..agent_tools import _ground_height, scene_vertices

        key = self._scene_key
        cached = self._ground_cache
        if key and cached is not None and cached[0] == key:
            vertices = cached[1]
        else:
            vertices = scene_vertices(scene)
            self._ground_cache = (key, vertices) if key else None
        if len(vertices) == 0:
            return False
        x, y, z = (float(v) for v in position)
        return y < _ground_height(vertices, x, z)

    def _remember_live_run(self, run_id: str) -> None:
        self._live_run_id = run_id

    async def _live_solve(self, config: dict, is_current) -> str:
        """Solve at the Tx's current position; "applied", "stale" or "cancelled"."""
        scene = self._owner.scene
        tx = scene.get_object_by_name(self._live_tx_name)
        if tx is None:
            raise RuntimeError(f"transmitter {self._live_tx_name!r} is gone")
        config["tx_position"] = [float(v) for v in tx.get_component("Transform").position]
        if self._below_ground(scene, config["tx_position"]):
            self._set_live_status("Tx is below the ground: move it up to update the radio map")
            return "skipped"
        cuda_retried = False
        for _ in range(3):
            send_scene = not self._scene_key
            if send_scene:
                self._scene_key = uuid.uuid4().hex
            config["scene_key"] = self._scene_key
            run = await asyncio.to_thread(
                api.solvers.solve,
                "witwin.channel.radiomap",
                scene=scene if send_scene else None,
                config=config,
                on_submitted=self._remember_live_run,
            )
            if run.status == "succeeded":
                break
            message = (run.error or {}).get("message") or "radiomap solve failed"
            if "scene_cache_miss" in message and not send_scene:
                self._scene_key = ""  # the solver host restarted: resend the Scene once
                continue
            if _cuda_error(message) and not cuda_retried:
                cuda_retried = True
                self._scene_key = ""  # the fresh host has no cached Scene
                await asyncio.sleep(_CUDA_RETRY_DELAY_S)
                continue
            if run.status == "cancelled":
                return "cancelled"
            self._set_live_status(f"Failed: {message}")
            Notifications.error("Channel", f"live radio map update failed: {message}")
            raise RuntimeError(message)
        else:
            return "cancelled"
        if is_current is not None and not is_current():
            return "stale"  # the Tx moved while this solve ran
        self._solver_result_handle = str((run.outputs or {}).get("resultHandle") or "")
        self._solver_run_id = run.run_id
        self._result = None
        self._persist_map = "grid_shape" not in config
        self.update_map()
        return "applied"

    def _notify_volume_progress(self, *args, **kwargs):
        # The core toasts "Volume Cached" for every plane publish; a radio map republishes
        # on every solve and live preview, and Live Status already says what is happening.
        return None

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
        data = self._on_own_grid(data)
        self._applying_map = True  # these writes are display state, not a grid edit
        try:
            self.set_volume_data(self._to_plane_volume(self._display_values(data)))  # normalizes + sets resolution
            self.set_field("render_mode", "Slices", force=True)
            self._export_map()                                     # volume resource -> plane shows
        finally:
            self._applying_map = False
        if bool(self.send_to_results):
            FieldViz.show_2d(data, title=label)
        return "Map updated"

    def _export_map(self) -> None:
        """Publish the plane; a full-grid result is also written to the project.

        In-memory volume assets are gone after a restart while the saved Scene still
        points at them, so every map Scene reopened showed only the green terrain. A
        full result goes to ``.witwin/runtime/volumes``; live previews and coarse passes
        stay in memory. The previous file is kept until the next one replaces it: the
        saved Scene points at it until the next autosave (every 30 s).
        """
        scene = getattr(self._owner, "scene", None)
        handler = (getattr(getattr(scene, "_server", None), "handlers", None) or {}).get("binary_assets")
        if not self._persist_map or handler is None or self._volume_data is None:
            self.export_volume_data()
            return
        volume = self._volume_data
        volume = volume.cpu().numpy() if hasattr(volume, "cpu") else np.asarray(volume)
        data = np.ascontiguousarray(np.asarray(volume, dtype=np.float32).transpose(2, 1, 0))
        current = str((self._values.get("volume") or {}).get("assetId") or "")
        kept = list(self._persisted_assets) or ([current] if current.startswith("radiomap-") else [])
        asset_id = f"radiomap-{uuid.uuid4().hex[:12]}"
        ref = handler.service.register_bytes(
            kind="volume", format=VOLUME_FLOAT32_FORMAT, data=data.tobytes(), scene_id="default",
            shape=[int(v) for v in volume.shape],
            attributes={"dtype": "float32", "layout": "zyx_for_webgl3d", "normalized": True},
            persistent=True, asset_id=asset_id,
        )
        self._publish_volume_ref(ref.to_dict())
        kept.append(asset_id)
        while len(kept) > 2:
            old = kept.pop(0)
            path = handler.service.resolve_file(old)
            handler.service.clear(old)
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass
        self._persisted_assets = tuple(kept)

    def _on_own_grid(self, data2d: np.ndarray) -> np.ndarray:
        """A coarse map (live preview, first refine pass) on the RadioMap's own grid.

        set_volume_data takes its resolution from the data, so showing a coarse map
        as is would shrink the RadioMap's grid and the next full solve would run coarse.
        """
        data = np.asarray(data2d)
        n_u, n_v = self._grid_shape()
        if data.shape == (n_v, n_u):
            return data
        rows = (np.arange(n_v) * data.shape[0]) // n_v
        cols = (np.arange(n_u) * data.shape[1]) // n_u
        return data[np.ix_(rows, cols)]

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
