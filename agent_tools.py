"""Agent tool: solve a radio map over the geometry already in the Scene.

Registered via ``contributes.tools`` -> ``register(ctx)``; the loader namespaces it as
``witwin.channel.run_radiomap``. The usual chain is ``witwin.streetmap.import_map``
(real terrain + buildings) followed by this tool, so one sentence such as "show me the
UCSD map and run a radio map" ends in a solved coverage map.

The tool plans everything the user did not say from the Scene geometry: the grid covers
the geometry's horizontal extent, the horizontal map plane sits ``rx_height_m`` above
the local ground at the grid center (imported terrain anchors its lowest point at
y=0, so the campus can sit far above it), and the transmitter is placed above the
center, clear of the local rooftops. It creates or reuses its own "Radio Map Tx" and
"Radio Map" objects and solves through ``RadioMapComponent.solve_for`` — the same
solver-host path as the Solve button — so the viewport plane updates exactly as a
manual solve would.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from witwin_server.api import SceneObject, Tool, ToolError, tool

from .adapter.channel_adapter import PATHSOLVER_COMPONENT, ChannelAdapter
from .adapter.endpoint_map import RADIOMAP_COMPONENT, RX_COMPONENT, TX_COMPONENT

TX_OBJECT_NAME = "Radio Map Tx"
MAP_OBJECT_NAME = "Radio Map"
# Cells at or below this level (dB) received nothing (the map floors 0 at -300 dB).
NO_COVERAGE_DB = -250.0
SOLVERS = ("deterministic", "montecarlo")
_GROUND_NEIGHBOURS = 16
_GROUND_BAND_M = 3.0
_ROOF_CLEARANCE_M = 3.0

_DESCRIPTION = (
    "Compute and show a radio map (radiomap, RF coverage map, signal strength / path "
    "gain heatmap) over the geometry already in the Scene, using the WiTwin Channel "
    "ray-tracing solver on the GPU. Use it whenever the user asks for a radio map, "
    "coverage map, signal coverage, wireless coverage or signal strength of a place. "
    "When the user asks to see, show or load the map of a place, first call "
    "witwin.streetmap.import_map for it, even if a map of that place is already open: "
    "it creates a new Scene and switches to it, and this tool then runs there. Skip "
    "import_map only for the map already on screen (e.g. 'run the radio map again'). "
    "Everything is optional: by default the grid "
    "covers the whole imported map, the map plane sits 1.5 m above the local ground, "
    "and one transmitter is placed above the map center, clear of nearby rooftops. "
    "Creates or reuses the objects 'Radio Map Tx' and 'Radio Map' and returns the "
    "path-gain statistics in dB."
)

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "frequency_ghz": {"type": "number", "minimum": 0.1, "maximum": 100.0,
                          "description": "Carrier frequency in GHz. Default 3.5."},
        "center_x": {"type": "number", "description": "Grid center world X (m). Default: map center."},
        "center_z": {"type": "number", "description": "Grid center world Z (m). Default: map center."},
        "size_m": {"type": "number", "minimum": 1.0, "maximum": 30000.0,
                   "description": "Grid side length (m). Default: the whole map extent."},
        "tx_height_m": {"type": "number", "minimum": 1.0, "maximum": 500.0,
                        "description": "Transmitter height above local ground (m). Default 30."},
        "rx_height_m": {"type": "number", "minimum": 0.1, "maximum": 100.0,
                        "description": "Map plane height above local ground (m). Default 1.5."},
        "resolution": {"type": "integer", "minimum": 2, "maximum": 1024,
                       "description": "Grid cells per side. Default 256."},
        "solver": {"type": "string", "enum": list(SOLVERS),
                   "description": "Ray-tracing solver. Default deterministic (reflections + "
                                  "edge diffraction); falls back to montecarlo if it fails."},
        "samples": {"type": "integer", "minimum": 1024, "maximum": 67108864,
                    "description": "Monte Carlo rays per transmitter. Default 1048576."},
        "max_bounces": {"type": "integer", "minimum": 0, "maximum": 5,
                        "description": "Reflection depth. Default 2."},
        "scene_id": {"type": "string",
                     "description": "Scene to solve in; Studio fills in the Scene the user is viewing."},
    },
    "additionalProperties": False,
}

_DEFAULTS = {
    "frequency_ghz": 3.5,
    "solver": "deterministic",
    "tx_height_m": 30.0,
    "rx_height_m": 1.5,
    "resolution": 256,
    "samples": 1 << 20,
    "max_bounces": 2,
}

_CHANNEL_COMPONENTS = (TX_COMPONENT, RX_COMPONENT, RADIOMAP_COMPONENT, PATHSOLVER_COMPONENT)


# --- planning (pure) -------------------------------------------------------------

def scene_vertices(scene: Any) -> np.ndarray:
    """World-space vertices of every structure the channel solver would trace."""
    from witwin_server.api import StructureMap

    chunks = []
    for obj in scene.objects.values():
        if any(obj.get_component(name) is not None for name in _CHANNEL_COMPONENTS):
            continue
        if obj.get_component("PlatformGeometry") is not None:
            geometry = StructureMap.to_platform(obj).geometry
            mesh = geometry if hasattr(geometry, "world_vertices") else geometry.to_mesh()
        elif ChannelAdapter._is_plain_mesh(obj):
            mesh = ChannelAdapter.mesh_geometry(obj)
        else:
            continue
        chunks.append(mesh.world_vertices.detach().cpu().numpy().reshape(-1, 3))
    if not chunks:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _ground_height(vertices: np.ndarray, x: float, z: float) -> float:
    # The nearest low-lying vertex: among the closest vertices in XZ, ignore rooftops
    # (anything more than a band above the lowest one) and take the nearest remaining.
    d2 = (vertices[:, 0] - x) ** 2 + (vertices[:, 2] - z) ** 2
    k = min(_GROUND_NEIGHBOURS, len(vertices))
    nearest = np.argpartition(d2, k - 1)[:k]
    ys = vertices[nearest, 1]
    low = nearest[ys <= ys.min() + _GROUND_BAND_M]
    return float(vertices[low[np.argmin(d2[low])], 1])


def plan_radiomap(vertices: np.ndarray, *, center_xz: Optional[Sequence[float]] = None,
                  size_m: Optional[float] = None, rx_height_m: float = 1.5,
                  tx_height_m: float = 30.0) -> Dict[str, Any]:
    """Plan the map grid and transmitter from world-space Scene vertices."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(v) == 0:
        raise ValueError("the Scene has no geometry to compute a radio map over")
    lo, hi = v.min(axis=0), v.max(axis=0)
    if center_xz is None:
        cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    else:
        cx, cz = float(center_xz[0]), float(center_xz[1])
    size = float(size_m) if size_m else float(max(hi[0] - lo[0], hi[2] - lo[2]))
    size = max(size, 1.0)

    ground = _ground_height(v, cx, cz)
    roof_radius = max(15.0, 0.02 * size)
    near = v[(v[:, 0] - cx) ** 2 + (v[:, 2] - cz) ** 2 <= roof_radius ** 2]
    roof = float(near[:, 1].max()) if len(near) else ground
    tx_y = max(ground + float(tx_height_m), roof + _ROOF_CLEARANCE_M)
    plane_y = ground + float(rx_height_m)
    return {
        "center": [cx, plane_y, cz],
        "size_m": size,
        "ground_y": ground,
        "plane_y": plane_y,
        "tx_position": [cx, tx_y, cz],
        "geometry_y_range": [float(lo[1]), float(hi[1])],
    }


# --- scene objects -------------------------------------------------------------

def _find_object(scene: Any, name: str, component: str) -> Optional[SceneObject]:
    for obj in scene.objects.values():
        if obj.name == name and obj.get_component(component) is not None:
            return obj
    return None


def ensure_radiomap_objects(scene: Any, plan: Dict[str, Any], *, frequency_hz: float,
                            resolution: int, samples: int,
                            max_bounces: int,
                            solver: str = "deterministic") -> Tuple[SceneObject, SceneObject]:
    """Create (or reuse) the tool's Tx + RadioMap objects and apply the plan.

    Deterministic is the default: with first-order UTD it lights shadowed streets that
    Monte Carlo basic leaves mostly empty at a million rays. Both configs are written so
    a fallback only has to switch ``solver``.
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown radio map solver: {solver}")
    from .components.endpoints import TxComponent
    from .components.radiomap import RadioMapComponent

    tx = _find_object(scene, TX_OBJECT_NAME, TX_COMPONENT)
    if tx is None:
        tx = SceneObject(name=TX_OBJECT_NAME, mesh_type="Empty")
        tx_comp = tx.add_component(TxComponent())
        tx_comp.polarization = [0.0, 1.0, 0.0]  # vertical dipole axis (Studio is Y up)
        tx_comp.power = 1.0
        scene.add_object(tx)
    tx.get_component("Transform").position = [float(v) for v in plan["tx_position"]]

    rmap = _find_object(scene, MAP_OBJECT_NAME, RADIOMAP_COMPONENT)
    if rmap is None:
        rmap = SceneObject(name=MAP_OBJECT_NAME, mesh_type="Empty")
        rmap.add_component(RadioMapComponent())
        scene.add_object(rmap)
    comp = rmap.get_component(RADIOMAP_COMPONENT)
    half = 0.5 * float(plan["size_m"])
    comp.frequency = float(frequency_hz)
    comp.grid_mode = "Resolution"
    comp.min_bound = [-half, 0.0, -half]
    comp.max_bound = [half, 0.0, half]
    comp.resolution = [float(resolution), 1.0, float(resolution)]
    comp.solver = solver
    comp.det_max_bounces = int(max_bounces)
    comp.det_max_diffraction_order = 1
    comp.mc_integrator = "basic"
    comp.mc_samples_per_tx = int(samples)
    comp.mc_max_bounces = int(max_bounces)
    comp.mc_max_diffraction_order = 1
    comp.metric = "path_gain"
    comp.log_scale = True
    rmap.get_component("Transform").position = [float(v) for v in plan["center"]]
    return tx, rmap


def _db_statistics(data: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(data, dtype=np.float64).ravel()
    covered = values[np.isfinite(values) & (values > NO_COVERAGE_DB)]
    stats: Dict[str, Any] = {"covered_fraction": float(len(covered)) / max(1, len(values))}
    if len(covered):
        stats.update(min=float(covered.min()), median=float(np.median(covered)),
                     max=float(covered.max()))
    return stats


# --- tool ------------------------------------------------------------------------

def _resolve_scene(ctx: Any, scene_id: Any = None) -> Any:
    # The radio map goes where the user is looking: Studio passes the viewed Scene.
    # With ctx.api.scene alone a map asked for in a new Scene was solved in the main one.
    api = getattr(ctx, "api", None)
    if scene_id:
        scene = api.get_scene(str(scene_id)) if api is not None else None
        if scene is None:
            raise ToolError(f"scene {scene_id!r} is not open", code="missing_scene",
                            detail={"scene_id": str(scene_id)})
        return scene
    scene = getattr(api, "scene", None)
    if scene is None:
        raise ToolError("no active scene", code="no_scene")
    return scene


def make_run_radiomap_tool(ctx: Any) -> Tool:
    @tool(
        name="run_radiomap",
        description=_DESCRIPTION,
        input_schema=_INPUT_SCHEMA,
        output_schema={"type": "object"},
        side_effects=True,
        permission_tier="scene_write",
        permission_scope="scene.write",
        timeout=900.0,
        tags=("channel", "radio map", "coverage"),
    )
    async def run_radiomap(args: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio

        opts = {**_DEFAULTS, **{k: v for k, v in (args or {}).items() if v is not None}}
        scene = _resolve_scene(ctx, (args or {}).get("scene_id"))
        vertices = await asyncio.to_thread(scene_vertices, scene)
        if len(vertices) == 0:
            raise ToolError(
                "the Scene has no map or buildings to compute a radio map over; import a "
                "place first with witwin.streetmap.import_map", code="no_geometry")
        center = None
        if "center_x" in opts or "center_z" in opts:
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            center = (float(opts.get("center_x", 0.5 * (lo[0] + hi[0]))),
                      float(opts.get("center_z", 0.5 * (lo[2] + hi[2]))))
        plan = await asyncio.to_thread(
            plan_radiomap, vertices, center_xz=center, size_m=opts.get("size_m"),
            rx_height_m=float(opts["rx_height_m"]), tx_height_m=float(opts["tx_height_m"]))
        frequency_hz = float(opts["frequency_ghz"]) * 1e9
        resolution = int(opts["resolution"])
        tx, rmap = ensure_radiomap_objects(
            scene, plan, frequency_hz=frequency_hz, resolution=resolution,
            samples=int(opts["samples"]), max_bounces=int(opts["max_bounces"]),
            solver=str(opts["solver"]))

        comp = rmap.get_component(RADIOMAP_COMPONENT)
        fallback = None
        started = time.perf_counter()
        message = await comp.solve_for(tx.name)
        if message != "Solve complete" and str(comp.solver) == "deterministic":
            # An older channel build cannot link the diffraction pipelines; Monte Carlo
            # still gives the user a map, and the result names the solver that ran.
            fallback = {"from": "deterministic", "reason": message}
            comp.solver = "montecarlo"
            message = await comp.solve_for(tx.name)
        solve_seconds = time.perf_counter() - started
        if message != "Solve complete":
            raise ToolError(f"radio map solve failed: {message}", code="solve_failed")
        payload = await asyncio.to_thread(comp._query_result, "extract_map", {
            "metric": "path_gain", "tx_index": 0, "log_scale": True})
        return {
            "ok": True,
            "transmitter": {"object_id": tx.id, "name": tx.name,
                            "position": [float(v) for v in plan["tx_position"]]},
            "radiomap": {
                "object_id": rmap.id, "name": rmap.name,
                "center": [float(v) for v in plan["center"]],
                "size_m": float(plan["size_m"]), "grid": [resolution, resolution],
                "frequency_hz": frequency_hz,
            },
            "ground_y": float(plan["ground_y"]),
            "geometry_y_range": plan["geometry_y_range"],
            "path_gain_db": _db_statistics(payload["data"]),
            "solve_seconds": round(solve_seconds, 3),
            "solver": str(comp.solver),
            **({"fallback": fallback} if fallback else {}),
        }

    return run_radiomap


def register(ctx: Any) -> None:
    """contributes.tools hook: register the run_radiomap agent tool.

    Also arms live Tx drag for RadioMaps loaded from disk: on every ``scene_loaded``
    and for the current Scene (it loads before plugins activate).
    """
    from .components.radiomap import watch_scene_radiomaps

    ctx.tools.register(make_run_radiomap_tool(ctx))
    lifecycle = getattr(ctx, "scene_lifecycle", None)
    if lifecycle is not None:
        lifecycle.on("scene_loaded", watch_scene_radiomaps)
    try:
        scene = ctx.api.scene
    except (RuntimeError, AttributeError):  # api not bound to a server (tests, solver host)
        scene = None
    if scene is not None:
        watch_scene_radiomaps(scene)
