"""ChannelAdapter — Core 0.4 ``witwin.core.Scene`` <-> studio ``Scene`` (the ``"channel"`` domain).

Implements the frozen ``SceneAdapter`` protocol (detect / to_studio / to_platform)
and is registered on the core ``PlatformSceneHandler`` by the plugin's ``__init__``.

witwin-channel 0.4+ has no scene type of its own: every solver takes the logical
``witwin.core.Scene`` (structures + ``AntennaState`` / ``ReceiverGrid`` endpoints) plus
an explicit reference frequency. The adapter therefore claims Core worlds that carry
endpoints (structures-only worlds stay with the core adapter) and keeps the channel
scene config (frequency / noise) in ``Scene.metadata["channel"]`` so a world round-trips.

Shared parts reuse the frozen base maps — structures (``StructureMap``), transform
(``TransformMap``), scalar material (``MaterialMap``) — and the channel-specific parts
are ITU material presets, endpoints (``Tx`` / ``Rx`` with inline antennas) and the
``RadioMap`` object that doubles as the receiver grid. Plain Studio meshes (e.g. an
imported street map) become structures with a default ITU concrete material.
"""
from typing import Any, Dict, List, Optional, Tuple

from witwin_server import Scene, SceneObject
from witwin_server.features.platform.bridge import StructureMap, TransformMap

from .common import num, vec
from .endpoint_map import RADIOMAP_COMPONENT, RX_COMPONENT, TX_COMPONENT, EndpointMap, endpoint_name
from .itu import itu_material
from .material_map import ChannelMaterialMap

PATHSOLVER_COMPONENT = "PathSolver"
CHANNEL_METADATA_KEY = "channel"
DEFAULT_MESH_MATERIAL = "concrete"
# StructureMeta tag for scene dressing that radio waves pass through (street-map roads
# drawn as ribbons, tree canopies): drawn in the viewport, left out of the radio world.
RF_TRANSPARENT_TAG = "rf:transparent"


class ChannelAdapter:
    """Bidirectional adapter between a Core 0.4 world and a studio ``Scene``."""

    def detect(self, scene: Any) -> bool:
        """Claim Core worlds that carry endpoints (graceful if Core is absent)."""
        try:
            import witwin.core as wc
        except ImportError:
            return False
        return isinstance(scene, wc.Scene) and len(tuple(scene.endpoints)) > 0

    # --- platform -> studio --------------------------------------------------

    def to_studio(self, scene: Any) -> Scene:
        """Translate a Core world into a studio ``Scene`` (batched)."""
        from witwin.core import ReceiverGrid

        config = dict((scene.metadata or {}).get(CHANNEL_METADATA_KEY, {}))
        frequency = config.get("frequency_hz")
        noise = config.get("noise_power_w")

        studio = Scene()
        studio.begin_batch()
        for structure in scene.structures:
            studio.add_object(self._structure_to_studio(structure))
        grids = [e for e in scene.endpoints if isinstance(e, ReceiverGrid)]
        points = [e for e in scene.endpoints if e.role == "rx" and not isinstance(e, ReceiverGrid)]
        for tx in (e for e in scene.endpoints if e.role == "tx"):
            studio.add_object(EndpointMap.tx_to_studio(tx))
        for grid in grids:
            studio.add_object(EndpointMap.radiomap_to_studio(
                grid, frequency=frequency, noise_power=noise))
        for rx in points:
            studio.add_object(EndpointMap.rx_to_studio(rx))
        # A PathSolver carries the config when there is no RadioMap, and is the path-solve
        # entry point whenever there are point receivers.
        if points or not grids:
            studio.add_object(self._pathsolver_object(frequency=frequency, noise_power=noise))
        studio.end_batch()
        return studio

    # --- studio -> platform --------------------------------------------------

    def to_platform(self, studio_scene: Scene) -> Any:
        """Rebuild a Core world from the studio scene (config discovered)."""
        frequency, noise = self._discover_config(studio_scene)
        return self.build(studio_scene, frequency=frequency, noise_power=noise)

    def build(self, studio_scene: Scene, *, frequency: Optional[float],
              noise_power: Optional[float] = None, transmitter: Optional[str] = None,
              receiver: Optional[str] = None,
              grid_shape: Optional[Tuple[int, int]] = None) -> Any:
        """Rebuild a Core world with explicit scene config (used by the solvers).

        ``transmitter`` / ``receiver`` restrict the world to the named Tx and receiver
        (point Rx or RadioMap grid); every solver solves all endpoints it is given.
        """
        import witwin.core as wc

        structures: List[Any] = []
        endpoints: List[Any] = []
        for obj in studio_scene.objects.values():
            if obj.get_component(TX_COMPONENT) is not None:
                if transmitter is None or obj.name == transmitter:
                    endpoints.append(EndpointMap.tx_to_platform(obj))
            elif obj.get_component(RX_COMPONENT) is not None:
                if receiver is None or obj.name == receiver:
                    endpoints.append(EndpointMap.rx_to_platform(obj))
            elif obj.get_component(RADIOMAP_COMPONENT) is not None:
                if receiver is None or obj.name == receiver:
                    endpoints.append(EndpointMap.radiomap_to_platform(obj, grid_shape=grid_shape))
            elif self._rf_transparent(obj):
                continue
            elif obj.get_component("PlatformGeometry") is not None:
                structures.append(self._structure_to_platform(obj, frequency))
            elif self._is_plain_mesh(obj):
                structures.append(self._mesh_to_platform(wc, obj, frequency))

        if transmitter is not None and not any(e.role == "tx" for e in endpoints):
            raise ValueError(f"transmitter '{transmitter}' not found in the scene")
        if receiver is not None and not any(e.role == "rx" for e in endpoints):
            raise ValueError(f"receiver '{receiver}' not found in the scene")
        config: Dict[str, Any] = {}
        if frequency is not None:
            config["frequency_hz"] = float(frequency)
        if noise_power is not None:
            config["noise_power_w"] = float(noise_power)
        return wc.Scene(structures=tuple(structures), endpoints=tuple(endpoints),
                        metadata={CHANNEL_METADATA_KEY: config})

    @staticmethod
    def has_structures(studio_scene: Scene) -> bool:
        """True if the scene has any geometry the solver can ray-trace against.

        Counts both channel/platform structures (``PlatformGeometry``) and plain studio
        meshes (used as default-material walls).
        """
        for obj in studio_scene.objects.values():
            if ChannelAdapter._rf_transparent(obj):
                continue
            if obj.get_component("PlatformGeometry") is not None or ChannelAdapter._is_plain_mesh(obj):
                return True
        return False

    @staticmethod
    def endpoint_by_name(world: Any, name: str) -> Any:
        """The endpoint built from the Studio object ``name``."""
        for endpoint in world.endpoints:
            if endpoint_name(endpoint) == name:
                return endpoint
        raise KeyError(name)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _rf_transparent(obj: SceneObject) -> bool:
        meta = obj.get_component("StructureMeta")
        return meta is not None and RF_TRANSPARENT_TAG in list(meta.tags or [])

    @staticmethod
    def _is_plain_mesh(obj: SceneObject) -> bool:
        # A regular studio mesh (no PlatformGeometry) usable as a default-material structure.
        if obj.get_component("PlatformGeometry") is not None:
            return False
        mesh = obj.get_component("Mesh")
        return mesh is not None and not mesh.is_empty

    @classmethod
    def _mesh_to_platform(cls, wc: Any, obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Build a surface structure from a plain studio mesh + its full Transform.
        return wc.Structure(cls.mesh_geometry(obj), cls._mesh_material(obj, scene_frequency),
                            name=obj.name)

    @staticmethod
    def mesh_geometry(obj: SceneObject) -> Any:
        """A plain studio mesh + its full Transform as a Core surface ``Mesh``."""
        import torch
        import witwin.core as wc

        mesh = obj.get_component("Mesh")
        transform = obj.get_component("Transform")
        verts = torch.as_tensor(mesh.get_vertices_numpy().reshape(-1, 3), dtype=torch.float32)
        faces = torch.as_tensor(mesh.get_faces_numpy().reshape(-1, 3).astype("int64"))
        return wc.Mesh(verts, faces, position=vec(transform.position),
                       rotation=TransformMap.euler_to_quat(transform.rotation),
                       scale=vec(transform.scale), recenter=False, fill_mode="surface")

    @staticmethod
    def _mesh_material(obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Plain meshes default to ITU concrete so every triangle has an assigned material;
        # an explicit MaterialPreset, an ITU pick or hand-edited EM scalars on the unified
        # Material component still win. Every SceneObject carries a Material component
        # whose untouched state is "Custom" with the component defaults, so only a
        # changed name or changed scalars signal intent.
        if obj.get_component("MaterialPreset") is not None:
            return ChannelMaterialMap.to_platform(obj, scene_frequency)
        material_comp = obj.get_component("Material")
        if material_comp is not None and (
                ChannelMaterialMap.itu_name(material_comp) is not None
                or ChannelMaterialMap.has_authored_scalars(material_comp)):
            return ChannelMaterialMap.to_platform(obj, scene_frequency)
        if scene_frequency is None:
            raise ValueError(f"{obj.name}: the default ITU concrete needs a scene frequency")
        return itu_material(DEFAULT_MESH_MATERIAL, scene_frequency)

    @classmethod
    def _discover_config(cls, studio_scene: Scene) -> Tuple[Optional[float], Optional[float]]:
        # Frequency / noise live on the RadioMap (preferred) or the PathSolver.
        for component_name in (RADIOMAP_COMPONENT, PATHSOLVER_COMPONENT):
            for obj in studio_scene.objects.values():
                comp = obj.get_component(component_name)
                if comp is not None:
                    return cls._config_from(comp)
        return (None, None)

    @staticmethod
    def _config_from(comp: Any) -> Tuple[Optional[float], Optional[float]]:
        frequency = num(comp.frequency)
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        return (frequency if frequency > 0.0 else None, noise)

    @staticmethod
    def _pathsolver_object(*, frequency: Optional[float],
                           noise_power: Optional[float]) -> SceneObject:
        from ..components.pathsolver import PathSolverComponent

        obj = SceneObject(name="Path Solver", mesh_type="Empty")
        comp = obj.add_component(PathSolverComponent())
        if frequency is not None:
            comp.frequency = float(frequency)
        if noise_power is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise_power)
        return obj

    @staticmethod
    def _structure_to_platform(obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Geometry + metadata via the frozen StructureMap; an ITU preset or ITU pick is
        # re-evaluated at the scene frequency (Core structures are immutable, so rebuild).
        import witwin.core as wc

        structure = StructureMap.to_platform(obj)
        material_comp = obj.get_component("Material")
        if obj.get_component("MaterialPreset") is None and (
                material_comp is None or ChannelMaterialMap.itu_name(material_comp) is None):
            return structure
        return wc.Structure(
            structure.geometry, ChannelMaterialMap.to_platform(obj, scene_frequency),
            name=structure.name, priority=structure.priority, enabled=structure.enabled,
            tags=structure.tags, metadata=dict(structure.metadata),
        )

    @staticmethod
    def _structure_to_studio(structure: Any) -> SceneObject:
        # The frozen StructureMap only fills mesh data when drjit is importable; Core 0.4
        # meshes are torch tensors, so fill it here or the mesh cannot be exported again.
        obj = StructureMap.to_studio(structure)
        mesh = obj.get_component("Mesh")
        geometry = structure.geometry
        if getattr(geometry, "kind", None) == "mesh" and mesh is not None and mesh.is_empty:
            vertices = geometry.vertices.detach().cpu().numpy().astype("float32")
            faces = geometry.faces.detach().cpu().numpy().astype("uint32")
            mesh.set_mesh_data(vertices, faces, notify=False)
        return obj
