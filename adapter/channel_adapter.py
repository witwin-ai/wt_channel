"""ChannelAdapter — ``channel.Scene`` <-> studio ``Scene`` (the ``"channel"`` domain).

Implements the frozen ``SceneAdapter`` protocol (detect / to_studio / to_platform)
and is registered on the core ``PlatformSceneHandler`` by the plugin's ``__init__``.

It reuses the frozen base maps for the shared parts — geometry (``GeometryMap``),
transform (``TransformMap``), scalar material (``MaterialMap``), structure metadata
(``StructureMap``) — and adds the channel-specific parts: structures + ITU material
presets, endpoints (``Tx`` / ``Rx`` with inline antennas), and the ``RadioMap`` object
that doubles as the receiver grid. The scene config (frequency / device / noise) is not
a separate object: it rides the ``RadioMap`` and ``PathSolver`` components, and is
discovered from them on export.
"""
from typing import Any, List, Optional, Tuple

from witwin_server import Scene, SceneObject
from witwin_server.features.platform.bridge import GeometryMap, StructureMap, TransformMap

from .common import device_name, num, vec
from .endpoint_map import RADIOMAP_COMPONENT, RX_COMPONENT, TX_COMPONENT, EndpointMap
from .material_map import ChannelMaterialMap

PATHSOLVER_COMPONENT = "PathSolver"


class ChannelAdapter:
    """Bidirectional adapter between a ``channel.Scene`` and a studio ``Scene``."""

    def detect(self, scene: Any) -> bool:
        """Claim the scene iff it is a ``channel.Scene`` (graceful if channel absent)."""
        try:
            import witwin.channel as ch
        except ImportError:
            return False
        return isinstance(scene, ch.Scene)

    # --- platform -> studio --------------------------------------------------

    def to_studio(self, scene: Any) -> Scene:
        """Translate a ``channel.Scene`` into a studio ``Scene`` (batched)."""
        studio = Scene()
        studio.begin_batch()
        frequency = scene.frequency
        device = device_name(scene.device)
        noise = getattr(scene, "noise_power", None)

        for structure in scene.structures:
            obj = StructureMap.to_studio(structure)
            ChannelMaterialMap.overlay_studio(obj, structure.material)
            studio.add_object(obj)
        for tx in scene.transmitters:
            studio.add_object(EndpointMap.tx_to_studio(tx, scene.tx_array))

        grids = [r for r in scene.receivers if hasattr(r, "axis")]
        points = [r for r in scene.receivers if not hasattr(r, "axis")]
        for grid in grids:
            studio.add_object(EndpointMap.radiomap_to_studio(
                grid, frequency=frequency, device=device, noise_power=noise))
        for rx in points:
            studio.add_object(EndpointMap.rx_to_studio(rx, scene.rx_array))
        # A PathSolver carries the config when there is no RadioMap, and is the path-solve
        # entry point whenever there are point receivers.
        if points or not grids:
            studio.add_object(self._pathsolver_object(
                frequency=frequency, device=device, noise_power=noise))
        studio.end_batch()
        return studio

    # --- studio -> platform --------------------------------------------------

    def to_platform(self, studio_scene: Scene) -> Any:
        """Rebuild a ``channel.Scene`` from the studio scene (config discovered)."""
        frequency, device, noise = self._discover_config(studio_scene)
        return self.build(studio_scene, frequency=frequency, device=device, noise_power=noise)

    def build(self, studio_scene: Scene, *, frequency: Optional[float], device: str,
              noise_power: Optional[float] = None) -> Any:
        """Rebuild a ``channel.Scene`` with explicit scene config (used by the solvers)."""
        import witwin.channel as ch

        structures: List[Any] = []
        transmitters: List[Any] = []
        receivers: List[Any] = []
        for obj in studio_scene.objects.values():
            if obj.get_component(TX_COMPONENT) is not None:
                transmitters.append(EndpointMap.tx_to_platform(ch, obj))
            elif obj.get_component(RX_COMPONENT) is not None:
                receivers.append(EndpointMap.rx_to_platform(ch, obj))
            elif obj.get_component(RADIOMAP_COMPONENT) is not None:
                receivers.append(EndpointMap.radiomap_to_platform(ch, obj))
            elif obj.get_component("PlatformGeometry") is not None:
                structures.append(self._structure_to_platform(ch, obj, frequency))
            elif self._is_plain_mesh(obj):
                structures.append(self._mesh_to_platform(ch, obj, frequency))

        scene = ch.Scene(structures=structures, transmitters=transmitters, receivers=receivers,
                         frequency=frequency, device=device)
        if noise_power is not None:
            scene.noise_power = noise_power
        return scene

    @staticmethod
    def has_structures(studio_scene: Scene) -> bool:
        """True if the scene has any geometry the solver can ray-trace against.

        A channel solve needs at least one structure mesh (the platform raises
        "active RayD runtime scene" otherwise). Counts both channel/platform structures
        (``PlatformGeometry``) and plain studio meshes (used as default-material walls).
        """
        for obj in studio_scene.objects.values():
            if obj.get_component("PlatformGeometry") is not None or ChannelAdapter._is_plain_mesh(obj):
                return True
        return False

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _is_plain_mesh(obj: SceneObject) -> bool:
        # A regular studio mesh (no PlatformGeometry) usable as a default-material structure.
        if obj.get_component("PlatformGeometry") is not None:
            return False
        mesh = obj.get_component("Mesh")
        return mesh is not None and not mesh.is_empty

    @classmethod
    def _mesh_to_platform(cls, ch: Any, obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Build a structure from a plain studio mesh + its Transform (default EM material).
        import witwin.core as wc

        mesh = obj.get_component("Mesh")
        transform = obj.get_component("Transform")
        verts = mesh.get_vertices_numpy().reshape(-1, 3)
        faces = mesh.get_faces_numpy().reshape(-1, 3)
        geometry = wc.Mesh(verts, faces, position=vec(transform.position),
                           rotation=TransformMap.euler_to_quat(transform.rotation),
                           scale=1.0, recenter=False)
        return ch.Structure(geometry, cls._mesh_material(ch, obj, scene_frequency), name=obj.name)

    @staticmethod
    def _mesh_material(ch: Any, obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Plain meshes default to ITU concrete so every triangle has an assigned material;
        # an explicit MaterialPreset / named scalar Material on the object still wins.
        # NB: in the material-system rewrite every SceneObject now carries a Material
        # component by default, so "has Material" is no longer a useful signal of intent.
        # We use ``material_name != ""`` instead — the user explicitly named the
        # scalar EM material on the unified Material component's Electromagnetic foldout.
        if obj.get_component("MaterialPreset") is not None:
            return ChannelMaterialMap.to_platform(ch, obj, scene_frequency)
        material_comp = obj.get_component("Material")
        if material_comp is not None and str(material_comp.material_name) != "":
            return ChannelMaterialMap.to_platform(ch, obj, scene_frequency)
        return ch.Material.from_itu("concrete")

    @classmethod
    def _discover_config(cls, studio_scene: Scene) -> Tuple[Optional[float], str, Optional[float]]:
        # Frequency / device / noise live on the RadioMap (preferred) or the PathSolver.
        for component_name in (RADIOMAP_COMPONENT, PATHSOLVER_COMPONENT):
            for obj in studio_scene.objects.values():
                comp = obj.get_component(component_name)
                if comp is not None:
                    return cls._config_from(comp)
        return (None, "cuda", None)

    @staticmethod
    def _config_from(comp: Any) -> Tuple[Optional[float], str, Optional[float]]:
        frequency = num(comp.frequency)
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        return (frequency if frequency > 0.0 else None, str(comp.device), noise)

    @staticmethod
    def _pathsolver_object(*, frequency: Optional[float], device: str,
                           noise_power: Optional[float]) -> SceneObject:
        from ..components.pathsolver import PathSolverComponent

        obj = SceneObject(name="Path Solver", mesh_type="Empty")
        comp = obj.add_component(PathSolverComponent())
        comp.frequency = 0.0 if frequency is None else float(frequency)
        comp.device = device
        if noise_power is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise_power)
        return obj

    @classmethod
    def _structure_to_platform(cls, ch: Any, obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        # Geometry + transform via the frozen base maps; material via channel resolver.
        geom = obj.get_component("PlatformGeometry")
        transform = obj.get_component("Transform")
        position = vec(transform.position)
        rotation_quat = TransformMap.euler_to_quat(transform.rotation)

        vertices = faces = None
        if str(geom.kind) == "mesh":
            mesh = obj.get_component("Mesh")
            if mesh is not None and not mesh.is_empty:
                vertices = mesh.get_vertices_numpy()
                faces = mesh.get_faces_numpy()
        geometry = GeometryMap.to_platform(geom, position, rotation_quat,
                                           vertices=vertices, faces=faces)

        material = ChannelMaterialMap.to_platform(ch, obj, scene_frequency)

        meta = obj.get_component("StructureMeta")
        if meta is None:
            return ch.Structure(geometry, material, name=obj.name)
        return ch.Structure(
            geometry, material, name=obj.name,
            priority=int(num(meta.priority)),
            enabled=bool(meta.enabled),
            tags=tuple(str(tag) for tag in meta.tags),
            metadata=dict(meta.metadata),
        )
