"""ChannelAdapter — ``channel.Scene`` <-> studio ``Scene`` (the ``"channel"`` domain).

Implements the frozen ``SceneAdapter`` protocol (detect / to_studio / to_platform)
and is registered on the core ``PlatformSceneHandler`` by the plugin's ``__init__``.

It reuses the frozen base maps for the shared parts — geometry (``GeometryMap``),
transform (``TransformMap``), scalar material (``MaterialMap``), structure metadata
(``StructureMap``) — and adds the channel-specific parts: the scene config singleton
(``SettingsMap``) and (later phases) ITU material presets, endpoints, antenna arrays,
solvers, and 2D radiomap visualization.
"""
from typing import Any, List, Optional

from witwin_server import Scene, SceneObject
from witwin_server.platform_bridge import GeometryMap, StructureMap, TransformMap

from .array_spec import ArraySpecMap
from .common import num, vec
from .endpoint_map import (
    DEFAULT_ARRAY_COMPONENT,
    GRID_COMPONENT,
    RX_COMPONENT,
    TX_COMPONENT,
    EndpointMap,
)
from .material_map import ChannelMaterialMap
from .settings_map import MARKER_COMPONENT, SettingsMap


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
        studio = Scene(kind="local")
        studio.begin_batch()
        studio.add_object(SettingsMap.to_studio(scene))
        for structure in scene.structures:
            obj = StructureMap.to_studio(structure)
            ChannelMaterialMap.overlay_studio(obj, structure.material)
            studio.add_object(obj)
        for tx in scene.transmitters:
            studio.add_object(EndpointMap.tx_to_studio(tx))
        for rx in scene.receivers:
            # ReceiverGrid carries an `axis`; a point Receiver does not.
            builder = EndpointMap.grid_to_studio if hasattr(rx, "axis") else EndpointMap.rx_to_studio
            studio.add_object(builder(rx))
        if scene.tx_array is not None:
            studio.add_object(EndpointMap.default_array_to_studio("tx", scene.tx_array))
        if scene.rx_array is not None:
            studio.add_object(EndpointMap.default_array_to_studio("rx", scene.rx_array))
        studio.end_batch()
        return studio

    # --- studio -> platform --------------------------------------------------

    def to_platform(self, studio_scene: Scene) -> Any:
        """Rebuild a ``channel.Scene`` from the studio scene's objects."""
        import witwin.channel as ch

        settings_obj = self._find_settings(studio_scene)
        if settings_obj is None:
            raise ValueError(
                "channel export requires a 'Channel Settings' object with a "
                "ChannelSettings component (scene frequency / device).")
        config = SettingsMap.build_config(settings_obj)

        structures: List[Any] = []
        transmitters: List[Any] = []
        receivers: List[Any] = []
        tx_array = rx_array = None
        for obj in studio_scene.objects.values():
            if obj is settings_obj:
                continue
            if obj.get_component(TX_COMPONENT) is not None:
                transmitters.append(EndpointMap.tx_to_platform(ch, obj))
            elif obj.get_component(RX_COMPONENT) is not None or obj.get_component(GRID_COMPONENT) is not None:
                receivers.append(EndpointMap.receiver_to_platform(ch, obj))
            elif obj.get_component(DEFAULT_ARRAY_COMPONENT) is not None:
                if str(obj.get_component(DEFAULT_ARRAY_COMPONENT).role) == "tx":
                    tx_array = ArraySpecMap.from_object(ch, obj)
                else:
                    rx_array = ArraySpecMap.from_object(ch, obj)
            elif obj.get_component("PlatformGeometry") is not None:
                structures.append(self._structure_to_platform(ch, obj, config.frequency))

        scene = ch.Scene(structures=structures, transmitters=transmitters, receivers=receivers,
                         frequency=config.frequency, device=config.device,
                         tx_array=tx_array, rx_array=rx_array)
        if config.noise_power is not None:
            scene.noise_power = config.noise_power
        return scene

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _find_settings(studio_scene: Scene) -> Optional[SceneObject]:
        # The Channel Settings object is the one carrying a ChannelSettings component.
        for obj in studio_scene.objects.values():
            if obj.get_component(MARKER_COMPONENT) is not None:
                return obj
        return None

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
