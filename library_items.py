"""Channel library prefabs — drag-to-create scene building blocks.

Registers a ``Channel`` library category with:

- **Channel Scene (Demo)** — Channel Settings (returned) + a Transmitter + a
  ReceiverGrid over a ground plane (ITU concrete). Drag it in, press **Solve** on
  the Channel Settings object, and the 2D radiomap appears in its figure / the
  Results panel (needs CUDA for the solve).
- **Channel Settings** — the scene-config singleton (frequency / device + the solver
  config + Solve button) for composing a custom scene.
- **Transmitter**, **Receiver**, **Receiver Grid**, **ULA Array**, **UPA Array** —
  individual building blocks.

Each prefab reuses the tested adapter maps so its components match the round-trip
exactly. The factory adds any extra objects to the drop scene and returns the primary
one. Result / runtime state is never part of a prefab.
"""
from witwin_server import Library

_CATEGORY = "channel"
_DEFAULT_FREQ = 3.5e9


def _settings_object(name="Channel Settings"):
    # Build the config singleton directly (no platform Scene needed, so no CUDA at drop).
    from witwin_server import SceneObject

    from .components.path import PathSolveComponent
    from .components.result import ChannelResultComponent
    from .components.settings import ChannelSettingsComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(ChannelSettingsComponent()).frequency = _DEFAULT_FREQ
    obj.add_component(ChannelResultComponent())
    obj.add_component(PathSolveComponent())
    return obj


def _ground_object():
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure
    from witwin_server.platform_bridge import StructureMap

    from .adapter.material_map import ChannelMaterialMap

    obj = StructureMap.to_studio(Structure(
        Box(size=(20.0, 20.0, 0.2), position=(0.0, 0.0, 0.0)),
        Material.from_itu("concrete"), name="ground"))
    ChannelMaterialMap.overlay_studio(obj, Material.from_itu("concrete"))
    return obj


def _tx_object(name="tx", position=(0.0, 0.0, 2.0)):
    import witwin.channel as ch
    from .adapter.endpoint_map import EndpointMap
    return EndpointMap.tx_to_studio(ch.Transmitter(name=name, position=position))


def _rx_object(name="rx", position=(5.0, 0.0, 1.5)):
    import witwin.channel as ch
    from .adapter.endpoint_map import EndpointMap
    return EndpointMap.rx_to_studio(ch.Receiver(name=name, position=position))


def _grid_object(name="rmap"):
    import witwin.channel as ch
    from .adapter.endpoint_map import EndpointMap
    return EndpointMap.grid_to_studio(ch.ReceiverGrid(
        name=name, axis="z", position=1.5, bounds=((-10.0, 10.0), (-10.0, 10.0)),
        grid_shape=(64, 64)))


def _array_object(name, array):
    # An Empty carrying just an AntennaArraySpec, ready to parent under an endpoint.
    from witwin_server import SceneObject

    from .adapter.array_spec import ArraySpecMap
    from .components.antenna import AntennaArraySpecComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    ArraySpecMap.to_studio(obj.add_component(AntennaArraySpecComponent()), array)
    return obj


# --- library factories (context.scene gets the extras; return the primary obj) ---

def _make_demo(ctx):
    """Solvable radiomap demo: Settings (returned) + Tx + ReceiverGrid + a point Rx + ground.

    The point Receiver ("rx") is the target for the Channel Settings' PathSolve button;
    the ReceiverGrid ("rmap") is the radiomap target for ChannelResult.
    """
    settings = _settings_object(ctx.name)
    ctx.scene.add_object(_ground_object())
    ctx.scene.add_object(_tx_object())
    ctx.scene.add_object(_rx_object())
    ctx.scene.add_object(_grid_object())
    return settings


def _make_settings(ctx):
    return _settings_object(ctx.name)


def _make_tx(ctx):
    obj = _tx_object()
    obj.name = ctx.name
    return obj


def _make_rx(ctx):
    obj = _rx_object()
    obj.name = ctx.name
    return obj


def _make_grid(ctx):
    obj = _grid_object()
    obj.name = ctx.name
    return obj


def _make_ula(ctx):
    import witwin.channel as ch
    return _array_object(ctx.name, ch.ULA(num_elements=4, spacing=0.5, axis="x"))


def _make_upa(ctx):
    import witwin.channel as ch
    return _array_object(ctx.name, ch.UPA(num_rows=2, num_cols=2))


def register() -> None:
    """Register the Channel library category + prefab items."""
    Library.register_category(_CATEGORY, "Channel", icon="radio", order=21)
    Library.register_item("channel_demo", "Channel Scene (Demo)", _CATEGORY,
                          icon="sparkles", object_type="empty", factory=_make_demo,
                          description="Tx + ReceiverGrid over a ground plane; Solve for a radiomap")
    Library.register_item("channel_settings", "Channel Settings", _CATEGORY,
                          icon="settings", object_type="empty", factory=_make_settings,
                          description="Scene config + solver + Solve button (build your own scene)")
    Library.register_item("channel_tx", "Transmitter", _CATEGORY,
                          icon="radio", object_type="empty", factory=_make_tx,
                          description="Channel transmitter endpoint")
    Library.register_item("channel_rx", "Receiver", _CATEGORY,
                          icon="antenna", object_type="empty", factory=_make_rx,
                          description="Channel point receiver endpoint")
    Library.register_item("channel_grid", "Receiver Grid", _CATEGORY,
                          icon="grid", object_type="empty", factory=_make_grid,
                          description="2D radiomap receiver grid")
    Library.register_item("channel_ula", "ULA Array", _CATEGORY,
                          icon="more-horizontal", object_type="empty", factory=_make_ula,
                          description="Uniform linear antenna array spec")
    Library.register_item("channel_upa", "UPA Array", _CATEGORY,
                          icon="grid", object_type="empty", factory=_make_upa,
                          description="Uniform planar antenna array spec")


register()
