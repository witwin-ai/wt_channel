"""Channel library prefabs — drag-to-create scene building blocks.

Registers a ``Channel`` library category with the four core building blocks plus a demo:

- **Channel Scene (Demo)** — a ground plane + wall, a **Tx**, an **Rx**, a **RadioMap**
  (returned), and a **Path Solver**. Drag it in, press **Solve** on the RadioMap for a
  2D radiomap, or **Solve Paths** on the Path Solver for the Tx<->Rx rays (needs CUDA).
- **Tx** / **Rx** — transmitter / point receiver endpoints (antenna defined inline).
- **RadioMap** — a 2D field that is its own receiver grid + solver + result viewer.
- **Path Solver** — point-to-point path solve; auto-finds the Tx/Rx in the scene.

Each prefab carries components straight from the registry (no platform ``Scene`` is
built, so dropping never needs CUDA). The factory adds any extra objects to the drop
scene and returns the primary one.
"""
from witwin_server import Library, SceneObject

_CATEGORY = "channel"


def _tx_object(name, position):
    from .components.endpoints import TxComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(TxComponent())
    obj.get_component("Transform").position = list(position)
    return obj


def _rx_object(name, position):
    from .components.endpoints import RxComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(RxComponent())
    obj.get_component("Transform").position = list(position)
    return obj


def _radiomap_object(name):
    from .components.radiomap import RadioMapComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(RadioMapComponent())
    return obj


def _pathsolver_object(name):
    from .components.pathsolver import PathSolverComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(PathSolverComponent())
    return obj


def _ground_object():
    import witwin.channel as ch  # noqa: F401  (ensures the channel material API is importable)
    from witwin.core import Box, Material, Structure
    from witwin_server.features.platform.bridge import StructureMap

    from .adapter.material_map import ChannelMaterialMap

    obj = StructureMap.to_studio(Structure(
        Box(size=(20.0, 0.2, 20.0), position=(0.0, 0.0, 0.0)),  # horizontal slab (Y up)
        Material.from_itu("concrete"), name="Channel Ground"))
    ChannelMaterialMap.overlay_studio(obj, Material.from_itu("concrete"))
    return obj


# --- library factories (context.scene gets the extras; return the primary obj) ---

def _make_demo(ctx):
    """Solvable demo: ground + Tx + Rx + Path Solver (extras) and a RadioMap (returned)."""
    ctx.scene.add_object(_ground_object())
    ctx.scene.add_object(_tx_object("Tx", (-5.0, 3.0, 0.0)))   # Y is up/height
    ctx.scene.add_object(_rx_object("Rx", (8.0, 1.5, 0.0)))
    ctx.scene.add_object(_pathsolver_object("Path Solver"))
    return _radiomap_object("RadioMap")


def _make_tx(ctx):
    return _tx_object(ctx.name, (0.0, 2.0, 0.0))


def _make_rx(ctx):
    return _rx_object(ctx.name, (5.0, 1.5, 0.0))


def _make_radiomap(ctx):
    return _radiomap_object(ctx.name)


def _make_pathsolver(ctx):
    return _pathsolver_object(ctx.name)


def register() -> None:
    """Register the Channel library category + prefab items."""
    Library.register_category(_CATEGORY, "Channel", icon="radio", order=21)
    Library.register_item("channel_demo", "Channel Scene (Demo)", _CATEGORY,
                          icon="sparkles", object_type="empty", factory=_make_demo,
                          description="Tx + Rx + RadioMap over a ground plane; Solve for a radiomap")
    Library.register_item("channel_tx", "Tx", _CATEGORY,
                          icon="radio", object_type="empty", factory=_make_tx,
                          description="Channel transmitter endpoint (antenna inline)")
    Library.register_item("channel_rx", "Rx", _CATEGORY,
                          icon="antenna", object_type="empty", factory=_make_rx,
                          description="Channel point receiver endpoint (antenna inline)")
    Library.register_item("channel_radiomap", "RadioMap", _CATEGORY,
                          icon="grid", object_type="empty", factory=_make_radiomap,
                          description="2D radiomap field: receiver grid + solver + result viewer")
    Library.register_item("channel_pathsolver", "Path Solver", _CATEGORY,
                          icon="route", object_type="empty", factory=_make_pathsolver,
                          description="Point-to-point path solve; auto-finds the Tx/Rx")


register()
