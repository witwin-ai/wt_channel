"""Endpoint interchange: channel endpoints <-> studio Empty objects.

Each platform endpoint becomes an ``Empty`` ``SceneObject`` carrying the matching
component, its world position on the object's ``Transform``, and its antenna inline on
the component. A platform ``Transmitter`` <-> a ``ChannelTx`` object; a point
``Receiver`` <-> a ``ChannelRx`` object; a ``ReceiverGrid`` <-> a ``RadioMap`` object
(the radiomap *is* the grid, named after itself). ``orientation`` is the antenna euler
(kept off the display Transform). Scene-default ``tx_array`` / ``rx_array`` are baked
into each endpoint that lacks an explicit array on import.
"""
import math
from typing import Any, Optional, Tuple

from witwin_server import SceneObject

from .array_spec import ArraySpecMap
from .common import num, opt_vec3, vec

# Component names that identify each endpoint object.
TX_COMPONENT = "ChannelTx"
RX_COMPONENT = "ChannelRx"
RADIOMAP_COMPONENT = "RadioMap"

# The RadioMap reuses the inherited Field region: plane <-> grid axis + tangential axes.
# plane -> (grid axis, normal index, tangential0 index, tangential1 index).
_PLANE_INFO = {"XY": ("z", 2, 0, 1), "XZ": ("y", 1, 0, 2), "YZ": ("x", 0, 1, 2)}
_AXIS_TO_PLANE = {"z": "XY", "y": "XZ", "x": "YZ"}
_SPEED_OF_LIGHT = 299792458.0


class EndpointMap:
    """Channel endpoints <-> studio Empty objects (antenna inline on the component)."""

    # --- platform -> studio --------------------------------------------------

    @staticmethod
    def tx_to_studio(tx: Any, scene_default_array: Optional[Any] = None) -> SceneObject:
        """Build a Tx Empty object from a platform ``Transmitter``."""
        from ..components.endpoints import TxComponent

        obj = SceneObject(name=tx.name, mesh_type="Empty")
        comp = obj.add_component(TxComponent())
        if tx.polarization is not None:
            comp.polarization = list(tx.polarization)
        comp.power = float(tx.power)
        EndpointMap._write_orientation(comp, tx.orientation)
        EndpointMap._set_position(obj, tx.position)
        ArraySpecMap.to_studio(comp, tx.array if tx.array is not None else scene_default_array)
        return obj

    @staticmethod
    def rx_to_studio(rx: Any, scene_default_array: Optional[Any] = None) -> SceneObject:
        """Build an Rx Empty object from a platform ``Receiver``."""
        from ..components.endpoints import RxComponent

        obj = SceneObject(name=rx.name, mesh_type="Empty")
        comp = obj.add_component(RxComponent())
        EndpointMap._write_polarization(comp, rx.polarization)
        EndpointMap._write_orientation(comp, rx.orientation)
        EndpointMap._set_position(obj, rx.position)
        ArraySpecMap.to_studio(comp, rx.array if rx.array is not None else scene_default_array)
        return obj

    @staticmethod
    def radiomap_to_studio(grid: Any, *, frequency: Optional[float], device: str,
                           noise_power: Optional[float] = None) -> SceneObject:
        """Build a RadioMap (Field subclass) object from a platform ``ReceiverGrid`` + config.

        The grid is stored in the inherited Field's local frame: the object Transform sits
        at the grid center and ``min_bound`` / ``max_bound`` hold the half-extents on the
        plane (so the grid follows the Transform).
        """
        from ..components.radiomap import RadioMapComponent

        obj = SceneObject(name=grid.name, mesh_type="Empty")
        comp = obj.add_component(RadioMapComponent())
        comp.frequency = 3.5e9 if frequency is None else float(frequency)
        comp.device = device
        if noise_power is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise_power)

        plane = _AXIS_TO_PLANE[str(grid.axis)]
        _, normal, t0, t1 = _PLANE_INFO[plane]
        b = grid.bounds
        c0, c1 = 0.5 * (b[0][0] + b[0][1]), 0.5 * (b[1][0] + b[1][1])
        center = [0.0, 0.0, 0.0]
        center[t0], center[t1], center[normal] = c0, c1, float(grid.position)
        mn, mx = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        mn[t0], mx[t0] = b[0][0] - c0, b[0][1] - c0
        mn[t1], mx[t1] = b[1][0] - c1, b[1][1] - c1

        comp.plane = plane
        comp.grid_mode = "Resolution"
        comp.min_bound = mn
        comp.max_bound = mx
        comp.resolution = EndpointMap._grid_resolution(grid, normal, t0, t1)
        EndpointMap._set_position(obj, center)
        return obj

    # --- studio -> platform --------------------------------------------------

    @staticmethod
    def tx_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``Transmitter`` from a Tx object."""
        comp = obj.get_component(TX_COMPONENT)
        return ch.Transmitter(
            name=obj.name, position=tuple(vec(EndpointMap._position(obj))),
            polarization=tuple(vec(comp.polarization)), power=num(comp.power),
            orientation=EndpointMap._read_orientation(comp),
            array=ArraySpecMap.to_platform(ch, comp))

    @staticmethod
    def rx_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``Receiver`` from an Rx object."""
        comp = obj.get_component(RX_COMPONENT)
        return ch.Receiver(
            name=obj.name, position=tuple(vec(EndpointMap._position(obj))),
            polarization=EndpointMap._read_polarization(comp),
            orientation=EndpointMap._read_orientation(comp),
            array=ArraySpecMap.to_platform(ch, comp))

    @staticmethod
    def radiomap_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``ReceiverGrid`` from a RadioMap object (the Field region).

        The Field's ``min_bound`` / ``max_bound`` are local; the world grid is the object
        Transform position plus those extents on the selected plane.
        """
        comp = obj.get_component(RADIOMAP_COMPONENT)
        pos = vec(EndpointMap._position(obj))
        mn, mx = vec(comp.min_bound), vec(comp.max_bound)
        res = vec(comp.resolution)
        axis, normal, t0, t1 = _PLANE_INFO[str(comp.plane)]
        bounds = ((pos[t0] + mn[t0], pos[t0] + mx[t0]), (pos[t1] + mn[t1], pos[t1] + mx[t1]))
        position = pos[normal] + mn[normal]
        size0 = abs(bounds[0][1] - bounds[0][0])
        size1 = abs(bounds[1][1] - bounds[1][0])
        grid_shape = _radiomap_grid_shape(
            getattr(comp, "grid_mode", "Resolution"),
            size0,
            size1,
            res[t0],
            res[t1],
            getattr(comp, "frequency", 0.0),
            getattr(comp, "points_per_wavelength", 4.0),
        )
        return ch.ReceiverGrid(name=obj.name, axis=axis, position=position,
                                bounds=bounds, grid_shape=grid_shape)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _set_position(obj: SceneObject, position: Any) -> None:
        obj.get_component("Transform").position = [float(v) for v in position]

    @staticmethod
    def _position(obj: SceneObject) -> Any:
        return obj.get_component("Transform").position

    @staticmethod
    def _write_orientation(comp: Any, orientation: Optional[Any]) -> None:
        if orientation is None:
            return
        comp.use_orientation = True
        comp.orientation = list(orientation)

    @staticmethod
    def _read_orientation(comp: Any) -> Optional[Tuple[float, float, float]]:
        return opt_vec3(comp.orientation) if bool(comp.use_orientation) else None

    @staticmethod
    def _write_polarization(comp: Any, polarization: Optional[Any]) -> None:
        if polarization is None:
            return
        comp.use_polarization = True
        comp.polarization = list(polarization)

    @staticmethod
    def _read_polarization(comp: Any) -> Optional[Tuple[float, float, float]]:
        return opt_vec3(comp.polarization) if bool(comp.use_polarization) else None

    @staticmethod
    def _grid_resolution(grid: Any, normal: int, t0: int, t1: int) -> list:
        # Field resolution vector3; tangential axes from grid_shape (or cell_size counts).
        res = [1, 1, 1]
        if grid.grid_shape is not None:
            res[t0], res[t1] = int(grid.grid_shape[0]), int(grid.grid_shape[1])
            return res
        cs = grid.cell_size
        d0 = cs[0] if isinstance(cs, (tuple, list)) else cs
        d1 = cs[1] if isinstance(cs, (tuple, list)) else cs
        e0 = abs(grid.bounds[0][1] - grid.bounds[0][0])
        e1 = abs(grid.bounds[1][1] - grid.bounds[1][0])
        res[t0] = max(1, round(e0 / float(d0)))
        res[t1] = max(1, round(e1 / float(d1)))
        return res


def _radiomap_grid_shape(
    grid_mode: Any,
    size0: float,
    size1: float,
    res0: float,
    res1: float,
    frequency: Any,
    points_per_wavelength: Any,
) -> tuple[int, int]:
    if str(grid_mode) == "PPW":
        freq = max(float(frequency or 0.0), 1e-30)
        ppw = max(float(points_per_wavelength or 4.0), 1e-30)
        cell = _SPEED_OF_LIGHT / freq / ppw
        return (max(1, int(math.ceil(size0 / cell))), max(1, int(math.ceil(size1 / cell))))
    return (max(1, int(res0)), max(1, int(res1)))
