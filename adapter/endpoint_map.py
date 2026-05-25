"""Endpoint interchange: channel endpoints <-> studio Empty objects.

Each platform endpoint becomes an ``Empty`` ``SceneObject`` carrying the matching
component, its world position on the object's ``Transform``, and its antenna inline on
the component. A platform ``Transmitter`` <-> a ``ChannelTx`` object; a point
``Receiver`` <-> a ``ChannelRx`` object; a ``ReceiverGrid`` <-> a ``RadioMap`` object
(the radiomap *is* the grid, named after itself). ``orientation`` is the antenna euler
(kept off the display Transform). Scene-default ``tx_array`` / ``rx_array`` are baked
into each endpoint that lacks an explicit array on import.
"""
from typing import Any, Optional, Tuple

from witwin_server import SceneObject

from .array_spec import ArraySpecMap
from .common import num, opt_vec3, vec

# Component names that identify each endpoint object.
TX_COMPONENT = "ChannelTx"
RX_COMPONENT = "ChannelRx"
RADIOMAP_COMPONENT = "RadioMap"


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
        """Build a RadioMap Empty object from a platform ``ReceiverGrid`` + scene config."""
        from ..components.radiomap import RadioMapComponent

        obj = SceneObject(name=grid.name, mesh_type="Empty")
        comp = obj.add_component(RadioMapComponent())
        comp.frequency = 0.0 if frequency is None else float(frequency)
        comp.device = device
        if noise_power is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise_power)
        comp.axis = str(grid.axis)
        comp.height = float(grid.position)
        comp.bounds = [float(grid.bounds[0][0]), float(grid.bounds[0][1]),
                       float(grid.bounds[1][0]), float(grid.bounds[1][1])]
        if grid.grid_shape is not None:
            comp.grid_mode = "grid_shape"
            comp.grid_nx, comp.grid_ny = int(grid.grid_shape[0]), int(grid.grid_shape[1])
        else:
            comp.grid_mode = "cell_size"
            comp.cell_dx, comp.cell_dy = float(grid.cell_size[0]), float(grid.cell_size[1])
        EndpointMap._set_position(obj, EndpointMap._grid_center(grid.axis, grid.position, grid.bounds))
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
        """Build a platform ``ReceiverGrid`` from a RadioMap object (the grid)."""
        comp = obj.get_component(RADIOMAP_COMPONENT)
        b = vec(comp.bounds)
        bounds = ((b[0], b[1]), (b[2], b[3]))
        grid_shape = cell_size = None
        if str(comp.grid_mode) == "grid_shape":
            grid_shape = (int(comp.grid_nx), int(comp.grid_ny))
        else:
            cell_size = (num(comp.cell_dx), num(comp.cell_dy))
        return ch.ReceiverGrid(name=obj.name, axis=str(comp.axis), position=num(comp.height),
                               bounds=bounds, grid_shape=grid_shape, cell_size=cell_size)

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
    def _grid_center(axis: str, position: float, bounds: Any) -> Tuple[float, float, float]:
        # Plane center (for display Transform placement); see core Grid.center.
        c0 = 0.5 * (float(bounds[0][0]) + float(bounds[0][1]))
        c1 = 0.5 * (float(bounds[1][0]) + float(bounds[1][1]))
        pos = float(position)
        if axis == "x":
            return (pos, c0, c1)
        if axis == "y":
            return (c0, pos, c1)
        return (c0, c1, pos)
