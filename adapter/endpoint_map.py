"""Endpoint interchange: channel endpoints <-> studio Empty objects.

Each platform endpoint (``Transmitter`` / ``Receiver`` / ``ReceiverGrid``) becomes an
``Empty`` ``SceneObject`` carrying the matching component, its world position on the
object's ``Transform``, and (when set) a sibling ``AntennaArraySpec`` for ``.array``.
Scene-default ``tx_array`` / ``rx_array`` ride dedicated ``ChannelDefaultArray``
objects. ``orientation`` is the antenna euler (kept off the display Transform).
"""
from typing import Any, List, Optional, Tuple

from witwin_server import SceneObject

from .array_spec import ArraySpecMap
from .common import num, opt_vec3, vec

# Component names that identify each endpoint object.
TX_COMPONENT = "Transmitter"
RX_COMPONENT = "Receiver"
GRID_COMPONENT = "ReceiverGrid"
DEFAULT_ARRAY_COMPONENT = "ChannelDefaultArray"
ENDPOINT_COMPONENTS = (TX_COMPONENT, RX_COMPONENT, GRID_COMPONENT)


class EndpointMap:
    """Channel endpoints + scene-default arrays <-> studio Empty objects."""

    # --- platform -> studio --------------------------------------------------

    @staticmethod
    def tx_to_studio(tx: Any) -> SceneObject:
        """Build a Transmitter Empty object from a platform ``Transmitter``."""
        from ..components.endpoints import TransmitterComponent

        obj = SceneObject(name=tx.name, mesh_type="Empty")
        comp = obj.add_component(TransmitterComponent())
        comp.polarization = list(tx.polarization)
        comp.power = float(tx.power)
        EndpointMap._write_orientation(comp, tx.orientation)
        EndpointMap._set_position(obj, tx.position)
        EndpointMap._attach_array(obj, tx.array)
        return obj

    @staticmethod
    def rx_to_studio(rx: Any) -> SceneObject:
        """Build a Receiver Empty object from a platform ``Receiver``."""
        from ..components.endpoints import ReceiverComponent

        obj = SceneObject(name=rx.name, mesh_type="Empty")
        comp = obj.add_component(ReceiverComponent())
        EndpointMap._write_polarization(comp, rx.polarization)
        EndpointMap._write_orientation(comp, rx.orientation)
        EndpointMap._set_position(obj, rx.position)
        EndpointMap._attach_array(obj, rx.array)
        return obj

    @staticmethod
    def grid_to_studio(grid: Any) -> SceneObject:
        """Build a ReceiverGrid Empty object from a platform ``ReceiverGrid``."""
        from ..components.endpoints import ReceiverGridComponent

        obj = SceneObject(name=grid.name, mesh_type="Empty")
        comp = obj.add_component(ReceiverGridComponent())
        comp.axis = str(grid.axis)
        comp.position = float(grid.position)
        comp.bounds = [float(grid.bounds[0][0]), float(grid.bounds[0][1]),
                       float(grid.bounds[1][0]), float(grid.bounds[1][1])]
        if grid.grid_shape is not None:
            comp.grid_mode = "grid_shape"
            comp.grid_nx, comp.grid_ny = int(grid.grid_shape[0]), int(grid.grid_shape[1])
        else:
            comp.grid_mode = "cell_size"
            comp.cell_dx, comp.cell_dy = float(grid.cell_size[0]), float(grid.cell_size[1])
        EndpointMap._write_polarization(comp, grid.polarization)
        EndpointMap._write_orientation(comp, grid.orientation)
        EndpointMap._set_position(obj, EndpointMap._grid_center(grid.axis, grid.position, grid.bounds))
        EndpointMap._attach_array(obj, grid.array)
        return obj

    @staticmethod
    def default_array_to_studio(role: str, array: Any) -> SceneObject:
        """Build a scene-default-array Empty (Scene.tx_array / rx_array)."""
        from ..components.antenna import (
            AntennaArraySpecComponent,
            ChannelDefaultArrayComponent,
        )

        obj = SceneObject(name=f"Channel {role.upper()} Array", mesh_type="Empty")
        obj.add_component(ChannelDefaultArrayComponent()).role = role
        ArraySpecMap.to_studio(obj.add_component(AntennaArraySpecComponent()), array)
        return obj

    # --- studio -> platform --------------------------------------------------

    @staticmethod
    def tx_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``Transmitter`` from a Transmitter object."""
        comp = obj.get_component(TX_COMPONENT)
        return ch.Transmitter(
            name=obj.name, position=tuple(vec(EndpointMap._position(obj))),
            polarization=tuple(vec(comp.polarization)), power=num(comp.power),
            orientation=EndpointMap._read_orientation(comp),
            array=ArraySpecMap.from_object(ch, obj))

    @staticmethod
    def rx_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``Receiver`` from a Receiver object."""
        comp = obj.get_component(RX_COMPONENT)
        return ch.Receiver(
            name=obj.name, position=tuple(vec(EndpointMap._position(obj))),
            polarization=EndpointMap._read_polarization(comp),
            orientation=EndpointMap._read_orientation(comp),
            array=ArraySpecMap.from_object(ch, obj))

    @staticmethod
    def grid_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Build a platform ``ReceiverGrid`` from a ReceiverGrid object."""
        comp = obj.get_component(GRID_COMPONENT)
        b = vec(comp.bounds)
        bounds = ((b[0], b[1]), (b[2], b[3]))
        grid_shape = cell_size = None
        if str(comp.grid_mode) == "grid_shape":
            grid_shape = (int(comp.grid_nx), int(comp.grid_ny))
        else:
            cell_size = (num(comp.cell_dx), num(comp.cell_dy))
        return ch.ReceiverGrid(
            name=obj.name, axis=str(comp.axis), position=num(comp.position), bounds=bounds,
            grid_shape=grid_shape, cell_size=cell_size,
            polarization=EndpointMap._read_polarization(comp),
            orientation=EndpointMap._read_orientation(comp),
            array=ArraySpecMap.from_object(ch, obj))

    @staticmethod
    def receiver_to_platform(ch: Any, obj: SceneObject) -> Any:
        """Dispatch a receiver object to a point Receiver or a ReceiverGrid."""
        if obj.get_component(GRID_COMPONENT) is not None:
            return EndpointMap.grid_to_platform(ch, obj)
        return EndpointMap.rx_to_platform(ch, obj)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _attach_array(obj: SceneObject, array: Optional[Any]) -> None:
        # Give the endpoint object an AntennaArraySpec iff it has an explicit array.
        if array is None:
            return
        from ..components.antenna import AntennaArraySpecComponent

        ArraySpecMap.to_studio(obj.add_component(AntennaArraySpecComponent()), array)

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
