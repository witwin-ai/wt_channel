"""Endpoint interchange: Core 0.4 endpoints <-> studio Empty objects.

Each endpoint is an ``Empty`` ``SceneObject`` carrying the matching component, its world
position on the object's ``Transform``, and its antenna inline on the component. A tx
``witwin.core.AntennaState`` <-> a ``ChannelTx`` object; a point rx ``AntennaState`` <-> a
``ChannelRx`` object; a ``witwin.core.ReceiverGrid`` <-> a ``RadioMap`` object (the
radiomap *is* the grid). Core endpoints carry no name, so the Studio object name rides
``metadata["name"]``. ``orientation`` is the antenna euler (kept off the display
Transform).

The grid samples are the RadioMap Field-region cell centers. Its axes are the plane's
tangential world axes (u -> ``x_axis``, v -> ``y_axis``) and ``shape`` is
``(n_u, n_v)``; the solvers publish a grid map as ``(T, n_v, n_u)``, which is exactly the
``data[iv, iu]`` layout the RadioMap plane visualizer consumes.

Only the plain single isotropic element is ported to witwin-channel 0.5; arrays and
non-isotropic patterns fail loudly instead of silently solving a different antenna.
"""
import math
from typing import Any, Optional, Tuple

from witwin_server.api import SceneObject

from .array_spec import ArraySpecMap
from .common import num, opt_vec3, vec

# Component names that identify each endpoint object.
TX_COMPONENT = "ChannelTx"
RX_COMPONENT = "ChannelRx"
RADIOMAP_COMPONENT = "RadioMap"

# The RadioMap reuses the inherited Field region: plane -> (normal, tangential0, tangential1).
_PLANE_INFO = {"XY": (2, 0, 1), "XZ": (1, 0, 2), "YZ": (0, 1, 2)}
_SPEED_OF_LIGHT = 299792458.0
# Studio is Y up. witwin.channel defaults an unpolarized endpoint to +Z, which is
# horizontal here, so receivers without an explicit polarization use Studio vertical.
STUDIO_VERTICAL = (0.0, 1.0, 0.0)


def endpoint_name(endpoint: Any) -> str:
    """The Studio object name an endpoint was built from ('' when unnamed)."""
    return str((endpoint.metadata or {}).get("name", ""))


class EndpointMap:
    """Core endpoints <-> studio Empty objects (antenna inline on the component)."""

    # --- platform -> studio --------------------------------------------------

    @staticmethod
    def tx_to_studio(tx: Any) -> SceneObject:
        """Build a Tx Empty object from a tx ``AntennaState``."""
        from ..components.endpoints import TxComponent

        obj = SceneObject(name=endpoint_name(tx) or "Tx", mesh_type="Empty")
        comp = obj.add_component(TxComponent())
        if tx.polarization is not None:
            comp.polarization = _floats(tx.polarization)
        if tx.power_w is not None:
            comp.power = num(tx.power_w)
        EndpointMap._write_orientation(comp, tx.orientation)
        EndpointMap._set_position(obj, _floats(tx.position))
        ArraySpecMap.to_studio(comp, None)
        return obj

    @staticmethod
    def rx_to_studio(rx: Any) -> SceneObject:
        """Build an Rx Empty object from a point rx ``AntennaState``."""
        from ..components.endpoints import RxComponent

        obj = SceneObject(name=endpoint_name(rx) or "Rx", mesh_type="Empty")
        comp = obj.add_component(RxComponent())
        EndpointMap._write_polarization(comp, rx.polarization)
        EndpointMap._write_orientation(comp, rx.orientation)
        EndpointMap._set_position(obj, _floats(rx.position))
        ArraySpecMap.to_studio(comp, None)
        return obj

    @staticmethod
    def radiomap_to_studio(grid: Any, *, frequency: Optional[float] = None,
                           noise_power: Optional[float] = None) -> SceneObject:
        """Build a RadioMap (Field subclass) object from a ``ReceiverGrid`` + config.

        The object Transform sits at the grid center and ``min_bound`` / ``max_bound`` hold
        the half-extents on the plane (so the grid follows the Transform).
        """
        from ..components.radiomap import RadioMapComponent

        obj = SceneObject(name=endpoint_name(grid) or "RadioMap", mesh_type="Empty")
        comp = obj.add_component(RadioMapComponent())
        if frequency is not None:
            comp.frequency = float(frequency)
        if noise_power is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise_power)

        normal, t0, t1 = _PLANE_INFO[str(comp.plane)]
        x_axis, y_axis = _floats(grid.x_axis), _floats(grid.y_axis)
        if abs(x_axis[t0] - 1.0) > 1e-6 or abs(y_axis[t1] - 1.0) > 1e-6:
            raise ValueError(
                f"ReceiverGrid axes {x_axis}/{y_axis} do not match the RadioMap "
                f"{comp.plane} plane (+u, +v).")
        n_u, n_v = int(grid.shape[0]), int(grid.shape[1])
        d_u, d_v = (float(v) for v in _floats(grid.spacing))
        origin = _floats(grid.position)
        half_u, half_v = 0.5 * n_u * d_u, 0.5 * n_v * d_v
        center = list(origin)
        center[t0] = origin[t0] - 0.5 * d_u + half_u
        center[t1] = origin[t1] - 0.5 * d_v + half_v
        mn, mx = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        mn[t0], mx[t0] = -half_u, half_u
        mn[t1], mx[t1] = -half_v, half_v

        comp.grid_mode = "Resolution"
        comp.min_bound = mn
        comp.max_bound = mx
        res = [1.0, 1.0, 1.0]
        res[t0], res[t1] = float(n_u), float(n_v)
        comp.resolution = res
        EndpointMap._set_position(obj, center)
        return obj

    # --- studio -> platform --------------------------------------------------

    @staticmethod
    def tx_to_platform(obj: SceneObject, antenna_id: Optional[int] = None) -> Any:
        """Build a tx ``AntennaState`` from a Tx object."""
        from witwin.core import AntennaState

        comp = obj.get_component(TX_COMPONENT)
        _require_plain_antenna(obj.name, comp)
        return AntennaState(
            _antenna_id(antenna_id), "tx", _floats(EndpointMap._position(obj)),
            orientation=EndpointMap._read_orientation(comp),
            polarization=tuple(vec(comp.polarization)),
            power_w=num(comp.power),
            metadata={"name": obj.name},
        )

    @staticmethod
    def rx_to_platform(obj: SceneObject, antenna_id: Optional[int] = None) -> Any:
        """Build a point rx ``AntennaState`` from an Rx object."""
        from witwin.core import AntennaState

        comp = obj.get_component(RX_COMPONENT)
        _require_plain_antenna(obj.name, comp)
        return AntennaState(
            _antenna_id(antenna_id), "rx", _floats(EndpointMap._position(obj)),
            orientation=EndpointMap._read_orientation(comp),
            polarization=EndpointMap._read_polarization(comp) or STUDIO_VERTICAL,
            metadata={"name": obj.name},
        )

    @staticmethod
    def radiomap_to_platform(obj: SceneObject, antenna_id: Optional[int] = None,
                             grid_shape: Optional[Tuple[int, int]] = None) -> Any:
        """Build a ``ReceiverGrid`` on the RadioMap Field region's cell centers.

        The Field's ``min_bound`` / ``max_bound`` are local; the world grid is the object
        Transform position plus those extents on the selected plane. ``grid_shape``
        ``(n_u, n_v)`` solves the same region on other cells (live previews, coarse refine).
        """
        from witwin.core import ReceiverGrid

        comp = obj.get_component(RADIOMAP_COMPONENT)
        pos = vec(EndpointMap._position(obj))
        mn, mx = vec(comp.min_bound), vec(comp.max_bound)
        res = vec(comp.resolution)
        normal, t0, t1 = _PLANE_INFO[str(comp.plane)]
        lo0, hi0 = sorted((pos[t0] + mn[t0], pos[t0] + mx[t0]))
        lo1, hi1 = sorted((pos[t1] + mn[t1], pos[t1] + mx[t1]))
        n_u, n_v = _radiomap_grid_shape(
            getattr(comp, "grid_mode", "Resolution"), hi0 - lo0, hi1 - lo1,
            res[t0], res[t1], getattr(comp, "frequency", 0.0),
            getattr(comp, "points_per_wavelength", 4.0),
        )
        if grid_shape is not None:
            n_u, n_v = (max(1, int(v)) for v in grid_shape)
        d_u, d_v = (hi0 - lo0) / n_u, (hi1 - lo1) / n_v
        origin = [0.0, 0.0, 0.0]
        origin[t0] = lo0 + 0.5 * d_u
        origin[t1] = lo1 + 0.5 * d_v
        origin[normal] = pos[normal] + mn[normal]
        x_axis, y_axis = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        x_axis[t0], y_axis[t1] = 1.0, 1.0
        return ReceiverGrid(
            _antenna_id(antenna_id), origin, x_axis, y_axis, (n_u, n_v), (d_u, d_v),
            polarization=STUDIO_VERTICAL, metadata={"name": obj.name},
        )

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
        values = _floats(orientation)
        if len(values) != 3:
            raise ValueError("only (yaw, pitch, roll) euler antenna orientations are supported")
        comp.use_orientation = True
        comp.orientation = values

    @staticmethod
    def _read_orientation(comp: Any) -> Optional[Tuple[float, float, float]]:
        return opt_vec3(comp.orientation) if bool(comp.use_orientation) else None

    @staticmethod
    def _write_polarization(comp: Any, polarization: Optional[Any]) -> None:
        if polarization is None:
            return
        comp.use_polarization = True
        comp.polarization = _floats(polarization)

    @staticmethod
    def _read_polarization(comp: Any) -> Optional[Tuple[float, float, float]]:
        return opt_vec3(comp.polarization) if bool(comp.use_polarization) else None


def _floats(value: Any) -> list:
    seq = value.detach().cpu().tolist() if hasattr(value, "detach") else vec(value)
    return [float(v) for v in seq]


def _antenna_id(antenna_id: Optional[int]) -> Any:
    from witwin.core import new_antenna_id

    return new_antenna_id() if antenna_id is None else int(antenna_id)


def _require_plain_antenna(name: str, comp: Any) -> None:
    # witwin-channel 0.5 radiomap solvers advertise no antenna arrays; only the single
    # isotropic element is ported, so anything else must not silently solve as one.
    if str(comp.array_kind) != "single":
        raise ValueError(f"{name}: antenna array '{comp.array_kind}' is not supported "
                         "on witwin-channel 0.5 yet (use a single element).")
    if str(comp.pattern) != "iso":
        raise ValueError(f"{name}: antenna pattern '{comp.pattern}' is not supported "
                         "on witwin-channel 0.5 yet (use 'iso').")


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
