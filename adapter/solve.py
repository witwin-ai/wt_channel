"""Solver hand-off + radiomap extraction (master §6 / §7).

``SolveRunner`` is the export->solve half of the edit loop: it rebuilds the Core world
from the live studio scene (via :class:`ChannelAdapter`), restricted to the chosen
transmitter + receiver grid, and runs one witwin-channel 0.6 radio-map estimator:
``radiomap.Coherent`` (the component's ``deterministic`` solver),
``radiomap.Stochastic`` (``montecarlo`` with the ``basic`` integrator) or
``experimental.bdpt`` (``montecarlo`` with ``bdpt``). All share the entry shape
``solve(scene, config, *, reference_frequency_hz)`` and need CUDA RayD.

``ConfigBuilder`` reads the RadioMap component's curated fields into the solver
config. witwin-channel 0.5 dropped the 0.3 tuning / integrator-option / filter /
edge-policy knobs; the component keeps those fields for saved scenes, but only depth,
diffraction order, sample count and seed reach the solver now. ``RadioMapMaps`` holds
the solved ``(T, n_v, n_u)`` path gain and derives ``rss`` / ``sinr``;
``RadioMapExtractor`` turns it into a 2D numpy map (+ extent) for the Results-panel /
figure surfaces. Result tensors are runtime state — never serialized into the scene.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .channel_adapter import ChannelAdapter
from .common import num

# The public diffraction capability of every 0.6 solver is first order.
_MAX_DIFFRACTION_ORDER = 1
# witwin-channel 0.6 refuses a grid sample closer to a transmitter than
# max(wavelength / (4 pi), 2^-20 * |coordinate|) (6.8 mm at 3.5 GHz). A sample this many
# tolerances away is clear of the refusal with room for float32 rounding.
_COINCIDENCE_MARGIN = 2.0
_SPEED_OF_LIGHT = 299792458.0


@dataclass
class MapImage:
    """A 2D radiomap ready for imshow: data + matplotlib-style extent + label."""

    data: np.ndarray
    extent: List[float]
    label: str


@dataclass
class RadioMapMaps:
    """Solved radiomap: ``path_gain`` is ``(T, n_v, n_u)``; ``tx_power_w`` is ``(T,)``."""

    path_gain: np.ndarray
    tx_power_w: np.ndarray
    noise_power_w: Optional[float] = None
    # What the solve changed to run at all (a grid moved off a transmitter), else None.
    note: Optional[str] = None

    def metrics(self) -> Dict[str, np.ndarray]:
        """Every metric this solve supports, by name."""
        names = ["path_gain", "rss"] + (["sinr"] if self.noise_power_w else [])
        return {name: self.metric(name) for name in names}

    def metric(self, name: str) -> np.ndarray:
        """``path_gain`` (linear), ``rss`` (W) or ``sinr`` (linear, needs noise power)."""
        gain = np.asarray(self.path_gain, dtype=np.float32)
        if name == "path_gain":
            return gain
        rss = gain * np.asarray(self.tx_power_w, dtype=np.float32).reshape(-1, 1, 1)
        if name == "rss":
            return rss
        if name == "sinr":
            if not self.noise_power_w:
                raise ValueError("sinr needs a noise power (enable use_noise_power)")
            return rss / np.float32(self.noise_power_w)
        raise ValueError(f"unknown radiomap metric: {name}")


def _components(max_bounces: int, diffraction_order: int) -> frozenset:
    components = {"los"}
    if max_bounces > 0:
        components.add("reflection")
    if diffraction_order > 0:
        components.add("diffraction")
    return frozenset(components)


class ConfigBuilder:
    """Read the RadioMap component into witwin-channel 0.6 solver configs.

    The method names follow the component's ``solver`` values, which saved scenes keep.
    """

    @staticmethod
    def deterministic(comp: Any) -> Any:
        """Build a ``radiomap.Coherent`` (complex path fields summed per cell)."""
        from witwin.channel import radiomap

        bounces = int(comp.det_max_bounces)
        order = min(int(comp.det_max_diffraction_order), _MAX_DIFFRACTION_ORDER)
        return radiomap.Coherent(
            max_depth=bounces,
            components=_components(bounces, order),
            return_field=False,
        )

    @staticmethod
    def montecarlo(comp: Any) -> Any:
        """Build a ``radiomap.Stochastic`` or, for ``bdpt``, an ``experimental.bdpt.Config``."""
        bounces = int(comp.mc_max_bounces)
        order = min(int(comp.mc_max_diffraction_order), _MAX_DIFFRACTION_ORDER)
        common = dict(samples=int(comp.mc_samples_per_tx), max_depth=bounces,
                      seed=int(comp.mc_seed), components=_components(bounces, order))
        if str(comp.mc_integrator) == "bdpt":
            from witwin.channel.experimental import bdpt

            return bdpt.Config(max_diffraction_order=max(order, 1), **common)
        from witwin.channel import radiomap

        return radiomap.Stochastic(**common)


class SolveRunner:
    """Rebuild the Core world from the studio scene and run a radiomap solve."""

    @staticmethod
    def run(studio_scene: Any, comp: Any, *, transmitter: str, receiver: str,
            grid_shape: Optional[Tuple[int, int]] = None) -> RadioMapMaps:
        """Run the chosen solver for the given Tx/grid (``grid_shape`` overrides its cells)."""
        frequency = num(comp.frequency)
        if frequency <= 0.0:
            raise ValueError("radiomap frequency must be positive")
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        world = ChannelAdapter().build(studio_scene, frequency=frequency, noise_power=noise,
                                       transmitter=transmitter, receiver=receiver,
                                       grid_shape=grid_shape)
        world, note = clear_grids_of_transmitters(world, frequency)
        tx = ChannelAdapter.endpoint_by_name(world, transmitter)
        if str(comp.solver) == "montecarlo":
            config = ConfigBuilder.montecarlo(comp)
            if str(comp.mc_integrator) == "bdpt":
                from witwin.channel.experimental import bdpt as solver
            else:
                from witwin.channel import radiomap as solver
        else:
            from witwin.channel import radiomap as solver

            config = ConfigBuilder.deterministic(comp)
        result = solver.solve(world, config, reference_frequency_hz=frequency)
        grid = ChannelAdapter.endpoint_by_name(world, receiver)
        gain = grid_map(_to_numpy(result.path_gain), tuple(grid.shape))
        return RadioMapMaps(path_gain=gain, tx_power_w=np.array([num(tx.power_w)], np.float32),
                            noise_power_w=noise, note=note)


def clear_grids_of_transmitters(world: Any, frequency: float) -> Tuple[Any, Optional[str]]:
    """Move a receiver grid a quarter cell off a transmitter it would sample.

    witwin-channel 0.6 refuses a grid whose sample lies within the coincidence tolerance
    of a transmitter (free space is undefined there). A quarter cell along both grid
    axes keeps every sample inside the cell the RadioMap draws it in, so the displayed
    map stays aligned with its region; half a cell, the upstream advice, would push the
    edge samples outside it. Returns the (possibly rebuilt) world and a note naming the
    shift, or ``None`` when no grid needed one. A grid too fine for a quarter cell to
    clear the tolerance raises ``ValueError`` naming the RadioMap and the Tx.
    """
    from witwin.core import ReceiverGrid

    transmitters = [e for e in world.endpoints if e.role == "tx"]
    tolerance = _SPEED_OF_LIGHT / float(frequency) / (4.0 * np.pi)
    endpoints, notes = [], []
    for endpoint in world.endpoints:
        if not isinstance(endpoint, ReceiverGrid):
            endpoints.append(endpoint)
            continue
        hit = _nearest_transmitter(endpoint, transmitters, tolerance)
        if hit is None:
            endpoints.append(endpoint)
            continue
        spacing = _floats(endpoint.spacing)
        offset = (0.25 * spacing[0] * _unit(endpoint.x_axis)
                  + 0.25 * spacing[1] * _unit(endpoint.y_axis))
        shifted = _with_origin(endpoint, np.asarray(_floats(endpoint.position)) + offset)
        if _nearest_transmitter(shifted, transmitters, tolerance) is not None:
            raise ValueError(
                f"RadioMap {endpoint_label(endpoint)!r} samples a point within "
                f"{tolerance * 1e3:.1f} mm of transmitter {endpoint_label(hit)!r}, where free "
                "space is undefined, and its cells are too small to step aside. Move the "
                "transmitter off the map plane or use coarser cells.")
        endpoints.append(shifted)
        notes.append(f"RadioMap {endpoint_label(endpoint)!r} moved a quarter cell off "
                     f"transmitter {endpoint_label(hit)!r}")
    if not notes:
        return world, None
    import witwin.core as wc

    rebuilt = wc.Scene(structures=tuple(world.structures), endpoints=tuple(endpoints),
                       metadata=dict(world.metadata or {}))
    return rebuilt, "; ".join(notes)


def endpoint_label(endpoint: Any) -> str:
    """The Studio object name of an endpoint, else its antenna id."""
    return str((endpoint.metadata or {}).get("name") or endpoint.antenna_id)


def _nearest_transmitter(grid: Any, transmitters: List[Any], tolerance: float) -> Any:
    """The first transmitter within ``_COINCIDENCE_MARGIN`` tolerances of a grid sample."""
    origin = np.asarray(_floats(grid.position), dtype=np.float64)
    axes = (_unit(grid.x_axis), _unit(grid.y_axis))
    spacing = _floats(grid.spacing)
    shape = tuple(int(v) for v in grid.shape)
    for tx in transmitters:
        position = np.asarray(_floats(tx.position), dtype=np.float64)
        rel = position - origin
        sample = origin.copy()
        for axis, step, count in zip(axes, spacing, shape):
            index = min(max(round(float(rel @ axis) / step), 0), count - 1)
            sample = sample + index * step * axis
        limit = max(tolerance, 2.0 ** -20 * float(np.abs(position).max()))
        if float(np.linalg.norm(sample - position)) < _COINCIDENCE_MARGIN * limit:
            return tx
    return None


def _with_origin(grid: Any, origin: np.ndarray) -> Any:
    from witwin.core import ReceiverGrid

    return ReceiverGrid(
        grid.antenna_id, [float(v) for v in origin], _floats(grid.x_axis),
        _floats(grid.y_axis), tuple(int(v) for v in grid.shape), tuple(_floats(grid.spacing)),
        orientation=grid.orientation, polarization=grid.polarization,
        synthetic_array=grid.synthetic_array, pattern=grid.pattern,
        metadata=dict(grid.metadata or {}),
    )


def _unit(axis: Any) -> np.ndarray:
    vector = np.asarray(_floats(axis), dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def _floats(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(v) for v in value]


class RadioMapExtractor:
    """Extract a 2D map (data + extent) from ``RadioMapMaps``."""

    @staticmethod
    def extract(maps: RadioMapMaps, *, metric: str, tx_index: int, bounds: List[float],
                log_scale: bool, vmin: Optional[float] = None,
                vmax: Optional[float] = None) -> MapImage:
        """Pick one Tx, read ``metric`` as a (n_v, n_u) array, build the extent.

        ``vmin`` / ``vmax`` (in the displayed units — dB when ``log_scale``) clamp the
        value window so the heatmap re-scales to exactly that range, without re-solving.
        """
        data = maps.metric(metric)
        data = data[max(0, min(int(tx_index), data.shape[0] - 1))]
        label = metric
        if log_scale:
            data = 10.0 * np.log10(np.abs(data) + 1e-30)
            label = f"{metric} (dB)"
        if vmin is not None or vmax is not None:
            data = np.clip(data, vmin, vmax)
        extent = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
        return MapImage(np.asarray(data, dtype=np.float32), extent, label)


def grid_map(gain: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """A solver grid result as ``(T, n_v, n_u)`` for a ReceiverGrid of shape ``(n_u, n_v)``.

    witwin-channel 0.6 returns every grid as ``[T, rows, cols]``. The RadioMap grid's
    rows run along u (``x_axis``) and its columns along v, so the solver map is
    ``(T, n_u, n_v)``; the plane visualizer reads ``data[iv, iu]``, hence the transpose.
    The transpose is unconditional because a square grid cannot tell the orders apart.
    """
    n_u, n_v = int(grid_shape[0]), int(grid_shape[1])
    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim != 3 or gain.shape[1:] != (n_u, n_v):
        raise ValueError(f"expected a (T, {n_u}, {n_v}) grid map, got shape {gain.shape}")
    return np.ascontiguousarray(gain.transpose(0, 2, 1))


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)
