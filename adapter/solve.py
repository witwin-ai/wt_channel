"""Solver hand-off + radiomap extraction (master §6 / §7).

``SolveRunner`` is the export->solve half of the edit loop: it rebuilds the Core world
from the live studio scene (via :class:`ChannelAdapter`), restricted to the chosen
transmitter + receiver grid, and runs the chosen witwin-channel 0.5 family —
``deterministic``, ``montecarlo.basic`` or ``montecarlo.bdpt`` — all of which share the
entry shape ``solve(scene, config, *, reference_frequency_hz)`` and need CUDA RayD.

``ConfigBuilder`` reads the RadioMap component's curated fields into the solver
``Config``. witwin-channel 0.5 dropped the 0.3 tuning / integrator-option / filter /
edge-policy knobs; the component keeps those fields for saved scenes, but only depth,
diffraction order, sample count and seed reach the solver now. ``RadioMapMaps`` holds
the solved ``(T, n_v, n_u)`` path gain and derives ``rss`` / ``sinr``;
``RadioMapExtractor`` turns it into a 2D numpy map (+ extent) for the Results-panel /
figure surfaces. Result tensors are runtime state — never serialized into the scene.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .channel_adapter import ChannelAdapter
from .common import num

# The public diffraction capability of every 0.5 solver is first order.
_MAX_DIFFRACTION_ORDER = 1


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
    """Read the RadioMap component into witwin-channel 0.5 solver configs."""

    @staticmethod
    def deterministic(comp: Any) -> Any:
        """Build a ``deterministic.Config`` from the component."""
        from witwin.channel import deterministic

        bounces = int(comp.det_max_bounces)
        order = min(int(comp.det_max_diffraction_order), _MAX_DIFFRACTION_ORDER)
        return deterministic.Config(
            max_depth=bounces,
            max_diffraction_order=max(order, 1),
            components=_components(bounces, order),
            return_field=False,
        )

    @staticmethod
    def montecarlo(comp: Any) -> Any:
        """Build a ``montecarlo.basic.Config`` / ``montecarlo.bdpt.Config``."""
        from witwin.channel.montecarlo import basic, bdpt

        bounces = int(comp.mc_max_bounces)
        order = min(int(comp.mc_max_diffraction_order), _MAX_DIFFRACTION_ORDER)
        common = dict(samples=int(comp.mc_samples_per_tx), max_depth=bounces,
                      seed=int(comp.mc_seed), components=_components(bounces, order))
        if str(comp.mc_integrator) == "bdpt":
            return bdpt.Config(max_diffraction_order=max(order, 1), **common)
        return basic.Config(**common)


class SolveRunner:
    """Rebuild the Core world from the studio scene and run a radiomap solve."""

    @staticmethod
    def run(studio_scene: Any, comp: Any, *, transmitter: str, receiver: str) -> RadioMapMaps:
        """Run the chosen solver family for the given Tx/grid."""
        frequency = num(comp.frequency)
        if frequency <= 0.0:
            raise ValueError("radiomap frequency must be positive")
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        world = ChannelAdapter().build(studio_scene, frequency=frequency, noise_power=noise,
                                       transmitter=transmitter, receiver=receiver)
        tx = ChannelAdapter.endpoint_by_name(world, transmitter)
        if str(comp.solver) == "montecarlo":
            from witwin.channel.montecarlo import basic, bdpt

            config = ConfigBuilder.montecarlo(comp)
            solver = bdpt if isinstance(config, bdpt.Config) else basic
        else:
            from witwin.channel import deterministic as solver

            config = ConfigBuilder.deterministic(comp)
        result = solver.solve(world, config, reference_frequency_hz=frequency)
        grid = ChannelAdapter.endpoint_by_name(world, receiver)
        gain = grid_map(_to_numpy(result.path_gain), tuple(grid.shape))
        return RadioMapMaps(path_gain=gain, tx_power_w=np.array([num(tx.power_w)], np.float32),
                            noise_power_w=noise)


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

    The deterministic solver publishes ``(T, n_v, n_u)``; Monte Carlo basic publishes
    ``(T, n_v * n_u)``, row-major over ``(v, u)`` (its component maps carry the 2D layout).
    """
    n_u, n_v = int(grid_shape[0]), int(grid_shape[1])
    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim == 3 and gain.shape[1:] == (n_v, n_u):
        return gain
    if gain.ndim == 2 and gain.shape[1] == n_v * n_u:
        return gain.reshape(gain.shape[0], n_v, n_u)
    raise ValueError(f"expected a (T, {n_v}, {n_u}) grid map, got shape {gain.shape}")


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)
