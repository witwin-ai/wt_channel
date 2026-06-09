"""Solver hand-off + radiomap extraction (master §6 / §7).

``SolveRunner`` is the export->solve half of the edit loop: it rebuilds the platform
``channel.Scene`` from the live studio scene (via :class:`ChannelAdapter`), resolves
the transmitter + receiver-grid by name, and runs the chosen family —
``deterministic.solve`` or ``montecarlo.solve`` — both of which share the entry shape
``solve(*, scene, transmitter, receiver, config) -> RadioMapResult`` and require
``scene.frequency`` plus CUDA RayD/OptiX.

``ConfigBuilder`` reads the ``ChannelResult`` component's curated fields into the
platform ``Config`` / ``Tuning`` / ``IntegratorOptions`` / ``FilterConfig`` /
``EdgePolicy``. ``RadioMapExtractor`` turns a ``RadioMapResult`` into a 2D numpy map
(+ extent) for the Results-panel / figure surfaces. Result tensors are runtime state —
never serialized into the scene.
"""
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .channel_adapter import ChannelAdapter
from .common import num, vec


@dataclass
class MapImage:
    """A 2D radiomap ready for imshow: data + matplotlib-style extent + label."""

    data: np.ndarray
    extent: List[float]
    label: str


class ConfigBuilder:
    """Read the ``ChannelResult`` component into platform solver configs."""

    _MC_ACCUMULATION_DEFAULT = "auto"

    @staticmethod
    def _str_field(comp: Any, name: str, default: str) -> str:
        value = getattr(comp, name, None)
        return default if value is None or value == "" else str(value)

    @staticmethod
    def edge_policy(ch: Any, comp: Any) -> Any:
        # Shared diffraction-edge policy; edge_diffraction 'default' leaves it to the platform.
        kwargs = {
            "vertical_ratio": num(comp.ep_vertical_ratio),
            "edge_selection_mode": str(comp.ep_edge_selection_mode),
        }
        diffraction = str(comp.ep_edge_diffraction)
        if diffraction == "on":
            kwargs["edge_diffraction"] = True
        elif diffraction == "off":
            kwargs["edge_diffraction"] = False
        return ch.EdgePolicy(**kwargs)

    @staticmethod
    def deterministic(ch: Any, comp: Any) -> Any:
        """Build a ``deterministic.Config`` from the component."""
        stratified = str(comp.det_quadrature_mode) == "stratified_fixed_n"
        tuning = ch.deterministic.Tuning(
            resolution_wavelength=num(comp.det_resolution_wavelength),
            solver_mode=str(comp.det_solver_mode),
            memory_profile=str(comp.det_memory_profile),
            reflection_field_backend=ConfigBuilder._str_field(
                comp, "det_reflection_field_backend", "native"),
        )
        return ch.deterministic.Config(
            metric=str(comp.det_metric),
            quadrature_mode=str(comp.det_quadrature_mode),
            samples_per_cell=int(comp.det_samples_per_cell) if stratified else None,
            shadow_boundary_correction=bool(comp.det_shadow_boundary_correction),
            num_samples=int(comp.det_num_samples),
            max_bounces=int(comp.det_max_bounces),
            max_diffraction_order=int(comp.det_max_diffraction_order),
            synthetic_array=bool(comp.det_synthetic_array),
            edge_policy=ConfigBuilder.edge_policy(ch, comp),
            tuning=tuning,
        )

    @staticmethod
    def montecarlo(ch: Any, comp: Any) -> Any:
        """Build a ``montecarlo.Config`` from the component."""
        integrator = str(comp.mc_integrator)
        order = int(comp.mc_max_diffraction_order)
        if integrator == "basic":
            order = min(order, 1)  # basic supports diffraction depth {0, 1}
        tuning = ch.montecarlo.Tuning(
            resolution_wavelength=num(comp.mc_resolution_wavelength),
            solver_mode=str(comp.mc_solver_mode),
            memory_profile=str(comp.mc_memory_profile),
            reflection_field_backend=ConfigBuilder._str_field(
                comp, "mc_reflection_field_backend", "native"),
        )
        options = ch.montecarlo.IntegratorOptions(
            integrator=integrator,
            samples_per_tx=int(comp.mc_samples_per_tx),
            seed=int(comp.mc_seed),
            rr_depth=int(comp.mc_rr_depth) if bool(comp.mc_use_rr_depth) else None,
            rr_prob=num(comp.mc_rr_prob) if bool(comp.mc_use_rr_prob) else None,
            stop_threshold=num(comp.mc_stop_threshold) if bool(comp.mc_use_stop_threshold) else None,
            bdpt_diffraction_sampling=str(comp.mc_bdpt_diffraction_sampling),
            accumulation_backend=ConfigBuilder._str_field(
                comp, "mc_accumulation_backend", ConfigBuilder._MC_ACCUMULATION_DEFAULT),
        )
        return ch.montecarlo.Config(
            num_samples=int(comp.mc_num_samples),
            max_bounces=int(comp.mc_max_bounces),
            max_diffraction_order=order,
            synthetic_array=bool(comp.mc_synthetic_array),
            edge_policy=ConfigBuilder.edge_policy(ch, comp),
            filtering=ConfigBuilder._filter(ch, comp),
            tuning=tuning,
            integrator_options=options,
        )

    @staticmethod
    def _filter(ch: Any, comp: Any) -> Optional[Any]:
        # One power-domain filter applied to both reflection + diffraction, when enabled.
        if not bool(comp.mc_use_filtering):
            return None
        component = ch.montecarlo.ComponentFilterConfig(
            method=str(comp.mc_filter_method),
            radius=int(comp.mc_filter_radius),
            sigma=num(comp.mc_filter_sigma),
            blend=num(comp.mc_filter_blend),
        )
        return ch.montecarlo.FilterConfig(reflection=component, diffraction=component)


class SolveRunner:
    """Rebuild the platform scene from the studio scene and run a radiomap solve."""

    @staticmethod
    def run(studio_scene: Any, comp: Any, *, transmitter: str, receiver: str) -> Any:
        """Run the chosen solver family for the given Tx/grid; returns a ``RadioMapResult``."""
        import witwin.channel as ch

        platform = SolveRunner._build_scene(studio_scene, comp)
        if str(comp.solver) == "montecarlo":
            config = ConfigBuilder.montecarlo(ch, comp)
            return ch.montecarlo.solve(scene=platform, transmitter=transmitter,
                                       receiver=receiver, config=config)
        config = ConfigBuilder.deterministic(ch, comp)
        return ch.deterministic.solve(scene=platform, transmitter=transmitter,
                                      receiver=receiver, config=config)

    @staticmethod
    def _build_scene(studio_scene: Any, comp: Any) -> Any:
        # Rebuild the platform scene using the component's own scene config.
        frequency = num(comp.frequency)
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        return ChannelAdapter().build(
            studio_scene, frequency=frequency if frequency > 0.0 else None,
            device=str(comp.device), noise_power=noise)


class RadioMapExtractor:
    """Extract a 2D map (data + extent) from a ``RadioMapResult``."""

    @staticmethod
    def extract(result: Any, *, metric: str, tx_index: int, bounds: List[float],
                log_scale: bool, vmin: Optional[float] = None,
                vmax: Optional[float] = None) -> MapImage:
        """Squeeze to one Tx, read ``metric`` as a (ny, nx) array, build the extent.

        ``vmin`` / ``vmax`` (in the displayed units — dB when ``log_scale``) clamp the
        value window so the heatmap re-scales to exactly that range, without re-solving.
        """
        single = result.squeeze_tx(int(tx_index))
        data = RadioMapExtractor._to_numpy(getattr(single, metric))
        label = metric
        if log_scale:
            data = 10.0 * np.log10(np.abs(data) + 1e-30)
            label = f"{metric} (dB)"
        if vmin is not None or vmax is not None:
            data = np.clip(data, vmin, vmax)
        extent = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
        return MapImage(np.asarray(data, dtype=np.float32), extent, label)

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        # RadioMapResult tensors are drjit FloatTensors; view as torch then numpy.
        import torch
        from witwin.channel.core.numerics.tensors import to_torch_view

        return to_torch_view(tensor, dtype=torch.float32).detach().cpu().numpy()
