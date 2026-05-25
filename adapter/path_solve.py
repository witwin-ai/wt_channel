"""Point-to-point path solve + CIR/CFR/taps extraction + viewport ray drawing (§6.3 / §7.4).

``path.solve(*, scene, transmitter, receiver, config) -> PathResult`` traces discrete
Tx<->Rx links (CIR / CFR / taps). With ``return_geometry=True`` the result carries
per-path interaction ``vertices``; :class:`PathLineBuilder` turns them into per-path
poly-lines (``Tx -> v0 -> ... -> Rx``) colored by interaction type for a declarative
viewport render spec. ``PathPlots`` reduces a ``PathResult`` to 1D plot data for the
component figure. Live path solve needs CUDA RayD/OptiX (the extractors are CPU-testable
against a synthetic ``PathResult``).
"""
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .channel_adapter import ChannelAdapter
from .common import num

# InteractionType codes (witwin.channel.path.result.InteractionType).
_REFLECTION = 1
_DIFFRACTION = 2

# Per-interaction poly-line colors.
COLORS = {"los": "#22d3ee", "reflection": "#f59e0b", "diffraction": "#a855f7"}


@dataclass
class PolyLine:
    """One ray path as a poly-line: world positions + its interaction class."""

    positions: List[List[float]]
    interaction: str

    @property
    def color(self) -> str:
        return COLORS[self.interaction]


@dataclass
class PlotData:
    """1D plot payload for the path figure (CIR / CFR / taps)."""

    x: np.ndarray
    y: np.ndarray
    kind: str   # "stem" | "line"
    xlabel: str
    ylabel: str
    title: str


class PathConfigBuilder:
    """Read the PathSolve component into a ``path.Config``."""

    @staticmethod
    def build(ch: Any, comp: Any) -> Any:
        max_num_paths = int(comp.max_num_paths) if bool(comp.use_max_num_paths) else None
        return ch.path.Config(
            max_num_paths=max_num_paths,
            max_diffraction_order=int(comp.max_diffraction_order),
            return_geometry=bool(comp.draw_paths),
            num_samples=int(comp.num_samples),
            max_bounces=int(comp.max_bounces),
            synthetic_array=bool(comp.synthetic_array),
        )


class PathSolveRunner:
    """Rebuild the platform scene and run a point-to-point path solve."""

    @staticmethod
    def run(studio_scene: Any, comp: Any) -> Any:
        """Run ``path.solve`` for the named Tx/Rx; returns a ``PathResult``."""
        import witwin.channel as ch

        platform = ChannelAdapter().to_platform(studio_scene)
        config = PathConfigBuilder.build(ch, comp)
        return ch.path.solve(scene=platform, transmitter=str(comp.transmitter),
                             receiver=str(comp.receiver), config=config)


class PathLineBuilder:
    """Turn a ``PathResult`` (with geometry) into per-path poly-lines for the viewport."""

    @staticmethod
    def build(result: Any, *, tx_index: int = 0, rx_index: int = 0) -> List[PolyLine]:
        """Build ``Tx -> interactions -> Rx`` poly-lines, colored by interaction."""
        import torch
        from witwin.channel.core.numerics.tensors import to_torch_view

        if result.vertices is None:
            return []
        valid = to_torch_view(result.valid, dtype=torch.bool)[rx_index, 0, tx_index, 0]
        types = to_torch_view(result.types, dtype=torch.int32)[rx_index, 0, tx_index, 0]
        verts = to_torch_view(result.vertices, dtype=torch.float32)[rx_index, 0, tx_index, 0]
        tx = [float(v) for v in to_torch_view(result.tx_positions, dtype=torch.float32)[tx_index]]
        rx = [float(v) for v in to_torch_view(result.rx_positions, dtype=torch.float32)[rx_index]]

        lines: List[PolyLine] = []
        for p in range(int(valid.shape[0])):
            if not bool(valid[p]):
                continue
            slots = types[p]
            positions = [tx]
            for d in range(int(slots.shape[0])):
                if int(slots[d]) != 0:
                    positions.append([float(c) for c in verts[p, d]])
            positions.append(rx)
            lines.append(PolyLine(positions, PathLineBuilder._classify(slots)))
        return lines

    @staticmethod
    def _classify(slots: Any) -> str:
        # A path is diffraction if it bends an edge, else reflection if it bounces, else LoS.
        codes = {int(v) for v in slots}
        if _DIFFRACTION in codes:
            return "diffraction"
        if _REFLECTION in codes:
            return "reflection"
        return "los"


class PathPlots:
    """Reduce a ``PathResult`` to 1D plot data for the figure."""

    @staticmethod
    def cir(result: Any, *, tx_index: int = 0, rx_index: int = 0) -> PlotData:
        """Channel impulse response: |a| vs delay (stem), valid paths only."""
        import torch

        coeff, tau = result.cir()
        link = (rx_index, 0, tx_index, 0)
        amp = torch.abs(coeff[link][:, 0]).detach().cpu().numpy()
        delay = tau[link].detach().cpu().numpy()
        keep = delay >= 0.0
        return PlotData(delay[keep] * 1e9, amp[keep], "stem", "delay (ns)", "|a|",
                        "Channel Impulse Response")

    @staticmethod
    def cfr(result: Any, *, num_points: int = 128, span_hz: float = 100e6,
            tx_index: int = 0, rx_index: int = 0) -> PlotData:
        """Channel frequency response: |H(f)| vs frequency (line)."""
        import torch

        f0 = float(result.frequency)
        freqs = torch.linspace(f0 - span_hz / 2.0, f0 + span_hz / 2.0, int(num_points))
        h = result.cfr(freqs)[rx_index, 0, tx_index, 0]
        mag = torch.abs(h[0] if h.ndim > 1 else h).detach().cpu().numpy()
        return PlotData(freqs.detach().cpu().numpy() / 1e9, mag, "line", "frequency (GHz)",
                        "|H(f)|", "Channel Frequency Response")

    @staticmethod
    def taps(result: Any, *, bandwidth: float, num_taps: int,
             tx_index: int = 0, rx_index: int = 0) -> PlotData:
        """Discrete-time taps: |tap| vs tap index (stem)."""
        import torch

        t = result.taps(float(bandwidth), int(num_taps))[rx_index, 0, tx_index, 0]
        mag = torch.abs(t[0]).detach().cpu().numpy()
        return PlotData(np.arange(mag.shape[0]), mag, "stem", "tap index", "|tap|",
                        f"Taps @ {num(bandwidth) / 1e6:.0f} MHz")
