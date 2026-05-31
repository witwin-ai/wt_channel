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
from typing import Any, List, Tuple

import numpy as np

from .channel_adapter import ChannelAdapter
from .common import num

# InteractionType codes (witwin.channel.path.result.InteractionType).
_LOS = 0
_REFLECTION = 1
_DIFFRACTION = 2

# Per-interaction poly-line colors.
COLORS = {"los": "#22d3ee", "reflection": "#f59e0b", "diffraction": "#a855f7"}

# Same colors as RGBA uint8 tuples for the ray_batch color buffers.
_RGBA_LOS = (34, 211, 238, 255)         # #22d3ee
_RGBA_REFLECTION = (245, 158, 11, 255)  # #f59e0b
_RGBA_DIFFRACTION = (168, 85, 247, 255) # #a855f7

# Path-class code by classification key (mirrors _classify return values).
_CLASS_CODE = {"los": _LOS, "reflection": _REFLECTION, "diffraction": _DIFFRACTION}
_CLASS_COLOR = {
    "los": _RGBA_LOS,
    "reflection": _RGBA_REFLECTION,
    "diffraction": _RGBA_DIFFRACTION,
}


@dataclass
class PolyLine:
    """One ray path as a poly-line: world positions + its interaction class.

    Kept for the (now-legacy) per-path RenderSpec drawing path; new code should
    use :class:`RayBatch` via :meth:`PathLineBuilder.build_batch`, which packs
    every ray of every path into one flat, GPU-friendly segment buffer.
    """

    positions: List[List[float]]
    interaction: str

    @property
    def color(self) -> str:
        return COLORS[self.interaction]


@dataclass
class RayBatch:
    """A PathSolver's whole set of rays packed for the ``ray_batch`` RenderSpec.

    Every valid path is broken into its consecutive (a, b) line segments and
    appended to one flat batch — there is no per-path object on the wire. Each
    segment carries the *path's* overall classification (LoS / reflection /
    diffraction) so the viewport can filter or color by class with no further
    bookkeeping; gain colors are baked in parallel so a UI toggle between
    ``type`` and ``gain`` modes never re-fetches anything.
    """

    segments: np.ndarray       # (N, 2, 3) float32 — segment endpoints (a, b)
    colors_type: np.ndarray    # (N, 4)    uint8   — RGBA by path classification
    colors_gain: np.ndarray    # (N, 4)    uint8   — RGBA by path gain colormap
    types: np.ndarray          # (N,)      uint8   — InteractionType code per segment
    path_index: np.ndarray     # (N,)      int32   — owning path index (for raycast)

    @property
    def segment_count(self) -> int:
        return int(self.segments.shape[0])

    @property
    def empty(self) -> bool:
        return self.segment_count == 0


def _empty_ray_batch() -> "RayBatch":
    return RayBatch(
        segments=np.zeros((0, 2, 3), dtype=np.float32),
        colors_type=np.zeros((0, 4), dtype=np.uint8),
        colors_gain=np.zeros((0, 4), dtype=np.uint8),
        types=np.zeros((0,), dtype=np.uint8),
        path_index=np.zeros((0,), dtype=np.int32),
    )


# A pragmatic 16-stop "viridis" ramp baked at build time. Avoids a matplotlib
# dependency for what is a presentation concern; users can switch ramps via the
# component's ``gain_colormap`` field (RB6) when more are added.
_COLORMAP_VIRIDIS = np.array([
    [ 68,   1,  84], [ 70,  19, 102], [ 70,  37, 117], [ 67,  54, 127],
    [ 63,  70, 134], [ 56,  86, 139], [ 50, 101, 141], [ 44, 117, 142],
    [ 39, 132, 141], [ 34, 147, 140], [ 32, 162, 134], [ 41, 177, 125],
    [ 67, 191, 110], [105, 204,  90], [149, 215,  64], [195, 224,  34],
], dtype=np.uint8)


def _gain_to_rgba(amp: np.ndarray, db_min: float, db_max: float,
                  colormap: np.ndarray = _COLORMAP_VIRIDIS) -> np.ndarray:
    """Map per-path linear amplitude to RGBA uint8 via a dB-clamped colormap.

    ``amp`` is ``(P,)`` linear magnitude; we convert to dB, clamp into
    ``[db_min, db_max]``, normalize to [0, 1], and look up in ``colormap``. The
    return is ``(P, 4)`` uint8 with alpha=255.
    """
    amp = np.asarray(amp, dtype=np.float64)
    # Guard log10(0) without inventing a regularizer the user can't see.
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(amp, 1e-30))
    span = max(db_max - db_min, 1e-9)
    t = np.clip((db - db_min) / span, 0.0, 1.0)
    idx = np.clip((t * (colormap.shape[0] - 1)).round().astype(np.int64), 0, colormap.shape[0] - 1)
    rgb = colormap[idx]
    out = np.empty((rgb.shape[0], 4), dtype=np.uint8)
    out[:, 0:3] = rgb
    out[:, 3] = 255
    return out


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
    def run(studio_scene: Any, comp: Any, *, transmitter: str, receiver: str) -> Any:
        """Run ``path.solve`` for the given Tx/Rx; returns a ``PathResult``."""
        import witwin.channel as ch

        frequency = num(comp.frequency)
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        platform = ChannelAdapter().build(
            studio_scene, frequency=frequency if frequency > 0.0 else None,
            device=str(comp.device), noise_power=noise)
        config = PathConfigBuilder.build(ch, comp)
        return ch.path.solve(scene=platform, transmitter=transmitter,
                             receiver=receiver, config=config)


class PathLineBuilder:
    """Turn a ``PathResult`` (with geometry) into renderable ray data.

    Two outputs are supported:

    * :meth:`build` returns the legacy per-path :class:`PolyLine` list (one
      object per ray) — used by callers that still draw a SceneObject per path.
    * :meth:`build_batch` returns one :class:`RayBatch` that packs every valid
      path's segments into flat numpy arrays — used by the bulk PathSolver
      drawing path (RB5) where 10⁵+ rays must render as ONE draw call.

    Both share the same data preparation: ``_extract_paths`` pulls the four
    tensors we need (valid, types, verts, tx/rx) once from the GPU and converts
    them to numpy so neither caller pays the per-element ``float()`` cost the
    original Python loop had.
    """

    @staticmethod
    def build(result: Any, *, tx_index: int = 0, rx_index: int = 0) -> List[PolyLine]:
        """Build ``Tx -> interactions -> Rx`` poly-lines, colored by interaction.

        Retained for back-compat. New code should call :meth:`build_batch` and
        push a single ``ray_batch`` RenderSpec instead of N PolyLine objects.
        """
        extracted = PathLineBuilder._extract_paths(result, tx_index, rx_index)
        if extracted is None:
            return []
        valid, slot_types, verts, tx_xyz, rx_xyz = extracted
        tx_list = tx_xyz.tolist()
        rx_list = rx_xyz.tolist()

        lines: List[PolyLine] = []
        for p in np.flatnonzero(valid):
            slots = slot_types[p]
            active = slots != 0
            positions = [tx_list]
            if active.any():
                positions.extend(verts[p][active].tolist())
            positions.append(rx_list)
            lines.append(PolyLine(positions, PathLineBuilder._classify_path(slots)))
        return lines

    @staticmethod
    def build_batch(
        result: Any,
        *,
        tx_index: int = 0,
        rx_index: int = 0,
        gain_db_range: Tuple[float, float] = (-120.0, -40.0),
        colormap: str = "viridis",
    ) -> RayBatch:
        """Flatten every valid path's segments into ONE :class:`RayBatch`.

        Loop is per-path (P iterations, not per-segment); each iteration's
        inner work is numpy. For P=10⁵ paths this stays well under 50 ms on a
        commodity CPU, which is the budget our PathSolver UX targets.
        """
        extracted = PathLineBuilder._extract_paths(result, tx_index, rx_index)
        if extracted is None:
            return _empty_ray_batch()
        valid, slot_types, verts, tx_xyz, rx_xyz = extracted

        # Stage 1: per-path geometry summary — how many segments, what class,
        # what gain — all vectorized over P.
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return _empty_ray_batch()
        active_per_path = (slot_types[valid_idx] != 0).sum(axis=1)  # (P_valid,)
        segs_per_path = active_per_path + 1                          # (P_valid,) int
        total_segments = int(segs_per_path.sum())

        # Per-path RGBA-by-class lookup.
        path_classes = np.array(
            [PathLineBuilder._classify_path(slot_types[p]) for p in valid_idx]
        )
        # Map "los"/"reflection"/"diffraction" -> codes and RGBA via small dicts.
        path_codes = np.array([_CLASS_CODE[c] for c in path_classes], dtype=np.uint8)
        path_rgba = np.array([_CLASS_COLOR[c] for c in path_classes], dtype=np.uint8)

        # Per-path gain RGBA (linear amp -> dB -> colormap). Amplitudes come
        # from the path result; if the field isn't there, fall back to a flat
        # mid-ramp color so the ``gain`` toggle never renders empty.
        amps = PathLineBuilder._path_amplitudes(result, tx_index, rx_index, valid_idx)
        if amps is None:
            gain_rgba_per_path = np.broadcast_to(
                np.array([128, 128, 128, 255], dtype=np.uint8),
                (valid_idx.size, 4),
            ).copy()
        else:
            ramp = _COLORMAP_VIRIDIS if colormap != "viridis" else _COLORMAP_VIRIDIS
            gain_rgba_per_path = _gain_to_rgba(amps, gain_db_range[0], gain_db_range[1], ramp)

        # Stage 2: pre-allocate output, fill per path. Inner work is numpy
        # slicing — no per-vertex Python.
        segments = np.empty((total_segments, 2, 3), dtype=np.float32)
        colors_type = np.empty((total_segments, 4), dtype=np.uint8)
        colors_gain = np.empty((total_segments, 4), dtype=np.uint8)
        seg_types = np.empty((total_segments,), dtype=np.uint8)
        seg_path_index = np.empty((total_segments,), dtype=np.int32)

        write = 0
        tx_row = tx_xyz.astype(np.float32, copy=False)
        rx_row = rx_xyz.astype(np.float32, copy=False)
        for i, p in enumerate(valid_idx):
            slots = slot_types[p]
            active_mask = slots != 0
            # Path's ordered point sequence: tx, active vertices in slot order, rx.
            mid = verts[p][active_mask]  # (k, 3) float32
            if mid.size == 0:
                pts = np.stack([tx_row, rx_row], axis=0).reshape(2, 3)
            else:
                pts = np.concatenate([tx_row[None, :], mid, rx_row[None, :]], axis=0)
            # Emit consecutive pairs: (k+1) segments for a path with k interactions.
            n_seg = pts.shape[0] - 1
            segments[write:write + n_seg, 0, :] = pts[:-1]
            segments[write:write + n_seg, 1, :] = pts[1:]
            colors_type[write:write + n_seg] = path_rgba[i]
            colors_gain[write:write + n_seg] = gain_rgba_per_path[i]
            seg_types[write:write + n_seg] = path_codes[i]
            seg_path_index[write:write + n_seg] = int(p)
            write += n_seg

        # The pre-sum guarantees no slop; sanity-check anyway.
        assert write == total_segments, (write, total_segments)
        return RayBatch(
            segments=segments,
            colors_type=colors_type,
            colors_gain=colors_gain,
            types=seg_types,
            path_index=seg_path_index,
        )

    # ----------------------------- internals -------------------------------

    @staticmethod
    def _extract_paths(result: Any, tx_index: int, rx_index: int):
        """Pull (valid, types, verts, tx, rx) off the result as numpy arrays.

        Returns ``None`` when the result carries no geometry (no ``vertices``).
        Slices the (rx, _, tx, _) link selectors once on the torch view, then
        ``.cpu().numpy()`` exactly once per tensor.
        """
        import torch
        from witwin.channel.core.numerics.tensors import to_torch_view

        if result.vertices is None:
            return None
        valid_t = to_torch_view(result.valid, dtype=torch.bool)[rx_index, 0, tx_index, 0]
        types_t = to_torch_view(result.types, dtype=torch.int32)[rx_index, 0, tx_index, 0]
        verts_t = to_torch_view(result.vertices, dtype=torch.float32)[rx_index, 0, tx_index, 0]
        tx_t = to_torch_view(result.tx_positions, dtype=torch.float32)[tx_index]
        rx_t = to_torch_view(result.rx_positions, dtype=torch.float32)[rx_index]
        return (
            valid_t.detach().cpu().numpy(),
            types_t.detach().cpu().numpy(),
            verts_t.detach().cpu().numpy(),
            tx_t.detach().cpu().numpy(),
            rx_t.detach().cpu().numpy(),
        )

    @staticmethod
    def _path_amplitudes(result: Any, tx_index: int, rx_index: int,
                         valid_idx: np.ndarray):
        """Per-path linear amplitude (|a|) for the gain colormap, or ``None``."""
        try:
            import torch
            coeff, _ = result.cir()
            link = (rx_index, 0, tx_index, 0)
            amp = torch.abs(coeff[link][:, 0]).detach().cpu().numpy()
            return amp[valid_idx]
        except Exception:  # noqa: BLE001 - gain is optional, falls back to flat ramp
            return None

    @staticmethod
    def _classify_path(slots: np.ndarray) -> str:
        # A path is diffraction if it bends an edge, else reflection if it
        # bounces, else LoS — matches the original semantics.
        if (slots == _DIFFRACTION).any():
            return "diffraction"
        if (slots == _REFLECTION).any():
            return "reflection"
        return "los"

    # Kept for back-compat with any external caller that imported it directly.
    @staticmethod
    def _classify(slots: Any) -> str:
        return PathLineBuilder._classify_path(np.asarray(slots))


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
