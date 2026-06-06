"""Out-of-process Channel solver entry."""
from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_channel  # noqa: F401  (registers Channel components before Scene.from_dict)
from witwin_server.features.solvers.sdk import query, serve
from wt_channel.adapter.path_solve import (
    PathLineBuilder,
    PathSolveRunner,
    RayBatch,
    _COLORMAP_VIRIDIS,
    _gain_to_rgba,
)
from wt_channel.adapter.solve import RadioMapExtractor, SolveRunner


@dataclass
class RadioMapCache:
    metrics: Dict[str, np.ndarray]


@dataclass
class PathCache:
    frequency: float
    coeff: np.ndarray
    tau: np.ndarray
    segments: np.ndarray
    colors_type: np.ndarray
    types: np.ndarray
    path_index: np.ndarray
    path_amplitudes: np.ndarray


def _component(config: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(config.get("component") or {}))


def _clamp_index(value: Any, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(int(value or 0), size - 1))


def solve(ctx, scene, config):
    config = config or {}
    mode = str(config.get("mode") or "radiomap")
    comp = _component(config)
    transmitter = str(config.get("transmitter") or "")
    receiver = str(config.get("receiver") or "")
    if mode == "path":
        ctx.progress(0.0, "building Channel path scene")
        result = PathSolveRunner.run(scene, comp, transmitter=transmitter, receiver=receiver)
        ctx.progress(0.9, "extracting Channel path cache")
        cache = _path_cache(result)
        ctx.progress(1.0, "Channel path solve complete")
        return cache
    ctx.progress(0.0, "building Channel radiomap scene")
    result = SolveRunner.run(scene, comp, transmitter=transmitter, receiver=receiver)
    ctx.progress(0.9, "extracting Channel radiomap cache")
    cache = _radiomap_cache(result)
    ctx.progress(1.0, "Channel radiomap solve complete")
    return cache


def _radiomap_cache(result) -> RadioMapCache:
    metrics = {}
    for name in ("path_gain", "rss", "sinr"):
        if hasattr(result, name):
            metrics[name] = np.asarray(
                RadioMapExtractor._to_numpy(getattr(result, name)),
                dtype=np.float32,
            )
    return RadioMapCache(metrics=metrics)


def _path_cache(result) -> PathCache:
    coeff, tau = result.cir(normalize_delays=True)
    coeff_np = coeff.detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy().astype(np.float32, copy=False)
    batch = PathLineBuilder.build_batch(result)
    path_amplitudes = _path_amplitudes(coeff_np)
    return PathCache(
        frequency=float(result.frequency),
        coeff=coeff_np.astype(np.complex64, copy=False),
        tau=tau_np,
        segments=batch.segments.astype(np.float32, copy=False),
        colors_type=batch.colors_type.astype(np.uint8, copy=False),
        types=batch.types.astype(np.uint8, copy=False),
        path_index=batch.path_index.astype(np.int32, copy=False),
        path_amplitudes=path_amplitudes,
    )


def _path_amplitudes(coeff: np.ndarray, *, tx_index: int = 0, rx_index: int = 0) -> np.ndarray:
    tx = _clamp_index(tx_index, coeff.shape[2])
    rx = _clamp_index(rx_index, coeff.shape[0])
    return np.abs(coeff[rx, 0, tx, 0, :, 0]).astype(np.float32, copy=False)


@query("extract_map")
def extract_map(ctx, result: RadioMapCache, params):
    metric = str(params.get("metric") or "path_gain")
    if metric not in result.metrics:
        raise ValueError(f"unknown radiomap metric: {metric}")
    data = result.metrics[metric]
    if data.ndim == 3:
        data = data[_clamp_index(params.get("tx_index"), data.shape[0])]
    else:
        data = np.asarray(data)
    label = metric
    if bool(params.get("log_scale", True)):
        data = 10.0 * np.log10(np.abs(data) + 1e-30)
        label = f"{metric} (dB)"
    vmin = params.get("vmin")
    vmax = params.get("vmax")
    if vmin is not None or vmax is not None:
        data = np.clip(
            data,
            None if vmin is None else float(vmin),
            None if vmax is None else float(vmax),
        )
    return {
        "data": np.asarray(data, dtype=np.float32),
        "extent": list(params.get("bounds") or [0.0, 1.0, 0.0, 1.0]),
        "label": label,
    }


@query("plot")
def plot(ctx, result: PathCache, params):
    mode = str(params.get("mode") or "cir")
    if mode == "cfr":
        data = _cfr_plot(result, params)
    elif mode == "taps":
        data = _taps_plot(result, params)
    else:
        data = _cir_plot(result, params)
    return data


def _path_link(result: PathCache, params):
    tx = _clamp_index(params.get("tx_index"), result.coeff.shape[2])
    rx = _clamp_index(params.get("rx_index"), result.coeff.shape[0])
    coeff = result.coeff[rx, 0, tx, 0]
    tau = result.tau[rx, 0, tx, 0]
    valid = tau >= 0.0
    return coeff, tau, valid


def _cir_plot(result: PathCache, params):
    coeff, tau, valid = _path_link(result, params)
    amp = np.abs(coeff[:, 0])
    return {
        "x": (tau[valid] * 1e9).astype(np.float32, copy=False),
        "y": amp[valid].astype(np.float32, copy=False),
        "kind": "stem",
        "xlabel": "delay (ns)",
        "ylabel": "|a|",
        "title": "Channel Impulse Response",
    }


def _cfr_plot(result: PathCache, params):
    coeff, tau, valid = _path_link(result, params)
    num_points = int(params.get("num_points") or 128)
    span_hz = float(params.get("span_hz") or 100e6)
    freqs = np.linspace(result.frequency - span_hz / 2.0,
                        result.frequency + span_hz / 2.0,
                        num_points, dtype=np.float32)
    if valid.any():
        phase = -2.0j * math.pi * tau[valid, None].astype(np.float32) * freqs[None, :]
        h = (coeff[valid, 0].astype(np.complex64)[:, None] * np.exp(phase)).sum(
            axis=0,
            dtype=np.complex64,
        )
        mag = np.abs(h)
    else:
        mag = np.zeros((num_points,), dtype=np.float32)
    return {
        "x": (freqs / 1e9).astype(np.float32),
        "y": np.asarray(mag, dtype=np.float32),
        "kind": "line",
        "xlabel": "frequency (GHz)",
        "ylabel": "|H(f)|",
        "title": "Channel Frequency Response",
    }


def _taps_plot(result: PathCache, params):
    coeff, tau, valid = _path_link(result, params)
    bandwidth = float(params.get("bandwidth") or 100e6)
    num_taps = int(params.get("num_taps") or 64)
    if bandwidth <= 0.0 or num_taps <= 0:
        raise ValueError("bandwidth and num_taps must be positive")
    tap_idx = np.rint(tau * bandwidth).astype(np.int64)
    keep = valid & (tap_idx >= 0) & (tap_idx < num_taps)
    taps = np.zeros((num_taps,), dtype=np.complex64)
    if keep.any():
        np.add.at(taps, tap_idx[keep], coeff[keep, 0])
    return {
        "x": np.arange(num_taps, dtype=np.float32),
        "y": np.abs(taps).astype(np.float32),
        "kind": "stem",
        "xlabel": "tap index",
        "ylabel": "|tap|",
        "title": f"Taps @ {float(bandwidth) / 1e6:.0f} MHz",
    }


@query("ray_batch")
def ray_batch(ctx, result: PathCache, params):
    batch = _ray_batch_from_cache(result, params)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        segments=batch.segments.astype(np.float32, copy=False),
        colors_type=batch.colors_type.astype(np.uint8, copy=False),
        colors_gain=batch.colors_gain.astype(np.uint8, copy=False),
        types=batch.types.astype(np.uint8, copy=False),
        path_index=batch.path_index.astype(np.int32, copy=False),
    )
    return {
        "resultRef": ctx.result_ref(
            buf.getvalue(),
            kind="npz",
            extension=".npz",
            metadata={"format": "numpy.npz"},
        ),
        "segmentCount": batch.segment_count,
        "empty": batch.empty,
    }


def _ray_batch_from_cache(result: PathCache, params) -> RayBatch:
    gain_range = params.get("gain_db_range") or [-120.0, -40.0]
    db_min, db_max = float(gain_range[0]), float(gain_range[1])
    path_index = result.path_index.astype(np.int64, copy=False)
    if path_index.size:
        segment_amps = result.path_amplitudes[path_index]
        colors_gain = _gain_to_rgba(segment_amps, db_min, db_max, _COLORMAP_VIRIDIS)
    else:
        colors_gain = np.zeros((0, 4), dtype=np.uint8)
    return RayBatch(
        segments=result.segments,
        colors_type=result.colors_type,
        colors_gain=colors_gain,
        types=result.types,
        path_index=result.path_index,
    )


if __name__ == "__main__":
    serve(solve, threaded_solve=False)
