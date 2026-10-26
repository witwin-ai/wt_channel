"""Out-of-process Channel solver entry."""
from __future__ import annotations

import io
import math
import sys
from collections import OrderedDict
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
from wt_channel.adapter.solve import SolveRunner


# Scenes this process received, by the caller's scene_key (newest last). Dragging the
# Tx re-solves the same geometry many times; shipping the Scene document each time
# costs more than a Monte Carlo preview solve, so a preview sends only the key and
# the Tx position.
_SCENES: "OrderedDict[str, Any]" = OrderedDict()
_MAX_SCENES = 2
_HEAVY_GRID_CELLS = 128 * 128
# The scene_key of the last radio map solve; a different one means another map.
_LAST_SCENE_KEY = None


def _scene_for(scene, config: Dict[str, Any]):
    key = str(config.get("scene_key") or "")
    if scene is not None:
        if key:
            _SCENES[key] = scene
            _SCENES.move_to_end(key)
            while len(_SCENES) > _MAX_SCENES:
                _SCENES.popitem(last=False)
    elif key in _SCENES:
        scene = _SCENES[key]
        _SCENES.move_to_end(key)
    else:
        raise LookupError(f"scene_cache_miss: no cached scene for key {key!r}")
    tx_position = config.get("tx_position")
    if tx_position is not None:
        name = str(config.get("transmitter") or "")
        tx = next((o for o in scene.objects.values() if o.name == name), None)
        if tx is None:
            raise ValueError(f"transmitter {name!r} is not in the cached scene")
        tx.get_component("Transform").position = [float(v) for v in tx_position]
    return scene


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
    scene = _scene_for(scene, config)
    global _LAST_SCENE_KEY
    key = str(config.get("scene_key") or "")
    if key and _LAST_SCENE_KEY and key != _LAST_SCENE_KEY:
        _forget_compiled_scenes()
    _LAST_SCENE_KEY = key or _LAST_SCENE_KEY
    grid_shape = config.get("grid_shape")
    grid_shape = None if grid_shape is None else tuple(int(v) for v in grid_shape)
    # Coarse live previews stay small; only solves that can crowd the GPU release it.
    heavy = str(getattr(comp, "solver", "")) == "deterministic" and (
        grid_shape is None or grid_shape[0] * grid_shape[1] >= _HEAVY_GRID_CELLS)
    if heavy:
        _release_torch_cache()
    failure = None
    try:
        result = _run_with_memory_retry(scene, comp, transmitter, receiver, grid_shape)
        _keep_newest_compiled_scene()
        ctx.progress(0.9, "extracting Channel radiomap cache")
        cache = _radiomap_cache(result)
        result = None
    except Exception as exc:  # noqa: BLE001 - re-raised below without its frames
        failure = (type(exc), str(exc))
    if failure is not None:
        # Outside the except block, so the failed attempt's frames - and the GPU tensors
        # they hold - are gone: Princeton ran out of memory and the solver host kept all
        # 16 GB, so every later solve (other maps, Monte Carlo retries) failed too.
        _forget_compiled_scenes()
        if "cuda error" in failure[1].lower():
            _exit_after_reply()
        raise _plain_error(*failure)
    if heavy:
        _release_torch_cache()
    ctx.progress(1.0, "Channel radiomap solve complete")
    return cache


def _exit_after_reply() -> None:
    """Exit right after the error reply goes out, so Studio starts a fresh host.

    A "CUDA error" (out of memory inside a kernel, illegal access) breaks the process's
    CUDA context: after one on Princeton every later solve in the same host failed the
    same way, even 64^2 previews of other maps.
    """
    import os
    import threading

    threading.Timer(0.5, os._exit, args=(3,)).start()


def _plain_error(kind, message: str) -> BaseException:
    """The same kind of error with only its message (no traceback, no chained cause)."""
    try:
        return kind(message)
    except Exception:  # noqa: BLE001 - an error type with another constructor
        return RuntimeError(message)


def _run_with_memory_retry(scene, comp, transmitter, receiver, grid_shape):
    """Solve; on a GPU out-of-memory error forget the compiled scenes and retry once."""
    kwargs = {} if grid_shape is None else {"grid_shape": grid_shape}
    try:
        return SolveRunner.run(scene, comp, transmitter=transmitter, receiver=receiver, **kwargs)
    except Exception as exc:  # noqa: BLE001 - only out-of-memory is retried
        if not _out_of_memory(exc):
            raise
    # Outside the except block, so the failed attempt's frames (and tensors) are gone.
    _forget_compiled_scenes()
    return SolveRunner.run(scene, comp, transmitter=transmitter, receiver=receiver, **kwargs)


def _out_of_memory(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "out_of_memory" in text or "memoryallocation" in text


def _forget_compiled_scenes() -> None:
    """Drop witwin-channel's compiled scenes and hand their GPU memory back.

    witwin-channel keeps up to 32 compiled scenes (RayD acceleration structures and
    edge caches on the GPU) in a process-wide LRU. Across ten maps in one Studio the
    solver host grew to the whole 16 GB GPU and later maps ran out of memory.
    """
    import gc

    from witwin.channel import scene as channel_scene

    channel_scene.clear_compile_cache()
    gc.collect()
    _release_torch_cache()


def _keep_newest_compiled_scene() -> None:
    """Drop every compiled scene but the newest from witwin-channel's LRU.

    Each live preview builds a new Core world (the Tx moved) and witwin-channel kept
    every compiled one: on MIT the allocated GPU memory grew 1.46 GB per preview
    (1.5 -> 8.8 GB over a single drag). The newest stays, so the next preview still
    reuses its RayD acceleration structure (previews stay ~0.5 s).
    """
    from witwin.channel.scene import compiler

    registry = getattr(compiler, "_REGISTRY", None)
    compiled = getattr(registry, "_compiled", None)
    if compiled is None:  # a witwin-channel without this LRU: nothing to trim
        return
    with registry._lock:
        while len(compiled) > 1:
            compiled.popitem(last=False)


def _release_torch_cache() -> None:
    """Hand torch's cached GPU blocks back to the driver around a deterministic solve.

    A deterministic radio map can peak near the whole GPU (UCSD: ~14 GB of 16 GB) and
    RayD/OptiX allocate outside torch, so blocks torch keeps cached from the previous
    solve or from Monte Carlo previews push it past physical memory (10 s -> 90-160 s,
    then the solver host is killed as unresponsive).
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _radiomap_cache(maps) -> RadioMapCache:
    metrics = {name: np.asarray(data, dtype=np.float32) for name, data in maps.metrics().items()}
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
