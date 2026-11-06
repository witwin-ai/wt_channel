"""Export function of the dataset Study ``studies/link-gain.toml``.

One sample is the radio channel from ``Tx`` to ``Rx`` with the receiver at
the Study's sampled ``rx_x`` and ``rx_z``: the total path gain in dB (the
label), the RMS delay spread and the number of paths.

The Channel Plugin's Path Solver runs inside Studio. This script calls the
same library (``witwin.channel`` 0.5 path solver, CUDA only) directly, so a
dataset Run needs no editor. It follows the Plugin's Solve Paths: the
buildings and the ground as boxes with their ITU material at the Path
Solver's frequency, the Tx and Rx as single isotropic antennas, and the
Path Solver's bounce and diffraction settings.

``read_setup`` and ``sample`` need only numpy; ``simulate`` is the GPU
part. ``tests/standin_export.py`` swaps ``simulate`` for free-space loss so
the tests run without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ITU-R P.2040 presets below 100 GHz: eps_r = a * f^b, sigma = c * f^d (S/m), f in GHz.
# The same coefficients as the Channel Plugin (wt_channel adapter/itu.py).
ITU = {
    "concrete": (5.24, 0.0, 0.0462, 0.7822),
    "brick": (3.91, 0.0, 0.0238, 0.16),
    "plasterboard": (2.73, 0.0, 0.0085, 0.9395),
    "wood": (1.99, 0.0, 0.0047, 1.0718),
    "glass": (6.31, 0.0, 0.0036, 1.3394),
    "chipboard": (2.58, 0.0, 0.0217, 0.7800),
    "marble": (7.074, 0.0, 0.0055, 0.9262),
    "metal": (1.0, 0.0, 1.0e7, 0.0),
}
# The Studio Rx's polarization when its component does not set one (wt_channel STUDIO_VERTICAL).
VERTICAL = (0.0, 1.0, 0.0)
C = 299_792_458.0


@dataclass(frozen=True)
class Box:
    centre: np.ndarray            # world metres
    size: np.ndarray              # full extents along x, y, z, metres
    eps_r: float
    sigma: float                  # S/m
    name: str

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(np.abs(point - self.centre) < self.size / 2.0))


@dataclass(frozen=True)
class Setup:
    """What one sample solves."""
    frequency_hz: float
    max_bounces: int
    diffraction: bool
    boxes: tuple[Box, ...]
    tx: np.ndarray
    tx_polarization: tuple[float, float, float]
    tx_power_w: float
    rx: np.ndarray
    rx_polarization: tuple[float, float, float]


def export(ctx) -> dict[str, Any] | None:
    """One sample: ``position`` (Rx x, z), ``gain_db``, ``delay_spread_ns``, ``num_paths``."""
    setup = read_setup(ctx)
    if any(box.contains(setup.rx) for box in setup.boxes):
        return None  # the receiver is inside a building: drop the sample
    power, delay = simulate(setup)
    return sample(setup, power, delay)


def read_setup(ctx) -> Setup:
    """The world of this sample, read from ``ctx.scene`` and the sampled values.

    ``ctx.scene`` holds the saved Scene; a sampled point changes only the
    fields bound to formulas, which ``ctx.values.bindings`` holds. So every
    position goes through ``_position``. Boxes must not be rotated.
    """
    scene = ctx.scene
    _, solver = _entity_with(scene, "PathSolver")
    frequency = float(solver["frequency"])
    tx_entity, tx = _entity_with(scene, "ChannelTx")
    rx_entity, rx = _entity_with(scene, "ChannelRx")
    boxes = []
    for entity in scene.objects.values():
        geometry = entity.components["PlatformGeometry"] if "PlatformGeometry" in entity.components else None
        if geometry is None or geometry.get("kind") != "box":
            continue
        if any(abs(float(a) - b) > 1e-9 for a, b in zip(entity.transform.rotation, (0.0, 0.0, 0.0, 1.0))):
            raise ValueError(f"{entity.name}: rotated boxes are not supported by this script")
        eps_r, sigma = _material(entity, frequency)
        size = np.asarray(geometry.get("size") or entity.transform.scale, dtype=float)
        boxes.append(Box(_position(ctx, entity), size, eps_r, sigma, entity.name))
    return Setup(
        frequency_hz=frequency,
        max_bounces=int(solver["max_bounces"]),
        diffraction=int(solver["max_diffraction_order"]) > 0,
        boxes=tuple(boxes),
        tx=_position(ctx, tx_entity),
        tx_polarization=tuple(float(v) for v in tx["polarization"]),
        tx_power_w=float(tx["power"]),
        rx=_position(ctx, rx_entity),
        rx_polarization=tuple(float(v) for v in rx["polarization"]) if rx.get("use_polarization") else VERTICAL,
    )


def simulate(setup: Setup) -> tuple[np.ndarray, np.ndarray]:
    """Power gain (linear) and delay (s) of each path from Tx to Rx."""
    import torch
    import witwin.core as core
    from witwin.channel import path

    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel solves paths on a CUDA GPU; this machine has none")
    structures = tuple(
        core.Structure(core.Box(position=tuple(b.centre), size=tuple(b.size)),
                       core.PhysicalMaterial(eps_r=b.eps_r, sigma_e=b.sigma), name=b.name)
        for b in setup.boxes)
    tx = core.AntennaState(core.new_antenna_id(), "tx", tuple(setup.tx),
                           polarization=setup.tx_polarization, power_w=setup.tx_power_w)
    rx = core.AntennaState(core.new_antenna_id(), "rx", tuple(setup.rx), polarization=setup.rx_polarization)
    components = {"los"} | ({"reflection"} if setup.max_bounces > 0 else set())
    components |= {"diffraction"} if setup.diffraction else set()
    config = path.Config(max_depth=max(setup.max_bounces, int(setup.diffraction)), components=frozenset(components))
    result = path.solve(core.Scene(structures=structures, endpoints=(tx, rx)), config,
                        reference_frequency_hz=setup.frequency_hz)
    # a: (rx, rx_ant, tx, tx_ant, path, time); tau and valid: (rx, rx_ant, tx, tx_ant, path).
    # A shadowed Rx has no paths: zero-size tensors, so sample() drops it.
    valid = result.valid.reshape(-1).bool()
    power = result.a.abs().square().sum(dim=-1).reshape(-1)
    return (power[valid].double().cpu().numpy(), result.tau.reshape(-1)[valid].double().cpu().numpy())


def sample(setup: Setup, power: np.ndarray, delay: np.ndarray) -> dict[str, Any] | None:
    """The dataset sample of one solved link; None (dropped) when no path reaches the Rx."""
    power, delay = np.asarray(power, dtype=float), np.asarray(delay, dtype=float)
    total = float(power.sum())
    if not total > 0.0:
        return None
    mean_delay = float(np.sum(power * delay) / total)
    spread = float(np.sqrt(max(np.sum(power * (delay - mean_delay) ** 2) / total, 0.0)))
    return {
        "position": np.array([setup.rx[0], setup.rx[2]], dtype=np.float32),
        "gain_db": 10.0 * np.log10(total),
        "delay_spread_ns": spread * 1e9,
        "num_paths": int(power.size),
    }


# (eps_r, sigma) of an entity: its ITU preset at the frequency, else the Material's scalars.
def _material(entity, frequency_hz: float) -> tuple[float, float]:
    components = entity.components
    preset = components["MaterialPreset"] if "MaterialPreset" in components else None
    if preset is not None:
        name = str(preset.get("preset"))
        if name not in ITU:
            raise ValueError(f"{entity.name}: ITU preset {name!r} is not in this script's table {sorted(ITU)}")
        f_ghz = float(preset["frequency_override"] if preset.get("use_frequency_override") else frequency_hz) / 1e9
        a, b, c, d = ITU[name]
        return a * f_ghz ** b, c * f_ghz ** d
    material = components["Material"]
    return float(material["eps_r"][0]), float(material.get("sigma_e") or 0.0)


def _entity_with(scene, type_name: str):
    for entity in scene.objects.values():
        if type_name in entity.components:
            return entity, entity.components[type_name]
    raise RuntimeError(f"the Scene has no {type_name}")


# A position with its bound components replaced by the sample's values.
def _position(ctx, entity) -> np.ndarray:
    position = np.asarray(entity.transform.position, dtype=float)
    for key, value in ctx.values.bindings.items():
        if key.entity_id == entity.id and key.target == "Transform.position" and key.index is not None:
            position[key.index] = float(value)
    return position
