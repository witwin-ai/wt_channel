"""A GPU-free stand-in for ``scripts/synthesize.py``'s export, used by the tests.

It reads the same setup from the Scene and returns the same sample, but the
channel comes from a formula instead of ``witwin.channel``: one direct path
with free-space gain (lambda / 4 pi d)^2, 20 dB weaker when a building
blocks the straight line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthesize  # noqa: E402


def blocked(setup: synthesize.Setup) -> bool:
    """True when the Tx-Rx segment passes through a building (any box but the ground)."""
    steps = np.linspace(0.0, 1.0, 200)[:, None]
    points = setup.tx + steps * (setup.rx - setup.tx)
    return any(box.contains(p) for box in setup.boxes if box.size[1] > 1.0 for p in points)


def export(ctx):
    setup = synthesize.read_setup(ctx)
    if any(box.contains(setup.rx) for box in setup.boxes):
        return None
    distance = float(np.linalg.norm(setup.rx - setup.tx))
    wavelength = synthesize.C / setup.frequency_hz
    gain = (wavelength / (4.0 * np.pi * distance)) ** 2 * (0.01 if blocked(setup) else 1.0)
    return synthesize.sample(setup, np.array([gain]), np.array([distance / synthesize.C]))
