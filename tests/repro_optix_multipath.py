"""Standalone solver repro: pure witwin.channel 0.6 + witwin.core, no Studio.

Not a pytest test (no ``test_`` prefix, so it is never collected). Run it directly in
whichever env you want to check, to see whether each solver the Plugin uses runs there:

    <project-or-server-python> tests/repro_optix_multipath.py

``python -m witwin.channel doctor`` reports the GPU, driver and OptiX probe; this script
then runs one small solve per solver three times and prints every failure verbatim. If
all three print OK, the gated live tests (``det_ready`` / ``mc_ready`` / ``path_ready``
in ``conftest.py``) run in that env.
"""
import torch
from witwin.channel import path, radiomap
from witwin.core import AntennaState, Box, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure

FREQUENCY_HZ = 3.5e9


def radiomap_scene():
    ground = Structure(
        Mesh(torch.tensor([[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]]),
             torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False, fill_mode="surface"),
        PhysicalMaterial(eps_r=5.24, sigma_e=0.12, name="concrete"), name="ground")
    return Scene(structures=[ground], endpoints=[
        AntennaState(1, "tx", torch.tensor([0.0, 0.0, 3.0]), power_w=1.0),
        ReceiverGrid(2, torch.tensor([-3.5, -3.5, 1.5]), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                     (8, 8), (1.0, 1.0)),
    ])


def path_scene():
    wall = Structure(Box(size=(0.3, 4.0, 4.0), position=(2.0, 0.0, 2.0)),
                     PhysicalMaterial(eps_r=5.24, sigma_e=0.12, name="concrete"), name="wall")
    return Scene(structures=[wall], endpoints=[
        AntennaState(1, "tx", torch.tensor([0.0, 0.0, 2.0]), power_w=1.0),
        AntennaState(2, "rx", torch.tensor([4.0, 0.0, 2.0])),
    ])


def probe(label, fn):
    for i in range(3):
        try:
            result = fn()
            print(f"[{label}] try {i}: OK -> {type(result).__name__}")
        except Exception as exc:  # noqa: BLE001 - repro: print every failure verbatim
            print(f"[{label}] try {i}: FAIL -> {exc}")


if __name__ == "__main__":
    probe("stochastic", lambda: radiomap.solve(radiomap_scene(), radiomap.Stochastic(samples=1024),
                                               reference_frequency_hz=FREQUENCY_HZ))
    probe("coherent  ", lambda: radiomap.solve(radiomap_scene(), radiomap.Coherent(),
                                               reference_frequency_hz=FREQUENCY_HZ))
    probe("path      ", lambda: path.solve(path_scene(), path.Config(),
                                           reference_frequency_hz=FREQUENCY_HZ))
