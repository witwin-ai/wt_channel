"""A small solvable channel world factory for live ``load_platform_scene`` checks.

``load_platform_scene`` with ``factory="wt_channel.examples.demo_scene:build_scene"``
imports this and calls it; the channel adapter detects the Core world (it carries
endpoints) and loads it into the editor (a RadioMap + a Path Solver + Tx/Rx over the
structures). Pressing **Solve** on the RadioMap then runs a radiomap (CUDA required).
Authored Y-up so the horizontal radiomap lands on the RadioMap's locked XZ plane.
"""
from typing import Any


def build_scene() -> Any:
    """Build a ground plane + a wall + a Tx + an Rx + a 96x96 grid at 3.5 GHz (Y up)."""
    from witwin.core import AntennaState, Box, ReceiverGrid, Scene, Structure

    from ..adapter.itu import itu_material

    frequency = 3.5e9
    cell = 24.0 / 96
    return Scene(
        structures=[
            Structure(Box(size=(30.0, 0.2, 30.0), position=(0.0, 0.0, 0.0)),
                      itu_material("concrete", frequency), name="ground"),
            Structure(Box(size=(0.3, 6.0, 8.0), position=(2.0, 3.0, 0.0)),
                      itu_material("brick", frequency), name="wall"),
        ],
        endpoints=[
            AntennaState(1, "tx", (-5.0, 3.0, 0.0), polarization=(0.0, 1.0, 0.0),
                         power_w=1.0, metadata={"name": "tx"}),
            AntennaState(2, "rx", (8.0, 1.5, 0.0), metadata={"name": "rx"}),
            ReceiverGrid(3, (-12.0 + cell / 2, 1.5, -12.0 + cell / 2), (1.0, 0.0, 0.0),
                         (0.0, 0.0, 1.0), (96, 96), (cell, cell), metadata={"name": "rmap"}),
        ],
        metadata={"channel": {"frequency_hz": frequency}},
    )
