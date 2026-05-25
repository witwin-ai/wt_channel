"""A small solvable channel scene factory for live ``load_platform_scene`` checks.

``load_platform_scene`` with ``factory="wt_channel.examples.demo_scene:build_scene"``
imports this and calls it; the channel adapter detects the ``channel.Scene`` and loads
it into the editor (a RadioMap + a Path Solver + Tx/Rx over the structures). Pressing
**Solve** on the RadioMap then runs a radiomap (CUDA required). Authored Y-up so the
horizontal radiomap lands on the RadioMap's locked XZ plane.
"""
from typing import Any


def build_scene() -> Any:
    """Build a ground-plane + a wall + a Transmitter + a ReceiverGrid at 3.5 GHz (Y up)."""
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure

    return ch.Scene(
        structures=[
            Structure(Box(size=(30.0, 0.2, 30.0), position=(0.0, 0.0, 0.0)),
                      Material.from_itu("concrete"), name="ground"),
            Structure(Box(size=(0.3, 6.0, 8.0), position=(2.0, 3.0, 0.0)),
                      Material.from_itu("brick"), name="wall"),
        ],
        transmitters=[ch.Transmitter(name="tx", position=(-5.0, 3.0, 0.0), power=1.0)],
        receivers=[
            ch.Receiver(name="rx", position=(8.0, 1.5, 0.0)),  # point Rx for the path solve
            ch.ReceiverGrid(name="rmap", axis="y", position=1.5,  # horizontal map (Y up)
                            bounds=((-12.0, 12.0), (-12.0, 12.0)), grid_shape=(96, 96)),
        ],
        frequency=3.5e9, device="cuda",
    )
