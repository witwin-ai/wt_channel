"""A small solvable channel scene factory for live ``load_platform_scene`` checks.

``load_platform_scene`` with ``factory="wt_channel.examples.demo_scene:build_scene"``
imports this and calls it; the channel adapter detects the ``channel.Scene`` and loads
it into the editor. Pressing **Solve** on the Channel Settings object then runs a
radiomap (CUDA required).
"""
from typing import Any


def build_scene() -> Any:
    """Build a ground-plane + a wall + a Transmitter + a ReceiverGrid at 3.5 GHz."""
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure

    return ch.Scene(
        structures=[
            Structure(Box(size=(30.0, 30.0, 0.2), position=(0.0, 0.0, 0.0)),
                      Material.from_itu("concrete"), name="ground"),
            Structure(Box(size=(0.3, 8.0, 6.0), position=(2.0, 0.0, 3.0)),
                      Material.from_itu("brick"), name="wall"),
        ],
        transmitters=[ch.Transmitter(name="tx", position=(-5.0, 0.0, 3.0), power=1.0)],
        receivers=[
            ch.Receiver(name="rx", position=(8.0, 0.0, 1.5)),  # point Rx for the path solve
            ch.ReceiverGrid(name="rmap", axis="z", position=1.5,
                            bounds=((-12.0, 12.0), (-12.0, 12.0)), grid_shape=(96, 96)),
        ],
        frequency=3.5e9, device="cuda",
    )
