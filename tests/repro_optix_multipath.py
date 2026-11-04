"""Standalone OptiX 'multipath' repro — pure witwin.channel, no studio / wt_channel.

Not a pytest test (no ``test_`` prefix, so it is never collected). Run it directly in
whichever env you want to check, to confirm whether the deterministic + path solvers'
shared 'multipath' OptiX pipeline compiles there:

    # conda witwin2 (the wt_channel test env)
    C:/Users/Asixa/miniconda3/envs/witwin2/python.exe repro_optix_multipath.py

    # or the env your running editor/server uses
    <server-env-python> repro_optix_multipath.py

Monte Carlo uses a different pipeline and is expected to work everywhere; deterministic
and path share the 'multipath' pipeline that fails to compile in conda witwin2. If all
three print OK in your env, the gated live tests (test_c3 deterministic / test_c4 path)
will run and pass there automatically — no code change needed.
"""
import witwin.channel as ch
from witwin.core import Box, Structure
from wt_channel.adapter.itu import itu_material


def radiomap_scene():
    return ch.Scene(
        structures=[Structure(Box(size=(8.0, 8.0, 0.1), position=(0.0, 0.0, 0.0)),
                              itu_material("concrete", 3.5e9), name="g")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 3.0))],
        receivers=[ch.ReceiverGrid(name="rmap", axis="z", position=1.5,
                                   bounds=((-4.0, 4.0), (-4.0, 4.0)), grid_shape=(8, 8))],
        frequency=3.5e9, device="cuda")


def path_scene():
    return ch.Scene(
        structures=[Structure(Box(size=(0.3, 4.0, 4.0), position=(2.0, 0.0, 2.0)),
                              itu_material("concrete", 3.5e9), name="wall")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 2.0))],
        receivers=[ch.Receiver(name="rx", position=(4.0, 0.0, 2.0))],
        frequency=3.5e9, device="cuda")


def probe(label, fn):
    for i in range(3):
        try:
            result = fn()
            print(f"[{label}] try {i}: OK -> {type(result).__name__}")
        except Exception as exc:  # noqa: BLE001 - repro: print every failure verbatim
            print(f"[{label}] try {i}: FAIL -> {exc}")


if __name__ == "__main__":
    probe("montecarlo  ", lambda: ch.montecarlo.solve(scene=radiomap_scene(),
                                                       transmitter="tx", receiver="rmap"))
    probe("deterministic", lambda: ch.deterministic.solve(scene=radiomap_scene(),
                                                          transmitter="tx", receiver="rmap"))
    probe("path        ", lambda: ch.path.solve(scene=path_scene(),
                                                 transmitter="tx", receiver="rx"))
