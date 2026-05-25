"""Pytest harness for the wt_channel round-trip suite.

Runs against the real environment: ``witwin_server`` (with the frozen
``platform_bridge``) and the channel solver (``witwin.channel`` / ``witwin.core``)
must be importable — e.g. ``pip install -e server`` and the platform packages on
``PYTHONPATH`` (the same packages the running server uses), so no extra PYTHONPATH is
needed at test time. This conftest only adds the plugins directory so
``importlib.import_module("wt_channel")`` resolves (the plugin itself is not
pip-installed; the server adds the same directory at startup). If the base/solver
deps are not importable, the whole suite is skipped rather than erroring.

Round-trip suites run on CPU with native-python tuples (``device='cpu'``). The live
solve suite (C3) needs CUDA RayD/OptiX and is gated by the ``solver_ready`` fixture.
"""
import importlib
import sys
from pathlib import Path

import pytest

# .../plugins/wt_channel/tests -> .../plugins
PLUGINS_DIR = Path(__file__).resolve().parents[2]
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

PKG = "wt_channel"

try:
    import witwin_server  # noqa: F401
    import witwin.channel  # noqa: F401
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Skip collection of every test module when the base/solver deps are absent.
if not _DEPS_OK:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session")
def wtc():
    """Import the plugin package (registers components + the channel adapter)."""
    return importlib.import_module(PKG)


@pytest.fixture(scope="session")
def adapter(wtc):
    """A fresh :class:`ChannelAdapter` instance."""
    return importlib.import_module(f"{PKG}.adapter.channel_adapter").ChannelAdapter()


@pytest.fixture(scope="session")
def device():
    """'cuda' when available, else 'cpu' (round-trip works on either)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def solver_ready(device):
    """Skip the live-solve suite if the channel solver can't run a tiny radiomap here."""
    if device != "cuda":
        pytest.skip("channel solve requires CUDA RayD/OptiX; round-trip tests run on CPU.")
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure
    try:
        scene = ch.Scene(
            structures=[Structure(Box(size=(4.0, 4.0, 0.1), position=(0.0, 0.0, 0.0)),
                                  Material.from_itu("concrete", 3.5e9), name="ground")],
            transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 2.0))],
            receivers=[ch.ReceiverGrid(name="rmap", axis="z", position=1.5,
                                       bounds=((-3.0, 3.0), (-3.0, 3.0)), grid_shape=(8, 8))],
            frequency=3.5e9, device="cuda",
        )
        ch.deterministic.solve(scene=scene, transmitter=scene.transmitter("tx"),
                               receiver=scene.receiver("rmap"))
    except Exception as exc:  # noqa: BLE001 - test gate: needs-user-verification if solver absent
        pytest.skip(f"channel solver unavailable in this environment: {exc}")
    return True
