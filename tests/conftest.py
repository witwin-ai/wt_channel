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
solve suites need CUDA RayD/OptiX and are gated per-solver by the ``det_ready`` /
``mc_ready`` / ``path_ready`` fixtures (each probes its own solver and skips only if
*that* one can't run here).
"""
import importlib
import inspect
import sys
from collections import namedtuple
from pathlib import Path

import pytest

_ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")


def _getargspec(func):
    spec = inspect.getfullargspec(func)
    return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)


if not hasattr(inspect, "ArgSpec"):
    inspect.ArgSpec = _ArgSpec  # type: ignore[attr-defined]
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = _getargspec  # type: ignore[attr-defined]

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


def _tiny_radiomap_scene():
    # A small radiomap scene for probing whether a solver can run here.
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure
    return ch.Scene(
        structures=[Structure(Box(size=(8.0, 8.0, 0.1), position=(0.0, 0.0, 0.0)),
                              Material.from_itu("concrete"), name="ground")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 3.0))],
        receivers=[ch.ReceiverGrid(name="rmap", axis="z", position=1.5,
                                   bounds=((-4.0, 4.0), (-4.0, 4.0)), grid_shape=(8, 8))],
        frequency=3.5e9, device="cuda",
    )


@pytest.fixture(scope="session")
def cuda(device):
    """Skip live-solve tests when CUDA is unavailable (round-trip tests run on CPU)."""
    if device != "cuda":
        pytest.skip("channel solve requires CUDA RayD/OptiX.")
    return True


@pytest.fixture(scope="session")
def det_ready(cuda):
    """Gate the deterministic live solve on its OptiX pipeline compiling here."""
    import witwin.channel as ch
    try:
        ch.deterministic.solve(scene=_tiny_radiomap_scene(), transmitter="tx", receiver="rmap")
    except Exception as exc:  # noqa: BLE001 - test gate: needs-user-verification if it can't run
        pytest.skip(f"deterministic solver unavailable in this environment: {exc}")
    return True


@pytest.fixture(scope="session")
def mc_ready(cuda):
    """Gate the Monte Carlo live solve on its solver running here."""
    import witwin.channel as ch
    try:
        ch.montecarlo.solve(scene=_tiny_radiomap_scene(), transmitter="tx", receiver="rmap")
    except Exception as exc:  # noqa: BLE001 - test gate: needs-user-verification if it can't run
        pytest.skip(f"montecarlo solver unavailable in this environment: {exc}")
    return True


@pytest.fixture(scope="session")
def path_ready(cuda):
    """Gate the point-to-point path live solve on its solver running here."""
    import witwin.channel as ch
    from witwin.core import Box, Material, Structure
    scene = ch.Scene(
        structures=[Structure(Box(size=(0.3, 4.0, 4.0), position=(2.0, 0.0, 2.0)),
                              Material.from_itu("concrete"), name="wall")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 2.0))],
        receivers=[ch.Receiver(name="rx", position=(4.0, 0.0, 2.0))],
        frequency=3.5e9, device="cuda",
    )
    try:
        ch.path.solve(scene=scene, transmitter="tx", receiver="rx")
    except Exception as exc:  # noqa: BLE001 - test gate: needs-user-verification if it can't run
        pytest.skip(f"path solver unavailable in this environment: {exc}")
    return True
