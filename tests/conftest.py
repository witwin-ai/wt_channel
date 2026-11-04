"""Pytest harness for the wt_channel round-trip suite.

Runs against the real environment: ``witwin_server`` (with the frozen
``platform_bridge``) and the channel solver (``witwin.channel`` / ``witwin.core``)
must be importable — e.g. ``pip install -e server`` and the platform packages on
``PYTHONPATH`` (the same packages the running server uses), so no extra PYTHONPATH is
needed at test time. This conftest only adds the plugins directory so
``importlib.import_module("wt_channel")`` resolves (the plugin itself is not
pip-installed; the server adds the same directory at startup). If the base/solver
deps are not importable, the whole suite is skipped rather than erroring.

Adapter / config suites run on CPU. The live
solve suites need CUDA RayD and are gated per-solver by the ``det_ready`` /
``mc_ready`` / ``path_ready`` fixtures (each probes its own solver and skips only if
*that* one can't run here).
"""
import importlib
import importlib.util
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
    if importlib.util.find_spec("witwin_server") is None:
        raise ImportError("witwin_server")
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


def _tiny_radiomap_world():
    # A small Core 0.4 radiomap world for probing whether a solver can run here. It is
    # built directly (not through the adapter) so an adapter bug still fails its test.
    import torch
    from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure
    ground = Structure(
        Mesh(torch.tensor([[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]]),
             torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False, fill_mode="surface"),
        PhysicalMaterial(eps_r=5.24, sigma_e=0.12, name="concrete"), name="ground")
    return Scene(structures=[ground], endpoints=[
        AntennaState(1, "tx", torch.tensor([0.0, 0.0, 3.0]), power_w=1.0),
        ReceiverGrid(2, torch.tensor([-3.5, -3.5, 1.5]), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                     (8, 8), (1.0, 1.0)),
    ])


@pytest.fixture(scope="session")
def cuda(device):
    """Skip live-solve tests when CUDA is unavailable (round-trip tests run on CPU)."""
    if device != "cuda":
        pytest.skip("channel solve requires CUDA RayD/OptiX.")
    return True


@pytest.fixture(scope="session")
def det_ready(cuda):
    """Gate the deterministic live solve on its OptiX pipeline compiling here."""
    from witwin.channel import deterministic
    try:
        deterministic.solve(_tiny_radiomap_world(), deterministic.Config(max_depth=1),
                            reference_frequency_hz=3.5e9)
    except Exception as exc:  # noqa: BLE001 - test gate: needs-user-verification if it can't run
        pytest.skip(f"deterministic solver unavailable in this environment: {exc}")
    return True


@pytest.fixture(scope="session")
def mc_ready(cuda):
    """Gate the Monte Carlo live solve on its solver running here."""
    from witwin.channel.montecarlo import basic
    try:
        basic.solve(_tiny_radiomap_world(), basic.Config(samples=1024, max_depth=1),
                    reference_frequency_hz=3.5e9)
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


@pytest.fixture
def solver_child(tmp_path):
    """Start one of this Plugin's solver children, as the ProjectServer does.

    ``solver_child("witwin.channel.radiomap")`` returns a started
    ``witwin.studio.solver.SolverProcess`` on this interpreter, with a scratch
    Project under ``tmp_path``; it is shut down after the test.
    """
    import tomllib
    from witwin.studio.solver import SolverProcess, SolverSpec

    plugin_root = PLUGINS_DIR / PKG
    manifest = tomllib.loads((plugin_root / "witwin.toml").read_text(encoding="utf-8"))
    processes = []

    def start(solver_id):
        item = next(s for s in manifest["contributes"]["solvers"] if s["id"] == solver_id)
        spec = SolverSpec.from_manifest(manifest["id"], plugin_root,
                                        {**item, "runtime": {"python": sys.executable}})
        process = SolverProcess(spec, tmp_path / "project")
        processes.append(process)
        process.start()
        return process

    yield start
    for process in processes:
        process.close()


@pytest.fixture(scope="session", autouse=True)
def _studio_system_plugins():
    """Activate Studio's system Plugins once, as a ProjectServer does.

    The EM fields of Material and the Transmitter, Receiver and Field
    components belong to the system Plugin ``studio.em`` (2026-11-03); Scenes
    built directly in these tests need it active.
    """
    try:
        from witwin_server.api import testing
    except ImportError:
        return
    testing.activate_system_plugins()
