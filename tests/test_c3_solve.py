"""C3 — solver config + library prefabs (CPU) and the live radiomap solve (CUDA).

The config builders map the RadioMap component's curated fields onto the witwin-channel
0.5 solver configs (``deterministic.Config``, ``montecarlo.basic.Config``,
``montecarlo.bdpt.Config``). The library prefabs build correct studio objects. The live
solve (``test_live_*``) needs CUDA RayD and is skipped otherwise — it must not be faked
as passing.
"""
import asyncio
import json
import tomllib
import shutil
import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pytest


# --- config builders (CPU) -----------------------------------------------------

def _result_component():
    from wt_channel.components.radiomap import RadioMapComponent
    return RadioMapComponent()


def test_deterministic_config_maps_fields():
    from witwin.channel import deterministic
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.solver = "deterministic"
    comp.det_max_bounces = 3
    comp.det_max_diffraction_order = 2

    config = ConfigBuilder.deterministic(comp)

    assert isinstance(config, deterministic.Config)
    assert config.max_depth == 3
    assert config.max_diffraction_order == 1  # public capability is first order
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.return_field is False  # the radiomap only reads path gain


def test_montecarlo_basic_config_maps_fields():
    from witwin.channel.montecarlo import basic
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.mc_integrator = "basic"
    comp.mc_samples_per_tx = 4096
    comp.mc_max_bounces = 2
    comp.mc_seed = 7

    config = ConfigBuilder.montecarlo(comp)

    assert isinstance(config, basic.Config)
    assert config.samples == 4096
    assert config.max_depth == 2
    assert config.seed == 7
    assert config.components == frozenset({"los", "reflection", "diffraction"})


def test_montecarlo_bdpt_config_maps_fields():
    from witwin.channel.montecarlo import bdpt
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.mc_integrator = "bdpt"
    comp.mc_samples_per_tx = 2048
    comp.mc_max_bounces = 3
    comp.mc_max_diffraction_order = 3

    config = ConfigBuilder.montecarlo(comp)

    assert isinstance(config, bdpt.Config)
    assert config.samples == 2048
    assert config.max_depth == 3
    assert config.max_diffraction_order == 1


def test_components_drop_reflection_and_diffraction_when_disabled():
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.mc_max_bounces = 0
    comp.mc_max_diffraction_order = 0

    config = ConfigBuilder.montecarlo(comp)

    assert config.components == frozenset({"los"})
    assert config.max_depth == 0


# --- result maps (CPU) -----------------------------------------------------------

def test_radiomap_maps_derive_rss_and_sinr():
    from wt_channel.adapter.solve import RadioMapMaps

    gain = np.array([[[1e-6, 1e-8], [1e-10, 0.0]]], dtype=np.float32)  # (T=1, n_v=2, n_u=2)
    maps = RadioMapMaps(path_gain=gain, tx_power_w=np.array([2.0]), noise_power_w=1e-12)

    np.testing.assert_allclose(maps.metric("path_gain"), gain)
    np.testing.assert_allclose(maps.metric("rss"), gain * 2.0)
    np.testing.assert_allclose(maps.metric("sinr"), gain * 2.0 / 1e-12, rtol=1e-6)
    assert set(maps.metrics()) == {"path_gain", "rss", "sinr"}


def test_radiomap_maps_without_noise_have_no_sinr():
    from wt_channel.adapter.solve import RadioMapMaps

    maps = RadioMapMaps(path_gain=np.ones((1, 2, 2), np.float32), tx_power_w=np.array([1.0]))

    assert set(maps.metrics()) == {"path_gain", "rss"}
    with pytest.raises(ValueError, match="sinr"):
        maps.metric("sinr")


def test_grid_map_unflattens_monte_carlo_basic_layout():
    # MC basic publishes a grid as (T, n_v * n_u), row-major over (v, u); the
    # deterministic solver already publishes (T, n_v, n_u).
    from wt_channel.adapter.solve import grid_map

    flat = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)  # T=2, grid (n_u=3, n_v=2)
    shaped = grid_map(flat, (3, 2))

    assert shaped.shape == (2, 2, 3)
    assert shaped[1, 1, 2] == flat[1, 1 * 3 + 2]
    np.testing.assert_array_equal(grid_map(shaped, (3, 2)), shaped)
    with pytest.raises(ValueError, match="grid map"):
        grid_map(np.zeros((1, 5), np.float32), (3, 2))


def test_value_range_clamps_without_resolve():
    from wt_channel.adapter.solve import RadioMapExtractor, RadioMapMaps

    maps = RadioMapMaps(path_gain=np.array([[[1.0, 10.0], [100.0, 1000.0]]], np.float32),
                        tx_power_w=np.array([1.0]))
    full = RadioMapExtractor.extract(maps, metric="path_gain", tx_index=0,
                                     bounds=[-1, 1, -1, 1], log_scale=True)
    np.testing.assert_allclose(sorted(full.data.flatten()), [0.0, 10.0, 20.0, 30.0], atol=1e-3)
    windowed = RadioMapExtractor.extract(maps, metric="path_gain", tx_index=0,
                                         bounds=[-1, 1, -1, 1], log_scale=True, vmin=5.0, vmax=25.0)
    assert float(windowed.data.min()) == pytest.approx(5.0)
    assert float(windowed.data.max()) == pytest.approx(25.0)


def test_solver_host_cache_keeps_every_metric():
    import wt_channel.solver_host as host
    from wt_channel.adapter.solve import RadioMapMaps

    maps = RadioMapMaps(path_gain=np.ones((1, 2, 3), np.float32), tx_power_w=np.array([4.0]))

    cache = host._radiomap_cache(maps)

    assert set(cache.metrics) == {"path_gain", "rss"}
    assert cache.metrics["rss"].shape == (1, 2, 3)
    np.testing.assert_allclose(cache.metrics["rss"], 4.0)


# --- library prefabs (CPU) -----------------------------------------------------

def test_library_items_registered(wtc):
    from witwin_server.features.library.service import _pending_items

    names = set(_pending_items)
    assert {"channel_demo", "channel_tx", "channel_rx", "channel_radiomap",
            "channel_pathsolver"} <= names


def test_platform_compat_restores_getargspec(monkeypatch):
    import inspect
    from wt_channel._compat import apply_platform_compat

    monkeypatch.delattr(inspect, "getargspec", raising=False)

    apply_platform_compat()

    args, varargs, keywords, defaults = inspect.getargspec(
        lambda value, *args, **kwargs: value
    )
    assert args == ["value"]
    assert varargs == "args"
    assert keywords == "kwargs"
    assert defaults is None


def test_demo_prefab_does_not_import_channel_platform(monkeypatch, wtc):
    import builtins
    from witwin_server import Scene as StudioScene
    from wt_channel import library_items

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "witwin.channel" or name == "witwin.core" or name.startswith("witwin.channel."):
            raise AssertionError(f"demo prefab imported platform package: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    studio = StudioScene()
    ctx = SimpleNamespace(scene=studio, name="RadioMap")

    obj = library_items._make_demo(ctx)

    assert obj.name == "RadioMap"
    assert {item.name for item in studio.objects.values()} == {"Channel Ground", "Tx", "Rx", "Path Solver"}


def test_demo_prefab_builds_core_world(adapter, wtc):
    from witwin_server import Scene as StudioScene
    from wt_channel import library_items

    studio = StudioScene()
    studio.begin_batch()
    ctx = SimpleNamespace(scene=studio, name="RadioMap")
    studio.add_object(library_items._make_demo(ctx))  # returns the RadioMap (primary)
    studio.end_batch()

    world = adapter.to_platform(studio)

    assert {s.name for s in world.structures} == {"Channel Ground"}
    assert sorted(e.metadata["name"] for e in world.endpoints) == ["RadioMap", "Rx", "Tx"]
    assert world.structures[0].material.name == "concrete"


def test_demo_scene_factory_builds_core_world(wtc):
    from witwin.core import Scene
    from wt_channel.examples.demo_scene import build_scene

    world = build_scene()

    assert isinstance(world, Scene)
    assert len(world.structures) >= 1
    assert {e.role for e in world.endpoints} == {"tx", "rx"}


def test_default_solver_is_montecarlo():
    from wt_channel.components.radiomap import RadioMapComponent

    assert str(RadioMapComponent().solver) == "montecarlo"


def test_radiomap_is_field_locked_to_xz():
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()
    assert "Field" in comp._get_base_class_names()  # inherits Field -> gizmo + plane viz
    assert str(comp.plane) == "XZ"
    assert comp._fields_meta["plane"].hidden is True  # locked (not switchable in UI)


def test_radiomap_plane_volume_orientation():
    # The solved map is (n_v=Z, n_u=X); the field volume must be world axis-aligned so
    # data[iv, iu] lands at world (t0=iu, t1=iv) on the XZ plane.
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()  # plane XZ -> t0=X(0), t1=Z(2), normal=Y(1)
    data = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)  # (nZ=2, nX=3)
    vol = comp._to_plane_volume(data)
    assert vol.shape == (3, 1, 2)  # (nX, 1, nZ)
    for iz in range(2):
        for ix in range(3):
            assert vol[ix, 0, iz] == data[iz, ix]


def test_dB_map_display_ignores_uncovered_cells():
    # Uncovered cells come back at -300 dB (log of 0); left in, the plane's min-max
    # normalization squeezes every covered cell (-78..-140 dB) into the top colors.
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()
    covered = np.linspace(-140.0, -78.0, 98).astype(np.float32)
    data = np.concatenate([covered, [-300.0, -300.0]]).reshape(10, 10)

    shown = comp._display_values(data)

    assert shown.max() == pytest.approx(-78.0)
    assert shown.min() >= np.percentile(covered, 1) - 1e-3
    assert shown.reshape(-1)[-1] == pytest.approx(shown.min())  # uncovered -> lowest color
    np.testing.assert_allclose(shown.reshape(-1)[50:98], data.reshape(-1)[50:98])


def test_linear_map_display_is_unchanged():
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()
    comp.log_scale = False
    data = np.array([[0.0, 1e-9], [1e-8, 1e-7]], np.float32)

    np.testing.assert_array_equal(comp._display_values(data), data)


def test_manifest_declares_rfc012_channel_solvers():
    manifest = tomllib.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "witwin.toml").read_text())
    solvers = {item["id"]: item for item in manifest["contributes"]["solvers"]}
    assert solvers["witwin.channel.radiomap"]["entry"] == "solver_host.py"
    assert solvers["witwin.channel.path"]["entry"] == "solver_host.py"
    assert solvers["witwin.channel.radiomap"]["runtime"]["python"] == "${project:python}"
    assert solvers["witwin.channel.path"]["runtime"]["python"] == "${project:python}"
    assert solvers["witwin.channel.radiomap"]["maxParallel"] == 1
    assert solvers["witwin.channel.path"]["maxParallel"] == 1


def test_radiomap_component_imports_api_facade_after_api_submodule_loaded():
    import importlib
    from witwin_server.api import WitwinApi
    import witwin_server.api as api_submodule
    import wt_channel.components.radiomap as radiomap_mod

    reloaded = importlib.reload(radiomap_mod)

    assert reloaded.api is not api_submodule
    assert isinstance(reloaded.api, WitwinApi)
    assert hasattr(reloaded.api, "solvers")


def test_radiomap_solver_config_is_json_serializable():
    from wt_channel.components.radiomap import RadioMapComponent

    config = RadioMapComponent()._solver_config("tx", "rmap")

    json.dumps(config)


def test_radiomap_component_uses_solver_api_for_solve(monkeypatch):
    import wt_channel.adapter.channel_adapter as channel_adapter_mod
    import wt_channel.components.radiomap as radiomap_mod
    from witwin_server.core import Scene, SceneObject as SO
    from wt_channel.components.radiomap import RadioMapComponent

    studio = Scene()
    host = SO(name="rmap", mesh_type="Empty")
    tx = SO(name="tx", mesh_type="Empty")
    studio.add_object(host)
    studio.add_object(tx)
    comp = RadioMapComponent()
    comp._owner = host
    host._components["RadioMap"] = comp
    tx._components["ChannelTx"] = object()
    comp.solver = "montecarlo"
    comp.mc_num_samples = 123

    class FakeRun:
        status = "succeeded"
        outputs = {"resultHandle": "map-handle"}
        run_id = "map-run"
        error = None

    class FakeSolvers:
        def __init__(self):
            self.solve_kwargs = None
            self.queries = []

        def solve(self, solver_id, **kwargs):
            self.solve_kwargs = {"solver_id": solver_id, **kwargs}
            return FakeRun()

        def query(self, solver_id, result_handle, op, params, **kwargs):
            self.queries.append((solver_id, result_handle, op, params, kwargs))
            return {
                "data": {
                    "data": np.ones((2, 3), dtype=np.float32),
                    "extent": [-1.0, 1.0, -1.0, 1.0],
                    "label": "path_gain",
                }
            }

    fake = FakeSolvers()
    notifications = {"success": [], "error": []}

    class FakeNotifications:
        @staticmethod
        def success(title, message):
            notifications["success"].append((title, message))

        @staticmethod
        def error(title, message):
            notifications["error"].append((title, message))

    monkeypatch.setattr(channel_adapter_mod.ChannelAdapter, "has_structures", staticmethod(lambda scene: True))
    monkeypatch.setattr(radiomap_mod, "api", type("FakeApi", (), {"solvers": fake})())
    monkeypatch.setattr(radiomap_mod, "Notifications", FakeNotifications)

    msg = asyncio.run(comp.solve())

    assert msg == "Solve complete"
    assert comp._solver_result_handle == "map-handle"
    assert comp._solver_run_id == "map-run"
    assert fake.solve_kwargs["solver_id"] == "witwin.channel.radiomap"
    assert fake.solve_kwargs["scene"] is studio
    assert "surface_progress" not in fake.solve_kwargs
    assert fake.solve_kwargs["config"]["mode"] == "radiomap"
    assert fake.solve_kwargs["config"]["component"]["solver"] == "montecarlo"
    assert fake.solve_kwargs["config"]["component"]["mc_num_samples"] == 123
    assert fake.queries[0][2] == "extract_map"
    assert notifications["success"] == []  # the map itself (and Live Status) report the solve
    assert notifications["error"] == []


# --- live solve (CUDA) ---------------------------------------------------------

def _live_studio():
    """20 m concrete ground (Y up), Tx near the +X/-Z corner, 16x16 map at 1.5 m."""
    from witwin_server import Scene as StudioScene, SceneObject
    from wt_channel.components.endpoints import TxComponent
    from wt_channel.components.radiomap import RadioMapComponent

    studio = StudioScene()
    ground = SceneObject(name="ground", mesh_type="Custom")
    verts = np.array([[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]], np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)
    ground.get_component("Mesh").set_mesh_data(verts, faces, notify=False)
    studio.add_object(ground)

    tx = SceneObject(name="tx", mesh_type="Empty")
    tx_comp = tx.add_component(TxComponent())
    tx_comp.polarization = [0.0, 1.0, 0.0]
    tx.get_component("Transform").position = [7.0, 4.0, -7.0]
    studio.add_object(tx)

    rmap = SceneObject(name="rmap", mesh_type="Empty")
    comp = rmap.add_component(RadioMapComponent())
    comp.min_bound = [-8.0, 0.0, -8.0]
    comp.max_bound = [8.0, 0.0, 8.0]
    comp.resolution = [16.0, 1.0, 16.0]
    rmap.get_component("Transform").position = [0.0, 1.5, 0.0]
    studio.add_object(rmap)
    return studio, comp


def _run_live(solver):
    from wt_channel.adapter.solve import SolveRunner

    studio, comp = _live_studio()
    comp.solver = solver
    comp.mc_samples_per_tx = 1 << 16
    # The deterministic solver adds paths coherently, so the ground reflection's
    # interference moves its strongest cell away from under the Tx; LoS alone does not.
    comp.det_max_bounces = 0
    comp.det_max_diffraction_order = 0
    return SolveRunner.run(studio, comp, transmitter="tx", receiver="rmap")


def _assert_map_oriented(maps):
    gain = maps.metric("path_gain")
    assert gain.shape == (1, 16, 16)  # (T, n_v=Z, n_u=X)
    assert np.isfinite(gain).all() and (gain > 0).any()
    # Tx sits above world (x=+7, z=-7): the (high u = +X, low v = -Z) quadrant holds the
    # most power; a transposed map would favour one of the two mixed quadrants. The
    # strongest cell is not a robust marker: a vertical dipole has a null straight below
    # it, so its peak is a ring around the Tx.
    g = gain[0]
    quadrants = {"tx": g[:8, 8:].sum(), "u_low_v_low": g[:8, :8].sum(),
                 "u_high_v_high": g[8:, 8:].sum(), "far": g[8:, :8].sum()}
    assert max(quadrants, key=quadrants.get) == "tx", quadrants


def test_live_montecarlo_radiomap_is_oriented(mc_ready):
    _assert_map_oriented(_run_live("montecarlo"))


def test_live_deterministic_radiomap_is_oriented(det_ready):
    _assert_map_oriented(_run_live("deterministic"))


def test_live_radiomap_solver_host_solves_and_queries(mc_ready):
    """Exercise the real RFC-012 child process path for radiomap solve + query."""
    from pathlib import Path
    from witwin_server.features.runs.store import RunStore
    from witwin_server.features.solvers.host import SolverHost
    from witwin_server.features.solvers.scene_ref import make_scene_ref
    from witwin_server.features.solvers.spec import SolverSpec

    plugin_root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((plugin_root / "witwin.toml").read_text())
    item = next(s for s in manifest["contributes"]["solvers"]
                if s["id"] == "witwin.channel.radiomap")
    spec = SolverSpec.from_manifest(
        "witwin.channel",
        plugin_root,
        {**item, "runtime": {"python": sys.executable}},
    )
    work_dir = (Path.cwd() / ".pytest_tmp_wt_channel" / f"radiomap-host-{uuid.uuid4().hex}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    host = SolverHost(spec, RunStore(work_dir / "runs.db"))
    try:
        studio, comp = _live_studio()
        comp.solver = "montecarlo"
        comp.mc_samples_per_tx = 2048
        comp.mc_max_bounces = 1
        comp.mc_max_diffraction_order = 0

        final = host.solve(
            scene_ref=make_scene_ref(studio),
            config=comp._solver_config("tx", "rmap"),
        )

        assert final.status == "succeeded", final.error
        handle = final.outputs["resultHandle"]
        queried = host.query(
            handle,
            "extract_map",
            {"metric": "path_gain", "tx_index": 0, "log_scale": True},
            run_id=final.run_id,
        )
        data = np.asarray(queried["data"]["data"], dtype=np.float32)
        assert data.shape == (16, 16)
        assert np.isfinite(data).any()
    finally:
        host.shutdown(force=True)
        shutil.rmtree(work_dir, ignore_errors=True)
