"""C3 — solver config + library prefabs (CPU) and the live radiomap solve (CUDA).

The config builders (deterministic + Monte Carlo + edge policy) are pure CPU: they map
the ChannelResult component's curated fields onto valid platform Config objects. The
library prefabs build correct studio objects that round-trip. The live solve
(``test_live_*``) needs CUDA RayD/OptiX and is skipped otherwise — it must not be
faked as passing.
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
from witwin.core import Box, Material, Structure

from _helpers import RT


# --- config builders (CPU) -----------------------------------------------------

def _result_component():
    from wt_channel.components.radiomap import RadioMapComponent
    return RadioMapComponent()


def test_deterministic_config_maps_fields():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.solver = "deterministic"
    comp.det_metric = "sinr"
    comp.det_num_samples = 5000
    comp.det_max_bounces = 3
    comp.det_max_diffraction_order = 2
    comp.det_reflection_field_backend = "drjit"
    comp.det_resolution_wavelength = 0.25
    config = ConfigBuilder.deterministic(ch, comp)
    assert isinstance(config, ch.deterministic.Config)
    assert config.metric == "sinr"
    assert config.num_samples == 5000
    assert config.max_bounces == 3
    assert config.max_diffraction_order == 2
    assert config.tuning.reflection_field_backend == "drjit"
    assert float(config.tuning.resolution_wavelength) == pytest.approx(0.25)


def test_deterministic_stratified_quadrature():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.det_quadrature_mode = "stratified_fixed_n"
    comp.det_samples_per_cell = 8
    config = ConfigBuilder.deterministic(ch, comp)
    assert config.quadrature_mode == "stratified_fixed_n"
    assert config.samples_per_cell == 8


def test_montecarlo_config_maps_fields_and_integrator():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.solver = "montecarlo"
    comp.mc_num_samples = 2048
    comp.mc_integrator = "bdpt"
    comp.mc_max_diffraction_order = 3
    comp.mc_samples_per_tx = 4096
    comp.mc_seed = 7
    comp.mc_use_rr_depth = True
    comp.mc_rr_depth = 3
    comp.mc_bdpt_diffraction_sampling = "hash"
    comp.mc_accumulation_backend = "rayd_reflection_accumulation"
    config = ConfigBuilder.montecarlo(ch, comp)
    assert isinstance(config, ch.montecarlo.Config)
    assert config.num_samples == 2048
    assert config.max_diffraction_order == 3
    assert config.integrator_options.integrator == "bdpt"
    assert config.integrator_options.samples_per_tx == 4096
    assert config.integrator_options.seed == 7
    assert config.integrator_options.rr_depth == 3
    assert config.integrator_options.bdpt_diffraction_sampling == "hash"
    assert config.integrator_options.accumulation_backend == "rayd_reflection_accumulation"


def test_montecarlo_legacy_config_uses_rayd_accumulation_default():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    legacy = SimpleNamespace(**{
        name: getattr(comp, name)
        for name in (
            "mc_resolution_wavelength", "mc_solver_mode", "mc_memory_profile",
            "mc_reflection_field_backend", "mc_integrator", "mc_max_diffraction_order",
            "mc_samples_per_tx", "mc_seed", "mc_use_rr_depth", "mc_rr_depth",
            "mc_use_rr_prob", "mc_rr_prob", "mc_use_stop_threshold", "mc_stop_threshold",
            "mc_bdpt_diffraction_sampling", "mc_num_samples", "mc_max_bounces",
            "mc_synthetic_array", "mc_use_filtering", "ep_vertical_ratio",
            "ep_edge_selection_mode", "ep_edge_diffraction",
        )
    })

    config = ConfigBuilder.montecarlo(ch, legacy)

    assert config.integrator_options.accumulation_backend == "rayd_reflection_accumulation"


def test_montecarlo_basic_clamps_diffraction_order():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.solver = "montecarlo"
    comp.mc_integrator = "basic"
    comp.mc_max_diffraction_order = 3  # basic supports 0/1 -> must clamp
    config = ConfigBuilder.montecarlo(ch, comp)
    assert config.max_diffraction_order == 1


def test_montecarlo_filtering_config():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.solver = "montecarlo"
    comp.mc_use_filtering = True
    comp.mc_filter_method = "bilateral"
    comp.mc_filter_radius = 2
    comp.mc_filter_sigma = 1.5
    config = ConfigBuilder.montecarlo(ch, comp)
    assert config.filtering is not None
    assert config.filtering.reflection.method == "bilateral"
    assert config.filtering.reflection.radius == 2
    assert config.filtering.diffraction.method == "bilateral"


def test_edge_policy_default_and_overrides():
    import witwin.channel as ch
    from wt_channel.adapter.solve import ConfigBuilder

    comp = _result_component()
    comp.ep_vertical_ratio = 0.6
    comp.ep_edge_selection_mode = "vertical_only"
    comp.ep_edge_diffraction = "off"
    policy = ConfigBuilder.edge_policy(ch, comp)
    assert float(policy.vertical_ratio) == pytest.approx(0.6)
    assert policy.edge_selection_mode == "vertical_only"
    assert policy.edge_diffraction is False
    assert policy.boundary_edge_policy == "exclude"


# --- library prefabs (CPU) -----------------------------------------------------

class _FakeResult:
    """Minimal RadioMapResult stand-in for testing the value-range windowing on CPU."""

    def __init__(self, arr):
        import torch
        self._arr = torch.as_tensor(arr, dtype=torch.float32)

    def squeeze_tx(self, index):
        return self

    @property
    def path_gain(self):
        return self._arr


def test_value_range_clamps_without_resolve():
    from wt_channel.adapter.solve import RadioMapExtractor

    result = _FakeResult([[1.0, 10.0], [100.0, 1000.0]])  # linear; 0/10/20/30 dB
    full = RadioMapExtractor.extract(result, metric="path_gain", tx_index=0,
                                     bounds=[-1, 1, -1, 1], log_scale=True)
    np.testing.assert_allclose(sorted(full.data.flatten()), [0.0, 10.0, 20.0, 30.0], atol=1e-3)
    # Windowing to [5, 25] dB clamps the extremes (re-window, same stored result).
    windowed = RadioMapExtractor.extract(result, metric="path_gain", tx_index=0,
                                         bounds=[-1, 1, -1, 1], log_scale=True, vmin=5.0, vmax=25.0)
    assert float(windowed.data.min()) == pytest.approx(5.0)
    assert float(windowed.data.max()) == pytest.approx(25.0)


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

    assert inspect.getargspec is inspect.getfullargspec


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


def test_demo_prefab_roundtrips(adapter, wtc):
    from witwin_server import Scene as StudioScene
    from wt_channel import library_items

    studio = StudioScene()
    studio.begin_batch()
    ctx = SimpleNamespace(scene=studio, name="RadioMap")
    studio.add_object(library_items._make_demo(ctx))  # returns the RadioMap (primary)
    studio.end_batch()
    # The RadioMap is the scene-config carrier; force CPU so the round trip needs no GPU.
    studio.get_object_by_name("RadioMap").get_component("RadioMap").device = "cpu"

    platform = adapter.to_platform(studio)
    assert {s.name for s in platform.structures} == {"Channel Ground"}
    assert {t.name for t in platform.transmitters} == {"Tx"}
    assert {r.name for r in platform.receivers} == {"RadioMap", "Rx"}  # grid + point Rx
    assert platform.structures[0].material.itu_descriptor == ("concrete", None)


def test_default_solver_is_montecarlo():
    # Deterministic needs its OptiX pipeline to compile; MC is the robust default.
    from wt_channel.components.radiomap import RadioMapComponent

    assert str(RadioMapComponent().solver) == "montecarlo"


def test_radiomap_is_field_locked_to_xz():
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()
    assert "Field" in comp._get_base_class_names()  # inherits Field -> gizmo + plane viz
    assert str(comp.plane) == "XZ"
    assert comp._fields_meta["plane"].hidden is True  # locked (not switchable in UI)


def test_radiomap_plane_volume_orientation():
    # The platform tensor is (n_v=bounds1, n_u=bounds0); the field volume must be world
    # axis-aligned so data[iv, iu] lands at world (t0=iu, t1=iv) on the XZ plane.
    from wt_channel.components.radiomap import RadioMapComponent

    comp = RadioMapComponent()  # plane XZ -> t0=X(0), t1=Z(2), normal=Y(1)
    data = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)  # (nZ=2, nX=3)
    vol = comp._to_plane_volume(data)
    assert vol.shape == (3, 1, 2)  # (nX, 1, nZ)
    for iz in range(2):
        for ix in range(3):
            assert vol[ix, 0, iz] == data[iz, ix]


def test_manifest_declares_rfc012_channel_solvers():
    manifest = tomllib.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "witwin.toml").read_text())
    solvers = {item["id"]: item for item in manifest["contributes"]["solvers"]}
    assert solvers["witwin.channel.radiomap"]["entry"] == "solver_host.py"
    assert solvers["witwin.channel.path"]["entry"] == "solver_host.py"
    assert solvers["witwin.channel.radiomap"]["runtime"]["python"] == "${backend:python}"
    assert solvers["witwin.channel.path"]["runtime"]["python"] == "${backend:python}"
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
    assert notifications["success"] == [("Channel", "montecarlo radiomap solved")]
    assert notifications["error"] == []


# --- live solve (CUDA) ---------------------------------------------------------

def _live_scene():
    import witwin.channel as ch
    return ch.Scene(
        structures=[Structure(Box(size=(20.0, 20.0, 0.2), position=(0.0, 0.0, 0.0)),
                              Material.from_itu("concrete"), name="ground")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 4.0))],
        receivers=[ch.ReceiverGrid(name="rmap", axis="z", position=1.5,
                                   bounds=((-8.0, 8.0), (-8.0, 8.0)), grid_shape=(16, 16))],
        frequency=3.5e9, device="cuda",
    )


def _run_live(adapter, solver):
    # Round-trip then solve via the adapter; return the (ny,nx) path-gain map.
    from wt_channel.adapter.solve import RadioMapExtractor, SolveRunner

    studio = adapter.to_studio(_live_scene())
    comp = studio.get_object_by_name("rmap").get_component("RadioMap")  # the grid is a RadioMap
    comp.solver = solver
    comp.det_num_samples = comp.mc_num_samples = 2000
    result = SolveRunner.run(studio, comp, transmitter="tx", receiver="rmap")
    return result, RadioMapExtractor._to_numpy(result.squeeze_tx(0).path_gain)


def test_live_deterministic_matches_direct(adapter, det_ready):
    """Round-trip then deterministic solve; assert the map matches a direct run's shape."""
    import witwin.channel as ch

    result, got = _run_live(adapter, "deterministic")
    ref = ch.deterministic.solve(scene=_live_scene(), transmitter="tx", receiver="rmap")
    assert tuple(result.grid_shape) == tuple(ref.grid_shape) == (16, 16)
    assert got.shape == (16, 16) and np.isfinite(got).any()


def test_live_montecarlo_matches_direct(adapter, mc_ready):
    """Round-trip then Monte Carlo solve; assert the map matches a direct run's shape."""
    import witwin.channel as ch

    result, got = _run_live(adapter, "montecarlo")
    ref = ch.montecarlo.solve(scene=_live_scene(), transmitter="tx", receiver="rmap")
    assert tuple(result.grid_shape) == tuple(ref.grid_shape) == (16, 16)
    assert got.shape == (16, 16) and np.isfinite(got).any()


def test_live_radiomap_solver_host_solves_and_queries(adapter, mc_ready):
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
        studio = adapter.to_studio(_live_scene())
        comp = studio.get_object_by_name("rmap").get_component("RadioMap")
        comp.solver = "montecarlo"
        comp.mc_num_samples = 128
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
