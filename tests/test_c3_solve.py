"""C3 — solver config + library prefabs (CPU) and the live radiomap solve (CUDA).

The config builders (deterministic + Monte Carlo + edge policy) are pure CPU: they map
the ChannelResult component's curated fields onto valid platform Config objects. The
library prefabs build correct studio objects that round-trip. The live solve
(``test_live_*``) needs CUDA RayD/OptiX and is skipped otherwise — it must not be
faked as passing.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from witwin.core import Box, Material, Structure

from _helpers import RT


# --- config builders (CPU) -----------------------------------------------------

def _result_component():
    from wt_channel.components.result import ChannelResultComponent
    return ChannelResultComponent()


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
    config = ConfigBuilder.montecarlo(ch, comp)
    assert isinstance(config, ch.montecarlo.Config)
    assert config.num_samples == 2048
    assert config.max_diffraction_order == 3
    assert config.integrator_options.integrator == "bdpt"
    assert config.integrator_options.samples_per_tx == 4096
    assert config.integrator_options.seed == 7
    assert config.integrator_options.rr_depth == 3
    assert config.integrator_options.bdpt_diffraction_sampling == "hash"


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

def test_library_items_registered(wtc):
    from witwin_server.handlers.library import _pending_items

    names = set(_pending_items)
    assert {"channel_demo", "channel_settings", "channel_tx", "channel_grid",
            "channel_ula", "channel_upa"} <= names


def test_demo_prefab_roundtrips(adapter, wtc):
    from witwin_server import Scene as StudioScene
    from wt_channel import library_items

    studio = StudioScene(kind="local")
    studio.begin_batch()
    ctx = SimpleNamespace(scene=studio, name="Channel Settings")
    studio.add_object(library_items._make_demo(ctx))
    studio.end_batch()
    # The demo settings default to CUDA; force CPU so the round trip needs no GPU.
    studio.get_object_by_name("Channel Settings").get_component("ChannelSettings").device = "cpu"

    platform = adapter.to_platform(studio)
    assert {s.name for s in platform.structures} == {"ground"}
    assert {t.name for t in platform.transmitters} == {"tx"}
    assert {r.name for r in platform.receivers} == {"rmap"}
    assert platform.structures[0].material.itu_descriptor == ("concrete", None)


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
    comp = studio.get_object_by_name("Channel Settings").get_component("ChannelResult")
    comp.solver, comp.transmitter, comp.receiver_grid = solver, "tx", "rmap"
    comp.det_num_samples = comp.mc_num_samples = 2000
    result = SolveRunner.run(studio, comp)
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
