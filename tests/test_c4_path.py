"""C4 — point-to-point path solve + viewport ray drawing.

The path config builder and the CIR/CFR/taps + poly-line extractors are CPU-testable
against a synthetic ``PathResult`` (built from a payload, no solver). The live path
solve (``test_live_*``) needs CUDA RayD/OptiX and is skipped otherwise.
"""
import numpy as np
import pytest
from witwin.core import Box, Material, Structure

_C = 299792458.0


def _synthetic_result():
    """A 2-path PathResult: a LoS path and a single-reflection path (geometry included)."""
    import torch
    from witwin.channel.path.result import PathResult

    ps = (1, 1, 1, 1, 2)        # path shape
    cs = (1, 1, 1, 1, 2, 1)     # coeff shape (+ time)
    ds = (1, 1, 1, 1, 2, 2)     # depth shape
    payload = {
        "name": "path", "num_rx": 1, "num_tx": 1, "num_rx_ant": 1, "num_tx_ant": 1,
        "num_time_steps": 1, "max_num_paths": 2, "max_depth": 2,
        "tx_pos": (0.0, 0.0, 2.0),
        "tx_positions": torch.tensor([[0.0, 0.0, 2.0]]),
        "rx_positions": torch.tensor([[6.0, 0.0, 1.5]]),
        "frequency": 3.5e9, "wavelength": _C / 3.5e9,
        "a": torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=torch.complex64).reshape(cs),
        "tau": torch.tensor([2.0e-8, 3.0e-8]).reshape(ps),
        "theta_t": torch.zeros(ps), "phi_t": torch.zeros(ps),
        "theta_r": torch.zeros(ps), "phi_r": torch.zeros(ps),
        "valid": torch.tensor([True, True]).reshape(ps),
        "types": torch.tensor([[0, 0], [1, 0]], dtype=torch.int32).reshape(ds),
        "num_paths": torch.tensor([2], dtype=torch.int32).reshape((1, 1, 1, 1)),
        "vertices": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
             [[3.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]).reshape(ds + (3,)),
        "metadata": {},
    }
    return PathResult._from_payload(payload)


def test_synthetic_result_builds():
    result = _synthetic_result()
    assert result.max_num_paths == 2
    assert result.vertices is not None


def test_path_lines_classified_and_routed():
    from wt_channel.adapter.path_solve import PathLineBuilder

    lines = PathLineBuilder.build(_synthetic_result())
    assert len(lines) == 2
    los, refl = lines
    assert los.interaction == "los"
    assert los.positions == [[0.0, 0.0, 2.0], [6.0, 0.0, 1.5]]   # Tx -> Rx
    assert refl.interaction == "reflection"
    assert refl.positions == [[0.0, 0.0, 2.0], [3.0, 0.0, 1.0], [6.0, 0.0, 1.5]]  # Tx -> bounce -> Rx
    assert los.color != refl.color


def test_cir_plot_data():
    from wt_channel.adapter.path_solve import PathPlots

    data = PathPlots.cir(_synthetic_result())
    assert data.kind == "stem"
    assert data.x.shape == (2,)
    # cir() normalizes delays to the first arrival -> [0, 10] ns (raw 20/30 ns).
    np.testing.assert_allclose(np.sort(data.x), [0.0, 10.0], atol=1e-3)
    np.testing.assert_allclose(np.sort(data.y), [0.5, 1.0], atol=1e-3)


def test_taps_plot_data():
    from wt_channel.adapter.path_solve import PathPlots

    data = PathPlots.taps(_synthetic_result(), bandwidth=100e6, num_taps=16)
    assert data.kind == "stem"
    assert data.x.shape == (16,)
    assert data.y.sum() > 0.0


def test_cfr_plot_data():
    from wt_channel.adapter.path_solve import PathPlots

    data = PathPlots.cfr(_synthetic_result(), num_points=64)
    assert data.kind == "line"
    assert data.x.shape == (64,) and data.y.shape == (64,)


def test_path_config_builder():
    import witwin.channel as ch
    from wt_channel.adapter.path_solve import PathConfigBuilder
    from wt_channel.components.path import PathSolveComponent

    comp = PathSolveComponent()
    comp.num_samples = 5000
    comp.max_bounces = 4
    comp.max_diffraction_order = 2
    comp.use_max_num_paths = True
    comp.max_num_paths = 25
    comp.draw_paths = True
    config = PathConfigBuilder.build(ch, comp)
    assert isinstance(config, ch.path.Config)
    assert config.num_samples == 5000
    assert config.max_bounces == 4
    assert config.max_diffraction_order == 2
    assert config.max_num_paths == 25
    assert config.return_geometry is True


# --- live path solve (CUDA) ----------------------------------------------------

def test_live_path_solve(adapter, solver_ready):
    """Round-trip then path-solve a discrete Tx<->Rx link; check geometry + CIR."""
    import witwin.channel as ch
    from wt_channel.adapter.path_solve import PathLineBuilder, PathSolveRunner

    scene = ch.Scene(
        structures=[Structure(Box(size=(0.3, 6.0, 6.0), position=(3.0, 0.0, 3.0)),
                              Material.from_itu("concrete"), name="wall")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 3.0))],
        receivers=[ch.Receiver(name="rx", position=(6.0, 0.0, 3.0))],
        frequency=3.5e9, device="cuda",
    )
    studio = adapter.to_studio(scene)
    comp = studio.get_object_by_name("Channel Settings").get_component("PathSolve")
    comp.transmitter, comp.receiver = "tx", "rx"
    comp.num_samples = 2000
    result = PathSolveRunner.run(studio, comp)
    assert result.vertices is not None
    lines = PathLineBuilder.build(result)
    assert len(lines) >= 1   # at least the LoS path
