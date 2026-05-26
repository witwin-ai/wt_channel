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


def test_build_batch_packs_all_paths_into_one_segment_buffer():
    """RB4: vectorized batch packs N paths' segments into flat numpy arrays."""
    from wt_channel.adapter.path_solve import PathLineBuilder

    batch = PathLineBuilder.build_batch(_synthetic_result())
    # LoS path -> 1 segment (Tx->Rx); 1-reflection path -> 2 segments (Tx->v0->Rx).
    assert batch.segment_count == 3
    assert batch.segments.shape == (3, 2, 3)
    assert batch.segments.dtype == np.float32
    assert batch.colors_type.shape == (3, 4)
    assert batch.colors_type.dtype == np.uint8
    assert batch.colors_gain.shape == (3, 4)
    assert batch.types.shape == (3,)
    assert batch.path_index.shape == (3,)

    # The LoS path is path 0 (1 segment), reflection is path 1 (2 segments).
    np.testing.assert_array_equal(batch.path_index, [0, 1, 1])
    # First segment is the LoS Tx->Rx exactly.
    np.testing.assert_allclose(batch.segments[0, 0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(batch.segments[0, 1], [6.0, 0.0, 1.5])
    # Reflection: Tx -> bounce (3, 0, 1) -> Rx, emitted as two segments.
    np.testing.assert_allclose(batch.segments[1, 0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(batch.segments[1, 1], [3.0, 0.0, 1.0])
    np.testing.assert_allclose(batch.segments[2, 0], [3.0, 0.0, 1.0])
    np.testing.assert_allclose(batch.segments[2, 1], [6.0, 0.0, 1.5])

    # Per-segment type code matches the owning path's classification
    # (LoS=0, reflection=1).
    np.testing.assert_array_equal(batch.types, [0, 1, 1])
    # Same classification -> same RGBA across both reflection segments.
    np.testing.assert_array_equal(batch.colors_type[1], batch.colors_type[2])
    # Different class -> different RGBA from LoS.
    assert not np.array_equal(batch.colors_type[0], batch.colors_type[1])


def test_build_batch_empty_result_returns_empty_batch():
    from wt_channel.adapter.path_solve import PathLineBuilder, RayBatch

    class _NoGeom:
        vertices = None

    batch = PathLineBuilder.build_batch(_NoGeom())
    assert isinstance(batch, RayBatch)
    assert batch.empty
    assert batch.segment_count == 0


def test_build_batch_gain_colormap_varies_across_amplitudes():
    """The gain channel must produce *different* colors for the two paths
    (their amplitudes differ: 1.0 vs 0.5)."""
    from wt_channel.adapter.path_solve import PathLineBuilder

    batch = PathLineBuilder.build_batch(_synthetic_result(),
                                        gain_db_range=(-20.0, 0.0))
    # All segments of the LoS path share its gain color; same for reflection;
    # the two paths' gain colors must differ.
    los_color = batch.colors_gain[0]
    refl_color = batch.colors_gain[1]
    np.testing.assert_array_equal(batch.colors_gain[2], refl_color)
    assert not np.array_equal(los_color, refl_color)


def test_build_matches_build_batch_geometry():
    """Legacy PolyLine output must agree with the batch on per-path geometry."""
    from wt_channel.adapter.path_solve import PathLineBuilder

    result = _synthetic_result()
    lines = PathLineBuilder.build(result)
    batch = PathLineBuilder.build_batch(result)

    # Reconstruct each path's point sequence from the batch and compare.
    by_path = {}
    for i in range(batch.segment_count):
        by_path.setdefault(int(batch.path_index[i]), []).append(batch.segments[i])

    for p_idx, segs in by_path.items():
        # The ordered point sequence is segs[0][0], segs[0][1] == segs[1][0], segs[1][1], ...
        seq = [segs[0][0].tolist()]
        for s in segs:
            seq.append(s[1].tolist())
        np.testing.assert_allclose(seq, lines[p_idx].positions)


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


def test_path_solver_draw_uses_single_anchor_and_ray_batch_spec(monkeypatch):
    """RB5: _draw creates ONE hidden anchor + pushes a ray_batch RenderSpec.

    No per-path SceneObject; the asset travels via the ray_batch_assets service
    (mocked here so the test doesn't need the live server bound).
    """
    from witwin_server.core import Scene
    from witwin_server import api
    from wt_channel.components.pathsolver import PathSolverComponent

    scene = Scene()
    # Host the component on its own SceneObject so _owner.scene resolves.
    from witwin_server.core import SceneObject as SO
    host = SO(name="Path Solver", mesh_type="Empty")
    scene.add_object(host)
    comp = PathSolverComponent()
    comp._owner = host
    host._components["PathSolver"] = comp
    comp._result = _synthetic_result()

    pushed = {}

    class _FakeViewport:
        def register_ray_batch(self, segments, *, colors_type=None, colors_gain=None,
                               types=None, scene_id="world"):
            pushed["asset"] = {
                "segments": segments,
                "colors_type": colors_type,
                "colors_gain": colors_gain,
                "types": types,
            }
            return {
                "asset_id": "r_test_fake",
                "uri": f"/scenes/{scene_id}/assets/raybatch/r_test_fake.raybatchbin",
                "format": "witwin.raybatch.v1",
                "content_hash": "sha256:fake",
                "segment_count": int(segments.shape[0]),
                "bounds": [0, 0, 0, 1, 1, 1],
            }

        def set_render_spec(self, object_id, spec):
            pushed["object_id"] = object_id
            pushed["spec"] = spec

        def clear_render_spec(self, object_id):
            pushed.setdefault("cleared", []).append(object_id)

    monkeypatch.setattr(api, "_viewport", _FakeViewport())

    drawn = comp._draw(scene)
    assert drawn == 3  # 1 LoS seg + 2 reflection segs (synthetic 2-path result)

    # Exactly one anchor SceneObject got created, and it's hidden from the user.
    anchors = [o for o in scene.objects.values() if o.tag.startswith("wt_channel:ray_batch:")]
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.system is True
    assert anchor.disable_transform is True
    assert anchor.allow_add_components is False

    # The spec is a ray_batch RenderSpec with the asset handle (binary), not
    # inline positions; this is the whole point of the bulk channel.
    spec = pushed["spec"]
    assert spec["geometry"]["kind"] == "ray_batch"
    assert spec["geometry"]["segment_count"] == 3
    assert spec["geometry"]["binary"]["asset_id"] == "r_test_fake"
    assert spec["geometry"]["color_channel"] == "type"
    assert "positions" not in spec["geometry"]  # bulk data NOT inline

    # A second _draw reuses the SAME anchor (no scene clutter on re-solve).
    comp._draw(scene)
    assert pushed["object_id"] == anchor.id
    assert len([o for o in scene.objects.values()
                if o.tag.startswith("wt_channel:ray_batch:")]) == 1


def test_path_solver_color_mode_switches_channel(monkeypatch):
    """RB6: color_mode='gain' flips the spec's color_channel + enables additive."""
    from witwin_server.core import Scene, SceneObject as SO
    from witwin_server import api
    from wt_channel.components.pathsolver import PathSolverComponent

    scene = Scene()
    host = SO(name="Path Solver", mesh_type="Empty")
    scene.add_object(host)
    comp = PathSolverComponent()
    comp._owner = host
    host._components["PathSolver"] = comp
    comp._result = _synthetic_result()
    comp.color_mode = "gain"

    pushed = {}

    class _FakeViewport:
        def register_ray_batch(self, segments, **kwargs):
            return {"asset_id": "r_gain", "uri": "/x", "format": "witwin.raybatch.v1",
                    "content_hash": "sha256:gain", "segment_count": int(segments.shape[0])}

        def set_render_spec(self, object_id, spec):
            pushed["spec"] = spec

        def clear_render_spec(self, object_id):
            pass

    monkeypatch.setattr(api, "_viewport", _FakeViewport())
    comp._draw(scene)

    assert pushed["spec"]["geometry"]["color_channel"] == "gain"
    assert pushed["spec"]["material"]["additive"] is True


def test_path_config_builder():
    import witwin.channel as ch
    from wt_channel.adapter.path_solve import PathConfigBuilder
    from wt_channel.components.pathsolver import PathSolverComponent

    comp = PathSolverComponent()
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

def test_live_path_solve(adapter, path_ready):
    """Round-trip then path-solve a discrete Tx<->Rx link; check geometry + CIR."""
    import witwin.channel as ch
    from wt_channel.adapter.path_solve import PathLineBuilder, PathSolveRunner

    # Keep the wall in the scene while leaving the Tx/Rx center line visible.
    scene = ch.Scene(
        structures=[Structure(Box(size=(0.3, 6.0, 6.0), position=(3.0, 4.0, 3.0)),
                              Material.from_itu("concrete"), name="wall")],
        transmitters=[ch.Transmitter(name="tx", position=(0.0, 0.0, 3.0))],
        receivers=[ch.Receiver(name="rx", position=(6.0, 0.0, 3.0))],
        frequency=3.5e9, device="cuda",
    )
    studio = adapter.to_studio(scene)
    comp = studio.get_object_by_name("Path Solver").get_component("PathSolver")
    comp.num_samples = 2000
    result = PathSolveRunner.run(studio, comp, transmitter="tx", receiver="rx")
    assert result.vertices is not None
    lines = PathLineBuilder.build(result)
    assert len(lines) >= 1
    assert any(line.interaction == "los" for line in lines)
