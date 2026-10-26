"""C12 — a solved radio map is still there after Studio restarts (CPU, fake solver API).

Every map Scene opened after a restart showed only the green terrain: the heatmap was
published as an in-memory binary asset, the saved Scene kept a reference to it, and
after the restart nothing answered that reference. The full-grid result is now written
to the project (one file per RadioMap, replaced by the next full result); live
previews stay in memory.
"""
import asyncio
from types import SimpleNamespace

from test_c9_live_radiomap import _move, _scene, solver_api  # noqa: F401  (fixture)


def _with_assets(studio, tmp_path):
    from witwin_server.features.assets.binary.service import BinaryAssetService

    volumes = tmp_path / "runtime" / "volumes"
    persistence = SimpleNamespace(runtime_volumes_dir=volumes, volume_candidate_dirs=lambda: [volumes])
    service = BinaryAssetService(persistence=persistence)
    studio._server = SimpleNamespace(handlers={"binary_assets": SimpleNamespace(service=service)})
    return service, persistence, volumes


def _asset_id(comp):
    return (comp._values.get("volume") or {}).get("assetId")


def test_a_solved_map_is_found_again_after_a_restart(solver_api, tmp_path):
    from witwin_server.features.assets.binary.service import BinaryAssetService

    studio, tx, comp = _scene()
    _, persistence, volumes = _with_assets(studio, tmp_path)

    assert asyncio.run(comp.solve_for("Tx")) == "Solve complete"

    asset = _asset_id(comp)
    restarted = BinaryAssetService(persistence=persistence)  # nothing in memory any more
    assert asset and restarted.resolve_file(asset) is not None
    assert [p.name for p in volumes.iterdir()] == [f"{asset}.bin"]


def test_only_full_maps_are_written_and_the_previous_one_is_kept_until_the_next(solver_api, tmp_path):
    # The saved Scene points at the previous file until the next autosave (every 30 s),
    # so deleting it right away left a dead reference if Studio closed in between.
    studio, tx, comp = _scene()
    _, _, volumes = _with_assets(studio, tmp_path)
    finals = []

    async def main():
        assert await comp.solve_for("Tx") == "Solve complete"
        finals.append(_asset_id(comp))
        _move(tx, 12.0)
        await comp._live.idle()
        finals.append(_asset_id(comp))
        assert await comp.solve_for("Tx") == "Solve complete"
        finals.append(_asset_id(comp))

    asyncio.run(main())

    # Two explicit solves and one drag (preview, coarse, full): three full maps written,
    # the newest two kept.
    assert len(solver_api.calls) == 5
    assert sorted(p.name for p in volumes.iterdir()) == sorted(f"{a}.bin" for a in finals[1:])


def test_previews_stay_in_memory(solver_api, tmp_path):
    studio, tx, comp = _scene()
    _, _, volumes = _with_assets(studio, tmp_path)
    comp._persist_map = False

    comp._solver_result_handle = "h"
    comp.update_map()

    assert _asset_id(comp) and not volumes.exists()


def test_a_cuda_error_is_retried_once_in_a_fresh_solver_host(solver_api, monkeypatch):
    import wt_channel.components.radiomap as radiomap_mod

    monkeypatch.setattr(radiomap_mod, "_CUDA_RETRY_DELAY_S", 0.0)
    studio, tx, comp = _scene()
    solver_api.failures.append("CUDA error: out of memory")

    async def main():
        assert await comp.solve_for("Tx") == "Solve complete"
        solver_api.failures.append("CUDA error: an illegal memory access was encountered")
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())

    explicit_failed, explicit, preview_failed, preview = solver_api.calls[:4]
    assert explicit_failed["scene_sent"] and explicit["scene_sent"]
    assert preview_failed["grid_shape"] == [64, 64] and preview["grid_shape"] == [64, 64]
    assert preview["scene_sent"]  # the fresh host has no cached Scene
    assert comp.live_status.startswith("Done")
