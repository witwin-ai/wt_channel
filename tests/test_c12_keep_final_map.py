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


def test_only_the_final_map_of_a_drag_is_written_and_it_replaces_the_old_one(solver_api, tmp_path):
    studio, tx, comp = _scene()
    _, _, volumes = _with_assets(studio, tmp_path)

    async def main():
        assert await comp.solve_for("Tx") == "Solve complete"
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())

    # Explicit solve, preview, coarse refine, full refine: one file, the latest full map.
    assert len(solver_api.calls) == 4
    assert [p.name for p in volumes.iterdir()] == [f"{_asset_id(comp)}.bin"]


def test_previews_stay_in_memory(solver_api, tmp_path):
    studio, tx, comp = _scene()
    _, _, volumes = _with_assets(studio, tmp_path)
    comp._persist_map = False

    comp._solver_result_handle = "h"
    comp.update_map()

    assert _asset_id(comp) and not volumes.exists()
