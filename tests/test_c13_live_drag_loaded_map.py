"""C13 — dragging the Tx updates a radio map loaded from disk (CPU, fake solver API).

Since full maps are kept on disk, a map Scene reopened after a restart shows its
heatmap at once, but dragging its Tx did nothing (Eiffel Tower): live updates were
armed only by a solve in the current run. A RadioMap now watches its Scene from the
start and updates live as soon as it shows a map.
"""
import asyncio
from types import SimpleNamespace

from test_c9_live_radiomap import _move, _scene, solver_api  # noqa: F401  (fixture)


def test_dragging_the_tx_updates_a_map_loaded_from_disk(solver_api):
    async def main():
        from wt_channel.components.radiomap import watch_scene_radiomaps

        studio, tx, comp = _scene()
        comp._values["volume"] = {"assetId": "radiomap-saved"}  # as loaded from the Scene file
        watch_scene_radiomaps(studio)  # scene_loaded
        _move(tx, 12.0)
        assert comp._live is not None
        await comp._live.idle()

    asyncio.run(main())

    preview, coarse, refine = solver_api.calls
    assert preview["scene_sent"]  # nothing cached in the solver host yet
    assert preview["grid_shape"] == [64, 64] and preview["tx_position"] == [12.0, 30.0, 0.0]
    assert coarse["grid_shape"] == [128, 128]
    assert "grid_shape" not in refine and refine["tx_position"] == [12.0, 30.0, 0.0]


def test_a_radiomap_without_a_map_still_ignores_the_tx(solver_api):
    async def main():
        from wt_channel.components.radiomap import watch_scene_radiomaps

        studio, tx, comp = _scene()
        watch_scene_radiomaps(studio)
        _move(tx, 12.0)
        await asyncio.sleep(0.05)

    asyncio.run(main())

    assert solver_api.calls == []


def test_the_plugin_watches_loaded_and_already_open_scenes():
    from wt_channel import agent_tools
    from wt_channel.components.radiomap import watch_scene_radiomaps

    studio, tx, comp = _scene()
    subscribed = []
    ctx = SimpleNamespace(
        tools=SimpleNamespace(register=lambda tool: None),
        scene_lifecycle=SimpleNamespace(on=lambda event, fn: subscribed.append((event, fn))),
        api=SimpleNamespace(server=SimpleNamespace(scenes={"main": studio})),
    )

    agent_tools.register(ctx)

    assert subscribed == [("scene_loaded", watch_scene_radiomaps)]
    assert comp._live_scene is studio
