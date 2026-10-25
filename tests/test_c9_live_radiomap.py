"""C9 — the RadioMap follows a dragged Tx (CPU, fake solver API).

After the map has been solved once, moving its Tx re-solves without shipping the Scene:
first a Monte Carlo basic preview, then (once the Tx is still) one solve with the
component's own solver. Any other edit to the Scene makes the next solve resend it.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest


class _Run:
    def __init__(self, error=None):
        self.status = "failed" if error else "succeeded"
        self.outputs = {} if error else {"resultHandle": "h"}
        self.run_id = "r"
        self.error = {"message": error} if error else None


@pytest.fixture
def solver_api(monkeypatch):
    import wt_channel.components.radiomap as radiomap_mod

    calls = []
    failures = []

    class Solvers:
        def solve(self, solver_id, *, scene=None, config=None, on_submitted=None, **kwargs):
            calls.append({"scene_sent": scene is not None, **config})
            return _Run(failures.pop(0) if failures else None)

        def query(self, solver_id, handle, op, params, **kwargs):
            return {"data": {"data": np.full((2, 2), -90.0, np.float32), "extent": [0, 1, 0, 1]}}

        def cancel(self, run_id, reason=None):
            calls.append({"cancel": run_id})
            return True

    monkeypatch.setattr(radiomap_mod, "api", SimpleNamespace(solvers=Solvers()))
    toasts = []
    monkeypatch.setattr(radiomap_mod, "Notifications", SimpleNamespace(
        info=lambda *a: toasts.append(("info",) + a),
        success=lambda *a: toasts.append(("success",) + a),
        error=lambda *a: toasts.append(("error",) + a)))
    return SimpleNamespace(calls=calls, failures=failures, toasts=toasts)


def _scene():
    from witwin_server import Scene as StudioScene, SceneObject
    from wt_channel.components.endpoints import TxComponent
    from wt_channel.components.radiomap import RadioMapComponent

    studio = StudioScene()
    ground = SceneObject(name="Ground", mesh_type="Custom")
    ground.get_component("Mesh").set_mesh_data(
        np.array([[-50, 0, -50], [50, 0, -50], [50, 0, 50], [-50, 0, 50]], np.float32),
        np.array([[0, 2, 1], [0, 3, 2]], np.uint32), notify=False)
    studio.add_object(ground)
    tx = SceneObject(name="Tx", mesh_type="Empty")
    tx.add_component(TxComponent())
    tx.get_component("Transform").position = [0.0, 30.0, 0.0]
    studio.add_object(tx)
    rmap = SceneObject(name="Map", mesh_type="Empty")
    comp = rmap.add_component(RadioMapComponent())
    comp.solver = "deterministic"
    comp.live_settle_s = 0.01
    studio.add_object(rmap)
    return studio, tx, comp


def _move(tx, x):
    tx.get_component("Transform").position = [float(x), 30.0, 0.0]


def test_moving_the_tx_previews_then_refines_without_resending_the_scene(solver_api):
    async def main():
        studio, tx, comp = _scene()
        assert await comp.solve_for("Tx") == "Solve complete"
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())
    explicit, preview, coarse, refine = solver_api.calls
    assert explicit["scene_sent"] and explicit["scene_key"]
    assert explicit["component"]["solver"] == "deterministic"

    assert not preview["scene_sent"]
    assert preview["scene_key"] == explicit["scene_key"]
    assert preview["tx_position"] == [12.0, 30.0, 0.0]
    # Previews are deterministic on a coarse grid so diffraction shows while dragging
    # (UCSD 64^2, one reflection: ~0.4 s, 78 % of shadow cells lit; Monte Carlo basic at
    # 256k rays lit ~12 %).
    assert preview["component"]["solver"] == "deterministic"
    assert preview["grid_shape"] == [64, 64]
    assert preview["component"]["det_max_bounces"] == 1
    assert preview["component"]["det_max_diffraction_order"] == 1

    # The refine comes in two passes: a coarse grid first (UCSD: 128^2 in ~1.6 s), then
    # the RadioMap's own grid (256^2, ~6.5 s).
    assert coarse["grid_shape"] == [128, 128]
    assert coarse["component"]["solver"] == "deterministic"
    assert not refine["scene_sent"]
    assert "grid_shape" not in refine
    assert refine["tx_position"] == [12.0, 30.0, 0.0]
    assert refine["component"]["solver"] == "deterministic"


def test_live_update_off_leaves_the_map_alone(solver_api):
    async def main():
        studio, tx, comp = _scene()
        comp.live_update = False
        await comp.solve_for("Tx")
        _move(tx, 12.0)
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert len(solver_api.calls) == 1


def test_moves_before_the_first_solve_do_nothing(solver_api):
    async def main():
        studio, tx, comp = _scene()
        _move(tx, 12.0)
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert solver_api.calls == []


def test_editing_other_geometry_resends_the_scene_on_the_next_preview(solver_api):
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        studio.get_object_by_name("Ground").get_component("Transform").position = [0.0, -1.0, 0.0]
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())
    explicit, preview = solver_api.calls[:2]
    assert preview["scene_sent"]
    assert preview["scene_key"] != explicit["scene_key"]


def test_a_solver_cache_miss_resends_the_scene_once(solver_api):
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        solver_api.failures.append("scene_cache_miss: no cached scene for key 'x'")
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())
    missed, resent = solver_api.calls[1:3]
    assert not missed["scene_sent"]
    assert resent["scene_sent"]
    assert resent["tx_position"] == [12.0, 30.0, 0.0]


def test_an_explicit_solve_drops_pending_live_work(solver_api):
    # The Agent tool moves the Tx and then solves explicitly: that move must not also
    # start a preview + refine of its own.
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        _move(tx, 12.0)
        await comp.solve_for("Tx")
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert [c["scene_sent"] for c in solver_api.calls] == [True, True]


def test_the_tx_asks_the_viewport_to_stream_its_drag():
    # The Studio viewport streams a drag to the server only for objects whose component
    # value live_drag is true; without it the map would update only on mouse-up.
    from wt_channel.components.endpoints import TxComponent

    values = TxComponent().to_dict()["values"]
    assert values["live_drag"] is True


def test_radio_map_refreshes_never_toast(solver_api, monkeypatch):
    # Each map refresh publishes the plane volume and the core toasted "Volume Cached"
    # for every publish; the user asked for no progress / done toasts at all.
    from witwin_server.core.components.scalar_grid_resource import ScalarGridResourcePublisher
    import wt_channel.components.radiomap as radiomap_mod

    toasts = []
    monkeypatch.setattr(ScalarGridResourcePublisher, "_notify_volume_progress",
                        lambda self, *a, **k: toasts.append(a))
    monkeypatch.setattr(radiomap_mod.RadioMapComponent, "export_volume_data",
                        lambda self, progress_id=None: self._notify_volume_progress(
                            progress_id, "Cached", "cached", 0.2, "asset"))

    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        explicit = len(toasts)
        _move(tx, 12.0)
        await comp._live.idle()
        return explicit

    explicit = asyncio.run(main())
    assert explicit == 0
    assert toasts == []
    assert not [t for t in solver_api.toasts if t[0] != "error"]


def test_the_refine_after_a_drag_reports_through_live_status_only(solver_api):
    # The last preview lands a moment after the drag and looked final; Live Status says
    # when the refine runs and when it is done. The user asked for no toasts.
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        solver_api.toasts.clear()
        statuses = []
        comp.set_field = _recording_set_field(comp, statuses)
        _move(tx, 12.0)
        await comp._live.idle()
        return comp, statuses

    comp, statuses = asyncio.run(main())
    assert solver_api.toasts == []
    assert statuses[0].lower().startswith("previewing")
    assert any(s.lower().startswith("refining") for s in statuses)
    assert str(comp.live_status).lower().startswith("done")


def test_a_refine_cut_short_by_another_move_does_not_claim_done(solver_api):
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        solver_api.toasts.clear()
        stale = {"left": 1}
        original = comp._live_refine

        async def refine_then_move(is_current):
            if stale["left"]:
                stale["left"] -= 1
                _move(tx, 20.0)  # the Tx moves while this refine runs
            await original(is_current)

        statuses = []
        comp.set_field = _recording_set_field(comp, statuses)
        comp._live = None
        _move(tx, 12.0)
        comp._live._refine = refine_then_move
        await comp._live.idle()
        return statuses

    statuses = asyncio.run(main())
    # The cut-short refine says the Tx moved; only the refine at the final position is done.
    assert statuses.count("Tx moved: refining again once it stops") == 1
    assert statuses[-1].startswith("Done")


def _recording_set_field(comp, statuses):
    original = comp.set_field

    def set_field(name, value, *args, **kwargs):
        if name == "live_status":
            statuses.append(str(value))
        return original(name, value, *args, **kwargs)

    return set_field


def test_a_map_no_bigger_than_the_coarse_grid_refines_once(solver_api):
    async def main():
        studio, tx, comp = _scene()
        comp.resolution = [64.0, 1.0, 64.0]
        await comp.solve_for("Tx")
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())
    refines = [c for c in solver_api.calls[2:] if c["component"]["solver"] == "deterministic"]
    assert [c.get("grid_shape") for c in refines] == [None]


def test_a_coarse_map_is_shown_on_the_radiomap_grid(solver_api, monkeypatch):
    # set_volume_data takes its resolution from the data; a coarse result must not
    # shrink the RadioMap's own grid, or the next full solve would run coarse too.
    import wt_channel.components.radiomap as radiomap_mod

    shown = []
    monkeypatch.setattr(radiomap_mod.RadioMapComponent, "set_volume_data",
                        lambda self, vol: shown.append(np.asarray(vol).shape))
    monkeypatch.setattr(radiomap_mod.RadioMapComponent, "export_volume_data",
                        lambda self, progress_id=None: None)

    async def main():
        studio, tx, comp = _scene()
        comp.resolution = [4.0, 1.0, 4.0]
        await comp.solve_for("Tx")  # the fake solver returns a 2x2 map
        return comp

    comp = asyncio.run(main())
    assert shown == [(4, 1, 4)]
    np.testing.assert_allclose(list(comp.resolution), [4.0, 1.0, 4.0])


def test_monte_carlo_previews_stay_available(solver_api):
    async def main():
        studio, tx, comp = _scene()
        comp.live_preview = "montecarlo"
        await comp.solve_for("Tx")
        _move(tx, 12.0)
        await comp._live.idle()

    asyncio.run(main())
    preview = solver_api.calls[1]
    assert preview["component"]["solver"] == "montecarlo"
    assert preview["component"]["mc_integrator"] == "basic"
    assert preview["component"]["mc_samples_per_tx"] == 262144
    assert "grid_shape" not in preview


def test_a_tx_dragged_below_the_ground_is_not_solved(solver_api):
    # A Tx dragged 70 m under the University of Michigan terrain made the deterministic
    # refine run out of GPU memory; nothing on screen said why. Below the ground there
    # is nothing to solve: Live Status says so and no solve is submitted.
    async def main():
        studio, tx, comp = _scene()
        await comp.solve_for("Tx")
        sent = len(solver_api.calls)
        tx.get_component("Transform").position = [12.0, -5.0, 0.0]
        await comp._live.idle()
        return comp, sent

    comp, sent = asyncio.run(main())
    assert len(solver_api.calls) == sent
    assert "below the ground" in str(comp.live_status)


def test_solve_refuses_a_tx_below_the_ground(solver_api):
    async def main():
        studio, tx, comp = _scene()
        tx.get_component("Transform").position = [0.0, -5.0, 0.0]
        return await comp.solve_for("Tx")

    message = asyncio.run(main())
    assert "below the ground" in message
    assert solver_api.calls == []
    assert any(kind == "error" for kind, *_ in solver_api.toasts)
