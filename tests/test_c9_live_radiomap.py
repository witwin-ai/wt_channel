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
    monkeypatch.setattr(radiomap_mod, "Notifications", SimpleNamespace(
        success=lambda *a: None, error=lambda *a: None))
    return SimpleNamespace(calls=calls, failures=failures)


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
    explicit, preview, refine = solver_api.calls
    assert explicit["scene_sent"] and explicit["scene_key"]
    assert explicit["component"]["solver"] == "deterministic"

    assert not preview["scene_sent"]
    assert preview["scene_key"] == explicit["scene_key"]
    assert preview["tx_position"] == [12.0, 30.0, 0.0]
    assert preview["component"]["solver"] == "montecarlo"
    assert preview["component"]["mc_integrator"] == "basic"
    assert preview["component"]["mc_samples_per_tx"] == 262144

    assert not refine["scene_sent"]
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
