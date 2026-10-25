"""C6 — the ``witwin.channel.run_radiomap`` Agent tool (CPU, fake solver).

One sentence ("show me the UCSD map and run a radio map") must end in a solved radio
map over whatever geometry is in the Scene (typically an imported street map). The
tool plans the grid from the Scene geometry (horizontal extent, local ground height,
a transmitter above the local rooftops), creates or reuses its own Tx + RadioMap
objects, solves through the same RadioMap path as the Solve button, and reports
value statistics. The real GPU solve is covered by the C3 live tests.
"""
import asyncio
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _terrain_and_building():
    """A 200 m sloped terrain (y rises 0 -> 20 m along +x) and a 30 m tall box at the center."""
    xs = np.linspace(-100.0, 100.0, 21)
    zs = np.linspace(-100.0, 100.0, 21)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    terrain = np.stack([gx, 0.1 * (gx + 100.0), gz], axis=-1).reshape(-1, 3)
    building = np.array([[x, y, z] for x in (-5.0, 5.0) for y in (10.0, 40.0) for z in (-5.0, 5.0)])
    return np.concatenate([terrain, building]).astype(np.float32)


def test_plan_covers_scene_extent_at_local_ground_height():
    from wt_channel.agent_tools import plan_radiomap

    plan = plan_radiomap(_terrain_and_building(), rx_height_m=1.5, tx_height_m=25.0)

    np.testing.assert_allclose(plan["center"][0], 0.0, atol=1e-4)
    np.testing.assert_allclose(plan["center"][2], 0.0, atol=1e-4)
    assert plan["size_m"] == pytest.approx(200.0)
    # Local ground at the center is y=10 (terrain), not the global minimum y=0.
    assert plan["plane_y"] == pytest.approx(11.5)
    # The Tx clears the 40 m roof at the center instead of sitting inside the building.
    assert plan["tx_position"][1] >= 43.0
    np.testing.assert_allclose([plan["tx_position"][0], plan["tx_position"][2]], [0.0, 0.0], atol=1e-4)


def test_plan_honours_explicit_center_and_size():
    from wt_channel.agent_tools import plan_radiomap

    plan = plan_radiomap(_terrain_and_building(), center_xz=(50.0, -20.0), size_m=40.0,
                         rx_height_m=2.0, tx_height_m=10.0)

    assert plan["center"][0] == pytest.approx(50.0)
    assert plan["center"][2] == pytest.approx(-20.0)
    assert plan["size_m"] == pytest.approx(40.0)
    assert plan["plane_y"] == pytest.approx(15.0 + 2.0)  # terrain y at x=50 is 15
    assert plan["tx_position"][1] == pytest.approx(15.0 + 10.0)


def test_plan_without_geometry_raises():
    from wt_channel.agent_tools import plan_radiomap

    with pytest.raises(ValueError, match="no geometry"):
        plan_radiomap(np.zeros((0, 3), np.float32))


def _studio_with_map():
    from witwin_server import Scene as StudioScene, SceneObject

    studio = StudioScene()
    terrain = SceneObject(name="Map Terrain", mesh_type="Custom")
    verts = np.array([[-50, 0, -50], [50, 0, -50], [50, 0, 50], [-50, 0, 50]], np.float32)
    terrain.get_component("Mesh").set_mesh_data(
        verts, np.array([[0, 2, 1], [0, 3, 2]], np.uint32), notify=False)
    studio.add_object(terrain)
    return studio


def test_ensure_objects_creates_then_reuses_tx_and_map():
    from wt_channel.agent_tools import ensure_radiomap_objects, plan_radiomap, scene_vertices

    studio = _studio_with_map()
    plan = plan_radiomap(scene_vertices(studio), tx_height_m=20.0)

    tx, rmap = ensure_radiomap_objects(studio, plan, frequency_hz=2.4e9, resolution=64,
                                       samples=4096, max_bounces=1)
    again_tx, again_map = ensure_radiomap_objects(studio, plan, frequency_hz=3.5e9, resolution=32,
                                                  samples=4096, max_bounces=1)

    assert (again_tx.id, again_map.id) == (tx.id, rmap.id)
    assert sum(1 for o in studio.objects.values() if o.get_component("ChannelTx")) == 1
    tx_comp = tx.get_component("ChannelTx")
    np.testing.assert_allclose(list(tx_comp.polarization), [0.0, 1.0, 0.0])  # vertical (Y up)
    comp = rmap.get_component("RadioMap")
    assert float(comp.frequency) == pytest.approx(3.5e9)
    np.testing.assert_allclose(list(comp.resolution), [32.0, 1.0, 32.0])
    np.testing.assert_allclose(list(comp.min_bound), [-50.0, 0.0, -50.0])
    np.testing.assert_allclose(list(comp.max_bound), [50.0, 0.0, 50.0])
    assert str(comp.solver) == "montecarlo"
    np.testing.assert_allclose(list(rmap.get_component("Transform").position), [0.0, 1.5, 0.0])


def test_scene_vertices_ignore_channel_objects():
    from wt_channel.agent_tools import ensure_radiomap_objects, plan_radiomap, scene_vertices

    studio = _studio_with_map()
    before = scene_vertices(studio)
    ensure_radiomap_objects(studio, plan_radiomap(before), frequency_hz=3.5e9, resolution=16,
                            samples=1024, max_bounces=1)

    np.testing.assert_allclose(scene_vertices(studio), before)


class _FakeTools:
    def __init__(self):
        self.registered = []

    def register(self, tool):
        self.registered.append(tool)


def _registered_tool(studio):
    from wt_channel import agent_tools

    tools = _FakeTools()
    agent_tools.register(SimpleNamespace(tools=tools, api=SimpleNamespace(scene=studio)))
    assert [t.name for t in tools.registered] == ["run_radiomap"]
    return tools.registered[0]


def test_tool_metadata_names_radio_map_wording():
    tool = _registered_tool(_studio_with_map())

    assert tool.permission_tier == "scene_write"
    assert tool.side_effects is True
    text = tool.description.lower()
    for phrase in ("radio map", "coverage", "signal strength", "import_map"):
        assert phrase in text
    assert tool.timeout >= 600


def _fake_solver(monkeypatch, data):
    import wt_channel.components.radiomap as radiomap_mod

    calls = {}

    class FakeRun:
        status = "succeeded"
        outputs = {"resultHandle": "h"}
        run_id = "r"
        error = None

    class FakeSolvers:
        def solve(self, solver_id, **kwargs):
            calls["solve"] = {"solver_id": solver_id, **kwargs}
            return FakeRun()

        def query(self, solver_id, handle, op, params, **kwargs):
            return {"data": {"data": data, "extent": [0, 1, 0, 1], "label": "path_gain (dB)"}}

    monkeypatch.setattr(radiomap_mod, "api", SimpleNamespace(solvers=FakeSolvers()))
    monkeypatch.setattr(radiomap_mod, "Notifications", SimpleNamespace(
        success=lambda *a: None, error=lambda *a: calls.setdefault("error", a)))
    return calls


def test_tool_solves_and_reports_statistics(monkeypatch):
    studio = _studio_with_map()
    data = np.array([[-60.0, -80.0], [-100.0, -300.0]], np.float32)
    calls = _fake_solver(monkeypatch, data)

    result = asyncio.run(_registered_tool(studio).run({"frequency_ghz": 2.4, "resolution": 2}))

    assert "error" not in calls, calls.get("error")
    assert calls["solve"]["solver_id"] == "witwin.channel.radiomap"
    assert calls["solve"]["config"]["transmitter"] == result["transmitter"]["name"]
    assert calls["solve"]["config"]["component"]["frequency"] == pytest.approx(2.4e9)
    assert result["ok"] is True
    assert result["radiomap"]["object_id"]
    assert result["radiomap"]["grid"] == [2, 2]
    stats = result["path_gain_db"]
    assert stats["max"] == pytest.approx(-60.0)
    assert stats["min"] == pytest.approx(-100.0)  # over covered cells only
    assert stats["median"] == pytest.approx(-80.0)
    assert stats["covered_fraction"] == pytest.approx(0.75)  # cells above -250 dB floor
    assert result["solve_seconds"] >= 0.0


def test_tool_without_geometry_asks_for_a_map(monkeypatch):
    from witwin_server import Scene as StudioScene
    from witwin_server.tools.base import ToolError

    _fake_solver(monkeypatch, np.zeros((1, 1), np.float32))
    with pytest.raises(ToolError, match="import_map"):
        asyncio.run(_registered_tool(StudioScene()).run({}))


def test_tool_reports_solver_failure(monkeypatch):
    import wt_channel.components.radiomap as radiomap_mod
    from witwin_server.tools.base import ToolError

    class FailedRun:
        status = "failed"
        outputs = {}
        run_id = "r"
        error = {"message": "out of memory"}

    monkeypatch.setattr(radiomap_mod, "api", SimpleNamespace(
        solvers=SimpleNamespace(solve=lambda *a, **k: FailedRun())))
    monkeypatch.setattr(radiomap_mod, "Notifications", SimpleNamespace(
        success=lambda *a: None, error=lambda *a: None))

    with pytest.raises(ToolError, match="out of memory"):
        asyncio.run(_registered_tool(_studio_with_map()).run({}))


def test_manifest_contributes_tool_and_skill():
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "witwin.toml").read_text())

    assert {"module": "agent_tools", "register": "register"} in manifest["contributes"]["tools"]
    assert manifest["contributes"]["skills"] == [{"root": "skills"}]
    assert "scene_write" in manifest["permissions"]["tools"]
    skill = (root / "skills" / "radio-map" / "SKILL.md").read_text(encoding="utf-8")
    assert "witwin.channel.run_radiomap" in skill
    assert "witwin.streetmap.import_map" in skill
