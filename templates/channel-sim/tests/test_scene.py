import importlib.util
from pathlib import Path

import witwin.studio as ws

PROJECT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_channel", PROJECT / "scripts" / "prepare_channel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_has_every_part_the_solvers_need():
    scene = ws.open_project(PROJECT).scene()
    script = load_script()
    for component in ("ChannelTx", "ChannelRx", "RadioMap", "PathSolver"):
        script.find(scene, component)


def test_prepare_script_reports_ready(capsys):
    assert load_script().main([]) == 0
    assert "Ready to solve" in capsys.readouterr().out
