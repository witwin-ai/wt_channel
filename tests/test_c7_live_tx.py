"""C7 — live radio map while the Tx is dragged (CPU, fake solver).

Moving only the Tx must not ship the whole Scene again: on UCSD the Scene document is
11 MB and serializing it both ways costs ~0.7 s, while a Monte Carlo basic preview at
256k rays solves in ~0.2 s. The solver host keeps the last Scene it received under a
``scene_key``; a preview sends no Scene, only that key and the Tx position.
"""
from types import SimpleNamespace

import numpy as np
import pytest


class _Ctx:
    def progress(self, *args, **kwargs):
        pass


def _studio_with_tx(position=(0.0, 30.0, 0.0)):
    from witwin_server import Scene as StudioScene, SceneObject
    from wt_channel.components.endpoints import TxComponent

    studio = StudioScene()
    tx = SceneObject(name="Radio Map Tx", mesh_type="Empty")
    tx.add_component(TxComponent())
    tx.get_component("Transform").position = list(position)
    studio.add_object(tx)
    return studio


@pytest.fixture
def host(monkeypatch):
    import wt_channel.solver_host as solver_host

    seen = []

    def fake_run(scene, comp, *, transmitter, receiver):
        obj = next(o for o in scene.objects.values() if o.name == transmitter)
        seen.append((scene, [float(v) for v in obj.get_component("Transform").position]))
        return SimpleNamespace(metrics=lambda: {"path_gain": np.ones((1, 2, 2), np.float32)})

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    solver_host._SCENES.clear()
    return SimpleNamespace(module=solver_host, seen=seen)


def _config(**extra):
    return {"mode": "radiomap", "transmitter": "Radio Map Tx", "receiver": "Radio Map",
            "component": {}, **extra}


def test_solver_host_reuses_the_cached_scene_and_moves_only_the_tx(host):
    studio = _studio_with_tx()

    host.module.solve(_Ctx(), studio, _config(scene_key="k1"))
    host.module.solve(_Ctx(), None, _config(scene_key="k1", tx_position=[5.0, 31.0, -2.0]))

    (first_scene, first_pos), (second_scene, second_pos) = host.seen
    assert second_scene is first_scene  # no Scene shipped, the cached one solved
    assert first_pos == [0.0, 30.0, 0.0]
    assert second_pos == [5.0, 31.0, -2.0]


def test_solver_host_reports_a_cache_miss_for_an_unknown_key(host):
    with pytest.raises(LookupError, match="scene_cache_miss"):
        host.module.solve(_Ctx(), None, _config(scene_key="never-sent", tx_position=[0, 0, 0]))


def test_solver_host_keeps_only_the_latest_scenes(host):
    for i in range(5):
        host.module.solve(_Ctx(), _studio_with_tx(), _config(scene_key=f"k{i}"))

    assert len(host.module._SCENES) <= 2
    assert "k4" in host.module._SCENES
