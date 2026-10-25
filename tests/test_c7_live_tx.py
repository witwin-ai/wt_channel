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

    def fake_run(scene, comp, *, transmitter, receiver, grid_shape=None):
        obj = next(o for o in scene.objects.values() if o.name == transmitter)
        seen.append((scene, [float(v) for v in obj.get_component("Transform").position]))
        return SimpleNamespace(metrics=lambda: {"path_gain": np.ones((1, 2, 2), np.float32)})

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    solver_host._SCENES.clear()
    monkeypatch.setattr(solver_host, "_LAST_SCENE_KEY", None)
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


def test_deterministic_solves_release_the_torch_cache_around_them(host, monkeypatch):
    # The deterministic UCSD solve peaks near 14 GB of a 16 GB GPU. RayD/OptiX allocate
    # outside torch, so blocks torch keeps cached from the previous solve (or from the
    # Monte Carlo previews) push the next deterministic solve past physical memory:
    # 10 s became 90-160 s and the Studio solver host was killed as unresponsive.
    import torch

    released = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(len(host.seen)))
    studio = _studio_with_tx()

    host.module.solve(_Ctx(), studio, _config(scene_key="k", component={"solver": "montecarlo"}))
    assert released == []  # previews keep the cache warm
    host.module.solve(_Ctx(), None, _config(scene_key="k", tx_position=[1, 30, 0],
                                            component={"solver": "deterministic"}))
    assert released == [1, 2]  # before and after the deterministic solve


def test_solver_host_passes_a_grid_shape_override_through(host, monkeypatch):
    shapes = []
    original = host.module.SolveRunner.run

    def run(scene, comp, *, transmitter, receiver, grid_shape=None):
        shapes.append(grid_shape)
        return original(scene, comp, transmitter=transmitter, receiver=receiver)

    monkeypatch.setattr(host.module.SolveRunner, "run", staticmethod(run))
    studio = _studio_with_tx()
    host.module.solve(_Ctx(), studio, _config(scene_key="k"))
    host.module.solve(_Ctx(), None, _config(scene_key="k", grid_shape=[64, 64]))

    assert shapes == [None, (64, 64)]


def test_coarse_deterministic_previews_keep_the_torch_cache(host, monkeypatch):
    # Emptying the cache costs a re-allocation; only solves big enough to crowd the
    # GPU (128^2 and up on UCSD) release it.
    import torch

    released = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(1))
    studio = _studio_with_tx()
    host.module.solve(_Ctx(), studio, _config(scene_key="k", grid_shape=[64, 64],
                                              component={"solver": "deterministic"}))
    assert released == []
    host.module.solve(_Ctx(), None, _config(scene_key="k", grid_shape=[128, 128],
                                            component={"solver": "deterministic"}))
    assert released == [1, 1]
