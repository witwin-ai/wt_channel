"""C11 — switching maps hands the previous map's GPU memory back (CPU, fake solver).

witwin-channel keeps up to 32 compiled scenes (RayD acceleration structures and edge
caches on the GPU) in a process-wide LRU. Going through ten maps in one Studio, the
solver host grew 2 -> 7.5 -> 10.7 -> 11.4 GB until MIT went "solver unresponsive" and
Sydney ran out of memory even at a reduced grid. The host now forgets the compiled
scenes of the previous map when a different Scene arrives, and on an out-of-memory
error it forgets them and retries once.
"""
from types import SimpleNamespace

import numpy as np
import pytest


class _Ctx:
    def progress(self, *args, **kwargs):
        pass


def _studio_with_tx():
    from witwin_server.api import Scene as StudioScene, SceneObject
    from wt_channel.components.endpoints import TxComponent

    studio = StudioScene()
    tx = SceneObject(name="Radio Map Tx", mesh_type="Empty")
    tx.add_component(TxComponent())
    studio.add_object(tx)
    return studio


def _config(key, **extra):
    return {"mode": "radiomap", "transmitter": "Radio Map Tx", "receiver": "Radio Map",
            "component": {"solver": "deterministic"}, "scene_key": key, **extra}


@pytest.fixture
def host(monkeypatch):
    import torch
    import witwin.channel.scene as channel_scene
    import wt_channel.solver_host as solver_host

    events = SimpleNamespace(cleared=0, runs=0, fail=0)

    def fake_run(scene, comp, *, transmitter, receiver, grid_shape=None):
        events.runs += 1
        if events.fail:
            events.fail -= 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return SimpleNamespace(metrics=lambda: {"path_gain": np.ones((1, 2, 2), np.float32)})

    def clear():
        events.cleared += 1

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    monkeypatch.setattr(channel_scene, "clear_compile_cache", clear)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    solver_host._SCENES.clear()
    monkeypatch.setattr(solver_host, "_LAST_SCENE_KEY", None)
    monkeypatch.setattr(solver_host, "_exit_after_reply", lambda: None)  # never exit pytest
    return SimpleNamespace(module=solver_host, events=events)


def test_a_different_map_forgets_the_previous_compiled_scenes(host):
    host.module.solve(_Ctx(), _studio_with_tx(), _config("ucsd"))
    host.module.solve(_Ctx(), None, _config("ucsd", tx_position=[1.0, 30.0, 0.0]))
    assert host.events.cleared == 0  # dragging the Tx on one map keeps its compiled scene

    host.module.solve(_Ctx(), _studio_with_tx(), _config("mit"))
    assert host.events.cleared == 1


def test_out_of_memory_forgets_compiled_scenes_and_retries_once(host):
    host.events.fail = 1

    cache = host.module.solve(_Ctx(), _studio_with_tx(), _config("sydney"))

    assert host.events.runs == 2 and host.events.cleared == 1
    assert cache.metrics["path_gain"].shape == (1, 2, 2)


def test_a_second_out_of_memory_is_reported(host):
    import torch

    host.events.fail = 2

    with pytest.raises(torch.cuda.OutOfMemoryError):
        host.module.solve(_Ctx(), _studio_with_tx(), _config("sydney"))
    assert host.events.runs == 2


def test_each_solve_keeps_only_the_newest_compiled_scene(host, monkeypatch):
    # Every live preview builds a new Core world (the Tx moved), and witwin-channel kept
    # each compiled world: on MIT the allocated GPU memory grew 1.46 GB per preview
    # (1.5 -> 8.8 GB over one drag). The newest one stays, so the next preview still
    # reuses its RayD acceleration structure.
    from witwin.channel.scene import compiler

    registry = compiler._CompileRegistry()
    monkeypatch.setattr(compiler, "_REGISTRY", registry)
    for i in range(3):
        registry.put(("older", i), object())

    host.module.solve(_Ctx(), _studio_with_tx(), _config("mit"))

    assert list(registry._compiled) == [("older", 2)]


def test_a_failed_solve_lets_go_of_its_gpu_tensors(host, monkeypatch):
    # Princeton ran out of memory at 39 s and the solver host kept all 16 GB: the
    # error carried the failed attempt's frames (and their tensors) out of solve(), so
    # every later solve - other maps, Monte Carlo retries - ran out of memory too.
    import gc
    import weakref

    held = []

    class Big:
        pass

    def failing_run(scene, comp, *, transmitter, receiver, grid_shape=None):
        big = Big()
        held.append(weakref.ref(big))
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(host.module.SolveRunner, "run", staticmethod(failing_run))

    with pytest.raises(RuntimeError, match="illegal memory access") as caught:
        host.module.solve(_Ctx(), _studio_with_tx(), _config("princeton"))
    gc.collect()

    assert held[0]() is None  # nothing keeps the failed attempt's locals alive
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert host.events.cleared == 1  # and the compiled scenes are dropped


def test_a_cuda_error_restarts_the_solver_host(host, monkeypatch):
    # After "CUDA error: out of memory" on Princeton every later solve in the same
    # process failed the same way, even 64^2 previews on other maps: the CUDA context
    # was broken. The host now exits after reporting it; Studio starts a fresh one.
    restarts = []
    monkeypatch.setattr(host.module, "_exit_after_reply", lambda: restarts.append(1))

    def broken(scene, comp, *, transmitter, receiver, grid_shape=None):
        raise RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported")

    monkeypatch.setattr(host.module.SolveRunner, "run", staticmethod(broken))
    with pytest.raises(RuntimeError, match="CUDA error"):
        host.module.solve(_Ctx(), _studio_with_tx(), _config("princeton"))
    assert restarts == [1]

    def other(scene, comp, *, transmitter, receiver, grid_shape=None):
        raise ValueError("transmitter 'Tx' is not in the scene")

    monkeypatch.setattr(host.module.SolveRunner, "run", staticmethod(other))
    with pytest.raises(ValueError):
        host.module.solve(_Ctx(), _studio_with_tx(), _config("ucsd"))
    assert restarts == [1]  # other errors keep the host


def test_a_cuda_error_while_switching_maps_also_restarts_the_host(host, monkeypatch):
    # The broken context surfaced in torch.cuda.empty_cache() while switching from
    # Princeton to MIT, before the solve itself, and the host kept running broken.
    restarts = []
    monkeypatch.setattr(host.module, "_exit_after_reply", lambda: restarts.append(1))
    host.module.solve(_Ctx(), _studio_with_tx(), _config("princeton"))

    import witwin.channel.scene as channel_scene

    def broken():
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(channel_scene, "clear_compile_cache", broken)
    with pytest.raises(RuntimeError, match="CUDA error"):
        host.module.solve(_Ctx(), _studio_with_tx(), _config("mit"))
    assert restarts == [1]
