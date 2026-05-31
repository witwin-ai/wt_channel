"""C0 — scene config + structures round-trip (CPU, scalar materials).

Builds a ``channel.Scene`` with a frequency and three structures (scalar materials,
priorities, tags), runs platform -> studio -> platform, and asserts the rebuilt scene
reproduces the frequency, device, and each structure's geometry args, material, and
metadata.
"""
import numpy as np
import pytest
from witwin.core import Box, Cylinder, Material, Sphere, Structure

from _helpers import RT


@pytest.fixture
def scene(device):
    import witwin.channel as ch
    return ch.Scene(
        structures=[
            Structure(Box(size=(4.0, 4.0, 0.2), position=(0.0, 0.0, 0.0)),
                      Material(eps_r=5.24, sigma_e=0.12, name="ground"),
                      name="ground", priority=3, tags=("floor", "static")),
            Structure(Sphere(radius=1.5, position=(3.0, 0.0, 1.0)),
                      Material(eps_r=2.7, sigma_e=0.0, name="ball"),
                      name="ball", enabled=False),
            Structure(Cylinder(radius=0.4, height=3.0, axis="z", position=(-2.0, 1.0, 1.5)),
                      Material(eps_r=3.9, sigma_e=0.01), name="post"),
        ],
        frequency=3.5e9, device="cpu",
    )


def test_scene_config_roundtrip(adapter, scene):
    rebuilt = RT.back(adapter, scene)
    assert rebuilt.frequency == pytest.approx(3.5e9)
    assert str(rebuilt.device) == "cpu"


def test_noise_power_roundtrip(adapter, device):
    import witwin.channel as ch
    scene = ch.Scene(structures=[Structure(Box(size=(1, 1, 1)), Material(), name="b")],
                     frequency=1e9, device="cpu")
    scene.noise_power = 2.5e-13
    rebuilt = RT.back(adapter, scene)
    assert float(rebuilt.noise_power) == pytest.approx(2.5e-13)


def test_structure_count_and_names(adapter, scene):
    rebuilt = RT.back(adapter, scene)
    assert RT.by_name(rebuilt.structures).keys() == {"ground", "ball", "post"}


def test_structure_geometry_roundtrip(adapter, scene):
    rebuilt = RT.by_name(RT.back(adapter, scene).structures)
    original = RT.by_name(scene.structures)
    for name in original:
        RT.assert_geom(rebuilt[name].geometry, original[name].geometry)


def test_structure_material_roundtrip(adapter, scene):
    rebuilt = RT.by_name(RT.back(adapter, scene).structures)
    original = RT.by_name(scene.structures)
    for name in original:
        RT.assert_material(rebuilt[name].material, original[name].material)


def test_structure_metadata_roundtrip(adapter, scene):
    rebuilt = RT.by_name(RT.back(adapter, scene).structures)
    assert int(rebuilt["ground"].priority) == 3
    assert set(rebuilt["ground"].tags) == {"floor", "static"}
    assert rebuilt["ball"].enabled is False
    assert rebuilt["post"].enabled is True


def test_export_empty_scene_is_empty(adapter):
    # Config now rides RadioMap / PathSolver; an empty studio scene exports an empty
    # channel scene (default frequency) rather than raising.
    from witwin_server import Scene as StudioScene
    empty = StudioScene()
    scene = adapter.to_platform(empty)
    assert list(scene.structures) == []
    assert list(scene.transmitters) == []
    assert list(scene.receivers) == []
