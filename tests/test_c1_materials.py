"""C1 — materials round-trip (explicit scalar + ITU presets).

Explicit scalar materials round-trip as ``core.Material`` (no preset companion). ITU
materials round-trip through their *identity* ``itu_descriptor`` ``(name, override)``
— reconstructed via ``Material.from_itu``, never via the evaluated eps_r/sigma_e — and
the preset's frequency band is validated on export.
"""
import pytest
from witwin.core import Box, Material, Structure

from _helpers import RT


def _scene(structure, *, frequency=3.5e9):
    import witwin.channel as ch
    return ch.Scene(structures=[structure], frequency=frequency, device="cpu")


def test_explicit_scalar_has_no_preset(adapter):
    scene = _scene(Structure(Box(size=(2, 2, 0.2)), Material(eps_r=4.5, sigma_e=0.02, name="wall"),
                             name="wall"))
    studio = adapter.to_studio(scene)
    obj = next(o for o in studio.objects.values() if o.name == "wall")
    assert obj.get_component("MaterialPreset") is None
    rebuilt = adapter.to_platform(studio).structures[0]
    assert getattr(rebuilt.material, "itu_descriptor", None) is None
    RT.assert_material(rebuilt.material, scene.structures[0].material)


def test_itu_preset_roundtrip_identity(adapter):
    scene = _scene(Structure(Box(size=(8, 8, 0.2)), Material.from_itu("concrete"), name="ground"))
    studio = adapter.to_studio(scene)
    obj = next(o for o in studio.objects.values() if o.name == "ground")
    preset = obj.get_component("MaterialPreset")
    assert preset is not None
    assert str(preset.preset) == "concrete"
    assert bool(preset.use_frequency_override) is False
    # Reconstructs via the descriptor identity, NOT the evaluated numbers.
    rebuilt = adapter.to_platform(studio).structures[0]
    assert rebuilt.material.itu_descriptor == ("concrete", None)


def test_itu_preset_frequency_override_roundtrip(adapter):
    # Scene at 3.5 GHz, but this material is pinned to 60 GHz (glass's high band).
    scene = _scene(Structure(Box(size=(2, 2, 0.1)), Material.from_itu("glass", 60e9), name="pane"))
    studio = adapter.to_studio(scene)
    preset = next(o for o in studio.objects.values() if o.name == "pane").get_component("MaterialPreset")
    assert bool(preset.use_frequency_override) is True
    assert float(preset.frequency_override) == pytest.approx(60e9)
    rebuilt = adapter.to_platform(studio).structures[0]
    assert rebuilt.material.itu_descriptor[0] == "glass"
    assert rebuilt.material.itu_descriptor[1] == pytest.approx(60e9)


def test_itu_alias_normalizes(adapter):
    scene = _scene(Structure(Box(size=(2, 2, 0.1)), Material.from_itu("itu_brick"), name="b"))
    rebuilt = RT.back(adapter, scene).structures[0]
    assert rebuilt.material.itu_descriptor == ("brick", None)


def test_itu_band_validation_on_export(adapter):
    # very_dry_ground is only defined over 1-10 GHz; exporting at 50 GHz must raise.
    scene = _scene(Structure(Box(size=(8, 8, 0.2)), Material.from_itu("very_dry_ground"), name="g"))
    studio = adapter.to_studio(scene)
    studio.get_object_by_name("Channel Settings").get_component("ChannelSettings").frequency = 50e9
    with pytest.raises(ValueError, match="very_dry_ground"):
        adapter.to_platform(studio)
