"""C1 — ITU material presets on the Core 0.4 world (CPU).

witwin-channel 0.4+ no longer ships ``Material.from_itu``; the plugin owns the ITU-R
P.2040 table and evaluates it into a scalar ``witwin.core.PhysicalMaterial`` at the
solve frequency: ``eps_r = a*f_GHz^b``, ``sigma_e = c*f_GHz^d``. Each preset is defined
only over its band, so out-of-band exports raise.
"""
import pytest


def test_concrete_evaluates_at_frequency():
    from witwin.core import PhysicalMaterial
    from wt_channel.adapter.itu import itu_material

    material = itu_material("concrete", 3.5e9)

    assert isinstance(material, PhysicalMaterial)
    assert material.name == "concrete"
    assert float(material.eps_r) == pytest.approx(5.24)
    assert float(material.sigma_e) == pytest.approx(0.0462 * 3.5 ** 0.7822)
    assert float(material.mu_r) == pytest.approx(1.0)


def test_glass_uses_its_high_band():
    from wt_channel.adapter.itu import itu_material

    material = itu_material("glass", 300e9)

    assert float(material.eps_r) == pytest.approx(5.79)
    assert float(material.sigma_e) == pytest.approx(0.0004 * 300.0 ** 1.658)


def test_itu_alias_normalizes():
    from wt_channel.adapter.itu import normalize_itu_name

    assert normalize_itu_name("itu_brick") == "brick"
    assert normalize_itu_name("Ceiling Board") == "ceiling_board"


def test_out_of_band_raises_with_name():
    from wt_channel.adapter.itu import itu_material

    with pytest.raises(ValueError, match="very_dry_ground"):
        itu_material("very_dry_ground", 50e9)


def test_unknown_preset_raises():
    from wt_channel.adapter.itu import itu_material

    with pytest.raises(ValueError, match="unknown ITU material"):
        itu_material("cheese", 3.5e9)


def test_component_preset_names_match_table():
    from wt_channel.adapter.itu import ITU_MATERIALS
    from wt_channel.components.materials import PRESET_NAMES

    assert set(PRESET_NAMES) == set(ITU_MATERIALS)
