"""MaterialPreset — ITU material companion (channel-only).

An ITU material's *identity* lives in its hidden ``itu_descriptor``
``(normalized_name, frequency_override_or_None)``, not in the evaluated
``eps_r`` / ``sigma_e`` (those are derived from the frequency at solve time). When a
structure's material is an ITU preset, the adapter attaches this companion component
(carrying the name + optional override) and exports it via ``Material.from_itu`` —
the unified ``Material`` scalar EM fields are ignored for that structure.

ITU formula (``material_presets.py``): ``eps_r = a*f_GHz^b``, ``sigma_e = c*f_GHz^d``,
``mu_r = 1``; each preset is only defined over a frequency band (the platform raises
outside it).
"""
from witwin_server.core.components import (
    Component,
    bool_field,
    component,
    float_field,
    string_field,
)

from .category import CATEGORY

# Preset names from witwin.channel material_presets.ITU_MATERIALS_PROPERTIES.
PRESET_NAMES = [
    "concrete", "brick", "plasterboard", "wood", "glass", "ceiling_board",
    "chipboard", "plywood", "marble", "floorboard", "metal",
    "very_dry_ground", "medium_dry_ground", "wet_ground",
]


@component(name="MaterialPreset", category=CATEGORY)
class MaterialPresetComponent(Component):
    """ITU material preset (name + optional per-material frequency override)."""

    preset = string_field(
        "concrete", options=PRESET_NAMES,
        description="ITU material preset; eps_r/sigma_e are evaluated at the frequency.",
    )
    use_frequency_override = bool_field(
        False, title="Frequency",
        description="Override the scene frequency for this material's ITU evaluation.",
    )
    frequency_override = float_field(
        3.5e9, min=0.0, show_if="use_frequency_override",
        description="Per-material ITU evaluation frequency in Hz.",
        info_box={"message": "Must fall in the preset's defined band.", "type": "info"},
    )
