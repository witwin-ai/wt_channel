"""Antenna array authoring — the reusable nested array spec + scene-default marker.

``AntennaArraySpec`` is the single array authoring surface (master / plan §4.4): one
component with a ``kind`` enum that keeps custom arrays and the ULA / UPA presets in
one place. It is attached to an endpoint's ``Empty`` object to give that endpoint an
explicit ``.array``; absent, the endpoint falls back to the scene-default array, then
to ``default_array()``.

Scene-default ``Scene.tx_array`` / ``Scene.rx_array`` are carried by dedicated
``Empty`` objects tagged with ``ChannelDefaultArray`` (role = tx / rx), each also
holding an ``AntennaArraySpec`` — so the array schema lives in exactly one component.
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    int_field,
    list_field,
    string_field,
    vector3_field,
)

from .settings import CATEGORY

_CUSTOM = {"field_name": "array_kind", "operator": "eq", "value": "custom"}
_ULA = {"field_name": "array_kind", "operator": "eq", "value": "ula"}
_UPA = {"field_name": "array_kind", "operator": "eq", "value": "upa"}
_POL_CUSTOM = {"field_name": "polarization", "operator": "eq", "value": "custom"}


@component(name="AntennaArraySpec", category=CATEGORY)
class AntennaArraySpecComponent(Component):
    """Reusable antenna array spec: custom element layout, or a ULA / UPA preset."""

    array_kind = string_field("custom", options=["custom", "ula", "upa"], enum_toggle=True,
                              title="Antenna Array")
    pattern = string_field("iso", options=["iso", "dipole", "tr38901"],
                           description="Per-element radiation pattern.")
    polarization = string_field("v", options=["v", "h", "vh", "cross", "custom"],
                                description="Polarization preset, or 'custom' complex slots.")

    # --- custom element layout (local offsets in meters) ---
    element_positions = list_field(vector3_field(), default=[[0.0, 0.0, 0.0]],
                                   show_if=_CUSTOM, description="Local element offsets (m).")
    use_element_orientations = bool_field(False, show_if=_CUSTOM)
    element_orientations = list_field(vector3_field(), default=[],
                                      show_if="use_element_orientations",
                                      description="Per-element euler orientation (rad).")

    # --- ULA preset ---
    num_elements = int_field(4, min=1, show_if=_ULA)
    spacing = float_field(0.5, min=0.0, show_if=_ULA, description="Element spacing (m).")
    axis = string_field("x", options=["x", "y"], enum_toggle=True, show_if=_ULA)

    # --- UPA preset ---
    num_rows = int_field(2, min=1, show_if=_UPA, horizontal_group="Grid")
    num_cols = int_field(2, min=1, show_if=_UPA, horizontal_group="Grid")
    vertical_spacing = float_field(0.5, min=0.0, show_if=_UPA, horizontal_group="Spacing")
    horizontal_spacing = float_field(0.5, min=0.0, show_if=_UPA, horizontal_group="Spacing")

    # --- custom polarization slots (real + optional imaginary 3-vectors) ---
    pol_slots = list_field(vector3_field(), default=[[1.0, 0.0, 0.0]], show_if=_POL_CUSTOM,
                           description="Custom polarization slots (real part).")
    pol_slots_imag = list_field(vector3_field(), default=[], show_if=_POL_CUSTOM,
                                description="Custom polarization slots (imaginary part; optional).")


@component(name="ChannelDefaultArray", category=CATEGORY)
class ChannelDefaultArrayComponent(Component):
    """Marks an Empty as a scene-default antenna array (Scene.tx_array / rx_array)."""

    role = string_field("tx", options=["tx", "rx"], enum_toggle=True,
                        description="Which scene-default slot this array fills.")
