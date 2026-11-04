"""Channel endpoints — the ``Tx`` and ``Rx`` scene components.

Each endpoint is an ``Empty`` ``SceneObject``: its world position is the object's
``Transform``, while the channel-specific fields (polarization, power, orientation)
and the antenna definition live on the component itself. There is no separate antenna
component — every Tx/Rx carries its own antenna inline under the ``Antenna`` group:
a per-element ``pattern`` plus an ``array_kind`` (single element, ULA, UPA, or a custom
element layout) and its polarization. ``array_kind="single"`` with ``pattern="iso"`` and
``array_polarization="v"`` means "no explicit array" (the solver's default element).

Each endpoint draws a viewport gizmo (a colored marker at its position) via
``on_draw_gizmos`` so it is visible even though the object itself is an Empty.

The antenna fields are declared on both components (rather than a shared base) so each
endpoint controls its own field ordering; the component metaclass collects fields from
``cls.__dict__`` only, so a plain mixin would be dropped. ``adapter.array_spec`` reads
these identically-named fields from either component.
"""
from witwin_server.api import (
    Component,
    GizmoContext,
    bool_field,
    component,
    float_field,
    int_field,
    list_field,
    string_field,
    vector3_field,
)

from .category import CATEGORY

_ULA = {"field_name": "array_kind", "operator": "eq", "value": "ula"}
_UPA = {"field_name": "array_kind", "operator": "eq", "value": "upa"}
_CUSTOM = {"field_name": "array_kind", "operator": "eq", "value": "custom"}
_POL_CUSTOM = {"field_name": "array_polarization", "operator": "eq", "value": "custom"}
_TX_COLOR = "#22c55e"  # green = transmitter
_RX_COLOR = "#ef4444"  # red = receiver


@component(name="ChannelTx", display_name="Tx", category=CATEGORY)
class TxComponent(Component):
    """Channel transmitter endpoint (position via the object's Transform).

    Named ``ChannelTx`` (not ``Transmitter``) to avoid colliding with the core
    ``Transmitter`` component in the global registry.
    """

    # Asks the Studio viewport to stream gizmo drags of this object to the server, so a
    # RadioMap with live_update follows the Tx while it moves (not only on mouse-up).
    live_drag = bool_field(True, hidden=True)
    polarization = vector3_field([1.0, 0.0, 0.0], description="Transmit field polarization vector.")
    power = float_field(1.0, min=1e-12, description="Transmit power (linear, > 0).")
    use_orientation = bool_field(False, title="Orientation")
    orientation = vector3_field([0.0, 0.0, 0.0], show_if="use_orientation",
                                description="Antenna orientation euler (yaw, pitch, roll; rad).")

    # --- antenna (inline) ---
    pattern = string_field("iso", options=["iso", "dipole", "tr38901"], group="Antenna",
                           title="Antenna", description="Per-element radiation pattern.")
    array_kind = string_field("single", options=["single", "ula", "upa", "custom"],
                              enum_toggle=True, group="Antenna",
                              description="Single element, a ULA/UPA preset, or a custom layout.")
    array_polarization = string_field("v", options=["v", "h", "vh", "cross", "custom"],
                                      group="Antenna", description="Element polarization preset.")
    num_elements = int_field(4, min=1, group="Antenna", show_if=_ULA)
    spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_ULA,
                          description="Element spacing (m).")
    ula_axis = string_field("x", options=["x", "y"], enum_toggle=True, group="Antenna", show_if=_ULA)
    num_rows = int_field(2, min=1, group="Antenna", show_if=_UPA, horizontal_group="Grid")
    num_cols = int_field(2, min=1, group="Antenna", show_if=_UPA, horizontal_group="Grid")
    vertical_spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_UPA,
                                   horizontal_group="Spacing")
    horizontal_spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_UPA,
                                     horizontal_group="Spacing")
    element_positions = list_field(vector3_field(), default=[[0.0, 0.0, 0.0]], group="Antenna",
                                   show_if=_CUSTOM, description="Local element offsets (m).")
    use_element_orientations = bool_field(False, group="Antenna", show_if=_CUSTOM)
    element_orientations = list_field(vector3_field(), default=[], group="Antenna",
                                      show_if="use_element_orientations",
                                      description="Per-element euler orientation (rad).")
    pol_slots = list_field(vector3_field(), default=[[1.0, 0.0, 0.0]], group="Antenna",
                           show_if=_POL_CUSTOM, description="Custom polarization slots (real part).")
    pol_slots_imag = list_field(vector3_field(), default=[], group="Antenna", show_if=_POL_CUSTOM,
                                description="Custom polarization slots (imaginary part; optional).")

    def on_draw_gizmos(self, ctx: GizmoContext) -> None:
        """A green marker + a forward arrow at the transmitter position."""
        ctx.color = _TX_COLOR
        ctx.draw_sphere(radius=0.15, wireframe=False)
        ctx.draw_arrow(length=0.45, direction=(0, 0, 1))


@component(name="ChannelRx", display_name="Rx", category=CATEGORY)
class RxComponent(Component):
    """Channel point receiver endpoint (position via the object's Transform).

    Named ``ChannelRx`` (not ``Receiver``) to avoid colliding with the core
    ``Receiver`` component in the global registry.
    """

    use_polarization = bool_field(False, title="Polarization")
    polarization = vector3_field([1.0, 0.0, 0.0], show_if="use_polarization",
                                 description="Receive field polarization vector.")
    use_orientation = bool_field(False, title="Orientation")
    orientation = vector3_field([0.0, 0.0, 0.0], show_if="use_orientation",
                                description="Antenna orientation euler (yaw, pitch, roll; rad).")

    # --- antenna (inline) ---
    pattern = string_field("iso", options=["iso", "dipole", "tr38901"], group="Antenna",
                           title="Antenna", description="Per-element radiation pattern.")
    array_kind = string_field("single", options=["single", "ula", "upa", "custom"],
                              enum_toggle=True, group="Antenna",
                              description="Single element, a ULA/UPA preset, or a custom layout.")
    array_polarization = string_field("v", options=["v", "h", "vh", "cross", "custom"],
                                      group="Antenna", description="Element polarization preset.")
    num_elements = int_field(4, min=1, group="Antenna", show_if=_ULA)
    spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_ULA,
                          description="Element spacing (m).")
    ula_axis = string_field("x", options=["x", "y"], enum_toggle=True, group="Antenna", show_if=_ULA)
    num_rows = int_field(2, min=1, group="Antenna", show_if=_UPA, horizontal_group="Grid")
    num_cols = int_field(2, min=1, group="Antenna", show_if=_UPA, horizontal_group="Grid")
    vertical_spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_UPA,
                                   horizontal_group="Spacing")
    horizontal_spacing = float_field(0.5, min=0.0, group="Antenna", show_if=_UPA,
                                     horizontal_group="Spacing")
    element_positions = list_field(vector3_field(), default=[[0.0, 0.0, 0.0]], group="Antenna",
                                   show_if=_CUSTOM, description="Local element offsets (m).")
    use_element_orientations = bool_field(False, group="Antenna", show_if=_CUSTOM)
    element_orientations = list_field(vector3_field(), default=[], group="Antenna",
                                      show_if="use_element_orientations",
                                      description="Per-element euler orientation (rad).")
    pol_slots = list_field(vector3_field(), default=[[1.0, 0.0, 0.0]], group="Antenna",
                           show_if=_POL_CUSTOM, description="Custom polarization slots (real part).")
    pol_slots_imag = list_field(vector3_field(), default=[], group="Antenna", show_if=_POL_CUSTOM,
                                description="Custom polarization slots (imaginary part; optional).")

    def on_draw_gizmos(self, ctx: GizmoContext) -> None:
        """A red marker + a ring at the receiver position."""
        ctx.color = _RX_COLOR
        ctx.draw_sphere(radius=0.15, wireframe=False)
        ctx.draw_circle(radius=0.28, segments=20)
