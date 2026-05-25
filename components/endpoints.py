"""Channel endpoints — Transmitter / Receiver / ReceiverGrid components.

Each endpoint is an ``Empty`` ``SceneObject``: its world position is the object's
``Transform`` (a point ``Receiver`` / ``Transmitter``), while the channel-specific
fields (polarization, power, orientation) live on the component. The endpoint's
antenna array, if any, is the sibling ``AntennaArraySpec`` component (plan §4.4).

``ReceiverGrid`` is the 2D radiomap receiver: an axis-aligned plane (``axis`` +
``position``) over a tangential ``bounds`` rectangle, sampled by **exactly one** of
``grid_shape`` or ``cell_size`` (the platform ``GridSpec`` enforces the XOR).
``orientation`` is a yaw/pitch/roll euler (radians), kept separate from the display
``Transform`` rotation (it is the antenna's electrical orientation, not a pose).
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    int_field,
    string_field,
    vector3_field,
    vector4_field,
)

from .settings import CATEGORY

_GRID_SHAPE = {"field_name": "grid_mode", "operator": "eq", "value": "grid_shape"}
_CELL_SIZE = {"field_name": "grid_mode", "operator": "eq", "value": "cell_size"}


@component(name="Transmitter", category=CATEGORY)
class TransmitterComponent(Component):
    """Scene transmitter endpoint (position via the object's Transform)."""

    polarization = vector3_field([1.0, 0.0, 0.0], description="Linear polarization vector.")
    power = float_field(1.0, min=1e-12, description="Transmit power (linear, > 0).")
    use_orientation = bool_field(False, title="Orientation")
    orientation = vector3_field([0.0, 0.0, 0.0], show_if="use_orientation",
                                description="Antenna orientation euler (yaw, pitch, roll; rad).")


@component(name="Receiver", category=CATEGORY)
class ReceiverComponent(Component):
    """Scene point receiver endpoint (position via the object's Transform)."""

    use_polarization = bool_field(False, title="Polarization")
    polarization = vector3_field([1.0, 0.0, 0.0], show_if="use_polarization",
                                 description="Linear polarization vector.")
    use_orientation = bool_field(False, title="Orientation")
    orientation = vector3_field([0.0, 0.0, 0.0], show_if="use_orientation",
                                description="Antenna orientation euler (yaw, pitch, roll; rad).")


@component(name="ReceiverGrid", category=CATEGORY)
class ReceiverGridComponent(Component):
    """Axis-aligned 2D radiomap receiver grid (plan §4.3)."""

    axis = string_field("z", options=["x", "y", "z"], enum_toggle=True,
                        description="Plane normal axis.")
    position = float_field(1.5, description="Plane offset along the axis (m).")
    bounds = vector4_field([-5.0, 5.0, -5.0, 5.0],
                           description="Tangential extents [min0, max0, min1, max1].")
    grid_mode = string_field("grid_shape", options=["grid_shape", "cell_size"], enum_toggle=True,
                             description="Discretize by cell counts or by cell size (exactly one).")
    grid_nx = int_field(64, min=1, show_if=_GRID_SHAPE, horizontal_group="Grid shape")
    grid_ny = int_field(64, min=1, show_if=_GRID_SHAPE, horizontal_group="Grid shape")
    cell_dx = float_field(0.5, min=1e-6, show_if=_CELL_SIZE, horizontal_group="Cell size")
    cell_dy = float_field(0.5, min=1e-6, show_if=_CELL_SIZE, horizontal_group="Cell size")

    use_polarization = bool_field(False, title="Polarization")
    polarization = vector3_field([1.0, 0.0, 0.0], show_if="use_polarization")
    use_orientation = bool_field(False, title="Orientation")
    orientation = vector3_field([0.0, 0.0, 0.0], show_if="use_orientation",
                                description="Antenna orientation euler (yaw, pitch, roll; rad).")
