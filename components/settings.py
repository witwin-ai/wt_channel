"""ChannelSettings — the scene-level config singleton (frequency / device / noise).

Lives on the single ``Channel Settings`` Empty object; its presence marks that
object as the scene-config carrier for :class:`ChannelAdapter`. ``frequency`` is
scene-level (``Scene.frequency``) and drives ITU material evaluation. ``noise_power``
is a loose, optional scene attribute consulted only by the deterministic SINR metric.
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    string_field,
)

CATEGORY = "Simulation/Channel"


@component(name="ChannelSettings", category=CATEGORY)
class ChannelSettingsComponent(Component):
    """Scene-level channel config carried by the Channel Settings object."""

    frequency = float_field(
        3.5e9, min=0.0,
        description="Scene carrier frequency in Hz (re-evaluates ITU materials).",
    )
    device = string_field(
        "cuda", options=["cuda", "cpu"], enum_toggle=True,
        description="Solve device. Round-trip on CPU; live solve needs CUDA RayD.",
    )
    use_noise_power = bool_field(
        False, title="Noise",
        description="Provide a noise power (used only by the deterministic SINR metric).",
    )
    noise_power = float_field(
        1e-13, min=0.0, show_if="use_noise_power",
        description="Linear noise power for the deterministic SINR metric.",
    )
