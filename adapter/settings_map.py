"""Scene-level config interchange: channel scene config <-> Channel Settings object.

``to_studio`` reads a ``channel.Scene`` (duck-typed) into a single ``Empty``
``SceneObject`` carrying the ``ChannelSettings`` component. ``build_config`` reads it
back into the plain values needed to construct a ``channel.Scene``.
"""
from dataclasses import dataclass
from typing import Any, Optional

from witwin_server import SceneObject

from .common import device_name, num

SETTINGS_NAME = "Channel Settings"
MARKER_COMPONENT = "ChannelSettings"


@dataclass
class SceneConfig:
    """The plain values needed to build a ``channel.Scene`` (see :class:`ChannelAdapter`)."""

    frequency: Optional[float]
    device: str
    noise_power: Optional[float]


class SettingsMap:
    """Channel scene config <-> the Channel Settings object's ``ChannelSettings``."""

    @staticmethod
    def to_studio(scene: Any) -> SceneObject:
        """Build the Channel Settings ``Empty`` object from a ``channel.Scene``."""
        from ..components.settings import ChannelSettingsComponent

        obj = SceneObject(name=SETTINGS_NAME, mesh_type="Empty")
        comp = obj.add_component(ChannelSettingsComponent())
        comp.frequency = 0.0 if scene.frequency is None else float(scene.frequency)
        comp.device = device_name(scene.device)
        noise = getattr(scene, "noise_power", None)
        if noise is not None:
            comp.use_noise_power = True
            comp.noise_power = float(noise)
        return obj

    @staticmethod
    def build_config(settings_obj: SceneObject) -> SceneConfig:
        """Read the ChannelSettings component back into a :class:`SceneConfig`."""
        comp = settings_obj.get_component(MARKER_COMPONENT)
        frequency = num(comp.frequency)
        noise = num(comp.noise_power) if bool(comp.use_noise_power) else None
        return SceneConfig(
            frequency=frequency if frequency > 0.0 else None,
            device=str(comp.device),
            noise_power=noise,
        )
