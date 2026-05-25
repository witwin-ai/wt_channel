"""Channel material interchange: structure material <-> core.Material / ITU preset.

Two authoring modes (master §4.1, channel keeps presets channel-only):

1. **Explicit scalar** — the unified ``EMMaterial`` component -> ``core.Material``
   (eps_r/mu_r/sigma_e/name) via the frozen ``MaterialMap``. Maxwell-only advanced
   fields are dropped (a Fresnel ray tracer cannot consume dispersion).
2. **ITU preset** — the ``MaterialPreset`` companion -> ``Material.from_itu(name,
   override)``. The round trip preserves the ``itu_descriptor`` *identity*
   ``(name, override)``, never the evaluated numbers.

``MaterialPreset`` takes precedence: if a structure carries it, the scalar
``EMMaterial`` is ignored on export.
"""
from typing import Any, Optional

from witwin_server import SceneObject
from witwin_server.platform_bridge import MaterialMap

from .common import num


class ChannelMaterialMap:
    """Structure material <-> ``core.Material`` (scalar) or ITU preset."""

    @staticmethod
    def overlay_studio(obj: SceneObject, material: Any) -> None:
        """Attach a ``MaterialPreset`` companion if ``material`` is an ITU preset."""
        descriptor = getattr(material, "itu_descriptor", None)
        if descriptor is None:
            return
        from ..components.materials import MaterialPresetComponent

        name, override = descriptor
        comp = obj.add_component(MaterialPresetComponent())
        comp.preset = str(name)
        if override is not None:
            comp.use_frequency_override = True
            comp.frequency_override = float(override)

    @staticmethod
    def to_platform(ch: Any, obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        """Build the platform material: ITU preset if present, else scalar EMMaterial."""
        preset = obj.get_component("MaterialPreset")
        if preset is not None:
            return ChannelMaterialMap._from_preset(ch, preset, scene_frequency)
        em = obj.get_component("EMMaterial")
        if em is not None:
            return MaterialMap.to_platform(em)
        return ch.Material()

    @staticmethod
    def _from_preset(ch: Any, preset: Any, scene_frequency: Optional[float]) -> Any:
        # Rebuild via from_itu(name, override); validate the effective frequency's band.
        from witwin.channel.core.scene.material_presets import (
            evaluate_itu_material,
            normalize_itu_name,
        )

        name = normalize_itu_name(str(preset.preset))
        override = num(preset.frequency_override) if bool(preset.use_frequency_override) else None
        effective = override if override is not None else scene_frequency
        if effective is None:
            raise ValueError(
                f"ITU material '{name}' requires a scene frequency or a frequency override.")
        evaluate_itu_material(name, float(effective))  # raises if outside the preset's band
        return ch.Material.from_itu(name, override)
