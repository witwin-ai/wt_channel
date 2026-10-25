"""Channel material interchange: structure material -> ``core.PhysicalMaterial``.

Two authoring modes (channel keeps presets channel-only):

1. **Explicit scalar** — the unified ``Material`` component -> ``PhysicalMaterial``
   (eps_r/mu_r/sigma_e/name) via the frozen ``MaterialMap``.
2. **ITU preset** — the ``MaterialPreset`` companion -> the ITU-R P.2040 preset
   evaluated at the effective frequency (the per-material override, else the scene
   frequency). Core 0.4 has no ITU identity on the material, so Studio keeps the
   preset on the companion component and re-evaluates it on every export.

``MaterialPreset`` takes precedence: if a structure carries it, the scalar
``Material``'s EM fields are ignored on export.
"""
from typing import Any, Optional

import numpy as np
from witwin_server import SceneObject
from witwin_server.features.platform.bridge import MaterialMap

from .common import num
from .itu import ITU_MATERIALS, itu_material, normalize_itu_name


class ChannelMaterialMap:
    """Structure material -> ``core.PhysicalMaterial`` (scalar or ITU preset)."""

    @staticmethod
    def to_platform(obj: SceneObject, scene_frequency: Optional[float]) -> Any:
        """Build the platform material: ITU preset if present, else scalar Material."""
        preset = obj.get_component("MaterialPreset")
        if preset is not None:
            return ChannelMaterialMap.from_preset(preset, scene_frequency)
        material_comp = obj.get_component("Material")
        if material_comp is not None:
            name = ChannelMaterialMap.itu_name(material_comp)
            if name is not None:
                if scene_frequency is None:
                    raise ValueError(f"ITU material '{name}' requires a scene frequency.")
                return itu_material(name, float(scene_frequency))
            return MaterialMap.to_platform(material_comp)
        from witwin.core import PhysicalMaterial

        return PhysicalMaterial()

    @staticmethod
    def itu_name(material_comp: Any) -> Optional[str]:
        """The ITU preset picked on the unified Material component, if any.

        Studio's picker auto-fills nominal 1 GHz scalars; the channel re-evaluates the
        preset at the solve frequency instead.
        """
        name = normalize_itu_name(str(material_comp.material_name or ""))
        return name if name in ITU_MATERIALS else None

    @staticmethod
    def has_authored_scalars(material_comp: Any) -> bool:
        """True when eps_r / mu_r / sigma_e differ from the component defaults."""
        defaults = type(material_comp)()
        for field in ("eps_r", "mu_r", "sigma_e"):
            actual = np.asarray(_plain(getattr(material_comp, field)), dtype=np.float64)
            default = np.asarray(_plain(getattr(defaults, field)), dtype=np.float64)
            if actual.shape != default.shape or not np.allclose(actual, default):
                return True
        return False

    @staticmethod
    def from_preset(preset: Any, scene_frequency: Optional[float]) -> Any:
        """Evaluate a ``MaterialPreset`` at its override or the scene frequency."""
        name = normalize_itu_name(str(preset.preset))
        override = num(preset.frequency_override) if bool(preset.use_frequency_override) else None
        effective = override if override is not None else scene_frequency
        if effective is None:
            raise ValueError(
                f"ITU material '{name}' requires a scene frequency or a frequency override.")
        return itu_material(name, float(effective))


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value
