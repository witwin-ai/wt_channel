"""ITU-R P.2040 material presets evaluated into Core 0.4 materials.

witwin-channel 0.4+ dropped ``Material.from_itu``; Core keeps only the scalar
``PhysicalMaterial`` contract. The plugin therefore owns the preset table (the same
coefficients the 0.3 channel loader shipped) and evaluates a preset at one frequency:
``eps_r = a * f_GHz**b``, ``sigma_e = c * f_GHz**d`` (S/m), ``mu_r = 1``. A preset is
defined only over its band(s); evaluating outside every band raises.
"""
from typing import Any, Dict, Tuple

# name -> ((f_min_GHz, f_max_GHz), (a, b, c, d)) per band.
ITU_MATERIALS: Dict[str, Tuple[Tuple[Tuple[float, float], Tuple[float, float, float, float]], ...]] = {
    "concrete": (((1.0, 100.0), (5.24, 0.0, 0.0462, 0.7822)),),
    "brick": (((1.0, 40.0), (3.91, 0.0, 0.0238, 0.16)),),
    "plasterboard": (((1.0, 100.0), (2.73, 0.0, 0.0085, 0.9395)),),
    "wood": (((0.001, 100.0), (1.99, 0.0, 0.0047, 1.0718)),),
    "glass": (
        ((0.1, 100.0), (6.31, 0.0, 0.0036, 1.3394)),
        ((220.0, 450.0), (5.79, 0.0, 0.0004, 1.658)),
    ),
    "ceiling_board": (
        ((1.0, 100.0), (1.48, 0.0, 0.0011, 1.0750)),
        ((220.0, 450.0), (1.52, 0.0, 0.0029, 1.029)),
    ),
    "chipboard": (((1.0, 100.0), (2.58, 0.0, 0.0217, 0.7800)),),
    "plywood": (((1.0, 40.0), (2.71, 0.0, 0.33, 0.0)),),
    "marble": (((1.0, 60.0), (7.074, 0.0, 0.0055, 0.9262)),),
    "floorboard": (((50.0, 100.0), (3.66, 0.0, 0.0044, 1.3515)),),
    "metal": (((1.0, 100.0), (1.0, 0.0, 1.0e7, 0.0)),),
    "very_dry_ground": (((1.0, 10.0), (3.0, 0.0, 0.00015, 2.52)),),
    "medium_dry_ground": (((1.0, 10.0), (15.0, -0.1, 0.035, 1.63)),),
    "wet_ground": (((1.0, 10.0), (30.0, -0.4, 0.15, 1.30)),),
}


def normalize_itu_name(name: str) -> str:
    """'itu_brick' / 'Ceiling Board' -> 'brick' / 'ceiling_board'."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key.startswith("itu_"):
        key = key[len("itu_"):]
    return key


def itu_parameters(name: str, frequency_hz: float) -> Tuple[float, float]:
    """Return ``(eps_r, sigma_e)`` for an ITU preset at ``frequency_hz``."""
    key = normalize_itu_name(name)
    bands = ITU_MATERIALS.get(key)
    if bands is None:
        raise ValueError(f"unknown ITU material {name!r}; expected one of {sorted(ITU_MATERIALS)}")
    f_ghz = float(frequency_hz) / 1e9
    for (f_min, f_max), (a, b, c, d) in bands:
        if f_min <= f_ghz <= f_max:
            return a * f_ghz ** b, c * f_ghz ** d
    ranges = ", ".join(f"{lo:g}-{hi:g} GHz" for (lo, hi), _ in bands)
    raise ValueError(f"ITU material '{key}' is not defined at {f_ghz:g} GHz (bands: {ranges})")


def itu_material(name: str, frequency_hz: float) -> Any:
    """Evaluate an ITU preset into a scalar ``witwin.core.PhysicalMaterial``."""
    from witwin.core import PhysicalMaterial

    eps_r, sigma_e = itu_parameters(name, frequency_hz)
    return PhysicalMaterial(eps_r=eps_r, sigma_e=sigma_e, name=normalize_itu_name(name))
