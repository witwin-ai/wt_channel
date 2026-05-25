"""Antenna array interchange: ``AntennaArraySpec`` component <-> platform array.

The platform offers a generic ``AntennaArray`` (explicit element positions) and the
``ULA`` / ``UPA`` presets (both subclasses of ``PlanarArray``). ``to_platform`` builds
the right class from the component's ``array_kind``; ``to_studio`` detects the class
(ULA, then UPA / PlanarArray, then generic) and fills the component. Polarization
round-trips as a preset name when it matches ``v/h/vh/cross``, else as raw complex
3-vector slots (real + imaginary lists).
"""
import math
from typing import Any, List, Optional, Tuple

from witwin_server import SceneObject

from .common import num, vec

TOL = 1e-6

# Preset polarization slots (real), mirrored from arrays._POL_PRESETS.
_S = 1.0 / math.sqrt(2.0)
_POL_PRESETS = {
    "v": ((1.0, 0.0, 0.0),),
    "h": ((0.0, 1.0, 0.0),),
    "vh": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "cross": ((_S, _S, 0.0), (_S, -_S, 0.0)),
}


class ArraySpecMap:
    """``AntennaArraySpec`` component <-> ``AntennaArray`` / ``ULA`` / ``UPA``."""

    @staticmethod
    def to_platform(ch: Any, spec: Any) -> Any:
        """Build the platform antenna array from an ``AntennaArraySpec`` component."""
        kind = str(spec.array_kind)
        pattern = str(spec.pattern)
        polarization = ArraySpecMap._pol_to_platform(spec)
        if kind == "ula":
            return ch.ULA(num_elements=int(spec.num_elements), spacing=num(spec.spacing),
                          axis=str(spec.axis), polarization=polarization, pattern=pattern)
        if kind == "upa":
            return ch.UPA(num_rows=int(spec.num_rows), num_cols=int(spec.num_cols),
                          vertical_spacing=num(spec.vertical_spacing),
                          horizontal_spacing=num(spec.horizontal_spacing),
                          polarization=polarization, pattern=pattern)
        positions = [vec(p) for p in spec.element_positions]
        orientations = None
        if bool(spec.use_element_orientations) and len(spec.element_orientations) > 0:
            orientations = [vec(o) for o in spec.element_orientations]
        return ch.AntennaArray(element_positions=positions, element_orientations=orientations,
                               polarization=polarization, pattern=pattern)

    @staticmethod
    def to_studio(spec: Any, array: Any) -> None:
        """Fill an ``AntennaArraySpec`` component from a platform antenna array."""
        spec.pattern = str(array.pattern)
        ArraySpecMap._pol_to_studio(spec, array.polarization_slots)
        kind = ArraySpecMap._detect_kind(array)
        spec.array_kind = kind
        if kind == "ula":
            ArraySpecMap._fill_ula(spec, array)
        elif kind == "upa":
            spec.num_rows = int(array.num_rows)
            spec.num_cols = int(array.num_cols)
            spec.vertical_spacing = float(array.vertical_spacing)
            spec.horizontal_spacing = float(array.horizontal_spacing)
        else:
            ArraySpecMap._fill_custom(spec, array)

    @staticmethod
    def resolve(endpoint_array: Optional[Any], scene_array: Optional[Any]) -> Optional[Any]:
        """Documented resolution order: endpoint array, else scene array, else None.

        ``None`` means the platform applies ``default_array()`` at solve time.
        """
        if endpoint_array is not None:
            return endpoint_array
        return scene_array

    @staticmethod
    def from_object(ch: Any, obj: SceneObject) -> Optional[Any]:
        """Build a platform array from an object's ``AntennaArraySpec``, or None."""
        spec = obj.get_component("AntennaArraySpec")
        return None if spec is None else ArraySpecMap.to_platform(ch, spec)

    # --- kind detection ------------------------------------------------------

    @staticmethod
    def _detect_kind(array: Any) -> str:
        # ULA is the most specific (subclass of PlanarArray); check it first.
        import witwin.channel as ch

        if isinstance(array, ch.ULA):
            return "ula"
        if isinstance(array, ch.PlanarArray):
            return "upa"
        return "custom"

    @staticmethod
    def _fill_ula(spec: Any, array: Any) -> None:
        # ULA stores its layout as num_rows/num_cols + per-axis spacing; recover axis.
        if int(array.num_rows) == 1:
            spec.axis = "x"
            spec.num_elements = int(array.num_cols)
            spec.spacing = float(array.horizontal_spacing)
        else:
            spec.axis = "y"
            spec.num_elements = int(array.num_rows)
            spec.spacing = float(array.vertical_spacing)

    @staticmethod
    def _fill_custom(spec: Any, array: Any) -> None:
        ep = array.element_positions
        spec.element_positions = [[float(x), float(y), float(z)]
                                  for x, y, z in zip(ep.x, ep.y, ep.z)]
        orient = array.element_orientations
        if orient is not None:
            spec.use_element_orientations = True
            spec.element_orientations = [[float(x), float(y), float(z)]
                                         for x, y, z in zip(orient.x, orient.y, orient.z)]

    # --- polarization --------------------------------------------------------

    @staticmethod
    def _pol_to_platform(spec: Any) -> Any:
        # Preset name passes straight through; 'custom' builds complex 3-vector slots.
        name = str(spec.polarization)
        if name != "custom":
            return name
        reals = [vec(s) for s in spec.pol_slots]
        imags = [vec(s) for s in spec.pol_slots_imag]
        slots = []
        for i, real in enumerate(reals):
            imag = imags[i] if i < len(imags) else (0.0, 0.0, 0.0)
            slots.append(tuple(complex(real[j], imag[j]) for j in range(3)))
        return slots

    @staticmethod
    def _pol_to_studio(spec: Any, slots: Tuple) -> None:
        # Match a preset by value; otherwise store raw complex slots (real + imag).
        preset = ArraySpecMap._match_preset(slots)
        if preset is not None:
            spec.polarization = preset
            return
        spec.polarization = "custom"
        spec.pol_slots = [[float(c.real) for c in slot] for slot in slots]
        if any(abs(c.imag) > TOL for slot in slots for c in slot):
            spec.pol_slots_imag = [[float(c.imag) for c in slot] for slot in slots]

    @staticmethod
    def _match_preset(slots: Tuple) -> Optional[str]:
        # Return the preset name whose (real) slots equal these, if any.
        if any(abs(c.imag) > TOL for slot in slots for c in slot):
            return None
        actual = tuple(tuple(float(c.real) for c in slot) for slot in slots)
        for name, preset in _POL_PRESETS.items():
            if len(preset) == len(actual) and all(
                all(abs(a - b) <= TOL for a, b in zip(pa, aa))
                for pa, aa in zip(preset, actual)
            ):
                return name
        return None
