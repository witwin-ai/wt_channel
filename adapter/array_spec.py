"""Antenna interchange: a Tx/Rx component's inline antenna fields <-> platform array.

The antenna lives directly on the endpoint component (``ChannelTx`` / ``ChannelRx``):
``array_kind`` selects a single element, a ULA / UPA preset, or a custom element layout,
with ``pattern`` and ``array_polarization`` shared across all kinds. ``to_platform``
returns ``None`` for the plain default (single iso/V element) so the platform applies its
own ``default_array()``; any non-default antenna becomes an explicit ``AntennaArray`` /
``ULA`` / ``UPA``. Polarization round-trips as a preset name (``v/h/vh/cross``) when it
matches, else as raw complex 3-vector slots (real + imaginary lists).
"""
import math
from typing import Any, Optional, Tuple

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
    """A Tx/Rx component's antenna fields <-> ``AntennaArray`` / ``ULA`` / ``UPA``."""

    @staticmethod
    def to_platform(ch: Any, comp: Any) -> Optional[Any]:
        """Build the platform antenna array, or ``None`` for the plain default element."""
        kind = str(comp.array_kind)
        pattern = str(comp.pattern)
        polarization = ArraySpecMap._pol_to_platform(comp)
        if kind == "single":
            if pattern == "iso" and str(comp.array_polarization) == "v":
                return None  # plain default element -> let the platform supply default_array()
            return ch.AntennaArray(element_positions=[(0.0, 0.0, 0.0)],
                                   polarization=polarization, pattern=pattern)
        if kind == "ula":
            return ch.ULA(num_elements=int(comp.num_elements), spacing=num(comp.spacing),
                          axis=str(comp.ula_axis), polarization=polarization, pattern=pattern)
        if kind == "upa":
            return ch.UPA(num_rows=int(comp.num_rows), num_cols=int(comp.num_cols),
                          vertical_spacing=num(comp.vertical_spacing),
                          horizontal_spacing=num(comp.horizontal_spacing),
                          polarization=polarization, pattern=pattern)
        positions = [vec(p) for p in comp.element_positions]
        orientations = None
        if bool(comp.use_element_orientations) and len(comp.element_orientations) > 0:
            orientations = [vec(o) for o in comp.element_orientations]
        return ch.AntennaArray(element_positions=positions, element_orientations=orientations,
                               polarization=polarization, pattern=pattern)

    @staticmethod
    def to_studio(comp: Any, array: Optional[Any]) -> None:
        """Fill a Tx/Rx component's antenna fields from a platform array (None -> default)."""
        if array is None:
            return  # defaults (single / iso / V) already on the component
        comp.pattern = str(array.pattern)
        ArraySpecMap._pol_to_studio(comp, array.polarization_slots)
        kind = ArraySpecMap._detect_kind(array)
        comp.array_kind = kind
        if kind == "ula":
            ArraySpecMap._fill_ula(comp, array)
        elif kind == "upa":
            comp.num_rows = int(array.num_rows)
            comp.num_cols = int(array.num_cols)
            comp.vertical_spacing = float(array.vertical_spacing)
            comp.horizontal_spacing = float(array.horizontal_spacing)
        elif kind == "custom":
            ArraySpecMap._fill_custom(comp, array)

    # --- kind detection ------------------------------------------------------

    @staticmethod
    def _detect_kind(array: Any) -> str:
        # ULA is the most specific (subclass of PlanarArray); check it first.
        import witwin.channel as ch

        if isinstance(array, ch.ULA):
            return "ula"
        if isinstance(array, ch.PlanarArray):
            return "upa"
        if array.num_elements == 1 and ArraySpecMap._is_origin(array):
            return "single"
        return "custom"

    @staticmethod
    def _is_origin(array: Any) -> bool:
        ep = array.element_positions
        return all(abs(float(c)) <= TOL for c in (ep.x[0], ep.y[0], ep.z[0]))

    @staticmethod
    def _fill_ula(comp: Any, array: Any) -> None:
        # ULA stores its layout as num_rows/num_cols + per-axis spacing; recover axis.
        if int(array.num_rows) == 1:
            comp.ula_axis = "x"
            comp.num_elements = int(array.num_cols)
            comp.spacing = float(array.horizontal_spacing)
        else:
            comp.ula_axis = "y"
            comp.num_elements = int(array.num_rows)
            comp.spacing = float(array.vertical_spacing)

    @staticmethod
    def _fill_custom(comp: Any, array: Any) -> None:
        ep = array.element_positions
        comp.element_positions = [[float(x), float(y), float(z)]
                                  for x, y, z in zip(ep.x, ep.y, ep.z)]
        orient = array.element_orientations
        if orient is not None:
            comp.use_element_orientations = True
            comp.element_orientations = [[float(x), float(y), float(z)]
                                         for x, y, z in zip(orient.x, orient.y, orient.z)]

    # --- polarization --------------------------------------------------------

    @staticmethod
    def _pol_to_platform(comp: Any) -> Any:
        # Preset name passes straight through; 'custom' builds complex 3-vector slots.
        name = str(comp.array_polarization)
        if name != "custom":
            return name
        reals = [vec(s) for s in comp.pol_slots]
        imags = [vec(s) for s in comp.pol_slots_imag]
        slots = []
        for i, real in enumerate(reals):
            imag = imags[i] if i < len(imags) else (0.0, 0.0, 0.0)
            slots.append(tuple(complex(real[j], imag[j]) for j in range(3)))
        return slots

    @staticmethod
    def _pol_to_studio(comp: Any, slots: Tuple) -> None:
        # Match a preset by value; otherwise store raw complex slots (real + imag).
        preset = ArraySpecMap._match_preset(slots)
        if preset is not None:
            comp.array_polarization = preset
            return
        comp.array_polarization = "custom"
        comp.pol_slots = [[float(c.real) for c in slot] for slot in slots]
        if any(abs(c.imag) > TOL for slot in slots for c in slot):
            comp.pol_slots_imag = [[float(c.imag) for c in slot] for slot in slots]

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
