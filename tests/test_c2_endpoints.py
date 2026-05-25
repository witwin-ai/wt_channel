"""C2 — endpoints + antenna arrays round-trip (CPU).

Per-endpoint round-trip (Transmitter / Receiver / ReceiverGrid) including position,
polarization, power, orientation (yaw/pitch/roll), and the antenna array; ReceiverGrid
grid_shape XOR cell_size; ULA / UPA reconstruction; and the scene-default array
fallback resolution order.
"""
import numpy as np
import pytest
from witwin.core import Box, Material, Structure

from _helpers import RT


def _scene(*, transmitters=(), receivers=(), tx_array=None, rx_array=None):
    import witwin.channel as ch
    return ch.Scene(
        structures=[Structure(Box(size=(2, 2, 0.2)), Material.from_itu("concrete"), name="g")],
        transmitters=list(transmitters), receivers=list(receivers),
        tx_array=tx_array, rx_array=rx_array, frequency=3.5e9, device="cpu",
    )


def test_transmitter_roundtrip(adapter):
    import witwin.channel as ch
    tx = ch.Transmitter(name="tx", position=(1.0, 2.0, 3.0), polarization=(0.0, 0.0, 1.0),
                        power=2.5, orientation=(0.1, 0.2, 0.3))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    np.testing.assert_allclose(rebuilt.position, (1.0, 2.0, 3.0), atol=1e-4)
    np.testing.assert_allclose(rebuilt.polarization, (0.0, 0.0, 1.0), atol=1e-4)
    assert float(rebuilt.power) == pytest.approx(2.5)
    np.testing.assert_allclose(rebuilt.orientation, (0.1, 0.2, 0.3), atol=1e-4)


def test_receiver_optional_fields_default_none(adapter):
    import witwin.channel as ch
    rx = ch.Receiver(name="rx", position=(4.0, 5.0, 6.0))
    rebuilt = RT.by_name(RT.back(adapter, _scene(receivers=[rx])).receivers)["rx"]
    np.testing.assert_allclose(rebuilt.position, (4.0, 5.0, 6.0), atol=1e-4)
    assert rebuilt.polarization is None
    assert rebuilt.orientation is None


def test_receiver_grid_grid_shape_roundtrip(adapter):
    import witwin.channel as ch
    grid = ch.ReceiverGrid(name="rm", axis="z", position=1.5, bounds=((-6, 6), (-4, 4)),
                           grid_shape=(20, 16))
    rebuilt = RT.by_name(RT.back(adapter, _scene(receivers=[grid])).receivers)["rm"]
    assert rebuilt.axis == "z"
    assert float(rebuilt.position) == pytest.approx(1.5)
    assert rebuilt.bounds == ((-6.0, 6.0), (-4.0, 4.0))
    assert rebuilt.grid_shape == (20, 16)
    assert rebuilt.cell_size is None


def test_receiver_grid_cell_size_resolves_to_counts(adapter):
    import witwin.channel as ch
    # The RadioMap (Field) stores grid resolution as counts, so a cell_size grid
    # round-trips as the equivalent grid_shape (extent / cell_size).
    grid = ch.ReceiverGrid(name="rm", axis="x", position=0.0, bounds=((-5, 5), (-5, 5)),
                           cell_size=(0.25, 0.5))
    rebuilt = RT.by_name(RT.back(adapter, _scene(receivers=[grid])).receivers)["rm"]
    assert rebuilt.axis == "x"
    assert rebuilt.grid_shape == (40, 20)  # 10m / 0.25, 10m / 0.5
    assert rebuilt.cell_size is None


def test_ula_array_roundtrip(adapter):
    import witwin.channel as ch
    tx = ch.Transmitter(name="tx", position=(0, 0, 2),
                        array=ch.ULA(num_elements=4, spacing=0.5, axis="y", pattern="dipole"))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    arr = rebuilt.array
    assert isinstance(arr, ch.ULA)
    assert arr.num_elements == 4
    assert arr.pattern == "dipole"
    np.testing.assert_allclose([float(v) for v in arr.element_positions.y],
                               [-0.75, -0.25, 0.25, 0.75], atol=1e-4)


def test_upa_array_roundtrip(adapter):
    import witwin.channel as ch
    tx = ch.Transmitter(name="tx", position=(0, 0, 2),
                        array=ch.UPA(num_rows=2, num_cols=3, vertical_spacing=0.4,
                                     horizontal_spacing=0.6, polarization="vh"))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    arr = rebuilt.array
    assert isinstance(arr, ch.UPA)
    assert (arr.num_rows, arr.num_cols) == (2, 3)
    assert arr.num_elements == 6
    assert arr.polarization_slots == (((1 + 0j), 0j, 0j), (0j, (1 + 0j), 0j))


def test_custom_array_roundtrip(adapter):
    import witwin.channel as ch
    tx = ch.Transmitter(name="tx", position=(0, 0, 2),
                        array=ch.AntennaArray(element_positions=[(0, 0, 0), (0.1, 0.2, 0.3)],
                                              polarization="cross"))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    arr = rebuilt.array
    assert arr.num_elements == 2
    np.testing.assert_allclose([float(v) for v in arr.element_positions.x], [0.0, 0.1], atol=1e-4)
    assert arr.polarization_slots == ch.AntennaArray(
        element_positions=[(0, 0, 0)], polarization="cross").polarization_slots


def test_custom_complex_polarization_roundtrip(adapter):
    import witwin.channel as ch
    slots = [(1 + 0j, 0.5j, 0j)]
    tx = ch.Transmitter(name="tx", position=(0, 0, 2),
                        array=ch.AntennaArray(element_positions=[(0, 0, 0)], polarization=slots))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    assert rebuilt.array.polarization_slots == (((1 + 0j), 0.5j, 0j),)


def test_tx_default_array_roundtrips_to_none(adapter):
    import witwin.channel as ch
    # A plain Tx (single iso/V element) exports no explicit array -> platform default.
    tx = ch.Transmitter(name="tx", position=(0, 0, 2))
    rebuilt = RT.by_name(RT.back(adapter, _scene(transmitters=[tx])).transmitters)["tx"]
    assert rebuilt.array is None


def test_scene_default_array_baked_into_endpoint(adapter):
    import witwin.channel as ch
    # No more scene-default arrays: the default is baked into each endpoint that lacks one.
    tx = ch.Transmitter(name="tx", position=(0, 0, 2))
    scene = _scene(transmitters=[tx], tx_array=ch.ULA(num_elements=8, spacing=0.5, axis="x"))
    rebuilt = RT.back(adapter, scene)
    assert rebuilt.tx_array is None
    arr = rebuilt.transmitter("tx").array
    assert isinstance(arr, ch.ULA)
    assert arr.num_elements == 8


def test_endpoint_array_overrides_scene_default(adapter):
    import witwin.channel as ch
    # An explicit endpoint array wins over the scene default (which is then ignored).
    tx = ch.Transmitter(name="tx", position=(0, 0, 2),
                        array=ch.ULA(num_elements=2, spacing=0.5, axis="x"))
    scene = _scene(transmitters=[tx], tx_array=ch.ULA(num_elements=8, spacing=0.5, axis="x"))
    rebuilt = RT.back(adapter, scene)
    assert rebuilt.transmitter("tx").array.num_elements == 2
