"""C2 — endpoints on the Core 0.4 world (CPU).

A ``ChannelTx`` object becomes a ``witwin.core.AntennaState`` (role ``tx``), a
``ChannelRx`` object a point ``AntennaState`` (role ``rx``), and a ``RadioMap`` object a
``witwin.core.ReceiverGrid`` whose samples are the Field-region cell centers. The grid
axes follow the RadioMap plane's tangential world axes (u -> ``x_axis``, v -> ``y_axis``)
so a solved ``(T, n_v, n_u)`` map keeps the ``data[iv, iu]`` layout the plane visualizer
expects. Only the plain single isotropic element is ported; other antennas fail loudly.
"""
import numpy as np
import pytest


def _studio():
    from witwin_server import Scene as StudioScene
    return StudioScene()


def _tx(name="tx", position=(1.0, 2.0, 3.0)):
    from witwin_server.core import SceneObject
    from wt_channel.components.endpoints import TxComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    comp = obj.add_component(TxComponent())
    obj.get_component("Transform").position = list(position)
    return obj, comp


def _radiomap(name="rm", position=(5.0, 1.5, -3.0)):
    from witwin_server.core import SceneObject
    from wt_channel.components.radiomap import RadioMapComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    comp = obj.add_component(RadioMapComponent())
    obj.get_component("Transform").position = list(position)
    return obj, comp


def test_transmitter_becomes_core_antenna_state():
    from witwin.core import AntennaState
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _tx()
    comp.polarization = [0.0, 1.0, 0.0]
    comp.power = 2.5
    comp.use_orientation = True
    comp.orientation = [0.1, 0.2, 0.3]

    tx = EndpointMap.tx_to_platform(obj)

    assert isinstance(tx, AntennaState)
    assert tx.role == "tx"
    assert tx.metadata["name"] == "tx"
    np.testing.assert_allclose(tx.position.tolist(), [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(tx.polarization.tolist(), [0.0, 1.0, 0.0], atol=1e-6)
    assert float(tx.power_w) == pytest.approx(2.5)
    np.testing.assert_allclose(tx.orientation.tolist(), [0.1, 0.2, 0.3], atol=1e-6)
    assert tx.pattern.kind == "isotropic"


def test_receivers_default_to_studio_vertical_polarization():
    # Studio is Y up. witwin.channel defaults an unpolarized endpoint to +Z, which in
    # Studio is horizontal: a vertical Tx then met cross-polarized receivers and the
    # deterministic radio map lost ~31 dB of LoS against Monte Carlo (UCSD, median).
    from witwin_server.core import SceneObject
    from wt_channel.adapter.endpoint_map import EndpointMap
    from wt_channel.components.endpoints import RxComponent

    rmap_obj, _ = _radiomap()
    rx_obj = SceneObject(name="rx", mesh_type="Empty")
    rx_obj.add_component(RxComponent())

    for endpoint in (EndpointMap.radiomap_to_platform(rmap_obj), EndpointMap.rx_to_platform(rx_obj)):
        np.testing.assert_allclose(endpoint.polarization.tolist(), [0.0, 1.0, 0.0])


def test_receiver_explicit_polarization_wins():
    from witwin_server.core import SceneObject
    from wt_channel.adapter.endpoint_map import EndpointMap
    from wt_channel.components.endpoints import RxComponent

    obj = SceneObject(name="rx", mesh_type="Empty")
    comp = obj.add_component(RxComponent())
    comp.use_polarization = True
    comp.polarization = [1.0, 0.0, 0.0]

    np.testing.assert_allclose(EndpointMap.rx_to_platform(obj).polarization.tolist(), [1.0, 0.0, 0.0])


def test_receiver_optional_fields_default_none():
    from witwin_server.core import SceneObject
    from wt_channel.adapter.endpoint_map import EndpointMap
    from wt_channel.components.endpoints import RxComponent

    obj = SceneObject(name="rx", mesh_type="Empty")
    obj.add_component(RxComponent())
    obj.get_component("Transform").position = [4.0, 5.0, 6.0]

    rx = EndpointMap.rx_to_platform(obj)

    assert rx.role == "rx"
    np.testing.assert_allclose(rx.position.tolist(), [4.0, 5.0, 6.0], atol=1e-6)
    assert rx.orientation is None


def test_radiomap_becomes_receiver_grid_on_cell_centers():
    from witwin.core import ReceiverGrid
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _radiomap()
    comp.min_bound = [-10.0, 0.0, -20.0]
    comp.max_bound = [10.0, 0.0, 20.0]
    comp.resolution = [4.0, 1.0, 8.0]

    grid = EndpointMap.radiomap_to_platform(obj)

    assert isinstance(grid, ReceiverGrid)
    assert grid.metadata["name"] == "rm"
    assert grid.shape == (4, 8)  # (n_u = X, n_v = Z)
    np.testing.assert_allclose(grid.x_axis.tolist(), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grid.y_axis.tolist(), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(grid.spacing.tolist(), [5.0, 5.0])
    # min corner (-5, 1.5, -23) plus half a cell on each tangential axis.
    np.testing.assert_allclose(grid.position.tolist(), [-2.5, 1.5, -20.5], atol=1e-6)


def test_radiomap_resolution_mode_defaults_to_256_grid():
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _radiomap()

    grid = EndpointMap.radiomap_to_platform(obj)

    assert str(comp.grid_mode) == "Resolution"
    assert grid.shape == (256, 256)


def test_radiomap_ppw_mode_derives_grid_shape_from_bounds_and_wavelength():
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _radiomap()
    comp.grid_mode = "PPW"
    comp.points_per_wavelength = 4.0
    comp.frequency = 299792458.0
    comp.min_bound = [-10.0, 0.0, -5.0]
    comp.max_bound = [10.0, 0.0, 5.0]

    grid = EndpointMap.radiomap_to_platform(obj)

    assert grid.shape == (80, 40)
    np.testing.assert_allclose(grid.spacing.tolist(), [0.25, 0.25])


@pytest.mark.parametrize("field,value", [("array_kind", "ula"), ("pattern", "dipole")])
def test_unported_antenna_fails_loudly(field, value):
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _tx()
    setattr(comp, field, value)

    with pytest.raises(ValueError, match="not supported"):
        EndpointMap.tx_to_platform(obj)


def test_grid_roundtrips_through_studio():
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _radiomap()
    comp.min_bound = [-6.0, 0.0, -4.0]
    comp.max_bound = [6.0, 0.0, 4.0]
    comp.resolution = [12.0, 1.0, 16.0]
    grid = EndpointMap.radiomap_to_platform(obj)

    rebuilt_obj = EndpointMap.radiomap_to_studio(grid, frequency=2.4e9)
    rebuilt = EndpointMap.radiomap_to_platform(rebuilt_obj)

    assert rebuilt_obj.name == "rm"
    assert rebuilt.shape == grid.shape
    np.testing.assert_allclose(rebuilt.position.tolist(), grid.position.tolist(), atol=1e-5)
    np.testing.assert_allclose(rebuilt.spacing.tolist(), grid.spacing.tolist(), atol=1e-6)
    assert float(rebuilt_obj.get_component("RadioMap").frequency) == pytest.approx(2.4e9)


def test_transmitter_roundtrips_through_studio():
    from wt_channel.adapter.endpoint_map import EndpointMap

    obj, comp = _tx(position=(7.0, 8.0, 9.0))
    comp.power = 3.0
    comp.polarization = [0.0, 1.0, 0.0]
    tx = EndpointMap.tx_to_platform(obj)

    rebuilt = EndpointMap.tx_to_platform(EndpointMap.tx_to_studio(tx))

    np.testing.assert_allclose(rebuilt.position.tolist(), [7.0, 8.0, 9.0], atol=1e-6)
    np.testing.assert_allclose(rebuilt.polarization.tolist(), [0.0, 1.0, 0.0], atol=1e-6)
    assert float(rebuilt.power_w) == pytest.approx(3.0)
    assert rebuilt.metadata["name"] == "tx"
