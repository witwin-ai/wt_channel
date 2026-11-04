"""C0 — the Studio scene as a Core 0.4 world (CPU).

``ChannelAdapter.build`` turns the live Studio scene into an immutable
``witwin.core.Scene``: platform-geometry objects and plain Studio meshes (e.g. an
imported street map) become ``Structure`` s, Tx / Rx / RadioMap objects become
endpoints. Plain meshes default to ITU concrete at the solve frequency; an explicit
``MaterialPreset`` or a named scalar ``Material`` wins. The channel config rides the
world's metadata so a Core world round-trips through Studio.
"""
import numpy as np
import pytest
import torch


def _studio():
    from witwin_server.api import Scene as StudioScene
    return StudioScene()


def _quad_mesh(name="ground", *, position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    from witwin_server.api import SceneObject

    obj = SceneObject(name=name, mesh_type="Custom")
    vertices = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
    obj.get_component("Mesh").set_mesh_data(vertices, faces, notify=False)
    transform = obj.get_component("Transform")
    transform.position = list(position)
    transform.scale = list(scale)
    return obj


def _add_tx(studio, name="tx", position=(0.0, 10.0, 0.0)):
    from witwin_server.api import SceneObject
    from wt_channel.components.endpoints import TxComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    obj.add_component(TxComponent())
    obj.get_component("Transform").position = list(position)
    studio.add_object(obj)
    return obj


def _add_radiomap(studio, name="rm"):
    from witwin_server.api import SceneObject
    from wt_channel.components.radiomap import RadioMapComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    comp = obj.add_component(RadioMapComponent())
    comp.resolution = [4.0, 1.0, 4.0]
    studio.add_object(obj)
    return obj


def test_build_returns_core_world_with_mesh_tx_and_grid(adapter):
    from witwin.core import ReceiverGrid, Scene

    studio = _studio()
    studio.add_object(_quad_mesh())
    _add_tx(studio)
    _add_radiomap(studio)

    world = adapter.build(studio, frequency=3.5e9)

    assert isinstance(world, Scene)
    assert [s.name for s in world.structures] == ["ground"]
    roles = sorted(endpoint.role for endpoint in world.endpoints)
    assert roles == ["rx", "tx"]
    assert any(isinstance(endpoint, ReceiverGrid) for endpoint in world.endpoints)
    assert world.metadata["channel"]["frequency_hz"] == pytest.approx(3.5e9)


def test_plain_mesh_defaults_to_itu_concrete_at_frequency(adapter):
    studio = _studio()
    studio.add_object(_quad_mesh())

    material = adapter.build(studio, frequency=3.5e9).structures[0].material

    assert material.name == "concrete"
    assert float(material.sigma_e) == pytest.approx(0.0462 * 3.5 ** 0.7822)


def test_plain_mesh_world_vertices_apply_transform(adapter):
    studio = _studio()
    studio.add_object(_quad_mesh(position=(10.0, 2.0, -5.0), scale=(3.0, 1.0, 2.0)))

    geometry = adapter.build(studio, frequency=3.5e9).structures[0].geometry
    world = geometry.world_vertices.detach().cpu().numpy()

    np.testing.assert_allclose(world.min(axis=0), [7.0, 2.0, -7.0], atol=1e-5)
    np.testing.assert_allclose(world.max(axis=0), [13.0, 2.0, -3.0], atol=1e-5)


def test_material_preset_overrides_default(adapter):
    from wt_channel.components.materials import MaterialPresetComponent

    studio = _studio()
    obj = _quad_mesh()
    preset = obj.add_component(MaterialPresetComponent())
    preset.preset = "metal"
    studio.add_object(obj)

    material = adapter.build(studio, frequency=3.5e9).structures[0].material

    assert material.name == "metal"
    assert float(material.sigma_e) == pytest.approx(1.0e7)


def test_studio_itu_pick_is_evaluated_at_frequency(adapter):
    studio = _studio()
    obj = _quad_mesh()
    obj.get_component("Material").material_name = "Very dry ground"
    studio.add_object(obj)

    material = adapter.build(studio, frequency=3.5e9).structures[0].material

    assert material.name == "very_dry_ground"
    assert float(material.sigma_e) == pytest.approx(0.00015 * 3.5 ** 2.52)


def test_hand_edited_scalars_win_over_default(adapter):
    studio = _studio()
    obj = _quad_mesh()
    material_comp = obj.get_component("Material")
    material_comp.eps_r = [3.0, 0.0]
    material_comp.sigma_e = 0.5
    studio.add_object(obj)

    material = adapter.build(studio, frequency=3.5e9).structures[0].material

    assert float(np.real(material.eps_r)) == pytest.approx(3.0)
    assert float(material.sigma_e) == pytest.approx(0.5)


def test_material_preset_band_is_validated(adapter):
    from wt_channel.components.materials import MaterialPresetComponent

    studio = _studio()
    obj = _quad_mesh()
    obj.add_component(MaterialPresetComponent()).preset = "very_dry_ground"
    studio.add_object(obj)

    with pytest.raises(ValueError, match="very_dry_ground"):
        adapter.build(studio, frequency=50e9)


def test_build_filters_to_named_transmitter_and_receiver(adapter):
    studio = _studio()
    studio.add_object(_quad_mesh())
    _add_tx(studio, "tx_a", (0.0, 5.0, 0.0))
    _add_tx(studio, "tx_b", (9.0, 5.0, 0.0))
    _add_radiomap(studio, "rm_a")
    _add_radiomap(studio, "rm_b")

    world = adapter.build(studio, frequency=3.5e9, transmitter="tx_b", receiver="rm_a")

    names = sorted(endpoint.metadata["name"] for endpoint in world.endpoints)
    assert names == ["rm_a", "tx_b"]


def test_has_structures_counts_plain_meshes(adapter):
    studio = _studio()
    assert adapter.has_structures(studio) is False
    studio.add_object(_quad_mesh())
    assert adapter.has_structures(studio) is True


def test_export_empty_scene_is_empty(adapter):
    world = adapter.to_platform(_studio())
    assert tuple(world.structures) == ()
    assert tuple(world.endpoints) == ()


def test_detect_claims_only_core_worlds_with_endpoints(adapter):
    from witwin.core import AntennaState, Scene

    tx = AntennaState(1, "tx", torch.tensor([0.0, 1.0, 0.0]), metadata={"name": "tx"})
    assert adapter.detect(Scene(endpoints=[tx])) is True
    assert adapter.detect(Scene()) is False
    assert adapter.detect(object()) is False


def test_core_world_roundtrips_through_studio(adapter):
    studio = _studio()
    studio.add_object(_quad_mesh())
    _add_tx(studio)
    _add_radiomap(studio)
    world = adapter.build(studio, frequency=2.4e9)

    rebuilt = adapter.to_platform(adapter.to_studio(world))

    assert [s.name for s in rebuilt.structures] == ["ground"]
    assert sorted(e.metadata["name"] for e in rebuilt.endpoints) == ["rm", "tx"]
    assert rebuilt.metadata["channel"]["frequency_hz"] == pytest.approx(2.4e9)


def _tag_rf_transparent(obj):
    from witwin_server.api import StructureMetaComponent

    meta = obj.add_component(StructureMetaComponent())
    meta.tags = ["rf:transparent"]
    return obj


def test_rf_transparent_meshes_stay_out_of_the_radio_world(adapter):
    # Imported street maps draw roads as flat ribbons and trees as canopy blobs. As
    # default-concrete walls they added spurious diffraction edges and cost ~40 % of a
    # deterministic UCSD refine (10.6 s -> 6.5 s without them, same coverage).
    from wt_channel.adapter.channel_adapter import RF_TRANSPARENT_TAG

    studio = _studio()
    studio.add_object(_quad_mesh())
    studio.add_object(_tag_rf_transparent(_quad_mesh("roads")))
    _add_tx(studio)
    _add_radiomap(studio)

    world = adapter.build(studio, frequency=3.5e9)

    assert RF_TRANSPARENT_TAG == "rf:transparent"
    assert [s.name for s in world.structures] == ["ground"]


def test_has_structures_ignores_rf_transparent_meshes(adapter):
    studio = _studio()
    studio.add_object(_tag_rf_transparent(_quad_mesh("roads")))
    assert adapter.has_structures(studio) is False
