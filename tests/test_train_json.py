"""Stigmatic-train design files (``raytracer.io.train_json``)."""

from pathlib import Path

import pytest

from raytracer.design import StigmaticTrain
from raytracer.io import read_materials, read_train, write_train
from raytracer.optics.materials import (
    AIR,
    ConstantIndex,
    MaterialLibrary,
    default_materials,
    sellmeier_glass,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def full_library() -> MaterialLibrary:
    return read_materials(DATA / "materials/formlabs_resins.csv", into=default_materials())


def test_round_trip_preserves_the_design(tmp_path):
    train = StigmaticTrain(
        media=(AIR, sellmeier_glass("N-BK7"), AIR),
        vertices=(0.0, 8.0),
        conjugates=(-60.0, 27.051444733688673, 60.0),  # full precision survives
        semidiameter=8.0,
    )
    path = tmp_path / "doublet.json"
    write_train(train, path, comment="test design")
    loaded = read_train(path, materials=default_materials())
    assert loaded.conjugates == train.conjugates
    assert loaded.vertices == train.vertices
    assert loaded.indices == train.indices
    assert loaded.media[1].name == "N-BK7"
    assert loaded.wavelength_um == train.wavelength_um


def test_constant_index_media_rebuild_without_a_catalog_entry(tmp_path):
    train = StigmaticTrain.singlet(n=1.5, d0=-60.0, d1=300.0, d2=60.0, thickness=8.0)
    path = tmp_path / "singlet.json"
    write_train(train, path)
    loaded = read_train(path, materials=MaterialLibrary())  # empty library
    assert isinstance(loaded.media[1], ConstantIndex)
    assert loaded.indices == train.indices


def test_unknown_dispersive_medium_is_rejected(tmp_path):
    train = StigmaticTrain(
        media=(AIR, sellmeier_glass("N-BK7"), AIR),
        vertices=(0.0, 8.0),
        conjugates=(-60.0, 300.0, 60.0),
    )
    path = tmp_path / "doublet.json"
    write_train(train, path)
    with pytest.raises(KeyError, match="material library"):
        read_train(path, materials=MaterialLibrary())


def test_catalog_drift_is_rejected(tmp_path):
    train = StigmaticTrain(
        media=(AIR, sellmeier_glass("N-BK7"), AIR),
        vertices=(0.0, 8.0),
        conjugates=(-60.0, 300.0, 60.0),
    )
    path = tmp_path / "doublet.json"
    write_train(train, path)
    drifted = MaterialLibrary()
    drifted.register(ConstantIndex("N-BK7", 1.6))  # not the BK7 this was built with
    with pytest.raises(ValueError, match="drifted"):
        read_train(path, materials=drifted)


def test_non_train_file_is_rejected(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"kind": "something_else"}')
    with pytest.raises(ValueError, match="not a stigmatic-train file"):
        read_train(path, materials=MaterialLibrary())


@pytest.mark.parametrize(
    "relative",
    [
        "optical_systems/telescopes/bk7_f2_achromatic_doublet.json",
        "optical_systems/microscopes/lak21_sf11_bk7_20x.json",
        "optical_systems/microscopes/formlabs_clear_10x.json",
        "optical_systems/projectors/sk16_f2_bk7_sk4_slide_projector.json",
        "optical_systems/ultrawide/sf11_bk7_virtual_object.json",
    ],
)
def test_shipped_designs_load_and_build(relative):
    train = read_train(DATA / relative, materials=full_library())
    system = train.to_system()
    assert len(system.rows) == train.n_surfaces
    assert system.wavelength_um == train.wavelength_um
