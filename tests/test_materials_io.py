"""Material catalogs from data files (``raytracer.io.materials_csv``) and
real-material stigmatic trains (``StigmaticTrain.from_materials``)."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.design import StigmaticTrain
from raytracer.io import read_materials
from raytracer.optics.materials import (
    AIR,
    CauchyMaterial,
    MaterialLibrary,
    default_materials,
    sellmeier_glass,
)
from raytracer.propagation import SequentialTracer, trace_from_object

DATA = Path(__file__).resolve().parents[1] / "data"
FORMLABS = DATA / "formlabs_resins.csv"

#: The manufacturer's measured table the shipped Cauchy fit was built from
#: ("Optical properties of selected Formlabs SLA resins", Clear Resin V4).
FORMLABS_MEASURED = [(0.486, 1.514), (0.546, 1.511), (0.589, 1.508), (0.656, 1.505)]


def test_formlabs_catalog_reproduces_the_measured_table():
    library = read_materials(FORMLABS)
    resin = library["FORMLABS_CLEAR_V4"]
    assert isinstance(resin, CauchyMaterial)
    for wavelength_um, measured in FORMLABS_MEASURED:
        assert resin.index(wavelength_um) == pytest.approx(measured, abs=5e-4)


def test_formlabs_v4_abbe_number_is_in_the_published_range():
    resin = read_materials(FORMLABS)["FORMLABS_CLEAR_V4"]
    nd = resin.index(0.58756)
    vd = (nd - 1.0) / (resin.index(0.48613) - resin.index(0.65627))
    assert 50.0 < vd < 60.0  # Optica OME 11, 3392 (2021) reports ~57


def test_read_into_extends_an_existing_library():
    library = read_materials(FORMLABS, into=default_materials())
    assert "N-BK7" in library and "FORMLABS_CLEAR_V4" in library
    assert isinstance(library, MaterialLibrary)


def test_reader_rejects_unknown_model_and_bad_arity(tmp_path):
    bad_model = tmp_path / "bad_model.csv"
    bad_model.write_text("name,model,c1\nX,polynomial,1.5\n")
    with pytest.raises(ValueError, match="unknown model"):
        read_materials(bad_model)

    bad_arity = tmp_path / "bad_arity.csv"
    bad_arity.write_text("name,model,c1,c2\nX,constant,1.5,0.1\n")
    with pytest.raises(ValueError, match="exactly c1"):
        read_materials(bad_arity)

    no_name = tmp_path / "no_name.csv"
    no_name.write_text("name,model,c1\n,constant,1.5\n")
    with pytest.raises(ValueError, match="empty material name"):
        read_materials(no_name)


def test_comment_lines_are_ignored(tmp_path):
    path = tmp_path / "with_comments.csv"
    path.write_text("# a comment\nname,model,c1\n# another\nX,constant,1.25\n")
    assert read_materials(path)["X"].index(0.5) == 1.25


# -- real-material trains ----------------------------------------------------


@pytest.fixture(scope="module")
def bk7_train() -> StigmaticTrain:
    return StigmaticTrain.from_materials(
        (AIR, sellmeier_glass("N-BK7"), AIR),
        vertices=(0.0, 8.0),
        conjugates=(-60.0, 300.0, 60.0),
        wavelength_um=0.58756,
        semidiameter=8.0,
    )


def _axial_focus_z(train: StigmaticTrain, wavelength_um: float) -> float:
    system = train.to_system()
    system.wavelength_um = wavelength_um
    system._rebuild()
    result = trace_from_object(SequentialTracer(system), (0.0, 0.0), (0.0, 0.05))
    d = result.direction / np.linalg.norm(result.direction)
    return float(result.image_point[2] - result.image_point[1] * d[2] / d[1])


def test_from_materials_derives_indices_at_the_design_wavelength(bk7_train):
    assert bk7_train.indices[1] == pytest.approx(
        sellmeier_glass("N-BK7").index(0.58756)
    )
    assert bk7_train.to_system().wavelength_um == 0.58756


def test_material_train_is_exact_at_design_wavelength_and_chromatic_off_it(bk7_train):
    assert _axial_focus_z(bk7_train, 0.58756) == pytest.approx(60.0, abs=1e-6)
    shift_fc = _axial_focus_z(bk7_train, 0.65627) - _axial_focus_z(bk7_train, 0.48613)
    assert shift_fc > 0.5  # a BK7 singlet's real axial color, in mm


def test_with_conjugates_preserves_the_materials(bk7_train):
    moved = bk7_train.with_conjugates([250.0])
    assert moved.materials == bk7_train.materials
    assert moved.wavelength_um == bk7_train.wavelength_um
    assert moved.indices == bk7_train.indices


def test_mismatched_material_index_is_rejected():
    with pytest.raises(ValueError, match="from_materials"):
        StigmaticTrain(
            indices=(1.0, 1.9, 1.0),
            vertices=(0.0, 8.0),
            conjugates=(-60.0, 300.0, 60.0),
            materials=(AIR, sellmeier_glass("N-BK7"), AIR),
            wavelength_um=0.58756,
        )


def test_materials_without_wavelength_are_rejected():
    with pytest.raises(ValueError, match="wavelength_um"):
        StigmaticTrain(
            indices=(1.0, 1.5168, 1.0),
            vertices=(0.0, 8.0),
            conjugates=(-60.0, 300.0, 60.0),
            materials=(AIR, sellmeier_glass("N-BK7"), AIR),
        )
