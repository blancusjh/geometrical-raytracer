"""Dispersion models: AbbeMaterial approximation and real Sellmeier glasses."""

import pytest

from raytracer.core.materials import (
    AbbeMaterial,
    SellmeierMaterial,
    _SELLMEIER_CATALOG,
    default_materials,
    sellmeier_glass,
)

F_LINE, D_LINE, C_LINE = 0.48613, 0.58756, 0.65627


@pytest.mark.parametrize("name", sorted(_SELLMEIER_CATALOG))
def test_sellmeier_glass_matches_catalog_nd(name):
    nd, _, _ = _SELLMEIER_CATALOG[name]
    glass = sellmeier_glass(name)
    assert glass.index(D_LINE) == pytest.approx(nd, abs=1e-5)


@pytest.mark.parametrize("name", sorted(_SELLMEIER_CATALOG))
def test_sellmeier_glass_shows_normal_dispersion(name):
    glass = sellmeier_glass(name)
    assert glass.index(F_LINE) > glass.index(D_LINE) > glass.index(C_LINE)


def test_sellmeier_glass_unknown_name_raises():
    with pytest.raises(KeyError):
        sellmeier_glass("NOT-A-REAL-GLASS")


def test_default_materials_uses_real_sellmeier_for_sk16_and_sk4():
    lib = default_materials()
    assert isinstance(lib.resolve("SK16"), SellmeierMaterial)
    assert isinstance(lib.resolve("SK4"), SellmeierMaterial)
    assert isinstance(lib.resolve("N-SK16"), SellmeierMaterial)
    assert isinstance(lib.resolve("N-SK4"), SellmeierMaterial)


def test_default_materials_falls_back_to_abbe_for_f4():
    lib = default_materials()
    glass = lib.resolve("F4")
    assert isinstance(glass, AbbeMaterial)
    assert glass.index(D_LINE) == pytest.approx(1.617, abs=1e-6)


def test_default_materials_includes_common_glass_catalog():
    lib = default_materials()
    for name in ("N-BK7", "N-F2", "N-SF11", "N-LAK21"):
        glass = lib.resolve(name)
        assert isinstance(glass, SellmeierMaterial)
        assert glass.index(F_LINE) > glass.index(D_LINE) > glass.index(C_LINE)
