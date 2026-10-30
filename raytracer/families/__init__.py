"""Families: parametrised generators of optical systems.

A family is not a system. Its members are systems, built on demand, and
its parameters are the degrees of freedom a design search moves through.
"""

from .stigmatic import DESIGN_WAVELENGTH_UM, StigmaticTrain

__all__ = ["DESIGN_WAVELENGTH_UM", "StigmaticTrain"]
