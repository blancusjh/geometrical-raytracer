"""Provenance and supported wavelength interval of a scalar index model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialMetadata:
    """Declared evidence, not a claim of accuracy outside measured conditions.

    The interval is the supported range of this implementation, in micrometres.
    A missing interval means unknown, not unlimited physical validity. Temperature
    is descriptive: no thermo-optic correction is applied by these models.
    """

    source: str = ""
    wavelength_range_um: tuple[float, float] | None = None
    reference_temperature_c: float | None = None
    index_reference: str = "unspecified"
    notes: str = ""

    def __post_init__(self):
        if self.wavelength_range_um is not None:
            bounds = tuple(self.wavelength_range_um)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds):
                raise ValueError("wavelength interval must contain two finite bounds")
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError("wavelength interval must satisfy 0 < lower <= upper")
            object.__setattr__(self, "wavelength_range_um", bounds)
        if self.reference_temperature_c is not None and not math.isfinite(
            self.reference_temperature_c
        ):
            raise ValueError("reference temperature must be finite")

    def check_wavelength(self, wavelength_um: float, name: str) -> None:
        if self.wavelength_range_um is None:
            return
        lower, upper = self.wavelength_range_um
        if not math.isfinite(wavelength_um) or not lower - 1e-12 <= wavelength_um <= upper + 1e-12:
            raise ValueError(
                f"{name}: wavelength {wavelength_um} um outside supported interval "
                f"[{lower}, {upper}] um; extrapolation is not enabled"
            )
