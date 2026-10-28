"""Reading and writing optical-system data.

A different nature from math/optics/design: serialization format, not what
a system is or how light moves through it. Covers prescriptions (surface
tables) and material catalogs (dispersion models as data files).
"""

from .analysis_json import write_analysis
from .materials_csv import read_materials
from .prescription_csv import FORMAT_READERS, FORMAT_WRITERS, read_csv, write_csv
from .system_json import read_system, write_system
from .train_json import read_train, write_train

__all__ = [
    "write_analysis",
    "read_system",
    "write_system",
    "read_csv",
    "write_csv",
    "read_materials",
    "read_train",
    "write_train",
    "FORMAT_READERS",
    "FORMAT_WRITERS",
]
