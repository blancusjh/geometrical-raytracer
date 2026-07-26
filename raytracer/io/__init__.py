"""Reading and writing optical-system data.

A different nature from math/physics/design: this is about serialization
format (a CSV dialect, today), not about what a system *is* or how light
moves through it.
"""

from .prescription_csv import FORMAT_READERS, FORMAT_WRITERS, read_csv, write_csv

__all__ = ["read_csv", "write_csv", "FORMAT_READERS", "FORMAT_WRITERS"]
