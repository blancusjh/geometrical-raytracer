"""Reading and writing optical-system data.

A different nature from math/optics/design: serialization format, not what
a system is or how light moves through it.
"""

from .prescription_csv import FORMAT_READERS, FORMAT_WRITERS, read_csv, write_csv

__all__ = ["read_csv", "write_csv", "FORMAT_READERS", "FORMAT_WRITERS"]
