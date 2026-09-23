"""Portable analysis records with embedded input and execution provenance."""

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import scipy

from .system_json import system_to_dict


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_analysis(path, *, system, settings, results):
    """Write strict JSON: undefined numerical results become null, never NaN.

    Settings must describe fields, sampling, wavelengths and analysis policies.
    Failure/status information belongs alongside results; null is not a zero.
    The embedded prescription and its canonical SHA-256 identify the physical
    model independently of the file name and of the platform's line endings.
    """
    prescription = system_to_dict(system)
    canonical = json.dumps(
        prescription, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    try:
        package_version = version("raytracer")
    except PackageNotFoundError:
        package_version = "uninstalled source checkout"
    implementation = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for source in sorted(package_root.rglob("*.py")):
        implementation.update(source.relative_to(package_root).as_posix().encode("utf-8") + b"\0")
        implementation.update(source.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    record = {
        "format": "geometrical-raytracer-analysis",
        "version": 1,
        "units": {"length": "mm", "wavelength": "um", "tilt": "deg"},
        "system_sha256": hashlib.sha256(canonical).hexdigest(),
        "system": prescription,
        "settings": _json_value(settings),
        "results": _json_value(results),
        "environment": {
            "raytracer": package_version,
            "python": platform.python_version(),
            "implementation_sha256": implementation.hexdigest(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    Path(path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record
