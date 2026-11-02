"""The public API: every name reachable through the packages' ``__all__``, from the root and each layer.

A name exported in several places counts once, under its shortest path (``Rays``, not ``rays.Rays``).
"""

import importlib
import types

import raytracer

LAYERS = ["geometry", "materials", "coatings", "interfaces", "elements", "system", "conditions", "rays", "sources",
          "tracing", "families", "io", "analysis", "optimize", "viz"]


def public_names() -> dict[str, object]:
    """``{path: object}`` for every public name, modules included."""

    paths: dict[int, set[str]] = {}
    objects: dict[int, object] = {}

    def walk(module, prefix):
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            path = f"{prefix}.{name}" if prefix else name
            paths.setdefault(id(obj), set()).add(path)
            objects[id(obj)] = obj
            if isinstance(obj, types.ModuleType):
                walk(obj, path)

    walk(raytracer, "")
    for layer in LAYERS:
        module = importlib.import_module(f"raytracer.{layer}")
        paths.setdefault(id(module), set()).add(layer)
        objects[id(module)] = module
        walk(module, layer)
    return {min(p, key=lambda k: (k.count("."), k)): objects[i] for i, p in paths.items()}
