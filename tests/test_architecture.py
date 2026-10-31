"""The layering is a claim about the code, so it is checked like one.

``raytracer`` is organized by the nature of each piece, and the packages
form a strictly descending chain: every import at module level goes to a
package *earlier* in :data:`LAYERS`. This test reads the AST of every
module and enforces that.

Two kinds of import are exempt, both deliberate:

- imports inside a function body, deferred to call time (this is how
  ``OpticalSystem.from_table`` reaches its builder module, and how a
  ``Prescription`` builds a tracer, without importing them at load time);
- imports inside ``if TYPE_CHECKING:``, erased at runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "raytracer"

#: The layer order the README documents, lowest first.
LAYERS = [
    "geometry",
    "materials",
    "coatings",
    "interfaces",
    "elements",
    "system",
    "rays",
    "sources",
    "tracing",
    "families",
    "io",
    "analysis",
    "optimize",
    "viz",
]
RANK = {name: i for i, name in enumerate(LAYERS)}


def _tests_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_nodes(node: ast.AST):
    """Every import executed when the module is first imported."""

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.If) and _tests_type_checking(child.test):
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child
        else:
            yield from _import_time_nodes(child)


def _imported_layers(path: Path) -> set[str]:
    """The layers *path* imports at module level, by name."""

    parts = path.relative_to(PACKAGE).parts
    tree = ast.parse(path.read_text(encoding="utf-8"))
    layers: set[str] = set()

    for node in _import_time_nodes(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import: "." is the containing package, and each
                # extra dot climbs one more level toward raytracer/.
                base = list(parts[: max(0, len(parts) - node.level)])
                targets.append(".".join(base + (node.module or "").split(".")))
            elif node.module and node.module.startswith("raytracer."):
                targets.append(node.module[len("raytracer.") :])
        else:
            targets += [
                alias.name[len("raytracer.") :]
                for alias in node.names
                if alias.name.startswith("raytracer.")
            ]
        layers.update(t.split(".")[0] for t in targets if t.split(".")[0] in RANK)

    return layers


def _modules_in(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).rglob("*.py"))


def test_every_layer_has_modules():
    """A typo in LAYERS would otherwise make this whole file vacuous."""

    for layer in LAYERS:
        assert _modules_in(layer), f"no modules found under raytracer/{layer}"


@pytest.mark.parametrize("layer", LAYERS)
def test_imports_only_reach_downward(layer: str):
    for path in _modules_in(layer):
        for imported in _imported_layers(path):
            if imported == layer:
                continue
            assert RANK[imported] < RANK[layer], (
                f"{path.relative_to(PACKAGE.parent)} imports {imported!r}, which "
                f"sits above {layer!r} in the layer order {LAYERS}"
            )


def test_geometry_depends_on_no_other_layer():
    """The bottom of the chain: shape and place, with no optics in it."""

    for path in _modules_in("geometry"):
        assert _imported_layers(path) <= {"geometry"}


def test_elements_know_nothing_of_systems_or_engines():
    """An element is a physical thing; how it is arranged or traced is not its business."""

    for path in _modules_in("elements"):
        assert _imported_layers(path).isdisjoint({"system", "tracing", "sources"})


def test_rays_are_light_alone():
    """A ray is light: what it meets, and how it is traced, are no part of it."""

    for path in _modules_in("rays"):
        assert not _imported_layers(path)


def test_system_reaches_io_and_tracing_only_lazily():
    """Builders defer to ``table``; the data model never imports an engine or a file format."""

    for path in _modules_in("system"):
        assert _imported_layers(path).isdisjoint({"tracing", "io", "analysis"})
