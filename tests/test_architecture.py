"""The layering is a claim about the code, so it is checked like one.

``raytracer`` is organized by the nature of each piece, and the packages
form a strictly descending chain: every import at module level goes to a
package *earlier* in :data:`LAYERS`. This test reads the AST of every
module and enforces that.

Two kinds of import are exempt, both deliberate:

- imports inside a function body, deferred to call time (this is how
  ``OpticalSystem.from_prescription`` reaches the ``io`` reader registry
  without the data model depending on a file format at import time);
- imports inside ``if TYPE_CHECKING:``, erased at runtime (this is how
  ``surfaces`` annotates a ``Ray`` it never imports).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "raytracer"

#: The layer order the README documents, lowest first.
LAYERS = [
    "math",
    "surfaces",
    "optics",
    "design",
    "propagation",
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


def test_math_depends_on_no_other_layer():
    """The bottom of the chain: geometry with no optics in it."""

    for path in _modules_in("math"):
        assert _imported_layers(path) <= {"math"}


def test_surfaces_reaches_optics_only_for_annotations():
    """A surface is hit-testable without knowing what a ray is."""

    ray_annotation = PACKAGE / "surfaces" / "surface.py"
    assert "optics" not in _imported_layers(ray_annotation)
    assert "TYPE_CHECKING" in ray_annotation.read_text(encoding="utf-8")


def test_design_reaches_io_only_from_inside_a_method():
    """``from_prescription`` is a convenience on the data model; the data
    model itself stays independent of any file format."""

    system = PACKAGE / "design" / "system.py"
    assert "io" not in _imported_layers(system)
    assert "from ..io.prescription_csv import" in system.read_text(encoding="utf-8")
