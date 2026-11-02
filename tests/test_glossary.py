"""The glossary and the public API name the same things (ontology §7)."""

import re
from pathlib import Path

from public_api import public_names

GLOSSARY = Path(__file__).resolve().parents[1] / "docs" / "design" / "glossary.md"
ENTRY = re.compile(r"^\| `([^`]+)` \| (\w+) \| (.+) \|$", re.M)


def _entries() -> dict[str, tuple[str, str]]:
    return {m.group(1): (m.group(2), m.group(3)) for m in ENTRY.finditer(GLOSSARY.read_text(encoding="utf-8"))}


def test_every_public_name_is_defined_in_the_glossary():
    missing = sorted(set(public_names()) - set(_entries()))
    assert not missing, f"public names with no glossary entry: {missing}"


def test_every_glossary_entry_names_something_public():
    stale = sorted(set(_entries()) - set(public_names()))
    assert not stale, f"glossary entries that no longer name anything public: {stale}"


def test_every_entry_says_what_it_is():
    vague = sorted(name for name, (_, what) in _entries().items() if len(what) < 12 or "TODO" in what)
    assert not vague, f"glossary entries without a real definition: {vague}"
