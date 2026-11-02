"""``docs/design/usage.md`` runs.

Its code blocks are executed in order, in one namespace, like a notebook.
A block preceded by ``<!-- planned: <feature> -->`` is expected to fail
until its feature lands — and strictly: once it passes, this test fails
until the marker is removed. The usage contract and the implementation
cannot drift apart without a red test.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs" / "design" / "usage.md"
BLOCK = re.compile(r"^(?:<!-- planned: (?P<planned>[^>]*?) -->\n)?```python\n(?P<code>.*?)^```$", re.M | re.S)


def _blocks() -> list[tuple[int, str, str | None]]:
    text = USAGE.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start("code")) + 1, m.group("code"), m.group("planned")) for m in BLOCK.finditer(text)]


BLOCKS = _blocks()
_namespace: dict = {}
_run: set[int] = set()


def _execute(index: int) -> None:
    _run.add(index)
    line, code, _ = BLOCKS[index]
    exec(compile(code, f"usage.md:{line}", "exec"), _namespace)


def _catch_up(index: int) -> None:
    """Run the blocks before *index* not run yet, so a selected block finds what it builds on."""

    for i in range(index):
        if i not in _run:
            try:
                _execute(i)
            except Exception:
                if not BLOCKS[i][2]:
                    raise


@pytest.mark.parametrize(
    "index", range(len(BLOCKS)),
    ids=[f"line{line}" + (f"-planned-{planned.replace(' ', '-')}" if planned else "") for line, _, planned in BLOCKS],
)
def test_usage_block(index, monkeypatch, request):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    _catch_up(index)
    planned = BLOCKS[index][2]
    if planned:
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"planned: {planned}"))
    _execute(index)


def test_there_is_something_to_run():
    assert len(BLOCKS) > 20 and any(planned is None for _, _, planned in BLOCKS)
