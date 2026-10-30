# Design: usage first, then ontology

**Status: proposal.** Nothing described in this folder is implemented
yet unless the migration table in `ontology.md` says so. Its purpose is to
fix *how the finished library is used*, and *what each noun in it is*,
before any of Tier 0–3 is written.

> The purpose of abstraction is not to be vague, but to create a new
> semantic level in which one can be absolutely precise.
> — E. W. Dijkstra, *The Humble Programmer* (EWD340)

The method follows from that sentence and from Dijkstra's *separation of
concerns* (EWD447): first write the programs a user will write, as if the
library were finished; then read the nouns and verbs off those programs;
then fix each noun's nature and the laws that keep natures apart. Code
comes last, and it is written against this text, not the other way round.

## Reading order

1. [`usage.md`](usage.md) — the library as if it existed. One continuous
   session, from describing a lens to tolerancing it, tracing it
   non-sequentially and propagating a beam through it. Every planned
   feature appears here at least once.
2. [`ontology.md`](ontology.md) — what each noun *is*: its nature, its
   package, whether it is stated or derived. The laws that keep the
   ontology sound, and how each law is enforced. The map from today's
   names to the new ones, and the decisions still open.

## Scope

Covered: Tier 0 (system model), Tier 1 (core analyses), Tier 2 except the
general optimiser (glass catalogues, import/export, tolerancing, surface
types, multi-configuration), Tier 3 except thermal analysis (polarisation
and coatings, non-sequential tracing, image simulation, Gaussian beams and
physical optics propagation).

Not covered, but left room for: the general optimiser (it reuses
`Parameter` and `Metric` from this design), thermal analysis (it would be
one more kind of `Parameter` plus temperature in `Conditions`).

## How the design is protected

A design that lives only in prose erodes. Three mechanisms, described in
`ontology.md` §7, turn it into something `pytest` checks:

- the **layer test** (`tests/test_architecture.py`, already in place),
  extended to the new layers and to third-party imports;
- a **glossary test**: every public name must be defined once in the
  glossary, so a new concept cannot enter the API without a definition;
- an **executable usage specification**: the snippets of `usage.md` run as
  tests, each marked `xfail(strict=True)` until its feature lands, so the
  usage contract cannot silently drift from the implementation.
