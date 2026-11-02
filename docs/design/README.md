# Design: usage first, then ontology

**Status.** Implemented: the physical model — placed elements in an
ambient medium, and light as `Rays` that both a sequential and a
non-sequential tracer carry through it — and Tier 0, how a system is used:
conditions, configurations, first-order optics, ray aiming and tracing a
configuration (`usage.md` §2–§6 and §12, `ontology.md` §1–4). The three
safeguards below are in force. Everything else here is the
plan, and its usage is fixed before any code is written for it.

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
   ontology sound, how each law is enforced, and the decisions still open.
3. [`glossary.md`](glossary.md) — every public name, defined once.

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

- the **layer test** (`tests/test_architecture.py`): the chain of packages,
  and no plotting or windowing library below `viz`;
- the **glossary test** (`tests/test_glossary.py`): every public name is
  defined once in `glossary.md`, so a new concept cannot enter the API
  without a definition;
- the **executable usage specification** (`tests/test_usage.py`): the
  blocks of `usage.md` run in order; a planned one is an expected failure,
  strictly, until its feature lands, so the usage contract cannot silently
  drift from the implementation.
