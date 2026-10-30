# Ontology: what each noun is, and the laws that keep it so

`usage.md` fixes how the library is used. This document reads the nouns
off that usage, gives each one a single nature and a single home, and
states the laws that keep the natures apart. **‹law n›** in `usage.md`
cites §6 here.

Sections 1–3 describe the architecture as implemented. Sections 4–5 and
the laws cover the rest of the plan as well.

---

## 1. The questions, and the chain of packages

Each package answers one question. Every module-level import goes to a
package earlier in the chain, and `tests/test_architecture.py` checks it:

```
geometry → materials → coatings → interfaces → elements → system → sources
         → tracing → families → io → analysis → optimize → viz
```

| Package | Question | Nature |
|---|---|---|
| `geometry` | Where is it, and what shape is it? (`Rotation`, `Pose`, surfaces, apertures, solids, intersection) | geometry, no optics |
| `materials` | What fills space? | physics of media |
| `coatings` | How much light does a boundary reflect and transmit? | physics of boundaries (energy) |
| `interfaces` | What is a surface as an optical boundary, and which laws act there? | physics of boundaries (direction) |
| `elements` | What physical things is an instrument made of? | description |
| `system` | How are they arranged, and what follows from the arrangement? | description and derived facts |
| `sources` | Where does light start? | description of emission |
| `tracing` | How does light get from here to there? | algorithms |
| `families` | Which systems does a design family generate? | generators of descriptions |
| `io` | How is a design stored? | serialisation |
| `analysis` | What do we measure? | measurement |
| `optimize` | What is better? | directed variation |
| `viz` | How is it shown? | presentation |

---

## 2. The nouns

**Stated** means the user says it and the object stores it. **Derived**
means it is computed from stated things. A derived quantity is never
stored as authoritative state. **‹law 3›**

| Noun | Is | Package | Stated / derived |
|---|---|---|---|
| `Rotation`, `Pose` | orientation; position + orientation (local → world) | geometry | stated |
| `Surface` and its family | a smooth boundary `z = s(x, y)` in its own frame; geometry only | geometry.surfaces | stated |
| `Aperture` (`Circle`, `Annulus`, `Rectangle`, `Polygon`) | a region of a surface's local (x, y) plane | geometry.apertures | stated |
| `Solid` (`Sphere`, `Cylinder`, `Box`, `Mesh`) | a closed region of space | geometry.solids | stated |
| `Material` | a medium: `index(λ)` | materials | stated |
| `Coating` | power split at a boundary: `Uncoated` (Fresnel), `AR`, `PerfectMirror`, `IdealBeamSplitter`, `Absorbing` | coatings | stated |
| `Interface` | a surface + its coating + its clear aperture | interfaces | stated |
| `Element` (`Lens`, `Mirror`, `BeamSplitter`, `Prism`, `Stop`, `Sensor`) | a named physical thing with a pose | elements | stated |
| `Assembly` | a named rigid group of components | system | stated |
| `OpticalSystem` | components in an ambient medium, optionally a stated path and stop | system | stated |
| `PhysicalFace` | one physical boundary, with the medium on each side | system | **derived** (coincident faces merge: contact) |
| `Step`, `path` | the sequential route: boundary + event | system | stated, or **derived** for a coaxial system |
| `Rays`, `Source` | a batch of rays; an emitter | sources | stated |
| `SequentialTracer`, `NonSequentialTracer` | the engines; the tracer holds the facts of use it needs (wavelength, object plane) | tracing | — |
| `StigmaticTrain` | a family of stigmatic systems; `.system()` is a member | families | stated |

### Surface hierarchy

Inheritance follows mathematical specialization. A subclass is a subset
of its parent's family.

```
Surface
├── SurfaceOfRevolution
│   ├── Conic ─┬─ Plane         (c = 0: every family's flat member)
│   │          ├─ Ellipsoid ── Sphere   (a sphere is the ellipsoid of equal axes)
│   │          ├─ Paraboloid
│   │          └─ Hyperboloid
│   ├── CartesianSurface        (Descartes' stigmatic surface)
│   └── Asphere ─┬─ EvenAsphere
│                └─ QTypeAsphere
├── ToroidalSurface ── CylindricalSurface   (a cylinder is the toroid of infinite rotation radius)
└── FreeformSurface
```

Where specialization is a lattice rather than a tree, the tree follows the
defining form. A conic is the zero-departure member of *every* asphere
family. The tree can carry only one of those relations, so it is
expressed by `Asphere.base` and `EvenAsphere.from_conic` instead.

---

## 3. Anatomy of a boundary

```
  element A (material, pose)        element B or ambient
      ┌───────────── Interface ─────────────┐
      │ surface         what it looks like  │   ← geometry
      │ coating         how much goes       │   ← physics (energy)
      │ clear_aperture  where it is specified│
      └─────────────────────────────────────┘
  medium on each side  ← derived by the system from the elements
  what light does      ← derived: media + coating + laws (never stated)
```

* A mirror is a surface with a reflective coating.
* A cemented face is two element faces that coincide.
* A beam splitter's branch is a choice of the sequential path, stated
  there (`"BS:reflect"`), because physics takes both branches.

---

## 4. Coordinates of the condition space

Every sequential quantity is a function of
**(configuration, field, wavelength, pupil, focus)**.

* *configuration* selects the instrument and its use;
* *field* is `Field(x, y)` in the declared field type, or normalised `(hx, hy)`;
* *wavelength* is in µm, an index into the spectrum, `"primary"` or `"all"`;
* *pupil* is normalised `(px, py)`. Aiming maps it onto the stop;
* *focus* is the image surface as described, unless the call says
  `focus=` explicitly.

An analysis fixes some of these and integrates or samples the others. A
`sweep` varies one of them. That is the whole algebra.

---

## 5. Values, not objects with state

`Rotation`, `Pose`, every `Surface`, `Aperture`, `Solid`, `Material`,
`Coating`, `Interface`, `Element`, `Assembly` and `OpticalSystem` are
immutable values, compared and hashed by their parameters. So will be the
planned `Conditions`, `Configuration`, `Parameter`, `Tolerance` and every
`Result`. `replace(...)`, `placed(...)`, `rotated(...)` and
`Parameter.set(...)` return new values.

Consequences that the usage depends on:

* a tolerancing trial *is* a configuration, so every analysis applies to it;
* a multi-configuration *is* a tuple, so a sweep over it needs no special case;
* an analysis can cache on its subject, because equal subjects give equal results;
* there is no hidden ordering between calls.

Performance-sensitive engines may hold private mutable buffers. These are
never reachable from a value.

---

## 6. The laws

Each law states an invariant, why it exists, and how it is enforced. A
law with no enforcement is a wish. Enforcement is either a test or, where
no test can see it, a review rule written in `CONTRIBUTING.md`.

**Law 1 — One concept, one slot.** Each concept is expressible in exactly
one place: a mirror is a coating, GRIN is a medium, a stop is an element,
the media of a boundary are derived from the elements around it. *Why:* combinations stay a
product, not a list of special cases. *Enforced:* the glossary test (§7);
review.

**Law 2 — Descriptions are values, and hardware is not use.**
`OpticalSystem` holds no wavelength, field, aperture spec or object
distance. All descriptions are immutable and hashable. *Why:* the same
instrument is analysed under many conditions, perturbed in tolerancing,
and repeated in multi-configuration, all without aliasing.
*Enforced:* a test that every public description class (in `geometry`,
`materials`, `coatings`, `interfaces`, `elements`, `system`) and every
`Result` refuses attribute assignment and hashes by value; a test that
`OpticalSystem` holds no wavelength, field or object distance.

**Law 3 — Stated, never derived, state.** An object stores what was
stated. Pupils, vertex frames, EFL and bodies are computed. *Why:* a
stored derived value goes stale the moment a stated one changes; today's
`OpticalSystem._rebuild` exists only to fight that. *Enforced:* follows
from law 2 for public state; review for caches, which must be private and
keyed by the value.

**Law 4 — Invalid descriptions cannot be built.** Contradictions are
rejected when a `Configuration` (or any description) is constructed, with
a message naming both sides. Examples are object heights with an object at
infinity, or a non-alternating sequence. *Why:* an error far from its cause
costs an afternoon. *Enforced:* a test per rejection rule.

**Law 5 — Measuring does not change the measured.** No analysis, metric,
sweep or tolerancing run mutates its inputs. *Why:* law 2 makes it
possible, and this law makes it promised. *Enforced:* a parametrised test
that runs every public analysis and checks that its subject is equal, and
equal in hash, before and after.

**Law 6 — A result says what it measured.** Every `Result` carries its
configuration, field, wavelength and focus, and names its units. *Why:* a
number without its conditions cannot be compared with Zemax or with
itself tomorrow. *Enforced:* a test over all `Result` subclasses for the
provenance attributes; a test that numeric public attributes end in a unit
suffix or belong to a dimensionless allow-list.

**Law 7 — Measurement is not presentation.** Nothing below `viz` imports
matplotlib, vispy or Qt. `Result.plot()` is a deferred call into the `viz`
registry, the same pattern as today's `from_prescription`. *Why:* headless
analysis, and results that outlive any plotting library. *Enforced:* the
layer test, extended to third-party imports.

**Law 8 — No hidden solves.** No analysis refocuses, re-aims, re-scales a
field or re-solves an object plane unless the call says so, and the result
then reports what was solved (`defocus_mm`, …). *Why:* a hidden solve
makes two numbers that look comparable incomparable. *Enforced:* a test
that, for every analysis, the default `focus` is the described image
surface and any solved quantity appears in the result.

**Law 9 — No silent choices.** An ambiguous name (one glass in two
catalogues), an unsupported file feature, a Zernike ordering or a sign
convention is never chosen silently. Ambiguity raises; conventions are
named arguments. *Why:* silent choices are where replication fails.
*Enforced:* tests for each ambiguity; the import tests must list every
unsupported feature of a fixture file.

**Law 10 — Same inputs, same outputs.** Randomness (random pupil
sampling, Monte Carlo, non-sequential runs) takes an explicit `seed`.
There is no global RNG. *Why:* a tolerance report that cannot be
reproduced is not a report. *Enforced:* a test that every public function
with a random component requires `seed`, and that two calls with equal
seeds are equal.

---

## 7. How the design is protected

Three checks turn this document into something `pytest` fails on.

1. **Layers.** `tests/test_architecture.py` enforces the chain of §1, plus
   two rules: elements never import the system or the engines, and the
   system never imports an engine or a file format. Still to add: a
   third-party rule (below `viz`, no `matplotlib`, `vispy` or `PySide6`;
   law 7).
2. **Glossary.** `docs/design/glossary.md`, to be written once D1–D7 (§9)
   are settled, defines every public name, once,
   with its nature and package. A test collects `raytracer.__all__`
   recursively and fails on any name missing from the glossary, and on any
   glossary entry that no longer exists. A new concept cannot enter the API
   without a definition. A renamed one cannot leave a stale definition
   behind.
3. **Executable usage.** Each fenced block of `usage.md` becomes a test
   case, run in order like a notebook. Each case is marked
   `xfail(strict=True, reason="<feature>")` until its feature lands. When
   the feature lands, the case passes, strict mode fails the run, and the
   marker must be removed by hand. The usage contract and the
   implementation cannot drift apart without a red test.

Laws 2, 5, 6, 8 and 10 each get the dedicated tests listed in §6.

---

## 8. What changed from the old code

| Old | Now |
|---|---|
| `math`, `surfaces` | `geometry` (vectors, `Rotation`/`Pose`, surfaces, apertures, solids, intersection) |
| `optics.materials` | `materials` |
| `optics.laws`, `optics.radiometry` | `interfaces.reflect/refract`; Fresnel is the `Uncoated` coating |
| `SurfaceRow`, `SurfaceKind` | `TableRow` (a view and a constructor); the model is placed elements |
| `SurfaceKind.STOP` | a `Stop` element (`virtual=True` for a reference plane inside glass) |
| `OpticalSystem(rows, wavelength_um, object_z)` | `OpticalSystem(components, ambient, path, stop)`; wavelength and object plane on the tracer |
| `set_thickness`, `set_asphere` (mutation) | new values: `replace`, `with_component`, `placed`, `rotated` |
| 2-D `Lens`/`Mirror`/`Screen`, `BranchingTracer` | the same 3-D elements and `Sensor`, `NonSequentialTracer`; `fan="y"` sources for meridional pictures |
| `propagation` | `tracing` |
| `design.StigmaticTrain.to_system()` | `families.StigmaticTrain.system()` / `.tracer()` |
| `OpticalSystem.from_prescription` | `io.read_prescription` → `Prescription(system, wavelength_um)` |

---

## 9. Open decisions

**D1 — Name of "a system under conditions".** `Configuration`, `Setup`, or
`Imaging`. *Recommended: `Configuration`.*

**D4 — Default ray aiming** once conditions exist: real or paraxial.
*Recommended: real.*

**D5 — `result.plot()`.** Allow it as a deferred convenience, or allow
only `rt.viz.plot(result)`. *Recommended: allow both.*

**D6 — Language of the design documents.** English, matching the code;
the README is in Spanish.

**D7 — Glass data shipped with the package.** *Recommended:* ship
refractiveindex.info data (CC0); read vendor `.agf` files from a user path.

Settled: the package is `geometry.surfaces`, not `shapes`. Prescription
tables keep the standard signed convention, while placed elements have
absolute poses. Old names were broken, not shimmed. Mutating verbs are
past participles that return new values (`rotated`, `translated`,
`placed`).
