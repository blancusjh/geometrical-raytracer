# Ontology: what each noun is, and the laws that keep it so

`usage.md` fixed how the library is used. This document reads the nouns
off that usage, gives each one a single nature and a single home, and
states the laws that keep the natures apart. Section numbers are cited
from `usage.md` as **‹law n›**.

---

## 1. The questions

Every piece of the library answers exactly one of these questions. Its
answer decides the package the piece lives in.

| Question | Nature | Package |
|---|---|---|
| What is the shape of a boundary? | geometry | `math`, `shapes` |
| What is light, and what laws does it obey in a medium and at a boundary? | physics | `optics` |
| What is this instrument, and how is it used? | description | `design` |
| How does light get from here to there? | algorithm | `propagation` |
| How is a description stored? | serialisation | `io` |
| What do we measure? | measurement | `analysis` |
| How sensitive is it to error? | variation (statistical) | `tolerance` |
| What is better? | variation (directed) | `optimize` |
| How is it shown? | presentation | `viz` |

The packages form a strictly descending chain. Every import at module level
goes to a package earlier in the chain:

```
math → shapes → optics → design → propagation → io → analysis → tolerance → optimize → viz
```

Compared with today, two things change. `surfaces` becomes `shapes`
(decision D2 in §9), and `tolerance` appears between measuring and
optimising. Tolerancing varies parameters but does not seek an optimum.
Its compensators are one-dimensional solves that need nothing from
`optimize`. A future desensitising optimiser may use `tolerance`, so
`tolerance` sits below `optimize`.

---

## 2. The nouns

**Stated** means the user says it and the object stores it. **Derived**
means it is computed from stated things on demand. A derived quantity is
never stored as authoritative state. **‹law 3›**

### 2.1 Geometry — `math`, `shapes`

| Noun | Is | Stated / derived |
|---|---|---|
| `Frame`, `RigidTransform` | a position and orientation in 3-D | stated |
| polynomial bases (`zernike`, `q_polynomials`, `chebyshev`) | orthogonal function families | — (pure math) |
| `Shape` (`Plane`, `Sphere`, `Conic`, `EvenAsphere`, `OddAsphere`, `QCon`, `QBFS`, `Toroid`, `Cylinder`, `Biconic`, `XYPolynomial`, `ZernikeSag`, `CartesianOval`) | a boundary's geometry in its own local frame: sag, implicit `f_Σ`, gradient, seed | stated |
| `Aperture` (`CircularAperture`, `AnnularAperture`, `RectangularAperture`, `EllipticalAperture`, `Obscuration`) | the region of a shape that exists | stated |

The intersection solvers stay in `math.intersections`. A shape declares
itself. It never solves its own intersection, which is the existing
decision, kept.

Polynomial bases move down to `math`. Zernike polynomials are used by a
shape (`ZernikeSag`) *and* by a measurement (wavefront fitting), so they
belong to neither.

### 2.2 Physics — `optics`

| Noun | Is | Stated / derived |
|---|---|---|
| `Material` (protocol), `ConstantIndex`, `SellmeierMaterial`, `CauchyMaterial`, `AbbeMaterial`, `ModelGlass`, catalogue formulas | a homogeneous medium: `index(λ)`, optional `internal_transmittance(λ, t)` | stated |
| `GradientIndex` | an inhomogeneous medium: `n(x, y, z, λ)` | stated |
| `MaterialLibrary` | ordered catalogues that resolve a name to one material | stated |
| `Interaction` (`Refract`, `Reflect`, `Diffract`) | what light does at a boundary: the direction law | stated |
| `PhaseFunction` (`LinearGrating`, `Binary2`, …) | the phase a diffractive boundary adds | stated |
| `Coating` (`Uncoated`, `ThinFilm`, `IdealCoating`) | how much light a boundary passes, per polarisation: the energy law | stated |
| `Ray`, `RaySet` | geometric light: one ray, or a batch with status | — (state of light) |
| `GaussianBeam` | paraxial beam light: complex `q`, `M²` | stated |
| `ComplexField` | wave light: a sampled complex amplitude | stated |
| `PolarizationState`, `Jones` algebra | the polarisation of light and its transfer | stated |
| `Source` (`PointSource`, `Collimated`, `Lambertian`, `ImageSource`, …) | an emitter: geometry + spectrum + power | stated |
| `Detector` | a surface that records what lands on it | stated |

`laws` (vector reflection, refraction and the grating equation) and
`radiometry` (Fresnel, thin-film transfer matrices) stay here. Physics
needs no system to exist.

### 2.3 Description — `design`

| Noun | Is | Stated / derived |
|---|---|---|
| `Surface` | the five facts of a sequential boundary: shape, aperture, interaction, coating, placement | stated |
| `Gap` | the medium between two surfaces, its signed axial length, and an optional change of axis (`then=`) | stated |
| `Placement` (`Decenter`, `Tilt`, compositions with `@`) | a rigid motion of a surface or element *relative to the axis* | stated |
| `Element` (`Singlet`, `Doublet`, `Mirror`, `Fold`, `Stop`, `Air`, `ImageSurface`, `Group`) | a named group of surfaces and gaps; a *way of writing* a system | stated (the grouping is kept) |
| `OpticalSystem` | the instrument: object-space medium, alternating surfaces and gaps, image surface, which surface is the stop | stated |
| vertices and frames, `bodies()`, `total_track_mm`, `table()` | facts of the hardware | **derived** |
| `Object`, `Aperture` spec, `Fields`, `Spectrum`, `Polarization`, `Aiming` | how the instrument is used | stated |
| `Conditions` | the five use-facts together | stated |
| `Configuration` | a system *under* conditions: the subject of every analysis | stated (a pair) |
| `MultiConfiguration` | an ordered, labelled tuple of configurations | stated |
| `Field`, `Pupil` | coordinates in field space and in the normalised pupil | stated |
| pupil samplings (`hexapolar`, `rings`, `grid`, `fan`, `random`) | point sets in the unit pupil | stated |
| `Parameter` (`Radius`, `Thickness`, `AirSpace`, `Defocus`, `Decenter`, …) | a named degree of freedom: `get(config)` and `set(config, v) → config` | stated |
| families (`StigmaticTrain`, …) | a parametrised generator of systems; `.system()` gives the member | stated |
| `Scene` | solids, sources and detectors placed in space, with no order | stated |
| `Solid` | a closed region of one medium bounded by faces | stated, or derived from a system's `bodies()` |

Two words are used precisely. **Aperture** (geometry) is the region of a
shape that exists. **`Aperture` spec** (description) is how big the
light cone is: EPD, F/#, NA, or float-by-stop. They are different nouns.
The spec lives in `design.conditions` and is written
`rt.Aperture.f_number(…)`. The geometric kinds are always written with
their shape, as in `CircularAperture`, so the two never meet unqualified.

Body deduction (which pair of surfaces encloses which medium, and where
the faces meet) moves from `viz.bodies` to `design`. It is a fact about
the hardware. The non-sequential engine needs it, and drawing is only
one of its readers.

### 2.4 Algorithms — `propagation`

| Kind of light | Engine | Module |
|---|---|---|
| paraxial ray (y, u) | y-nu trace, ABCD | `propagation.paraxial` |
| real ray, fixed order | sequential, vectorised over (field, λ, pupil) | `propagation.sequential` |
| real ray, any order | non-sequential, with ray splitting and genealogy (3-D; 2-D as the meridional case) | `propagation.nonsequential` |
| Gaussian beam | complex-`q` ABCD | `propagation.beams` |
| complex field | angular spectrum / Fresnel between surfaces, ray-traced phase at them | `propagation.physical_optics` |

Ray aiming (from pupil coordinates to real rays) is
`propagation.aiming`. It is the only place that turns a `Pupil` into a
direction. It replaces today's `chief_ray_slope(s)` and the per-call
`na_object_sine`.

The engine for a sequential analysis is chosen by its subject, a
`Configuration`. Users do not construct tracers for ordinary work.

### 2.5 Measurement — `analysis`

| Noun | Is |
|---|---|
| `Result` | base of every analysis output: the measured values, the `configuration` they were measured on, the exact `field`, `wavelength` and `focus`, and units in field names |
| analysis functions | `Configuration → Result`, pure |
| `Metric` (`RmsWavefront`, `RmsSpot`, `MtfAt`, `EncircledEnergyAt`, …) | an analysis reduced to one number by a stated rule: `Configuration → float` |
| `sweep` | an analysis evaluated along an axis (field, wavelength, configuration or `Parameter`) → `Sweep` |
| `focus` | explicit focusing: `focus.paraxial`, `focus.best(metric)` → a new configuration plus the shift |

Sub-packages stay split by *what* is measured. `first_order`. `imaging`
covers spot, PSF, MTF, encircled energy, image simulation and the
lithographic aerial image. `aberrations` covers fans, Seidel, fitted
Seidel, wavefront, distortion, field curvature, chromatic, stigmatism and
aplanatism. `radiometry` covers transmission, relative illumination,
vignetting, footprint and ghosts. `polarization` covers the pupil maps.

### 2.6 Variation — `tolerance`, `optimize`

| Noun | Is |
|---|---|
| `Tolerance` (`Power`, `Irregularity`, `Thickness`, `AirSpace`, `Decenter`, `Tilt`, `Index`, `Abbe`, …) | a *physical* fabrication or assembly error, in shop units, with a distribution; it maps onto one or more `Parameter`s |
| `Compensator` | a `Parameter` re-solved after each perturbation, within bounds |
| `sensitivity`, `inverse`, `monte_carlo` | tolerancing runs → reports whose members are real `Configuration`s |

`optimize` keeps today's content. When the general optimiser comes, its
variables are `Parameter`s and its objective is built from `Metric`s,
so no new nouns will be needed.

### 2.7 Presentation — `viz`

`plot(result)` is a registry dispatched on the result type. `layout`,
`view3d` and `report` are the other entry points. Today's `viz.scene.Scene`
(a list of draw items) is renamed `Drawing`. That frees `Scene` for the
non-sequential description, where it is the standard word.

---

## 3. Anatomy of a boundary

The single most important cut in the design: at a sequential boundary,
five independent facts. Between boundaries, one.

```
            ┌──────────── Surface ────────────┐
   Gap      │ shape        what it looks like │      Gap
 (medium,   │ aperture     where it exists    │   (medium,
  length,   │ interaction  where light goes   │    length,
  then=…)   │ coating      how much goes      │    then=…)
            │ placement    where it sits      │
            └─────────────────────────────────┘
```

Each of these is replaceable without touching the others:

* An aspheric grating is `Surface(EvenAsphere(…), interaction=Diffract(LinearGrating(…)))`.
* A coated mirror is `Surface(Conic(…), interaction=Reflect(), coating=ThinFilm(…))`.
* A GRIN rod is `Gap(10.0, GradientIndex(…))` between two plane surfaces.

Zemax encodes many of these combinations as distinct "surface types".
Here there are no surface types, only the product of five
vocabularies. **‹law 1›**

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

`Shape`, `Aperture`, `Material`, `Surface`, `Gap`, `Element`,
`OpticalSystem`, `Conditions`, `Configuration`, `Parameter`, `Tolerance`,
`Scene` and every `Result` are frozen dataclasses: immutable, comparable,
hashable. `replace(...)` and `Parameter.set(...)` return new values.

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
one place: a grating is an interaction, GRIN is a medium, a stop is a
designation on the system, not a surface kind. *Why:* combinations stay a
product, not a list of special cases. *Enforced:* the glossary test (§7);
review.

**Law 2 — Descriptions are values, and hardware is not use.**
`OpticalSystem` holds no wavelength, field, aperture spec or object
distance. All descriptions are immutable and hashable. *Why:* the same
instrument is analysed under many conditions, perturbed in tolerancing,
and repeated in multi-configuration, all without aliasing.
*Enforced:* a test that every public class in `shapes`, `optics`
(descriptions), `design` and every `Result` is a frozen dataclass; a test
that `OpticalSystem`'s fields contain no condition types.

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

1. **Layers.** `tests/test_architecture.py` already enforces the chain.
   Extend `LAYERS` to the new chain. Add a third-party rule: below `viz`,
   no `matplotlib`, `vispy` or `PySide6` (law 7).
2. **Glossary.** `docs/design/glossary.md`, written once D1–D8 (§9) are
   settled, defines every public name, once,
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

## 8. Migration: from today's names

Nothing here is renamed until the corresponding feature is implemented.
The executable usage specification (§7.3) is the switch.

| Today | Becomes | Why |
|---|---|---|
| package `surfaces` | `shapes` (geometry only) | a shape is not yet a surface; the surface is the five-fact composite (§3) |
| `SurfaceRow` | `Surface` + `Gap`; the row survives as `OpticalSystem.table()` | a row mixes a boundary with the medium after it |
| `SurfaceKind.REFRACT` / `MIRROR` | `Refract()` / `Reflect()` interactions | the direction law, now extensible to `Diffract` |
| `SurfaceKind.STOP` | `OpticalSystem(..., stop="name")` | being the stop is a designation, not a kind of surface |
| `OpticalSystem(rows, wavelength_um, object_space, object_z)` | `OpticalSystem` (hardware) + `Conditions` (`spectrum`, `object`) | law 2 |
| `OpticalSystem.set_asphere`, `set_thickness` | `replace(...)`, `Parameter.set(...)` | law 2; no mutation |
| `OpticalSystem.vertices`, `n_before`, `n_after`, `image_z` (stored) | derived methods | law 3 |
| `AsphereProfile` | `shapes.EvenAsphere` (with `Sphere`, `Conic`, `Plane` as named cases) | one name per shape |
| `CartesianOvalProfile` | `shapes.CartesianOval` | — |
| `ProfileSurface`, `ConicSurface`, `EllipseSurface`, … (2-D, placed, with media) | shapes placed in a `Scene` | they were non-sequential objects all along |
| `FieldPoint(x, y)` | `Field(x, y)` in a declared field type | fields have types (angle, height) |
| `na_object_sine=` on every call | `Aperture.object_na(...)` in `Conditions` | stated once |
| `stop_index=` on every call | `OpticalSystem.stop` | stated once |
| object at `1e6` mm as "infinity" | `Object.at_infinity()` | it is not an approximation any more |
| `PupilSampling(kind=...)` | `rt.pupil.hexapolar(...)`, `rings`, `grid`, `fan`, `random(seed=)` | — |
| `trace_pupil(tracer, field, ...)` | `rt.trace(config, field=, pupil=)` | the subject is the configuration |
| `trace_from_object` | `rt.trace_ray(config, field=, pupil=)` | — |
| `chief_ray_slope(s)` | internal to `propagation.aiming` | aiming is one mechanism |
| `solve_object_plane`, `differential_conjugates` | `rt.focus.paraxial(config)`, `first_order` | explicit solve (law 8) |
| `ParaxialModel` | `rt.paraxial.trace` (y-nu) + `analysis.first_order` | — |
| `TraceResult`, `BatchTraceResult`, `TraceStatus` | `RayPath`, `RaySet`, `RayStatus` | — |
| `BranchingTracer` | `nonsequential.trace` on `scene.meridional()` | one engine, two dimensions |
| `spot_data(pupil)` | `analysis.spot_diagram(config)`; `spot_data_from_points` stays low-level | — |
| `exit_pupil_wavefront(..., reference_radius=500.0)` | `analysis.wavefront(config)`, referenced through the exit pupil | the 500 mm default was a placeholder |
| `pupil_function` + `scalar_psf` (unit-disk pupil) | `analysis.psf(method="fft")` on the traced pupil | vignetting and pupil distortion included |
| `seidel_coefficients` | `analysis.fitted_seidel`; new `analysis.seidel` is per-surface theory | two quantities, two names |
| `zernike`, `zernike_modes`, `fit_opd`, `fit_transverse` | polynomials to `math.polynomials`; fits as `Wavefront.zernike(...)` | shared by a shape and a measurement |
| `distortion_grid(tracer, magnification=, half_field=)` | `analysis.distortion_grid(config, n=)` | magnification and field come from conditions |
| `chief_ray_distortion`, `field_metrics` | `analysis.distortion`, `analysis.field_curvature` | one measurement per name |
| `axial_color`, `lateral_color`, `chromatic_spots` | `focal_shift`, `longitudinal_aberration`, `lateral_color`, `spot_diagram(wavelength="all")` | — |
| `ray_fans` | `ray_fan`, plus new `opd_fan` | — |
| `abbe_image`, `contrast_curve`, `BinaryMask` | `analysis.imaging.lithography` | specialised; kept |
| `StigmaticTrain`, `.to_system()` | `design.families.StigmaticTrain`, `.system()`, `.under(c)` | a family is not a system |
| `optics.elements.Lens`, `Mirror` (2-D) | `Singlet`, `Mirror` elements | one description of a lens, not two |
| `optics.instruments.Screen` | `Detector` | — |
| `viz.bodies` (deduction) | `OpticalSystem.bodies()` in `design` | a fact of the hardware |
| `viz.scene.Scene` | `viz.Drawing` | frees `Scene` for non-sequential |
| `io.read_csv`, `io.read_train` | `io.read_system` (dispatch by format) | named by what it returns |
| `optimize.Constraint`, `EvaluationContext` | unchanged now; later expressed with `Metric` and `Parameter` | out of scope |

---

## 9. Open decisions

Each has a recommendation. Each changes names in `usage.md`, so it should
be settled before any code is written.

**D1 — Name of "a system under conditions".** `Configuration` (the
optical-design word, and multi-configuration falls out of it), `Setup`, or
`Imaging`. *Recommended: `Configuration`.*

**D2 — Rename `surfaces` to `shapes`, and move `Surface` to `design` as
the five-fact composite.** Today's `surfaces.Surface` is a 2-D placed
shape with media, which is really a non-sequential object. *Recommended:
yes.*

**D3 — Sign convention.** Keep the standard signed thicknesses (negative
after an odd number of mirrors) and global radius signs, or make gaps
positive along the light path. *Recommended: keep standard*, so patents
and `.zmx` files compare digit for digit. That is a core value of this
repo.

**D4 — Default ray aiming.** Real (correct off-axis, slower) or paraxial
(Zemax's default). *Recommended: real.* The library already aims the
chief ray in real space.

**D5 — `result.plot()`.** Allow it as a deferred convenience, or allow
only `rt.viz.plot(result)`. *Recommended: allow both*, with the method
being one line that defers to the registry.

**D6 — Language of the design documents.** The README is in Spanish and
the code is in English. These documents are in English to match the code
and the API names. *Your call.*

**D7 — Glass data shipped with the package.** refractiveindex.info data
is CC0 and can ship. Vendor `.agf` catalogues have unclear redistribution
terms. *Recommended:* ship the refractiveindex.info subset; read vendor
`.agf` files from a user path.

**D8 — Compatibility.** Break the old names (version 0.x, one main
author), or keep deprecation shims. *Recommended: break*, migrating the
notebooks and tests in the same commits that rename.
