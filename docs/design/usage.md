# The library as if it existed

**Status.** §2 (describing a system) and the core of §12 (non-sequential
tracing) are implemented; every other section is marked *Planned* and is
the specification of what comes next.

One continuous session. The subject is a Cooke triplet, later a zoom, a
folded telescope and a laser line. Every planned feature appears at least
once. Comments marked **‹law›** point to the invariant in `ontology.md` §6
that the line depends on.

Conventions, fixed once for the whole library: lengths in **mm**,
wavelengths in **µm**, angles in **degrees** at the public surface (radians
only inside). A name whose unit departs from that convention says so
(`rms_radius_um`, `opd_nm`, `frequency_cyc_per_mm`).

```python
import numpy as np
import raytracer as rt
```

---

## 1. Materials come from catalogues

*Planned.*

A material is a *medium*: what fills space between surfaces. Catalogues are
data files. They are read by `io` and collected in a library that resolves
names.

```python
schott = rt.io.read_agf("data/materials/catalogs/SCHOTT.AGF")
ohara  = rt.io.read_agf("data/materials/catalogs/OHARA.AGF")
caf2   = rt.io.read_refractiveindex("data/materials/rii/CaF2_Malitson.yml")

glass = rt.MaterialLibrary(schott, ohara, extra=[caf2])

bk7 = glass["N-BK7"]
bk7.index(0.5876)                                   # 1.5168
bk7.nd, bk7.vd                                      # 1.5168, 64.17
bk7.internal_transmittance(0.40, thickness_mm=10)   # 0.997
glass.find(nd=1.62, vd=36.4, tolerance=0.01)        # nearest catalogue glasses

model = rt.ModelGlass(nd=1.62, vd=36.4)             # a glass that is only (nd, Vd)
```

`ConstantIndex`, `SellmeierMaterial`, `CauchyMaterial` and `AbbeMaterial`
stay. They are the dispersion laws a catalogue entry instantiates. A name
that resolves to two catalogues raises; it never picks silently. **‹law 9›**

---

## 2. Describing what the system *is* — implemented

A system is hardware: **elements placed in an ambient medium**. It knows
nothing about field angles, F-numbers or wavelengths of use. Those are
*conditions*, in §3. **‹law 2›** This section is implemented (see
`raytracer/elements`, `raytracer/system`).

```python
import raytracer as rt
from raytracer import elements as el, coatings
from raytracer.geometry import surfaces as sf, apertures as ap
from raytracer.interfaces import Interface
from raytracer.materials import default_materials

glass = default_materials()
```

### 2.1 Placed components (the model itself)

Every element has a name and a pose. Its local origin is the vertex of its
first face; its local +z runs into the element.

```python
cooke = rt.OpticalSystem(
    name="Cooke triplet",
    ambient=rt.AIR,
    components=[
        el.Lens("L1", front=sf.Sphere(21.25), back=sf.Sphere(-158.65), thickness=2.0,
                material=glass["N-SK16"], semi_diameter=6.5, z=0.0),
        el.Lens("L2", front=sf.Sphere(-20.25), back=sf.Sphere(19.30), thickness=1.0,
                material=glass["F4"], semi_diameter=5.0, z=8.0),
        el.Stop("STO", aperture=4.2, z=10.5),
        el.Lens("L3", front=sf.Sphere(141.25), back=sf.Sphere(-17.285), thickness=2.0,
                material=glass["N-SK16"], semi_diameter=6.5, z=15.0),
        el.Sensor("IMG", z=60.0),
    ],
)
```

Placement keywords, the same on every element:

* `pos=(x, y, z)` or `z=` for the position;
* `rot=` for the orientation: `(x, y, z)` Euler degrees (intrinsic
  x → y′ → z″, so `R = Rx @ Ry @ Rz`), `(angle, (ax, ay, az))` axis-angle,
  a 3×3 matrix, or a `rt.Rotation`;
* `axis=` / `normal=`, which give where the local +z points;
* or a complete `pose=rt.Pose(...)`.

Element vocabulary: `Lens`, `Mirror`, `BeamSplitter`, `Prism` (a
polyhedral solid), `Stop` (an opaque plane with an opening; `virtual=True`
for a patent's reference plane), `Sensor`. `rt.Assembly(name, components,
pose)` moves a group rigidly.

### 2.2 Concise writing: composition and tables

Both are constructors of the same value.

```python
cooke_same = rt.OpticalSystem.compose(
    el.Lens("L1", front=sf.Sphere(21.25), back=sf.Sphere(-158.65), thickness=2.0,
            material=glass["N-SK16"], semi_diameter=6.5),
    rt.Gap(6.0),
    el.Lens("L2", ...), rt.Gap(1.5), el.Stop("STO", aperture=4.2), rt.Gap(4.5),
    el.Lens("L3", ...), rt.Gap(43.0), el.Sensor("IMG"),
)

duv = rt.io.read_prescription("data/optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv")
duv.system.summary()        # double-pass lens, two mirrors, virtual stop, immersion in contact
duv.system.table()          # the rows again: the table is a view of the system
```

A prescription table is read into physical elements:
- every run of glass between two surfaces becomes a lens;
- a surface met twice is one surface;
- bodies sharing a surface are in contact;
- a stop plane inside a lens is a virtual stop;
- the image plane becomes a sensor.

### 2.3 What a boundary is made of

A **surface** is geometry only. An **interface** is a surface plus what
makes it optical *on its own*: a coating and a clear aperture. A bare
surface is accepted wherever an interface is.

```python
el.Lens("L1",
    front=Interface(sf.Sphere(21.25), coating=coatings.AR(), clear_aperture=ap.Circle(6.2)),
    back=sf.Sphere(-158.65),
    thickness=2.0, material=glass["N-SK16"], semi_diameter=6.5, pos=(0, 0, 25), rot=(30, 45, 0))
```

What light *does* at a boundary is not stated anywhere. It follows from
three things:
- the media on the two sides, resolved by the system from the elements;
- the coating: `Uncoated` (Fresnel), `AR`, `PerfectMirror`,
  `IdealBeamSplitter`, `Absorbing`;
- the laws of reflection and refraction.

A mirror is a surface with a reflective coating. **‹law 1›**

Surfaces, with inheritance following mathematical specialization:

```
Surface
├── SurfaceOfRevolution
│   ├── Conic ─┬─ Plane (c = 0)
│   │          ├─ Ellipsoid ── Sphere
│   │          ├─ Paraboloid
│   │          └─ Hyperboloid
│   ├── CartesianSurface      (Descartes' stigmatic surface; conjugates from the vertex)
│   └── Asphere ─┬─ EvenAsphere
│                └─ QTypeAsphere   (Forbes Q-con; Q-bfs not yet)
├── ToroidalSurface ── CylindricalSurface
└── FreeformSurface          (XY polynomial on a conic base)
```

Apertures (`Circle`, `Annulus`, `Rectangle`, `Polygon`) are regions of a
surface's local (x, y) plane. Solids (`Sphere`, `Cylinder`, `Box`, `Mesh`)
are closed regions, used by prisms.

### 2.4 What the system derives

```python
cooke.faces          # every physical boundary, with the medium on each side
cooke.path           # the sequential route: stated (path=[...]) or inferred if coaxial
cooke.stop           # the aperture stop's name
cooke.rotated(45, axis=(1, 1, 0), about=(10, 0, 5))     # a new system; nothing mutates
```

Where two faces coincide (same pose, same surface), they are one boundary
between two materials: cemented lenses, an immersion liquid against a lens.
A folded or split system states its path, with `":reflect"` where a
beam splitter would otherwise be ambiguous:

```python
arm = system.replace(path=["BS:reflect", "M1", "L2.front", "L2.back", "D2"])
```

## 3. Stating how it is *used*: conditions

*Planned.*

```python
day = rt.Conditions(
    object=rt.Object.at_infinity(),
    aperture=rt.Aperture.f_number(4.5),             # or .epd(…), .object_na(…), .image_na(…), .from_stop()
    fields=rt.Fields.angles([0.0, 10.0, 14.0, 20.0], weights=[1, 1, 1, 0.5]),
    spectrum=rt.Spectrum.fdc(),                      # F, d, C; primary = d
    polarization=rt.Polarization.unpolarized(),
)

triplet = cooke.under(day)                           # a Configuration
```

A **Configuration** is a system *under* conditions. It is the subject of
every analysis. Inconsistent combinations are rejected here, at
construction, and not halfway through an analysis. Examples are
`Fields.object_heights` with an object at infinity, or a stop with no
aperture when `Aperture.from_stop()` is asked for. **‹law 4›**

```python
rt.Object.at_infinity()
rt.Object.finite(distance=250.0)          # 250 mm in front of the first vertex
rt.Object.finite(distance=-8.0)           # virtual object, 8 mm behind it

rt.Fields.angles([...])                   # degrees
rt.Fields.object_heights([...])           # mm
rt.Fields.image_heights([...], real=True) # mm; solved by real (or paraxial) chief rays

rt.Spectrum([0.4861, 0.5876, 0.6563], weights=[1, 2, 1], primary=0.5876)
rt.Spectrum.single(0.193)
```

How pupil coordinates turn into real rays is part of the conditions,
because every analysis must agree on it:

```python
day = day.replace(aiming=rt.Aiming.real())       # default; .paraxial() is the faster option
```

Configurations are values too: `triplet.replace(conditions=…)`,
`triplet.replace(system=…)`.

Reading a lens file gives you whatever the file describes, and the function
name says which:

```python
system  = rt.io.read_system("data/optical_systems/photographic/cooke_triplet_prescription.csv")
config  = rt.io.read_configuration("data/optical_systems/photographic/double_gauss.zmx")
configs = rt.io.read_configurations("zoom.zmx")      # a MultiConfiguration

rt.io.write(config, "out/double_gauss.zmx")
rt.io.write(system, "out/cooke.json")
```

A file feature the reader cannot represent raises `UnsupportedFeature`. The
error lists every such feature. Nothing is dropped silently. **‹law 9›**

---

## 4. Where in field, pupil and spectrum: coordinates

*Planned.*

Every sequential quantity is a function of three things: a field point, a
pupil point and a wavelength.

```python
rt.Field(y=14.0)                    # absolute, in the unit of the declared field type (here degrees)
rt.Field.normalized(hy=1.0)         # relative to the largest declared field
2                                   # as an argument: the third *declared* field

rt.Pupil(px=0.0, py=1.0)            # normalized pupil coordinates; |p| ≤ 1 fills the stop image

rt.pupil.hexapolar(rings=8)         # samplings: hexapolar, rings, square grid, fan, random(seed=…)
rt.pupil.fan("tangential", n=41)
```

Analyses take `field=` and `wavelength=`. Each accepts a single value, an
index into the declared list, or `"all"` (the default). `wavelength=
"primary"` is the primary wavelength.

---

## 5. First-order optics

*Planned.*

```python
fo = rt.analysis.first_order(triplet)
fo.efl_mm, fo.bfl_mm, fo.ffl_mm                  # 50.0, 42.95, -37.2
fo.f_number, fo.working_f_number, fo.image_na
fo.entrance_pupil.position_mm, fo.entrance_pupil.diameter_mm
fo.exit_pupil.position_mm,     fo.exit_pupil.diameter_mm
fo.principal_planes, fo.nodal_points
fo.lagrange_invariant
fo.paraxial_image_height_mm, fo.magnification
print(fo)                                        # the "system data" report

ynu = rt.paraxial.trace(triplet)                 # marginal and chief rays, surface by surface
ynu.marginal.y, ynu.marginal.u, ynu.chief.y, ynu.chief.u
```

Each quantity is computed by the paraxial engine (y-nu trace). None of them
is stored on the system. **‹law 3›**

---

## 6. Tracing, when you want rays and not a measurement

*Planned.*

```python
rays = rt.trace(triplet, field="all", pupil=rt.pupil.hexapolar(rings=6), wavelength="all")
rays.shape                       # (fields, wavelengths, pupil samples)
rays.image_points                # (..., 3) mm
rays.directions, rays.opl_mm, rays.status         # status: OK, VIGNETTED, TIR, MISSED, DIVERGED
rays.where(rt.RayStatus.OK)

ray = rt.trace_ray(triplet, field=rt.Field.normalized(hy=1.0), pupil=rt.Pupil(py=1.0))
print(ray.table())               # per-surface: point, normal, angle of incidence, OPL, medium

rays = rt.trace(triplet, field=0, pupil=rt.pupil.hexapolar(8), polarization=True)
rays.jones                       # polarisation ray-tracing matrices, per ray
```

`rt.trace` is the sequential engine. It is chosen by the subject (a
`Configuration`) and never constructed by hand for ordinary work.
`rt.SequentialTracer` stays public for those who want the engine itself.

---

## 7. Measuring image quality

*Planned.*

Every analysis is a function `Configuration → Result`. A result is a value.
It carries what it was computed from (its configuration and the exact
field, wavelength and focus) and its units. It never mutates its subject,
never refocuses silently, and is never drawn by itself. **‹laws 5, 6, 7, 8›**

```python
A = rt.analysis
```

### 7.1 Rays

```python
spots = A.spot_diagram(triplet, pupil=rt.pupil.hexapolar(8))
spots[2].rms_radius_um, spots[2].geo_radius_um, spots[2].centroid_mm, spots.airy_radius_um

A.ray_fan(triplet)                         # transverse ray aberration, tangential and sagittal
A.opd_fan(triplet)                         # the same, in waves of OPD
A.longitudinal_aberration(triplet)         # focus shift vs pupil zone, per wavelength
A.field_curvature(triplet)                 # tangential and sagittal focus vs field
A.distortion(triplet, kind="f_tan")        # or "f_theta", or a calibrated mapping
A.distortion_grid(triplet, n=11)
A.lateral_color(triplet)
A.focal_shift(triplet, wavelengths=np.linspace(0.40, 0.70, 31))
A.vignetting(triplet)                      # unvignetted fraction vs field
A.footprint(triplet, surface="L2.1")       # where each field's beam lands on a surface
A.relative_illumination(triplet)
```

### 7.2 Third order: theory and measurement are named apart

```python
s = A.seidel(triplet)                      # third-order theory, from the paraxial rays
s.by_surface                               # table: SI SII SIII SIV SV CL CT per surface
s.total.SI

A.fitted_seidel(triplet, field=3)          # the same five terms read off the *real* wavefront
```

Both measure "Seidel coefficients", but they are different quantities that
disagree exactly where third-order theory stops being valid. That is why
they have two names.

### 7.3 Wavefront and diffraction

```python
wf = A.wavefront(triplet, field=3, wavelength="primary")
wf.rms_waves, wf.pv_waves
wf.map                                     # on the exit pupil, referenced to the sphere centred
                                           # on the chief-ray image point through the exit pupil
z = wf.zernike(terms=37, ordering="fringe")   # or "ansi"; the ordering is never implicit
z["defocus"], z.rms_waves(exclude=("piston", "tilt"))

psf = A.psf(triplet, field=3, method="huygens")   # or "fft"; both use the *traced* pupil,
psf.strehl, psf.pixel_um                          # vignetting and pupil distortion included
psf.encircled_energy(radius_um=5.0)
psf.ensquared_energy(half_width_um=5.0)

mtf = A.mtf(triplet, method="fft")         # or "geometric", "huygens"
mtf[2].tangential, mtf[2].sagittal, mtf.diffraction_limit
mtf.at(frequency_cyc_per_mm=50)
```

### 7.4 Sweeps: one law instead of N variants

"MTF vs field", "MTF through focus" and "RMS vs wavelength" are not new
analyses. They are one analysis evaluated along an axis. The axis is a
field, a wavelength, a configuration, or a `Parameter` of the system (§9).

```python
through_focus = rt.sweep(A.mtf, triplet,
                         over=rt.param.Defocus(), values=np.linspace(-0.1, 0.1, 21),
                         field=0)
through_focus.map(lambda m: m.at(50).tangential)   # → curve vs defocus

vs_field = rt.sweep(A.mtf, triplet, over="field",
                    values=rt.Fields.angles(np.linspace(0, 20, 21)))
```

The familiar named views are one-line definitions over `sweep`:
`A.mtf_through_focus`, `A.mtf_vs_field`, `A.rms_vs_field`.

### 7.5 Focus is explicit

```python
best = rt.focus.best(triplet, metric=rt.metrics.RmsWavefront(field="all"))
best.defocus_mm                            # how far the image surface moved
triplet_focused = best.configuration       # a new configuration; triplet is unchanged

A.mtf(triplet, focus="best")               # allowed, and the result reports .defocus_mm
```

### 7.6 Polarisation, transmission, ghosts, images

```python
A.transmission(triplet)                                   # Fresnel + coatings + bulk absorption
A.polarization_pupil(triplet, field=0)                    # diattenuation and retardance maps
A.jones_pupil(triplet, field=0)

A.ghosts(triplet, bounces=2)              # sequential ghost enumeration: every surface pair,
                                          # ghost focus position, ghost irradiance at the image

scene_image = rt.io.read_image("data/images/resolution_chart.png")
A.simulate_image(triplet, scene_image, field_of_view_deg=40.0, psf_grid=(5, 5))
```

### 7.7 Exact optics (already the library's speciality)

```python
train = rt.families.StigmaticTrain(media=(rt.AIR, glass["N-BK7"], rt.AIR), ...)
A.stigmatism_report(train.under(conditions))
A.aplanatism_report(train.under(conditions))
```

A `StigmaticTrain` is a *family*: a parametrised generator of systems. It
is not a system. `train.system()` gives the member, and `train.under(c)` is
`train.system().under(c)`.

---

## 8. Showing results

*Planned.*

```python
rt.viz.plot(mtf)                     # one entry point, dispatched on the result type
mtf.plot()                           # the same; a deferred convenience, not a dependency
rt.viz.layout(triplet, fields="all") # 2-D section with bodies and rays
rt.viz.view3d(triplet)               # the HDR OpenGL viewer
rt.viz.report(triplet, [A.spot_diagram, A.mtf, A.field_curvature, A.distortion])
```

No analysis imports matplotlib. **‹law 7›**

---

## 9. Parameters: how the library refers to a degree of freedom

*Planned.*

One noun serves sweeps, multi-configuration, tolerancing and (later)
optimisation. A parameter reads a value from a configuration and returns a
*new* configuration with the value changed.

```python
p = rt.param.Radius("L3.2")
p.get(triplet)                       # -17.285
p.set(triplet, -17.0)                # a new Configuration

rt.param.Curvature("L1.1"); rt.param.Conic("L3.2"); rt.param.AsphereCoefficient("L3.2", order=4)
rt.param.Thickness("L2")             # centre thickness of element L2
rt.param.AirSpace("L1", "L2")        # the gap between two elements
rt.param.Defocus()                   # the gap before the image surface
rt.param.Decenter("L2", axis="x"); rt.param.Tilt("L2", axis="y")
rt.param.Index("L1"); rt.param.Abbe("L1")
rt.param.ObjectDistance(); rt.param.FieldScale()      # parameters of the conditions
```

---

## 10. Multi-configuration

*Planned.*

```python
zoom = rt.MultiConfiguration.vary(
    base=zoom_lens.under(day),
    labels=("wide", "mid", "tele"),
    values={rt.param.AirSpace("G1", "G2"): [30.0, 15.0, 2.0],
            rt.param.AirSpace("G2", "G3"): [5.0, 18.0, 33.0],
            rt.param.FieldScale():         [1.0, 0.6, 0.35]},
)
zoom["tele"]                                 # an ordinary Configuration
rt.sweep(A.first_order, zoom, over="configuration").map(lambda f: f.efl_mm)
```

A `MultiConfiguration` is an ordered, labelled tuple of `Configuration`s.
Every analysis takes one of them, and a sweep takes all of them.

---

## 11. Tolerancing

*Planned.*

A tolerance is a *physical* error (fabrication or assembly), stated in the
units an optical shop uses. It maps onto parameters. It is not a raw field
of the data model.

```python
tolerances = [
    rt.tol.Power("L1.1", fringes=3, test_wavelength_um=0.6328),
    rt.tol.Irregularity("L1.1", fringes=0.5),
    rt.tol.Thickness("L2", plus_minus_mm=0.05),
    rt.tol.AirSpace("L1", "L2", plus_minus_mm=0.05),
    rt.tol.Decenter("L2", plus_minus_mm=0.02),
    rt.tol.Tilt("L2", plus_minus_arcmin=1.0),
    rt.tol.Index("L1", plus_minus=0.0005),
    rt.tol.Abbe("L1", plus_minus_percent=0.8),
]
tolerances = rt.tol.standard(triplet, grade="precision")      # or a default set, per surface and element

criterion    = rt.metrics.RmsWavefront(field="all")           # a Metric: Configuration → float
compensators = [rt.comp.Free(rt.param.Defocus(), within_mm=0.2)]

sens = rt.tolerance.sensitivity(triplet, tolerances, criterion, compensators=compensators)
sens.ranked()[:5]                                             # worst offenders
inv = rt.tolerance.inverse(triplet, tolerances, criterion, budget=0.02, ...)

mc = rt.tolerance.monte_carlo(triplet, tolerances, criterion,
                              compensators=compensators, trials=500, seed=7)
mc.percentile(90)                                             # criterion at 90 % yield
mc.yield_below(0.07)
lot = mc.worst(3)                                             # three *real* configurations
A.mtf(lot[0])                                                 # every analysis works on them
```

Because perturbed systems are ordinary values, nothing about tolerancing is
special to the analyses. Randomness always takes an explicit `seed`. **‹law 10›**

---

## 12. Non-sequential: the same system, no order — core implemented

There is no separate "scene". The non-sequential engine reads the very
same `OpticalSystem`: the same faces, the same media on each side, the
same coatings. Only the order is dropped.

```python
from raytracer.sources import CollimatedSource
from raytracer.tracing import NonSequentialTracer

source = CollimatedSource((0, 0, -40), direction=(0, 0, 1), beam_radius=2.0,
                          rays=2000, wavelength_um=0.55)
tree = NonSequentialTracer(system, max_interactions=12, split=True).trace(source)
tree.hits("D1").power.sum()                      # power reaching a sensor
tree.hits("D1").histogram(axis=1, bins=200)      # irradiance profile
tree.segments()                                  # every segment, with generation/parent/kind
```

* `split=True` divides power between reflected and transmitted children,
  per coating (Fresnel for bare surfaces).
* `split=False` follows the branch a sequential trace would take, which
  gives the clean picture for drawing.
* `fan="y"` on a source gives the meridional fan of a 2-D figure.
* Planned: irradiance maps in W/mm², ghost-path queries, Monte Carlo
  splitting with `seed=`.

---

## 13. Beams and waves

*Planned.*

A Gaussian beam is a different *kind of light*, like a ray, and it has its
own engine.

```python
beam = rt.GaussianBeam(waist_um=500.0, wavelength_um=1.064, m2=1.1, waist_at_mm=-100.0)
bt = rt.beams.trace(laser_line.system, beam)       # complex q through every surface
bt.at("image").waist_um, bt.at("image").rayleigh_range_mm

field = rt.ComplexField.gaussian(beam, grid=1024, extent_mm=20.0)
pop = rt.physical_optics.propagate(laser_line, field)    # angular spectrum between surfaces,
pop.irradiance("image")                                  # ray-traced phase at each surface
pop.coupling_efficiency(rt.ComplexField.fiber_mode(mfd_um=6.2, wavelength_um=1.064))
```

---

## What the session needed, and nothing more

Nouns: `Material`, `MaterialLibrary`, `Surface` (and its family),
`Aperture`, `Solid`, `Coating`, `Interface`, `Element` (`Lens`, `Mirror`,
`BeamSplitter`, `Prism`, `Stop`, `Sensor`), `Pose`, `Rotation`,
`Assembly`, `OpticalSystem`, `Conditions`, `Configuration`,
`MultiConfiguration`, `Field`, `Pupil`, `RaySet`, `Result`, `Parameter`,
`Metric`, `Tolerance`, `Compensator`, `Source`, `GaussianBeam`,
`ComplexField`.

Verbs: `compose`, `under`, `replace`, `trace`, `sweep`, the analyses, the
tolerancing runs, `read` and `write`, `plot`.

`ontology.md` defines each of these once.
