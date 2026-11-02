# Glossary: every public name, defined once

This is the vocabulary of the library: every name reachable through the
packages' public namespaces — from the top level, and from each layer of
`ontology.md` §1 — with its nature and what it is. A name exported in
several places appears once, under its shortest path.

`tests/test_glossary.py` holds the two in step: a public name missing here
fails the run, and so does an entry here that no longer names anything. A new
concept cannot enter the API without a definition, and a renamed one cannot
leave a stale definition behind. **‹law 1›**

## (top level)

| Name | Nature | What it is |
|---|---|---|
| `Aiming` | class | How a pupil point becomes a ray. |
| `AIR` | constant | Air, taken at index 1 for every wavelength: the default ambient medium. |
| `analysis` | namespace | Optical analysis, split by what is being measured. |
| `Assembly` | class | A named rigid group of components, with a pose of its own. |
| `coatings` | namespace | Coatings: how much light an interface reflects and transmits. |
| `Conditions` | class | How a system is used: where the object is, how much light enters, which fields, in which colours. |
| `conditions` | namespace | Conditions: how an optical system is used. |
| `Configuration` | class | An optical system under conditions of use: the subject of every analysis. |
| `elements` | namespace | Optical elements: the physical things an optical system is made of. |
| `families` | namespace | Families: parametrised generators of optical systems. |
| `Field` | class | A field point: ``(x, y)`` in the unit of the declared fields, or relative to the largest of them. |
| `Fields` | class | The declared field points, all of one kind. |
| `Gap` | class | A stretch of ambient medium in a composition, *distance* mm along the light. |
| `geometry` | namespace | Geometry: where things are and what shape they have — no optics. |
| `Interface` | class | A surface, its coating, and the part of it that is optically specified. |
| `interfaces` | namespace | Interfaces: a surface as an optical boundary, and the laws obeyed there. |
| `io` | namespace | Reading and writing optical-system data. |
| `materials` | namespace | Optical materials: the media that fill space, and their refractive index. |
| `Object` | class | Where the object is. |
| `OpticalSystem` | class | Elements and assemblies placed in space, immersed in an ambient medium. |
| `optimize` | namespace | Design optimization: searching a family of exact systems for a property. |
| `paraxial` | namespace | Paraxial optics of a coaxial path. |
| `Polarization` | class | The polarisation of the light entering: unpolarised, linear at ``angle`` degrees from x, or circular. |
| `Pose` | class | A position and an orientation: the local frame of a placed thing. |
| `Pupil` | class | A point of the pupil in normalized coordinates: ``\|p\| = 1`` is the rim the aperture fills. |
| `pupil` | namespace | Samplings of the pupil: which normalized pupil points to trace, and the share of the pupil each stands for. |
| `Rays` | class | Rays of light, a row each: where each ray is and where it goes, the light it carries, the way it came. |
| `rays` | namespace | Rays: light as geometrical optics knows it, one thing however it is traced. |
| `Rotation` | class | A proper rotation of 3-D space (orthonormal, determinant +1). |
| `sources` | namespace | Sources: emitters that start light. |
| `Spectrum` | class | The wavelengths of use (µm), how much each weighs, and the primary — where first-order optics is read. |
| `system` | namespace | Optical systems: elements and assemblies placed in an ambient medium. |
| `SystemAperture` | class | How much light the system takes in: the paraxial marginal ray it fixes. |
| `TableRow` | class | One row of a prescription table. |
| `trace` | function | Rays of *configuration*: from each field, at each wavelength, through each pupil point — aimed, then traced. |
| `tracing` | namespace | Tracing: the algorithms that carry light through an optical system. |
| `VACUUM` | constant | Vacuum: index 1 at every wavelength. |
| `viz` | namespace | Visualization: the drawing plane, analysis figures, and the matplotlib and OpenGL backends. |

## analysis

| Name | Nature | What it is |
|---|---|---|
| `analysis.abbe_image` | function | Partially coherent Abbe image of *mask* through the pupil. |
| `analysis.airy_radius_mm` | function | The radius of the first dark ring of the Airy pattern, 1.22 λ / (2 NA), in mm. |
| `analysis.angles_of_incidence` | function | ``(N, V)``: the angle (degrees) between each ray and the normal of the boundary at each of its points. |
| `analysis.aplanatic_image_surface` | function | The surface where the images of displaced object points form. |
| `analysis.aplanatism_map` | function | Per-ray ``M`` of the paper's Eq. (25) — 1 for every ray iff aplanatic. |
| `analysis.aplanatism_report` | function | Trace a meridional fan and evaluate the aplanatism condition on it. |
| `analysis.AplanatismReport` | class | Ray-by-ray aplanatism diagnosis of a stigmatic train. |
| `analysis.axial_color` | function | Paraxial focus position vs. wavelength, object plane held fixed. |
| `analysis.AxialColor` | class | Paraxial on-axis focus position vs. wavelength. |
| `analysis.BinaryMask` | class | Binary amplitude mask on a square pixel grid (dimensions in nm). |
| `analysis.chief_ray_distortion` | function | Chief-ray distortion swept over a 1-D field range. |
| `analysis.chromatic_spots` | function | Trace one pupil bundle per wavelength onto a shared image plane. |
| `analysis.ChromaticSpots` | class | Per-wavelength geometric spots sharing one common reference frame. |
| `analysis.coherent_cutoff_half_pitch_nm` | function | Partially-coherent resolution limit lambda / [2 NA (1 + sigma)] in nm. |
| `analysis.contrast_curve` | function | Image contrast of 1-D equal line/space gratings vs half-pitch (nm). |
| `analysis.detected_on` | function | The rays *detector* detected: where each landed, in the detector's own frame ``(K, 3)``, and the power it brought. |
| `analysis.distortion_grid` | function | Trace an ``n x n`` object-space grid spanning ``center +/- half_field``. |
| `analysis.DistortionGrid` | class | Object- and image-space square grids, plus the ideal linear map. |
| `analysis.exit_pupil_wavefront` | function | Compute the exit-pupil OPD of a traced bundle (in waves). |
| `analysis.export_metrics_csv` | function | Write the rows of field metrics to a CSV file. |
| `analysis.FanData` | class | 1-D transverse-aberration fans for one field point. |
| `analysis.field_metrics` | function | Aberration metrics per field point. |
| `analysis.first_order` | function | The first-order (paraxial) data of *configuration*: focal lengths and points, pupils, NAs, magnification. |
| `analysis.FirstOrder` | class | First-order data of a configuration at one wavelength (the primary unless told otherwise). |
| `analysis.fit_opd` | function | Directly fit sampled OPD values (same length unit as *opd*). |
| `analysis.fit_transverse` | function | Fit Zernike gradients to transverse ray aberrations. |
| `analysis.ImageSurface` | class | Best-focus image points for a sweep of object heights. |
| `analysis.lateral_color` | function | Chief-ray image height vs. wavelength, at the fixed nominal image plane. |
| `analysis.LateralColor` | class | Chief-ray image height vs. wavelength, at the reference focal plane. |
| `analysis.Mode` | class | One Zernike mode: its ANSI index, radial order n, azimuthal order m and name. |
| `analysis.normal_axis_crossing` | function | ``V_k C_k``: axis crossing of the surface normal, per the paper's Eq. (24). |
| `analysis.point_line_distances` | function | Perpendicular distance from *target* to each ray's infinite line (2-D or 3-D). |
| `analysis.pupil_function` | function | Build the complex pupil ``P = 1_disk * exp(i 2 pi W / lambda)``. |
| `analysis.PupilGrid` | class | Complex pupil function sampled on an FFT frequency grid. |
| `analysis.ray_fans` | function | Trace meridional and sagittal fans and return transverse aberrations. |
| `analysis.rays_by_generation` | function | Starts and directions of the light after ``generation - 1`` interactions: every ray's piece from that point. |
| `analysis.scalar_psf` | function | Intensity PSF (centred) of the pupil function. |
| `analysis.seidel_coefficients` | function | Fit Seidel-style coefficients from wavefronts traced at several fields. |
| `analysis.SeidelCoefficients` | class | Per-field low-order wavefront coefficients (waves) and their fits. |
| `analysis.spot_data` | function | Compute area-weighted spot statistics from a pupil trace. |
| `analysis.spot_data_from_points` | function | Spot statistics from raw ``(N, 2)`` image-plane points (mm), equal-weighted by default: the same convergence metric (RMS radius from the centroid, in um) for points with no :class:`PupilTrace` behind them, such as where a detector met the rays of a non-sequential trace (:func:`~raytracer.analysis.footprints.detected_on`). |
| `analysis.SpotData` | class | Weighted image-plane spot statistics for one field point. |
| `analysis.stigmatism_report` | function | Compute the spot-size (and, given a wavelength/NA, OPD) convergence metrics. |
| `analysis.StigmatismReport` | class | RMS spot size, and (if a wavelength/NA were given) peak OPD, for one field. |
| `analysis.WavefrontSamples` | class | Scattered exit-pupil wavefront samples in waves. |
| `analysis.weighted_mean` | function | The mean of values under weights. |
| `analysis.weighted_quantile` | function | The value below which a given share of the weight lies. |
| `analysis.zernike` | function | ANSI/OSA RMS-normalized Zernike polynomial on the unit disk. |
| `analysis.zernike_modes` | function | The Zernike modes, as (index, n, m) in the library's ordering. |
| `analysis.ZernikeExpansion` | class | Fitted Zernike coefficients (in the length unit of the fit, mm). |

## coatings

| Name | Nature | What it is |
|---|---|---|
| `coatings.Absorbing` | class | Absorbs everything: a blackened edge, a diaphragm blade, a detector. |
| `coatings.AR` | class | Ideal anti-reflection coating: everything transmitted (except under TIR). |
| `coatings.Coating` | class | What an interface does with the power arriving on it. |
| `coatings.IdealBeamSplitter` | class | Reflects ``R`` and transmits ``T`` of the power, independent of angle and polarisation. |
| `coatings.PerfectMirror` | class | Reflects everything, on either side. |
| `coatings.PowerCoefficients` | class | Power reflectance and transmittance for s and p polarisation, per ray. |
| `coatings.Uncoated` | class | A bare interface: the Fresnel equations, with total internal reflection. |

## elements

| Name | Nature | What it is |
|---|---|---|
| `elements.BeamSplitter` | class | A thin partially reflecting surface (an :class:`~raytracer.coatings.IdealBeamSplitter` by default). |
| `elements.BLADE` | constant | The role of a diaphragm's opaque face: it absorbs, and bounds its opening. |
| `elements.DETECTOR` | constant | The role of a sensor's face: it detects the rays that reach it. |
| `elements.EDGE` | constant | The role of a lens rim: a cylindrical wall that absorbs. |
| `elements.Element` | class | A named, placed physical component. |
| `elements.ElementFace` | class | One boundary of an element, in the element's own frame. |
| `elements.Lens` | class | Glass between optical surfaces: one body, or several cemented into one. |
| `elements.Mirror` | class | A single reflecting surface (coated :class:`~raytracer.coatings.PerfectMirror` unless told otherwise). |
| `elements.OPTICAL` | constant | The role of an optical face: rays reflect or refract there. |
| `elements.Prism` | class | A polyhedral body of one material; every facet is an optical interface. |
| `elements.Sensor` | class | A detector surface: it absorbs and records the light reaching it. |
| `elements.Stop` | class | A diaphragm: an opaque plane with an opening. |

## families

| Name | Nature | What it is |
|---|---|---|
| `families.DESIGN_WAVELENGTH_UM` | constant | The wavelength (µm) a stigmatic train is designed at unless told otherwise. |
| `families.StigmaticTrain` | class | N Cartesian surfaces delimiting a chain of N+1 media. |

## geometry

| Name | Nature | What it is |
|---|---|---|
| `geometry.apertures` | namespace | Apertures: regions of the local (x, y) plane where a surface exists. |
| `geometry.apertures.Annulus` | class | The ring between an inner and an outer radius, about a centre. |
| `geometry.apertures.Aperture` | class | A closed region of the local (x, y) plane. |
| `geometry.apertures.Circle` | class | A disk of a radius, about a centre. |
| `geometry.apertures.Polygon` | class | A simple polygon given by its vertices in order (either orientation). |
| `geometry.apertures.Rectangle` | class | A rectangle of a width and a height, about a centre. |
| `geometry.direction_from_angles` | function | Unit direction tilted from +z by field angles (degrees) in the x-z and y-z planes. |
| `geometry.direction_from_slopes` | function | Unit direction from transverse slopes ``(dx/dz, dy/dz)``. |
| `geometry.normalize` | function | Return the unit direction associated with *vec*. |
| `geometry.solids` | namespace | Solids: closed regions of space, as geometry only. |
| `geometry.solids.Box` | class | A rectangular block centred on the local origin. |
| `geometry.solids.Cylinder` | class | Right circular cylinder of *radius* along local z, from ``z = 0`` to ``z = length``. |
| `geometry.solids.Mesh` | class | Closed polyhedron from vertices and faces (vertex indices, counter-clockwise from outside). |
| `geometry.solids.Polyhedron` | class | A solid bounded by planar polygonal facets. |
| `geometry.solids.Solid` | class | A closed region of space in its local frame. |
| `geometry.solids.Sphere` | class | Ball of *radius* centred on the local origin. |
| `geometry.surfaces` | namespace | Optical surfaces: smooth boundaries, as geometry only. |
| `geometry.surfaces.Asphere` | class | A conic base ``(R, k)`` plus a departure term. |
| `geometry.surfaces.CartesianSurface` | class | The stigmatic surface imaging ``z0`` (index ``n0``) onto ``zi`` (index ``ni``). |
| `geometry.surfaces.Conic` | class | General conic of revolution, vertex radius ``R`` and conic constant ``k``. |
| `geometry.surfaces.CylindricalSurface` | class | Cylinder curved in one direction: the toroid with infinite radius of rotation. |
| `geometry.surfaces.Ellipsoid` | class | Ellipsoid of revolution, ``k > −1`` (prolate for ``k < 0``, oblate for ``k > 0``). |
| `geometry.surfaces.EvenAsphere` | class | ``s(r) = conic(r) + Σ a_p r^p`` over even powers ``p ≥ 2``. |
| `geometry.surfaces.FreeformSurface` | class | XY-polynomial departure ``{(i, j): c_ij}`` on a conic base ``(R, k)``. |
| `geometry.surfaces.Hyperboloid` | class | Hyperboloid of revolution (one sheet of it), ``k < −1``. |
| `geometry.surfaces.Paraboloid` | class | Paraboloid of revolution (``k = −1``), given by ``R`` or by its focal distance ``f = R/2``. |
| `geometry.surfaces.Plane` | class | The flat surface ``z = 0``: the ``c = 0`` member of every conic family. |
| `geometry.surfaces.QTypeAsphere` | class | Forbes Q-type asphere on a conic base (the "Q-con" form). |
| `geometry.surfaces.Sphere` | class | Sphere of radius ``R``: the ellipsoid with equal semi-axes (``k = 0``). |
| `geometry.surfaces.Surface` | class | A smooth surface ``z = s(x, y)`` in its local frame. |
| `geometry.surfaces.SurfaceOfRevolution` | class | A surface symmetric about its local z axis: ``s(x, y) = s(r)``. |
| `geometry.surfaces.ToroidalSurface` | class | Toroid: conic ``(Ry, k)`` in the y-z plane, rotated with radius ``Rx`` about an axis parallel to y. |

## interfaces

| Name | Nature | What it is |
|---|---|---|
| `interfaces.reflect` | function | Law of reflection for ``(N, 3)`` unit directions and normals (either orientation). |
| `interfaces.refract` | function | Snell's law in vector form; *normals* must face the incoming ray. |

## io

| Name | Nature | What it is |
|---|---|---|
| `io.FORMAT_READERS` | constant | The readers of design files, by format name. |
| `io.FORMAT_WRITERS` | constant | The writers of design files, by format name. |
| `io.Prescription` | class | A system read from a lens file, with the wavelength the file is tabulated at. |
| `io.read_csv` | function | Read a CSV prescription into placed elements. |
| `io.read_materials` | function | Read a material-catalog CSV into a :class:`MaterialLibrary`. |
| `io.read_prescription` | function | Read a lens file by format (default: from its extension). |
| `io.read_rows` | function | The table rows of a CSV prescription, and its wavelength (from the index column). |
| `io.read_train` | function | Load a design file back into a :class:`StigmaticTrain`. |
| `io.write_csv` | function | Write the coaxial path of *system* as a CSV prescription. |
| `io.write_train` | function | Save *train* as a JSON design file. |

## materials

| Name | Nature | What it is |
|---|---|---|
| `materials.AbbeMaterial` | class | Refractive index from (n_d, V_d) via a 2-term Cauchy dispersion curve. |
| `materials.CauchyMaterial` | class | Refractive index from an explicit Cauchy series. |
| `materials.ConstantIndex` | class | A medium whose index does not depend on wavelength: enough for single-wavelength work. |
| `materials.default_materials` | function | Vacuum, air and a small common-glass catalog spanning crown to flint. |
| `materials.Material` | class | Anything exposing a name and a wavelength-dependent refractive index. |
| `materials.MaterialLibrary` | class | Name-keyed material registry with tolerant resolution. |
| `materials.sellmeier_glass` | function | A :class:`SellmeierMaterial` for one of the built-in catalog glasses. |
| `materials.SellmeierMaterial` | class | Refractive index from a measured 3-term Sellmeier dispersion curve. |

## optimize

| Name | Nature | What it is |
|---|---|---|
| `optimize.AplanatFit` | class | Outcome of :func:`optimize_aplanat`. |
| `optimize.Aplanatism` | class | The Abbe sine condition: one residual ``M - 1`` per fan ray. |
| `optimize.aplanatism_residuals` | function | Residual vector: per-ray ``M - 1``, plus weighted image-surface sagitta. |
| `optimize.AxialColor` | class | Zero axial color: other wavelengths must focus on the design plane. |
| `optimize.Constraint` | class | One property to drive to zero; subclasses define its residual block. |
| `optimize.constraint_residuals` | function | Concatenated residual blocks of *constraints* on one candidate train. |
| `optimize.Distortion` | class | Zero relative distortion, ``y(h) / (g_t h) - 1``, at the given heights. |
| `optimize.EvaluationContext` | class | One candidate train plus lazily cached traces shared by constraints. |
| `optimize.FlatImageSurface` | class | Zero sagitta of the aplanatic image surface at the given heights. |
| `optimize.optimize_aplanat` | function | The original single-purpose entry point: aplanatism, optionally flat. |
| `optimize.optimize_train` | function | Vary the finite intermediate conjugates to satisfy *constraints*. |
| `optimize.TargetMagnification` | class | Pin the transverse magnification ``g_t`` to a design value. |

## paraxial

| Name | Nature | What it is |
|---|---|---|
| `paraxial.ConjugateSolution` | class | An object plane recovered for a tracer, with the magnification it gives. |
| `paraxial.differential_conjugates` | function | Recover the object plane imposing B = 0 at the detector. |
| `paraxial.Paraxial` | class | The paraxial optics of a configuration at one wavelength (the primary unless told otherwise). |
| `paraxial.ParaxialModel` | class | Analytic ABCD model of a coaxial path, in the reduced state ``(y, n u)``. |
| `paraxial.ParaxialPupil` | class | A paraxial pupil: its axial position ``z_mm`` (world; infinite when telecentric) and diameter. |
| `paraxial.ParaxialRay` | class | A paraxial ray step by step: its height on each vertex plane, and its slope ``u = dy/dz`` leaving it. |
| `paraxial.ParaxialTrace` | class | The paraxial marginal ray, and the chief ray of the largest declared field, of a configuration. |
| `paraxial.solve_object_plane` | function | Find the object plane conjugate to the detector, and (by default) set it on *tracer*. |
| `paraxial.trace` | function | The paraxial marginal and chief rays of *configuration*, step by step (at the primary wavelength by default). |

## pupil

| Name | Nature | What it is |
|---|---|---|
| `pupil.fan` | function | *n* points across the pupil: ``"tangential"`` along y, ``"sagittal"`` along x. |
| `pupil.grid` | function | An *n* × *n* grid across the pupil, clipped to the unit disk. |
| `pupil.hexapolar` | function | The centre and *rings* rings of 6, 12, 18, … points: every point stands for an equal area. |
| `pupil.polar` | function | The centre and *rings* rings of *arms* points each. |
| `pupil.random` | function | *n* points uniform in the disk, drawn from *seed* — the same seed, the same points. |
| `pupil.Sampling` | class | A sampling of the pupil: :meth:`points` ``(N, 2)`` in normalized coordinates, :meth:`weights` summing to 1. |

## rays

| Name | Nature | What it is |
|---|---|---|
| `rays.Status` | class | How a ray ended, or that it has not. |

## sources

| Name | Nature | What it is |
|---|---|---|
| `sources.CollimatedSource` | class | A parallel beam of radius *beam_radius*, centred on *origin*, travelling along *direction*. |
| `sources.ImageSource` | class | An extended object: a 1-D intensity profile along the segment ``p0 → p1``. |
| `sources.PointSource` | class | Rays from one point, filling a cone of half-angle *half_angle_deg* about *direction*. |
| `sources.Source` | class | An emitter of rays. |

## system

| Name | Nature | What it is |
|---|---|---|
| `system.DETECT` | constant | The event of a path step on a detector: the ray is detected there. |
| `system.PhysicalFace` | class | One physical boundary, in world coordinates. |
| `system.REFLECT` | constant | The event of a path step where the ray reflects. |
| `system.Step` | class | One visit of the sequential path: a boundary and what happens there. |
| `system.STOP` | constant | The event of a path step on the aperture stop: the ray passes, or is clipped. |
| `system.TRANSMIT` | constant | The event of a path step where the ray refracts into the medium beyond. |

## tracing

| Name | Nature | What it is |
|---|---|---|
| `tracing.aim` | function | Rays of one *field* at one wavelength, crossing the stop at the normalized *pupil_points* ``(N, 2)``. |
| `tracing.Bundle` | class | Rays traced through a configuration, a row per ray, with what each samples. |
| `tracing.chief_ray_slopes` | function | General ``(s_x, s_y)`` whose ray crosses the axis at the stop surface. |
| `tracing.FieldPoint` | class | Object-plane transverse coordinates (mm). |
| `tracing.NonSequentialTracer` | class | Trace rays through a system with no assumed order. |
| `tracing.object_field` | function | The object-side field — angles (degrees) from infinity, else object heights (mm) — that *field* names. |
| `tracing.PupilSampling` | class | Normalized pupil sample layout. |
| `tracing.PupilTrace` | class | A pupil-filling bundle for one field point: its rays, the pupil point each samples, and the chief ray. |
| `tracing.SequentialTracer` | class | Exact sequential tracer along an :class:`OpticalSystem`'s path. |
| `tracing.trace_from_object` | function | Trace a single ray from the object plane given transverse slopes. |
| `tracing.trace_pupil` | function | Trace a pupil-filling bundle for *field*. |

## viz

| Name | Nature | What it is |
|---|---|---|
| `viz.Drawing` | class | Drawable items in drawing coordinates ``(z, y)``, with optional view limits. |
| `viz.GLBackend` | class | The OpenGL viewer behind the backend interface (``render``, ``update``, ``show``, ``save``). |
| `viz.LensBodyItem` | class | A filled body: its closed outline and, for GL fills, its triangles. |
| `viz.MarkerItem` | class | Points to mark on a drawing, with their colour and size. |
| `viz.plots` | namespace | Matplotlib analysis figures of sequential systems; each returns its :class:`~matplotlib.figure.Figure`. |
| `viz.plots.aberrations_figure` | function | 2×2 field summary: RMS spot, distortion, field curvature, astigmatism. |
| `viz.plots.aerial_figure` | function | Mask, Abbe aerial image, and thresholded resist panels. |
| `viz.plots.chromatic_figure` | function | Axial colour, and optionally lateral colour, against wavelength. |
| `viz.plots.chromatic_spots_figure` | function | Every wavelength's spot in one reference frame, coloured blue to red across the spectrum. |
| `viz.plots.contrast_figure` | function | Image contrast against half-pitch: the resolving-power curve. |
| `viz.plots.dispersion_figure` | function | Refractive index against wavelength; the legend names each material's model (Sellmeier, Abbe, …). |
| `viz.plots.distortion_grid_figure` | function | A traced square grid over the ideal one, its departure from ideal amplified by ``exaggeration``. |
| `viz.plots.distortion_grids_figure` | function | Several distortion grids side by side, each with its own exaggeration (one value, or one per grid). |
| `viz.plots.draw_system` | function | Meridional drawing of a system with traced ray fans; returns the traced paths ``(rays, points, 3)`` or None. |
| `viz.plots.fans_figure` | function | Tangential and sagittal transverse-aberration fans. |
| `viz.plots.layout_figure` | function | Meridional layout of a sequential trace: bodies, mirrors, the object, image and stop planes, ray fans. |
| `viz.plots.psf_figure` | function | Central crop of the scalar PSF. |
| `viz.plots.spots_figure` | function | Spot diagrams about the weighted centroid, one panel per field. |
| `viz.plots.wavefront_figure` | function | Scattered exit-pupil OPD samples in milli-waves. |
| `viz.plots.zernike_figure` | function | Coefficient bars, and the wavefront at the nominal plane and refocused. |
| `viz.RaySegmentsItem` | class | RaySegmentsItem(segments: 'np.ndarray', colors: 'np.ndarray', intensities: 'np.ndarray', escaping: 'np.ndarray', directions: 'np.ndarray') |
| `viz.SensorItem` | class | A detector's section, with the drawing-plane points it recorded. |
| `viz.show` | function | Render *drawing* with the chosen backend and return the backend object. |
| `viz.SurfaceItem` | class | SurfaceItem(polyline: 'np.ndarray', color: 'tuple' = (1.0, 1.0, 1.0, 1.0), width: 'float' = 2.0, name: 'str' = '') |
