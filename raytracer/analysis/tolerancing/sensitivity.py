"""Sensitivity and seeded tolerance sampling with fixed incident illumination."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ...math.transforms import RigidTransform
from ...propagation.fields import FieldPoint, PupilSampling, trace_pupil
from ...propagation.sequential import SequentialTracer, TraceStatus
from ..aberrations.chromatic import _wavelengths, spectral_weights
from .perturbations import Perturbation


@dataclass
class FieldPerformance:
    centroid_xy_mm: np.ndarray
    rms_radius_mm: float
    reference_rms_mm: float
    geometric_throughput: float
    focus_shift_mm: float
    status_counts: dict[str, int]
    valid_masks: tuple[np.ndarray, ...]


@dataclass
class Performance:
    fields: tuple[FieldPerformance, ...]

    def metrics(self):
        values = {}
        for index, field in enumerate(self.fields):
            prefix = f"field_{index}."
            values.update(
                {
                    prefix + "centroid_x_mm": float(field.centroid_xy_mm[0]),
                    prefix + "centroid_y_mm": float(field.centroid_xy_mm[1]),
                    prefix + "rms_radius_mm": field.rms_radius_mm,
                    prefix + "reference_rms_mm": field.reference_rms_mm,
                    prefix + "geometric_throughput": field.geometric_throughput,
                    prefix + "focus_shift_mm": field.focus_shift_mm,
                }
            )
        return values


class GeometricEvaluator:
    """Evaluate identical incident bundles at a fixed nominal detector pose.

    Each wavelength's launch bundle is aimed once in the nominal prescription
    and reused under perturbation. This preserves illumination and clipping.
    The nominal measure is stop area, not a universal radiometric source model.
    ``focus_policy='refocus'`` analytically minimizes the combined spectral RMS
    along the detector normal, independently for each field. It is a diagnostic
    per-field focus, not a single common detector position for all fields.
    Fresnel, absorption, scattering and thermal changes are outside this model.
    """

    def __init__(
        self,
        tracer: SequentialTracer,
        fields,
        wavelengths_um,
        *,
        wavelength_weights=None,
        sampling=None,
        focus_policy="fixed",
    ):
        if focus_policy not in ("fixed", "refocus"):
            raise ValueError("focus_policy must be fixed or refocus")
        self.tracer = tracer.with_system(deepcopy(tracer.system))
        self.fields = tuple(fields)
        if not self.fields or any(not isinstance(f, FieldPoint) for f in self.fields):
            raise ValueError("fields must be nonempty FieldPoint instances")
        self.wavelengths_um = _wavelengths(wavelengths_um).copy()
        self.wavelength_weights = spectral_weights(wavelength_weights, len(self.wavelengths_um))
        self.sampling = sampling or PupilSampling(kind="gauss", radial=8, azimuth=32)
        if self.sampling.kind not in ("gauss", "rings", "grid"):
            raise ValueError("sensitivity requires a two-dimensional pupil sampling")
        self.focus_policy = focus_policy
        self.detector = deepcopy(self.tracer.system.image_frame)
        self.launches = tuple(
            tuple(
                trace_pupil(
                    self.tracer.with_system(self.tracer.system.at_wavelength(wl)),
                    field,
                    sampling=self.sampling,
                )
                for wl in self.wavelengths_um
            )
            for field in self.fields
        )
        for pupils in self.launches:
            for pupil in pupils:
                if pupil.aiming_residual_mm is not None and (
                    not np.all(np.isfinite(pupil.aiming_residual_mm))
                    or np.max(pupil.aiming_residual_mm) > 1e-9
                ):
                    raise ValueError(
                        "nominal pupil aiming failed; cannot define fixed illumination"
                    )
        self.nominal = self.evaluate(self.tracer.system)

    def evaluate(self, system):
        candidate = deepcopy(system)
        candidate._rebuild()
        # Hold the physical detector fixed even if a thickness moves nominal vertices.
        candidate.image_placement = RigidTransform(
            candidate.frame.to_local(self.detector.origin) - [0, 0, candidate.image_z],
            candidate.frame.rotation.T @ self.detector.rotation,
        )
        results = []
        for pupils in self.launches:
            points, slopes, weights, masks = [], [], [], []
            status_counts = {status.name: 0 for status in TraceStatus}
            for wavelength, spectrum, nominal in zip(
                self.wavelengths_um, self.wavelength_weights, pupils
            ):
                batch = self.tracer.with_system(candidate.at_wavelength(wavelength)).trace_batch(
                    nominal.launch_origins,
                    nominal.launch_directions,
                )
                masks.append(batch.valid.copy())
                for status in TraceStatus:
                    status_counts[status.name] += int(np.count_nonzero(batch.status == status))
                unexpected = ~np.isin(
                    batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED, TraceStatus.TIR]
                )
                if unexpected.any():
                    raise RuntimeError(f"perturbed trace failed: {status_counts}")
                local = self.detector.to_local(batch.image_points[batch.valid])
                directions = self.detector.direction_to_local(batch.directions[batch.valid])
                points.append(local[:, :2])
                slopes.append(directions[:, :2] / directions[:, 2:])
                weights.append(spectrum * nominal.sample_weights[batch.valid])
            positions, slopes, weights = (
                np.vstack(points),
                np.vstack(slopes),
                np.concatenate(weights),
            )
            throughput = float(weights.sum())
            if throughput <= 0:
                raise RuntimeError("no weighted ray survives the perturbed system")
            weights /= throughput
            centroid = weights @ positions
            centered = positions - centroid
            spread = slopes - weights @ slopes
            shift = 0.0
            if self.focus_policy == "refocus":
                denominator = np.sum(weights[:, None] * spread**2)
                if denominator <= 1e-28:
                    raise RuntimeError("refocus is undefined for parallel outgoing rays")
                shift = float(-np.sum(weights[:, None] * centered * spread) / denominator)
                positions = positions + shift * slopes
                centroid = weights @ positions
                centered = positions - centroid
            reference = self.detector.to_local(pupils[0].chief.image_point)[:2]
            if not np.all(np.isfinite(reference)):
                raise RuntimeError("nominal chief reference does not reach the detector")
            results.append(
                FieldPerformance(
                    centroid,
                    float(np.sqrt(weights @ np.sum(centered**2, axis=1))),
                    float(np.sqrt(weights @ np.sum((positions - reference) ** 2, axis=1))),
                    throughput,
                    shift,
                    status_counts,
                    tuple(masks),
                )
            )
        return Performance(tuple(results))

    def settings(self):
        return {
            "fields": self.fields,
            "wavelengths_um": self.wavelengths_um,
            "wavelength_weights": self.wavelength_weights,
            "sampling": self.sampling,
            "focus_policy": self.focus_policy,
            "illumination": "fixed nominal launch rays",
            "detector_policy": "fixed nominal pose; optional per-field axial refocus",
            "reference": "nominal chief ray at first listed wavelength",
            "tracer": {key: value for key, value in vars(self.tracer).items() if key != "system"},
        }


@dataclass
class SensitivityResult:
    parameter: Perturbation
    parameter_unit: str
    step: float
    nominal: dict[str, float]
    minus: dict[str, float]
    plus: dict[str, float]
    derivative: dict[str, float]
    derivative_step_difference: dict[str, float]
    one_sided_difference: dict[str, float]
    changed_ray_membership: bool


def sensitivity(evaluator: GeometricEvaluator, parameter: Perturbation, step: float):
    """Central differences at h and h/2, retaining endpoints and nonsmooth clues.

    Derivatives use h/2. The difference between step sizes is a convergence
    diagnostic, not a certified error bound. RMS norms and changing vignetting
    can be nondifferentiable; inspect one-sided differences and ray membership.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("sensitivity step must be finite and positive")
    nominal = evaluator.nominal.metrics()
    trials = [
        evaluator.evaluate(parameter.apply(evaluator.tracer.system, delta))
        for delta in (-step, step, -step / 2, step / 2)
    ]
    minus, plus, half_minus, half_plus = [trial.metrics() for trial in trials]
    derivative = {key: (half_plus[key] - half_minus[key]) / step for key in nominal}
    change = {key: abs(derivative[key] - (plus[key] - minus[key]) / (2 * step)) for key in nominal}
    one_sided = {
        key: abs((half_plus[key] - nominal[key]) - (nominal[key] - half_minus[key])) / (step / 2)
        for key in nominal
    }
    changed = any(
        not np.array_equal(before, after)
        for trial in trials
        for reference, field in zip(evaluator.nominal.fields, trial.fields)
        for before, after in zip(reference.valid_masks, field.valid_masks)
    )
    return SensitivityResult(
        parameter,
        parameter.unit,
        step,
        nominal,
        minus,
        plus,
        derivative,
        change,
        one_sided,
        changed,
    )


@dataclass(frozen=True)
class Tolerance:
    """Uniform half-width or normal standard deviation, in parameter units."""

    parameter: Perturbation
    scale: float
    distribution: str = "uniform"

    def __post_init__(self):
        if (
            self.distribution not in ("uniform", "normal")
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("tolerance requires a positive scale and uniform/normal distribution")


@dataclass
class MonteCarloResult:
    seed: int
    tolerances: tuple[Tolerance, ...]
    draws: np.ndarray
    metrics: dict[str, np.ndarray]
    failures: dict[int, str]

    @property
    def successful_fraction(self):
        return 1 - len(self.failures) / len(self.draws)


def monte_carlo(evaluator: GeometricEvaluator, tolerances, *, samples: int, seed: int):
    """Independent parameter draws with explicit order, seed and retained failures.

    Perturbations are applied in input order. Group rotations generally do not
    commute. Choose explicit pivots when combining translations and rotations.
    This estimates the specified distribution; it does not infer manufacturing
    yield without an acceptance criterion and credible process distributions.
    """
    tolerances = tuple(tolerances)
    if not tolerances or not isinstance(samples, int) or samples < 1:
        raise ValueError("Monte Carlo requires tolerances and a positive integer sample count")
    rng = np.random.default_rng(seed)
    draws = np.column_stack(
        [
            rng.uniform(-tol.scale, tol.scale, samples)
            if tol.distribution == "uniform"
            else rng.normal(0, tol.scale, samples)
            for tol in tolerances
        ]
    )
    metrics = {key: np.full(samples, np.nan) for key in evaluator.nominal.metrics()}
    failures = {}
    for index, draw in enumerate(draws):
        system = evaluator.tracer.system
        try:
            for tolerance, amount in zip(tolerances, draw):
                system = tolerance.parameter.apply(system, amount)
            result = evaluator.evaluate(system).metrics()
        except (ValueError, RuntimeError) as error:
            failures[index] = str(error)
            continue
        for key, value in result.items():
            metrics[key][index] = value
    return MonteCarloResult(seed, tolerances, draws, metrics, failures)
