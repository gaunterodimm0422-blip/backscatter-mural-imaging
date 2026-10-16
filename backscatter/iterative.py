from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

from .electronics import energy_sigma_keV
from .kinematics import (
    compton_scattered_energy_keV,
    incident_direction_from_tag,
)


@dataclass(frozen=True)
class ReconstructionGrid2D:
    """Cartesian x-depth grid used after integrating over the cylinder y-axis."""

    x_edges_mm: np.ndarray
    depth_edges_mm: np.ndarray

    def __post_init__(self) -> None:
        x_edges = np.asarray(self.x_edges_mm, dtype=np.float64)
        depth_edges = np.asarray(self.depth_edges_mm, dtype=np.float64)
        if x_edges.ndim != 1 or depth_edges.ndim != 1:
            raise ValueError("grid edges must be one-dimensional")
        if len(x_edges) < 2 or len(depth_edges) < 2:
            raise ValueError("each grid axis must contain at least one bin")
        if np.any(np.diff(x_edges) <= 0.0) or np.any(
            np.diff(depth_edges) <= 0.0
        ):
            raise ValueError("grid edges must be strictly increasing")
        object.__setattr__(self, "x_edges_mm", x_edges)
        object.__setattr__(self, "depth_edges_mm", depth_edges)

    @property
    def x_centers_mm(self) -> np.ndarray:
        return 0.5 * (self.x_edges_mm[:-1] + self.x_edges_mm[1:])

    @property
    def depth_centers_mm(self) -> np.ndarray:
        return 0.5 * (self.depth_edges_mm[:-1] + self.depth_edges_mm[1:])

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.depth_centers_mm), len(self.x_centers_mm))

    @property
    def n_pixels(self) -> int:
        n_depth, n_x = self.shape
        return n_depth * n_x


@dataclass(frozen=True)
class RayLikelihoodSystem:
    """Sparse event-by-pixel likelihood matrix and its retained event indices."""

    matrix: csr_matrix
    event_indices: np.ndarray
    diagnostics: dict[str, Any]


def _positions(table: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack(
        [
            table[f"{prefix}_x_mm"],
            table[f"{prefix}_y_mm"],
            table[f"{prefix}_z_mm"],
        ]
    ).astype(np.float64)


def _klein_nishina_relative(
    theta_rad: np.ndarray,
    scattered_energy_keV: np.ndarray,
    incident_energy_keV: float,
) -> np.ndarray:
    ratio = np.asarray(scattered_energy_keV, dtype=np.float64) / float(
        incident_energy_keV
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = ratio**2 * (
            ratio + 1.0 / ratio - np.sin(theta_rad) ** 2
        )
    return np.clip(weight, 0.0, None)


def build_ray_likelihood_system(
    pairs: dict[str, np.ndarray],
    selected: np.ndarray,
    grid: ReconstructionGrid2D,
    *,
    incident_energy_keV: float = 511.0,
    electron_rest_energy_keV: float = 511.0,
    d1_energy_fwhm_fraction_at_511: float = 0.08,
    d1_position_resolution_mm: tuple[float, float, float] = (1.55, 1.55, 5.0),
    d2_position_resolution_mm: tuple[float, float, float] = (1.0, 1.0, 2.5),
    y_half_width_mm: float = 150.0,
    likelihood_sigma_limit: float = 4.0,
    doppler_sigma_keV: float = 0.0,
    include_geometric_weight: bool = True,
    chunk_size: int = 1024,
) -> RayLikelihoodSystem:
    """Build the list-mode probability model for tagged backscatter events.

    D2 and the source define one incident ray per event.  Candidate points are
    sampled where that ray crosses each depth bin.  The measured D1 energy is
    compared with the Compton energy predicted by the candidate-to-D1 geometry.
    Rows are normalized, so each row is an event-wise location probability over
    the retained x-depth pixels rather than a full absolute detector response.
    """
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != np.asarray(pairs["key"]).shape:
        raise ValueError("selected mask must align with the pairs table")
    if likelihood_sigma_limit <= 0.0:
        raise ValueError("likelihood sigma limit must be positive")
    if y_half_width_mm <= 0.0:
        raise ValueError("y half-width must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    event_indices = np.flatnonzero(selected)
    source_all = _positions(pairs, "source")
    d1_all = _positions(pairs, "d1")
    d2_all = _positions(pairs, "d2")
    energy_all = np.asarray(pairs["d1_energy_keV"], dtype=np.float64)

    depth = grid.depth_centers_mm
    n_depth, n_x = grid.shape
    x_lo = float(grid.x_edges_mm[0])
    x_hi = float(grid.x_edges_mm[-1])
    x_step = float(np.median(np.diff(grid.x_edges_mm)))
    if not np.allclose(np.diff(grid.x_edges_mm), x_step):
        raise ValueError("x bins must have uniform width")

    d1_resolution = np.asarray(d1_position_resolution_mm, dtype=np.float64)
    d2_resolution = np.asarray(d2_position_resolution_mm, dtype=np.float64)
    if np.any(d1_resolution <= 0.0) or np.any(d2_resolution <= 0.0):
        raise ValueError("position resolution values must be positive")
    # The supplied spatial resolutions are treated as voxel widths.  A uniform
    # quantization error therefore has sigma = width/sqrt(12).
    d1_position_sigma = float(np.linalg.norm(d1_resolution / np.sqrt(12.0)))
    d2_position_sigma = float(np.linalg.norm(d2_resolution / np.sqrt(12.0)))

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = [0]
    retained_events: list[int] = []
    candidate_counts: list[int] = []

    for start in range(0, len(event_indices), chunk_size):
        chosen = event_indices[start : start + chunk_size]
        source = source_all[chosen]
        d1 = d1_all[chosen]
        d2 = d2_all[chosen]
        measured_energy = energy_all[chosen]
        direction, direction_ok = incident_direction_from_tag(source, d2)

        with np.errstate(divide="ignore", invalid="ignore"):
            path = (
                -depth[None, :] - source[:, 2, None]
            ) / direction[:, 2, None]
        candidate = source[:, None, :] + path[:, :, None] * direction[:, None, :]
        candidate_x = candidate[:, :, 0]
        candidate_y = candidate[:, :, 1]

        outgoing = d1[:, None, :] - candidate
        outgoing_distance = np.linalg.norm(outgoing, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            outgoing_direction = outgoing / outgoing_distance[:, :, None]
        cosine = np.einsum(
            "ijk,ik->ij",
            outgoing_direction,
            direction,
        )
        theta = np.arccos(np.clip(cosine, -1.0, 1.0))
        predicted_energy = compton_scattered_energy_keV(
            np.degrees(theta),
            incident_energy_keV=incident_energy_keV,
            electron_rest_energy_keV=electron_rest_energy_keV,
        )

        energy_sigma = energy_sigma_keV(
            predicted_energy,
            fwhm_fraction_at_511=d1_energy_fwhm_fraction_at_511,
        )
        tag_distance = np.linalg.norm(d2 - source, axis=1)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            angular_sigma = np.sqrt(
                (d1_position_sigma / outgoing_distance) ** 2
                + (d2_position_sigma / tag_distance) ** 2
            )
        denominator = 1.0 + (
            incident_energy_keV / electron_rest_energy_keV
        ) * (1.0 - np.cos(theta))
        derivative_keV_per_rad = (
            incident_energy_keV
            * (incident_energy_keV / electron_rest_energy_keV)
            * np.sin(theta)
            / denominator**2
        )
        total_sigma = np.sqrt(
            energy_sigma**2
            + (derivative_keV_per_rad * angular_sigma) ** 2
            + float(doppler_sigma_keV) ** 2
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            residual_sigma = (
                measured_energy[:, None] - predicted_energy
            ) / total_sigma
            likelihood = np.exp(-0.5 * residual_sigma**2) / total_sigma

        valid = (
            direction_ok[:, None]
            & np.isfinite(path)
            & (path > 0.0)
            & np.isfinite(candidate_x)
            & np.isfinite(candidate_y)
            & (np.abs(candidate_y) <= float(y_half_width_mm))
            & (candidate_x >= x_lo)
            & (candidate_x < x_hi)
            & np.isfinite(outgoing_distance)
            & (outgoing_distance > 0.0)
            & np.isfinite(total_sigma)
            & (total_sigma > 0.0)
            & np.isfinite(residual_sigma)
            & (np.abs(residual_sigma) <= float(likelihood_sigma_limit))
        )

        if include_geometric_weight:
            likelihood *= _klein_nishina_relative(
                theta,
                predicted_energy,
                incident_energy_keV,
            ) / np.maximum(outgoing_distance**2, 1.0)
        likelihood = np.where(valid, likelihood, 0.0)

        for local_index, original_index in enumerate(chosen):
            keep = np.flatnonzero(likelihood[local_index] > 0.0)
            if len(keep) == 0:
                continue
            row_weights = likelihood[local_index, keep]
            row_sum = float(row_weights.sum())
            if not np.isfinite(row_sum) or row_sum <= 0.0:
                continue
            row_weights = row_weights / row_sum
            x_index = np.floor(
                (candidate_x[local_index, keep] - x_lo) / x_step
            ).astype(np.int64)
            pixel_index = keep.astype(np.int64) * n_x + x_index
            data_parts.append(row_weights.astype(np.float64))
            index_parts.append(pixel_index)
            indptr.append(indptr[-1] + len(pixel_index))
            retained_events.append(int(original_index))
            candidate_counts.append(int(len(pixel_index)))

    if data_parts:
        data = np.concatenate(data_parts)
        indices = np.concatenate(index_parts)
    else:
        data = np.empty(0, dtype=np.float64)
        indices = np.empty(0, dtype=np.int64)
    matrix = csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(retained_events), grid.n_pixels),
    )
    if matrix.shape[0]:
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            raise RuntimeError("event likelihood rows failed normalization")

    counts = np.asarray(candidate_counts, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "n_selected_input": int(selected.sum()),
        "n_rows_retained": int(matrix.shape[0]),
        "n_rows_dropped": int(selected.sum() - matrix.shape[0]),
        "n_nonzero": int(matrix.nnz),
        "candidate_count_median": float(np.median(counts)) if len(counts) else None,
        "candidate_count_p05": float(np.percentile(counts, 5)) if len(counts) else None,
        "candidate_count_p95": float(np.percentile(counts, 95)) if len(counts) else None,
        "row_normalized": True,
        "geometric_weight": bool(include_geometric_weight),
        "likelihood_sigma_limit": float(likelihood_sigma_limit),
        "doppler_sigma_keV": float(doppler_sigma_keV),
        "d1_energy_fwhm_fraction_at_511": float(
            d1_energy_fwhm_fraction_at_511
        ),
        "d1_position_resolution_mm": d1_resolution.tolist(),
        "d2_position_resolution_mm": d2_resolution.tolist(),
    }
    return RayLikelihoodSystem(
        matrix=matrix,
        event_indices=np.asarray(retained_events, dtype=np.int64),
        diagnostics=diagnostics,
    )


def smooth_reference_sensitivity(
    reference_matrix: csr_matrix,
    grid: ReconstructionGrid2D,
    *,
    reference_exposure: float,
    gaussian_sigma_bins: float = 1.0,
    floor_fraction_of_max: float = 0.01,
) -> np.ndarray:
    """Estimate a stable per-primary flat-field from homogeneous reference data."""
    if reference_exposure <= 0.0:
        raise ValueError("reference exposure must be positive")
    if gaussian_sigma_bins < 0.0:
        raise ValueError("smoothing sigma cannot be negative")
    if not 0.0 <= floor_fraction_of_max < 1.0:
        raise ValueError("floor fraction must be in [0, 1)")
    sensitivity = np.asarray(reference_matrix.sum(axis=0)).ravel()
    sensitivity = sensitivity.reshape(grid.shape) / float(reference_exposure)
    if gaussian_sigma_bins > 0.0:
        sensitivity = gaussian_filter(
            sensitivity,
            sigma=float(gaussian_sigma_bins),
            mode="nearest",
        )
    positive = sensitivity[np.isfinite(sensitivity) & (sensitivity > 0.0)]
    if len(positive) == 0:
        raise ValueError("reference matrix has no positive sensitivity")
    floor = float(np.max(positive) * floor_fraction_of_max)
    return np.maximum(sensitivity, floor).ravel()


def listmode_mlem(
    target_matrix: csr_matrix,
    sensitivity_per_exposure: np.ndarray,
    *,
    target_exposure: float,
    iterations: int,
    initial_image: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Reference-calibrated list-mode MLEM on a row-normalized event model."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if target_exposure <= 0.0:
        raise ValueError("target exposure must be positive")
    sensitivity = np.asarray(sensitivity_per_exposure, dtype=np.float64)
    if sensitivity.shape != (target_matrix.shape[1],):
        raise ValueError("sensitivity length must equal the number of pixels")
    if np.any(~np.isfinite(sensitivity)) or np.any(sensitivity <= 0.0):
        raise ValueError("sensitivity must be finite and strictly positive")
    if initial_image is None:
        image = np.ones(target_matrix.shape[1], dtype=np.float64)
    else:
        image = np.asarray(initial_image, dtype=np.float64).copy()
        if image.shape != sensitivity.shape:
            raise ValueError("initial image must match sensitivity")
        image = np.maximum(image, epsilon)

    history: list[dict[str, float]] = []
    for iteration in range(1, iterations + 1):
        prediction = np.asarray(target_matrix @ image).ravel()
        safe_prediction = np.maximum(prediction, epsilon)
        backprojection = np.asarray(
            target_matrix.T @ (1.0 / safe_prediction)
        ).ravel() / float(target_exposure)
        update = backprojection / sensitivity
        image *= update
        image = np.maximum(image, epsilon)
        log_likelihood = float(np.sum(np.log(safe_prediction)))
        history.append(
            {
                "iteration": float(iteration),
                "log_likelihood_without_poisson_rate_term": log_likelihood,
                "image_min": float(np.min(image)),
                "image_median": float(np.median(image)),
                "image_max": float(np.max(image)),
            }
        )
    return image, history


def denoise_tv_chambolle(
    image: np.ndarray,
    *,
    weight: float,
    max_iterations: int = 100,
    tolerance: float = 2e-4,
) -> np.ndarray:
    """Small dependency-free 2-D ROF total-variation proximal operator."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("TV denoising expects a two-dimensional image")
    if weight < 0.0:
        raise ValueError("TV weight cannot be negative")
    if weight == 0.0:
        return image.copy()
    if max_iterations <= 0:
        raise ValueError("max iterations must be positive")

    px = np.zeros_like(image)
    py = np.zeros_like(image)
    output = image.copy()
    tau = 0.125
    previous_energy = None
    for _ in range(max_iterations):
        divergence = np.zeros_like(image)
        divergence[:, 0] = px[:, 0]
        divergence[:, 1:] += px[:, 1:] - px[:, :-1]
        divergence[0, :] += py[0, :]
        divergence[1:, :] += py[1:, :] - py[:-1, :]
        output = image + weight * divergence

        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)
        gradient_x[:, :-1] = np.diff(output, axis=1)
        gradient_y[:-1, :] = np.diff(output, axis=0)
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        scale = 1.0 + (tau / weight) * gradient_norm
        px_new = (px + (tau / weight) * gradient_x) / scale
        py_new = (py + (tau / weight) * gradient_y) / scale

        energy = float(
            0.5 * np.mean((output - image) ** 2)
            + weight * np.mean(gradient_norm)
        )
        px, py = px_new, py_new
        if previous_energy is not None:
            relative = abs(previous_energy - energy) / max(abs(previous_energy), 1e-12)
            if relative < tolerance:
                break
        previous_energy = energy
    return output
