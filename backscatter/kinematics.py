from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScatterSolution:
    points_mm: np.ndarray
    path_length_mm: np.ndarray
    incident_direction: np.ndarray
    theta_deg: np.ndarray
    alpha_deg: np.ndarray
    valid: np.ndarray
    failure_code: np.ndarray


def compton_scattered_energy_keV(
    theta_deg: np.ndarray | float,
    incident_energy_keV: float = 511.0,
    electron_rest_energy_keV: float = 511.0,
) -> np.ndarray:
    """Photon energy after a single Compton scatter."""
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    denominator = 1.0 + (incident_energy_keV / electron_rest_energy_keV) * (
        1.0 - np.cos(theta)
    )
    return incident_energy_keV / denominator


def compton_angle_from_energy_keV(
    scattered_energy_keV: np.ndarray | float,
    incident_energy_keV: float = 511.0,
    electron_rest_energy_keV: float = 511.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (theta_deg, valid_kinematics) without clipping invalid events.

    Invalid deposited energies must not be forced to 0 or 180 degrees.
    """
    energy = np.asarray(scattered_energy_keV, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = 1.0 - electron_rest_energy_keV * (
            1.0 / energy - 1.0 / incident_energy_keV
        )
    valid = (
        np.isfinite(cos_theta)
        & (energy > 0.0)
        & (energy <= incident_energy_keV)
        & (cos_theta >= -1.0)
        & (cos_theta <= 1.0)
    )
    theta = np.full(energy.shape, np.nan, dtype=np.float64)
    theta[valid] = np.degrees(np.arccos(cos_theta[valid]))
    return theta, valid


def incident_direction_from_tag(
    source_mm: np.ndarray, d2_hit_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Infer the object-going photon direction from the opposite D2 photon."""
    source = np.asarray(source_mm, dtype=np.float64)
    d2 = np.asarray(d2_hit_mm, dtype=np.float64)
    if source.ndim == 1:
        source = np.broadcast_to(source, d2.shape)
    tag_vector = d2 - source
    norm = np.linalg.norm(tag_vector, axis=1)
    valid = np.isfinite(norm) & (norm > 0.0)
    direction = np.full_like(tag_vector, np.nan, dtype=np.float64)
    direction[valid] = -tag_vector[valid] / norm[valid, None]
    return direction, valid


def incident_polar_angle_deg(source_mm: np.ndarray, d2_hit_mm: np.ndarray) -> np.ndarray:
    """D2 tag angle relative to +z, equal to object-going angle relative to -z."""
    source = np.asarray(source_mm, dtype=np.float64)
    d2 = np.asarray(d2_hit_mm, dtype=np.float64)
    if source.ndim == 1:
        source = np.broadcast_to(source, d2.shape)
    vector = d2 - source
    norm = np.linalg.norm(vector, axis=1)
    angle = np.full(norm.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norm) & (norm > 0.0)
    cosine = np.clip(vector[valid, 2] / norm[valid], -1.0, 1.0)
    angle[valid] = np.degrees(np.arccos(cosine))
    return angle


def solve_scatter_points(
    source_mm: np.ndarray,
    d1_hit_mm: np.ndarray,
    d2_hit_mm: np.ndarray,
    scattered_energy_keV: np.ndarray,
    *,
    incident_energy_keV: float = 511.0,
    electron_rest_energy_keV: float = 511.0,
    path_min_mm: float = 1.0,
    path_max_mm: float = 280.0,
) -> ScatterSolution:
    """
    Closed-form tagged-backscatter reconstruction.

    Measurement notation used in the 2026-07-31 system schematic:

      * D2 tag detector records (E1, X1, t1).  X1 and the known source
        position S define the incident ray l1 in the opposite direction.
      * D1 backscatter detector records (E2, X2, t2).  E2 is the scattered
        photon energy used in the Compton equation, and X2 is the outgoing
        ray endpoint.

    In this function ``d2_hit_mm`` is X1, ``d1_hit_mm`` is X2, and
    ``scattered_energy_keV`` is E2.  The event pairing/time requirement is
    applied before this solver by ``pipeline.build_selection``.

    With v=-(X1-S)/|X1-S| and alpha=angle(v,X2-S), the unique solution is

        t = |X2-S| sin(theta-alpha) / sin(theta)
        P = S + t v.

    Failure codes:
      0 valid
      1 invalid D2/source direction
      2 invalid single-Compton energy
      3 theta <= alpha or singular geometry
      4 solution outside the allowed path interval
    """
    d1 = np.asarray(d1_hit_mm, dtype=np.float64)
    d2 = np.asarray(d2_hit_mm, dtype=np.float64)
    energy = np.asarray(scattered_energy_keV, dtype=np.float64)
    source = np.asarray(source_mm, dtype=np.float64)
    if source.ndim == 1:
        source = np.broadcast_to(source, d1.shape)
    if not (source.shape == d1.shape == d2.shape):
        raise ValueError("source, D1 and D2 positions must share shape (N,3)")
    if len(energy) != len(d1):
        raise ValueError("energy length must match position arrays")

    direction, direction_ok = incident_direction_from_tag(source, d2)
    theta_deg, energy_ok = compton_angle_from_energy_keV(
        energy,
        incident_energy_keV=incident_energy_keV,
        electron_rest_energy_keV=electron_rest_energy_keV,
    )
    theta = np.radians(theta_deg)

    source_to_d1 = d1 - source
    distance = np.linalg.norm(source_to_d1, axis=1)
    alpha = np.full(distance.shape, np.nan, dtype=np.float64)
    alpha_ok = direction_ok & np.isfinite(distance) & (distance > 0.0)
    cosine_alpha = np.full(distance.shape, np.nan, dtype=np.float64)
    cosine_alpha[alpha_ok] = (
        np.einsum(
            "ij,ij->i", direction[alpha_ok], source_to_d1[alpha_ok]
        )
        / distance[alpha_ok]
    )
    alpha[alpha_ok] = np.arccos(np.clip(cosine_alpha[alpha_ok], -1.0, 1.0))

    sin_theta = np.sin(theta)
    geometry_ok = (
        direction_ok
        & energy_ok
        & alpha_ok
        & np.isfinite(theta)
        & (theta > alpha)
        & (np.abs(sin_theta) > 1e-12)
    )
    path = np.full(distance.shape, np.nan, dtype=np.float64)
    path[geometry_ok] = (
        distance[geometry_ok]
        * np.sin(theta[geometry_ok] - alpha[geometry_ok])
        / sin_theta[geometry_ok]
    )
    range_ok = (
        geometry_ok
        & np.isfinite(path)
        & (path > float(path_min_mm))
        & (path < float(path_max_mm))
    )

    points = np.full_like(d1, np.nan, dtype=np.float64)
    points[range_ok] = source[range_ok] + path[range_ok, None] * direction[range_ok]

    failure = np.zeros(len(d1), dtype=np.int8)
    failure[~direction_ok] = 1
    failure[direction_ok & ~energy_ok] = 2
    failure[direction_ok & energy_ok & ~geometry_ok] = 3
    failure[geometry_ok & ~range_ok] = 4

    return ScatterSolution(
        points_mm=points,
        path_length_mm=np.where(range_ok, path, np.nan),
        incident_direction=direction,
        theta_deg=theta_deg,
        alpha_deg=np.degrees(alpha),
        valid=range_ok,
        failure_code=failure,
    )


def synthetic_selftest(n: int = 20_000, seed: int = 7) -> dict[str, float | int | bool]:
    """Generate known vertices and verify exact reconstruction closure."""
    rng = np.random.default_rng(seed)
    source = np.broadcast_to(np.array([0.0, 0.0, 17.5]), (n, 3)).copy()

    cos_beta = rng.uniform(np.cos(np.radians(45.0)), 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_beta = np.sqrt(1.0 - cos_beta**2)
    incident = np.column_stack(
        [sin_beta * np.cos(phi), sin_beta * np.sin(phi), -cos_beta]
    )

    depth = rng.uniform(5.0, 95.0, n)
    path_true = (17.5 + depth) / cos_beta
    point_true = source + path_true[:, None] * incident

    tag_path = (60.0 - 17.5) / cos_beta
    d2 = source - tag_path[:, None] * incident

    radius = rng.uniform(20.0, 200.0, n)
    dphi = rng.uniform(0.0, 2.0 * np.pi, n)
    d1 = np.column_stack(
        [
            radius * np.cos(dphi),
            radius * np.sin(dphi),
            rng.uniform(10.0, 25.0, n),
        ]
    )

    outgoing = d1 - point_true
    outgoing /= np.linalg.norm(outgoing, axis=1, keepdims=True)
    cosine_theta = np.einsum("ij,ij->i", incident, outgoing)
    theta = np.degrees(np.arccos(np.clip(cosine_theta, -1.0, 1.0)))
    energy = compton_scattered_energy_keV(theta)

    solution = solve_scatter_points(source, d1, d2, energy)
    error = np.linalg.norm(solution.points_mm - point_true, axis=1)
    finite_error = error[solution.valid]
    maximum = float(np.max(finite_error)) if len(finite_error) else float("inf")
    median = float(np.median(finite_error)) if len(finite_error) else float("inf")

    invalid_energy = np.full(n, 160.0)
    invalid_solution = solve_scatter_points(source, d1, d2, invalid_energy)
    invalid_rejected = int((~invalid_solution.valid).sum())

    passed = (
        int(solution.valid.sum()) == n
        and maximum < 1e-8
        and invalid_rejected == n
    )
    return {
        "passed": passed,
        "n_generated": n,
        "n_reconstructed": int(solution.valid.sum()),
        "max_vertex_error_mm": maximum,
        "median_vertex_error_mm": median,
        "invalid_160keV_rejected": invalid_rejected,
    }
