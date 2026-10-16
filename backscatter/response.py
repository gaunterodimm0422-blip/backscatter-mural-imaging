from __future__ import annotations

from typing import Any

import numpy as np

from .electronics import FWHM_TO_SIGMA, smear_energy_keV


def quantize_positions_to_voxel_centers(
    positions_mm: np.ndarray,
    *,
    resolution_mm: np.ndarray | list[float] | tuple[float, float, float],
    origin_mm: np.ndarray | list[float] | tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> np.ndarray:
    """Map continuous global coordinates to the centres of readout voxels."""
    positions = np.asarray(positions_mm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    resolution = np.asarray(resolution_mm, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64)
    if resolution.shape != (3,) or origin.shape != (3,):
        raise ValueError("resolution and origin must each have three components")
    if np.any(~np.isfinite(resolution)) or np.any(resolution <= 0.0):
        raise ValueError("position resolution values must be finite and positive")
    return (
        np.floor((positions - origin) / resolution) + 0.5
    ) * resolution + origin


def _positions(table: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack(
        [
            table[f"{prefix}_x_mm"],
            table[f"{prefix}_y_mm"],
            table[f"{prefix}_z_mm"],
        ]
    ).astype(np.float64)


def apply_detector_response(
    pairs: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    random_seed: int = 24073101,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the versioned D1/D2 measurement response to event-level pairs.

    Energy broadening is applied after hit energies have been summed per GATE
    event.  Position response is then applied to the energy-weighted event
    position.  The returned table is a copy; the ideal GATE table is unchanged.
    """
    measured = {name: np.asarray(value).copy() for name, value in pairs.items()}
    rng = np.random.default_rng(int(random_seed))
    detector_reports: dict[str, Any] = {}

    for prefix in ("d1", "d2"):
        detector = config["detectors"][prefix]
        energy_response = detector.get("energy_response")
        if energy_response is None:
            raise ValueError(f"{prefix.upper()} energy response is missing")
        energy_model = str(energy_response.get("model"))
        if energy_model != "gaussian_sqrt_energy":
            raise ValueError(f"unsupported {prefix.upper()} energy-response model")
        fraction = float(energy_response["fwhm_fraction_at_511"])
        energy_key = f"{prefix}_energy_keV"
        ideal_energy = np.asarray(pairs[energy_key], dtype=np.float64)
        measured[energy_key] = smear_energy_keV(
            ideal_energy,
            fwhm_fraction_at_511=fraction,
            standard_normal=rng.standard_normal(len(ideal_energy)),
        )

        position_response = detector.get("position_response")
        if position_response is None:
            raise ValueError(f"{prefix.upper()} position response is missing")
        position_model = str(position_response.get("model"))
        resolution = np.asarray(
            position_response["resolution_mm"],
            dtype=np.float64,
        )
        ideal_position = _positions(pairs, prefix)
        if position_model == "voxel_center_quantization_global_xyz":
            measured_position = quantize_positions_to_voxel_centers(
                ideal_position,
                resolution_mm=resolution,
                origin_mm=position_response.get("origin_mm", (0.0, 0.0, 0.0)),
            )
        elif position_model == "gaussian_fwhm_global_xyz":
            sigma = resolution / FWHM_TO_SIGMA
            measured_position = ideal_position + rng.standard_normal(
                ideal_position.shape
            ) * sigma
        else:
            raise ValueError(f"unsupported {prefix.upper()} position-response model")

        measured[f"{prefix}_x_mm"] = measured_position[:, 0]
        measured[f"{prefix}_y_mm"] = measured_position[:, 1]
        measured[f"{prefix}_z_mm"] = measured_position[:, 2]
        detector_reports[prefix] = {
            "material": detector["material"],
            "energy_model": energy_model,
            "energy_fwhm_fraction_at_511": fraction,
            "energy_fwhm_keV_at_511": fraction * 511.0,
            "energy_sigma_keV_at_511": fraction * 511.0 / FWHM_TO_SIGMA,
            "position_model": position_model,
            "position_resolution_mm": resolution.tolist(),
        }

    report = {
        "schema_version": "1.0",
        "random_seed": int(random_seed),
        "application_order": (
            "event-summed ideal GATE pairs -> independent D1/D2 energy "
            "broadening -> D1/D2 position response"
        ),
        "detectors": detector_reports,
        "ideal_input_preserved": True,
    }
    return measured, report
