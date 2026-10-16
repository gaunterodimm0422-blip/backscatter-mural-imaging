from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the authoritative project configuration is inconsistent."""


def _interval(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two values")
    lo, hi = float(value[0]), float(value[1])
    if not lo < hi:
        raise ConfigError(f"{name} must satisfy lower < upper")
    return lo, hi


def _positive_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three values")
    result = tuple(float(component) for component in value)
    if any(component <= 0.0 for component in result):
        raise ConfigError(f"{name} values must all be positive")
    return result


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "project",
        "simulation",
        "source",
        "detectors",
        "phantom",
        "selection",
        "reconstruction",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"missing top-level fields: {', '.join(missing)}")

    source = config["source"]
    if len(source["position_mm"]) != 3:
        raise ConfigError("source.position_mm must have three components")

    d1 = config["detectors"]["d1"]
    d2 = config["detectors"]["d2"]
    rmin, rmax = _interval(d1["radius_mm"], "detectors.d1.radius_mm")
    if rmin < 0:
        raise ConfigError("D1 inner radius cannot be negative")
    if float(d1["thickness_mm"]) <= 0:
        raise ConfigError("D1 thickness must be positive")
    if float(d1.get("density_g_cm3", 0.0)) <= 0.0:
        raise ConfigError("D1 material density must be positive")

    array = d2["array"]
    size = d2["crystal_size_mm"]
    pitch = d2["pitch_mm"]
    if len(array) != 3 or len(size) != 3 or len(pitch) != 3:
        raise ConfigError("D2 array, crystal size and pitch must have three components")
    if int(array[0]) <= 0 or int(array[1]) <= 0:
        raise ConfigError("D2 array dimensions must be positive")
    if float(size[0]) > float(pitch[0]) or float(size[1]) > float(pitch[1]):
        raise ConfigError("D2 crystal width cannot exceed its pitch")
    if float(d2.get("density_g_cm3", 0.0)) <= 0.0:
        raise ConfigError("D2 material density must be positive")

    for detector_name, detector in (("d1", d1), ("d2", d2)):
        energy_response = detector.get("energy_response")
        if energy_response is not None:
            fraction = float(energy_response["fwhm_fraction_at_511"])
            if not 0.0 <= fraction < 1.0:
                raise ConfigError(
                    f"detectors.{detector_name}.energy_response."
                    "fwhm_fraction_at_511 must lie in [0, 1)"
                )
            if energy_response.get("model") != "gaussian_sqrt_energy":
                raise ConfigError(
                    f"unsupported {detector_name.upper()} energy-response model"
                )

        position_response = detector.get("position_response")
        if position_response is not None:
            _positive_triplet(
                position_response["resolution_mm"],
                f"detectors.{detector_name}.position_response.resolution_mm",
            )
            if position_response.get("model") not in {
                "voxel_center_quantization_global_xyz",
                "gaussian_fwhm_global_xyz",
            }:
                raise ConfigError(
                    f"unsupported {detector_name.upper()} position-response model"
                )

    phantom = config["phantom"]
    if len(phantom["size_mm"]) != 3 or len(phantom["center_mm"]) != 3:
        raise ConfigError("phantom size and center must have three components")
    z_lo, z_hi = _interval(phantom["z_interval_mm"], "phantom.z_interval_mm")
    center_z = float(phantom["center_mm"][2])
    size_z = float(phantom["size_mm"][2])
    calculated = (center_z - size_z / 2.0, center_z + size_z / 2.0)
    if max(abs(calculated[0] - z_lo), abs(calculated[1] - z_hi)) > 1e-6:
        raise ConfigError(
            "phantom z_interval_mm disagrees with center_mm and size_mm"
        )

    d1_bottom = float(d1["center_z_mm"]) - float(d1["thickness_mm"]) / 2.0
    if d1_bottom < z_hi:
        raise ConfigError("D1 overlaps the phantom in z")

    selection = config["selection"]
    _interval(selection["d1_energy_window_keV"], "selection.d1_energy_window_keV")
    _interval(selection["d2_energy_window_keV"], "selection.d2_energy_window_keV")
    srmin, srmax = _interval(
        selection["d1_radius_mm"], "selection.d1_radius_mm"
    )
    if srmin < rmin or srmax > rmax:
        raise ConfigError("selection D1 radius lies outside the physical detector")
    _interval(selection["incident_angle_deg"], "selection.incident_angle_deg")
    _interval(selection["scatter_angle_deg"], "selection.scatter_angle_deg")
    _interval(
        config["reconstruction"]["path_length_mm"],
        "reconstruction.path_length_mm",
    )

    if float(config["simulation"]["incident_energy_keV"]) <= 0:
        raise ConfigError("incident energy must be positive")
    if int(config["simulation"]["primary_events"]) <= 0:
        raise ConfigError("primary event count must be positive")

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return validate_config(config)
