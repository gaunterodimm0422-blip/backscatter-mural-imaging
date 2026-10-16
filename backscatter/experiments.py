from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import validate_config


STANDARD_TARGET_KINDS = {
    "thin_layer",
    "rod",
    "double_rod",
    "point",
    "double_point",
    "edge",
}


def build_blind_analysis_config(
    base_config: dict[str, Any],
    *,
    primary_events: int = 1_000_000,
) -> dict[str, Any]:
    """Create one neutral observable-only config for every blinded case.

    The same config is used whether a private GATE case contains a void or a
    homogeneous slab.  It deliberately contains neither a class label nor a
    target position and prevents phantom truth from entering event selection.
    """
    if primary_events <= 0:
        raise ValueError("primary event count must be positive")
    config = copy.deepcopy(base_config)
    config["project"]["name"] = (
        "tagged_511keV_backscatter_blind_unknown_concrete_case"
    )
    config["project"]["baseline_id"] = (
        f"blind_unknown_concrete_{int(primary_events)}_primaries"
    )
    config["simulation"]["primary_events"] = int(primary_events)
    config["phantom"]["name"] = "unknown_concrete_case"
    config["phantom"]["measurement_role"] = "blind_void_detection"
    config["selection"]["phantom_truth_policy"] = "ignore"
    return validate_config(config)


def build_concrete_void_config(
    base_config: dict[str, Any],
    *,
    depth_mm: float,
    radius_mm: float,
    length_mm: float = 300.0,
    x_mm: float = 0.0,
    primary_events: int = 2_000_000,
    random_seed: int | None = None,
    d1_rmax_mm: float | None = None,
) -> dict[str, Any]:
    """Create a full concrete-slab configuration with a cylindrical void.

    The cylinder is parallel to global y.  Its center depth is measured from
    the concrete top surface, matching the project's positive-depth reporting
    convention.
    """
    if depth_mm <= 0.0 or radius_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("void depth, radius and length must be positive")
    if primary_events <= 0:
        raise ValueError("primary event count must be positive")
    if random_seed is not None and random_seed <= 0:
        raise ValueError("random seed must be positive")

    config = copy.deepcopy(base_config)
    phantom = config["phantom"]
    size_x, size_y, size_z = map(float, phantom["size_mm"])
    center_x, center_y, center_z = map(float, phantom["center_mm"])
    z_top = center_z + size_z / 2.0
    z_bottom = center_z - size_z / 2.0
    void_z = z_top - float(depth_mm)
    if void_z + radius_mm > z_top or void_z - radius_mm < z_bottom:
        raise ValueError("void cross-section must remain inside the concrete slab")
    if abs(float(x_mm) - center_x) + radius_mm > size_x / 2.0:
        raise ValueError("void cross-section must remain inside the slab in x")
    if length_mm > size_y:
        raise ValueError("void length cannot exceed the concrete slab y size")

    config["project"]["name"] = "tagged_511keV_backscatter_concrete_void"
    config["project"]["baseline_id"] = (
        f"concrete_void_diameter_{2.0 * radius_mm:g}mm_"
        f"depth_{depth_mm:g}mm_x_{x_mm:g}mm"
    )
    config["simulation"]["primary_events"] = int(primary_events)
    if random_seed is not None:
        config["simulation"]["random_seed"] = int(random_seed)

    config["phantom"] = {
        **phantom,
        "name": "concrete_with_cylindrical_void",
        "gate_volume": "concrete",
        "measurement_role": "defect_detectability",
        "void": {
            "name": "void",
            "shape": "cylinder",
            "material": "Vacuum",
            "axis": "y",
            "center_mm": [float(x_mm), float(center_y), float(void_z)],
            "depth_mm": float(depth_mm),
            "radius_mm": float(radius_mm),
            "diameter_mm": 2.0 * float(radius_mm),
            "length_mm": float(length_mm),
            "local_z_mm": float(void_z - center_z),
        },
    }
    # A real reconstruction cannot know which histories scattered in concrete.
    # Truth remains recorded for validation, but it must not define the image.
    config["selection"]["phantom_truth_policy"] = "ignore"

    config = build_config_variant(
        config,
        d1_rmax_mm=d1_rmax_mm,
    )
    return validate_config(config)


def build_config_variant(
    base_config: dict[str, Any],
    *,
    primary_events: int | None = None,
    random_seed: int | None = None,
    d1_rmax_mm: float | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Create an auditable seed/count/D1-geometry variant of any config."""
    config = copy.deepcopy(base_config)
    suffixes: list[str] = []

    if primary_events is not None:
        if primary_events <= 0:
            raise ValueError("primary event count must be positive")
        config["simulation"]["primary_events"] = int(primary_events)
        suffixes.append(f"n_{int(primary_events)}")

    if random_seed is not None:
        if random_seed <= 0:
            raise ValueError("random seed must be positive")
        config["simulation"]["random_seed"] = int(random_seed)
        suffixes.append(f"seed_{int(random_seed)}")

    if d1_rmax_mm is not None:
        d1_rmin_mm = float(config["detectors"]["d1"]["radius_mm"][0])
        if d1_rmax_mm <= d1_rmin_mm:
            raise ValueError("D1 outer radius must exceed its inner radius")
        config["detectors"]["d1"]["radius_mm"][1] = float(d1_rmax_mm)
        config["selection"]["d1_radius_mm"][1] = float(d1_rmax_mm)
        radius_grid = config.get("sweep_grid", {}).get(
            "d1_radius_max_mm",
            [],
        )
        if radius_grid:
            config["sweep_grid"]["d1_radius_max_mm"] = [
                float(value)
                for value in radius_grid
                if float(value) <= float(d1_rmax_mm)
            ]
        suffixes.append(f"d1r_{float(d1_rmax_mm):g}mm")

    if variant_id:
        config["project"]["baseline_id"] = str(variant_id)
    elif suffixes:
        config["project"]["baseline_id"] += "_" + "_".join(suffixes)
    return validate_config(config)


def build_standard_config(
    base_config: dict[str, Any],
    *,
    kind: str,
    depth_mm: float,
    thickness_mm: float = 1.0,
    lateral_mm: float = 300.0,
    radius_mm: float = 2.5,
    separation_mm: float = 10.0,
    primary_events: int = 2_000_000,
    material: str = "Concrete",
    random_seed: int | None = None,
    d1_rmax_mm: float | None = None,
) -> dict[str, Any]:
    if kind not in STANDARD_TARGET_KINDS:
        raise ValueError(
            f"unknown standard target kind {kind!r}; "
            f"choose from {sorted(STANDARD_TARGET_KINDS)}"
        )
    if depth_mm <= 0.0 or thickness_mm <= 0.0 or lateral_mm <= 0.0:
        raise ValueError("depth, thickness and lateral size must be positive")
    if radius_mm <= 0.0:
        raise ValueError("target radius must be positive")
    if kind in {"double_point", "double_rod"} and separation_mm <= 2.0 * radius_mm:
        raise ValueError("double-target separation must exceed the target diameter")
    if primary_events <= 0:
        raise ValueError("primary event count must be positive")
    if material not in {"Concrete", "Water", "Iron"}:
        raise ValueError("standard target material must be Concrete, Water or Iron")
    if random_seed is not None and random_seed <= 0:
        raise ValueError("random seed must be positive")

    config = copy.deepcopy(base_config)
    config["project"]["name"] = "tagged_511keV_backscatter_standard_target"
    config["project"]["coordinate_convention"] = (
        "+z points from the target toward the detectors; z=0 mm is the "
        "system axial reference plane"
    )
    config["project"]["baseline_id"] = (
        f"{material.lower()}_{kind}_depth_{depth_mm:g}mm"
    )
    config["simulation"]["primary_events"] = int(primary_events)
    if random_seed is not None:
        config["simulation"]["random_seed"] = int(random_seed)

    if d1_rmax_mm is not None:
        d1_rmin_mm = float(config["detectors"]["d1"]["radius_mm"][0])
        if d1_rmax_mm <= d1_rmin_mm:
            raise ValueError("D1 outer radius must exceed its inner radius")
        config["detectors"]["d1"]["radius_mm"][1] = float(d1_rmax_mm)
        config["selection"]["d1_radius_mm"][1] = float(d1_rmax_mm)
        radius_grid = config.get("sweep_grid", {}).get(
            "d1_radius_max_mm",
            [],
        )
        if radius_grid:
            config["sweep_grid"]["d1_radius_max_mm"] = [
                float(value)
                for value in radius_grid
                if float(value) <= float(d1_rmax_mm)
            ]

    center_z = -float(depth_mm)
    if kind == "thin_layer":
        size = [float(lateral_mm), float(lateral_mm), float(thickness_mm)]
        center = [0.0, 0.0, center_z]
        target_centers = [[0.0, 0.0, center_z]]
    elif kind == "rod":
        diameter = 2.0 * float(radius_mm)
        size = [diameter, float(lateral_mm), diameter]
        center = [0.0, 0.0, center_z]
        target_centers = [[0.0, 0.0, center_z]]
    elif kind == "double_rod":
        diameter = 2.0 * float(radius_mm)
        size = [
            float(separation_mm) + diameter,
            float(lateral_mm),
            diameter,
        ]
        center = [0.0, 0.0, center_z]
        target_centers = [
            [-float(separation_mm) / 2.0, 0.0, center_z],
            [float(separation_mm) / 2.0, 0.0, center_z],
        ]
    elif kind == "point":
        diameter = 2.0 * float(radius_mm)
        size = [diameter, diameter, float(thickness_mm)]
        center = [0.0, 0.0, center_z]
        target_centers = [[0.0, 0.0, center_z]]
    elif kind == "double_point":
        diameter = 2.0 * float(radius_mm)
        size = [
            float(separation_mm) + diameter,
            diameter,
            float(thickness_mm),
        ]
        center = [0.0, 0.0, center_z]
        target_centers = [
            [-float(separation_mm) / 2.0, 0.0, center_z],
            [float(separation_mm) / 2.0, 0.0, center_z],
        ]
    else:
        size = [300.0, float(lateral_mm), float(thickness_mm)]
        center = [150.0, 0.0, center_z]
        target_centers = []
    z_interval = [
        center_z - float(size[2]) / 2.0,
        center_z + float(size[2]) / 2.0,
    ]

    config["phantom"] = {
        "name": kind,
        "gate_volume": "target",
        "material": material,
        "target_kind": kind,
        "size_mm": size,
        "center_mm": center,
        "z_interval_mm": z_interval,
        "depth_mm": float(depth_mm),
        "thickness_mm": float(size[2]),
        "radius_mm": float(radius_mm),
        "diameter_mm": 2.0 * float(radius_mm),
        "separation_mm": float(separation_mm),
        "target_centers_mm": target_centers,
        "edge_x_mm": 0.0 if kind == "edge" else None,
        "environment_material": "Vacuum",
        "measurement_role": "intrinsic_geometric_response",
        "phantom_sd": True,
        "truth_actor": "TruthTarget",
    }
    config["selection"]["phantom_truth_policy"] = (
        "prefer_single_phantom_compton"
    )
    config["selection"]["phantom_truth_volume_contains"] = "target"
    suffixes = []
    if random_seed is not None:
        suffixes.append(f"seed_{int(random_seed)}")
    if d1_rmax_mm is not None:
        suffixes.append(f"d1r_{float(d1_rmax_mm):g}mm")
    if suffixes:
        config["project"]["baseline_id"] += "_" + "_".join(suffixes)
    return validate_config(config)


def write_standard_config(
    path: str | Path,
    config: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
