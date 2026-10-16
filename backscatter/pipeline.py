from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .kinematics import (
    compton_angle_from_energy_keV,
    compton_scattered_energy_keV,
    incident_polar_angle_deg,
    solve_scatter_points,
)
from .rootio import (
    aggregate_hits_tree,
    build_pairs,
    pack_event_tables,
    read_gate_metadata,
    save_npz,
)
from .truth import match_truth_to_pairs, read_compton_truth


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positions(pairs: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack(
        [
            pairs[f"{prefix}_x_mm"],
            pairs[f"{prefix}_y_mm"],
            pairs[f"{prefix}_z_mm"],
        ]
    ).astype(np.float64)


def _source_positions(pairs: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [pairs["source_x_mm"], pairs["source_y_mm"], pairs["source_z_mm"]]
    ).astype(np.float64)


def build_selection(
    pairs: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[np.ndarray, OrderedDict[str, int], dict[str, Any]]:
    selection = config["selection"]
    n = len(pairs["key"])
    mask = np.ones(n, dtype=bool)
    chain: OrderedDict[str, int] = OrderedDict()

    def apply(name: str, condition: np.ndarray) -> None:
        nonlocal mask
        mask &= np.asarray(condition, dtype=bool)
        chain[name] = int(mask.sum())

    apply(
        "positive_detector_energy",
        (pairs["d1_energy_keV"] > 0.0) & (pairs["d2_energy_keV"] > 0.0),
    )
    if selection["require_different_primary_photons"]:
        apply("different_primary_photons", pairs["different_primary_photons"])

    apply(
        "coincidence_time_window",
        np.abs(pairs["dt_ns"]) <= float(selection["coincidence_window_ns"]),
    )

    d2_lo, d2_hi = map(float, selection["d2_energy_window_keV"])
    apply(
        "d2_energy_window",
        (pairs["d2_energy_keV"] >= d2_lo)
        & (pairs["d2_energy_keV"] <= d2_hi),
    )

    d1_lo, d1_hi = map(float, selection["d1_energy_window_keV"])
    apply(
        "d1_energy_window",
        (pairs["d1_energy_keV"] >= d1_lo)
        & (pairs["d1_energy_keV"] <= d1_hi),
    )

    d1_radius = np.hypot(pairs["d1_x_mm"], pairs["d1_y_mm"])
    r_lo, r_hi = map(float, selection["d1_radius_mm"])
    apply("d1_active_radius", (d1_radius >= r_lo) & (d1_radius <= r_hi))

    half = float(selection["d2_half_width_mm"])
    apply(
        "d2_active_area",
        (np.abs(pairs["d2_x_mm"]) <= half)
        & (np.abs(pairs["d2_y_mm"]) <= half),
    )

    source = _source_positions(pairs)
    d2 = _positions(pairs, "d2")
    incident_angle = incident_polar_angle_deg(source, d2)
    beta_lo, beta_hi = map(float, selection["incident_angle_deg"])
    apply(
        "incident_angle",
        np.isfinite(incident_angle)
        & (incident_angle >= beta_lo)
        & (incident_angle <= beta_hi),
    )

    theta, kinematic = compton_angle_from_energy_keV(
        pairs["d1_energy_keV"],
        incident_energy_keV=float(config["simulation"]["incident_energy_keV"]),
        electron_rest_energy_keV=float(
            config["simulation"]["electron_rest_energy_keV"]
        ),
    )
    theta_lo, theta_hi = map(float, selection["scatter_angle_deg"])
    apply(
        "single_compton_kinematics_and_scatter_angle",
        kinematic & (theta >= theta_lo) & (theta <= theta_hi),
    )

    volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
    truth_available = bool(
        np.any(pairs["d1_n_phantom_compton"] > 0)
        or np.any((volumes != "null") & (volumes != "unavailable"))
    )
    truth_applied = False
    policy = selection.get("phantom_truth_policy", "ignore")
    if policy in {
        "prefer_single_concrete_compton",
        "prefer_single_phantom_compton",
    } and truth_available:
        truth_volume = str(
            selection.get("phantom_truth_volume_contains", "concrete")
        ).lower()
        truth_condition = (
            (pairs["d1_n_phantom_compton"] == 1)
            & (pairs["d2_n_phantom_compton"] == 0)
            & (np.char.find(volumes, truth_volume) >= 0)
        )
        apply("truth_single_concrete_compton", truth_condition)
        truth_applied = True

    diagnostics = {
        "incident_angle_deg": incident_angle,
        "scatter_angle_deg": theta,
        "kinematic_valid": kinematic,
        "d1_radius_mm": d1_radius,
        "phantom_truth_available": truth_available,
        "phantom_truth_applied": truth_applied,
        "phantom_truth_policy": policy,
    }
    return mask, chain, diagnostics


def _histogram(values: np.ndarray, lo: float, hi: float, step: float) -> dict[str, Any]:
    edges = np.arange(lo, hi + step * 0.5, step)
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_width": step,
        "bin_centers": centers.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def _peak(values: np.ndarray, lo: float, hi: float, step: float = 1.0) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= lo) & (values <= hi)]
    if len(values) == 0:
        return None
    edges = np.arange(lo, hi + step, step)
    counts, edges = np.histogram(values, bins=edges)
    return float(0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]))


def _fraction(values: np.ndarray, lo: float, hi: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(((values >= lo) & (values <= hi)).mean())


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _truth_error_summary(
    reconstruction_points: np.ndarray,
    reconstruction_valid: np.ndarray,
    selected: np.ndarray,
    matched_truth: dict[str, np.ndarray],
) -> dict[str, Any]:
    usable = selected & reconstruction_valid & matched_truth["matched"]
    if not np.any(usable):
        return {
            "n_matched_selected_valid": 0,
            "note": "No truth vertices matched. Check ParentID-to-primaryID mapping and actor schema.",
        }
    truth_points = np.column_stack(
        [
            matched_truth["x_mm"],
            matched_truth["y_mm"],
            matched_truth["z_mm"],
        ]
    )
    delta = reconstruction_points[usable] - truth_points[usable]
    radial = np.linalg.norm(delta, axis=1)
    return {
        "n_matched_selected_valid": int(usable.sum()),
        "bias_mm": {
            "x": float(delta[:, 0].mean()),
            "y": float(delta[:, 1].mean()),
            "z": float(delta[:, 2].mean()),
        },
        "rmse_mm": {
            "x": float(np.sqrt(np.mean(delta[:, 0] ** 2))),
            "y": float(np.sqrt(np.mean(delta[:, 1] ** 2))),
            "z": float(np.sqrt(np.mean(delta[:, 2] ** 2))),
            "3d": float(np.sqrt(np.mean(radial**2))),
        },
        "absolute_error_quantiles_mm": {
            "x": _quantiles(np.abs(delta[:, 0])),
            "y": _quantiles(np.abs(delta[:, 1])),
            "z": _quantiles(np.abs(delta[:, 2])),
            "3d": _quantiles(radial),
        },
        "note": "This is event-wise reconstruction error against actor vertices, not the PSF of a standard target.",
    }


def analyze_root(
    *,
    root_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    truth_path: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root_path)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    d1_tree = config["detectors"]["d1"]["root_tree"]
    d2_tree = config["detectors"]["d2"]["root_tree"]

    print(f"[1/6] 聚合 {d1_tree} ...")
    d1_events = aggregate_hits_tree(root_path, d1_tree)
    print(f"      D1 事件数: {len(d1_events['key']):,}")
    print(f"[2/6] 聚合 {d2_tree} ...")
    d2_events = aggregate_hits_tree(root_path, d2_tree)
    print(f"      D2 事件数: {len(d2_events['key']):,}")

    metadata = read_gate_metadata(root_path)
    metadata["sha256"] = _sha256(root_path)
    save_npz(
        out_dir / "events.npz",
        pack_event_tables(d1_events, d2_events, metadata),
    )

    print("[3/6] 构建同一 GATE 事件的双探测器配对 ...")
    pairs = build_pairs(d1_events, d2_events)
    selected, chain, diagnostics = build_selection(pairs, config)
    pairs_to_save = dict(pairs)
    pairs_to_save.update(
        {
            "selected": selected,
            "incident_angle_deg": diagnostics["incident_angle_deg"],
            "scatter_angle_deg": diagnostics["scatter_angle_deg"],
            "kinematic_valid": diagnostics["kinematic_valid"],
            "d1_radius_mm": diagnostics["d1_radius_mm"],
        }
    )
    save_npz(out_dir / "pairs.npz", pairs_to_save)

    print("[4/6] 运行不裁剪非法能量的闭式散射点重建 ...")
    source = _source_positions(pairs)
    d1 = _positions(pairs, "d1")
    d2 = _positions(pairs, "d2")
    path_lo, path_hi = map(
        float, config["reconstruction"]["path_length_mm"]
    )
    solution = solve_scatter_points(
        source,
        d1,
        d2,
        pairs["d1_energy_keV"],
        incident_energy_keV=float(config["simulation"]["incident_energy_keV"]),
        electron_rest_energy_keV=float(
            config["simulation"]["electron_rest_energy_keV"]
        ),
        path_min_mm=path_lo,
        path_max_mm=path_hi,
    )
    selected_valid = selected & solution.valid

    truth_report: dict[str, Any] = {
        "available": False,
        "path": None,
    }
    matched_truth = {
        "matched": np.zeros(len(selected), dtype=bool),
        "x_mm": np.full(len(selected), np.nan),
        "y_mm": np.full(len(selected), np.nan),
        "z_mm": np.full(len(selected), np.nan),
        "n_compton_vertices": np.zeros(len(selected), dtype=np.int16),
    }
    if truth_path is not None:
        truth_file = Path(truth_path)
        print("[5/6] 读取并匹配 PhaseSpaceActor 康普顿顶点 ...")
        truth_table = read_compton_truth(truth_file)
        matched_truth = match_truth_to_pairs(pairs, truth_table)
        truth_report = {
            "available": True,
            "path": str(truth_file.resolve()),
            "sha256": _sha256(truth_file),
            "tree_name": truth_table["tree_name"],
            "n_actor_rows": truth_table["n_actor_rows"],
            "n_compton_rows": truth_table["n_compton_rows"],
            "n_unique_event_parent": int(len(truth_table["key"])),
            "n_pairs_matched": int(matched_truth["matched"].sum()),
        }
    else:
        print("[5/6] 未提供 truth actor 文件，跳过顶点真值匹配。")

    reconstruction = {
        "key": pairs["key"],
        "selected": selected,
        "valid": solution.valid,
        "selected_valid": selected_valid,
        "x_mm": solution.points_mm[:, 0],
        "y_mm": solution.points_mm[:, 1],
        "z_mm": solution.points_mm[:, 2],
        "path_length_mm": solution.path_length_mm,
        "theta_deg": solution.theta_deg,
        "alpha_deg": solution.alpha_deg,
        "failure_code": solution.failure_code,
        "truth_matched": matched_truth["matched"],
        "truth_x_mm": matched_truth["x_mm"],
        "truth_y_mm": matched_truth["y_mm"],
        "truth_z_mm": matched_truth["z_mm"],
        "truth_n_compton_vertices": matched_truth["n_compton_vertices"],
    }
    save_npz(out_dir / "reconstruction.npz", reconstruction)

    print("[6/6] 生成审计摘要 ...")
    e0 = float(config["simulation"]["incident_energy_keV"])
    mec2 = float(config["simulation"]["electron_rest_energy_keV"])
    backscatter_energy = compton_scattered_energy_keV(
        np.array([180.0, 90.0]),
        incident_energy_keV=e0,
        electron_rest_energy_keV=mec2,
    )
    back_lo, back_hi = float(backscatter_energy[0]), float(backscatter_energy[1])

    n_primary = int(
        metadata.get(
            "primary_event_count",
            config["simulation"]["primary_events"],
        )
    )
    n_selected = int(selected.sum())
    n_valid = int(selected_valid.sum())
    efficiency = n_valid / n_primary if n_primary > 0 else None
    sigma_eff = np.sqrt(n_valid) / n_primary if n_primary > 0 else None

    recon_points = solution.points_mm[selected_valid]
    phantom_z_lo, phantom_z_hi = map(float, config["phantom"]["z_interval_mm"])

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "inputs": {
            "root": str(root_path.resolve()),
            "root_sha256": metadata["sha256"],
            "truth": truth_report,
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
        },
        "gate_metadata": metadata,
        "canonical_geometry": {
            "source_position_mm": config["source"]["position_mm"],
            "d1_radius_mm": config["detectors"]["d1"]["radius_mm"],
            "d1_center_z_mm": config["detectors"]["d1"]["center_z_mm"],
            "d2_center_z_mm": config["detectors"]["d2"]["center_z_mm"],
            "d2_pitch_mm": config["detectors"]["d2"]["pitch_mm"],
            "phantom_z_interval_mm": config["phantom"]["z_interval_mm"],
        },
        "event_counts": {
            "primaries": n_primary,
            "d1_aggregated_events": int(len(d1_events["key"])),
            "d2_aggregated_events": int(len(d2_events["key"])),
            "common_event_pairs_before_filters": int(len(pairs["key"])),
            "selection_chain": chain,
            "selected_before_geometric_solver": n_selected,
            "selected_and_reconstructed": n_valid,
            "solver_failure_codes_all_pairs": {
                str(code): int((solution.failure_code == code).sum())
                for code in range(5)
            },
        },
        "phantom_truth": {
            "hits_truth_available": diagnostics["phantom_truth_available"],
            "hits_truth_applied_to_selection": diagnostics["phantom_truth_applied"],
            "d1_events_nPhantomCompton_0": int(
                (d1_events["n_phantom_compton"] == 0).sum()
            ),
            "d1_events_nPhantomCompton_1": int(
                (d1_events["n_phantom_compton"] == 1).sum()
            ),
            "d1_events_nPhantomCompton_gt1": int(
                (d1_events["n_phantom_compton"] > 1).sum()
            ),
            "actor": truth_report,
        },
        "energy_validation": {
            "incident_energy_keV": e0,
            "single_backscatter_band_keV_for_90_to_180_deg": [
                back_lo,
                back_hi,
            ],
            "d1_all_event_mean_keV": float(d1_events["energy_keV"].mean()),
            "d2_all_event_mean_keV": float(d2_events["energy_keV"].mean()),
            "d2_photopeak_mode_keV_450_to_530": _peak(
                d2_events["energy_keV"], 450.0, 530.0
            ),
            "d2_fraction_500_to_522_keV": _fraction(
                d2_events["energy_keV"], 500.0, 522.0
            ),
            "d1_fraction_in_theoretical_backscatter_band": _fraction(
                d1_events["energy_keV"], back_lo, back_hi
            ),
            "d1_spectrum_all_events": _histogram(
                d1_events["energy_keV"], 0.0, 600.0, 2.0
            ),
            "d2_spectrum_all_events": _histogram(
                d2_events["energy_keV"], 0.0, 600.0, 2.0
            ),
            "note": "These are event-summed raw deposited energies; no electronics broadening has been added.",
        },
        "angles": {
            "incident_angle_selected_quantiles_deg": _quantiles(
                diagnostics["incident_angle_deg"][selected]
            ),
            "scatter_angle_selected_quantiles_deg": _quantiles(
                diagnostics["scatter_angle_deg"][selected]
            ),
        },
        "reconstruction_distribution_not_resolution": {
            "n": n_valid,
            "x_quantiles_mm": _quantiles(recon_points[:, 0])
            if n_valid
            else None,
            "y_quantiles_mm": _quantiles(recon_points[:, 1])
            if n_valid
            else None,
            "z_quantiles_mm": _quantiles(recon_points[:, 2])
            if n_valid
            else None,
            "fraction_inside_concrete_z_interval": _fraction(
                recon_points[:, 2] if n_valid else np.array([]),
                phantom_z_lo,
                phantom_z_hi,
            ),
            "note": "The homogeneous slab distribution width is deliberately not reported as spatial resolution.",
        },
        "truth_reconstruction_error": _truth_error_summary(
            solution.points_mm, solution.valid, selected, matched_truth
        ),
        "ideal_true_coincidence_sensitivity": {
            "selected_valid_per_primary": efficiency,
            "poisson_1sigma_per_primary": float(sigma_eff)
            if sigma_eff is not None
            else None,
            "cps_per_MBq_low_rate_equivalent": efficiency * 1.0e6
            if efficiency is not None
            else None,
            "note": "Assumes one primary event per decay and excludes randoms, dead time and pile-up.",
        },
        "warnings": [],
    }

    if not diagnostics["phantom_truth_available"]:
        summary["warnings"].append(
            "Hits phantom truth is unavailable (all nPhantomCompton are zero or volume names are NULL). "
            "This is expected for the legacy macro but fails the new baseline acceptance criterion."
        )
    if truth_path is not None and truth_report.get("n_pairs_matched", 0) == 0:
        summary["warnings"].append(
            "Truth actor file was read but no pair matched EventID/ParentID; inspect the actor schema and primary-track mapping."
        )
    if metadata.get("primary_event_count") is None:
        summary["warnings"].append(
            "ROOT primary count was not found; sensitivity normalization used the configured count."
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
