from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .kinematics import solve_scatter_points
from .pipeline import build_selection
from .standards import estimate_fwhm_kde


FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def quantize_to_channel_centers(
    coordinate_mm: np.ndarray,
    *,
    pitch_mm: float,
) -> np.ndarray:
    """Quantize a symmetric even-channel array to half-pitch centers."""
    if pitch_mm <= 0.0:
        raise ValueError("channel pitch must be positive")
    coordinate = np.asarray(coordinate_mm, dtype=np.float64)
    return (
        np.floor(coordinate / float(pitch_mm)) + 0.5
    ) * float(pitch_mm)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _positions(table: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack(
        [
            table[f"{prefix}_x_mm"],
            table[f"{prefix}_y_mm"],
            table[f"{prefix}_z_mm"],
        ]
    ).astype(np.float64)


def _truth_signal_mask(
    pairs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    volumes = np.char.lower(pairs["d1_compt_volume"].astype(str))
    volume_name = str(
        config["selection"].get(
            "phantom_truth_volume_contains",
            config["phantom"].get("gate_volume", "concrete"),
        )
    ).lower()
    return (
        (pairs["d1_n_phantom_compton"] == 1)
        & (pairs["d2_n_phantom_compton"] == 0)
        & (np.char.find(volumes, volume_name) >= 0)
    )


def evaluate_position_readout_sensitivity(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    d1_xy_fwhm_mm: Iterable[float] = (0.0, 2.0, 5.0, 10.0),
    random_seed: int = 24062002,
) -> dict[str, Any]:
    """Evaluate ideal coordinates versus channelized/no-DOI readout models."""
    analysis_dir = Path(analysis_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    pairs = _load_npz(analysis_dir / "pairs.npz")
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    if len(pairs["key"]) != len(reconstruction["key"]) or not np.array_equal(
        pairs["key"],
        reconstruction["key"],
    ):
        raise ValueError("pair and reconstruction event keys are not aligned")
    with (analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)
    n_primaries = int(summary["event_counts"]["primaries"])

    fwhm_values = sorted(set(float(value) for value in d1_xy_fwhm_mm))
    if not fwhm_values:
        raise ValueError("at least one D1 position-resolution scenario is required")
    if any(value < 0.0 for value in fwhm_values):
        raise ValueError("D1 position FWHM cannot be negative")

    rng = np.random.default_rng(int(random_seed))
    normal_x = rng.standard_normal(len(pairs["key"]))
    normal_y = rng.standard_normal(len(pairs["key"]))
    source = _positions(pairs, "source")
    d1_true = _positions(pairs, "d1")
    d2_true = _positions(pairs, "d2")
    d2_pitch_x = float(config["detectors"]["d2"]["pitch_mm"][0])
    d2_pitch_y = float(config["detectors"]["d2"]["pitch_mm"][1])
    d2_center_z = float(config["detectors"]["d2"]["center_z_mm"])
    d1_center_z = float(config["detectors"]["d1"]["center_z_mm"])
    path_lo, path_hi = map(
        float,
        config["reconstruction"]["path_length_mm"],
    )
    truth_signal = _truth_signal_mask(pairs, config)
    truth_matched = reconstruction["truth_matched"].astype(bool)

    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "ideal_continuous_xyz",
            "d1_xy_fwhm_mm": None,
            "d1_depth_of_interaction": "ideal_continuous",
            "d2_readout": "ideal_continuous_xyz",
        }
    ]
    scenarios.extend(
        {
            "scenario": f"channelized_no_doi_d1xy_{value:g}mm",
            "d1_xy_fwhm_mm": float(value),
            "d1_depth_of_interaction": "none_use_detector_center",
            "d2_readout": "1mm_channel_centers_no_doi",
        }
        for value in fwhm_values
    )
    rows: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios):
        scenario_pairs = dict(pairs)
        if index == 0:
            d1 = d1_true.copy()
            d2 = d2_true.copy()
        else:
            position_fwhm = float(scenario["d1_xy_fwhm_mm"])
            sigma = position_fwhm / FWHM_TO_SIGMA
            d1 = d1_true.copy()
            d1[:, 0] += sigma * normal_x
            d1[:, 1] += sigma * normal_y
            d1[:, 2] = d1_center_z
            d2 = d2_true.copy()
            d2[:, 0] = quantize_to_channel_centers(
                d2[:, 0],
                pitch_mm=d2_pitch_x,
            )
            d2[:, 1] = quantize_to_channel_centers(
                d2[:, 1],
                pitch_mm=d2_pitch_y,
            )
            d2[:, 2] = d2_center_z

        scenario_pairs["d1_x_mm"] = d1[:, 0]
        scenario_pairs["d1_y_mm"] = d1[:, 1]
        scenario_pairs["d1_z_mm"] = d1[:, 2]
        scenario_pairs["d2_x_mm"] = d2[:, 0]
        scenario_pairs["d2_y_mm"] = d2[:, 1]
        scenario_pairs["d2_z_mm"] = d2[:, 2]
        observable_config = copy.deepcopy(config)
        observable_config["selection"]["phantom_truth_policy"] = "ignore"
        selected, _, _ = build_selection(scenario_pairs, observable_config)
        solution = solve_scatter_points(
            source,
            d1,
            d2,
            scenario_pairs["d1_energy_keV"],
            incident_energy_keV=float(
                config["simulation"]["incident_energy_keV"]
            ),
            electron_rest_energy_keV=float(
                config["simulation"]["electron_rest_energy_keV"]
            ),
            path_min_mm=path_lo,
            path_max_mm=path_hi,
        )
        usable = selected & solution.valid
        signal_usable = usable & truth_signal
        matched_usable = usable & truth_matched
        if np.any(matched_usable):
            depth_error = (
                -solution.points_mm[matched_usable, 2]
                + reconstruction["truth_z_mm"][matched_usable]
            )
            depth_rmse = float(np.sqrt(np.mean(depth_error**2)))
            error_fwhm = estimate_fwhm_kde(depth_error)
        else:
            depth_rmse = None
            error_fwhm = {"status": "insufficient_events", "fwhm": None}

        target_kind = config["phantom"].get("target_kind")
        if target_kind in {"thin_layer", "rod", "point"}:
            response_values = -solution.points_mm[signal_usable, 2]
            response_key = "target_axial_response_fwhm_mm"
        else:
            response_values = (
                -solution.points_mm[matched_usable, 2]
                + reconstruction["truth_z_mm"][matched_usable]
            )
            response_key = "eventwise_axial_error_fwhm_mm"
        response_fwhm = estimate_fwhm_kde(response_values)
        n_valid = int(usable.sum())
        n_signal = int(signal_usable.sum())
        row = {
            **scenario,
            "n_observable_selected_valid": n_valid,
            "selected_valid_per_primary": float(n_valid / n_primaries),
            "n_truth_single_scatter_signal": n_signal,
            "truth_signal_fraction_score_only": (
                float(n_signal / n_valid) if n_valid else None
            ),
            "n_actor_truth_matched": int(matched_usable.sum()),
            "eventwise_depth_rmse_mm": depth_rmse,
            "eventwise_depth_error_fwhm_mm": error_fwhm.get("fwhm"),
            response_key: response_fwhm.get("fwhm"),
            "response_fwhm_status": response_fwhm["status"],
        }
        rows.append(row)

    csv_path = out_dir / "position_readout_sensitivity.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False

    labels = ["Ideal"] + [f"{value:g}" for value in fwhm_values]
    x = np.arange(len(rows))
    efficiency = np.asarray(
        [row["selected_valid_per_primary"] for row in rows]
    )
    purity = np.asarray(
        [row["truth_signal_fraction_score_only"] for row in rows]
    )
    response_key = (
        "target_axial_response_fwhm_mm"
        if any("target_axial_response_fwhm_mm" in row for row in rows)
        else "eventwise_axial_error_fwhm_mm"
    )
    response_fwhm = np.asarray(
        [
            row.get(response_key)
            if row.get(response_key) is not None
            else np.nan
            for row in rows
        ]
    )
    rmse = np.asarray(
        [
            row["eventwise_depth_rmse_mm"]
            if row["eventwise_depth_rmse_mm"] is not None
            else np.nan
            for row in rows
        ]
    )
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    axes[0].plot(
        x,
        efficiency,
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=3.5,
    )
    axes[0].set_ylabel("Selected valid events / primary")
    axes[1].plot(
        x,
        purity,
        "o-",
        color="#B64342",
        linewidth=1.1,
        markersize=3.5,
    )
    axes[1].set_ylabel("Truth signal fraction (score only)")
    axes[2].plot(
        x,
        response_fwhm,
        "o-",
        color="#0F4D92",
        linewidth=1.1,
        markersize=3.5,
        label="Axial FWHM",
    )
    axes[2].plot(
        x,
        rmse,
        "s--",
        color="#767676",
        linewidth=1.0,
        markersize=3.0,
        label="Event-wise RMSE",
    )
    axes[2].set_ylabel("Axial error / response (mm)")
    axes[2].legend(fontsize=6.0)
    for label, ax in zip("abc", axes):
        ax.set_xticks(x, labels)
        ax.set_xlabel(
            "D1 xy FWHM (mm)\n"
            "(channelized rows use no D1/D2 DOI)"
        )
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.8)
    output_base = out_dir / "position_readout_sensitivity"
    figures: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figures.append(str(path.resolve()))
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "random_seed": int(random_seed),
        "common_random_numbers_across_d1_blur_scenarios": True,
        "model": {
            "ideal_row": (
                "Current GATE energy-weighted continuous interaction centroids"
            ),
            "channelized_rows": (
                "D2 x/y quantized to 1 mm crystal centers; D2 z fixed to "
                "detector center; D1 z fixed to annulus center (no DOI); D1 "
                "x/y Gaussian blur with the listed FWHM"
            ),
            "calibration_status": "hypothetical_readout_sensitivity",
        },
        "selection_uses_monte_carlo_truth": False,
        "truth_use": "post-selection scoring only",
        "rows": rows,
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "The ideal row is not directly achievable by a monolithic crystal "
            "without a position-decoding method. The channelized rows are "
            "sensitivity scenarios, not a finalized segmentation or optical "
            "readout design; measured position and DOI calibration must replace "
            "them before hardware claims."
        ),
    }
    with (out_dir / "position_readout_sensitivity.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
