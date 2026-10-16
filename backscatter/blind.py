from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import beta

from .config import load_config
from .voids import _control_centers, _observable_points, compute_void_cnr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_axis(
    minimum_mm: float,
    maximum_mm: float,
    step_mm: float,
) -> np.ndarray:
    if not np.isfinite([minimum_mm, maximum_mm, step_mm]).all():
        raise ValueError("search limits and step must be finite")
    if maximum_mm < minimum_mm:
        raise ValueError("search maximum must not be below its minimum")
    if step_mm <= 0.0:
        raise ValueError("search step must be positive")
    values = np.arange(
        float(minimum_mm),
        float(maximum_mm) + 0.5 * float(step_mm),
        float(step_mm),
        dtype=np.float64,
    )
    return values[values <= float(maximum_mm) + 1.0e-9]


def _tree_count(
    tree: cKDTree,
    *,
    center_x_mm: float,
    center_depth_mm: float,
    radius_mm: float,
) -> int:
    return int(
        tree.query_ball_point(
            [float(center_x_mm), float(center_depth_mm)],
            r=float(radius_mm),
            return_length=True,
        )
    )


def score_blind_points(
    *,
    sample_x_mm: np.ndarray,
    sample_depth_mm: np.ndarray,
    reference_x_mm: np.ndarray,
    reference_depth_mm: np.ndarray,
    roi_radius_mm: float,
    search_x_interval_mm: tuple[float, float],
    search_depth_interval_mm: tuple[float, float],
    search_step_mm: float,
    slab_x_interval_mm: tuple[float, float],
) -> dict[str, Any]:
    """Score an unknown sample without reading its class or target position.

    Each candidate location receives the same control-normalized Poisson
    deficit CNR used by the known-location pilot.  The blind case score is the
    maximum local CNR over the predeclared search grid.
    """
    if roi_radius_mm <= 0.0:
        raise ValueError("ROI radius must be positive")
    sample_x = np.asarray(sample_x_mm, dtype=np.float64)
    sample_depth = np.asarray(sample_depth_mm, dtype=np.float64)
    reference_x = np.asarray(reference_x_mm, dtype=np.float64)
    reference_depth = np.asarray(reference_depth_mm, dtype=np.float64)
    if len(sample_x) != len(sample_depth):
        raise ValueError("sample x and depth arrays must have equal lengths")
    if len(reference_x) != len(reference_depth):
        raise ValueError(
            "reference x and depth arrays must have equal lengths"
        )
    if not (
        np.isfinite(sample_x).all()
        and np.isfinite(sample_depth).all()
        and np.isfinite(reference_x).all()
        and np.isfinite(reference_depth).all()
    ):
        raise ValueError("blind-score point arrays must be finite")
    if len(sample_x) == 0 or len(reference_x) == 0:
        raise ValueError("sample and reference must both contain events")

    x_values = _candidate_axis(
        search_x_interval_mm[0],
        search_x_interval_mm[1],
        search_step_mm,
    )
    depth_values = _candidate_axis(
        search_depth_interval_mm[0],
        search_depth_interval_mm[1],
        search_step_mm,
    )
    sample_tree = cKDTree(np.column_stack((sample_x, sample_depth)))
    reference_tree = cKDTree(
        np.column_stack((reference_x, reference_depth))
    )

    rows: list[dict[str, Any]] = []
    for depth_index, depth_center in enumerate(depth_values):
        for x_index, x_center in enumerate(x_values):
            controls = _control_centers(
                void_x_mm=float(x_center),
                radius_mm=float(roi_radius_mm),
                slab_x_interval_mm=slab_x_interval_mm,
            )
            sample_roi = _tree_count(
                sample_tree,
                center_x_mm=float(x_center),
                center_depth_mm=float(depth_center),
                radius_mm=roi_radius_mm,
            )
            reference_roi = _tree_count(
                reference_tree,
                center_x_mm=float(x_center),
                center_depth_mm=float(depth_center),
                radius_mm=roi_radius_mm,
            )
            sample_controls = [
                _tree_count(
                    sample_tree,
                    center_x_mm=float(control_x),
                    center_depth_mm=float(depth_center),
                    radius_mm=roi_radius_mm,
                )
                for control_x in controls
            ]
            reference_controls = [
                _tree_count(
                    reference_tree,
                    center_x_mm=float(control_x),
                    center_depth_mm=float(depth_center),
                    radius_mm=roi_radius_mm,
                )
                for control_x in controls
            ]
            row: dict[str, Any] = {
                "x_index": int(x_index),
                "depth_index": int(depth_index),
                "candidate_x_mm": float(x_center),
                "candidate_depth_mm": float(depth_center),
                "roi_radius_mm": float(roi_radius_mm),
                "control_x_centers_mm": controls,
                "sample_control_counts_by_center": sample_controls,
                "reference_control_counts_by_center": reference_controls,
                "sample_roi_count": int(sample_roi),
                "reference_roi_count": int(reference_roi),
                "sample_control_count": int(sum(sample_controls)),
                "reference_control_count": int(sum(reference_controls)),
                "valid_score": False,
                "control_normalization_scale": None,
                "expected_no_void_roi_count": None,
                "event_deficit": None,
                "event_deficit_standard_deviation": None,
                "contrast": None,
                "cnr": None,
            }
            try:
                metrics = compute_void_cnr(
                    void_roi_count=sample_roi,
                    reference_roi_count=reference_roi,
                    void_control_count=sum(sample_controls),
                    reference_control_count=sum(reference_controls),
                )
            except ValueError:
                rows.append(row)
                continue
            row.update(
                {
                    "valid_score": True,
                    "control_normalization_scale": metrics[
                        "control_normalization_scale"
                    ],
                    "expected_no_void_roi_count": metrics[
                        "expected_no_void_roi_count"
                    ],
                    "event_deficit": metrics["event_deficit"],
                    "event_deficit_standard_deviation": metrics[
                        "event_deficit_standard_deviation"
                    ],
                    "contrast": metrics["contrast"],
                    "cnr": metrics["cnr"],
                }
            )
            rows.append(row)

    valid_rows = [row for row in rows if row["valid_score"]]
    if not valid_rows:
        raise ValueError("no search-grid candidate has a valid blind score")
    maximum = max(valid_rows, key=lambda row: float(row["cnr"]))
    return {
        "x_values_mm": x_values,
        "depth_values_mm": depth_values,
        "rows": rows,
        "maximum": maximum,
        "n_candidates": int(len(rows)),
        "n_valid_candidates": int(len(valid_rows)),
    }


def score_blind_void_case(
    *,
    sample_analysis_dir: str | Path,
    reference_analysis_dir: str | Path,
    selection_config_path: str | Path,
    out_dir: str | Path,
    roi_radius_mm: float = 15.0,
    x_min_mm: float = -30.0,
    x_max_mm: float = 30.0,
    depth_min_mm: float = 20.0,
    depth_max_mm: float = 40.0,
    search_step_mm: float = 2.0,
    y_half_width_mm: float = 150.0,
    candidate_threshold: float = 4.0,
) -> dict[str, Any]:
    """Blindly score one unknown concrete case against a fixed reference."""
    if y_half_width_mm <= 0.0:
        raise ValueError("y half-width must be positive")
    if not np.isfinite(candidate_threshold):
        raise ValueError("candidate threshold must be finite")

    sample_analysis_dir = Path(sample_analysis_dir)
    reference_analysis_dir = Path(reference_analysis_dir)
    selection_config_path = Path(selection_config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(selection_config_path)

    sample_points = _observable_points(
        sample_analysis_dir,
        config,
        y_half_width_mm=y_half_width_mm,
    )
    reference_points = _observable_points(
        reference_analysis_dir,
        config,
        y_half_width_mm=y_half_width_mm,
    )
    slab_center_x = float(config["phantom"]["center_mm"][0])
    slab_size_x = float(config["phantom"]["size_mm"][0])
    slab_x_interval = (
        slab_center_x - slab_size_x / 2.0,
        slab_center_x + slab_size_x / 2.0,
    )
    score = score_blind_points(
        sample_x_mm=sample_points["x_mm"],
        sample_depth_mm=sample_points["depth_mm"],
        reference_x_mm=reference_points["x_mm"],
        reference_depth_mm=reference_points["depth_mm"],
        roi_radius_mm=roi_radius_mm,
        search_x_interval_mm=(x_min_mm, x_max_mm),
        search_depth_interval_mm=(depth_min_mm, depth_max_mm),
        search_step_mm=search_step_mm,
        slab_x_interval_mm=slab_x_interval,
    )

    csv_path = out_dir / "blind_score_map.csv"
    csv_fields = [
        "candidate_x_mm",
        "candidate_depth_mm",
        "roi_radius_mm",
        "sample_roi_count",
        "reference_roi_count",
        "sample_control_count",
        "reference_control_count",
        "control_normalization_scale",
        "expected_no_void_roi_count",
        "event_deficit",
        "event_deficit_standard_deviation",
        "contrast",
        "cnr",
        "valid_score",
        "control_x_centers_mm",
        "sample_control_counts_by_center",
        "reference_control_counts_by_center",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in score["rows"]:
            output_row = {name: row.get(name) for name in csv_fields}
            for list_name in (
                "control_x_centers_mm",
                "sample_control_counts_by_center",
                "reference_control_counts_by_center",
            ):
                output_row[list_name] = ";".join(
                    str(value) for value in row[list_name]
                )
            writer.writerow(output_row)

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
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8

    x_values = score["x_values_mm"]
    depth_values = score["depth_values_mm"]
    score_grid = np.full(
        (len(depth_values), len(x_values)),
        np.nan,
        dtype=np.float64,
    )
    for row in score["rows"]:
        if row["valid_score"]:
            score_grid[row["depth_index"], row["x_index"]] = float(
                row["cnr"]
            )
    finite_scores = score_grid[np.isfinite(score_grid)]
    color_low = min(float(np.quantile(finite_scores, 0.02)), 0.0)
    color_high = max(
        float(np.quantile(finite_scores, 0.98)),
        float(candidate_threshold),
        1.0,
    )
    mm_to_inch = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=(90.0 * mm_to_inch, 78.0 * mm_to_inch))
    image = ax.imshow(
        score_grid,
        extent=[
            float(x_values[0] - search_step_mm / 2.0),
            float(x_values[-1] + search_step_mm / 2.0),
            float(depth_values[-1] + search_step_mm / 2.0),
            float(depth_values[0] - search_step_mm / 2.0),
        ],
        aspect="auto",
        cmap="viridis",
        vmin=color_low,
        vmax=color_high,
        interpolation="nearest",
        rasterized=True,
    )
    maximum = score["maximum"]
    ax.plot(
        maximum["candidate_x_mm"],
        maximum["candidate_depth_mm"],
        marker="*",
        color="white",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=9,
    )
    if (
        float(np.nanmin(score_grid)) <= candidate_threshold
        <= float(np.nanmax(score_grid))
    ):
        ax.contour(
            x_values,
            depth_values,
            score_grid,
            levels=[candidate_threshold],
            colors=["white"],
            linewidths=[0.8],
        )
    ax.set_xlabel("Candidate x (mm)")
    ax.set_ylabel("Candidate reconstructed depth (mm)")
    ax.set_title("Blind void score map (truth not used)")
    ax.text(
        0.03,
        0.97,
        f"max CNR = {float(maximum['cnr']):.2f}\n"
        f"threshold = {candidate_threshold:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7,
        bbox={
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.55,
            "pad": 2.0,
        },
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("Local control-normalized CNR")
    fig.tight_layout(pad=1.0)
    figure_base = out_dir / "blind_score_map"
    figure_paths: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = figure_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figure_paths.append(str(path.resolve()))
    plt.close(fig)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "blind_concrete_void_case_score",
        "sample_analysis_dir": str(sample_analysis_dir.resolve()),
        "reference_analysis_dir": str(reference_analysis_dir.resolve()),
        "selection_config": str(selection_config_path.resolve()),
        "selection_config_sha256": _sha256(selection_config_path),
        "blindness": {
            "uses_monte_carlo_truth": False,
            "reads_sample_class": False,
            "reads_target_position": False,
            "selection_truth_policy_forced_to": "ignore",
        },
        "task": {
            "target_class_for_search": (
                "20 mm-diameter y-axis cylindrical low-density void"
            ),
            "roi_radius_mm": float(roi_radius_mm),
            "search_x_interval_mm": [float(x_min_mm), float(x_max_mm)],
            "search_depth_interval_mm": [
                float(depth_min_mm),
                float(depth_max_mm),
            ],
            "search_step_mm": float(search_step_mm),
            "y_half_width_mm": float(y_half_width_mm),
            "candidate_threshold": float(candidate_threshold),
            "threshold_status": (
                "Frozen workflow-qualification threshold; a fixed 5% "
                "false-positive claim requires independent negative cases."
            ),
        },
        "event_counts": {
            "sample_observable_selected_valid_all_y": int(
                sample_points["n_observable_selected_valid_all_y"][0]
            ),
            "sample_inside_y_extent": int(len(sample_points["x_mm"])),
            "reference_observable_selected_valid_all_y": int(
                reference_points["n_observable_selected_valid_all_y"][0]
            ),
            "reference_inside_y_extent": int(
                len(reference_points["x_mm"])
            ),
        },
        "score_definition": (
            "Maximum control-normalized Poisson count-deficit CNR over every "
            "candidate on the frozen two-dimensional search grid."
        ),
        "n_candidates": score["n_candidates"],
        "n_valid_candidates": score["n_valid_candidates"],
        "maximum_score": float(maximum["cnr"]),
        "estimated_center_mm": {
            "x": float(maximum["candidate_x_mm"]),
            "depth": float(maximum["candidate_depth_mm"]),
        },
        "candidate_threshold": float(candidate_threshold),
        "candidate_flag": bool(
            float(maximum["cnr"]) >= float(candidate_threshold)
        ),
        "score_map_csv": str(csv_path.resolve()),
        "figures": figure_paths,
        "interpretation_guardrail": (
            "This file is one truth-blind case score. Detection probability, "
            "false-positive rate, and localization accuracy require a frozen "
            "multi-case protocol followed by answer-key reveal."
        ),
    }
    report_path = out_dir / "blind_case_score.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path.resolve())
    return report


def _binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> list[float] | None:
    if trials <= 0:
        return None
    if successes < 0 or successes > trials:
        raise ValueError("binomial successes must lie between zero and trials")
    alpha = 1.0 - float(confidence)
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta.ppf(
                1.0 - alpha / 2.0,
                successes + 1,
                trials - successes,
            )
        )
    )
    return [lower, upper]


def evaluate_blind_void_campaign(
    *,
    score_bundle_path: str | Path,
    committed_score_hash_path: str | Path,
    truth_key_path: str | Path,
    truth_salt_path: str | Path,
    truth_commitment_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Reveal a frozen blind campaign only after both commitments verify."""
    score_bundle_path = Path(score_bundle_path)
    committed_score_hash_path = Path(committed_score_hash_path)
    truth_key_path = Path(truth_key_path)
    truth_salt_path = Path(truth_salt_path)
    truth_commitment_path = Path(truth_commitment_path)
    out_dir = Path(out_dir)

    score_bytes = score_bundle_path.read_bytes()
    actual_score_hash = hashlib.sha256(score_bytes).hexdigest().upper()
    committed_score_hash = (
        committed_score_hash_path.read_text(encoding="ascii")
        .strip()
        .split()[0]
        .upper()
    )
    if (
        len(committed_score_hash) != 64
        or actual_score_hash != committed_score_hash
    ):
        raise ValueError("frozen score-bundle SHA-256 verification failed")
    scores = json.loads(score_bytes.decode("utf-8-sig"))
    if not scores.get("scores_frozen_before_truth_reveal"):
        raise ValueError("score bundle is not marked frozen before reveal")
    if scores.get("contains_truth_labels_or_positions"):
        raise ValueError("score bundle already contains private truth")

    truth_bytes = truth_key_path.read_bytes()
    salt = bytes.fromhex(
        truth_salt_path.read_text(encoding="ascii").strip()
    )
    expected_truth_commitment = (
        truth_commitment_path.read_text(encoding="ascii").strip().upper()
    )
    actual_truth_commitment = hashlib.sha256(
        salt + truth_bytes
    ).hexdigest().upper()
    if actual_truth_commitment != expected_truth_commitment:
        raise ValueError("truth-key commitment verification failed")
    if (
        str(scores["public_truth_commitment_sha256"]).upper()
        != expected_truth_commitment
    ):
        raise ValueError("score bundle was bound to a different truth key")
    truth = json.loads(truth_bytes.decode("utf-8"))
    if truth["campaign_tag"] != scores["campaign_tag"]:
        raise ValueError("truth and score campaign tags do not match")

    score_by_case = {
        str(row["case_id"]): row for row in scores["cases"]
    }
    truth_by_case = {
        str(row["case_id"]): row for row in truth["cases"]
    }
    if set(score_by_case) != set(truth_by_case):
        raise ValueError("truth and score case sets do not match")
    protocol = scores["frozen_protocol"]
    threshold = float(protocol["candidate_threshold"])
    tolerance = float(protocol["localization_tolerance_mm_after_reveal"])

    rows: list[dict[str, Any]] = []
    positive_errors: list[float] = []
    correct_detections = 0
    false_alarms = 0
    positive_count = 0
    negative_count = 0
    for case_id in sorted(score_by_case):
        score_row = score_by_case[case_id]
        truth_row = truth_by_case[case_id]
        maximum_score = float(score_row["maximum_score"])
        candidate_flag = bool(score_row["candidate_flag"])
        if candidate_flag != (maximum_score >= threshold):
            raise ValueError(f"candidate flag is inconsistent for {case_id}")
        estimated_x = float(score_row["estimated_x_mm"])
        estimated_depth = float(score_row["estimated_depth_mm"])
        private_class = str(truth_row["class"])
        row: dict[str, Any] = {
            "case_id": case_id,
            "class": private_class,
            "maximum_score": maximum_score,
            "candidate_threshold": threshold,
            "candidate_flag": candidate_flag,
            "estimated_x_mm": estimated_x,
            "estimated_depth_mm": estimated_depth,
            "true_x_mm": truth_row["x_mm"],
            "true_depth_mm": truth_row["depth_mm"],
            "localization_error_mm": None,
            "correct_localized_detection": None,
            "false_alarm": None,
        }
        if private_class == "void":
            positive_count += 1
            error = float(
                np.hypot(
                    estimated_x - float(truth_row["x_mm"]),
                    estimated_depth - float(truth_row["depth_mm"]),
                )
            )
            positive_errors.append(error)
            correct = bool(candidate_flag and error <= tolerance)
            correct_detections += int(correct)
            row["localization_error_mm"] = error
            row["correct_localized_detection"] = correct
            row["false_alarm"] = False
        elif private_class == "no_void":
            negative_count += 1
            false_alarms += int(candidate_flag)
            row["correct_localized_detection"] = False
            row["false_alarm"] = candidate_flag
        else:
            raise ValueError(f"unknown private class for {case_id}")
        rows.append(row)

    detection_rate = (
        float(correct_detections / positive_count)
        if positive_count
        else None
    )
    false_alarm_rate = (
        float(false_alarms / negative_count) if negative_count else None
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "blind_void_qualification_reveal",
        "campaign_tag": scores["campaign_tag"],
        "commitment_verification": {
            "score_bundle_sha256_verified": True,
            "score_bundle_sha256": actual_score_hash,
            "truth_key_commitment_verified": True,
            "truth_key_commitment_sha256": actual_truth_commitment,
        },
        "frozen_protocol": protocol,
        "positive_cases": {
            "n": positive_count,
            "correct_localized_detections": correct_detections,
            "detection_rate": detection_rate,
            "exact_binomial_95_ci": _binomial_interval(
                correct_detections,
                positive_count,
            ),
            "localization_error_median_mm": (
                float(np.median(positive_errors))
                if positive_errors
                else None
            ),
            "localization_error_90th_percentile_mm": (
                float(np.quantile(positive_errors, 0.9))
                if positive_errors
                else None
            ),
        },
        "negative_cases": {
            "n": negative_count,
            "false_alarms": false_alarms,
            "false_alarm_rate": false_alarm_rate,
            "exact_binomial_95_ci": _binomial_interval(
                false_alarms,
                negative_count,
            ),
        },
        "workflow_qualification_rule": {
            "requires_at_least_5_of_6_correct_localized_detections": True,
            "allows_at_most_1_of_6_false_alarms": True,
            "passed": bool(
                positive_count == 6
                and negative_count == 6
                and correct_detections >= 5
                and false_alarms <= 1
            ),
        },
        "cases": rows,
        "interpretation_guardrail": (
            "This 6-positive/6-negative batch qualifies the blinded workflow. "
            "Its binomial intervals are too wide for a final detector "
            "detection-probability or fixed-5%-false-positive claim."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

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
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False

    negative_rows = [row for row in rows if row["class"] == "no_void"]
    positive_rows = [row for row in rows if row["class"] == "void"]
    negative_jitter = np.linspace(-0.12, 0.12, len(negative_rows))
    positive_jitter = np.linspace(-0.12, 0.12, len(positive_rows))
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(135.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    axes[0].scatter(
        negative_jitter,
        [row["maximum_score"] for row in negative_rows],
        color="#767676",
        edgecolor="white",
        linewidth=0.4,
        s=24,
        label="No void",
        zorder=3,
    )
    axes[0].scatter(
        1.0 + positive_jitter,
        [row["maximum_score"] for row in positive_rows],
        color="#0F4D92",
        edgecolor="white",
        linewidth=0.4,
        s=24,
        label="20 mm void",
        zorder=3,
    )
    axes[0].axhline(
        threshold,
        color="#B64342",
        linestyle="--",
        linewidth=1.0,
        label=f"Threshold = {threshold:g}",
    )
    axes[0].set_xticks([0.0, 1.0], ["No void", "20 mm void"])
    axes[0].set_xlim(-0.35, 1.35)
    axes[0].set_ylabel("Frozen blind maximum CNR")
    axes[0].set_title("Blind score separation")
    axes[0].legend(fontsize=6.4, loc="upper left")

    axes[1].scatter(
        np.arange(1, len(positive_rows) + 1),
        [row["localization_error_mm"] for row in positive_rows],
        color="#0F4D92",
        edgecolor="white",
        linewidth=0.4,
        s=24,
        zorder=3,
    )
    axes[1].axhline(
        tolerance,
        color="#B64342",
        linestyle="--",
        linewidth=1.0,
        label=f"Tolerance = {tolerance:g} mm",
    )
    axes[1].set_xlabel("Positive case number after reveal")
    axes[1].set_ylabel("Two-dimensional localization error (mm)")
    axes[1].set_title("Localization after reveal")
    axes[1].set_xticks(np.arange(1, len(positive_rows) + 1))
    axes[1].set_ylim(
        0.0,
        max(
            tolerance * 1.08,
            max(
                float(row["localization_error_mm"])
                for row in positive_rows
            )
            * 1.18,
        ),
    )
    axes[1].legend(
        fontsize=6.4,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )

    for label, axis in zip("ab", axes):
        axis.text(
            -0.18,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.0, w_pad=2.0)
    figure_base = out_dir / "blind_qualification_summary"
    figure_paths: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = figure_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figure_paths.append(str(path.resolve()))
    plt.close(fig)
    report["figures"] = figure_paths

    json_path = out_dir / "blind_qualification_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    positive_ci = report["positive_cases"]["exact_binomial_95_ci"]
    negative_ci = report["negative_cases"]["exact_binomial_95_ci"]
    markdown_lines = [
        "# 水泥孔洞盲检流程验收结果",
        "",
        f"- 批次：`{scores['campaign_tag']}`",
        (
            f"- 正样本正确检出：{correct_detections}/{positive_count}，"
            f"比例 {detection_rate:.1%}，95% 区间 "
            f"{positive_ci[0]:.1%}–{positive_ci[1]:.1%}"
        ),
        (
            f"- 负样本误报：{false_alarms}/{negative_count}，"
            f"比例 {false_alarm_rate:.1%}，95% 区间 "
            f"{negative_ci[0]:.1%}–{negative_ci[1]:.1%}"
        ),
        (
            "- 流程验收："
            + (
                "通过"
                if report["workflow_qualification_rule"]["passed"]
                else "未通过"
            )
        ),
        "",
        "该批次只有 6 个正样本和 6 个负样本，只用于验收盲检流程，"
        "不能当作最终检出概率。",
        "",
        "| case | 答案 | 评分 | 报警 | 估计 x | 估计深度 | 定位误差 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        error_text = (
            "—"
            if row["localization_error_mm"] is None
            else f"{float(row['localization_error_mm']):.2f} mm"
        )
        markdown_lines.append(
            f"| {row['case_id']} | {row['class']} | "
            f"{row['maximum_score']:.2f} | "
            f"{'是' if row['candidate_flag'] else '否'} | "
            f"{row['estimated_x_mm']:.1f} mm | "
            f"{row['estimated_depth_mm']:.1f} mm | {error_text} |"
        )
    markdown_path = out_dir / "blind_qualification_results.md"
    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    report["json_path"] = str(json_path.resolve())
    report["markdown_path"] = str(markdown_path.resolve())
    return report
