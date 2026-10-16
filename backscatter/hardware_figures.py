from __future__ import annotations

import copy
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .kinematics import solve_scatter_points
from .pipeline import build_selection
from .response import apply_detector_response
from .standards import estimate_fwhm_kde


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
    pairs: dict[str, np.ndarray], config: dict[str, Any]
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


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    return plt


def _save_figure(fig: Any, output_base: Path) -> list[str]:
    figures: list[str] = []
    for extension, dpi in (
        ("svg", 300),
        ("pdf", 300),
        ("tiff", 600),
        ("png", 300),
    ):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        figures.append(str(path.resolve()))
    return figures


def resolve_hardware_profile_for_variant(
    *,
    base_config_path: str | Path,
    hardware_profile_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Apply the hardware response while preserving a variant target geometry."""
    base_config_path = Path(base_config_path)
    hardware_profile_path = Path(hardware_profile_path)
    output_path = Path(output_path)
    with base_config_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    hardware = load_config(hardware_profile_path)
    resolved["schema_version"] = "1.1"
    resolved["project"]["baseline_id"] = (
        f"{resolved['project']['baseline_id']}_hardware_response_20260731"
    )
    resolved["simulation"]["simultaneous_source_count"] = hardware[
        "simulation"
    ]["simultaneous_source_count"]
    for key in (
        "catalog_model",
        "nominal_activity_Bq_per_source",
        "inventory_quantity",
        "capsule_outer_diameter_mm",
        "capsule_height_mm",
        "active_part_diameter_mm",
        "production_period_weeks",
        "radionuclide_status",
        "transport_model_status",
    ):
        resolved["source"][key] = hardware["source"][key]
    for detector_name in ("d1", "d2"):
        resolved["detectors"][detector_name]["material"] = "GAGG"
        resolved["detectors"][detector_name]["energy_response"] = copy.deepcopy(
            hardware["detectors"][detector_name]["energy_response"]
        )
        resolved["detectors"][detector_name]["position_response"] = copy.deepcopy(
            hardware["detectors"][detector_name]["position_response"]
        )
    resolved["selection"]["d1_energy_window_keV"] = copy.deepcopy(
        hardware["selection"]["d1_energy_window_keV"]
    )
    resolved["selection"]["d2_energy_window_keV"] = copy.deepcopy(
        hardware["selection"]["d2_energy_window_keV"]
    )
    resolved["reconstruction"] = copy.deepcopy(hardware["reconstruction"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
    load_config(output_path)
    return output_path


def prepare_hardware_response_analysis(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    random_seed: int = 24073101,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    ideal_pairs = _load_npz(analysis_dir / "pairs.npz")
    ideal_reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        ideal_summary = json.load(handle)

    measured_pairs, response_report = apply_detector_response(
        ideal_pairs,
        config,
        random_seed=random_seed,
    )
    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    selected, selection_chain, diagnostics = build_selection(
        measured_pairs,
        observable_config,
    )
    source = _positions(measured_pairs, "source")
    d1 = _positions(measured_pairs, "d1")
    d2 = _positions(measured_pairs, "d2")
    path_lo, path_hi = map(float, config["reconstruction"]["path_length_mm"])
    solution = solve_scatter_points(
        source,
        d1,
        d2,
        measured_pairs["d1_energy_keV"],
        incident_energy_keV=float(config["simulation"]["incident_energy_keV"]),
        electron_rest_energy_keV=float(
            config["simulation"]["electron_rest_energy_keV"]
        ),
        path_min_mm=path_lo,
        path_max_mm=path_hi,
    )
    selected_valid = selected & solution.valid

    measured_pairs_to_save = dict(measured_pairs)
    for prefix in ("d1", "d2"):
        measured_pairs_to_save[f"ideal_{prefix}_energy_keV"] = ideal_pairs[
            f"{prefix}_energy_keV"
        ]
        for axis in "xyz":
            measured_pairs_to_save[f"ideal_{prefix}_{axis}_mm"] = ideal_pairs[
                f"{prefix}_{axis}_mm"
            ]
    measured_pairs_to_save["selected_hardware_response"] = selected
    measured_pairs_to_save["incident_angle_deg_hardware_response"] = diagnostics[
        "incident_angle_deg"
    ]
    measured_pairs_to_save["scatter_angle_deg_hardware_response"] = diagnostics[
        "scatter_angle_deg"
    ]
    np.savez_compressed(out_dir / "pairs_response.npz", **measured_pairs_to_save)

    response_reconstruction = {
        "key": measured_pairs["key"],
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
        "truth_matched": ideal_reconstruction["truth_matched"],
        "truth_x_mm": ideal_reconstruction["truth_x_mm"],
        "truth_y_mm": ideal_reconstruction["truth_y_mm"],
        "truth_z_mm": ideal_reconstruction["truth_z_mm"],
        "truth_n_compton_vertices": ideal_reconstruction[
            "truth_n_compton_vertices"
        ],
    }
    np.savez_compressed(
        out_dir / "reconstruction_response.npz",
        **response_reconstruction,
    )

    n_primaries = int(ideal_summary["event_counts"]["primaries"])
    truth_matched = ideal_reconstruction["truth_matched"].astype(bool)
    scored = selected_valid & truth_matched
    if np.any(scored):
        true_depth = -ideal_reconstruction["truth_z_mm"][scored]
        reconstructed_depth = -solution.points_mm[scored, 2]
        residual = reconstructed_depth - true_depth
        depth_bias = float(np.mean(residual))
        depth_rmse = float(np.sqrt(np.mean(residual**2)))
    else:
        depth_bias = None
        depth_rmse = None

    report = {
        "schema_version": "1.0",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "response": response_report,
        "selection_chain": selection_chain,
        "n_primaries": n_primaries,
        "n_initial_pairs": int(len(measured_pairs["key"])),
        "n_selected": int(selected.sum()),
        "n_selected_valid": int(selected_valid.sum()),
        "selected_valid_fraction_of_pairs": float(
            selected_valid.sum() / len(selected_valid)
        ),
        "n_selected_valid_truth_matched": int(scored.sum()),
        "eventwise_depth_bias_mm": depth_bias,
        "eventwise_depth_rmse_mm": depth_rmse,
        "outputs": {
            "pairs": str((out_dir / "pairs_response.npz").resolve()),
            "reconstruction": str(
                (out_dir / "reconstruction_response.npz").resolve()
            ),
        },
    }
    with (out_dir / "response_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def materialize_response_analysis_view(
    *,
    source_analysis_dir: str | Path,
    response_dir: str | Path,
) -> dict[str, Any]:
    """Expose response-applied pairs/reconstruction to existing evaluators.

    The standard, void, sweep and detector-region evaluators expect the
    conventional ``pairs.npz``, ``reconstruction.npz`` and ``summary.json``
    filenames.  This helper creates a response-specific analysis view without
    altering the original ideal-response analysis directory.  Large NPZ files
    are hard-linked when possible and copied only as a fallback.

    ``events.npz`` is intentionally not exposed because it contains the raw
    GATE detector-event energies before the detector-response model.
    """
    source_analysis_dir = Path(source_analysis_dir)
    response_dir = Path(response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)
    with (response_dir / "response_analysis.json").open(
        "r", encoding="utf-8"
    ) as handle:
        response_report = json.load(handle)
    with (source_analysis_dir / "summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)

    aliases: dict[str, dict[str, str]] = {}
    for output_name, response_name in (
        ("pairs.npz", "pairs_response.npz"),
        ("reconstruction.npz", "reconstruction_response.npz"),
    ):
        source = response_dir / response_name
        destination = response_dir / output_name
        if destination.exists():
            destination.unlink()
        method = "hardlink"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
            method = "copy"
        aliases[output_name] = {
            "source": str(source.resolve()),
            "path": str(destination.resolve()),
            "method": method,
        }

    event_counts = summary.setdefault("event_counts", {})
    event_counts["common_event_pairs_before_filters"] = int(
        response_report["n_initial_pairs"]
    )
    event_counts["selection_chain"] = response_report["selection_chain"]
    event_counts["selected_before_geometric_solver"] = int(
        response_report["n_selected"]
    )
    event_counts["selected_and_reconstructed"] = int(
        response_report["n_selected_valid"]
    )
    summary["detector_response_analysis"] = {
        "status": "parameterized_postprocessing_response_applied",
        "response_report": str(
            (response_dir / "response_analysis.json").resolve()
        ),
        "raw_events_exposed": False,
    }
    warnings = summary.setdefault("warnings", [])
    response_warning = (
        "This analysis view contains D1/D2 response-applied paired events and "
        "reconstruction. Raw events.npz is deliberately omitted."
    )
    if response_warning not in warnings:
        warnings.append(response_warning)
    summary_path = response_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    manifest = {
        "schema_version": "1.0",
        "source_analysis_dir": str(source_analysis_dir.resolve()),
        "response_dir": str(response_dir.resolve()),
        "aliases": aliases,
        "summary": str(summary_path.resolve()),
        "guardrail": (
            "Use this directory only for paired-event/reconstruction "
            "evaluators. Raw event spectra must be read from the source "
            "analysis and explicitly passed through the response model."
        ),
    }
    with (response_dir / "analysis_view_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def plot_paired_spectra_before_after(
    *,
    analysis_dir: str | Path,
    response_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    response_dir = Path(response_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    pairs = _load_npz(response_dir / "pairs_response.npz")
    reconstruction = _load_npz(response_dir / "reconstruction_response.npz")
    events = _load_npz(analysis_dir / "events.npz")
    before = np.ones(len(pairs["key"]), dtype=bool)
    after = reconstruction["selected_valid"].astype(bool)
    denominator = float(len(before))
    edges = np.arange(0.0, 625.0, 5.0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, float]] = []
    spectra: dict[str, dict[str, np.ndarray]] = {}
    for stage_name, mask in (("before_selection", before), ("after_selection", after)):
        spectra[stage_name] = {}
        for prefix in ("d1", "d2"):
            counts, _ = np.histogram(pairs[f"{prefix}_energy_keV"][mask], bins=edges)
            fraction = counts.astype(np.float64) / denominator
            spectra[stage_name][prefix] = fraction
            for center, count, value in zip(centers, counts, fraction):
                rows.append(
                    {
                        "stage": stage_name,
                        "detector": prefix.upper(),
                        "energy_bin_center_keV": float(center),
                        "events": int(count),
                        "fraction_of_initial_pairs_per_5keV": float(value),
                    }
                )

    csv_path = out_dir / "paired_spectra_before_after.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw_d2 = np.asarray(events["d2_energy_keV"], dtype=np.float64)
    raw_photopeak = (raw_d2 >= 510.0) & (raw_d2 < 512.0)
    d2_lo, d2_hi = map(float, config["selection"]["d2_energy_window_keV"])
    measured_d2_window = (
        (pairs["d2_energy_keV"] >= d2_lo)
        & (pairs["d2_energy_keV"] <= d2_hi)
    )
    d1_lo, d1_hi = map(float, config["selection"]["d1_energy_window_keV"])

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(183.0 * mm_to_inch, 108.0 * mm_to_inch),
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )
    blue = "#165A9F"
    orange = "#D67A1F"
    grey = "#6E7781"
    for ax, stage_name, panel, label in (
        (axes[0], "before_selection", "a", "Detector response applied; before event selection"),
        (axes[1], "after_selection", "b", "After coincidence, energy, geometry and valid-solution cuts"),
    ):
        ax.step(
            centers,
            spectra[stage_name]["d1"],
            where="mid",
            color=blue,
            linewidth=1.35,
            label="D1 backscatter detector",
        )
        ax.step(
            centers,
            spectra[stage_name]["d2"],
            where="mid",
            color=orange,
            linewidth=1.35,
            label="D2 tag detector",
        )
        ax.axvspan(d1_lo, d1_hi, color=blue, alpha=0.08)
        ax.axvspan(d2_lo, d2_hi, color=orange, alpha=0.08)
        ax.set_ylabel("Fraction of initial pairs / 5 keV")
        ax.set_ylim(bottom=0.0)
        ax.text(
            0.01,
            0.94,
            panel,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
        ax.text(
            0.06,
            0.94,
            label,
            transform=ax.transAxes,
            color=grey,
            fontsize=7.2,
            va="top",
        )
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.52, 0.98), fontsize=7)
    axes[0].text(
        0.99,
        0.60,
        (
            f"D1: 8% FWHM @ 511 keV\n"
            f"D2: 12% FWHM @ 511 keV\n"
            f"Initial paired events: {len(before):,}"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    axes[1].text(
        0.99,
        0.92,
        (
            f"Retained: {after.sum():,} / {len(after):,} "
            f"({after.mean():.2%})\n"
            f"D1 window: {d1_lo:.2f}–{d1_hi:.1f} keV; "
            f"D2 window: {d2_lo:.0f}–{d2_hi:.0f} keV"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    axes[1].set_xlabel("Measured event energy (keV)")
    axes[1].set_xlim(0.0, 620.0)
    fig.tight_layout(pad=0.9)
    figures = _save_figure(fig, out_dir / "paired_spectra_before_after")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "n_initial_pairs": int(len(before)),
        "n_selected_valid": int(after.sum()),
        "selected_valid_fraction_of_pairs": float(after.mean()),
        "raw_gate_d2_photopeak_definition": "all D2 events with 510 <= E < 512 keV before detector broadening",
        "raw_gate_d2_photopeak_count": int(raw_photopeak.sum()),
        "raw_gate_d2_event_count": int(len(raw_d2)),
        "raw_gate_d2_photopeak_fraction": float(raw_photopeak.mean()),
        "measured_d2_window_fraction_of_initial_pairs": float(
            measured_d2_window.mean()
        ),
        "source_data": str(csv_path.resolve()),
        "figures": figures,
        "normalization": "each bin is divided by the same initial D1-D2 pair count",
    }
    with (out_dir / "paired_spectra_before_after.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def plot_depth_truth_projection_and_bias(
    *,
    response_dir: str | Path,
    out_dir: str | Path,
    truth_depth_centres_mm: Iterable[float] = (10.0, 30.0, 50.0, 70.0, 90.0),
    truth_bin_width_mm: float = 10.0,
    scatter_seed: int = 24073102,
    maximum_scatter_points: int = 15000,
) -> dict[str, Any]:
    response_dir = Path(response_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = _load_npz(response_dir / "reconstruction_response.npz")
    usable = (
        reconstruction["selected_valid"].astype(bool)
        & reconstruction["truth_matched"].astype(bool)
    )
    true_depth = -np.asarray(reconstruction["truth_z_mm"][usable], dtype=np.float64)
    reconstructed_depth = -np.asarray(reconstruction["z_mm"][usable], dtype=np.float64)
    residual = reconstructed_depth - true_depth
    if len(true_depth) == 0:
        raise ValueError("no selected valid truth-matched events are available")

    depth_centres = [float(value) for value in truth_depth_centres_mm]
    edges = np.arange(-20.0, 161.0, 2.0)
    x = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    projection_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    half_width = float(truth_bin_width_mm) / 2.0
    for centre in depth_centres:
        mask = (true_depth >= centre - half_width) & (
            true_depth < centre + half_width
        )
        counts, _ = np.histogram(reconstructed_depth[mask], bins=edges)
        density = counts.astype(np.float64)
        if density.sum() > 0.0:
            density /= density.sum() * float(edges[1] - edges[0])
            density = np.convolve(density, kernel, mode="same")
        curves.append(
            {
                "truth_depth_mm": centre,
                "n": int(mask.sum()),
                "density": density,
            }
        )
        for centre_x, count, value in zip(x, counts, density):
            projection_rows.append(
                {
                    "truth_depth_mm": centre,
                    "truth_bin_left_mm": centre - half_width,
                    "truth_bin_right_mm": centre + half_width,
                    "reconstructed_depth_bin_center_mm": float(centre_x),
                    "events": int(count),
                    "smoothed_probability_density": float(value),
                }
            )
    projection_csv = out_dir / "depth_truth_projection.csv"
    with projection_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projection_rows[0]))
        writer.writeheader()
        writer.writerows(projection_rows)

    event_csv = out_dir / "depth_bias_scatter_source.csv"
    with event_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["true_depth_mm", "reconstructed_depth_mm", "depth_residual_mm"]
        )
        writer.writerows(zip(true_depth, reconstructed_depth, residual))

    summary_edges = np.arange(0.0, 101.0, 10.0)
    summary_centres = 0.5 * (summary_edges[:-1] + summary_edges[1:])
    medians = np.full(len(summary_centres), np.nan)
    q25 = np.full(len(summary_centres), np.nan)
    q75 = np.full(len(summary_centres), np.nan)
    counts = np.zeros(len(summary_centres), dtype=int)
    for index, (left, right) in enumerate(zip(summary_edges[:-1], summary_edges[1:])):
        mask = (true_depth >= left) & (true_depth < right)
        counts[index] = int(mask.sum())
        if counts[index] > 0:
            q25[index], medians[index], q75[index] = np.quantile(
                residual[mask], [0.25, 0.5, 0.75]
            )
    summary_csv = out_dir / "depth_bias_binned_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "true_depth_bin_center_mm",
                "n_events",
                "residual_q25_mm",
                "residual_median_mm",
                "residual_q75_mm",
            ]
        )
        writer.writerows(zip(summary_centres, counts, q25, medians, q75))

    rng = np.random.default_rng(int(scatter_seed))
    if len(true_depth) > int(maximum_scatter_points):
        chosen = rng.choice(
            len(true_depth),
            size=int(maximum_scatter_points),
            replace=False,
        )
    else:
        chosen = np.arange(len(true_depth))

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(150.0 * mm_to_inch, 78.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.04, 1.0]},
    )
    colors = ["#225EA8", "#1D91C0", "#41B6C4", "#FE9929", "#CC4C02"]
    ax = axes[0]
    for curve, color in zip(curves, colors):
        ax.plot(
            x,
            curve["density"],
            color=color,
            linewidth=1.3,
            label=(
                f"truth {curve['truth_depth_mm']:.0f}±{half_width:.0f} mm "
                f"(n={curve['n']:,})"
            ),
        )
        ax.axvline(
            curve["truth_depth_mm"],
            color=color,
            linestyle="--",
            linewidth=0.65,
            alpha=0.75,
        )
    ax.set_xlabel("Reconstructed depth (mm)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(-10.0, 145.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=6.0, ncol=1, loc="upper right")
    ax.text(-0.12, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1]
    ax.scatter(
        true_depth[chosen],
        residual[chosen],
        s=5,
        alpha=0.13,
        color="#2B6CB0",
        edgecolors="none",
        rasterized=True,
        label="truth-matched events",
    )
    ax.axhline(0.0, color="#6E7781", linestyle="--", linewidth=0.9)
    finite = np.isfinite(medians)
    ax.plot(
        summary_centres[finite],
        medians[finite],
        color="#B53A3A",
        marker="o",
        markersize=3.2,
        linewidth=1.3,
        label="10 mm-bin median",
    )
    ax.fill_between(
        summary_centres[finite],
        q25[finite],
        q75[finite],
        color="#B53A3A",
        alpha=0.16,
        linewidth=0.0,
        label="interquartile range",
    )
    y_lo, y_hi = np.quantile(residual, [0.01, 0.99])
    padding = 0.08 * max(float(y_hi - y_lo), 1.0)
    ax.set_ylim(float(y_lo - padding), float(y_hi + padding))
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("True Compton depth (mm)")
    ax.set_ylabel(r"Depth residual $d_{rec}-d_{true}$ (mm)")
    ax.legend(fontsize=6.0, loc="upper left")
    ax.text(-0.12, 1.02, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout(pad=0.9, w_pad=1.7)
    figures = _save_figure(fig, out_dir / "depth_truth_projection_and_bias")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "n_selected_valid_truth_matched": int(len(true_depth)),
        "truth_depth_centres_mm": depth_centres,
        "truth_bin_width_mm": float(truth_bin_width_mm),
        "eventwise_depth_bias_mm": float(np.mean(residual)),
        "eventwise_depth_rmse_mm": float(np.sqrt(np.mean(residual**2))),
        "scatter_preview_points": int(len(chosen)),
        "source_data": [
            str(projection_csv.resolve()),
            str(event_csv.resolve()),
            str(summary_csv.resolve()),
        ],
        "figures": figures,
        "definitions": {
            "true_depth": "minus the GATE truth first-Compton z coordinate",
            "reconstructed_depth": "minus the closed-form reconstructed z coordinate",
            "depth_residual": "reconstructed depth minus true depth",
        },
    }
    with (out_dir / "depth_truth_projection_and_bias.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def plot_d2_threshold_linear(
    *,
    response_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    lower_thresholds_keV: Iterable[float] = (
        1.0,
        50.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
    ),
    upper_energy_keV: float = 600.0,
    current_lower_bound_keV: float = 400.0,
) -> dict[str, Any]:
    response_dir = Path(response_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    pairs = _load_npz(response_dir / "pairs_response.npz")
    reconstruction = _load_npz(response_dir / "reconstruction_response.npz")
    with (response_dir / "response_analysis.json").open(
        "r", encoding="utf-8"
    ) as handle:
        response_summary = json.load(handle)
    n_primaries = int(response_summary["n_primaries"])
    truth_signal = _truth_signal_mask(pairs, config)
    truth_matched = reconstruction["truth_matched"].astype(bool)
    thresholds = sorted(set(float(value) for value in lower_thresholds_keV))
    if float(current_lower_bound_keV) not in thresholds:
        thresholds.append(float(current_lower_bound_keV))
        thresholds.sort()

    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"
    rows: list[dict[str, Any]] = []
    for lower in thresholds:
        scenario = copy.deepcopy(observable_config)
        scenario["selection"]["d2_energy_window_keV"] = [
            float(lower),
            float(upper_energy_keV),
        ]
        selected, _, _ = build_selection(pairs, scenario)
        usable = selected & reconstruction["valid"].astype(bool)
        signal = usable & truth_signal
        scored = usable & truth_matched
        if np.any(scored):
            residual = (
                -reconstruction["z_mm"][scored]
                + reconstruction["truth_z_mm"][scored]
            )
            rmse = float(np.sqrt(np.mean(residual**2)))
            fwhm_estimate = estimate_fwhm_kde(residual, grid_size=512)
            fwhm = fwhm_estimate.get("fwhm")
            fwhm_status = fwhm_estimate["status"]
        else:
            rmse = None
            fwhm = None
            fwhm_status = "insufficient_events"
        rows.append(
            {
                "d2_lower_threshold_keV": float(lower),
                "d2_upper_threshold_keV": float(upper_energy_keV),
                "n_observable_valid": int(usable.sum()),
                "observable_valid_per_million_primaries": float(
                    usable.sum() / n_primaries * 1.0e6
                ),
                "n_truth_single_compton": int(signal.sum()),
                "truth_signal_per_million_primaries": float(
                    signal.sum() / n_primaries * 1.0e6
                ),
                "truth_signal_fraction_score_only": (
                    float(signal.sum() / usable.sum()) if np.any(usable) else None
                ),
                "eventwise_depth_rmse_mm": rmse,
                "eventwise_depth_residual_fwhm_mm": fwhm,
                "eventwise_depth_residual_fwhm_status": fwhm_status,
            }
        )
    baseline = next(
        row
        for row in rows
        if np.isclose(row["d2_lower_threshold_keV"], current_lower_bound_keV)
    )
    for row in rows:
        row["truth_signal_gain_vs_current"] = (
            row["truth_signal_per_million_primaries"]
            / baseline["truth_signal_per_million_primaries"]
            - 1.0
        )

    threshold_csv = out_dir / "d2_threshold_response.csv"
    with threshold_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_energy_config = copy.deepcopy(observable_config)
    all_energy_config["selection"]["d2_energy_window_keV"] = [
        0.001,
        float(upper_energy_keV),
    ]
    selected_all, _, _ = build_selection(pairs, all_energy_config)
    all_usable = selected_all & reconstruction["valid"].astype(bool)
    signal_usable = all_usable & truth_signal
    edges = np.arange(0.0, upper_energy_keV + 5.0, 5.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    observable_counts, _ = np.histogram(
        pairs["d2_energy_keV"][all_usable], bins=edges
    )
    signal_counts, _ = np.histogram(
        pairs["d2_energy_keV"][signal_usable], bins=edges
    )
    spectrum_csv = out_dir / "d2_spectrum_after_other_cuts_linear.csv"
    with spectrum_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "d2_energy_bin_center_keV",
                "observable_valid_events",
                "truth_single_compton_events_score_only",
            ]
        )
        writer.writerows(zip(centres, observable_counts, signal_counts))

    threshold = np.asarray([row["d2_lower_threshold_keV"] for row in rows])
    observable = np.asarray(
        [row["observable_valid_per_million_primaries"] for row in rows]
    )
    signal = np.asarray(
        [row["truth_signal_per_million_primaries"] for row in rows]
    )
    rmse = np.asarray(
        [
            np.nan
            if row["eventwise_depth_rmse_mm"] is None
            else row["eventwise_depth_rmse_mm"]
            for row in rows
        ]
    )
    fwhm = np.asarray(
        [
            np.nan
            if row["eventwise_depth_residual_fwhm_mm"] is None
            else row["eventwise_depth_residual_fwhm_mm"]
            for row in rows
        ]
    )
    purity = np.asarray(
        [
            np.nan
            if row["truth_signal_fraction_score_only"] is None
            else row["truth_signal_fraction_score_only"]
            for row in rows
        ]
    )

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183.0 * mm_to_inch, 66.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
    )
    blue = "#165A9F"
    red = "#B64342"
    grey = "#737373"
    orange = "#D67A1F"
    ax = axes[0]
    ax.step(
        centres,
        observable_counts,
        where="mid",
        color=grey,
        linewidth=1.0,
        label="observable valid",
    )
    ax.step(
        centres,
        signal_counts,
        where="mid",
        color=blue,
        linewidth=1.15,
        label="truth signal (score only)",
    )
    ax.axvline(
        current_lower_bound_keV,
        color=red,
        linestyle="--",
        linewidth=0.9,
        label="current lower bound",
    )
    ax.set_xlim(0.0, upper_energy_keV)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Measured D2 energy (keV)")
    ax.set_ylabel("Events / 5 keV (linear scale)")
    ax.legend(fontsize=5.7, loc="upper left")

    ax = axes[1]
    ax.plot(
        threshold,
        observable,
        "o-",
        color=grey,
        linewidth=1.0,
        markersize=3.0,
        label="observable valid",
    )
    ax.plot(
        threshold,
        signal,
        "o-",
        color=blue,
        linewidth=1.2,
        markersize=3.2,
        label="truth signal (score only)",
    )
    ax.axvline(current_lower_bound_keV, color=red, linestyle="--", linewidth=0.9)
    ax.set_xlabel("D2 lower threshold (keV)")
    ax.set_ylabel(r"Valid events / $10^6$ primaries")
    ax.legend(fontsize=5.7)
    lowest = rows[0]
    ax.text(
        0.04,
        0.06,
        f"Signal gain vs {current_lower_bound_keV:g} keV: "
        f"{lowest['truth_signal_gain_vs_current']:+.1%}",
        transform=ax.transAxes,
        fontsize=6.0,
    )

    ax = axes[2]
    ax.plot(
        threshold,
        fwhm,
        "o-",
        color=orange,
        linewidth=1.2,
        markersize=3.2,
        label="axial residual FWHM",
    )
    ax.plot(
        threshold,
        rmse,
        "s--",
        color=grey,
        linewidth=1.0,
        markersize=2.8,
        label="event-wise RMSE",
    )
    ax.axvline(current_lower_bound_keV, color=red, linestyle="--", linewidth=0.9)
    ax.set_xlabel("D2 lower threshold (keV)")
    ax.set_ylabel("Axial error (mm)")
    ax.legend(fontsize=5.7)
    ax.text(
        0.04,
        0.06,
        f"Truth fraction: {np.nanmin(purity):.1%}–{np.nanmax(purity):.1%}",
        transform=ax.transAxes,
        fontsize=6.0,
    )
    for label, ax in zip("abc", axes):
        ax.text(
            -0.17,
            1.03,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.tight_layout(pad=0.9, w_pad=1.7)
    figures = _save_figure(fig, out_dir / "d2_threshold_response_linear")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "current_window_keV": [
            float(current_lower_bound_keV),
            float(upper_energy_keV),
        ],
        "panel_a_y_scale": "linear",
        "rows": rows,
        "baseline": baseline,
        "source_data": [
            str(threshold_csv.resolve()),
            str(spectrum_csv.resolve()),
        ],
        "figures": figures,
        "interpretation_guardrail": (
            "Truth labels are used only to score signal fraction and depth error. "
            "The selection itself uses observable fields. Electronic noise, "
            "trigger efficiency, pile-up and dead time are not included."
        ),
    }
    with (out_dir / "d2_threshold_response_linear.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def build_hardware_update_assets(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    threshold_analysis_dir: str | Path | None = None,
    threshold_base_config_path: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    response_dir = out_dir / "response_analysis"
    figures_dir = out_dir / "figures"
    response = prepare_hardware_response_analysis(
        analysis_dir=analysis_dir,
        config_path=config_path,
        out_dir=response_dir,
    )
    spectra = plot_paired_spectra_before_after(
        analysis_dir=analysis_dir,
        response_dir=response_dir,
        config_path=config_path,
        out_dir=figures_dir / "slide07_spectra",
    )
    depth = plot_depth_truth_projection_and_bias(
        response_dir=response_dir,
        out_dir=figures_dir / "slide09_depth",
    )
    if threshold_analysis_dir is None:
        threshold_response_dir = response_dir
        threshold_config_path = Path(config_path)
        threshold_response = response
    else:
        if threshold_base_config_path is None:
            raise ValueError(
                "threshold_base_config_path is required with threshold_analysis_dir"
            )
        threshold_config_path = resolve_hardware_profile_for_variant(
            base_config_path=threshold_base_config_path,
            hardware_profile_path=config_path,
            output_path=(
                out_dir
                / "resolved_configs"
                / "thin_layer_depth30_hardware_20260731.json"
            ),
        )
        threshold_response_dir = out_dir / "threshold_response_analysis"
        threshold_response = prepare_hardware_response_analysis(
            analysis_dir=threshold_analysis_dir,
            config_path=threshold_config_path,
            out_dir=threshold_response_dir,
            random_seed=24073103,
        )
    threshold = plot_d2_threshold_linear(
        response_dir=threshold_response_dir,
        config_path=threshold_config_path,
        out_dir=figures_dir / "slide12_d2_threshold",
    )
    report = {
        "schema_version": "1.0",
        "analysis_dir": str(Path(analysis_dir).resolve()),
        "config": str(Path(config_path).resolve()),
        "response": response,
        "spectra": spectra,
        "depth": depth,
        "threshold_response": threshold_response,
        "d2_threshold": threshold,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hardware_update_assets.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
