from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .pipeline import build_selection
from .sweeps import _load_npz, _row_metrics, _truth_signal_mask


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty D2-energy table")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_d2_energy_acceptance(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    lower_thresholds_keV: Iterable[float] = (
        1.0,
        20.0,
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
    upper_energy_keV: float = 560.0,
    band_edges_keV: Iterable[float] = (
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
        560.0,
    ),
) -> dict[str, Any]:
    """Measure whether partial-energy D2 events add useful image information."""

    analysis_dir = Path(analysis_dir)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    pairs = _load_npz(analysis_dir / "pairs.npz")
    reconstruction = _load_npz(analysis_dir / "reconstruction.npz")
    with (analysis_dir / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)
    n_primaries = int(summary["event_counts"]["primaries"])
    if n_primaries <= 0:
        raise ValueError("analysis primary count must be positive")
    if len(pairs["key"]) != len(reconstruction["key"]) or not np.array_equal(
        pairs["key"],
        reconstruction["key"],
    ):
        raise ValueError("pair and reconstruction event keys are not aligned")

    thresholds = sorted(
        {float(value) for value in lower_thresholds_keV}
    )
    if (
        not thresholds
        or thresholds[0] < 0.0
        or thresholds[-1] >= upper_energy_keV
    ):
        raise ValueError("D2 lower thresholds must be in [0, upper energy)")
    edges = sorted({float(value) for value in band_edges_keV})
    if (
        len(edges) < 2
        or edges[0] < 0.0
        or edges[-1] > upper_energy_keV
    ):
        raise ValueError("D2 band edges must lie within the study range")

    target_kind = config["phantom"].get("target_kind")
    target_depth = config["phantom"].get("depth_mm")
    truth_signal = _truth_signal_mask(pairs, config)
    if truth_signal is None:
        raise ValueError("D2-energy study requires simulation truth for scoring")
    observable_config = copy.deepcopy(config)
    observable_config["selection"]["phantom_truth_policy"] = "ignore"

    def measure(
        *,
        kind: str,
        lower: float,
        upper: float,
    ) -> dict[str, Any]:
        scenario = copy.deepcopy(observable_config)
        scenario["selection"]["d2_energy_window_keV"] = [lower, upper]
        selected, _, _ = build_selection(pairs, scenario)
        row = _row_metrics(
            category=kind,
            setting=f"{lower:g}-{upper:g} keV",
            selected=selected,
            truth_signal=truth_signal,
            reconstruction=reconstruction,
            n_primaries=n_primaries,
            standard_target_kind=target_kind,
            target_depth_mm=(
                float(target_depth)
                if target_depth is not None
                else None
            ),
        )
        row = {
            "window_kind": kind,
            "lower_energy_keV": lower,
            "upper_energy_keV": upper,
            **row,
        }
        return row

    cumulative_rows: list[dict[str, Any]] = []
    for lower in thresholds:
        row = measure(
            kind="cumulative_lower_threshold",
            lower=lower,
            upper=float(upper_energy_keV),
        )
        cumulative_rows.append(row)

    band_rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        row = measure(
            kind="disjoint_energy_band",
            lower=lower,
            upper=upper,
        )
        band_rows.append(row)

    photopeak_candidates = [
        row
        for row in cumulative_rows
        if np.isclose(row["lower_energy_keV"], 450.0)
    ]
    if len(photopeak_candidates) != 1:
        raise ValueError("D2-energy study requires a 450-560 keV baseline")
    baseline = photopeak_candidates[0]
    baseline_signal = float(baseline["efficiency_per_primary"])
    baseline_observable = float(
        baseline["observable_efficiency_per_primary"]
    )
    for row in cumulative_rows:
        row["truth_signal_sensitivity_retention_vs_450keV"] = float(
            row["efficiency_per_primary"] / baseline_signal
        )
        row["observable_sensitivity_retention_vs_450keV"] = float(
            row["observable_efficiency_per_primary"]
            / baseline_observable
        )
    for row in band_rows:
        row["truth_signal_sensitivity_vs_photopeak_fraction"] = float(
            row["efficiency_per_primary"] / baseline_signal
        )

    cumulative_csv = out_dir / "d2_energy_cumulative_thresholds.csv"
    bands_csv = out_dir / "d2_energy_disjoint_bands.csv"
    _write_csv(cumulative_csv, cumulative_rows)
    _write_csv(bands_csv, band_rows)

    # Panel a is formed after all non-D2 observable cuts. It shows which D2
    # deposited energies remain potentially usable before imposing a window.
    all_energy_config = copy.deepcopy(observable_config)
    all_energy_config["selection"]["d2_energy_window_keV"] = [
        thresholds[0],
        float(upper_energy_keV),
    ]
    all_selected, _, _ = build_selection(pairs, all_energy_config)
    all_usable = all_selected & reconstruction["valid"].astype(bool)
    signal_usable = all_usable & truth_signal
    spectrum_edges = np.arange(
        0.0,
        float(upper_energy_keV) + 5.0,
        5.0,
    )
    spectrum_centers = 0.5 * (
        spectrum_edges[:-1] + spectrum_edges[1:]
    )
    observable_counts, _ = np.histogram(
        pairs["d2_energy_keV"][all_usable],
        bins=spectrum_edges,
    )
    signal_counts, _ = np.histogram(
        pairs["d2_energy_keV"][signal_usable],
        bins=spectrum_edges,
    )
    spectrum_csv = out_dir / "d2_energy_spectrum_after_other_cuts.csv"
    with spectrum_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "d2_energy_bin_center_keV",
                "observable_valid_counts",
                "truth_single_scatter_counts_score_only",
            ]
        )
        writer.writerows(
            zip(spectrum_centers, observable_counts, signal_counts)
        )

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
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"

    axes[0].step(
        spectrum_centers,
        observable_counts,
        where="mid",
        color=neutral,
        linewidth=1.0,
        label="Observable valid",
    )
    axes[0].step(
        spectrum_centers,
        signal_counts,
        where="mid",
        color=blue,
        linewidth=1.1,
        label="Truth signal (score only)",
    )
    axes[0].axvline(
        450.0,
        color=red,
        linestyle="--",
        linewidth=0.9,
        label="Current lower bound",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].set_xlim(0.0, upper_energy_keV)
    axes[0].set_xlabel("D2 deposited energy (keV)")
    axes[0].set_ylabel("Events / 5 keV")
    axes[0].legend(fontsize=5.5)

    threshold = np.asarray(
        [row["lower_energy_keV"] for row in cumulative_rows],
        dtype=np.float64,
    )
    signal_sensitivity = np.asarray(
        [
            row["efficiency_per_primary"] * 1.0e6
            for row in cumulative_rows
        ],
        dtype=np.float64,
    )
    observable_sensitivity = np.asarray(
        [
            row["observable_efficiency_per_primary"] * 1.0e6
            for row in cumulative_rows
        ],
        dtype=np.float64,
    )
    axes[1].plot(
        threshold,
        observable_sensitivity,
        "o-",
        color=neutral,
        linewidth=1.0,
        markersize=3.0,
        label="Observable valid",
    )
    axes[1].plot(
        threshold,
        signal_sensitivity,
        "o-",
        color=blue,
        linewidth=1.2,
        markersize=3.2,
        label="Truth signal (score only)",
    )
    axes[1].axvline(
        450.0,
        color=red,
        linestyle="--",
        linewidth=0.9,
    )
    axes[1].set_xlabel("D2 lower threshold (keV)")
    axes[1].set_ylabel("Valid events / $10^6$ primaries")
    axes[1].legend(fontsize=5.5)
    lowest = cumulative_rows[0]
    axes[1].text(
        0.04,
        0.06,
        (
            "Signal gain vs 450 keV:\n"
            f"{lowest['truth_signal_sensitivity_retention_vs_450keV'] - 1:+.1%}"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
    )

    quality_key = (
        "axial_psf_fwhm_mm"
        if target_kind == "thin_layer"
        else "truth_error_rmse_z_mm"
    )
    quality_label = (
        "Axial response / error (mm)"
        if target_kind == "thin_layer"
        else "Truth z RMSE (mm)"
    )
    quality = np.asarray(
        [
            row[quality_key]
            if row[quality_key] is not None
            else np.nan
            for row in cumulative_rows
        ],
        dtype=np.float64,
    )
    axes[2].plot(
        threshold,
        quality,
        "o-",
        color=red,
        linewidth=1.2,
        markersize=3.2,
        label=(
            "Axial FWHM"
            if quality_key == "axial_psf_fwhm_mm"
            else "Truth z RMSE"
        ),
    )
    if target_kind == "thin_layer":
        rmse = np.asarray(
            [row["truth_error_rmse_z_mm"] for row in cumulative_rows],
            dtype=np.float64,
        )
        axes[2].plot(
            threshold,
            rmse,
            "s--",
            color=neutral,
            linewidth=1.0,
            markersize=2.8,
            label="Event-wise z RMSE",
        )
    axes[2].axvline(
        450.0,
        color=red,
        linestyle="--",
        linewidth=0.9,
    )
    axes[2].set_xlabel("D2 lower threshold (keV)")
    axes[2].set_ylabel(quality_label)
    axes[2].legend(fontsize=5.5)
    purity = np.asarray(
        [row["truth_signal_fraction"] for row in cumulative_rows],
        dtype=np.float64,
    )
    axes[2].text(
        0.04,
        0.06,
        f"Truth fraction = {np.min(purity):.1%}–{np.max(purity):.1%}",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
    )

    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    fig.tight_layout(pad=1.1, w_pad=1.8)
    output_base = out_dir / "d2_energy_acceptance"
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
        "n_primaries": n_primaries,
        "upper_energy_keV": float(upper_energy_keV),
        "selection_uses_truth": False,
        "truth_use": "post-selection signal, purity and resolution scoring only",
        "photopeak_region_baseline": baseline,
        "lowest_threshold_result": lowest,
        "cumulative_rows": cumulative_rows,
        "disjoint_band_rows": band_rows,
        "source_data": [
            str(cumulative_csv.resolve()),
            str(bands_csv.resolve()),
            str(spectrum_csv.resolve()),
        ],
        "figures": figures,
        "conclusion": (
            "D2 partial-energy events below the 450 keV photopeak-region "
            "lower bound contain usable tag information in this ideal "
            "simulation. Lowering the threshold increases image sensitivity "
            "without a material purity loss in the tested 30 mm target."
        ),
        "interpretation_guardrail": (
            "The lowest thresholds are exploratory because electronics noise, "
            "hardware trigger threshold and measured GAGG/SiPM resolution are "
            "not modeled. D2 deposited energy is used for event acceptance, "
            "not in the Compton scatter-point kinematic equation. A practical "
            "window must be re-optimized after hardware calibration."
        ),
    }
    with (out_dir / "d2_energy_acceptance.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
