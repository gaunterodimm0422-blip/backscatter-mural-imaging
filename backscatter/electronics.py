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


def energy_sigma_keV(
    deposited_energy_keV: np.ndarray,
    *,
    fwhm_fraction_at_511: float,
) -> np.ndarray:
    """Statistical scintillator-resolution scenario.

    The absolute FWHM scales as sqrt(E), so the relative resolution scales as
    1/sqrt(E).  This is a sensitivity model, not a calibrated detector claim.
    """
    if fwhm_fraction_at_511 < 0.0:
        raise ValueError("energy-resolution fraction cannot be negative")
    energy = np.clip(
        np.asarray(deposited_energy_keV, dtype=np.float64),
        0.0,
        None,
    )
    fwhm = (
        float(fwhm_fraction_at_511)
        * np.sqrt(energy * 511.0)
    )
    return fwhm / FWHM_TO_SIGMA


def smear_energy_keV(
    deposited_energy_keV: np.ndarray,
    *,
    fwhm_fraction_at_511: float,
    standard_normal: np.ndarray,
) -> np.ndarray:
    energy = np.asarray(deposited_energy_keV, dtype=np.float64)
    standard_normal = np.asarray(standard_normal, dtype=np.float64)
    if energy.shape != standard_normal.shape:
        raise ValueError("energy and standard-normal arrays must have one shape")
    sigma = energy_sigma_keV(
        energy,
        fwhm_fraction_at_511=fwhm_fraction_at_511,
    )
    return np.clip(energy + sigma * standard_normal, 0.0, None)


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


def _peak_mode(values: np.ndarray, lo: float, hi: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= lo) & (values <= hi)]
    if len(values) == 0:
        return None
    counts, edges = np.histogram(values, bins=np.arange(lo, hi + 1.0, 1.0))
    index = int(np.argmax(counts))
    return float(0.5 * (edges[index] + edges[index + 1]))


def evaluate_electronics_sensitivity(
    *,
    analysis_dir: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    fwhm_percent_at_511: Iterable[float] = (0.0, 5.0, 10.0, 15.0),
    random_seed: int = 24062001,
) -> dict[str, Any]:
    """Re-run observable selection and geometry under energy-blur scenarios."""
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
    if n_primaries <= 0:
        raise ValueError("analysis primary count must be positive")

    percentages = [float(value) for value in fwhm_percent_at_511]
    if not percentages:
        raise ValueError("at least one electronics scenario is required")
    if any(value < 0.0 for value in percentages):
        raise ValueError("energy-resolution percentages cannot be negative")
    percentages = sorted(set(percentages))

    rng = np.random.default_rng(int(random_seed))
    normal_d1 = rng.standard_normal(len(pairs["key"]))
    normal_d2 = rng.standard_normal(len(pairs["key"]))
    source = _positions(pairs, "source")
    d1 = _positions(pairs, "d1")
    d2 = _positions(pairs, "d2")
    path_lo, path_hi = map(
        float,
        config["reconstruction"]["path_length_mm"],
    )
    truth_signal = _truth_signal_mask(pairs, config)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, np.ndarray | float]] = []

    for percentage in percentages:
        fraction = percentage / 100.0
        scenario_pairs = dict(pairs)
        if fraction == 0.0:
            scenario_pairs["d1_energy_keV"] = np.asarray(
                pairs["d1_energy_keV"],
                dtype=np.float64,
            )
            scenario_pairs["d2_energy_keV"] = np.asarray(
                pairs["d2_energy_keV"],
                dtype=np.float64,
            )
        else:
            scenario_pairs["d1_energy_keV"] = smear_energy_keV(
                pairs["d1_energy_keV"],
                fwhm_fraction_at_511=fraction,
                standard_normal=normal_d1,
            )
            scenario_pairs["d2_energy_keV"] = smear_energy_keV(
                pairs["d2_energy_keV"],
                fwhm_fraction_at_511=fraction,
                standard_normal=normal_d2,
            )

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
        truth_usable = usable & reconstruction["truth_matched"].astype(bool)

        if np.any(truth_usable):
            depth_error = (
                -solution.points_mm[truth_usable, 2]
                + reconstruction["truth_z_mm"][truth_usable]
            )
            depth_rmse = float(np.sqrt(np.mean(depth_error**2)))
            error_fwhm = estimate_fwhm_kde(depth_error)
        else:
            depth_rmse = None
            error_fwhm = {"status": "insufficient_events", "fwhm": None}

        target_kind = config["phantom"].get("target_kind")
        if target_kind in {"thin_layer", "rod", "point"}:
            response_values = -solution.points_mm[signal_usable, 2]
            response_label = "target_axial_response_fwhm_mm"
        else:
            response_values = (
                -solution.points_mm[truth_usable, 2]
                + reconstruction["truth_z_mm"][truth_usable]
            )
            response_label = "eventwise_axial_error_fwhm_mm"
        response_fwhm = estimate_fwhm_kde(response_values)
        if response_fwhm.get("status") == "ok":
            curves.append(
                {
                    "fwhm_percent_at_511": percentage,
                    "grid": np.asarray(response_fwhm["grid"]),
                    "density": np.asarray(response_fwhm["density"]),
                }
            )

        n_selected = int(selected.sum())
        n_valid = int(usable.sum())
        n_signal = int(signal_usable.sum())
        row = {
            "fwhm_percent_at_511": percentage,
            "fwhm_keV_at_511": 5.11 * percentage,
            "n_observable_selected": n_selected,
            "n_observable_selected_valid": n_valid,
            "selected_valid_per_primary": float(n_valid / n_primaries),
            "n_truth_single_scatter_signal": n_signal,
            "truth_signal_fraction_of_selected_valid": (
                float(n_signal / n_valid) if n_valid else None
            ),
            "n_actor_truth_matched_selected_valid": int(truth_usable.sum()),
            "eventwise_depth_rmse_mm": depth_rmse,
            "eventwise_depth_error_fwhm_mm": error_fwhm.get("fwhm"),
            response_label: response_fwhm.get("fwhm"),
            "response_fwhm_status": response_fwhm.get("status"),
            "d2_pair_energy_mode_keV_450_to_560": _peak_mode(
                scenario_pairs["d2_energy_keV"],
                450.0,
                560.0,
            ),
        }
        rows.append(row)

    csv_path = out_dir / "electronics_sensitivity.csv"
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

    x = np.asarray(
        [row["fwhm_percent_at_511"] for row in rows],
        dtype=np.float64,
    )
    efficiency = np.asarray(
        [row["selected_valid_per_primary"] for row in rows],
        dtype=np.float64,
    )
    purity = np.asarray(
        [
            row["truth_signal_fraction_of_selected_valid"]
            if row["truth_signal_fraction_of_selected_valid"] is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    fwhm_key = (
        "target_axial_response_fwhm_mm"
        if any("target_axial_response_fwhm_mm" in row for row in rows)
        else "eventwise_axial_error_fwhm_mm"
    )
    axial_fwhm = np.asarray(
        [
            row.get(fwhm_key)
            if row.get(fwhm_key) is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    depth_rmse = np.asarray(
        [
            row["eventwise_depth_rmse_mm"]
            if row["eventwise_depth_rmse_mm"] is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"

    axes[0].plot(x, efficiency, "o-", color=blue, linewidth=1.2, markersize=3)
    axes[0].set_xlabel("Assumed energy FWHM at 511 keV (%)")
    axes[0].set_ylabel("Selected valid events / primary")
    if efficiency[0] > 0.0:
        axes[0].text(
            0.97,
            0.05,
            f"15% / raw = {efficiency[-1] / efficiency[0]:.2f}",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
        )

    axes[1].plot(x, purity, "o-", color=red, linewidth=1.2, markersize=3)
    axes[1].set_xlabel("Assumed energy FWHM at 511 keV (%)")
    axes[1].set_ylabel("Truth signal fraction (score only)")
    axes[1].set_ylim(0.0, min(1.0, max(0.1, np.nanmax(purity) * 1.15)))

    axes[2].plot(
        x,
        axial_fwhm,
        "o-",
        color=blue,
        linewidth=1.2,
        markersize=3,
        label="Axial FWHM",
    )
    axes[2].plot(
        x,
        depth_rmse,
        "s--",
        color=neutral,
        linewidth=1.0,
        markersize=2.8,
        label="Event-wise RMSE",
    )
    axes[2].set_xlabel("Assumed energy FWHM at 511 keV (%)")
    axes[2].set_ylabel("Axial error / response (mm)")
    axes[2].legend(fontsize=6.2)

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
    output_base = out_dir / "electronics_sensitivity"
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

    raw_efficiency = rows[0]["selected_valid_per_primary"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis_dir": str(analysis_dir.resolve()),
        "config": str(config_path.resolve()),
        "random_seed": int(random_seed),
        "common_random_numbers_across_scenarios": True,
        "resolution_model": {
            "formula": (
                "FWHM(E) = r_511 * sqrt(E * 511 keV); "
                "Gaussian broadening applied independently to event-summed "
                "D1 and D2 energies"
            ),
            "calibration_status": "hypothetical_sensitivity_scenarios",
        },
        "selection_uses_monte_carlo_truth": False,
        "truth_use": "post-selection performance scoring only",
        "rows": rows,
        "relative_efficiency_to_raw": [
            (
                float(row["selected_valid_per_primary"] / raw_efficiency)
                if raw_efficiency
                else None
            )
            for row in rows
        ],
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "These curves show sensitivity to assumed electronics resolution. "
            "They are not specifications for the eventual GAGG/SiPM channel. "
            "A measured calibration curve must replace the scenarios before a "
            "hardware-performance claim."
        ),
    }
    with (out_dir / "electronics_sensitivity.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
