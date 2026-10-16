from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def d1_mass_kg(config: dict[str, Any], outer_radius_mm: float) -> float:
    d1 = config["detectors"]["d1"]
    inner_radius_mm = float(d1["radius_mm"][0])
    if outer_radius_mm <= inner_radius_mm:
        raise ValueError("D1 outer radius must exceed its inner radius")
    volume_mm3 = (
        np.pi
        * (outer_radius_mm**2 - inner_radius_mm**2)
        * float(d1["thickness_mm"])
    )
    return float(volume_mm3 * float(d1["density_g_cm3"]) / 1.0e6)


def d2_active_mass_kg(config: dict[str, Any]) -> float:
    d2 = config["detectors"]["d2"]
    count = int(np.prod(np.asarray(d2["array"], dtype=np.int64)))
    crystal_volume_mm3 = float(
        np.prod(np.asarray(d2["crystal_size_mm"], dtype=np.float64))
    )
    return float(
        count
        * crystal_volume_mm3
        * float(d2["density_g_cm3"])
        / 1.0e6
    )


def _outer_radius(setting: str) -> float:
    match = re.fullmatch(
        r"\s*[-+]?\d+(?:\.\d+)?-([-+]?\d+(?:\.\d+)?)\s+mm\s*",
        setting,
    )
    if not match:
        raise ValueError(f"cannot parse D1 radius setting: {setting!r}")
    return float(match.group(1))


def evaluate_detector_size(
    *,
    sweep_csv: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(sweep_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    radius_rows = [
        row
        for row in sweep_rows
        if row["category"] == "d1_radius_cumulative"
    ]
    if not radius_rows:
        raise ValueError("sweep table contains no d1_radius_cumulative rows")

    physical_outer = float(config["detectors"]["d1"]["radius_mm"][1])
    physical_mass = d1_mass_kg(config, physical_outer)
    baseline_rows = [row for row in sweep_rows if row["category"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError("sweep table must contain exactly one baseline row")
    baseline_efficiency = float(baseline_rows[0]["efficiency_per_primary"])

    rows: list[dict[str, Any]] = []
    for source in radius_rows:
        radius = _outer_radius(source["setting"])
        mass = d1_mass_kg(config, radius)
        efficiency = float(source["efficiency_per_primary"])
        quality = (
            float(source["quality_value_mm"])
            if source["quality_value_mm"]
            else None
        )
        rows.append(
            {
                "outer_radius_mm": radius,
                "d1_mass_kg": mass,
                "mass_fraction_vs_200mm": mass / physical_mass,
                "mass_reduction_pct_vs_200mm": 100.0
                * (1.0 - mass / physical_mass),
                "n_selected_valid": int(source["n_selected_valid"]),
                "efficiency_per_primary": efficiency,
                "sensitivity_per_million_primaries": efficiency * 1.0e6,
                "efficiency_retention": efficiency / baseline_efficiency,
                "quality_metric": source["quality_metric"],
                "quality_value_mm": quality,
                "posthoc_only": True,
            }
        )
    rows.sort(key=lambda row: row["outer_radius_mm"])

    def smallest_meeting(retention: float) -> dict[str, Any] | None:
        candidates = [
            row for row in rows if row["efficiency_retention"] >= retention
        ]
        return candidates[0] if candidates else None

    csv_path = out_dir / "d1_size_tradeoff.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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

    radius = np.asarray([row["outer_radius_mm"] for row in rows])
    mass = np.asarray([row["d1_mass_kg"] for row in rows])
    sensitivity = np.asarray(
        [row["sensitivity_per_million_primaries"] for row in rows]
    )
    quality = np.asarray([row["quality_value_mm"] for row in rows])
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"

    axes[0].plot(radius, mass, color=blue, marker="o", linewidth=1.3)
    axes[0].set_xlabel("D1 outer radius (mm)")
    axes[0].set_ylabel("GAGG mass (kg)")

    axes[1].plot(radius, sensitivity, color=blue, marker="o", linewidth=1.3)
    axes[1].axhline(
        0.95 * sensitivity[-1],
        color=neutral,
        linestyle="--",
        linewidth=0.8,
        label="95% of R = 200 mm",
    )
    axes[1].axhline(
        0.98 * sensitivity[-1],
        color=neutral,
        linestyle=":",
        linewidth=0.8,
        label="98% of R = 200 mm",
    )
    axes[1].set_xlabel("D1 outer radius (mm)")
    axes[1].set_ylabel(
        "Imaging sensitivity, $S(R)$\n"
        "(valid events / $10^6$ primaries)"
    )
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(fontsize=5.3)

    axes[2].plot(radius, quality, color=red, marker="s", linewidth=1.3)
    axes[2].set_xlabel("D1 outer radius (mm)")
    quality_label = {
        "truth_error_rmse_z_mm": "Truth z RMSE (mm)",
        "axial_psf_fwhm_mm": "Axial FWHM (mm)",
        "transverse_psf_fwhm_x_mm": "Transverse x FWHM (mm)",
    }.get(rows[0]["quality_metric"], rows[0]["quality_metric"].replace("_", " "))
    axes[2].set_ylabel(quality_label)
    axes[2].set_ylim(bottom=0.0)

    for label, ax in zip("abc", axes):
        ax.text(
            -0.18,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.set_xticks(radius)

    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "d1_size_tradeoff"
    figures = []
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
        "d1_physical_outer_radius_mm": physical_outer,
        "d1_physical_mass_kg": physical_mass,
        "d2_active_gagg_mass_kg": d2_active_mass_kg(config),
        "sensitivity_definition": (
            "S(R) = N_selected_valid(R) / N_primaries; the plotted values "
            "are scaled to valid events per 1,000,000 primaries."
        ),
        "relationship_type": (
            "Empirical sampled response curve with connected points; no "
            "unsupported analytic functional form is fitted."
        ),
        "posthoc_rows": rows,
        "smallest_radius_retaining_95pct_efficiency": smallest_meeting(0.95),
        "smallest_radius_retaining_98pct_efficiency": smallest_meeting(0.98),
        "source_data": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "These are post-hoc active-area cuts inside the 200 mm detector. "
            "Any size recommendation must be confirmed by rerunning GATE with "
            "the physical D1 radius changed."
        ),
    }
    with (out_dir / "d1_size_tradeoff.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
