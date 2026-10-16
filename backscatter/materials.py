from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config


def compare_standard_materials(
    *,
    metrics_paths: Iterable[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in metrics_paths]
    if len(paths) < 2:
        raise ValueError("material comparison requires at least two runs")
    metrics: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            metrics.append(json.load(handle))
        configs.append(load_config(metrics[-1]["config"]))
    if any(item.get("kind") != metrics[0].get("kind") for item in metrics):
        raise ValueError("material runs must use one standard-target kind")
    if any(
        float(item["target_depth_mm"])
        != float(metrics[0]["target_depth_mm"])
        for item in metrics
    ):
        raise ValueError("material runs must use one target depth")
    if any(
        config["phantom"]["size_mm"] != configs[0]["phantom"]["size_mm"]
        for config in configs
    ):
        raise ValueError("material runs must use identical target dimensions")

    rows: list[dict[str, Any]] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for path, item, config in zip(paths, metrics, configs):
        material = str(config["phantom"]["material"])
        estimate = item["fwhm_method"]["estimate"]
        ci = item["fwhm_method"]["bootstrap"].get("fwhm_95_ci")
        rows.append(
            {
                "material": material,
                "metrics_path": str(path.resolve()),
                "random_seed": int(config["simulation"]["random_seed"]),
                "n_primaries": int(item["n_primaries"]),
                "n_selected_valid": int(item["n_selected_valid"]),
                "efficiency_per_primary": float(
                    item["efficiency_per_primary"]
                ),
                "axial_fwhm_mm": (
                    float(estimate["fwhm"])
                    if estimate.get("fwhm") is not None
                    else None
                ),
                "axial_fwhm_ci_low_mm": (
                    float(ci[0]) if ci is not None else None
                ),
                "axial_fwhm_ci_high_mm": (
                    float(ci[1]) if ci is not None else None
                ),
                "peak_depth_bias_mm": float(item["peak_depth_bias_mm"]),
                "event_truth_rmse_mm": float(
                    item["event_truth_error"]["rmse_mm"]
                ),
            }
        )
        with Path(item["source_data"]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            curve_rows = list(csv.DictReader(handle))
        curves.append(
            (
                np.asarray(
                    [float(row["depth_mm"]) for row in curve_rows],
                    dtype=np.float64,
                ),
                np.asarray(
                    [float(row["density_normalized"]) for row in curve_rows],
                    dtype=np.float64,
                ),
            )
        )

    reference_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["material"].lower() == "concrete"
        ),
        0,
    )
    reference_efficiency = rows[reference_index]["efficiency_per_primary"]
    for row in rows:
        row["efficiency_relative_to_reference"] = float(
            row["efficiency_per_primary"] / reference_efficiency
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "material_comparison.csv"
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
    plt.rcParams["legend.frameon"] = False

    colors = ["#0F4D92", "#3D8D7A", "#B64342", "#767676"]
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.35, 0.9, 0.9]},
    )
    for row, curve, color in zip(rows, curves, colors):
        axes[0].plot(
            curve[0],
            curve[1],
            color=color,
            linewidth=1.15,
            label=row["material"],
        )
    axes[0].set_xlabel("Reconstructed depth (mm)")
    axes[0].set_ylabel("Normalized response")
    axes[0].legend(fontsize=6.2)

    labels = [row["material"] for row in rows]
    efficiency = [row["efficiency_per_primary"] for row in rows]
    axes[1].bar(labels, efficiency, color=colors[: len(rows)], width=0.65)
    axes[1].set_ylabel("Selected valid events / primary")
    axes[1].tick_params(axis="x", rotation=25)

    fwhm = [row["axial_fwhm_mm"] for row in rows]
    lower_error = [
        row["axial_fwhm_mm"] - row["axial_fwhm_ci_low_mm"]
        for row in rows
    ]
    upper_error = [
        row["axial_fwhm_ci_high_mm"] - row["axial_fwhm_mm"]
        for row in rows
    ]
    axes[2].errorbar(
        labels,
        fwhm,
        yerr=np.asarray([lower_error, upper_error]),
        fmt="o",
        color="#0F4D92",
        ecolor="#767676",
        capsize=3,
        markersize=4,
    )
    axes[2].set_ylabel("Axial FWHM (mm)")
    axes[2].tick_params(axis="x", rotation=25)
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
    output_base = out_dir / "material_comparison"
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
        "kind": metrics[0]["kind"],
        "target_depth_mm": float(metrics[0]["target_depth_mm"]),
        "target_size_mm": configs[0]["phantom"]["size_mm"],
        "reference_material": rows[reference_index]["material"],
        "rows": rows,
        "source_csv": str(csv_path.resolve()),
        "figures": figures,
        "interpretation_guardrail": (
            "This comparison uses isolated equal-size targets in vacuum. It "
            "measures material-dependent interaction yield and intrinsic "
            "response under identical geometry; it is not a bulk-material "
            "penetration or material-classification result."
        ),
    }
    with (out_dir / "material_comparison.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
