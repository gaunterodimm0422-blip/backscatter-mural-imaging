from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .electronics import smear_energy_keV
from .kinematics import compton_scattered_energy_keV


def _configure_matplotlib() -> Any:
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
    return plt


def _histogram_fwhm(
    x: np.ndarray,
    counts: np.ndarray,
    lo: float,
    hi: float,
) -> dict[str, float]:
    """Estimate a histogram peak FWHM using linear crossing interpolation."""

    use = (x >= lo) & (x <= hi)
    local_x = np.asarray(x[use], dtype=np.float64)
    local_counts = np.asarray(counts[use], dtype=np.float64)
    if len(local_x) < 3 or len(local_x) != len(local_counts):
        raise ValueError("histogram FWHM range contains insufficient data")
    peak_index = int(np.argmax(local_counts))
    peak_x = float(local_x[peak_index])
    peak_count = float(local_counts[peak_index])
    if peak_count <= 0.0:
        raise ValueError("histogram FWHM peak must contain positive counts")
    half = 0.5 * peak_count
    left_below = np.flatnonzero(local_counts[:peak_index] < half)
    right_below = np.flatnonzero(local_counts[peak_index + 1 :] < half)
    if len(left_below) == 0 or len(right_below) == 0:
        raise ValueError("histogram peak FWHM is not bracketed")
    left0 = int(left_below[-1])
    left1 = left0 + 1
    right1 = peak_index + 1 + int(right_below[0])
    right0 = right1 - 1

    def crossing(index0: int, index1: int) -> float:
        x0 = float(local_x[index0])
        x1 = float(local_x[index1])
        y0 = float(local_counts[index0])
        y1 = float(local_counts[index1])
        if y1 == y0:
            return 0.5 * (x0 + x1)
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    left = crossing(left0, left1)
    right = crossing(right0, right1)
    return {
        "mode_keV": peak_x,
        "half_max_count": half,
        "left_half_max_keV": left,
        "right_half_max_keV": right,
        "fwhm_keV": float(right - left),
        "resolution_percent": float(100.0 * (right - left) / peak_x),
    }


def plot_energy_spectra(
    *,
    analysis_dir: str | Path,
    out_dir: str | Path | None = None,
    d2_fwhm_percent_at_511: float = 10.0,
    random_seed: int = 24062001,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    validation = summary["energy_validation"]
    d1 = validation["d1_spectrum_all_events"]
    d2 = validation["d2_spectrum_all_events"]
    centers = np.asarray(d1["bin_centers"], dtype=np.float64)
    d1_counts = np.asarray(d1["counts"], dtype=np.int64)
    d2_centers = np.asarray(d2["bin_centers"], dtype=np.float64)
    d2_counts = np.asarray(d2["counts"], dtype=np.int64)
    if not np.array_equal(centers, d2_centers):
        raise ValueError("D1 and D2 spectra use different energy grids")
    if d2_fwhm_percent_at_511 <= 0.0:
        raise ValueError("D2 energy-resolution percentage must be positive")
    with np.load(
        analysis_dir / "events.npz",
        allow_pickle=False,
    ) as events:
        d2_event_energy = np.asarray(
            events["d2_energy_keV"],
            dtype=np.float64,
        )
    d2_total_events = int(len(d2_event_energy))
    if d2_total_events <= 0:
        raise ValueError("analysis contains no D2 events")
    photopeak_lo_keV = 510.0
    photopeak_hi_keV = 512.0
    photopeak_count = int(
        np.count_nonzero(
            (d2_event_energy >= photopeak_lo_keV)
            & (d2_event_energy < photopeak_hi_keV)
        )
    )
    photopeak_fraction = photopeak_count / d2_total_events
    selection_window_count = int(
        np.count_nonzero(
            (d2_event_energy >= 450.0) & (d2_event_energy < 560.0)
        )
    )
    selection_window_fraction = selection_window_count / d2_total_events
    rng = np.random.default_rng(int(random_seed))
    broadened_d2_energy = smear_energy_keV(
        d2_event_energy,
        fwhm_fraction_at_511=d2_fwhm_percent_at_511 / 100.0,
        standard_normal=rng.standard_normal(len(d2_event_energy)),
    )
    bin_width = float(np.median(np.diff(centers)))
    spectrum_edges = np.concatenate(
        [
            [centers[0] - 0.5 * bin_width],
            centers + 0.5 * bin_width,
        ]
    )
    broadened_d2_counts, _ = np.histogram(
        broadened_d2_energy,
        bins=spectrum_edges,
    )

    broadened_peak = _histogram_fwhm(
        centers,
        broadened_d2_counts,
        420.0,
        600.0,
    )

    csv_path = out_dir / "energy_spectra_source.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_bin_center_keV",
                "d1_raw_counts",
                "d2_raw_counts",
                (
                    "d2_broadened_counts_"
                    f"{d2_fwhm_percent_at_511:g}pct_fwhm_at_511"
                ),
            ]
        )
        writer.writerows(
            zip(
                centers,
                d1_counts,
                d2_counts,
                broadened_d2_counts,
            )
        )

    scatter_angle_deg = np.linspace(0.0, 180.0, 361)
    scattered_energy_keV = compton_scattered_energy_keV(scatter_angle_deg)
    angle_csv_path = out_dir / "energy_angle_relation.csv"
    with angle_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "compton_scatter_angle_deg",
                "scattered_photon_energy_keV",
            ]
        )
        writer.writerows(zip(scatter_angle_deg, scattered_energy_keV))

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        4,
        # Keep the four-panel figure close to the 16:9 slide image frame so
        # labels remain legible after insertion into the presentation.
        figsize=(220.0 * mm_to_inch, 74.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.92, 1.0]},
    )
    blue = "#0F4D92"
    red = "#B64342"
    neutral = "#767676"
    band = "#D8D8D8"
    back_lo, back_hi = validation[
        "single_backscatter_band_keV_for_90_to_180_deg"
    ]

    ax = axes[0]
    ax.step(centers, d1_counts, where="mid", color=blue, linewidth=1.0)
    ax.axvspan(
        back_lo,
        back_hi,
        color=band,
        alpha=0.8,
    )
    ax.set_xlim(0.0, 350.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("D1 event energy (keV)")
    ax.set_ylabel("Events / 2 keV")
    ax.text(
        0.58,
        0.07,
        "90–180° single\nbackscatter band",
        transform=ax.transAxes,
        fontsize=6,
        ha="left",
        va="bottom",
    )

    ax = axes[1]
    ax.step(centers, d2_counts, where="mid", color=blue, linewidth=1.0)
    ax.axvline(511.0, color=red, linestyle="--", linewidth=0.9)
    ax.axvspan(450.0, 560.0, color=band, alpha=0.6)
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("D2 event energy (keV)")
    ax.set_ylabel("Events / 2 keV")
    ax.text(
        0.03,
        0.94,
        (
            "Raw GATE spectrum (linear axis)\n"
            f"510–512 keV peak bin: {photopeak_count:,} / "
            f"{d2_total_events:,} = {photopeak_fraction:.1%}\n"
            f"450–560 keV window: {selection_window_fraction:.1%}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
    )

    ax = axes[2]
    zoom = (centers >= 420.0) & (centers <= 600.0)
    ax.step(
        centers[zoom],
        broadened_d2_counts[zoom],
        where="mid",
        color=blue,
        linewidth=1.2,
    )
    peak = validation["d2_photopeak_mode_keV_450_to_530"]
    ax.axvline(511.0, color=neutral, linestyle="--", linewidth=0.8)
    ax.hlines(
        broadened_peak["half_max_count"],
        broadened_peak["left_half_max_keV"],
        broadened_peak["right_half_max_keV"],
        color=red,
        linewidth=1.0,
    )
    for boundary in (
        broadened_peak["left_half_max_keV"],
        broadened_peak["right_half_max_keV"],
    ):
        ax.vlines(
            boundary,
            0.0,
            broadened_peak["half_max_count"],
            color=red,
            linestyle=":",
            linewidth=0.8,
        )
    ax.text(
        0.04,
        0.95,
        (
            f"Hypothetical $R_{{511}}$ = "
            f"{d2_fwhm_percent_at_511:.1f}%\n"
            f"FWHM = {broadened_peak['fwhm_keV']:.1f} keV\n"
            f"Measured $R$ = "
            f"{broadened_peak['resolution_percent']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 1.0,
            "pad": 1.5,
        },
    )
    ax.set_xlim(420.0, 600.0)
    ax.set_ylim(
        0.0,
        1.38 * float(np.max(broadened_d2_counts[zoom])),
    )
    ax.set_xlabel("Broadened D2 energy (keV)")
    ax.set_ylabel("Events / 2 keV")

    ax = axes[3]
    ax.plot(
        scatter_angle_deg,
        scattered_energy_keV,
        color=blue,
        linewidth=1.2,
    )
    ax.axvspan(90.0, 180.0, color=band, alpha=0.65)
    ax.scatter(
        [0.0, 90.0, 180.0],
        [511.0, 255.5, 511.0 / 3.0],
        color=red,
        s=10,
        zorder=3,
    )
    ax.text(4.0, 490.0, "0°: 511.0 keV", fontsize=5.5, va="top")
    ax.text(94.0, 270.0, "90°: 255.5 keV", fontsize=5.5, va="bottom")
    ax.text(
        176.0,
        166.0,
        "180°: 170.3 keV",
        fontsize=5.5,
        ha="right",
        va="top",
    )
    ax.text(
        0.36,
        0.95,
        r"$E'=511/[2-\cos(\theta)]$ keV",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 1.0,
        },
    )
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(150.0, 530.0)
    ax.set_xticks([0.0, 45.0, 90.0, 135.0, 180.0])
    ax.set_xlabel(r"Compton scatter angle $\theta$ (deg)")
    ax.set_ylabel("Scattered-photon energy (keV)")

    for label, ax in zip("abcd", axes):
        ax.text(
            -0.17,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )

    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "energy_spectra"
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
        "analysis_dir": str(analysis_dir.resolve()),
        "source_data": str(csv_path.resolve()),
        "angle_relation_source_data": str(angle_csv_path.resolve()),
        "figures": figures,
        "d2_photopeak_mode_keV": peak,
        "d2_photopeak_occupancy": {
            "definition": "raw D2 event energy in [510, 512) keV",
            "count": photopeak_count,
            "all_d2_events": d2_total_events,
            "fraction_of_all_d2_events": float(photopeak_fraction),
            "selection_window_450_to_560_keV_count": selection_window_count,
            "selection_window_fraction_of_all_d2_events": float(
                selection_window_fraction
            ),
        },
        "d2_broadening": {
            "model": (
                "Gaussian FWHM(E) = r_511 * sqrt(E * 511 keV)"
            ),
            "assumed_fwhm_percent_at_511": float(
                d2_fwhm_percent_at_511
            ),
            "assumed_fwhm_keV_at_511": float(
                5.11 * d2_fwhm_percent_at_511
            ),
            "random_seed": int(random_seed),
            **broadened_peak,
            "resolution_definition": (
                "R = FWHM / photopeak mode energy * 100%"
            ),
            "calibration_status": "hypothetical_not_measured",
        },
        "d1_single_backscatter_band_keV": [back_lo, back_hi],
        "interpretation": (
            "Panels a-b are raw event-summed GATE deposited-energy spectra "
            "with linear count axes. Panel b states both the raw 510-512 keV "
            "photopeak-bin occupancy and the 450-560 keV window occupancy. "
            "Panel c applies a stated "
            "hypothetical Gaussian electronics broadening to D2 and marks "
            "the histogram FWHM; it is not a measured hardware calibration. "
            "Panel d is the theoretical 511 keV single-Compton relationship "
            "between scattering angle and scattered-photon energy."
        ),
    }
    with (out_dir / "energy_spectra.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def plot_reconstruction_distribution(
    *,
    analysis_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(
        analysis_dir / "reconstruction.npz",
        allow_pickle=False,
    ) as data:
        reconstruction = {name: data[name] for name in data.files}
    with (analysis_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    usable = (
        reconstruction["selected"].astype(bool)
        & reconstruction["valid"].astype(bool)
    )
    x = np.asarray(reconstruction["x_mm"][usable], dtype=np.float64)
    y = np.asarray(reconstruction["y_mm"][usable], dtype=np.float64)
    depth = -np.asarray(
        reconstruction["z_mm"][usable],
        dtype=np.float64,
    )
    if len(depth) == 0:
        raise ValueError("analysis contains no selected valid reconstruction")

    phantom_z_lo, phantom_z_hi = map(
        float,
        summary["canonical_geometry"]["phantom_z_interval_mm"],
    )
    phantom_depth = (-phantom_z_hi, -phantom_z_lo)
    inside = (depth >= phantom_depth[0]) & (depth <= phantom_depth[1])

    csv_path = out_dir / "reconstruction_points_source.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "y_mm", "depth_mm", "inside_phantom_depth"])
        writer.writerows(zip(x, y, depth, inside.astype(int)))

    plt = _configure_matplotlib()
    mm_to_inch = 1.0 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180.0 * mm_to_inch, 62.0 * mm_to_inch),
        gridspec_kw={"width_ratios": [1.05, 1.15, 0.9]},
    )
    blue = "#0F4D92"
    neutral = "#767676"
    band = "#D8D8D8"

    ax = axes[0]
    scatter = ax.scatter(
        x,
        y,
        c=depth,
        cmap="viridis",
        s=7,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("Reconstructed x (mm)")
    ax.set_ylabel("Reconstructed y (mm)")
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("Depth (mm)")

    ax = axes[1]
    ax.axhspan(
        phantom_depth[0],
        phantom_depth[1],
        color=band,
        alpha=0.65,
    )
    ax.scatter(
        x,
        depth,
        s=7,
        color=blue,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(0.0, color=neutral, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Reconstructed x (mm)")
    ax.set_ylabel("Reconstructed depth (mm)")
    ax.invert_yaxis()

    ax = axes[2]
    edges = np.arange(
        np.floor(np.quantile(depth, 0.005) / 2.0) * 2.0,
        np.ceil(np.quantile(depth, 0.995) / 2.0) * 2.0 + 2.0,
        2.0,
    )
    ax.hist(
        depth,
        bins=edges,
        color="#B4CFE8",
        edgecolor=blue,
        linewidth=0.45,
    )
    ax.axvspan(
        phantom_depth[0],
        phantom_depth[1],
        color=band,
        alpha=0.45,
    )
    ax.set_xlabel("Reconstructed depth (mm)")
    ax.set_ylabel("Events / 2 mm")
    ax.text(
        0.98,
        0.96,
        f"n = {len(depth):,}\n"
        f"Inside slab = {inside.mean():.1%}\n"
        f"Median depth = {np.median(depth):.1f} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
    )

    for label, ax in zip("abc", axes):
        ax.text(
            -0.17,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )

    fig.tight_layout(pad=1.2, w_pad=2.0)
    output_base = out_dir / "reconstruction_distribution"
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
        "analysis_dir": str(analysis_dir.resolve()),
        "n_selected_valid": int(len(depth)),
        "fraction_inside_phantom_depth": float(inside.mean()),
        "median_depth_mm": float(np.median(depth)),
        "source_data": str(csv_path.resolve()),
        "figures": figures,
        "guardrail": (
            "This homogeneous-slab reconstruction is an imaging-chain "
            "demonstration and depth distribution, not a spatial-resolution "
            "measurement."
        ),
    }
    with (out_dir / "reconstruction_distribution.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
