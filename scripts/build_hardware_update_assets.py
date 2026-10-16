from pathlib import Path

from backscatter.hardware_figures import build_hardware_update_assets


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = build_hardware_update_assets(
        analysis_dir=(
            root / "runs" / "concrete_slab_2m_seed24061873_local"
        ),
        config_path=root / "configs" / "hardware_20260731.json",
        out_dir=root / "outputs" / "hardware_update_20260731",
        threshold_analysis_dir=(
            root / "runs" / "depth_psf_30mm_seed24061873_local"
        ),
        threshold_base_config_path=(
            root / "configs" / "generated" / "thin_layer_depth30.json"
        ),
    )
    print(result["spectra"]["figures"][-1])
    print(result["depth"]["figures"][-1])
    print(result["d2_threshold"]["figures"][-1])
