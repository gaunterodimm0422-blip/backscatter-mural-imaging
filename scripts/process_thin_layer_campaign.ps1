param(
    [string]$SharedGateRoot = "E:\vbgate\share\beisanshe_v1\gate",
    [int]$Bootstrap = 200
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$depths = @(10, 30, 50, 70, 90)
$metricPaths = @()

Push-Location $projectRoot
try {
    foreach ($depth in $depths) {
        $runId = "depth_psf_${depth}mm_seed24061873_local"
        $gateDir = Join-Path $SharedGateRoot "output\$runId"
        $root = Join-Path $gateDir "hits.root"
        $truth = Join-Path $gateDir "hits_truth_target.root"
        $log = Join-Path $gateDir "gate.log"
        $config = "configs\generated\thin_layer_depth${depth}.json"
        $analysis = "runs\$runId"

        foreach ($required in @($root, $truth, $log)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Incomplete exported run: missing $required"
            }
        }
        if (-not (Select-String -LiteralPath $log -Quiet -Pattern "Number of events processed\s*:\s*2000000")) {
            throw "GATE completion count is missing from $log"
        }

        & python -m backscatter analyze `
            --root $root `
            --truth $truth `
            --config $config `
            --out-dir $analysis
        if ($LASTEXITCODE -ne 0) {
            throw "Analysis failed for $runId"
        }

        & python -m backscatter evaluate-standard `
            --analysis-dir $analysis `
            --config $config `
            --bootstrap $Bootstrap
        if ($LASTEXITCODE -ne 0) {
            throw "Standard-target evaluation failed for $runId"
        }
        $metricPaths += Join-Path $analysis "standard_metrics\metrics.json"

        if ($depth -eq 30) {
            & python -m backscatter sweep `
                --analysis-dir $analysis `
                --config $config
            if ($LASTEXITCODE -ne 0) {
                throw "Filter/region sweep failed for $runId"
            }

            & python -m backscatter detector-size `
                --sweep-csv (Join-Path $analysis "sweeps\sweep_metrics.csv") `
                --config $config `
                --out-dir (Join-Path $analysis "detector_size")
            if ($LASTEXITCODE -ne 0) {
                throw "Detector-size evaluation failed for $runId"
            }

            & python -m backscatter d2-energy-study `
                --analysis-dir $analysis `
                --config $config `
                --out-dir (Join-Path $analysis "d2_energy_study")
            if ($LASTEXITCODE -ne 0) {
                throw "D2 partial-energy study failed for $runId"
            }
        }
    }

    & python -m backscatter depth-series `
        --metrics $metricPaths `
        --out-dir "runs\intrinsic_axial_series_seed24061873"
    if ($LASTEXITCODE -ne 0) {
        throw "Thin-layer axial series summary failed"
    }
}
finally {
    Pop-Location
}
