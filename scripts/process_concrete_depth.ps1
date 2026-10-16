param(
    [string]$SharedGateRoot = "E:\vbgate\share\beisanshe_v1\gate",
    [int]$Bootstrap = 100
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runId = "concrete_slab_2m_seed24061873_local"
$gateDir = Join-Path $SharedGateRoot "output\$runId"
$root = Join-Path $gateDir "hits.root"
$truth = Join-Path $gateDir "hits_truth_concrete.root"
$log = Join-Path $gateDir "gate.log"
$config = "configs\generated\concrete_slab_2m.json"
$analysis = "runs\$runId"

foreach ($required in @($root, $truth, $log)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Incomplete exported run: missing $required"
    }
}
if (-not (Select-String -LiteralPath $log -Quiet -Pattern "Number of events processed\s*:\s*2000000")) {
    throw "GATE completion count is missing from $log"
}

Push-Location $projectRoot
try {
    & python -m backscatter analyze `
        --root $root `
        --truth $truth `
        --config $config `
        --out-dir $analysis
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed for $runId"
    }

    & python -m backscatter plot-spectra `
        --analysis-dir $analysis
    if ($LASTEXITCODE -ne 0) {
        throw "D1/D2 spectrum plotting failed for $runId"
    }

    & python -m backscatter plot-reconstruction `
        --analysis-dir $analysis
    if ($LASTEXITCODE -ne 0) {
        throw "Concrete reconstruction plotting failed for $runId"
    }

    & python -m backscatter evaluate-concrete-depth `
        --analysis-dir $analysis `
        --config $config `
        --bin-width-mm 10 `
        --minimum-events 100 `
        --bootstrap $Bootstrap
    if ($LASTEXITCODE -ne 0) {
        throw "Concrete depth-profile evaluation failed for $runId"
    }

    & python -m backscatter sweep `
        --analysis-dir $analysis `
        --config $config
    if ($LASTEXITCODE -ne 0) {
        throw "Concrete observable-cut/truth-score sweep failed for $runId"
    }

    & python -m backscatter detector-size `
        --sweep-csv (Join-Path $analysis "sweeps\sweep_metrics.csv") `
        --config $config `
        --out-dir (Join-Path $analysis "detector_size")
    if ($LASTEXITCODE -ne 0) {
        throw "Concrete D1 detector-size evaluation failed for $runId"
    }

    & python -m backscatter d2-energy-study `
        --analysis-dir $analysis `
        --config $config `
        --out-dir (Join-Path $analysis "d2_energy_study")
    if ($LASTEXITCODE -ne 0) {
        throw "Concrete D2 partial-energy study failed for $runId"
    }
}
finally {
    Pop-Location
}
