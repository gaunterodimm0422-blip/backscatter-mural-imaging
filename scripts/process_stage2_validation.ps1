param(
    [string]$RawRoot = "runs\vgate_fast_transfer\output",
    [int]$StandardBootstrap = 200,
    [int]$ConcreteBootstrap = 100
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Assert-CompletedGateRun {
    param(
        [string]$RunDirectory,
        [string]$TruthFileName,
        [int]$ExpectedPrimaries = 2000000
    )

    foreach ($name in @(
        "hits.root",
        $TruthFileName,
        "gate.log",
        "run_parameters.txt",
        "macro_sha256.txt"
    )) {
        $path = Join-Path $RunDirectory $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Incomplete GATE run: missing $path"
        }
        if ((Get-Item -LiteralPath $path).Length -le 0) {
            throw "Incomplete GATE run: empty $path"
        }
    }
    $log = Join-Path $RunDirectory "gate.log"
    $completionPattern = (
        "Number of events processed\s*:\s*" +
        [regex]::Escape([string]$ExpectedPrimaries)
    )
    if (-not (
        Select-String -LiteralPath $log -Quiet -Pattern $completionPattern
    )) {
        throw "Expected completion count is absent from $log"
    }
    if (
        Select-String -LiteralPath $log -Quiet `
            -Pattern "ERROR|FATAL|Aborted|core dumped|G4Exception"
    ) {
        throw "Fatal marker found in $log"
    }
}

function Invoke-Backscatter {
    param([string[]]$Arguments)
    & python -m backscatter @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Backscatter command failed: $($Arguments -join ' ')"
    }
}

function Process-StandardRun {
    param(
        [string]$RunId,
        [string]$Config
    )
    $raw = Join-Path $RawRoot $RunId
    $analysis = Join-Path "runs" $RunId
    Assert-CompletedGateRun `
        -RunDirectory $raw `
        -TruthFileName "hits_truth_target.root"
    Invoke-Backscatter -Arguments @(
        "analyze",
        "--root", (Join-Path $raw "hits.root"),
        "--truth", (Join-Path $raw "hits_truth_target.root"),
        "--config", $Config,
        "--out-dir", $analysis
    )
    Invoke-Backscatter -Arguments @(
        "evaluate-standard",
        "--analysis-dir", $analysis,
        "--config", $Config,
        "--bootstrap", [string]$StandardBootstrap
    )
}

function Process-ConcreteRun {
    param(
        [string]$RunId,
        [string]$Config
    )
    $raw = Join-Path $RawRoot $RunId
    $analysis = Join-Path "runs" $RunId
    Assert-CompletedGateRun `
        -RunDirectory $raw `
        -TruthFileName "hits_truth_concrete.root"
    Invoke-Backscatter -Arguments @(
        "analyze",
        "--root", (Join-Path $raw "hits.root"),
        "--truth", (Join-Path $raw "hits_truth_concrete.root"),
        "--config", $Config,
        "--out-dir", $analysis
    )
    Invoke-Backscatter -Arguments @(
        "evaluate-concrete-depth",
        "--analysis-dir", $analysis,
        "--config", $Config,
        "--bin-width-mm", "10",
        "--minimum-events", "100",
        "--bootstrap", [string]$ConcreteBootstrap
    )
}

Push-Location $projectRoot
try {
    $standardRuns = @(
        @{
            RunId = "repeat_thin30_seed24061874_local"
            Config = "configs\generated\thin_layer_depth30_seed24061874.json"
        },
        @{
            RunId = "repeat_thin90_seed24061874_local"
            Config = "configs\generated\thin_layer_depth90_seed24061874.json"
        },
        @{
            RunId = "physical_d1r150_thin30_seed24061873_local"
            Config = "configs\generated\thin_layer_depth30_d1r150.json"
        },
        @{
            RunId = "physical_d1r175_thin30_seed24061873_local"
            Config = "configs\generated\thin_layer_depth30_d1r175.json"
        }
    )
    foreach ($run in $standardRuns) {
        Process-StandardRun -RunId $run.RunId -Config $run.Config
    }

    Process-ConcreteRun `
        -RunId "repeat_concrete_seed24061874_local" `
        -Config "configs\generated\concrete_slab_2m_seed24061874.json"
    Process-ConcreteRun `
        -RunId "physical_d1r150_concrete_seed24061873_local" `
        -Config "configs\generated\concrete_slab_2m_d1r150.json"

    Invoke-Backscatter -Arguments @(
        "compare-standard-repeats",
        "--metrics",
        "runs\depth_psf_30mm_seed24061873_local\standard_metrics\metrics.json",
        "runs\repeat_thin30_seed24061874_local\standard_metrics\metrics.json",
        "--out-dir", "runs\stage2_validation\repeat_thin30"
    )
    Invoke-Backscatter -Arguments @(
        "compare-standard-repeats",
        "--metrics",
        "runs\depth_psf_90mm_seed24061873_local\standard_metrics\metrics.json",
        "runs\repeat_thin90_seed24061874_local\standard_metrics\metrics.json",
        "--out-dir", "runs\stage2_validation\repeat_thin90"
    )
    Invoke-Backscatter -Arguments @(
        "compare-concrete-repeats",
        "--metrics",
        "runs\concrete_slab_2m_seed24061873_local\concrete_depth_profile\metrics.json",
        "runs\repeat_concrete_seed24061874_local\concrete_depth_profile\metrics.json",
        "--out-dir", "runs\stage2_validation\repeat_concrete"
    )
    Invoke-Backscatter -Arguments @(
        "validate-d1-sizes",
        "--full-metrics",
        "runs\depth_psf_30mm_seed24061873_local\standard_metrics\metrics.json",
        "--physical-metrics",
        "runs\physical_d1r150_thin30_seed24061873_local\standard_metrics\metrics.json",
        "runs\physical_d1r175_thin30_seed24061873_local\standard_metrics\metrics.json",
        "--posthoc-report",
        "runs\depth_psf_30mm_seed24061873_local\detector_size\d1_size_tradeoff.json",
        "--out-dir", "runs\stage2_validation\physical_d1"
    )
    Invoke-Backscatter -Arguments @(
        "validate-concrete-d1",
        "--full-analysis-dir",
        "runs\concrete_slab_2m_seed24061873_local",
        "--physical-analysis-dir",
        "runs\physical_d1r150_concrete_seed24061873_local",
        "--full-config", "configs\generated\concrete_slab_2m.json",
        "--physical-config",
        "configs\generated\concrete_slab_2m_d1r150.json",
        "--out-dir", "runs\stage2_validation\physical_d1_concrete",
        "--bin-width-mm", "10"
    )
}
finally {
    Pop-Location
}
