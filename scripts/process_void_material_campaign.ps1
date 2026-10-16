param(
    [string]$RawRoot = "runs\vgate_fast_transfer\output",
    [int]$Bootstrap = 200,
    [ValidateSet("all", "voids", "materials")]
    [string]$Campaign = "all",
    [string]$VoidRunSuffix = "",
    [string]$MaterialRunSuffix = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
foreach ($suffixSpec in @(
    @{ Name = "VoidRunSuffix"; Value = $VoidRunSuffix },
    @{ Name = "MaterialRunSuffix"; Value = $MaterialRunSuffix }
)) {
    if (
        $suffixSpec.Value -and
        $suffixSpec.Value -notmatch "^_[A-Za-z0-9][A-Za-z0-9_.-]*$"
    ) {
        throw (
            "$($suffixSpec.Name) must be empty or a safe suffix " +
            "beginning with underscore."
        )
    }
}

function Assert-CompletedRun {
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
    if (-not (
        Select-String -LiteralPath $log -Quiet `
            -Pattern "Number of events processed\s*:\s*$ExpectedPrimaries"
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

Push-Location $projectRoot
try {
    if ($Campaign -in @("all", "voids")) {
        foreach ($diameter in @(10, 20)) {
            foreach ($depth in @(30, 60)) {
                $runId = (
                    "concrete_void_d${diameter}_depth${depth}_" +
                    "2m_seed24061875_local" +
                    $VoidRunSuffix
                )
                $config = (
                    "configs\generated\concrete_void_d${diameter}_" +
                    "depth${depth}_2m.json"
                )
                $raw = Join-Path $RawRoot $runId
                $analysis = Join-Path "runs" $runId
                Assert-CompletedRun `
                    -RunDirectory $raw `
                    -TruthFileName "hits_truth_concrete.root"
                Invoke-Backscatter -Arguments @(
                    "analyze",
                    "--root", (Join-Path $raw "hits.root"),
                    "--truth", (Join-Path $raw "hits_truth_concrete.root"),
                    "--config", $config,
                    "--out-dir", $analysis
                )
                Invoke-Backscatter -Arguments @(
                    "evaluate-void",
                    "--void-analysis-dir", $analysis,
                    "--reference-analysis-dir",
                    "runs\concrete_slab_2m_seed24061873_local",
                    "--config", $config,
                    "--out-dir", (Join-Path $analysis "void_metrics"),
                    "--roi-radius-scale", "1.5",
                    "--map-bin-width-mm", "2",
                    "--smoothing-sigma-bins", "1"
                )
            }
        }
    }

    if ($Campaign -in @("all", "materials")) {
        $materialMetricPaths = @(
            "runs\depth_psf_30mm_seed24061873_local\standard_metrics\metrics.json"
        )
        foreach ($material in @("water", "iron")) {
            $runId = (
                "${material}_thin30_2m_seed24061873_local" +
                $MaterialRunSuffix
            )
            $config = (
                "configs\generated\${material}_thin_layer_depth30_2m.json"
            )
            $raw = Join-Path $RawRoot $runId
            $analysis = Join-Path "runs" $runId
            Assert-CompletedRun `
                -RunDirectory $raw `
                -TruthFileName "hits_truth_target.root"
            Invoke-Backscatter -Arguments @(
                "analyze",
                "--root", (Join-Path $raw "hits.root"),
                "--truth", (Join-Path $raw "hits_truth_target.root"),
                "--config", $config,
                "--out-dir", $analysis
            )
            Invoke-Backscatter -Arguments @(
                "evaluate-standard",
                "--analysis-dir", $analysis,
                "--config", $config,
                "--bootstrap", [string]$Bootstrap
            )
            $materialMetricPaths += (
                "runs\$runId\standard_metrics\metrics.json"
            )
        }
        $materialSummaryArguments = @(
            "compare-materials",
            "--metrics"
        ) + $materialMetricPaths + @(
            "--out-dir", "runs\stage2_validation\materials_thin30"
        )
        Invoke-Backscatter -Arguments $materialSummaryArguments
    }
}
finally {
    Pop-Location
}
