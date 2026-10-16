param(
    [string]$RawRoot = "runs\vgate_fast_transfer\output",
    [int]$Bootstrap = 200,
    [ValidateSet(
        "all",
        "rods",
        "double_rods",
        "double_rods_a",
        "double_rods_b",
        "edge"
    )]
    [string]$Campaign = "all",
    [string]$RodRunSuffix = "",
    [string]$DoubleRodRunSuffix = "",
    [string]$EdgeRunSuffix = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
foreach ($suffixSpec in @(
    @{ Name = "RodRunSuffix"; Value = $RodRunSuffix },
    @{ Name = "DoubleRodRunSuffix"; Value = $DoubleRodRunSuffix },
    @{ Name = "EdgeRunSuffix"; Value = $EdgeRunSuffix }
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
        [int]$ExpectedPrimaries
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

function Process-Target {
    param(
        [string]$RunId,
        [string]$Config,
        [int]$Primaries,
        [switch]$RunSweep
    )
    $raw = Join-Path $RawRoot $RunId
    $analysis = Join-Path "runs" $RunId
    Assert-CompletedRun `
        -RunDirectory $raw `
        -TruthFileName "hits_truth_target.root" `
        -ExpectedPrimaries $Primaries
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
        "--bootstrap", [string]$Bootstrap
    )
    if ($RunSweep) {
        Invoke-Backscatter -Arguments @(
            "sweep",
            "--analysis-dir", $analysis,
            "--config", $Config
        )
    }
}

Push-Location $projectRoot
try {
    if ($Campaign -in @("all", "rods")) {
        $rodMetricPaths = @()
        foreach ($width in @(1, 2, 5)) {
            $runId = (
                "rod_width${width}_depth30_5m_seed24061873_local" +
                $RodRunSuffix
            )
            Process-Target `
                -RunId $runId `
                -Config "configs\generated\rod_width${width}_depth30_5m.json" `
                -Primaries 5000000 `
                -RunSweep:($width -eq 2)
            $rodMetricPaths += (
                "runs\$runId\standard_metrics\metrics.json"
            )
        }
        $rodSummaryArguments = @(
            "summarize-rods",
            "--metrics"
        ) + $rodMetricPaths + @(
            "--out-dir", "runs\stage2_validation\rod_width_series"
        )
        Invoke-Backscatter -Arguments $rodSummaryArguments
    }

    if (
        $Campaign -in @(
            "all",
            "double_rods",
            "double_rods_a",
            "double_rods_b"
        )
    ) {
        $doubleRodSeparations = if ($Campaign -eq "double_rods_a") {
            @(5, 10, 15)
        }
        elseif ($Campaign -eq "double_rods_b") {
            @(20, 30)
        }
        else {
            @(5, 10, 15, 20, 30)
        }
        $doubleRodMetricPaths = @()
        foreach ($separation in $doubleRodSeparations) {
            $runId = (
                "double_rod_width2_sep${separation}_" +
                "depth30_5m_seed24061873_local" +
                $DoubleRodRunSuffix
            )
            Process-Target `
                -RunId $runId `
                -Config "configs\generated\double_rod_width2_sep${separation}_depth30_5m.json" `
                -Primaries 5000000
            $doubleRodMetricPaths += (
                "runs\$runId\standard_metrics\metrics.json"
            )
        }
        $doubleRodSummaryDir = if ($Campaign -eq "double_rods_a") {
            "runs\stage2_validation\double_rod_series_a"
        }
        elseif ($Campaign -eq "double_rods_b") {
            "runs\stage2_validation\double_rod_series_b"
        }
        else {
            "runs\stage2_validation\double_rod_series"
        }
        $doubleRodSummaryArguments = @(
            "summarize-double-rods",
            "--metrics"
        ) + $doubleRodMetricPaths + @(
            "--out-dir", $doubleRodSummaryDir
        )
        Invoke-Backscatter -Arguments $doubleRodSummaryArguments
    }

    if ($Campaign -in @("all", "edge")) {
        $edgeMetricPaths = @()
        foreach ($depth in @(30, 70)) {
            $runId = (
                "edge_depth${depth}_2m_seed24061873_local" +
                $EdgeRunSuffix
            )
            $config = "configs\generated\edge_depth${depth}_2m.json"
            $raw = Join-Path $RawRoot $runId
            $analysis = Join-Path "runs" $runId
            Assert-CompletedRun `
                -RunDirectory $raw `
                -TruthFileName "hits_truth_target.root" `
                -ExpectedPrimaries 2000000
            Invoke-Backscatter -Arguments @(
                "analyze",
                "--root", (Join-Path $raw "hits.root"),
                "--truth", (Join-Path $raw "hits_truth_target.root"),
                "--config", $config,
                "--out-dir", $analysis
            )
            Invoke-Backscatter -Arguments @(
                "evaluate-edge",
                "--edge-analysis-dir", $analysis,
                "--reference-analysis-dir",
                "runs\depth_psf_${depth}mm_seed24061873_local",
                "--config", $config,
                "--out-dir", (Join-Path $analysis "edge_mtf"),
                "--bin-width-mm", "1",
                "--minimum-reference-count", "5",
                "--bootstrap", [string]$Bootstrap
            )
            $edgeMetricPaths += (
                "runs\$runId\edge_mtf\metrics.json"
            )
        }
        $edgeSummaryArguments = @(
            "summarize-edges",
            "--metrics"
        ) + $edgeMetricPaths + @(
            "--out-dir", "runs\stage2_validation\edge_depth_series"
        )
        Invoke-Backscatter -Arguments $edgeSummaryArguments
    }
}
finally {
    Pop-Location
}
