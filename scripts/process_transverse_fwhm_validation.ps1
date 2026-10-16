param(
    [string]$RunId = (
        "rod_width2_depth30_5m_seed24080573_local_" +
        "fwhm5_validation"
    ),
    [string]$RawRoot = "runs\vgate_fast_transfer\output",
    [int]$Bootstrap = 300
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$expectedPrimaries = 5000000
$expectedGateSeed = 24080573

if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]*$") {
    throw "Unsafe run ID: $RunId"
}
if ($Bootstrap -le 0) {
    throw "Bootstrap must be a positive integer."
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    $raw = Join-Path $RawRoot $RunId
    $hits = Join-Path $raw "hits.root"
    $truth = Join-Path $raw "hits_truth_target.root"
    $gateLog = Join-Path $raw "gate.log"
    $parametersPath = Join-Path $raw "run_parameters.txt"
    foreach ($path in @(
        $hits,
        $truth,
        $gateLog,
        $parametersPath,
        (Join-Path $raw "macro_sha256.txt"),
        (Join-Path $raw "transfer_manifest.json")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Transferred validation run is incomplete: missing $path"
        }
        if ((Get-Item -LiteralPath $path).Length -le 0) {
            throw "Transferred validation run contains an empty file: $path"
        }
    }

    $parameters = @{}
    foreach ($line in Get-Content -LiteralPath $parametersPath) {
        if ($line -match "^([^=]+)=(.*)$") {
            $parameters[$matches[1]] = $matches[2]
        }
    }
    if ([int64]$parameters["N_PRIMARIES"] -ne $expectedPrimaries) {
        throw (
            "Expected $expectedPrimaries primaries, got " +
            $parameters["N_PRIMARIES"]
        )
    }
    if ([int64]$parameters["RANDOM_SEED"] -ne $expectedGateSeed) {
        throw (
            "Expected GATE seed $expectedGateSeed, got " +
            $parameters["RANDOM_SEED"]
        )
    }
    foreach ($expected in @{
        "TARGET_DEPTH_MM" = "30"
        "TARGET_RADIUS_MM" = "1"
        "TARGET_LATERAL_MM" = "300"
    }.GetEnumerator()) {
        if ([string]$parameters[$expected.Key] -ne $expected.Value) {
            throw (
                "Run parameter $($expected.Key) must be " +
                "$($expected.Value), got $($parameters[$expected.Key])"
            )
        }
    }
    if (-not (
        Select-String -LiteralPath $gateLog -Quiet `
            -Pattern "Number of events processed\s*:\s*5000000"
    )) {
        throw "The exact 5,000,000-event completion marker is absent."
    }
    if (
        Select-String -LiteralPath $gateLog -Quiet `
            -Pattern "ERROR|FATAL|Aborted|core dumped|G4Exception"
    ) {
        throw "Fatal marker found in the transferred GATE log."
    }

    $analysis = Join-Path "runs" $RunId
    if (Test-Path -LiteralPath $analysis) {
        throw (
            "Analysis directory already exists: $analysis. " +
            "Keep it immutable or choose a new run ID."
        )
    }
    $output = Join-Path (
        "outputs\transverse_fwhm_5mm_20260805\" +
        "independent_gate_seed"
    ) $RunId
    if (Test-Path -LiteralPath $output) {
        throw (
            "Validation output already exists: $output. " +
            "Keep it immutable or choose a new run ID."
        )
    }

    $config = "configs\transverse_fwhm_5mm_frozen.json"
    Write-Output "[1/2] Extracting event pairs and truth from the transferred ROOT files..."
    Write-Output (
        "The detector response is intentionally applied in step 2, " +
        "after ideal GATE hits have been summed per event."
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "backscatter", "analyze",
        "--root", $hits,
        "--truth", $truth,
        "--config", $config,
        "--out-dir", $analysis
    )

    $pairExtractionManifest = [ordered]@{
        schema_version = "1.0"
        run_id = $RunId
        purpose = (
            "Extract ideal event-summed GATE pairs and truth. " +
            "Detector energy/position response is not applied by analyze."
        )
        raw_root = (Resolve-Path -LiteralPath $hits).Path
        raw_truth = (Resolve-Path -LiteralPath $truth).Path
        frozen_config = (Resolve-Path -LiteralPath $config).Path
        gate_seed = $expectedGateSeed
        primaries = $expectedPrimaries
        next_step = "validate_transverse_fwhm_gate_seed.py"
    }
    $pairExtractionManifest |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (
            Join-Path $analysis "pair_extraction_manifest.json"
        ) -Encoding UTF8

    Write-Output "[2/2] Applying the frozen detector response and 5 mm acceptance test..."
    Invoke-CheckedPython -Arguments @(
        "scripts\validate_transverse_fwhm_gate_seed.py",
        "--analysis-dir", $analysis,
        "--config", $config,
        "--output-dir", $output,
        "--gate-seed", [string]$expectedGateSeed,
        "--response-seed-start", "24080601",
        "--response-seed-count", "25",
        "--bootstrap", [string]$Bootstrap
    )

    Write-Output "Independent transverse-FWHM validation passed."
    Write-Output (
        "Report: " +
        (Join-Path $output "独立GATE种子_横向FWHM5mm验收.md")
    )
}
finally {
    Pop-Location
}
