param(
    [Parameter(Mandatory = $true)]
    [string]$CampaignTag,

    [string]$PublicRoot = "runs\blind_void_qualification\public",
    [string]$AnalysisRoot = "runs\blind_void_qualification\analysis",
    [string]$ReferenceAnalysisDir = (
        "runs\concrete_slab_2m_seed24061873_local"
    ),
    [string]$NeutralConfig = (
        "configs\generated\blind_void_unknown_1m.json"
    )
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($CampaignTag -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "Unsafe campaign tag: $CampaignTag"
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
    $publicCampaign = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot (Join-Path $PublicRoot $CampaignTag))
    )
    $analysisCampaign = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot (Join-Path $AnalysisRoot $CampaignTag))
    )
    if (-not (
        Test-Path -LiteralPath $publicCampaign -PathType Container
    )) {
        throw "Public blind campaign is absent: $publicCampaign"
    }
    if (Test-Path -LiteralPath $analysisCampaign) {
        throw (
            "Refusing to overwrite an existing blind analysis campaign: " +
            $analysisCampaign
        )
    }
    foreach ($required in @(
        $ReferenceAnalysisDir,
        $NeutralConfig
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required blind-analysis input is absent: $required"
        }
    }

    $forbidden = @(
        Get-ChildItem -LiteralPath $publicCampaign -Recurse -File |
            Where-Object {
                $_.Name -match (
                    "truth_key|truth_salt|hits_truth|run_parameters|" +
                    "gate\.log|launch_plan"
                )
            }
    )
    if ($forbidden.Count -gt 0) {
        throw (
            "Refusing to score a public bundle containing private data: " +
            $forbidden[0].FullName
        )
    }
    $campaign = Get-Content -LiteralPath (
        Join-Path $publicCampaign "campaign_public.json"
    ) -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $campaign.campaign_tag -ne $CampaignTag -or
        [int]$campaign.case_count -ne 12
    ) {
        throw "Public campaign metadata is inconsistent."
    }
    if ([bool]$campaign.contains_case_labels_or_positions) {
        throw "Public campaign is not truth-blind."
    }

    New-Item -ItemType Directory -Path $analysisCampaign -Force |
        Out-Null
    $scoreRows = @()
    foreach ($caseId in @($campaign.case_ids)) {
        if (
            $caseId -notmatch (
                "^" + [regex]::Escape($CampaignTag) + "_case\d{3}$"
            )
        ) {
            throw "Unsafe or unexpected case ID: $caseId"
        }
        $publicCase = Join-Path $publicCampaign $caseId
        $publicManifestPath = Join-Path (
            $publicCase
        ) "public_manifest.json"
        $hitsPath = Join-Path $publicCase "hits.root"
        foreach ($required in @($publicManifestPath, $hitsPath)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Public case input is absent: $required"
            }
        }
        $publicManifest = Get-Content -LiteralPath (
            $publicManifestPath
        ) -Raw -Encoding UTF8 | ConvertFrom-Json
        $hitsHash = (
            Get-FileHash -LiteralPath $hitsPath -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        if (
            $hitsHash -ne (
                [string]$publicManifest.hits_sha256
            ).ToUpperInvariant()
        ) {
            throw "Public hits.root hash mismatch before scoring: $caseId"
        }

        $caseAnalysis = Join-Path $analysisCampaign $caseId
        $caseScore = Join-Path $caseAnalysis "blind_score"
        Write-Output "Analyzing sealed case $caseId without truth..."
        Invoke-Backscatter -Arguments @(
            "analyze",
            "--root", $hitsPath,
            "--config", $NeutralConfig,
            "--out-dir", $caseAnalysis
        )
        Invoke-Backscatter -Arguments @(
            "score-blind-void",
            "--sample-analysis-dir", $caseAnalysis,
            "--reference-analysis-dir", $ReferenceAnalysisDir,
            "--selection-config", $NeutralConfig,
            "--out-dir", $caseScore,
            "--roi-radius-mm", "15",
            "--x-min-mm", "-30",
            "--x-max-mm", "30",
            "--depth-min-mm", "20",
            "--depth-max-mm", "40",
            "--search-step-mm", "2",
            "--y-half-width-mm", "150",
            "--candidate-threshold", "4"
        )
        $scorePath = Join-Path $caseScore "blind_case_score.json"
        $score = Get-Content -LiteralPath (
            $scorePath
        ) -Raw -Encoding UTF8 | ConvertFrom-Json
        if (
            [bool]$score.blindness.uses_monte_carlo_truth -or
            [bool]$score.blindness.reads_sample_class -or
            [bool]$score.blindness.reads_target_position
        ) {
            throw "Blindness guard failed for $caseId"
        }
        if ([double]$score.candidate_threshold -ne 4.0) {
            throw "Unexpected candidate threshold for $caseId"
        }
        $scoreRows += [ordered]@{
            case_id = $caseId
            input_hits_sha256 = $hitsHash
            maximum_score = [double]$score.maximum_score
            estimated_x_mm = [double]$score.estimated_center_mm.x
            estimated_depth_mm = (
                [double]$score.estimated_center_mm.depth
            )
            candidate_threshold = [double]$score.candidate_threshold
            candidate_flag = [bool]$score.candidate_flag
            score_report_sha256 = (
                Get-FileHash -LiteralPath $scorePath -Algorithm SHA256
            ).Hash.ToUpperInvariant()
        }
    }

    $lockedScores = [ordered]@{
        schema_version = "1.0"
        kind = "blind_void_scores_locked_before_reveal"
        campaign_tag = $CampaignTag
        created_at = (Get-Date).ToString("o")
        scores_frozen_before_truth_reveal = $true
        contains_truth_labels_or_positions = $false
        public_truth_commitment_sha256 = (
            [string]$campaign.truth_commitment_sha256
        ).ToUpperInvariant()
        frozen_protocol = [ordered]@{
            target = "20 mm cylindrical void near 30 mm depth"
            roi_radius_mm = 15.0
            search_x_interval_mm = @(-30.0, 30.0)
            search_depth_interval_mm = @(20.0, 40.0)
            search_step_mm = 2.0
            candidate_threshold = 4.0
            localization_tolerance_mm_after_reveal = 15.0
        }
        case_count = $scoreRows.Count
        cases = $scoreRows
    }
    $scoreBundlePath = Join-Path (
        $analysisCampaign
    ) "blind_scores_locked.json"
    $lockedScores |
        ConvertTo-Json -Depth 7 |
        Set-Content -LiteralPath $scoreBundlePath -Encoding UTF8
    $scoreBundleHash = (
        Get-FileHash -LiteralPath $scoreBundlePath -Algorithm SHA256
    ).Hash.ToUpperInvariant()
    $hashPath = Join-Path (
        $analysisCampaign
    ) "blind_scores_locked.sha256"
    Set-Content -LiteralPath $hashPath -Encoding ASCII -Value (
        "$scoreBundleHash  blind_scores_locked.json"
    )

    Write-Output ""
    Write-Output "All 12 scores are frozen before reveal."
    Write-Output "Score bundle: $scoreBundlePath"
    Write-Output "SCORE_BUNDLE_SHA256=$scoreBundleHash"
    Write-Output (
        "Only now may the VM reveal script be run with this exact hash."
    )
}
finally {
    Pop-Location
}
