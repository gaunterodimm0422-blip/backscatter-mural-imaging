param(
    [Parameter(Mandatory = $true)]
    [string]$CampaignTag,

    [string]$DestinationRoot = "runs\blind_void_qualification\public",
    [string]$VmName = "vGate-9.4.1",
    [string]$VmUser = "vgate",
    [switch]$DeleteGuestPublicAfterValidation
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$vboxManage = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$vmPassword = $env:VGATE_PASSWORD
if ($CampaignTag -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "Unsafe campaign tag: $CampaignTag"
}
if (-not $vmPassword) {
    throw "Set the VGATE_PASSWORD environment variable for this process."
}
if (-not (Test-Path -LiteralPath $vboxManage -PathType Leaf)) {
    throw "VBoxManage was not found at $vboxManage"
}

function Invoke-Guest {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    $output = & $vboxManage guestcontrol $VmName run `
        --username $VmUser `
        --password $vmPassword `
        --wait-stdout `
        --wait-stderr `
        --timeout 30000 `
        --exe $Executable `
        -- @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Guest command failed: $Executable $($Arguments -join ' ')"
    }
    return ($output -join "`n").Trim()
}

function Get-GuestSha256 {
    param([string]$GuestPath)
    $line = Invoke-Guest `
        -Executable "/usr/bin/sha256sum" `
        -Arguments @($GuestPath)
    $hash = ($line -split "\s+")[0].ToUpperInvariant()
    if ($hash -notmatch "^[0-9A-F]{64}$") {
        throw "Could not parse guest SHA-256 for $GuestPath"
    }
    return $hash
}

Push-Location $projectRoot
try {
    $destinationRootPath = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot $DestinationRoot)
    )
    New-Item -ItemType Directory -Path $destinationRootPath -Force |
        Out-Null
    $finalPath = Join-Path $destinationRootPath $CampaignTag
    $stagingPath = Join-Path $destinationRootPath ".$CampaignTag.partial"
    if (Test-Path -LiteralPath $finalPath) {
        throw "Final host campaign already exists: $finalPath"
    }
    if (Test-Path -LiteralPath $stagingPath) {
        throw "Partial host campaign already exists: $stagingPath"
    }

    $guestCampaign = (
        "/home/vgate/backscatter_blind_public/" + $CampaignTag
    )
    $resolvedGuest = Invoke-Guest `
        -Executable "/usr/bin/realpath" `
        -Arguments @($guestCampaign)
    if (
        $resolvedGuest -ne $guestCampaign -or
        -not $resolvedGuest.StartsWith(
            "/home/vgate/backscatter_blind_public/"
        )
    ) {
        throw "Unsafe or unexpected guest source: $resolvedGuest"
    }

    Write-Output "Copying sealed public campaign $CampaignTag from the VM..."
    & $vboxManage guestcontrol $VmName copyfrom `
        --username $VmUser `
        --password $vmPassword `
        --recursive `
        $guestCampaign `
        $stagingPath
    if ($LASTEXITCODE -ne 0) {
        throw "VBoxManage copyfrom failed for $CampaignTag"
    }

    foreach ($required in @(
        "campaign_public.json",
        "truth_commitment.sha256"
    )) {
        $path = Join-Path $stagingPath $required
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Copied public campaign is missing $path"
        }
        if ((Get-Item -LiteralPath $path).Length -le 0) {
            throw "Copied public campaign contains empty file $path"
        }
    }
    $forbidden = @(
        Get-ChildItem -LiteralPath $stagingPath -Recurse -File |
            Where-Object {
                $_.Name -match (
                    "truth_key|truth_salt|hits_truth|run_parameters|" +
                    "gate\.log|launch_plan"
                )
            }
    )
    if ($forbidden.Count -gt 0) {
        throw (
            "Public bundle leaked private artifact: " +
            $forbidden[0].FullName
        )
    }

    $campaign = Get-Content -LiteralPath (
        Join-Path $stagingPath "campaign_public.json"
    ) -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($campaign.campaign_tag -ne $CampaignTag) {
        throw "Public campaign tag does not match the requested tag."
    }
    if (
        [int]$campaign.case_count -ne 12 -or
        @($campaign.case_ids).Count -ne 12
    ) {
        throw "Expected exactly 12 public blind cases."
    }
    if ([bool]$campaign.contains_case_labels_or_positions) {
        throw "Public campaign declares private labels or positions."
    }

    $caseRows = @()
    foreach ($caseId in @($campaign.case_ids)) {
        if (
            $caseId -notmatch (
                "^" + [regex]::Escape($CampaignTag) + "_case\d{3}$"
            )
        ) {
            throw "Unsafe or unexpected case ID: $caseId"
        }
        $casePath = Join-Path $stagingPath $caseId
        foreach ($required in @(
            "hits.root",
            "public_manifest.json",
            "gate_version.txt",
            "macro_sha256.txt"
        )) {
            $path = Join-Path $casePath $required
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Public case is missing $path"
            }
            if ((Get-Item -LiteralPath $path).Length -le 0) {
                throw "Public case contains empty file $path"
            }
        }
        $publicManifest = Get-Content -LiteralPath (
            Join-Path $casePath "public_manifest.json"
        ) -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($publicManifest.case_id -ne $caseId) {
            throw "Public case manifest ID mismatch for $caseId"
        }
        if (
            [int64]$publicManifest.expected_primaries -ne
            [int64]$campaign.primary_events_per_case
        ) {
            throw "Primary-event mismatch for $caseId"
        }
        if ([bool]$publicManifest.contains_truth_or_case_label) {
            throw "Public case manifest declares private information."
        }
        if (
            -not [bool]$publicManifest.gate_log_completion_verified_privately -or
            -not [bool]$publicManifest.fatal_log_markers_absent_privately
        ) {
            throw "Private GATE validation did not pass for $caseId"
        }

        $hostHash = (
            Get-FileHash -LiteralPath (
                Join-Path $casePath "hits.root"
            ) -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        $manifestHash = (
            [string]$publicManifest.hits_sha256
        ).ToUpperInvariant()
        $guestHash = Get-GuestSha256 (
            "$guestCampaign/$caseId/hits.root"
        )
        if ($hostHash -ne $manifestHash -or $hostHash -ne $guestHash) {
            throw "Public hits.root SHA-256 mismatch for $caseId"
        }
        $caseRows += [ordered]@{
            case_id = $caseId
            expected_primaries = [int64]$publicManifest.expected_primaries
            hits_sha256 = $hostHash
        }
        Write-Output "Validated sealed public case: $caseId"
    }

    $transferManifest = [ordered]@{
        schema_version = "1.0"
        campaign_tag = $CampaignTag
        copied_at = (Get-Date).ToString("o")
        guest_source = $guestCampaign
        case_count = $caseRows.Count
        private_truth_transferred = $false
        truth_commitment_sha256 = (
            Get-Content -LiteralPath (
                Join-Path $stagingPath "truth_commitment.sha256"
            ) -Raw
        ).Trim().ToUpperInvariant()
        cases = $caseRows
    }
    $transferManifest |
        ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (
            Join-Path $stagingPath "host_transfer_manifest.json"
        ) -Encoding UTF8

    Move-Item -LiteralPath $stagingPath -Destination $finalPath
    Write-Output "Validated sealed host campaign: $finalPath"

    if ($DeleteGuestPublicAfterValidation) {
        & $vboxManage guestcontrol $VmName rmdir `
            --username $VmUser `
            --password $vmPassword `
            --recursive `
            $guestCampaign
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Host copy is valid, but guest public cleanup failed: " +
                $guestCampaign
            )
        }
        Write-Output (
            "Removed validated guest public copy. " +
            "The private answer package was not touched."
        )
    }
}
finally {
    Pop-Location
}
