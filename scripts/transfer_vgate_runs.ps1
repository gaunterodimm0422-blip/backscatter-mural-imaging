param(
    [Parameter(Mandatory = $true)]
    [string[]]$RunIds,

    [string]$DestinationRoot = "runs\vgate_fast_transfer\output",
    [string]$VmName = "vGate-9.4.1",
    [string]$VmUser = "vgate",
    [switch]$DeleteGuestAfterValidation
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$vboxManage = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$vmPassword = $env:VGATE_PASSWORD
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

    foreach ($runId in $RunIds) {
        if ($runId -notmatch "^[A-Za-z0-9_.-]+$") {
            throw "Unsafe run ID: $runId"
        }
        $guestRunRoot = "/home/vgate/backscatter_runs/$runId"
        $guestResult = "$guestRunRoot/gate/output/$runId"
        $resolvedGuest = Invoke-Guest `
            -Executable "/usr/bin/realpath" `
            -Arguments @($guestResult)
        if (
            $resolvedGuest -ne $guestResult -or
            -not $resolvedGuest.StartsWith(
                "/home/vgate/backscatter_runs/"
            )
        ) {
            throw "Unsafe or unexpected guest source: $resolvedGuest"
        }

        $finalPath = Join-Path $destinationRootPath $runId
        $stagingPath = Join-Path $destinationRootPath ".$runId.partial"
        if (Test-Path -LiteralPath $finalPath) {
            throw "Final host directory already exists: $finalPath"
        }
        if (Test-Path -LiteralPath $stagingPath) {
            throw "Partial host directory already exists: $stagingPath"
        }

        Write-Output "Copying $runId from the VM..."
        & $vboxManage guestcontrol $VmName copyfrom `
            --username $VmUser `
            --password $vmPassword `
            --recursive `
            $guestResult `
            $stagingPath
        if ($LASTEXITCODE -ne 0) {
            throw "VBoxManage copyfrom failed for $runId"
        }

        foreach ($required in @(
            "hits.root",
            "gate.log",
            "run_parameters.txt",
            "macro_sha256.txt",
            "gate_version.txt"
        )) {
            $path = Join-Path $stagingPath $required
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Copied run is missing $path"
            }
            if ((Get-Item -LiteralPath $path).Length -le 0) {
                throw "Copied run contains empty file $path"
            }
        }
        $truthFiles = @(
            Get-ChildItem -LiteralPath $stagingPath `
                -Filter "hits_truth_*.root" `
                -File
        )
        if ($truthFiles.Count -ne 1) {
            throw "Expected exactly one truth ROOT in $stagingPath"
        }
        $parameters = @{}
        foreach (
            $line in Get-Content -LiteralPath (
                Join-Path $stagingPath "run_parameters.txt"
            )
        ) {
            if ($line -match "^([^=]+)=(.*)$") {
                $parameters[$matches[1]] = $matches[2]
            }
        }
        $expectedPrimaries = [int64]$parameters["N_PRIMARIES"]
        if ($expectedPrimaries -le 0) {
            throw "Invalid N_PRIMARIES in copied run $runId"
        }
        $gateLog = Join-Path $stagingPath "gate.log"
        if (-not (
            Select-String -LiteralPath $gateLog -Quiet `
                -Pattern (
                    "Number of events processed\s*:\s*" +
                    [regex]::Escape([string]$expectedPrimaries)
                )
        )) {
            throw "Completion count is missing from $gateLog"
        }
        if (
            Select-String -LiteralPath $gateLog -Quiet `
                -Pattern "ERROR|FATAL|Aborted|core dumped|G4Exception"
        ) {
            throw "Fatal marker was found in $gateLog"
        }

        $hostHits = (
            Get-FileHash -LiteralPath (
                Join-Path $stagingPath "hits.root"
            ) -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        $guestHits = Get-GuestSha256 "$guestResult/hits.root"
        if ($hostHits -ne $guestHits) {
            throw "hits.root SHA-256 mismatch for $runId"
        }
        $hostTruth = (
            Get-FileHash -LiteralPath $truthFiles[0].FullName `
                -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        $guestTruth = Get-GuestSha256 (
            "$guestResult/$($truthFiles[0].Name)"
        )
        if ($hostTruth -ne $guestTruth) {
            throw "Truth ROOT SHA-256 mismatch for $runId"
        }

        $manifest = [ordered]@{
            schema_version = "1.0"
            run_id = $runId
            copied_at = (Get-Date).ToString("o")
            guest_source = $guestResult
            expected_primaries = $expectedPrimaries
            hits_sha256 = $hostHits
            truth_file = $truthFiles[0].Name
            truth_sha256 = $hostTruth
            gate_log_completion_verified = $true
            fatal_log_markers_absent = $true
        }
        $manifest |
            ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath (
                Join-Path $stagingPath "transfer_manifest.json"
            ) -Encoding UTF8

        Move-Item -LiteralPath $stagingPath -Destination $finalPath
        Write-Output "Validated host copy: $finalPath"

        if ($DeleteGuestAfterValidation) {
            $resolvedRoot = Invoke-Guest `
                -Executable "/usr/bin/realpath" `
                -Arguments @($guestRunRoot)
            if (
                $resolvedRoot -ne $guestRunRoot -or
                -not $resolvedRoot.StartsWith(
                    "/home/vgate/backscatter_runs/"
                )
            ) {
                throw "Unsafe guest deletion target: $resolvedRoot"
            }
            & $vboxManage guestcontrol $VmName rmdir `
                --username $VmUser `
                --password $vmPassword `
                --recursive `
                $guestRunRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Validated copy kept, but guest cleanup failed: $runId"
            }
            Write-Output "Removed validated guest copy: $guestRunRoot"
        }
    }
}
finally {
    Pop-Location
}
