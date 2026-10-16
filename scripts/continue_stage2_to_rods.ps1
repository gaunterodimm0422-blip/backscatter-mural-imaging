param(
    [string]$VmName = "vGate-9.4.1",
    [string]$VmUser = "vgate",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$vboxManage = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$vmPassword = $env:VGATE_PASSWORD
if (-not $vmPassword) {
    throw "Set VGATE_PASSWORD in the controller process environment."
}
if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10."
}

$controllerRoot = Join-Path $projectRoot "runs\vgate_controller"
$stage2Root = Join-Path $controllerRoot "stage2"
$logPath = Join-Path $controllerRoot "continue_stage2_to_rods.log"
$rodStdout = Join-Path $controllerRoot "rods_batch.stdout.txt"
$rodStderr = Join-Path $controllerRoot "rods_batch.stderr.txt"
New-Item -ItemType Directory -Path $controllerRoot -Force | Out-Null

function Write-ControllerLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Get-GuestGateProcessCount {
    $output = & $vboxManage guestcontrol $VmName run `
        --username $VmUser `
        --password $vmPassword `
        --wait-stdout `
        --wait-stderr `
        --timeout 15000 `
        --exe /usr/bin/pgrep `
        -- -c -x Gate 2>$null
    $exitCode = $LASTEXITCODE
    $count = 0
    $parsed = [int]::TryParse(
        ($output -join "").Trim(),
        [ref]$count
    )
    if ($parsed -and $count -eq 0 -and $exitCode -in @(1, 33)) {
        # VBoxManage maps a guest exit status of 1 to host status 33.
        return 0
    }
    if ($exitCode -ne 0) {
        throw "Could not query GATE processes in the VM (exit $exitCode)."
    }
    if (-not $parsed) {
        throw "Could not parse the GATE process count."
    }
    return $count
}

function Assert-Stage2ControllersSucceeded {
    $expectedRunIds = @(
        "repeat_thin30_seed24061874_local",
        "repeat_thin90_seed24061874_local",
        "repeat_concrete_seed24061874_local",
        "physical_d1r150_thin30_seed24061873_local",
        "physical_d1r175_thin30_seed24061873_local",
        "physical_d1r150_concrete_seed24061873_local"
    )
    foreach ($runId in $expectedRunIds) {
        $stdout = Join-Path $stage2Root "$runId.stdout.txt"
        $stderr = Join-Path $stage2Root "$runId.stderr.txt"
        if (-not (Test-Path -LiteralPath $stdout -PathType Leaf)) {
            throw "Missing stage-2 controller output: $stdout"
        }
        $reportedReady = (
            Select-String -LiteralPath $stdout -Quiet `
                -SimpleMatch "Local result ready:"
        )
        $reportedCompleted = (
            Select-String -LiteralPath $stdout -Quiet `
                -SimpleMatch "Completed: output/$runId"
        )
        if (-not ($reportedReady -or $reportedCompleted)) {
            throw "Stage-2 controller did not report success: $stdout"
        }
        if (
            (Test-Path -LiteralPath $stderr -PathType Leaf) -and
            (Get-Item -LiteralPath $stderr).Length -gt 0
        ) {
            $stderrText = Get-Content -LiteralPath $stderr -Raw
            $knownPostCompletionWrapperError = (
                $reportedCompleted -and
                $stderrText -match (
                    "run_local_export\.sh: line 43: " +
                    "ete_void\.sh: command not found"
                )
            )
            if (-not $knownPostCompletionWrapperError) {
                throw "Stage-2 controller wrote stderr: $stderr"
            }
            Write-ControllerLog (
                "Accepted known post-completion shared-script offset error " +
                "for $runId; raw GATE files still require full validation."
            )
        }
    }
    return $expectedRunIds
}

function Start-RodBatch {
    foreach ($path in @($rodStdout, $rodStderr)) {
        if (Test-Path -LiteralPath $path) {
            throw "Rod controller output already exists: $path"
        }
    }
    $arguments = @(
        "guestcontrol",
        $VmName,
        "run",
        "--username=$VmUser",
        "--password=$vmPassword",
        "--wait-stdout",
        "--wait-stderr",
        "--putenv=PATH=/home/vgate/Software/Gate/bin:/usr/local/bin:/usr/bin:/bin",
        "--putenv=LD_LIBRARY_PATH=/home/vgate/Software/Geant4/install/lib:/home/vgate/Software/root/bin/lib:/home/vgate/Software/libtorch/lib",
        "--exe=/usr/bin/bash",
        "--",
        "/media/sf_share/beisanshe_v1/gate/run_campaign_batch.sh",
        "rods"
    )
    $process = Start-Process `
        -FilePath $vboxManage `
        -ArgumentList $arguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $rodStdout `
        -RedirectStandardError $rodStderr `
        -PassThru
    Write-ControllerLog "Started rod batch; host PID=$($process.Id)."
}

Push-Location $projectRoot
try {
    Write-ControllerLog "Continuation controller started."
    while ($true) {
        $count = Get-GuestGateProcessCount
        Write-ControllerLog "Active GATE processes: $count."
        if ($count -eq 0) {
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }

    # Allow the six waiting VBoxManage controllers to receive their final output.
    $runIds = $null
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        Start-Sleep -Seconds 10
        try {
            $runIds = Assert-Stage2ControllersSucceeded
            break
        }
        catch {
            if ($attempt -eq 12) {
                throw
            }
            Write-ControllerLog (
                "Waiting for stage-2 controller finalization " +
                "(attempt $attempt/12)."
            )
        }
    }
    if ($null -eq $runIds) {
        throw "Stage-2 controller outputs were not finalized."
    }
    Write-ControllerLog "All six stage-2 GATE controllers reported success."

    & (Join-Path $PSScriptRoot "transfer_vgate_runs.ps1") `
        -RunIds $runIds `
        -DestinationRoot "runs\vgate_fast_transfer\output" `
        -DeleteGuestAfterValidation
    if ($LASTEXITCODE -ne 0) {
        throw "Validated stage-2 transfer failed."
    }
    Write-ControllerLog "Stage-2 files transferred, hashed and guest copies removed."

    Start-RodBatch

    & (Join-Path $PSScriptRoot "process_stage2_validation.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-2 host analysis failed."
    }
    Write-ControllerLog "Stage-2 analysis completed successfully."
}
catch {
    Write-ControllerLog "FAILED: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
