param(
    [string]$AtakanRepo = "",
    [string]$ArdaBranch = "samuel-keyboard-hover",
    [string]$AtakanBranch = "samuel",
    [string]$Python = "python",
    [string]$TargetKey = "h",
    [string]$TargetMap = "",
    [switch]$SkipGitSwitch,
    [switch]$SkipInstall,
    [switch]$AllowDirty,
    [switch]$StartInferenceServer,
    [switch]$RunLiveDetection
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$StartDir)
    $root = git -C $StartDir rev-parse --show-toplevel 2>$null
    if (-not $root) {
        throw "Could not resolve git root from $StartDir"
    }
    return (Resolve-Path $root).Path
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
}

function Invoke-CommandChecked {
    param(
        [string]$WorkingDirectory,
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Host "[$WorkingDirectory] $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Assert-CleanForSwitch {
    param(
        [string]$Repo,
        [string]$Branch
    )
    $current = git -C $Repo branch --show-current
    $status = git -C $Repo status --porcelain
    if ($status -and $current -ne $Branch -and -not $AllowDirty) {
        throw "$Repo has uncommitted changes on branch '$current'. Commit/stash them or rerun with -AllowDirty."
    }
}

function Switch-ToBranch {
    param(
        [string]$Repo,
        [string]$Branch
    )
    Assert-CleanForSwitch -Repo $Repo -Branch $Branch
    $current = git -C $Repo branch --show-current
    if ($current -eq $Branch) {
        Write-Host "$Repo already on $Branch"
        return
    }

    git -C $Repo fetch --all --prune
    $local = git -C $Repo branch --list $Branch
    if ($local) {
        git -C $Repo switch $Branch
        return
    }

    $remote = git -C $Repo branch --remotes --list "origin/$Branch"
    if ($remote) {
        git -C $Repo switch -c $Branch --track "origin/$Branch"
        return
    }

    throw "Branch '$Branch' was not found locally or at origin in $Repo."
}

function New-FixtureTargetMap {
    param([string]$Path)
    $payload = [ordered]@{
        schema_version = "0.1"
        calibration = [ordered]@{
            accepted = $true
            reason = "setup fixture"
        }
        key_targets = [ordered]@{
            $TargetKey = [ordered]@{
                center_px = @(500.0, 400.0)
                confidence = 0.9
                source = "detected"
            }
        }
    }
    $dir = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force $dir | Out-Null
    $json = $payload | ConvertTo-Json -Depth 8
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArdaRepo = Resolve-RepoRoot -StartDir $ScriptDir
if (-not $AtakanRepo) {
    $AtakanRepo = Join-Path (Split-Path -Parent $ArdaRepo) "atakan-code"
}
$AtakanRepo = (Resolve-Path $AtakanRepo).Path

Write-Host "Arda repo:   $ArdaRepo"
Write-Host "Atakan repo: $AtakanRepo"
Write-Host "Python:      $Python"

if (-not $SkipGitSwitch) {
    Invoke-Step "Switch arda-code to $ArdaBranch" {
        Switch-ToBranch -Repo $ArdaRepo -Branch $ArdaBranch
    }
    Invoke-Step "Switch atakan-code to $AtakanBranch" {
        Switch-ToBranch -Repo $AtakanRepo -Branch $AtakanBranch
    }
}

if (-not $SkipInstall) {
    Invoke-Step "Install atakan-code Python requirements" {
        Invoke-CommandChecked -WorkingDirectory $AtakanRepo -FilePath $Python -Arguments @(
            "-m", "pip", "install", "-r", (Join-Path $AtakanRepo "requirements.txt")
        )
    }
    Invoke-Step "Install arda-code LeRobot hover requirements" {
        Invoke-CommandChecked -WorkingDirectory $ArdaRepo -FilePath $Python -Arguments @(
            "-m", "pip", "install", "-e", "$ArdaRepo`[feetech,kinematics]"
        )
    }
}

$ServerProcess = $null
if ($StartInferenceServer) {
    Invoke-Step "Start local Roboflow inference server" {
        $envFile = Join-Path $AtakanRepo ".env.inference"
        if (-not (Test-Path $envFile)) {
            throw "Missing $envFile. Copy .env.inference.example and set ROBOFLOW_API_KEY first."
        }
        $bash = Get-Command bash -ErrorAction SilentlyContinue
        if (-not $bash) {
            throw "bash was not found. Start $(Join-Path $AtakanRepo 'start_inference.sh') manually in Git Bash/WSL."
        }
        $script = Join-Path $AtakanRepo "start_inference.sh"
        $ServerProcess = Start-Process `
            -FilePath $bash.Source `
            -ArgumentList @($script) `
            -WorkingDirectory $AtakanRepo `
            -PassThru `
            -WindowStyle Hidden
        Write-Host "Started inference server process id $($ServerProcess.Id). Waiting 10s for startup..."
        Start-Sleep -Seconds 10
    }
}

Invoke-Step "Run atakan-code unit tests" {
    Invoke-CommandChecked -WorkingDirectory $AtakanRepo -FilePath $Python -Arguments @(
        "-m", "unittest", "discover", "-s", "tests"
    )
}

Invoke-Step "Run atakan-code visual servo smoke test" {
    Invoke-CommandChecked -WorkingDirectory $AtakanRepo -FilePath $Python -Arguments @(
        "scripts\simulate_visual_servo.py",
        "--target-x", "500",
        "--target-y", "400",
        "--tip-x", "200",
        "--tip-y", "200",
        "--output", "outputs\setup_visual_servo_smoke.json"
    )
}

if ($RunLiveDetection) {
    Invoke-Step "Run live Atakan detector on sample images" {
        Invoke-CommandChecked -WorkingDirectory $AtakanRepo -FilePath $Python -Arguments @(
            "scripts\batch_detect_keyboards.py",
            "--input-dir", "keyboard_pictures",
            "--output-dir", "outputs\setup_batch_detection",
            "--threshold", "0.50"
        )
    }
}

if (-not $TargetMap) {
    $TargetMap = Join-Path $ArdaRepo "outputs\setup_fixture_target_map.json"
    Invoke-Step "Create fixture target map for Arda dry-run" {
        New-FixtureTargetMap -Path $TargetMap
        Write-Host "Wrote $TargetMap"
    }
}

Invoke-Step "Run arda-code hover controller help check" {
    Invoke-CommandChecked -WorkingDirectory $ArdaRepo -FilePath $Python -Arguments @(
        "scripts\project4\keyboard_hover_controller.py", "--help"
    )
}

Invoke-Step "Run arda-code target-map dry-run" {
    Invoke-CommandChecked -WorkingDirectory $ArdaRepo -FilePath $Python -Arguments @(
        "scripts\project4\keyboard_hover_controller.py",
        "--dry-run",
        "--target-map", $TargetMap,
        "--target-key", $TargetKey,
        "--output-dir", "outputs\setup_keyboard_hover_dry_run"
    )
}

Write-Host ""
Write-Host "Setup and offline tests completed." -ForegroundColor Green
Write-Host ""
Write-Host "Inputs:"
Write-Host "  - Atakan repo: $AtakanRepo"
Write-Host "  - Arda repo:   $ArdaRepo"
Write-Host "  - Target key:  $TargetKey"
Write-Host "  - Target map:  $TargetMap"
Write-Host ""
Write-Host "Outputs:"
Write-Host "  - Atakan servo smoke: $AtakanRepo\outputs\setup_visual_servo_smoke.json"
Write-Host "  - Arda dry-run report: $ArdaRepo\outputs\setup_keyboard_hover_dry_run\hover_report.json"
if ($RunLiveDetection) {
    Write-Host "  - Live detector outputs: $AtakanRepo\outputs\setup_batch_detection"
}
if ($StartInferenceServer -and $ServerProcess) {
    Write-Host ""
    Write-Host "Inference server is still running as process $($ServerProcess.Id). Stop it manually when done."
}
