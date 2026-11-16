# Pull source changes and rebuild the exact GUI locally; no artifact download.
# The existing GUI is replaced only after the staged EXE verifies its build ID.
[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipPull,
    [switch]$SkipSync,
    [switch]$SkipBuildTools,
    [switch]$NoLaunch,
    [string]$PythonExecutable = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Program"
    }
}

function Assert-GuiStopped([string]$ExePath) {
    foreach ($running in @(Get-Process -Name VideoTranslate -ErrorAction SilentlyContinue)) {
        try { $runningPath = $running.Path } catch { continue }
        if ($runningPath -and [string]::Equals(
                [IO.Path]::GetFullPath($runningPath), $ExePath,
                [StringComparison]::OrdinalIgnoreCase)) {
            throw "Close the GUI at $ExePath (PID $($running.Id)), then rerun this updater. No process was terminated."
        }
    }
    if (Test-Path -LiteralPath $ExePath -PathType Leaf) {
        $probe = $null
        try {
            $probe = [IO.File]::Open($ExePath, [IO.FileMode]::Open,
                [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        } catch {
            throw "The current EXE is locked or not writable: $ExePath. Close it and rerun the updater."
        } finally {
            if ($null -ne $probe) { $probe.Dispose() }
        }
    }
}

$stagingRoot = $null
$preserveStaging = $false
$savedPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$savedQtPlatform = [Environment]::GetEnvironmentVariable('QT_QPA_PLATFORM', 'Process')
$locationPushed = $false

try {
    if ($env:OS -ne 'Windows_NT') { throw 'This GUI updater requires Windows.' }
    $project = (Resolve-Path -LiteralPath $ProjectDir).Path
    if (-not (Test-Path -LiteralPath (Join-Path $project 'pyproject.toml') -PathType Leaf)) {
        throw "Not a project folder: $project"
    }
    $guiSource = Join-Path $project 'src/auto_translate/gui.py'
    if (-not (Test-Path -LiteralPath $guiSource -PathType Leaf)) { throw "GUI source missing: $guiSource" }
    $target = Join-Path $project 'dist/VideoTranslate'
    $targetExe = Join-Path $target 'VideoTranslate.exe'
    Assert-GuiStopped $targetExe
    Push-Location -LiteralPath $project
    $locationPushed = $true

    if (-not $SkipPull) {
        # Do not package a locally modified old GUI under a newer commit ID.
        $sourceChanges = @(& git status --porcelain -- 'src/auto_translate/gui*.py')
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect local GUI source changes.' }
        if ($sourceChanges.Count -gt 0) {
            throw 'Local GUI source changes were found. Commit or set them aside before updating; nothing was overwritten.'
        }
        Invoke-Checked 'git' @('pull', '--ff-only', 'origin', 'feature/dual-asr-review-improvements')
    }
    $commitOutput = @(& git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0) { throw 'Could not identify the source commit.' }
    $commit = ($commitOutput -join '').Trim()
    if ($commit -notmatch '^[0-9a-fA-F]{40}$') { throw 'Source commit ID is invalid.' }
    $commit = $commit.ToLowerInvariant()
    $sourceHash = (Get-FileHash -LiteralPath $guiSource -Algorithm SHA256).Hash.ToLowerInvariant()

    if (-not $SkipSync) {
        Invoke-Checked 'uv' @('sync', '--locked', '--extra', 'gui')
    }
    if (-not $PythonExecutable) {
        $PythonExecutable = Join-Path $project '.venv/Scripts/python.exe'
    }
    $python = (Resolve-Path -LiteralPath $PythonExecutable).Path
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python interpreter missing: $python" }
    if (-not $SkipBuildTools) {
        Invoke-Checked 'uv' @('pip', 'install', '--python', $python,
            'pyinstaller==6.22.3', 'pyinstaller-hooks-contrib==2026.10')
    }

    $buildRoot = Join-Path $project 'build/GUI-update'
    $null = New-Item -ItemType Directory -Path $buildRoot -Force
    $stagingRoot = Join-Path $project ('dist/.gui-update-' + [guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $stagingRoot -Force
    $manifest = Join-Path $buildRoot 'gui-build.json'
    $buildInfo = [ordered]@{ schema_version = 1; commit = $commit; gui_sha256 = $sourceHash; protocol = 1 }
    # No BOM: the bundled Python reader uses ordinary UTF-8 JSON.
    [IO.File]::WriteAllText($manifest, ($buildInfo | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    $env:PYTHONPATH = Join-Path $project 'src'
    Write-Host "Building GUI commit $commit from $guiSource"
    Write-Host "GUI source SHA256: $sourceHash"
    Invoke-Checked $python @('-m', 'PyInstaller', '--clean', '--noconfirm', '--onedir', '--windowed',
        '--name', 'VideoTranslate', '--paths', (Join-Path $project 'src'),
        '--workpath', (Join-Path $buildRoot 'work'), '--specpath', (Join-Path $buildRoot 'spec'),
        '--distpath', $stagingRoot, '--add-data', ($manifest + ';.'), $guiSource)
    $stagedGui = Join-Path $stagingRoot 'VideoTranslate'
    $stagedExe = Join-Path $stagedGui 'VideoTranslate.exe'
    if (-not (Test-Path -LiteralPath $stagedExe -PathType Leaf)) { throw 'Build did not produce the staged EXE.' }

    $smokeLog = Join-Path $project 'logs/gui-packaged-smoke.log'
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $smokeLog) -Force
    if (Test-Path -LiteralPath $smokeLog) { Remove-Item -LiteralPath $smokeLog -Force }
    $env:QT_QPA_PLATFORM = 'offscreen'
    $smoke = Start-Process -FilePath $stagedExe -WorkingDirectory $project -PassThru -ArgumentList @(
        '--smoke-test', '--verify-build', $commit, '--verify-source-sha', $sourceHash)
    if (-not $smoke.WaitForExit(30000)) {
        # This is only the updater's own model-free smoke process.
        $smoke.Kill()
        $smoke.WaitForExit()
        throw 'Staged GUI smoke test timed out; existing GUI was not changed.'
    }
    $smoke.Refresh()
    if ($smoke.ExitCode -ne 0) { throw 'Staged GUI build verification failed; existing GUI was not changed.' }
    if (-not (Test-Path -LiteralPath $smokeLog -PathType Leaf) -or
            -not ([IO.File]::ReadAllText($smokeLog).Contains('Transcript/files/report smoke passed'))) {
        throw 'Staged GUI did not emit its smoke success marker; existing GUI was not changed.'
    }
    # A source change during a lengthy build must not be silently mislabeled.
    if ((Get-FileHash -LiteralPath $guiSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash) {
        throw 'GUI source changed during the build; existing GUI was not changed. Rerun the updater.'
    }
    Assert-GuiStopped $targetExe
    $preserveStaging = $true
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    Move-Item -LiteralPath $stagedGui -Destination $target
    $preserveStaging = $false
    $exeHash = (Get-FileHash -LiteralPath $targetExe -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Updated EXE: $targetExe"
    Write-Host "Build commit: $commit"
    Write-Host "EXE SHA256: $exeHash"
    # The real GUI must not inherit the temporary offscreen smoke platform.
    [Environment]::SetEnvironmentVariable('QT_QPA_PLATFORM', $savedQtPlatform, 'Process')
    if (-not $NoLaunch) { Start-Process -FilePath $targetExe -WorkingDirectory $project }
} catch {
    Write-Host "GUI UPDATE ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($preserveStaging) { Write-Host "Validated build retained for recovery at: $stagingRoot" }
    exit 1
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $savedPythonPath, 'Process')
    [Environment]::SetEnvironmentVariable('QT_QPA_PLATFORM', $savedQtPlatform, 'Process')
    if ($locationPushed) { Pop-Location }
    if (-not $preserveStaging -and $stagingRoot -and (Test-Path -LiteralPath $stagingRoot)) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
