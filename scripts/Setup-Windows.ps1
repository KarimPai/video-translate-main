# Windows 11 x64, NVIDIA CUDA + Vulkan; run through ../Setup.bat.
# Installer sources: WinGet, nyosegawa/speech.cpp, ggml-org/llama.cpp.
param([Parameter(Mandatory = $true)][string]$ProjectDir)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user;$env:Path"
}
function Require-Command([string]$Command, [string]$Hint) {
    Refresh-Path
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Cannot find $Command. $Hint. Restart Windows Terminal and rerun Setup.bat."
    }
}
function Test-MpvAvailable {
    # shinchiro.mpv normally installs here without adding mpv.exe to PATH.
    # The application also checks this same location when launching MPV.
    if (Get-Command mpv -ErrorAction SilentlyContinue) { return $true }
    return (Test-Path -LiteralPath 'C:\Program Files\MPV Player\mpv.exe' -PathType Leaf)
}
function Install-Winget([string]$Command, [string]$PackageId) {
    Refresh-Path
    if ($Command -eq 'mpv') {
        if (Test-MpvAvailable) {
            Write-Host 'Already installed: MPV (PATH or C:\Program Files\MPV Player\mpv.exe)'
            return
        }
    } elseif (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "Already installed: $Command"
        return
    }
    Step "Installing $PackageId using WinGet"
    & winget install --source winget --exact --id $PackageId --accept-package-agreements --accept-source-agreements
    $wingetExitCode = $LASTEXITCODE
    # WinGet returns APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE for an
    # already-installed, up-to-date package. It is not an installation failure.
    if ($wingetExitCode -eq -1978335189) {
        Write-Host "WinGet reports $PackageId is already installed and up to date."
    } elseif ($wingetExitCode -ne 0) {
        throw "WinGet installation failed: $PackageId (exit $wingetExitCode)"
    }
    Refresh-Path
    if ($Command -eq 'mpv') {
        if (Test-MpvAvailable) {
            Write-Host 'MPV executable found (PATH or C:\Program Files\MPV Player\mpv.exe).'
            return
        }
        throw 'MPV is reported installed but mpv.exe was not found. Check C:\Program Files\MPV Player\mpv.exe or add its folder to PATH, then rerun Setup.bat.'
    }
    Require-Command $Command "WinGet should have installed $PackageId"
}
function Fetch-Asset([object]$Asset, [string]$Destination) {
    Write-Host "Downloading $($Asset.name)..."
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $Destination -UseBasicParsing
    if ($Asset.digest -match '^sha256:([0-9a-fA-F]{64})$') {
        $expected = $Matches[1]
        $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($actual -ine $expected) { throw "SHA256 mismatch for $($Asset.name)" }
        Write-Host "SHA256 verified: $($Asset.name)"
    } else {
        Write-Warning "No SHA256 published for $($Asset.name); using official GitHub release download."
    }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') {
        throw 'Windows x64 is required.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir 'pyproject.toml'))) {
        throw 'Run Setup.bat from the fully extracted project folder (pyproject.toml missing).'
    }
    $adapterNames = @((Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name))
    if (@($adapterNames | Where-Object { $_ -match 'NVIDIA' }).Count -eq 0) {
        throw 'This branch requires an NVIDIA GPU with CUDA and a Vulkan-capable driver for FireRedVAD/Haas.'
    }
    Write-Host 'Detected video adapters:' ($adapterNames -join '; ')

    Step 'Checking Windows Package Manager'
    Require-Command winget 'Install Microsoft App Installer from Microsoft Store'
    Install-Winget uv 'astral-sh.uv'
    Install-Winget 'yt-dlp' 'yt-dlp.yt-dlp'
    Install-Winget ffmpeg 'yt-dlp.FFmpeg'
    Install-Winget mpv 'shinchiro.mpv'

    Step 'Installing Python 3.12 and project dependencies'
    Push-Location -LiteralPath $ProjectDir
    try {
        & uv python install 3.12
        if ($LASTEXITCODE -ne 0) { throw 'uv python install 3.12 failed.' }
        & uv sync --locked
        if ($LASTEXITCODE -ne 0) { throw 'uv sync --locked failed. Review Python/CUDA package compatibility.' }
    } finally { Pop-Location }

    Step 'Installing speech.cpp (official installer)'
    if (-not (Get-Command speech -ErrorAction SilentlyContinue)) {
        $speechScript = Join-Path $env:TEMP ('speech-install-' + [guid]::NewGuid().ToString('N') + '.ps1')
        try {
            Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1' -OutFile $speechScript -UseBasicParsing
            & $speechScript
            if (-not $?) { throw 'speech.cpp installer failed.' }
        } finally { Remove-Item -LiteralPath $speechScript -Force -ErrorAction SilentlyContinue }
    } else { Write-Host 'speech.cpp already installed.' }
    Require-Command speech 'Check the speech.cpp Windows Vulkan installer'
    Step 'Downloading Haas-ja-v1.0 ASR model'
    & speech pull haas-ja-v1.0
    if ($LASTEXITCODE -ne 0) { throw 'speech pull haas-ja-v1.0 failed.' }
    & speech info haas-ja-v1.0
    if ($LASTEXITCODE -ne 0) { throw 'speech info haas-ja-v1.0 failed.' }
    Write-Host 'Available speech.cpp devices:'
    & speech devices
    if ($LASTEXITCODE -ne 0) { throw 'speech devices failed. Check your Vulkan driver.' }
    Write-Host 'The selected Haas device must be Vulkan0 (see device list above).'

    Step 'Installing CUDA 12.4 llama.cpp binaries and matching runtime'
    $installRoot = Join-Path $env:LOCALAPPDATA 'HaasPLaMo'
    $llamaRoot = Join-Path $installRoot 'llama-cuda12'
    $pathFile = Join-Path $installRoot 'llama-bin-path.txt'
    New-Item -ItemType Directory -Path $llamaRoot -Force | Out-Null
    $existingServer = Get-ChildItem -LiteralPath $llamaRoot -Recurse -Filter 'llama-server.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $existingServer) {
        $headers = @{ 'User-Agent' = 'HaasPLaMo-Setup'; 'Accept' = 'application/vnd.github+json' }
        # Version-tag releases may not provide Windows assets. Find the newest with BOTH CUDA archives.
        $releases = @(Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=25' -Headers $headers)
        $build = $null
        $runtime = $null
        foreach ($release in $releases) {
            if ($release.prerelease -or $release.draft) { continue }
            $a = @($release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1)
            $b = @($release.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip' } | Select-Object -First 1)
            if ($a.Count -and $b.Count) { $build = $a[0]; $runtime = $b[0]; break }
        }
        if (-not $build -or -not $runtime) {
            throw 'No matching CUDA 12.4 llama.cpp Windows release found. See https://github.com/ggml-org/llama.cpp/releases'
        }
        $downloadDir = Join-Path $env:TEMP ('haas-plamo-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
        try {
            $buildZip = Join-Path $downloadDir 'llama.zip'
            $runtimeZip = Join-Path $downloadDir 'cudart.zip'
            Fetch-Asset $build $buildZip
            Fetch-Asset $runtime $runtimeZip
            Expand-Archive -LiteralPath $buildZip -DestinationPath $llamaRoot -Force
            Expand-Archive -LiteralPath $runtimeZip -DestinationPath $llamaRoot -Force
        } finally { Remove-Item -LiteralPath $downloadDir -Recurse -Force -ErrorAction SilentlyContinue }
        $existingServer = Get-ChildItem -LiteralPath $llamaRoot -Recurse -Filter 'llama-server.exe' -File | Select-Object -First 1
        if (-not $existingServer) { throw 'llama-server.exe missing after extracting official archives.' }
    } else { Write-Host 'CUDA 12.4 llama-server.exe already downloaded.' }
    $binDir = $existingServer.Directory.FullName
    Set-Content -LiteralPath $pathFile -Value $binDir -Encoding ASCII
    $env:Path = "$binDir;$env:Path"
    & $existingServer.FullName --version
    if ($LASTEXITCODE -ne 0) { throw 'llama-server.exe failed to launch. Update the NVIDIA driver or check CUDA runtime DLLs.' }

    Step 'Checking Python environment'
    Push-Location -LiteralPath $ProjectDir
    try {
        & uv run --no-sync python -c 'import numpy, fireredvad, auto_translate; print("Python dependency check OK")'
        if ($LASTEXITCODE -ne 0) { throw 'Python import check failed.' }
    } finally { Pop-Location }

    Write-Host "`nSetup complete." -ForegroundColor Green
    Write-Host 'Haas downloaded; PLaMo 3 Translate 2B Q4_K_M (1.81 GB) downloads on first Start translate.bat run.'
    Write-Host 'A CUDA-capable NVIDIA driver and working Vulkan0 device are required.'
    exit 0
} catch {
    Write-Host "`nSETUP ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Rerun Setup.bat after fixing this issue. Existing successful downloads are reused.'
    exit 1
}
