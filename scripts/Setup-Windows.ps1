# Windows 11 x64, NVIDIA CUDA; run through ../Setup.bat.
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
function Install-Winget([string]$Command, [string]$PackageId) {
    Refresh-Path
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "Already installed: $Command"
        return
    }
    Step "Installing $PackageId using WinGet"
    & winget install --source winget --exact --id $PackageId --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "WinGet installation failed: $PackageId (exit $LASTEXITCODE)" }
    Require-Command $Command "WinGet should have installed $PackageId"
}
function Find-Mpv {
    $command = Get-Command mpv.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}, "$env:LOCALAPPDATA\Programs")
    foreach ($root in $roots) {
        if (-not $root) { continue }
        foreach ($folder in @('MPV Player', 'mpv', 'mpv.net')) {
            $candidate = Join-Path $root "$folder\mpv.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }
    foreach ($scope in @('HKCU:', 'HKLM:')) {
        $key = "$scope\Software\Microsoft\Windows\CurrentVersion\App Paths\mpv.exe"
        if (Test-Path $key) {
            $candidate = (Get-Item $key).GetValue('')
            if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
        }
    }
    $packages = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    Get-ChildItem -Path "$packages\shinchiro.mpv_*\*" -Filter mpv.exe -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
function Fetch-Asset([object]$Asset, [string]$Destination, [string]$PinnedDigest) {
    Write-Host "Downloading $($Asset.name)..."
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $Destination -UseBasicParsing
    if ($Asset.digest -match '^sha256:([0-9a-fA-F]{64})$') {
        $expected = $Matches[1]
        if ($expected -ine $PinnedDigest) { throw "Release digest differs from pinned SHA256 for $($Asset.name)" }
        $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($actual -ine $expected) { throw "SHA256 mismatch for $($Asset.name)" }
        Write-Host "SHA256 verified: $($Asset.name)"
    } else {
        throw "Missing SHA256 for $($Asset.name)"
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
        throw 'This branch requires an NVIDIA GPU with a current CUDA-capable driver for Qwen3-ASR + HY-MT.'
    }
    Write-Host 'Detected video adapters:' ($adapterNames -join '; ')

    Step 'Checking Windows Package Manager'
    Require-Command winget 'Install Microsoft App Installer from Microsoft Store'
    Install-Winget uv 'astral-sh.uv'
    Install-Winget 'yt-dlp' 'yt-dlp.yt-dlp'
    Install-Winget deno 'DenoLand.Deno'
    Install-Winget ffmpeg 'yt-dlp.FFmpeg'
    $mpv = Find-Mpv
    if (-not $mpv) {
        Step 'Installing shinchiro.mpv using WinGet'
        & winget install --source winget --exact --id shinchiro.mpv --accept-package-agreements --accept-source-agreements
        Refresh-Path
        $mpv = Find-Mpv
        if (-not $mpv) { throw 'MPV executable could not be located. Add the folder containing mpv.exe to PATH and rerun Setup.bat.' }
    }
    Write-Host "Using MPV: $mpv"
    $cache = Join-Path $ProjectDir '.cache'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $cache 'mpv-path.txt'), $mpv)


    Step 'Installing Python 3.12 and project dependencies'
    Push-Location -LiteralPath $ProjectDir
    try {
        & uv python install 3.12
        if ($LASTEXITCODE -ne 0) { throw 'uv python install 3.12 failed.' }
        & uv sync --locked --extra gui --extra diagnostics
        if ($LASTEXITCODE -ne 0) { throw 'uv sync --locked --extra gui --extra diagnostics failed. Review Python/CUDA package compatibility.' }
    } finally { Pop-Location }

    Refresh-Path
    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        throw 'nvidia-smi.exe is unavailable. Install a current NVIDIA driver, restart Windows, then rerun Setup.bat.'
    }
    & nvidia-smi.exe --query-gpu=name,driver_version,memory.total,memory.free --format=csv
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver query failed. Repair/update the NVIDIA driver before using CUDA llama.cpp.' }

    Step 'Installing pinned CUDA 12.4 llama.cpp runtime for Qwen3-ASR + HY-MT'
    $installRoot = Join-Path $ProjectDir '.runtime'
    $llamaRoot = Join-Path $installRoot 'llama-b12318-cuda12'
    $pathFile = Join-Path $cache 'llama-bin-path.txt'
    New-Item -ItemType Directory -Path $llamaRoot -Force | Out-Null
    $existingServer = Get-ChildItem -LiteralPath $llamaRoot -Recurse -Filter 'llama-server.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $existingServer) {
        $headers = @{ 'User-Agent' = 'VideoTranslate-Qwen3-ASR-HY-MT-Setup'; 'Accept' = 'application/vnd.github+json' }
        $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b12318' -Headers $headers
        $build = $release.assets | Where-Object name -eq 'llama-b12318-bin-win-cuda-12.4-x64.zip'
        $runtime = $release.assets | Where-Object name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip'
        if (-not $build -or -not $runtime) {
            throw 'No matching CUDA 12.4 llama.cpp Windows release found. See https://github.com/ggml-org/llama.cpp/releases'
        }
        $downloadDir = Join-Path $ProjectDir ('.cache\qwen3-asr-hy-mt-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
        try {
            $buildZip = Join-Path $downloadDir 'llama.zip'
            $runtimeZip = Join-Path $downloadDir 'cudart.zip'
            Fetch-Asset $build $buildZip 20281408c618dcd13080bc0b3deed250c413bbe6541421a1c989acf5a376e684
            Fetch-Asset $runtime $runtimeZip 8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6
            Expand-Archive -LiteralPath $buildZip -DestinationPath $llamaRoot -Force
            Expand-Archive -LiteralPath $runtimeZip -DestinationPath $llamaRoot -Force
        } finally { Remove-Item -LiteralPath $downloadDir -Recurse -Force -ErrorAction SilentlyContinue }
        $existingServer = Get-ChildItem -LiteralPath $llamaRoot -Recurse -Filter 'llama-server.exe' -File | Select-Object -First 1
        if (-not $existingServer) { throw 'llama-server.exe missing after extracting official archives.' }
    } else { Write-Host 'CUDA 12.4 llama-server.exe already downloaded.' }
    $binDir = $existingServer.Directory.FullName
    [IO.File]::WriteAllText($pathFile, $binDir)
    $env:Path = "$binDir;$env:Path"
    & $existingServer.FullName --version
    if ($LASTEXITCODE -ne 0) { throw 'llama-server.exe failed to launch. Update the NVIDIA driver or check CUDA runtime DLLs.' }

    Step 'Checking Python environment'
    Push-Location -LiteralPath $ProjectDir
    try {
        & uv run --no-sync python -c 'import numpy, fireredvad, torch, onnxruntime, PySide6, auto_translate; assert torch.version.cuda is None'
        if ($LASTEXITCODE -ne 0) { throw 'Python import check failed.' }
        & uv run --no-sync python -m auto_translate.models
        if ($LASTEXITCODE -ne 0) { throw 'Could not download/verify Qwen3-ASR + HY-MT models. Rerun Setup to resume.' }
        Write-Host 'Python dependencies, Qwen3-ASR Q8_0 and HY-MT Q4_K_M model integrity checks OK'
    } finally { Pop-Location }

    Write-Host "`nSetup complete." -ForegroundColor Green
    Write-Host 'Qwen3-ASR Q8_0 and audio projector ready under models\qwen3-asr; HY-MT Q4_K_M ready under models\hy-mt.'
    Write-Host 'A CUDA-capable NVIDIA driver is required. FireRedVAD runs on CPU.'
    exit 0
} catch {
    Write-Host "`nSETUP ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Rerun Setup.bat after fixing this issue. Existing successful downloads are reused.'
    exit 1
}
