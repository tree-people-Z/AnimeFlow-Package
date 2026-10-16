$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$url = "http://127.0.0.1:7860"

# Reuse a running AnimeFlow instance before loading Python or checking packages.
$existing = $null
try {
    $existing = Invoke-RestMethod "$url/config" -TimeoutSec 3
} catch { $existing = $null }
if ($null -ne $existing) {
    # Windows PowerShell 5.1 cannot parse Gradio's config JSON with empty keys.
    if ($existing -is [string]) {
        $isAnimeFlow = $existing -match '<h1>AnimeFlow</h1>'
    } else {
        $isAnimeFlow = @($existing.components | Where-Object {
            $_.type -eq "html" -and $_.props.value -match '<h1>AnimeFlow</h1>'
        }).Count -gt 0
    }
    if ($isAnimeFlow) {
        Write-Host "AnimeFlow is already running: $url"
        if ($env:ANIMEFLOW_NO_BROWSER -ne "1") { Start-Process $url }
        exit 2
    }
    Write-Host "Port 7860 is used by another application. Close that application and try again."
    exit 1
}

# Validate local packages without contacting pip or downloading anything.
$probe = @'
import importlib.util, importlib.metadata as m, json, sys
names = ['gradio', 'PIL', 'numpy', 'ncnn', 'torch', 'diffusers', 'transformers', 'accelerate', 'peft']
missing = [name for name in names if importlib.util.find_spec(name) is None]
if missing:
    print('Missing local packages: ' + ', '.join(missing)); sys.exit(1)
v = tuple(int(x) for x in m.version('gradio').split('.')[:2])
if not (v >= (6, 30) and v < (7, 0)):
    print('Gradio 6.30.x or newer 6.x is required; found ' + m.version('gradio')); sys.exit(1)
import torch
print(json.dumps({'python': sys.executable, 'cuda': torch.cuda.is_available()}))
'@

function Test-Runtime($executable, $prefixArgs) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $executable @prefixArgs -c $probe 2>$null
        if ($LASTEXITCODE -eq 0) {
            return ($output | Select-Object -Last 1 | ConvertFrom-Json)
        }
        Write-Host "$executable : $output"
    } catch { }
    finally { $ErrorActionPreference = $savedPreference }
    return $null
}

# An integrated package must run from its OWN bundled interpreter, so that it
# works on a machine with no Python installed at all.  Preference order:
#   1. runtime\python-embed  - the embedded interpreter shipped in this package
#   2. .venv                 - a development virtualenv, if one exists
#   3. a system Python       - last resort for running this tree as a checkout
$candidates = @(
    @{ Path = Join-Path $root "runtime\python-embed\python.exe"; Label = "整合包内置解释器" },
    @{ Path = Join-Path $root ".venv\Scripts\python.exe";        Label = "开发用 .venv" },
    @{ Path = "py -3";                                           Label = "系统 Python（回退）" }
)

$runtime = $null
$runtimeLabel = ""
foreach ($candidate in $candidates) {
    if ($candidate.Path -eq "py -3") {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) { continue }
        $probed = Test-Runtime "py" @("-3")
    } else {
        if (-not (Test-Path -LiteralPath $candidate.Path)) { continue }
        $probed = Test-Runtime $candidate.Path @()
    }
    if ($null -ne $probed) {
        $runtime = $probed
        $runtimeLabel = $candidate.Label
        break
    }
}

if ($null -eq $runtime) {
    Write-Host "No complete local Python environment was found."
    Write-Host "Expected the bundled interpreter at runtime\python-embed\python.exe."
    Write-Host "The package may be incomplete or damaged."
    exit 1
}
Write-Host "Python: $($runtime.python)  [$runtimeLabel]"
Write-Host "CUDA available: $($runtime.cuda)"
if (-not $runtime.cuda) {
    Write-Host "WARNING: CUDA is not available. GMFSS interpolation and ToonOut cutout require CUDA."
}
Write-Host "Starting AnimeFlow at $url"
$logDir = Join-Path $root "work"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "startup.log"
# Use Windows PowerShell to open the browser after the embedded runtime has
# started listening.  This is deliberately outside the embedded Python process:
# it behaves the same as the working development launcher on Windows.
if ($env:ANIMEFLOW_NO_BROWSER -ne "1") {
    $browserScript = @"
`$url = '$url'
for (`$i = 0; `$i -lt 60; `$i++) {
    try {
        Invoke-WebRequest -UseBasicParsing (`$url + '/config') -TimeoutSec 1 | Out-Null
        Start-Process `$url
        break
    } catch { Start-Sleep -Milliseconds 250 }
}
"@
    Start-Process -WindowStyle Hidden -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $browserScript
    ) | Out-Null
}
$ErrorActionPreference = "Continue"
$env:ANIMEFLOW_PARENT_PID = "$PID"
& $runtime.python -u (Join-Path $root "launcher.py")
$applicationExit = $LASTEXITCODE
if ($applicationExit -ne 0) { Write-Host "Startup failed. Log: $logPath" }
exit $applicationExit
