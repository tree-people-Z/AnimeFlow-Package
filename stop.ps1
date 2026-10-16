$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $root "app.py"
$found = $false
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
    if ($_.CommandLine -and $_.CommandLine.Contains($appPath)) {
        $found = $true
        & taskkill.exe /PID $_.ProcessId /T /F
    }
}
if ($found) { Write-Host "AnimeFlow stopped." } else { Write-Host "AnimeFlow is not running in this package." }
