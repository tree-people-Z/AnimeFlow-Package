@echo off
setlocal
set "ANIMEFLOW_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ANIMEFLOW_ROOT%stop.ps1"
pause
endlocal
