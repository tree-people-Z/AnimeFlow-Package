@echo off
setlocal
set "ANIMEFLOW_ROOT=%~dp0"
cd /d "%ANIMEFLOW_ROOT%"
echo Starting AnimeFlow...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ANIMEFLOW_ROOT%start.ps1"
set "ANIMEFLOW_EXIT=%ERRORLEVEL%"
if "%ANIMEFLOW_EXIT%"=="2" (
    echo AnimeFlow is running. Use stop.bat to stop the background application.
    pause
    set "ANIMEFLOW_EXIT=0"
) else if not "%ANIMEFLOW_EXIT%"=="0" (
    echo.
    echo AnimeFlow failed to start. See the error above.
    pause
)
endlocal & exit /b %ANIMEFLOW_EXIT%
