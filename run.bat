@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found on this machine.
        echo Install Python 3.12 from https://www.python.org/downloads/ ^(check
        echo "Add python.exe to PATH" during install^), then re-run this script.
        pause
        exit /b 1
    )

    py -3.12 -c "" >nul 2>nul
    if errorlevel 1 (
        echo Python 3.12 is required but was not found via the 'py' launcher.
        echo Install it from https://www.python.org/downloads/, then re-run this script.
        pause
        exit /b 1
    )

    echo Setting up virtual environment - this only happens once...
    py -3.12 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m pip install -e .
    if not exist ".venv\Scripts\python.exe" (
        echo Setup failed - see the errors above.
        pause
        exit /b 1
    )
)

set target=%1
if "%target%"=="" (
    echo.
    echo   1^) Web app  - clinical UI, odontogram + report ^(recommended^)
    echo   2^) Gradio   - developer tool, raw model controls
    echo.
    set /p target="Choose [1/2]: "
)

if "%target%"=="2" goto gradio
if /i "%target%"=="gradio" goto gradio

echo Starting web app on http://127.0.0.1:8000 ...
.venv\Scripts\python.exe -m uvicorn web.backend.main:app --port 8000
goto :eof

:gradio
echo Starting Gradio dev tool ...
.venv\Scripts\python.exe app\app.py
