@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%LOCALAPPDATA%\mnist-digit-recognition-venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating the Python environment in a short Windows path...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto setup_error
)

echo Installing or checking project packages...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 goto setup_error

echo Starting Handwritten Digit Recognition...
"%VENV_DIR%\Scripts\python.exe" -m streamlit run "%PROJECT_DIR%app.py"
pause
exit /b

:setup_error
echo.
echo Setup did not finish. Check that Python is installed and try again.
pause
exit /b 1
