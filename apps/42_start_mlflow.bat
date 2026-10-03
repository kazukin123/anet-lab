@echo off

setlocal

call "%~dp0runner\tools\resolve_workspace.bat" "%~1"
if errorlevel 1 exit /b 1

set VENV_PYTHON=..\..\.venv\Scripts\python.exe
set MLFLOW_EXE=..\..\.venv\Scripts\mlflow.exe
set MLFLOW_REQUIREMENTS=..\..\viewers\metrics-tools\requirements.txt

cd /d "%~dp0runner"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python virtual environment was not found: "%VENV_PYTHON%"
    echo [INFO] Create it from the repository root: C:\Python314\python.exe -m venv .venv
    exit /b 1
)

"%VENV_PYTHON%" -c "import mlflow, sys; sys.exit(0 if mlflow.__version__ == '3.13.0' else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required MLflow version 3.13.0 is not installed in: "%VENV_PYTHON%"
    echo [INFO] Install it with: "%VENV_PYTHON%" -m pip install -r "%MLFLOW_REQUIREMENTS%"
    exit /b 1
)

set "MLFLOW_DB=%RUNS_DIR%\mlflow.db"
set "MLFLOW_DB_URI=%MLFLOW_DB:\=/%"
start "" "%MLFLOW_EXE%" server --backend-store-uri "sqlite:///%MLFLOW_DB_URI%"
timeout /t 16 /nobreak >nul
start "" "http://localhost:5000/"

endlocal
