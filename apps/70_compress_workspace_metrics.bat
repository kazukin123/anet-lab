@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "WORKSPACE_ARGUMENT="
set "DRY_RUN_ARGUMENT="
set "NO_PAUSE="
set "INTERACTIVE_MODE="

:parse_arguments
if "%~1"=="" goto :arguments_parsed
call :parse_argument "%~1"
if errorlevel 1 goto :finish
shift /1
goto :parse_arguments

:parse_argument
if /i "%~1"=="--dry-run" (
    set "DRY_RUN_ARGUMENT=--dry-run"
    exit /b 0
)
if /i "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    exit /b 0
)
set "PARSED_ARGUMENT=%~1"
setlocal EnableDelayedExpansion
if "!PARSED_ARGUMENT:~0,2!"=="--" (
    endlocal
    echo [ERROR] Unknown option: "%~1"
    set "RESULT=1"
    exit /b 1
)
endlocal
if defined WORKSPACE_ARGUMENT (
    echo [ERROR] Specify at most one workspace.
    set "RESULT=1"
    exit /b 1
)
set "WORKSPACE_ARGUMENT=%~1"
exit /b 0

:arguments_parsed
if not defined WORKSPACE_ARGUMENT set "INTERACTIVE_MODE=1"

:interactive_workspace_loop
set "WORKSPACE_SELECTION_EXIT="
if defined WORKSPACE_ARGUMENT (
    echo [INFO] Resolving workspace: "%WORKSPACE_ARGUMENT%"
    call "%~dp0runner\tools\resolve_workspace.bat" "%WORKSPACE_ARGUMENT%"
) else (
    echo.
    echo Available workspaces:
    call "%~dp0runner\tools\resolve_workspace.bat" --select-if-empty
)
if errorlevel 1 (
    set "RESULT=1"
    goto :finish
)
if defined WORKSPACE_SELECTION_EXIT (
    set "RESULT=0"
    set "NO_PAUSE=1"
    goto :finish
)

set "VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python virtual environment was not found: "%VENV_PYTHON%"
    echo [INFO] Create it from the repository root: C:\Python314\python.exe -m venv .venv
    set "RESULT=1"
    goto :finish
)

"%VENV_PYTHON%" "%~dp0runner\tools\compress_workspace_metrics.py" --workspace-root "%WORKSPACE_ROOT%" %DRY_RUN_ARGUMENT%
set "RESULT=%ERRORLEVEL%"
if defined INTERACTIVE_MODE (
    if not defined NO_PAUSE pause
    goto :interactive_workspace_loop
)

:finish
if not defined RESULT set "RESULT=1"
if not defined NO_PAUSE pause
exit /b %RESULT%
