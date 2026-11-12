@echo off
setlocal EnableExtensions DisableDelayedExpansion

for /f "tokens=2 delims=:" %%C in ('chcp') do set "ORIGINAL_CODE_PAGE=%%C"
set "ORIGINAL_CODE_PAGE=%ORIGINAL_CODE_PAGE: =%"
chcp 65001 >nul
for %%I in ("%~dp0..") do set "RUNNER_ROOT=%%~fI"

set "WORKSPACE_INPUT=%~1"
set "WORKSPACE_SOURCE=argument"
if /i "%WORKSPACE_INPUT%"=="--select-if-empty" (
    set "WORKSPACE_INPUT="
    set "WORKSPACE_SOURCE=selection"
    call :select_workspace
    if errorlevel 1 goto :failure
    if defined WORKSPACE_SELECTION_EXIT goto :selection_exit
)
if not defined WORKSPACE_INPUT (
    set "WORKSPACE_SOURCE=last_workspace"
    call :load_last_workspace
)

if not defined WORKSPACE_INPUT (
    set "WORKSPACE_INPUT=_default"
    set "WORKSPACE_SOURCE=default"
)

call :trim WORKSPACE_INPUT
call :validate "%WORKSPACE_INPUT%"
if errorlevel 1 goto :unavailable
call :resolve "%WORKSPACE_INPUT%"
if errorlevel 1 goto :unavailable
goto :success

:unavailable
if /i "%WORKSPACE_SOURCE%"=="last_workspace" (
    echo [WARN] Last workspace is unavailable. Falling back to _default: "%WORKSPACE_INPUT%"
    set "WORKSPACE_INPUT=_default"
    set "WORKSPACE_SOURCE=default"
    call :resolve "_default"
    if not errorlevel 1 goto :success
)
echo [ERROR] Workspace is unavailable: "%WORKSPACE_INPUT%"
goto :failure

:success
if not exist "%WORKSPACE_ROOT%\runs\" (
    echo [ERROR] Workspace runs directory was not found: "%WORKSPACE_ROOT%\runs"
    goto :failure
)
for %%I in ("%WORKSPACE_ROOT%\runs") do set "RUNS_DIR=%%~fI"
call :restore_code_page
endlocal & set "WORKSPACE_INPUT=%WORKSPACE_INPUT%" & set "WORKSPACE_ROOT=%WORKSPACE_ROOT%" & set "RUNS_DIR=%RUNS_DIR%" & set "WORKSPACE_SELECTION_EXIT="
echo WORKSPACE_ROOT=%WORKSPACE_ROOT%
exit /b 0

:selection_exit
call :restore_code_page
endlocal & set "WORKSPACE_SELECTION_EXIT=1"
exit /b 0

:failure
call :restore_code_page
endlocal
exit /b 1

:trim
setlocal EnableDelayedExpansion
set "TAB=	"
for %%V in (%1) do set "VALUE=!%%V!"
for /f "tokens=*" %%A in ("!VALUE!") do set "VALUE=%%A"
:trim_tail
if "!VALUE:~-1!"==" " set "VALUE=!VALUE:~0,-1!" & goto :trim_tail
if "!VALUE:~-1!"=="!TAB!" set "VALUE=!VALUE:~0,-1!" & goto :trim_tail
endlocal & set "%1=%VALUE%"
exit /b 0

:select_workspace
setlocal EnableDelayedExpansion
set "WORKSPACE_COUNT=0"
echo   [0] EXIT
for /f "delims=" %%D in ('dir /b /ad /o:n "%RUNNER_ROOT%\workspaces" 2^>nul') do (
    if exist "%RUNNER_ROOT%\workspaces\%%D\runs\" (
        set /a WORKSPACE_COUNT+=1
        set "WORKSPACE_CANDIDATE_!WORKSPACE_COUNT!=%%D"
        echo   [!WORKSPACE_COUNT!] %%D
    )
)

:select_workspace_prompt
set "WORKSPACE_CHOICE="
set /p "WORKSPACE_CHOICE=Select workspace by number: "
if errorlevel 1 (
    endlocal & set "WORKSPACE_SELECTION_EXIT=1"
    exit /b 0
)
if not defined WORKSPACE_CHOICE (
    goto :invalid_workspace_choice
)
set "WORKSPACE_CHOICE_REMAINDER="
for /f "delims=0123456789" %%A in ("!WORKSPACE_CHOICE!") do set "WORKSPACE_CHOICE_REMAINDER=%%A"
if defined WORKSPACE_CHOICE_REMAINDER (
    goto :invalid_workspace_choice
)
if "!WORKSPACE_CHOICE!"=="0" (
    endlocal & set "WORKSPACE_SELECTION_EXIT=1"
    exit /b 0
)
if !WORKSPACE_CHOICE! lss 1 (
    goto :invalid_workspace_choice
)
if !WORKSPACE_CHOICE! gtr !WORKSPACE_COUNT! (
    goto :invalid_workspace_choice
)
for %%N in (!WORKSPACE_CHOICE!) do set "SELECTED_WORKSPACE=!WORKSPACE_CANDIDATE_%%N!"
endlocal & set "WORKSPACE_INPUT=%SELECTED_WORKSPACE%"
exit /b 0

:invalid_workspace_choice
if !WORKSPACE_COUNT! equ 0 (
    echo [ERROR] Enter 0 to exit.
) else (
    echo [ERROR] Enter 0 to exit or a workspace number from 1 to !WORKSPACE_COUNT!.
)
goto :select_workspace_prompt

:load_last_workspace
if exist "%RUNNER_ROOT%\appdata\" (
    set "LAST_WORKSPACE_FILE=%RUNNER_ROOT%\appdata\last_workspace.txt"
) else if defined APPDATA (
    set "LAST_WORKSPACE_FILE=%APPDATA%\anet-lab\runner\last_workspace.txt"
)
if defined LAST_WORKSPACE_FILE if exist "%LAST_WORKSPACE_FILE%" (
    set /p WORKSPACE_INPUT=<"%LAST_WORKSPACE_FILE%"
)
exit /b 0

:validate
set "VALUE=%~1"
if not defined VALUE exit /b 1
if not "%VALUE:#=%"=="%VALUE%" exit /b 1
if not "%VALUE://=%"=="%VALUE%" exit /b 1
if "%VALUE:~-1%"==";" exit /b 1
set "NORMALIZED_VALUE=%VALUE:/=\%"
if "%NORMALIZED_VALUE:~0,2%"=="\\" exit /b 1
exit /b 0

:resolve
set "VALUE=%~1"
set "IS_ABSOLUTE="
if "%VALUE:~1,2%"==":\" set "IS_ABSOLUTE=1"
if "%VALUE:~1,2%"==":/" set "IS_ABSOLUTE=1"
if defined IS_ABSOLUTE (
    for %%I in ("%VALUE%") do set "WORKSPACE_ROOT=%%~fI"
) else (
    for %%I in ("%RUNNER_ROOT%\workspaces\%VALUE%") do set "WORKSPACE_ROOT=%%~fI"
)
if not exist "%WORKSPACE_ROOT%\" exit /b 1
exit /b 0

:restore_code_page
if defined ORIGINAL_CODE_PAGE chcp %ORIGINAL_CODE_PAGE% >nul
exit /b 0
