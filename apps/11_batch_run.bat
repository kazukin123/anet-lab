@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner_ab.exe" 2>nul | find /I "AnetRLRunner_ab.exe" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit"
SET "NET12=run.@btrnet>run.@btrsn12"
SET "NETP4=run.@btrnet>run.@btrpool4"

SET "WS=--workspace atari-04"
echo === 1. wiring: fixed binary sn12 cap512k 100k (2m) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET12%>run.@cap512k>run.@pl_check"
echo === 2. post-fix: RR4 sn12 envs128 cap512k lane 4096 50M (7h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET12%>run.@cap512k>run.@eval2ch" "app.run_name=run_{t}_rr4_btrsn12_envs128_cap512k_50m_fix"
echo === 3. pool: RR4 btrpool4 envs128 cap2M 50M (8h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NETP4%>run.@cap2m>run.@eval2ch" "app.run_name=run_{t}_rr4_btrpool4_envs128_cap2m_50m"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %WS% %*
%RUNNER% %WS% %* %FIX2%
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_succeeded
SET /A SUCCEEDED_RUNS+=1
echo   %DATE% %TIME% END   %*
exit /b 0

:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
