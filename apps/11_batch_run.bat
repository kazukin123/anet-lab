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

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR1=run.@hard125>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "EV=run.@eval2ch_r1>run.@eval2ch_r1_50m"
SET "TP=%A5%>%RR1%>%RF%>run.@cap4m>%EV%>run.@batch"

SET "WS=--workspace atari5-01"
echo === 1. wiring: a5 rr1 qbert cap4m 400k (RAM check) ===
call :run_exe "run.$=%TP%>run.@to_400k" E1.game=qbert "app.run_name=run_{t}_tmp_wiring"
echo === 2. a5 name_this_game rr1 cap4m 50M ===
call :run_exe "run.$=%TP%" E1.game=name_this_game "app.run_name=run_{t}_a5_name_this_game_rr1_cap4m_50m"
echo === 3. a5 phoenix rr1 cap4m 50M ===
call :run_exe "run.$=%TP%" E1.game=phoenix "app.run_name=run_{t}_a5_phoenix_rr1_cap4m_50m"
echo === 4. a5 qbert rr1 cap4m 50M ===
call :run_exe "run.$=%TP%" E1.game=qbert "app.run_name=run_{t}_a5_qbert_rr1_cap4m_50m"

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
%RUNNER% %WS% %*
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
