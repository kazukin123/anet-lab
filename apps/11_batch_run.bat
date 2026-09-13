@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-3rd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "BK=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit"
SET "NET=run.@btrnet>run.@btrsn"
SET "R=workspaces/atari-3rd/runs/run_20260912-074000_rr4_btrfull_munch"
SET "NOSN=workspaces/atari-3rd/runs/run_20260913-025232_rr4_btrnosn_20m/agent_close.anet"
SET "EV0=%A5%>run.@munch>%NET%>run.@evalonly>run.@eval2only>run.@greedy_eval>run.@to_50"

echo === 0a. wiring: btrtrunknosn 400k (7m) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrtrunknosn>run.@envs64>run.@to_400k>run.@btreval_r4" "app.run_name=run_{t}_tmp_wiring_btrtrunknosn"
echo === 0b. wiring: btrnosn resume load 400k (7m) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@envs64>run.@to_400k>run.@btreval_r4" "A3.auto_load_file=%NOSN%" "app.run_name=run_{t}_tmp_wiring_resume"
echo === 1. btrnosn resume +30M, cumulative 50M (5.6h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@envs64>run.@to_30m>run.@btreval_r4" "A3.auto_load_file=%NOSN%" "app.run_name=run_{t}_rr4_btrnosn_resume30m"
echo === 2-E1. btrfull 10m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_10311680.anet" "app.run_name=run_{t}_ev_btrfull_10m"
echo === 2-E2. btrfull 20m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_19976000.anet" "app.run_name=run_{t}_ev_btrfull_20m"
echo === 2-E3. btrfull 30m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_30033664.anet" "app.run_name=run_{t}_ev_btrfull_30m"
echo === 2-E4. btrfull 37m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_37069888.anet" "app.run_name=run_{t}_ev_btrfull_37m"
echo === 2-E5. btrfull 40m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_40349824.anet" "app.run_name=run_{t}_ev_btrfull_40m"
echo === 2-E6. btrfull 43m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_43399808.anet" "app.run_name=run_{t}_ev_btrfull_43m"
echo === 2-E7. btrfull 45m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_45030848.anet" "app.run_name=run_{t}_ev_btrfull_45m"
echo === 2-E8. btrfull 47m eval2 greedy (18m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_47005376.anet" "app.run_name=run_{t}_ev_btrfull_47m"
echo === 3. RR4 + Pool6 + trunk, no SN, 20M (3.7h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrtrunknosn>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunknosn_20m"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %*
%EXE% %* %BK% %FIX2%
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
