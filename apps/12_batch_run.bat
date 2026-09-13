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
SET "RF=run.@rfit"
SET "NET=run.@btrnet>run.@btrsn"
SET "R=workspaces/atari-3rd/runs/run_20260912-074000_rr4_btrfull_munch"
SET "EV0=%A5%>run.@munch>%NET%>run.@evalonly>run.@eval2only>run.@greedy_eval>run.@to_50"

echo === 1. btrfull 50M close eval2 greedy, smoke (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_close.anet" "app.run_name=run_{t}_ev_btrfull_50m"
echo === 2. RR4 + BTR net + SN, trunk restored, 20M (3.5h) ===
call :run_exe "run.$=%A5%>run.@hard500>run.@rr4>run.@munch>%RF%>run.@btrtrunk>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunk_20m"
echo === 3. RR4 + BTR net, no SN, 20M (3.5h) ===
call :run_exe "run.$=%A5%>run.@hard500>run.@rr4>run.@munch>%RF%>run.@btrnet>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrnosn_20m"
echo === 4-E1. btrfull 10m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_10311680.anet" "app.run_name=run_{t}_ev_btrfull_10m"
echo === 4-E2. btrfull 20m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_19976000.anet" "app.run_name=run_{t}_ev_btrfull_20m"
echo === 4-E3. btrfull 30m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_30033664.anet" "app.run_name=run_{t}_ev_btrfull_30m"
echo === 4-E4. btrfull 37m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_37069888.anet" "app.run_name=run_{t}_ev_btrfull_37m"
echo === 4-E5. btrfull 40m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_40349824.anet" "app.run_name=run_{t}_ev_btrfull_40m"
echo === 4-E6. btrfull 43m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_43399808.anet" "app.run_name=run_{t}_ev_btrfull_43m"
echo === 4-E7. btrfull 45m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_45030848.anet" "app.run_name=run_{t}_ev_btrfull_45m"
echo === 4-E8. btrfull 47m eval2 greedy (15m) ===
call :run_exe "run.$=%EV0%" "A3.auto_load_file=%R%/agent_47005376.anet" "app.run_name=run_{t}_ev_btrfull_47m"
echo === 5. RR1 + BTR structure 50M (3h) ===
call :run_exe "run.$=%A5%>run.@hard125>run.@munch>%RF%>%NET%>run.@btreval_r1" "app.run_name=run_{t}_rr1_btrnet_munch"
echo === 6. RR2 + BTR structure 50M (5h) ===
call :run_exe "run.$=%A5%>run.@hard250>run.@rr2>run.@munch>%RF%>%NET%>run.@btreval_r2" "app.run_name=run_{t}_rr2_btrnet_munch"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, 0 FAILED ===
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
