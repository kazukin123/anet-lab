@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-3rd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RF=run.@rfit"
SET "NET=run.@btrnet>run.@btrsn"
SET "NOSN=workspaces/atari-3rd/runs/run_20260913-072109_rr4_btrnosn_resume30m/agent_close.anet"

echo === 1. ev: btrnosn 50M close eval2 greedy (20m) ===
call :run_exe "run.$=%A5%>run.@munch>run.@btrnet>run.@evalonly>run.@eval2only>run.@greedy_eval>run.@to_50" "A3.auto_load_file=%NOSN%" "app.run_name=run_{t}_ev_btrnosn_50m"
echo === 2. RR1 + BTR net + SN 50M (3.5h) ===
call :run_exe "run.$=%A5%>run.@hard125>run.@munch>%RF%>%NET%>run.@envs64>run.@btreval_r1" "app.run_name=run_{t}_rr1_btrfull_50m"
echo === 3. RR2 + BTR net + SN 50M (5.5h) ===
call :run_exe "run.$=%A5%>run.@hard250>run.@rr2>run.@munch>%RF%>%NET%>run.@envs64>run.@btreval_r2" "app.run_name=run_{t}_rr2_btrfull_50m"
echo === 4. RR4 + BTR net + SN 50M, new eval (10.5h) ===
call :run_exe "run.$=%A5%>run.@hard500>run.@rr4>run.@munch>%RF%>%NET%>run.@envs64>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrfull_r2_50m"
echo === 5. RR4 + Pool4 trunk SN 20M (4h) ===
call :run_exe "run.$=%A5%>run.@hard500>run.@rr4>run.@munch>%RF%>run.@btrtrunkpool4>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunkpool4_20m"
echo === 6. RR4 + Pool3 trunk SN 20M (4h) ===
call :run_exe "run.$=%A5%>run.@hard500>run.@rr4>run.@munch>%RF%>run.@btrtrunkpool3>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunkpool3_20m"

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
%EXE% %* %FIX2%
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
