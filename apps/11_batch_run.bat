@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-2nd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "BK=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex"
SET "EV=run.@evalN10"
SET "BM=run.@breakout_metrics"
SET "BR=app.$=app.batchrun>P1"
SET "S400K=@vars.max_exp_step=400000"
SET "X400K=app.batchrun.exp_exit_step=400000"

echo === 0. PRD067 ON smoke - risk biased soft target, retry with UQE train policy ===
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "A3.use_optimistic_target=true" "A3.train_policy.policy_type=UQE" "app.run_name=run_{t}_tmp_smoke_067_target_risk_${E1.game}"

echo === 1. ARM - hard125 + Munchausen(target), 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>run.@m_on>%EV%"

echo === 2. CONTROL - hard125 + use_double_dqn=false, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>%BM%>run.@m_ctrl>%EV%"

echo === 3. ARM r2 - hard125 + Munchausen(target) replicate, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>run.@m_on>%EV%"

echo === 4. BASELINE r4 - hard125 reference replicate, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>%BM%>%EV%"

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
