@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_base.exe" goto :no_base
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET "NEW=bin\%BUILD%\AnetRLRunner_ab.exe"
SET "BASE=bin\%BUILD%\AnetRLRunner_base.exe"
SET EXE="%NEW%" --workspace atari-2nd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "BK=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex"
SET "EV=run.@evalN10"
SET "BM=run.@breakout_metrics"
SET "S400K=@vars.max_exp_step=400000"
SET "X400K=app.batchrun.exp_exit_step=400000"
SET "BR=app.$=app.batchrun>P1"

echo === 0. PRD067 OFF equivalence - base commit vs new binary, deterministic ===
SET "BK=backend.$=backend.@deterministic"
SET EXE="%BASE%" --workspace atari-2nd
call:run_exe "run.$=run.@v5_iqn_impala_x2" "%BR%" "%S400K%" "%X400K%" "app.run_name=run_{t}_tmp_off_base"
SET EXE="%NEW%" --workspace atari-2nd
call:run_exe "run.$=run.@v5_iqn_impala_x2" "%BR%" "%S400K%" "%X400K%" "app.run_name=run_{t}_tmp_off_new"
SET "BK=backend.$=backend.@non-deterministic"

echo === 1. PRD067 ON smoke x5 - 400k each ===
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "app.run_name=run_{t}_tmp_smoke_067_target_${E1.game}"
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "A3.learner.munchausen.log_policy_mode=online" "app.run_name=run_{t}_tmp_smoke_067_online_${E1.game}"
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "A3.learner.munchausen.log_policy_mode=online_reuse" "app.run_name=run_{t}_tmp_smoke_067_online_reuse_${E1.game}"
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "A3.learner.per_initial_priority_mode=actor_approx" "app.run_name=run_{t}_tmp_smoke_067_target_actor_approx_${E1.game}"
call:run_exe "run.$=run.@munchausen" "%BR%" "%S400K%" "%X400K%" "A3.use_optimistic_target=true" "app.run_name=run_{t}_tmp_smoke_067_target_risk_${E1.game}"

echo === 2. wiring check - m_ctrl / m_on on hard125 base, 400k each ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>%BM%>run.@m_ctrl>%EV%>run.@to_400k" "app.run_name=run_{t}_tmp_wiring_mu0"
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>run.@m_on>%EV%>run.@to_400k" "app.run_name=run_{t}_tmp_wiring_mu1"

echo === 3. CONTROL - hard125 + use_double_dqn=false, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>%BM%>run.@m_ctrl>%EV%"

echo === 4. ARM - hard125 + Munchausen(target), 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125>run.@m_on>%EV%"

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


:no_base
echo *** bin\%BUILD%\AnetRLRunner_base.exe not found. Nothing was run.
pause
exit /b 1

:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
