@echo off
cd /d "%~dp0runner"
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\Release\AnetRLRunner.exe"


call:run_tau_mode stratified
call:run_tau_mode antithetic
call:run_tau_mode systematic
call:run_tau_mode fixed

pause
exit /b


:run_tau_mode
call:run_exe ^
  app.run_name=run_{t}_dm_iqn32-%1-b256-rr100 ^
  train.seed=1 ^
  DefaultDQNAgent.train_policy.tau_rule.sample_mode=%1 ^
  DefaultDQNAgent.learner.iqn.current_taus.sample_mode=%1 ^
  DefaultDQNAgent.learner.iqn.target_taus.sample_mode=%1
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
