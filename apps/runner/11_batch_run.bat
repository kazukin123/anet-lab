@echo off
SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
REM SET EXE="bin\Release\AnetRLRunner.exe"

REM call:run_exe RunnerApp.run_name=run_{t}

REM call:run_exe RunnerApp.run_name=run_{t}-00
call:run_exe RunnerApp.run_name=run_{t}-01 A.learner.alpha=1e-3
call:run_exe RunnerApp.run_name=run_{t}-02 A.learner.alpha=1e-4
call:run_exe RunnerApp.run_name=run_{t}-03 A.learner.update_warmup_steps=0
call:run_exe RunnerApp.run_name=run_{t}-04 A.learner.update_warmup_steps=5000
call:run_exe RunnerApp.run_name=run_{t}-05 A.reward_scaler.reference_q_std=3
call:run_exe RunnerApp.run_name=run_{t}-06 A.reward_scaler.reference_q_std=10
call:run_exe RunnerApp.run_name=run_{t}-07 A.reward_scaler.reference_q_std=50


REM call:run_exe RunnerApp.$=RunnerApp.batchrun RunnerApp.use_image_log=true  RunnerApp.use_per_image_log=true  RunnerApp.run_name=run_{t}_ev100v
REM call:run_exe RunnerApp.$=RunnerApp.batchrun  train.seed=
REM call:run_exe RunnerApp.$=RunnerApp.batchrun RunnerApp.run_name=run_20251226-bs=64  train.seed=



pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
