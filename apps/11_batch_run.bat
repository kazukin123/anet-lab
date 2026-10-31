@echo off
title %~nx0

cd /d "%~dp0runner"

SET "BUILD=Release"
if not exist "bin\%BUILD%\AnetRLRunner_t106frz.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner_t106frz.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "SKIPMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2>metrics.scalar.@noisy>metrics.scalar.@skip"
SET "BTRENV=run.train.num_envs=64 A2.learner.replay_capacity=1048576 A2.actor.[train].sync_interval=1"
SET "BTREPSPH=A2.actor.[train].policy.spatial.enabled=false A2.actor.[train].policy.epsilon.type=phased "A2.actor.[train].policy.epsilon.phases=decay off" A2.actor.[train].policy.epsilon.phase.[decay].type=exponential A2.actor.[train].policy.epsilon.phase.[decay].start=1.0 A2.actor.[train].policy.epsilon.phase.[decay].end=0.01 A2.actor.[train].policy.epsilon.phase.[decay].time_constant=2000000 A2.actor.[train].policy.epsilon.phase.[decay].steps=25000000 A2.actor.[train].policy.epsilon.phase.[off].type=constant A2.actor.[train].policy.epsilon.phase.[off].value=0 A2.actor.[train].policy.epsilon.phase.[off].steps=1"
SET "BTRNZ=net.block.[AtariHeadFC512].init.mode=default net.block.[AtariHeadFC512].noisy.sigma0=0.5 A2.head.init.mode=default A2.head.projection.noisy.sigma0=0.5"
SET "ALNINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32SN].init1.mode=default net.block.[AtariImpalaRes32SN].init2.mode=default net.block.[AtariImpalaRes64SN].init1.mode=default net.block.[AtariImpalaRes64SN].init2.mode=default net.block.[AtariIQNTauProj2304].init.mode=default"
SET "ALNM=A2.learner.munchausen.log_policy_mode=online"
SET "MERGE=DefaultDQNAgent.learner.skip.feature_gradient=merge"
SET "SKGS=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@pixel_area>run.@skip>run.@batch"
SET "C30=DefaultDQNAgent.learner.grad_clip_tau=30"

call :use_ws Atari5-01 Atari.txt
echo === 1. PRD103 stage 1 qbert 50m seed 1: TempoRL merge, gradient clip threshold 30 instead of 10, plus single greedy eval ===
call :find_done a5_qb_aln_t98c30_s1
if not defined DONE_DIR call :run_exe "%SKGS%" "%SKIPMET%" E1.game=qbert run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %MERGE% %C30% "app.run_name=run_{t}_a5_qb_aln_t98c30_s1"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:use_ws
SET "WS=--workspace %~1"
SET "RUNS=workspaces\%~1\runs"
if exist "workspaces\%~1\config\_main.txt" exit /b 0
mkdir "workspaces\%~1\config" 2>nul
>"workspaces\%~1\config\_main.txt" echo $include ^<%~2^>
exit /b 0

:find_done
SET "DONE_DIR="
for /f "delims=" %%d in ('dir /b /ad "%RUNS%\run_*_%~1" 2^>nul') do (
  findstr /m /c:"Stop training requested" "%RUNS%\%%d\*.log" >nul 2>&1 && SET "DONE_DIR=%%d"
)
if defined DONE_DIR echo %DATE% %TIME% SKIP %~1 already finished in %DONE_DIR%
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
echo *** bin\%BUILD%\AnetRLRunner_t106frz.exe not found or copy failed. Nothing was run.
pause
exit /b 1
