@echo off

cd /d "%~dp0runner"

SET "BUILD=Release"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)
REM Another bat may start its next stage within a second; confirm that no runner appears for 15 s.
ping -n 16 127.0.0.1 >nul
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 goto :waitprev

if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET RUNNER="bin\%BUILD%\AnetRLRunner.exe"
SET "WS=--workspace Atari5-01"
SET "RUNS=workspaces\Atari5-01\runs"
SET "BTRSE=run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@pixel_area>run.@batch"
SET "BTRENV=run.train.num_envs=64 A2.learner.replay_capacity=1048576 A2.actor.[train].sync_interval=1"
SET "BTRNZ=net.block.[AtariHeadFC512].init.mode=default net.block.[AtariHeadFC512].noisy.sigma0=0.5 A2.head.init.mode=default A2.head.projection.noisy.sigma0=0.5"
SET "BTREPS=A2.actor.[train].policy.spatial.enabled=false A2.actor.[train].policy.epsilon.type=phased "A2.actor.[train].policy.epsilon.phases=decay off" A2.actor.[train].policy.epsilon.phase.[decay].type=exponential A2.actor.[train].policy.epsilon.phase.[decay].start=1.0 A2.actor.[train].policy.epsilon.phase.[decay].end=0.01 A2.actor.[train].policy.epsilon.phase.[decay].time_constant=2000000 A2.actor.[train].policy.epsilon.phase.[decay].steps=25000000 A2.actor.[train].policy.epsilon.phase.[off].type=constant A2.actor.[train].policy.epsilon.phase.[off].value=0 A2.actor.[train].policy.epsilon.phase.[off].steps=1"
SET "ALNINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32SN].init1.mode=default net.block.[AtariImpalaRes32SN].init2.mode=default net.block.[AtariImpalaRes64SN].init1.mode=default net.block.[AtariImpalaRes64SN].init2.mode=default net.block.[AtariIQNTauProj2304].init.mode=default"
SET "ALNM=A2.learner.munchausen.log_policy_mode=online"
SET "ALNFP=A2.learner.use_amp=false A2.learner.use_amp_bf16=false"
SET "B10M=app.batchrun.exp_exit_step=10000000"
SET "RT50S=run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard125>run.@munch>run.@rfit>run.@a5_metrics>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@batch"
SET "RTINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32].init1.mode=default net.block.[AtariImpalaRes64].init1.mode=default net.block.[AtariLinear512].init.mode=default net.block.[AtariIQNTauProj512].init.mode=default"
SET "WLAD=A2.actor.[train].policy.spatial.enabled=true A2.actor.[train].policy.spatial.epsilon.first=0.4 A2.actor.[train].policy.spatial.epsilon.last=0"
:wait_disk_a0
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_a0
)
echo === 1. qbert realtime profile 10M seed 1: init default for the layers before LayerNorm (residual conv2 stays zero) ===
call :run_exe "%RT50S%" E1.game=qbert run.seed=1 %RTINIT% %B10M% "app.run_name=run_{t}_a5_qb_rt_aln_s1"
:wait_disk_a1
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_a1
)
echo === 2. qbert 50M seed 1: btreps + init default + Munchausen log pi online, BF16 kept ===
call :run_exe "%BTRSE%" E1.game=qbert run.seed=1 %BTRENV% %BTREPS% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_qb_aln_btreps_s1"
:wait_disk_a2
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_a2
)
echo === 3. battle_zone 50M seed 1: arm 2 setup on battle_zone (BTR epsilon, init default, Munchausen log pi online) ===
call :run_exe "%BTRSE%" E1.game=battle_zone run.seed=1 %BTRENV% %BTREPS% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_bz_aln_btreps_s1"

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

:find_ckpt
SET "CKPT="
SET "CKPT_DIR="
if not "%RUN_EXIT_CODE%"=="0" exit /b 0
for /f "delims=" %%d in ('dir /b /ad /o-n "%RUNS%\%~1" 2^>nul') do if not defined CKPT_DIR SET "CKPT_DIR=%%d"
if not defined CKPT_DIR goto :find_ckpt_missing
if exist "%RUNS%\%CKPT_DIR%\agent_close.anet" SET "CKPT=%RUNS:\=/%/%CKPT_DIR%/agent_close.anet"
if defined CKPT exit /b 0
:find_ckpt_missing
echo %DATE% %TIME% [ERROR] agent_close.anet not found: %RUNS%\%~1 newest=%CKPT_DIR%
SET /A FAILED_RUNS+=1
exit /b 0

:no_exe
echo *** bin\%BUILD%\AnetRLRunner_ab.exe or AnetRLRunner.exe not found. Nothing was run.
pause
exit /b 1
