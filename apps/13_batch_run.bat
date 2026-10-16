@echo off

cd /d "%~dp0runner"

SET "BUILD=Release"

:waitprev
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto :waitprev
)
ping -n 16 127.0.0.1 >nul
tasklist /FI "IMAGENAME eq AnetRLRunner*" 2>nul | find /I "AnetRLRunner" >nul
if not errorlevel 1 goto :waitprev

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR1=run.@hard125>run.@munch"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "RT=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@to_100m>run.@batch"
SET "RT50=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@batch"
SET "SE=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@batch"

SET "RT50NZ=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@nz_learning>run.@batch"
SET "EPS0=A2.actor.[train].policy.use_spatial_exploration=false A2.actor.[train].policy.eps_start=0 A2.actor.[train].policy.eps_end=0"
SET "NZSIG=net.block.[AtariHeadFC512].noisy.sigma0=0.87 A2.head_projection.noisy.sigma0=1.2"
SET "BOMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@breakout>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2>metrics.scalar.@noisy"
SET "BOMET0=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@breakout>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2"
SET "NOCLIP=E1.reward_clip=false A3.reward_scaler.constant_scale=0.005"
SET "ELIFE=E1.episodic_life=true"
SET "STICKY0=E1.repeat_action_probability=0"
SET "UQE=A2.actor.[train].policy.policy_type=UQE"
SET "RT30=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_30m>run.@batch"
SET "LOWEPS=A2.actor.[train].policy.eps_start=0.05 A2.actor.[train].policy.eps_end=0.0001 A2.actor.[train].policy.uqe_eps_end=0.0001"
SET "PERHI=A2.learner.per_alpha=0.6 A2.learner.per_beta_start=0.4 A2.learner.per_beta_end=1.0"
SET "BOSS=%STICKY0% %LOWEPS% %NOCLIP% %PERHI%"
SET "N10=A2.learner.n_step=10"
SET "G999=A2.learner.gamma=0.999"

call :use_ws AtariSub-01 Atari.txt
echo === 1. kung_fu_master rt 30m seed 1: no sticky actions only ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=1 %STICKY0% "app.run_name=run_{t}_kung_fu_master_rr1_30m_sticky0_seed1"
echo === 2. kung_fu_master rt 30m seed 1: boss set = no sticky, low eps ladder, raw reward 1/200, per alpha 0.6 ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=1 %BOSS% "app.run_name=run_{t}_kung_fu_master_rr1_30m_boss_seed1"
echo === 3. kung_fu_master rt 30m seed 2: boss set ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=2 %BOSS% "app.run_name=run_{t}_kung_fu_master_rr1_30m_boss_seed2"
echo === 4. kung_fu_master rt 30m seed 1: boss set + n-step 10 ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=1 %BOSS% %N10% "app.run_name=run_{t}_kung_fu_master_rr1_30m_boss_n10_seed1"
echo === 5. kung_fu_master rt 30m seed 1: boss set + uqe train policy ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=1 %BOSS% %UQE% "app.run_name=run_{t}_kung_fu_master_rr1_30m_boss_uqe_seed1"
echo === 6. kung_fu_master rt 30m seed 1: boss set + n-step 10 + gamma 0.999 ===
call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=1 %BOSS% %N10% %G999% "app.run_name=run_{t}_kung_fu_master_rr1_30m_boss_n10_g999_seed1"


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
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
