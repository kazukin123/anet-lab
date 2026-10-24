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
SET "NZSIG=net.block.[AtariHeadFC512].noisy.sigma0=0.87 A2.head.projection.noisy.sigma0=1.2"
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
SET "UQB=%STICKY0% %NOCLIP% %PERHI% %UQE%"
SET "UQHI=A2.actor.[train].policy.uqe_eps_start=0.4 A2.actor.[train].policy.uqe_eps_end=0.01"
SET "UQLO=A2.actor.[train].policy.uqe_eps_start=0.05 A2.actor.[train].policy.uqe_eps_end=0.0001"
SET "UQ99=A2.actor.[train].policy.uqe_tau_start=0.99 A2.actor.[train].policy.uqe_tau_end=0.9"
SET "OPT=A3.use_optimistic_target=true"
SET "RT10=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_10m>run.@batch"
SET "B1K=A2.learner.replay_batch_size=1024"
SET "RT10V=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_10m>run.@vitva>run.@batch"
SET "SENZ=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@batch"
SET "NZSIGSE=net.block.[AtariHeadFC512].noisy.sigma0=1.1 A2.head.projection.noisy.sigma0=1.2"
SET "BTRENV=run.train.num_envs=64 A2.learner.replay_capacity=1048576 A2.actor.[train].sync_interval.value=1"
SET "BTREPS=A2.actor.[train].policy.use_spatial_exploration=false A2.actor.[train].policy.eps_start=1.0 A2.actor.[train].policy.eps_end=0.01 A2.actor.[train].policy.eps_decay_steps=4000000"
SET "BTRNZ=net.block.[AtariHeadFC512].init.mode=default net.block.[AtariHeadFC512].noisy.sigma0=0.5 A2.head.init.mode=default A2.head.projection.noisy.sigma0=0.5"

call :use_ws AtariSub-01 Atari.txt
echo === 1. kung_fu_master rt 10m seed 4: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s04
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=4 "app.run_name=run_{t}_kfm_10m_s04"
echo === 2. kung_fu_master rt 10m seed 5: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s05
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=5 "app.run_name=run_{t}_kfm_10m_s05"
echo === 3. kung_fu_master rt 10m seed 6: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s06
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=6 "app.run_name=run_{t}_kfm_10m_s06"
echo === 4. kung_fu_master rt 10m seed 7: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s07
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=7 "app.run_name=run_{t}_kfm_10m_s07"
echo === 5. kung_fu_master rt 10m seed 8: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s08
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=8 "app.run_name=run_{t}_kfm_10m_s08"
echo === 6. kung_fu_master rt 10m seed 9: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s09
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=9 "app.run_name=run_{t}_kfm_10m_s09"
echo === 7. kung_fu_master rt 10m seed 10: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s10
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=10 "app.run_name=run_{t}_kfm_10m_s10"
echo === 8. kung_fu_master rt 10m seed 11: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s11
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=11 "app.run_name=run_{t}_kfm_10m_s11"
echo === 9. kung_fu_master rt 10m seed 12: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s12
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=12 "app.run_name=run_{t}_kfm_10m_s12"
echo === 10. kung_fu_master rt 10m seed 13: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s13
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=13 "app.run_name=run_{t}_kfm_10m_s13"
echo === 11. kung_fu_master rt 10m seed 14: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s14
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=14 "app.run_name=run_{t}_kfm_10m_s14"
echo === 12. kung_fu_master rt 10m seed 15: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s15
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=15 "app.run_name=run_{t}_kfm_10m_s15"
echo === 13. kung_fu_master rt 10m seed 16: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s16
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=16 "app.run_name=run_{t}_kfm_10m_s16"
echo === 14. kung_fu_master rt 10m seed 17: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s17
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=17 "app.run_name=run_{t}_kfm_10m_s17"
echo === 15. kung_fu_master rt 10m seed 18: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s18
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=18 "app.run_name=run_{t}_kfm_10m_s18"
echo === 16. kung_fu_master rt 10m seed 19: default settings, does the boss route ignite early ===
call :find_done kfm_10m_s19
if not defined DONE_DIR call :run_exe "%RT10%" E1.game=kung_fu_master run.seed=19 "app.run_name=run_{t}_kfm_10m_s19"
echo === 17. kung_fu_master rt 30m seed 4: batch 1024, four times the experience per update ===
call :find_done kfm_b1k_s04
if not defined DONE_DIR call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=4 %B1K% "app.run_name=run_{t}_kfm_b1k_s04"
echo === 18. kung_fu_master rt 30m seed 5: batch 1024, four times the experience per update ===
call :find_done kfm_b1k_s05
if not defined DONE_DIR call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=5 %B1K% "app.run_name=run_{t}_kfm_b1k_s05"
echo === 19. kung_fu_master rt 30m seed 6: batch 1024, four times the experience per update ===
call :find_done kfm_b1k_s06
if not defined DONE_DIR call :run_exe "%RT30%" E1.game=kung_fu_master run.seed=6 %B1K% "app.run_name=run_{t}_kfm_b1k_s06"
echo === 20. kung_fu_master rt 10m seed 4: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s04
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=4 "app.run_name=run_{t}_kfm_vit_s04"
echo === 21. kung_fu_master rt 10m seed 5: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s05
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=5 "app.run_name=run_{t}_kfm_vit_s05"
echo === 22. kung_fu_master rt 10m seed 6: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s06
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=6 "app.run_name=run_{t}_kfm_vit_s06"
echo === 23. kung_fu_master rt 10m seed 7: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s07
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=7 "app.run_name=run_{t}_kfm_vit_s07"
echo === 24. kung_fu_master rt 10m seed 8: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s08
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=8 "app.run_name=run_{t}_kfm_vit_s08"
echo === 25. kung_fu_master rt 10m seed 9: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s09
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=9 "app.run_name=run_{t}_kfm_vit_s09"
echo === 26. kung_fu_master rt 10m seed 10: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s10
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=10 "app.run_name=run_{t}_kfm_vit_s10"
echo === 27. kung_fu_master rt 10m seed 11: impala plus vit trunk, does the boss fight improve ===
call :find_done kfm_vit_s11
if not defined DONE_DIR call :run_exe "%RT10V%" E1.game=kung_fu_master run.seed=11 "app.run_name=run_{t}_kfm_vit_s11"
echo === 28. battle_zone se 50m seed 1: noisynet on the sample efficiency profile, epsilon ladder kept ===
call :find_done a5_bz_se_nz_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=battle_zone run.seed=1 %NZSIGSE% "app.run_name=run_{t}_a5_bz_se_nz_s1"
echo === 29. qbert se 50m seed 1: noisynet on the sample efficiency profile, epsilon ladder kept ===
call :find_done a5_qb_se_nz_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=qbert run.seed=1 %NZSIGSE% "app.run_name=run_{t}_a5_qb_se_nz_s1"
echo === 30. double_dunk se 50m seed 1: noisynet on the sample efficiency profile, epsilon ladder kept ===
call :find_done a5_dd_se_nz_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=double_dunk run.seed=1 %NZSIGSE% "app.run_name=run_{t}_a5_dd_se_nz_s1"
echo === 31. qbert se 50m seed 1: aligned to btr, 64 env 1m replay, actor synced every step, decaying epsilon, btr noisy init ===
call :find_done a5_qb_btrlike_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=qbert run.seed=1 %BTRENV% %BTREPS% %BTRNZ% "app.run_name=run_{t}_a5_qb_btrlike_s1"
echo === 32. phoenix se 50m seed 1: noisynet on the sample efficiency profile, epsilon ladder kept ===
call :find_done a5_ph_se_nz_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=phoenix run.seed=1 %NZSIGSE% "app.run_name=run_{t}_a5_ph_se_nz_s1"
echo === 33. name_this_game se 50m seed 1: noisynet on the sample efficiency profile, epsilon ladder kept ===
call :find_done a5_ntg_se_nz_s1
if not defined DONE_DIR call :run_exe "%SENZ%" E1.game=name_this_game run.seed=1 %NZSIGSE% "app.run_name=run_{t}_a5_ntg_se_nz_s1"


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
