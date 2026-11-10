@echo off
title %~nx0

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
SET "EPS0=A2.actor.[train].policy.spatial.enabled=false A2.actor.[train].policy.epsilon=0 A2.actor.[train].policy.epsilon.start=0 A2.actor.[train].policy.spatial.epsilon.first=0 A2.actor.[train].policy.epsilon.end=0 A2.actor.[train].policy.spatial.epsilon.last=0"
SET "NZSIG=net.block.[AtariHeadFC512].noisy.sigma0=0.87 A2.head.projection.noisy.sigma0=1.2"
SET "BOMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@breakout>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2>metrics.scalar.@noisy"
SET "BOMET0=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@breakout>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2"
SET "NOCLIP=E1.reward_clip=false A3.reward_scaler.constant_scale=0.005"
SET "ELIFE=E1.episodic_life=true"
SET "STICKY0=E1.repeat_action_probability=0"
SET "UQE=A2.actor.[train].policy.policy_type=UQE"
SET "RT30=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_30m>run.@batch"
SET "LOWEPS=A2.actor.[train].policy.epsilon=0.05 A2.actor.[train].policy.epsilon.start=0.05 A2.actor.[train].policy.spatial.epsilon.first=0.05 A2.actor.[train].policy.epsilon.end=0.0001 A2.actor.[train].policy.spatial.epsilon.last=0.0001 A2.actor.[train].policy.uqe.epsilon.end=0.0001 A2.actor.[train].policy.spatial.uqe.epsilon.last=0.0001"
SET "PERHI=A2.learner.per_alpha=0.6 A2.learner.per_beta.start=0.4 A2.learner.per_beta.end=1.0"
SET "BOSS=%STICKY0% %LOWEPS% %NOCLIP% %PERHI%"
SET "N10=A2.learner.n_step=10"
SET "G999=A2.learner.gamma=0.999"
SET "UQB=%STICKY0% %NOCLIP% %PERHI% %UQE%"
SET "UQHI=A2.actor.[train].policy.uqe.epsilon=0.4 A2.actor.[train].policy.uqe.epsilon.start=0.4 A2.actor.[train].policy.spatial.uqe.epsilon.first=0.4 A2.actor.[train].policy.uqe.epsilon.end=0.01 A2.actor.[train].policy.spatial.uqe.epsilon.last=0.01"
SET "UQLO=A2.actor.[train].policy.uqe.epsilon=0.05 A2.actor.[train].policy.uqe.epsilon.start=0.05 A2.actor.[train].policy.spatial.uqe.epsilon.first=0.05 A2.actor.[train].policy.uqe.epsilon.end=0.0001 A2.actor.[train].policy.spatial.uqe.epsilon.last=0.0001"
SET "UQ99=A2.actor.[train].policy.uqe.tau=0.99 A2.actor.[train].policy.uqe.tau.start=0.99 A2.actor.[train].policy.spatial.uqe.tau.first=0.99 A2.actor.[train].policy.uqe.tau.end=0.9 A2.actor.[train].policy.spatial.uqe.tau.last=0.9"
SET "OPT=A3.use_optimistic_target=true"
SET "RT10=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_10m>run.@batch"
SET "RT10RND=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_10m>run.@rnd>run.@batch"
SET "EVFIX=DefaultDQNAgent.actor.[eval].policy.tau_rule.sample_mode=fixed"
SET "B1K=A2.learner.replay_batch_size=1024"
SET "RT10V=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@eval2ch_r1_50m>run.@to_10m>run.@vitva>run.@batch"
SET "SENZ=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@batch"
SET "NZSIGSE=net.block.[AtariHeadFC512].noisy.sigma0=1.1 A2.head.projection.noisy.sigma0=1.2"
SET "BTRENV=run.train.num_envs=64 A2.learner.replay_capacity=1048576 A2.actor.[train].sync_interval=1"
SET "BTREPS=A2.actor.[train].policy.spatial.enabled=false A2.actor.[train].policy.epsilon=1.0 A2.actor.[train].policy.epsilon.start=1.0 A2.actor.[train].policy.spatial.epsilon.first=1.0 A2.actor.[train].policy.epsilon.end=0.01 A2.actor.[train].policy.spatial.epsilon.last=0.01 A2.actor.[train].policy.epsilon.type=linear A2.actor.[train].policy.epsilon.steps=4000000"
SET "BTRNZ=net.block.[AtariHeadFC512].init.mode=default net.block.[AtariHeadFC512].noisy.sigma0=0.5 A2.head.init.mode=default A2.head.projection.noisy.sigma0=0.5"
SET "SENZPA=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@pixel_area>run.@batch"
SET "ALNINIT=net.block.[AtariImpalaConv32].init.mode=default net.block.[AtariImpalaConv64].init.mode=default net.block.[AtariImpalaRes32SN].init1.mode=default net.block.[AtariImpalaRes32SN].init2.mode=default net.block.[AtariImpalaRes64SN].init1.mode=default net.block.[AtariImpalaRes64SN].init2.mode=default net.block.[AtariIQNTauProj2304].init.mode=default"
SET "ALNM=A2.learner.munchausen.log_policy_mode=online"
SET "WLAD=A2.actor.[train].policy.spatial.enabled=true A2.actor.[train].policy.spatial.epsilon.first=0.4 A2.actor.[train].policy.spatial.epsilon.last=0"
SET "LADEXP=A2.actor.[train].policy.spatial.enabled=true A2.actor.[train].policy.spatial.epsilon.first.type=exponential A2.actor.[train].policy.spatial.epsilon.first.start=0.4 A2.actor.[train].policy.spatial.epsilon.first.end=0 A2.actor.[train].policy.spatial.epsilon.first.time_constant=5000000 A2.actor.[train].policy.spatial.epsilon.last=0"
SET "BTREPSPH=A2.actor.[train].policy.spatial.enabled=false A2.actor.[train].policy.epsilon.type=phased "A2.actor.[train].policy.epsilon.phases=decay off" A2.actor.[train].policy.epsilon.phase.[decay].type=exponential A2.actor.[train].policy.epsilon.phase.[decay].start=1.0 A2.actor.[train].policy.epsilon.phase.[decay].end=0.01 A2.actor.[train].policy.epsilon.phase.[decay].time_constant=2000000 A2.actor.[train].policy.epsilon.phase.[decay].steps=25000000 A2.actor.[train].policy.epsilon.phase.[off].type=constant A2.actor.[train].policy.epsilon.phase.[off].value=0 A2.actor.[train].policy.epsilon.phase.[off].steps=1"
SET "SENZPASK=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@nz_learning>run.@pixel_area>run.@skip>run.@batch"
SET "SKIPMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2>metrics.scalar.@noisy>metrics.scalar.@skip"
SET "MERGE=DefaultDQNAgent.learner.skip.feature_gradient=merge"
SET "FW10=app.batchrun.exp_exit_step=10000000 run.eval_schedule.[greedy].interval=15781"
SET "EVTAU=DefaultDQNAgent.learner.enabled=false app.save_agent_on_close=false run.train.num_envs=1 app.batchrun.exp_exit_step=5 run.eval_schedule.[eval].interval=10 run.eval_schedule.[greedy].interval=10 run.eval_schedule.[greedy_single].interval=10 run.eval.[eval].eval_batch_size=16 run.eval.[eval].eval_episodes=100 run.eval.[greedy].eval_batch_size=16 run.eval.[greedy].eval_episodes=100 run.eval.[greedy_single].eval_batch_size=16 run.eval.[greedy_single].eval_episodes=100 DefaultDQNAgent.actor.[eval].policy.tau_rule.sample_mode=random DefaultDQNAgent.actor.[greedy].policy.tau_rule.sample_mode=random DefaultDQNAgent.actor.[greedy_single].policy.tau_rule.sample_mode=fixed run.eval_schedule.[eval].warmup_count=0 run.eval_schedule.[greedy].warmup_count=0 run.eval_schedule.[greedy_single].warmup_count=0"

call :use_ws AtariSub-01 Atari.txt
echo === 1. freeway 10m seed 1: btr aligned, pixel_area, init default, munchausen log pi online, epsilon 1.0 to 0.01 with time constant 2M, greedy eval every 1.01M, no skip ===
call :find_done a5_fw_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=freeway run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %FW10% "app.run_name=run_{t}_a5_fw_aln_btreps_s1"
echo === 2. freeway 10m seed 1: same setup as 1 plus TempoRL sep-merge ===
call :find_done a5_fw_aln_t98mrg_s1
if not defined DONE_DIR call :run_exe "%SENZPASK%" "%SKIPMET%" E1.game=freeway run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %MERGE% %FW10% "app.run_name=run_{t}_a5_fw_aln_t98mrg_s1"
echo === 3. beam_rider 50m seed 1: btr aligned, pixel_area, init default, munchausen log pi online, epsilon 1.0 to 0.01 with time constant 2M then 0 from 25M, no skip ===
call :find_done a5_br_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=beam_rider run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_br_aln_btreps_s1"
echo === 4. beam_rider 50m seed 1: same setup as 3 plus TempoRL sep-merge ===
call :find_done a5_br_aln_t98mrg_s1
if not defined DONE_DIR call :run_exe "%SENZPASK%" "%SKIPMET%" E1.game=beam_rider run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %MERGE% "app.run_name=run_{t}_a5_br_aln_t98mrg_s1"
echo === 5. beam_rider 50m seed 2: same setup as 3 ===
call :find_done a5_br_aln_btreps_s2
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=beam_rider run.seed=2 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_br_aln_btreps_s2"
echo === 6. battle_zone 50m seed 1: same setup as 3 ===
call :find_done a5_bz_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=battle_zone run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_bz_aln_btreps_s1"
echo === 7. name_this_game 50m seed 1: same setup as 3 ===
call :find_done a5_ntg_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=name_this_game run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_ntg_aln_btreps_s1"
echo === 8. phoenix 50m seed 1: same setup as 3 ===
call :find_done a5_ph_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=phoenix run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_ph_aln_btreps_s1"
echo === 9. double_dunk 50m seed 1: same setup as 3 ===
call :find_done a5_dd_aln_btreps_s1
if not defined DONE_DIR call :run_exe "%SENZPA%" E1.game=double_dunk run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% "app.run_name=run_{t}_a5_dd_aln_btreps_s1"
echo === 10. battle_zone eval only on the 50m checkpoint of the same setup as 3: eval and greedy with random taus, greedy_single with fixed taus, 100 episodes each ===
SET "RUN_EXIT_CODE=0"
call :find_done a5_bz_aln_evtau_s1
if not defined DONE_DIR call :find_ckpt "run_*_a5_bz_aln_btreps_s1"
if not defined DONE_DIR if defined CKPT call :run_exe "%SENZPA%" E1.game=battle_zone run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %EVTAU% "DefaultDQNAgent.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_bz_aln_evtau_s1"
echo === 11. name_this_game eval only on the 50m checkpoint of the same setup as 3: eval and greedy with random taus, greedy_single with fixed taus, 100 episodes each ===
SET "RUN_EXIT_CODE=0"
call :find_done a5_ntg_aln_evtau_s1
if not defined DONE_DIR call :find_ckpt "run_*_a5_ntg_aln_btreps_s1"
if not defined DONE_DIR if defined CKPT call :run_exe "%SENZPA%" E1.game=name_this_game run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %EVTAU% "DefaultDQNAgent.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_ntg_aln_evtau_s1"
echo === 12. phoenix eval only on the 50m checkpoint of the same setup as 3: eval and greedy with random taus, greedy_single with fixed taus, 100 episodes each ===
SET "RUN_EXIT_CODE=0"
call :find_done a5_ph_aln_evtau_s1
if not defined DONE_DIR call :find_ckpt "run_*_a5_ph_aln_btreps_s1"
if not defined DONE_DIR if defined CKPT call :run_exe "%SENZPA%" E1.game=phoenix run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %EVTAU% "DefaultDQNAgent.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_ph_aln_evtau_s1"
echo === 13. double_dunk eval only on the 50m checkpoint of the same setup as 3: eval and greedy with random taus, greedy_single with fixed taus, 100 episodes each ===
SET "RUN_EXIT_CODE=0"
call :find_done a5_dd_aln_evtau_s1
if not defined DONE_DIR call :find_ckpt "run_*_a5_dd_aln_btreps_s1"
if not defined DONE_DIR if defined CKPT call :run_exe "%SENZPA%" E1.game=double_dunk run.seed=1 %BTRENV% %BTREPSPH% %BTRNZ% %ALNINIT% %ALNM% %EVTAU% "DefaultDQNAgent.auto_load_file=%CKPT%" "app.run_name=run_{t}_a5_dd_aln_evtau_s1"
echo === 14. kung_fu_master 10m seed 4: same as kfm_10m_s04 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s04
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=4 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s04"
echo === 15. kung_fu_master 10m seed 5: same as kfm_10m_s05 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s05
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=5 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s05"
echo === 16. kung_fu_master 10m seed 6: same as kfm_10m_s06 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s06
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=6 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s06"
echo === 17. kung_fu_master 10m seed 7: same as kfm_10m_s07 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s07
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=7 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s07"
echo === 18. kung_fu_master 10m seed 8: same as kfm_10m_s08 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s08
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=8 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s08"
echo === 19. kung_fu_master 10m seed 9: same as kfm_10m_s09 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s09
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=9 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s09"
echo === 20. kung_fu_master 10m seed 10: same as kfm_10m_s10 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s10
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=10 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s10"
echo === 21. kung_fu_master 10m seed 11: same as kfm_10m_s11 plus RND with the default settings (beta 0.1) ===
call :find_done kfm_10m_rnd_b01_s11
if not defined DONE_DIR call :run_exe "%RT10RND%" E1.game=kung_fu_master run.seed=11 DefaultDQNAgent.intrinsic.beta=0.1 %EVFIX% "app.run_name=run_{t}_kfm_10m_rnd_b01_s11"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
goto :beep_forever

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
goto :beep_forever


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
goto :beep_forever

:beep_forever
powershell -NoProfile -Command "[console]::Beep(1000,800)" >nul 2>&1
ping -n 301 127.0.0.1 >nul
goto :beep_forever
