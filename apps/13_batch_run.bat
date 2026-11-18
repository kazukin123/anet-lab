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
SET "R123=DefaultDQNAgent.intrinsic.beta=0.1 DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000"
SET "R123EMA=%R123% DefaultDQNAgent.intrinsic.rnd.error_norm.average_mode=ema"
SET "R123CE=DefaultDQNAgent.intrinsic.beta=0.1 DefaultDQNAgent.intrinsic.rnd.warmup_mode=collect_errors DefaultDQNAgent.intrinsic.rnd.warmup_steps=12800"
SET "R123NC=DefaultDQNAgent.intrinsic.beta=0.01 DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000 DefaultDQNAgent.intrinsic.rnd.error_norm.use_centering=false"
SET "R123B003=DefaultDQNAgent.intrinsic.beta=0.03 DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000"
SET "B003W1M=DefaultDQNAgent.intrinsic.beta=0.03 DefaultDQNAgent.intrinsic.rnd.warmup_steps=1000000"
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

SET "VD003=DefaultDQNAgent.intrinsic.value_mode=decomposed DefaultDQNAgent.intrinsic.reward_scaler.constant_scale=0.03 DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000"
SET "VDMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2>metrics.scalar.@value_decomposition"
SET "VDG1=metrics.scalar.[77_agent_value/41_greedy_disagree]=extrinsic_q_session_argmax_disagreement_rate $eval.[greedy] $actor @session_end"
SET "VDG2=metrics.scalar.[77_agent_value/42_greedy_regret]=extrinsic_q_session_regret_mean $eval.[greedy] $actor @session_end"
SET "VDG3=metrics.scalar.[77_agent_value/43_greedy_ext_gap]=extrinsic_q_session_gap_mean $eval.[greedy] $actor @session_end"
SET "VDG4=metrics.scalar.[77_agent_value/44_greedy_int_gap]=intrinsic_q_session_gap_mean $eval.[greedy] $actor @session_end"
SET "K32=DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus=32 DefaultDQNAgent.actor.[eval].policy.tau_rule.num_taus=32"
SET "TGCH=run.$=%A5%>%RR4%>%RF%>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@to_30m>run.@rnd>run.@batch"
SET "TGQ=DefaultDQNAgent.learner.replay_ratio=8 DefaultDQNAgent.model.hard_update_interval=250 DefaultDQNAgent.learner.per_alpha=0.5 DefaultDQNAgent.learner.per_beta.start=0.5 DefaultDQNAgent.learner.per_beta.end=0.5"
SET "TGEV=run.eval_schedule.[eval].interval=62528 run.eval_schedule.[greedy].interval=31250 run.eval_schedule.[greedy_single].interval=31250"
SET "TGRND=DefaultDQNAgent.intrinsic.learn_ratio=1 DefaultDQNAgent.intrinsic.learn_batch_size=128"
SET "MZT=E1.game=montezuma_revenge run.seed=1 run.eval.[greedy].eval_episodes=16"
SET "NMDD=DefaultDQNAgent.learner.munchausen.enabled=false DefaultDQNAgent.learner.use_double_dqn=true"
SET "VD=DefaultDQNAgent.intrinsic.value_mode=decomposed DefaultDQNAgent.intrinsic.reward_scaler.constant_scale=0.01 DefaultDQNAgent.intrinsic.beta=1"
SET "G99NE=DefaultDQNAgent.learner.intrinsic.discount_mode=separate DefaultDQNAgent.learner.intrinsic.gamma=0.99 DefaultDQNAgent.learner.intrinsic.terminal_mode=non_episodic"
SET "FGS=DefaultDQNAgent.learner.intrinsic.feature_gradient=split"
SET "PL1=metrics.scalar.[34_agent_plasticity/41_probe_dormant_ratio]=$agent plasticity_probe_dormant_ratio @learn $learn_step interval:5000"
SET "PL2=metrics.scalar.[34_agent_plasticity/42_probe_dead_ratio]=$agent plasticity_probe_dead_ratio @learn $learn_step interval:5000"
SET "PL3=metrics.scalar.[34_agent_plasticity/43_probe_feature_norm]=$agent plasticity_probe_feature_norm @learn $learn_step interval:5000"
SET "PL4=metrics.scalar.[34_agent_plasticity/44_probe_srank]=$agent plasticity_probe_srank @learn $learn_step interval:5000"
SET "PL5=metrics.scalar.[34_agent_plasticity/45_probe_srank_ratio]=$agent plasticity_probe_srank_ratio @learn $learn_step interval:5000"
SET "MZW=DefaultDQNAgent.intrinsic.rnd.warmup_steps=200000"
SET "RBWTG=run.$=%A5%>run.@hard125>run.@munch>%RF%>run.@cap2m>run.@eval2ch_r1>run.@to_100m>run.@rainbow_like>run.@rnd>run.@rnd_taiga>run.@batch"
SET "FP32=DefaultDQNAgent.learner.use_amp=false DefaultDQNAgent.learner.use_amp_bf16=false"
SET "EPSH=DefaultDQNAgent.actor.[train].policy.epsilon.type=phased DefaultDQNAgent.actor.[train].policy.epsilon.phase.[hold].value=1.0 DefaultDQNAgent.actor.[train].policy.epsilon.phase.[hold].steps=20000 DefaultDQNAgent.actor.[train].policy.epsilon.phase.[decay].type=linear DefaultDQNAgent.actor.[train].policy.epsilon.phase.[decay].start=1.0 DefaultDQNAgent.actor.[train].policy.epsilon.phase.[decay].end=0.01 DefaultDQNAgent.actor.[train].policy.epsilon.phase.[decay].steps=250000"
SET "EPSHP=DefaultDQNAgent.actor.[train].policy.epsilon.phases=hold decay"
SET "ASYNC1=A2.actor.[train].sync_interval=1"
SET "MZR2=E1.game=montezuma_revenge run.seed=2 run.train.num_envs=4"
SET "E64=run.train.num_envs=64 A2.actor.[train].sync_interval=800"
SET "RBW=run.$=%A5%>run.@hard125>run.@munch>%RF%>run.@cap2m>run.@eval2ch_r1>run.@to_100m>run.@rainbow_like>run.@batch"
SET "DQIQN=DefaultDQNAgent.learner.use_per=false DefaultDQNAgent.learner.learning_rate=5e-5 DefaultDQNAgent.learner.adam_eps=3.125e-4 DefaultDQNAgent.learner.iqn.current_taus.num_taus=64 DefaultDQNAgent.learner.iqn.target_taus.num_taus=64 DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus=32 DefaultDQNAgent.actor.[eval].policy.tau_rule.num_taus=32 DefaultDQNAgent.actor.[greedy].policy.tau_rule.num_taus=32"
SET "NOGS=run.eval_schedule.[greedy_single].interval=0"
SET "BO1=E1.game=breakout run.seed=1 run.train.num_envs=4"
SET "QB1=E1.game=qbert run.seed=1 run.train.num_envs=4"

call :use_ws AtariSub-01 Atari.txt

echo === 50. breakout 5m seed 1: rainbow_like aligned to Dopamine IQN (uniform replay, Adam 5e-5 eps 3.125e-4, taus 64/64/32), no RND, 4 env ===
call :find_done bo_dqiqn_s1
if not defined DONE_DIR call :run_exe "%RBW%" %BO1% %FP32% %EPSH% "%EPSHP%" %ASYNC1% %DQIQN% %NOGS% app.batchrun.exp_exit_step=5000000 "app.run_name=run_{t}_bo_dqiqn_s1"

echo === 51. qbert 5m seed 1: same as 50 ===
call :find_done qb_dqiqn_s1
if not defined DONE_DIR call :run_exe "%RBW%" %QB1% %FP32% %EPSH% "%EPSHP%" %ASYNC1% %DQIQN% %NOGS% app.batchrun.exp_exit_step=5000000 "app.run_name=run_{t}_qb_dqiqn_s1"

echo === 52. breakout 5m seed 1: same Q side as mz_rbw_rndtg (rainbow_like with PER, taus 8) without RND, 4 env ===
call :find_done bo_rbw_s1
if not defined DONE_DIR call :run_exe "%RBW%" %BO1% %FP32% %EPSH% "%EPSHP%" %ASYNC1% %NOGS% app.batchrun.exp_exit_step=5000000 "app.run_name=run_{t}_bo_rbw_s1"

echo === 53. qbert 5m seed 1: same as 52 ===
call :find_done qb_rbw_s1
if not defined DONE_DIR call :run_exe "%RBW%" %QB1% %FP32% %EPSH% "%EPSHP%" %ASYNC1% %NOGS% app.batchrun.exp_exit_step=5000000 "app.run_name=run_{t}_qb_rbw_s1"

echo === 54. montezuma_revenge 30m seed 1: same as mz_tg_g99ne_split_s1 but 64 env (replay lane window 32,768 instead of 16,384 with the same total capacity), train actor sync every 800 train steps (same exp-step interval) and RND learn batch 64 (all lanes every step, as 128 with 128 env) ===
call :find_done mz_tg_split_e64_s1
if not defined DONE_DIR call :run_exe "%TGCH%" %MZT% %K32% %ALNINIT% %TGQ% %TGRND% %TGEV% %NMDD% %MZW% %VD% %G99NE% %FGS% "%VDMET%" "%VDG1%" "%VDG2%" "%VDG3%" "%VDG4%" "%PL1%" "%PL2%" "%PL3%" "%PL4%" "%PL5%" %E64% DefaultDQNAgent.intrinsic.learn_batch_size=64 app.batchrun.exp_exit_step=30000000 "app.run_name=run_{t}_mz_tg_split_e64_s1"

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
