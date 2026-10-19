@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"

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

SET "WS=--workspace DropMerge-02"
SET "RUNS=workspaces\DropMerge-02\runs"
SET "BASE=run.$=run.@iqn32_stratified"
SET "QRBASE=run.$=run.@qr51_control"
SET "BATCH=app.$=app.batchrun>P1"
SET "B100M=app.batchrun.exp_exit_step=100000000"
SET "MUNCH=A3.learner.munchausen.enabled=true A3.learner.use_double_dqn=false"
SET "MMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@iqn_search_p0>metrics.scalar.@munchausen>M1>M2"
SET "MTAU3=A3.learner.munchausen.entropy_tau=0.03 A3.learner.munchausen.clip_value_min=-1"
SET "MAINRELU=DefaultDQNAgent.net.branch.[main_feature].structure=DMImpalaConv64 > DMImpalaPool > DMImpalaRes64(*2) > DMImpalaConv128 > DMImpalaPool > DMImpalaRes128(*2) > SiLU > PosEmbed2D > DMImpalaTrans128 > Flatten > Linear_2048 > ReLU"
SET "TAURELU=DefaultDQNAgent.net.branch.[tau_embedding].structure=IQNCosEmb > IQNTauProj2048 > ReLU"
SET "VALRELU=DefaultDQNAgent.net.branch.[value_stream].structure=HeadFC1024 > ReLU"
SET "ADVRELU=DefaultDQNAgent.net.branch.[adv_stream].structure=HeadFC1024 > ReLU"
SET "EVONLY=A3.learner.enabled=false app.save_agent_on_close=false run.train.num_envs=1 run.eval.[eval_target].eval_batch_size=16 run.eval.[eval_target].eval_episodes=100 run.eval.[eval].eval_batch_size=16 run.eval.[eval].eval_episodes=100 run.eval_schedule.[eval_target].interval=10 run.eval_schedule.[eval].interval=11 app.batchrun.exp_exit_step=40"
SET "GEVONLY=run.eval.[eval_target_greedy].actor=eval_panel run.eval.[eval_target_greedy].eval_batch_size=16 run.eval.[eval_target_greedy].eval_episodes=100 run.eval_schedule.[eval_target_greedy].interval=12 run.eval_schedule.[eval_target_greedy].use_background=true"
SET "GEV=run.eval.[eval_target_greedy].actor=eval_panel run.eval.[eval_target_greedy].eval_batch_size=1 run.eval.[eval_target_greedy].eval_episodes=1 run.eval_schedule.[eval_target_greedy].interval=200 run.eval_schedule.[eval_target_greedy].use_background=true"
SET "GMET="M2.[21_eval/11_greedy_reward]=mean.episode_return $runner @session_end $eval.[eval_target_greedy]" "M2.[21_eval/13_greedy_reward_ema]=mean.episode_return $runner @session_end $eval.[eval_target_greedy] $ema ema_alpha:0.02" "M2.[21_eval/15_greedy_ep_steps]=$runner mean.episode_steps @session_end $eval.[eval_target_greedy]" "M2.[53_eval3/00_ep_step_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.ep_step $ema ema_alpha:0.02" "M2.[53_eval3/01_ep_maxrank_max]=$eval.[eval_target_greedy] @session_end $env max.ep_max_rank" "M2.[53_eval3/13_double_suika_created_mean]=$eval.[eval_target_greedy] @session_end $env $exp_step mean.ep_double_suika_created" "M2.[53_eval3/61_ep_maxrank_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.ep_max_rank $ema ema_alpha:0.02" "M2.[53_eval3/73_tr_blk_mean]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_spawn_blocked" "M2.[53_eval3/74_tr_timeout_mean]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_no_drop_timeout" "M2.[53_eval3/84_tr_timeout_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_no_drop_timeout $ema ema_alpha:0.02""
SET "SGMET="M2.[36_agent_munchausen/07_soft_gap]=munchausen_soft_gap @learn $update_result""
SET "NDT20=E1.no_drop_timeout_gameover_penalty=-20"
SET "NSTEP3=A3.learner.n_step=3"
SET "SNBLK=net.block.[DMImpalaRes64SNR].type=ResBlock net.block.[DMImpalaRes64SNR].res.channels=64 net.block.[DMImpalaRes64SNR].res.kernel_size=3 net.block.[DMImpalaRes64SNR].res.activation=silu net.block.[DMImpalaRes64SNR].res.activation_mode=pre net.block.[DMImpalaRes64SNR].res.norm_type=none net.block.[DMImpalaRes64SNR].res.conv1_bias=true net.block.[DMImpalaRes64SNR].res.conv2_bias=true net.block.[DMImpalaRes64SNR].res.droppath_rate=0.0 net.block.[DMImpalaRes64SNR].weight_norm.mode=spectral net.block.[DMImpalaRes64SNR].init2.mode=he net.block.[DMImpalaRes128SNR].type=ResBlock net.block.[DMImpalaRes128SNR].res.channels=128 net.block.[DMImpalaRes128SNR].res.kernel_size=3 net.block.[DMImpalaRes128SNR].res.activation=silu net.block.[DMImpalaRes128SNR].res.activation_mode=pre net.block.[DMImpalaRes128SNR].res.norm_type=none net.block.[DMImpalaRes128SNR].res.conv1_bias=true net.block.[DMImpalaRes128SNR].res.conv2_bias=true net.block.[DMImpalaRes128SNR].res.droppath_rate=0.0 net.block.[DMImpalaRes128SNR].weight_norm.mode=spectral net.block.[DMImpalaRes128SNR].init2.mode=he"
SET "SNSTRUCT=DefaultDQNAgent.net.branch.[main_feature].structure=DMImpalaConv64 > DMImpalaPool > DMImpalaRes64SNR(*2) > DMImpalaConv128 > DMImpalaPool > DMImpalaRes128SNR(*2) > SiLU > PosEmbed2D > DMImpalaTrans128 > Flatten > Linear_2048 > SiLU"
SET "SNMET="M2.[34_agent_plasticity/63_weight_norm_feature_effective]=plasticity_weight_norm_feature_effective @learn $learn_step $update_result interval:500" "M2.[34_agent_plasticity/64_weight_norm_readout_effective]=plasticity_weight_norm_readout_effective @learn $learn_step $update_result interval:500" "M2.[34_agent_plasticity/65_spectral_sigma_feature]=plasticity_spectral_sigma_feature @learn $learn_step $update_result interval:500" "M2.[34_agent_plasticity/66_spectral_sigma_readout]=plasticity_spectral_sigma_readout @learn $learn_step $update_result interval:500""
SET "RR4=A3.learner.replay_ratio=4 A3.model.soft_update_tau=0.00125 run.eval_schedule.[eval_target].interval=400 run.eval_schedule.[eval].interval=400 run.eval_schedule.[eval_target_greedy].interval=800"
SET "NZNET=net.block.[HeadFC1024].noisy.enabled=true net.block.[HeadFC1024].noisy.sigma0=0.5 A3.head_projection.noisy.enabled=true A3.head_projection.noisy.sigma0=0.5"
SET "NZACT=A3.nn_forward.[act_lane].noisy.mode=sample A3.nn_forward.[act_lane].noisy.hold=call A3.nn_forward.[act_lane].noisy.share=sample A3.actor.[train].nn_forward_key=act_lane"
SET "NZREP=A3.actor.[train].policy.uqe_eps_start=0"
SET "NZMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@iqn_search_p0>metrics.scalar.@munchausen>M1>M2>metrics.scalar.@noisy"
SET "PMAX=A3.learner.per_initial_priority_mode=max"
SET "P003=A3.learner.per_initial_priority=0.03"
SET "B512RR1=A3.learner.replay_batch_size=512 A3.learner.replay_ratio=1 A3.model.soft_update_tau=0.01 run.eval_schedule.[eval_target].interval=50 run.eval_schedule.[eval].interval=50 run.eval_schedule.[eval_target_greedy].interval=100"
SET "B512RR2=A3.learner.replay_batch_size=512 A3.learner.replay_ratio=2"
SET "TF8=net.block.[DMImpalaTrans128].tf.num_layers=8"
SET "G999=A3.learner.gamma=0.999"
SET "B1024RR1=A3.learner.replay_batch_size=1024 A3.learner.replay_ratio=1 A3.model.soft_update_tau=0.02 run.eval_schedule.[eval_target].interval=25 run.eval_schedule.[eval].interval=25 run.eval_schedule.[eval_target_greedy].interval=50"
SET "NZGMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@iqn_search_p0>metrics.scalar.@munchausen>M1>M2>metrics.scalar.@noisy>metrics.scalar.@gradnoise>metrics.scalar.@attention"
SET "LRHALF=A3.learner.alpha=5e-5"
SET "B2048RR1=A3.learner.replay_batch_size=2048 A3.learner.replay_ratio=1 A3.model.soft_update_tau=0.04 run.eval_schedule.[eval_target].interval=12 run.eval_schedule.[eval].interval=12 run.eval_schedule.[eval_target_greedy].interval=25"
SET "B4096RR1=A3.learner.replay_batch_size=4096 A3.learner.replay_ratio=1 A3.model.soft_update_tau=0.08 run.eval_schedule.[eval_target].interval=6 run.eval_schedule.[eval].interval=6 run.eval_schedule.[eval_target_greedy].interval=12"
SET "B256RR2=A3.learner.replay_ratio=2 A3.model.soft_update_tau=0.0025 run.eval_schedule.[eval_target].interval=200 run.eval_schedule.[eval].interval=200 run.eval_schedule.[eval_target_greedy].interval=400"

SET RUNNER="bin\%BUILD%\AnetRLRunner.exe"
SET "WS=--workspace DropMerge-03"
SET "EVBIG=run.eval.[eval_target].eval_episodes=1250 run.eval.[eval].eval_episodes=1250 run.eval.[eval_target_greedy].eval_episodes=1250 run.eval.[eval_target].eval_batch_size=64 run.eval.[eval].eval_batch_size=64 run.eval.[eval_target_greedy].eval_batch_size=64"
SET "EV8=run.eval.[eval_target].eval_batch_size=8 run.eval.[eval_target].eval_episodes=8 run.eval.[eval_target_greedy].eval_batch_size=8 run.eval.[eval_target_greedy].eval_episodes=8"
:wait_disk_v0
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_v0
)
echo === ev only 5000 x 3: 42i final (230M) ===
SET "RUNS=workspaces\DropMerge-03\runs"
SET "RUN_EXIT_CODE=0"
call :find_ckpt "run_*_dmfb_42i_34a_b256rr2"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "app.run_name=run_{t}_dmfb_42i_ev_b256rr2"
:wait_disk_v1
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_v1
)
echo === 42G. +60M from 34a (170M to 230M): B256/RR1, duplicate of 42A, in-training eval x8 ===
SET "RUNS=workspaces\DropMerge-03\runs"
SET "RUN_EXIT_CODE=0"
call :find_ckpt "run_20261015-081844_*"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %B100M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% "A3.auto_load_file=%CKPT%" app.batchrun.exp_exit_step=60000000 %EV8% "app.run_name=run_{t}_dmfb_42g_34a_ctl2"
:wait_disk_v2
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_v2
)
echo === ev only 5000 x 3: 42g final (230M) ===
SET "RUNS=workspaces\DropMerge-03\runs"
SET "RUN_EXIT_CODE=0"
call :find_ckpt "run_*_dmfb_42g_34a_ctl2"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "app.run_name=run_{t}_dmfb_42g_ev_ctl2"
:wait_disk_v3
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_v3
)
echo === 42B. +60M from 34a (170M to 230M): B256/RR1 lr 7.1e-5, in-training eval x8 ===
SET "RUNS=workspaces\DropMerge-03\runs"
SET "RUN_EXIT_CODE=0"
call :find_ckpt "run_20261015-081844_*"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %B100M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% "A3.auto_load_file=%CKPT%" A3.learner.alpha=7.1e-5 app.batchrun.exp_exit_step=60000000 %EV8% "app.run_name=run_{t}_dmfb_42b_34a_lr7.1e-5"
:wait_disk_v4
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 15 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 15 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk_v4
)
echo === ev only 5000 x 3: 42b final (230M) ===
SET "RUNS=workspaces\DropMerge-03\runs"
SET "RUN_EXIT_CODE=0"
call :find_ckpt "run_*_dmfb_42b_34a_lr7.1e-5"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "app.run_name=run_{t}_dmfb_42b_ev_lr7.1e-5"

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
