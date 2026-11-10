@echo off
title %~nx0

cd /d "%~dp0runner"

REM 2026-11-01: DropMerge x TempoRL phase 1 (q47). From scratch to 100M with the main-line recipe (B256/RR1),
REM TempoRL sep-merge first, then the TempoRL-off control, each followed by its eval-only run.
REM Every run carries the Greedy sessions [greedy] (53_evalg, full) and [greedy_single] (54_evalgs, decide every step).
REM 2026-11-01 evening: 47a/47b are done and skipped. 47c repeats the 47b control with seed 2 to 50M to measure the
REM run-to-run spread (47b ran 12-20% below the old-build runs; the 10/15 and 11/01 exes learn identically).
REM 47c uses the exact exe of 47b (11/01 01:31), not the current build.
REM 2026-11-02 night: 47c is done and skipped. 47b and 47c (new build) both ran below the old-build runs from 30M on,
REM although the 10/15 and 11/01 exes learn identically in deterministic mode. Two checks, 50M each:
REM   47d: the 10/15 exe with the resolved config of 40_s2 (old keys), seed 1, rerun on today's machine.
REM   47e: the 47b recipe with the UQE epsilon ladder at the old effective value (1e-4 on all lanes; PRD101 D14 made it 0).
REM Queue order (>= 16 h): 47d, 47f (TempoRL auxiliary-only, PRD104), 47e, 47g (control seed 3, spread of the new build).
REM 2026-11-02 morning: 47d (10/15 exe, run today) came back in the old-build band, so 47f was stopped at about 10M and the
REM queue now starts with the long deterministic pair: the 10/15 exe and the 11/01 exe on the regression-hunt configs
REM (backend deterministic, evals off, seed 1, the new exe with the UQE epsilon ladder at 1e-4) to 30M each, into the
REM regression-hunt runs folder. A watcher stops rg_e7L at the first learning series that differs from rg_e0L.
REM Then 47e, 47g and 47f (from scratch).

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

SET "BUILD=Release"
SET "SRC_EXE=%~dp0..\.scratch\dm-regression-20261101\bin\AnetRLRunner_ab12_20261101.exe"
if not exist "%SRC_EXE%" goto :no_exe
copy /Y "%SRC_EXE%" "bin\%BUILD%\AnetRLRunner_ab12.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab12.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab12.exe"
REM 47d runs the 10/15 exe on the resolved config dump of 40_s2 (the old exe does not read the current config files).
SET "OLD_EXE_SRC=%~dp0..\.scratch\prd093\before\apps\runner\bin\Release\AnetRLRunner.exe"
if not exist "%OLD_EXE_SRC%" goto :no_exe
copy /Y "%OLD_EXE_SRC%" "bin\%BUILD%\AnetRLRunner_ab12old.exe" >nul
if errorlevel 1 goto :no_exe
SET RUNNER_OLD="bin\%BUILD%\AnetRLRunner_ab12old.exe"
SET "CFG_OLD=%~dp0..\.scratch\dm-regression-20261101\cfg_old.txt"
if not exist "%CFG_OLD%" goto :no_exe
for %%i in ("%~dp0..\.scratch\dm-regression-20261101") do SET "RG=%%~fi"
SET "CFG_NEW=%RG%\cfg_new.txt"
if not exist "%CFG_NEW%" goto :no_exe

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0
SET "RUN_EXIT_CODE=0"

SET "BASE=run.$=run.@iqn32_stratified"
SET "SKBASE=run.$=run.@iqn32_stratified>run.@skip"
SET "BATCH=app.$=app.batchrun>P1"
SET "B100M=app.batchrun.exp_exit_step=100000000"
SET "B50M=app.batchrun.exp_exit_step=50000000"
SET "MUNCH=A3.learner.munchausen.enabled=true A3.learner.use_double_dqn=false"
SET "NZGMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@iqn_search_p0>metrics.scalar.@munchausen>M1>M2>metrics.scalar.@noisy>metrics.scalar.@gradnoise>metrics.scalar.@attention"
SET "SKMET=metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@iqn_search_p0>metrics.scalar.@munchausen>M1>M2>metrics.scalar.@noisy>metrics.scalar.@gradnoise>metrics.scalar.@attention>metrics.scalar.@skip"
SET "MERGE=DefaultDQNAgent.learner.skip.feature_gradient=merge"
SET "MTAU3=A3.learner.munchausen.entropy_tau=0.03 A3.learner.munchausen.clip_value_min=-1"
SET "GEV=run.eval.[eval_target_greedy].actor=eval_panel run.eval.[eval_target_greedy].eval_batch_size=1 run.eval.[eval_target_greedy].eval_episodes=1 run.eval_schedule.[eval_target_greedy].interval=200 run.eval_schedule.[eval_target_greedy].use_background=true"
SET "GMET="M2.[21_eval/11_greedy_reward]=mean.episode_return $runner @session_end $eval.[eval_target_greedy]" "M2.[21_eval/13_greedy_reward_ema]=mean.episode_return $runner @session_end $eval.[eval_target_greedy] $ema ema_alpha:0.02" "M2.[21_eval/15_greedy_ep_steps]=$runner mean.episode_steps @session_end $eval.[eval_target_greedy]" "M2.[53_eval3/00_ep_step_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.ep_step $ema ema_alpha:0.02" "M2.[53_eval3/01_ep_maxrank_max]=$eval.[eval_target_greedy] @session_end $env max.ep_max_rank" "M2.[53_eval3/13_double_suika_created_mean]=$eval.[eval_target_greedy] @session_end $env $exp_step mean.ep_double_suika_created" "M2.[53_eval3/61_ep_maxrank_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.ep_max_rank $ema ema_alpha:0.02" "M2.[53_eval3/73_tr_blk_mean]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_spawn_blocked" "M2.[53_eval3/74_tr_timeout_mean]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_no_drop_timeout" "M2.[53_eval3/84_tr_timeout_mean_ema]=$eval.[eval_target_greedy] @session_end $env mean.term_reason_no_drop_timeout $ema ema_alpha:0.02""
SET "SGMET="M2.[36_agent_munchausen/07_soft_gap]=munchausen_soft_gap @learn $update_result""
SET "NDT20=E1.no_drop_timeout_gameover_penalty=-20"
SET "NZNET=net.block.[HeadFC1024].noisy.enabled=true net.block.[HeadFC1024].noisy.sigma0=0.5 A3.head.projection.noisy.enabled=true A3.head.projection.noisy.sigma0=0.5"
SET "NZACT=A3.nn_forward.[act_lane].noisy.mode=sample A3.nn_forward.[act_lane].noisy.hold=call A3.nn_forward.[act_lane].noisy.share=sample A3.actor.[train].nn_forward_key=act_lane"
SET "NZREP=A3.actor.[train].policy.uqe.epsilon=0 A3.actor.[train].policy.uqe.epsilon.start=0 A3.actor.[train].policy.spatial.uqe.epsilon.first=0"
REM Long deterministic pair: the arguments of the regression hunt (.scratch/dm-regression-20261101/rg.py) with 30M exp steps.
SET "DET=app.exp_exit_step=30000000 app.batchrun.exp_exit_step=30000000 app.save_agent_on_close=false app.show_error_dialog=false app.train_auto_start=true run.seed=1 backend.cudnn_benchmark=false backend.cudnn_deterministic=true backend.deterministic_algorithms=true backend.deterministic_warn_only=false run.eval_schedule.[eval_target].interval=0 run.eval_schedule.[eval].interval=0 run.eval_schedule.[eval_target_greedy].interval=0"
SET "DETNEW=run.eval_schedule.[greedy].interval=0 run.eval_schedule.[greedy_single].interval=0 DefaultDQNAgent.actor.[train].policy.spatial.uqe.epsilon.first=0.0001 DefaultDQNAgent.actor.[train].policy.spatial.uqe.epsilon.last=0.0001"
SET "EPS4=A3.actor.[train].policy.spatial.uqe.epsilon.first=0.0001 A3.actor.[train].policy.spatial.uqe.epsilon.last=0.0001"
REM TempoRL auxiliary-only (PRD104 run.@skip_aux): no action is held while training or in the in-training evals;
REM the skip loss only adds its gradient to the shared features. PRD104 rejects any holding Actor with this window,
REM so [greedy] (copied from [eval_target]) is single too and 53_evalg equals 54_evalgs, as in the TempoRL-off runs.
SET "AUX=DefaultDQNAgent.learner.skip.target_window=action_change DefaultDQNAgent.actor.[train].skip.max_length_mode=single DefaultDQNAgent.actor.[eval].skip.max_length_mode=single DefaultDQNAgent.actor.[eval_target].skip.max_length_mode=single DefaultDQNAgent.actor.[eval_panel].skip.max_length_mode=single"
SET "EV8=run.eval.[eval_target].eval_batch_size=8 run.eval.[eval_target].eval_episodes=8 run.eval.[eval_target_greedy].eval_batch_size=8 run.eval.[eval_target_greedy].eval_episodes=8"
REM Greedy sessions on the target network: [greedy] keeps the skip length (full), [greedy_single] decides every step.
REM In a run without TempoRL the two are the same policy; they are always placed so that 53_evalg and 54_evalgs line up across runs.
REM GSEL has a '>' and is passed quoted, like BASE.
SET "GSEL=DefaultDQNAgent.actor.[greedy].$=DefaultDQNAgent.actor.[eval_target]>DefaultDQNAgent.actor.@greedy"
SET "GACT=DefaultDQNAgent.actor.[greedy_single].$=DefaultDQNAgent.actor.[greedy] DefaultDQNAgent.actor.[greedy_single].skip.max_length_mode=single run.eval.[greedy].actor=greedy run.eval.[greedy_single].actor=greedy_single run.eval_schedule.[greedy].use_background=true run.eval_schedule.[greedy_single].use_background=true"
REM 19531 learn steps = 5.0M exp steps at B256/RR1: 20 sessions of 128 episodes in 100M.
SET "GSES=run.eval.[greedy].eval_batch_size=32 run.eval.[greedy].eval_episodes=128 run.eval.[greedy_single].eval_batch_size=32 run.eval.[greedy_single].eval_episodes=128 run.eval_schedule.[greedy].interval=19531 run.eval_schedule.[greedy_single].interval=19531"
SET "GMETG="M2.[53_evalg/00_ep_step_mean]=$eval.[greedy] @session_end $env mean.ep_step" "M2.[53_evalg/01_ep_maxrank_max]=$eval.[greedy] @session_end $env max.ep_max_rank" "M2.[53_evalg/02_ep_maxrank_mean]=$eval.[greedy] @session_end $env mean.ep_max_rank" "M2.[53_evalg/10_reward_mean]=mean.episode_return $runner @session_end $eval.[greedy]" "M2.[53_evalg/16_reward_max]=max.episode_return $runner @session_end $eval.[greedy]" "M2.[53_evalg/13_double_suika_created_mean]=$eval.[greedy] @session_end $env $exp_step mean.ep_double_suika_created" "M2.[53_evalg/73_tr_blk_mean]=$eval.[greedy] @session_end $env mean.term_reason_spawn_blocked" "M2.[53_evalg/74_tr_timeout_mean]=$eval.[greedy] @session_end $env mean.term_reason_no_drop_timeout" "M2.[53_evalg/75_tr_nolg_mean]=$eval.[greedy] @session_end $env mean.term_reason_no_legal_drop" "M2.[53_evalg/81_same_drop_col_ratio_mean]=$eval.[greedy] @session_end $env mean.ep_same_drop_col_ratio""
SET "GMETS="M2.[54_evalgs/00_ep_step_mean]=$eval.[greedy_single] @session_end $env mean.ep_step" "M2.[54_evalgs/01_ep_maxrank_max]=$eval.[greedy_single] @session_end $env max.ep_max_rank" "M2.[54_evalgs/02_ep_maxrank_mean]=$eval.[greedy_single] @session_end $env mean.ep_max_rank" "M2.[54_evalgs/10_reward_mean]=mean.episode_return $runner @session_end $eval.[greedy_single]" "M2.[54_evalgs/16_reward_max]=max.episode_return $runner @session_end $eval.[greedy_single]" "M2.[54_evalgs/13_double_suika_created_mean]=$eval.[greedy_single] @session_end $env $exp_step mean.ep_double_suika_created" "M2.[54_evalgs/73_tr_blk_mean]=$eval.[greedy_single] @session_end $env mean.term_reason_spawn_blocked" "M2.[54_evalgs/74_tr_timeout_mean]=$eval.[greedy_single] @session_end $env mean.term_reason_no_drop_timeout" "M2.[54_evalgs/75_tr_nolg_mean]=$eval.[greedy_single] @session_end $env mean.term_reason_no_legal_drop" "M2.[54_evalgs/81_same_drop_col_ratio_mean]=$eval.[greedy_single] @session_end $env mean.ep_same_drop_col_ratio""
REM Eval-only: 4 sessions x 1250 episodes on each of the five slots.
SET "EVONLY=A3.learner.enabled=false app.save_agent_on_close=false run.train.num_envs=1 run.eval.[eval_target].eval_batch_size=16 run.eval.[eval_target].eval_episodes=100 run.eval.[eval].eval_batch_size=16 run.eval.[eval].eval_episodes=100 run.eval_schedule.[eval_target].interval=10 run.eval_schedule.[eval].interval=11 app.batchrun.exp_exit_step=40"
SET "GEVONLY=run.eval.[eval_target_greedy].actor=eval_panel run.eval.[eval_target_greedy].eval_batch_size=16 run.eval.[eval_target_greedy].eval_episodes=100 run.eval_schedule.[eval_target_greedy].interval=12 run.eval_schedule.[eval_target_greedy].use_background=true"
SET "EVBIG=run.eval.[eval_target].eval_episodes=1250 run.eval.[eval].eval_episodes=1250 run.eval.[eval_target_greedy].eval_episodes=1250 run.eval.[eval_target].eval_batch_size=64 run.eval.[eval].eval_batch_size=64 run.eval.[eval_target_greedy].eval_batch_size=64"
SET "GSESBIG=run.eval.[greedy].eval_batch_size=64 run.eval.[greedy].eval_episodes=1250 run.eval.[greedy_single].eval_batch_size=64 run.eval.[greedy_single].eval_episodes=1250 run.eval_schedule.[greedy].interval=12 run.eval_schedule.[greedy_single].interval=12"

call :use_ws DropMerge-03 DropMerge.txt
REM 47a (closed by hand at 61M), 47b, 47c and 47d are done; skip them.
goto :q47det

call :wait_disk
echo === 47a. from scratch to 100M: TempoRL sep-merge, J=10, B256/RR1, in-training eval x8, Greedy full/single every 5M ===
call :find_done dmfb_47a_scr_skmrg
if not defined DONE_DIR call :run_exe "%SKBASE%" "%BATCH%" run.seed=1 %B100M% %MUNCH% "%SKMET%" %MERGE% %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47a_scr_skmrg"

call :wait_disk
echo === ev only 5000 x 5: 47a final (100M), Greedy full and single ===
call :find_ckpt "run_*_dmfb_47a_scr_skmrg"
if defined CKPT call :run_exe "%SKBASE%" "%BATCH%" run.seed=1 %MERGE% %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "%GSEL%" %GACT% %GSESBIG% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47a_ev_skmrg"

SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47b. from scratch to 100M: TempoRL off (control, 21_nzrep recipe), B256/RR1, in-training eval x8, Greedy full/single every 5M ===
call :find_done dmfb_47b_scr_ctl
if not defined DONE_DIR call :run_exe "%BASE%" "%BATCH%" run.seed=1 %B100M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47b_scr_ctl"

call :wait_disk
echo === ev only 5000 x 5: 47b final (100M) ===
call :find_ckpt "run_*_dmfb_47b_scr_ctl"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "%GSEL%" %GACT% %GSESBIG% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47b_ev_ctl"

:q47c
SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47c. from scratch to 50M: TempoRL off control, seed 2 (spread of 47b), same exe as 47b ===
call :find_done dmfb_47c_scr_ctl_s2
if not defined DONE_DIR call :run_exe "%BASE%" "%BATCH%" run.seed=2 %B50M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47c_scr_ctl_s2"

call :wait_disk
REM Eval-only keeps run.seed=1 so that its episodes match the 47a/47b eval-only runs.
echo === ev only 5000 x 5: 47c final (50M) ===
call :find_ckpt "run_*_dmfb_47c_scr_ctl_s2"
if defined CKPT call :run_exe "%BASE%" "%BATCH%" run.seed=1 %EVONLY% %GEVONLY% %GMET% %NDT20% %NZNET% "A3.auto_load_file=%CKPT%" %EVBIG% "%GSEL%" %GACT% %GSESBIG% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47c_ev_ctl_s2"

:q47det
SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === rg_e0L. 10/15 exe, regression-hunt config, deterministic, evals off, seed 1, to 30M ===
SET "RC_EXE=bin\%BUILD%\AnetRLRunner_ab12old.exe"
SET "RC_CFG=%CFG_OLD%"
call :run_cfg %DET% "app.runs_dir=%RG%\runs" "app.run_name=run_{t}_rg_e0L"

SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === rg_e7L. 11/01 exe (47b), regression-hunt config, deterministic, evals off, UQE epsilon 1e-4, seed 1, to 30M ===
SET "RC_EXE=bin\%BUILD%\AnetRLRunner_ab12.exe"
SET "RC_CFG=%CFG_NEW%"
call :run_cfg %DET% %DETNEW% "app.runs_dir=%RG%\runs" "app.run_name=run_{t}_rg_e7L"

:q47de
SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47d. 10/15 exe with the 40_s2 resolved config, seed 1, to 50M, in-training eval x8 ===
call :find_done dmfb_47d_ctl_old
if not defined DONE_DIR call :run_old run.seed=1 app.exp_exit_step=50000000 app.batchrun.exp_exit_step=50000000 %EV8% "app.runs_dir=%RUNS%" "app.run_name=run_{t}_dmfb_47d_ctl_old"

SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47e. 47b recipe with the UQE epsilon ladder at 1e-4 on all lanes, seed 1, to 50M ===
call :find_done dmfb_47e_ctl_eps1e4
if not defined DONE_DIR call :run_exe "%BASE%" "%BATCH%" run.seed=1 %B50M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EPS4% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47e_ctl_eps1e4"

SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47g. 47b recipe, seed 3, to 50M (run-to-run spread of the new build) ===
call :find_done dmfb_47g_scr_ctl_s3
if not defined DONE_DIR call :run_exe "%BASE%" "%BATCH%" run.seed=3 %B50M% %MUNCH% "%NZGMET%" %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47g_scr_ctl_s3"

SET "RUN_EXIT_CODE=0"
call :wait_disk
echo === 47f. TempoRL auxiliary-only (sep-merge, train and in-training evals single), seed 1, to 50M ===
call :find_done dmfb_47f_aux_mrg
if not defined DONE_DIR call :run_exe "%SKBASE%" "%BATCH%" run.seed=1 %B50M% %MUNCH% "%SKMET%" %MERGE% %AUX% %MTAU3% %GEV% %GMET% %SGMET% %NDT20% %NZNET% %NZACT% %NZREP% %EV8% "%GSEL%" %GACT% %GSES% %GMETG% %GMETS% "app.run_name=run_{t}_dmfb_47f_aux_mrg"

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

REM A 100M run writes about 3 GB; wait while C: has less than 6 GB free.
:wait_disk
SET "FREE_GB=0"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive C).Free/1GB)"') do SET "FREE_GB=%%f"
if %FREE_GB% LSS 6 (
  echo %DATE% %TIME% waiting for disk space: C: has %FREE_GB% GB free, need 6 GB
  ping -n 301 127.0.0.1 >nul
  goto :wait_disk
)
exit /b 0

:find_done
SET "DONE_DIR="
for /f "delims=" %%d in ('dir /b /ad "%RUNS%\run_*_%~1" 2^>nul') do (
  findstr /m /c:"Stop training requested" "%RUNS%\%%d\*.log" >nul 2>&1 && SET "DONE_DIR=%%d"
)
if defined DONE_DIR echo %DATE% %TIME% SKIP %~1 already finished in %DONE_DIR%
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

:run_exe
echo %DATE% %TIME% START %WS% %*
%RUNNER% %WS% %*
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_old
echo %DATE% %TIME% START old %*
%RUNNER_OLD% --config "%CFG_OLD%" %*
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_cfg
echo %DATE% %TIME% START "%RC_EXE%" --config "%RC_CFG%" %*
"%RC_EXE%" --config "%RC_CFG%" %*
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
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
goto :beep_forever

:beep_forever
powershell -NoProfile -Command "[console]::Beep(1000,800)" >nul 2>&1
ping -n 301 127.0.0.1 >nul
goto :beep_forever
