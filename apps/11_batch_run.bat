@echo off
title %~nx0

cd /d "%~dp0runner"

SET "BUILD=Release"
if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe
if not exist "bin\%BUILD%\AnetRLRunner_ab.exe" goto :no_exe

SET RUNNER="bin\%BUILD%\AnetRLRunner_ab.exe"

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "LLMRG=run.$=run.@munch>run.@skip"
SET "LLOFF=run.$=run.@munch"
SET "LLAPP=app.$=app.batchrun>P1"
SET "EVONLY=run.seed=1 DefaultDQNAgent.learner.enabled=false app.save_agent_on_close=false run.train.num_envs=1 app.batchrun.exp_exit_step=40 run.eval_schedule.[eval_target].interval=0"
SET "EVUQE=run.eval.[eval].eval_batch_size=64 run.eval.[eval].eval_episodes=400 run.eval_schedule.[eval].interval=12"
SET "EVGRD=run.eval.[greedy].eval_batch_size=64 run.eval.[greedy].eval_episodes=400 run.eval_schedule.[greedy].interval=12"
SET "EVGS=DefaultDQNAgent.actor.[greedy_single].$=DefaultDQNAgent.actor.[greedy] DefaultDQNAgent.actor.[greedy_single].skip.max_length_mode=single run.eval.[greedy_single].actor=greedy_single run.eval.[greedy_single].eval_batch_size=64 run.eval.[greedy_single].eval_episodes=400 run.eval_schedule.[greedy_single].interval=12 run.eval_schedule.[greedy_single].use_background=true"
SET "EVUS=DefaultDQNAgent.actor.[eval_single].$=DefaultDQNAgent.actor.[eval] DefaultDQNAgent.actor.[eval_single].skip.max_length_mode=single run.eval.[eval_single].actor=eval_single run.eval.[eval_single].eval_batch_size=64 run.eval.[eval_single].eval_episodes=400 run.eval_schedule.[eval_single].interval=12 run.eval_schedule.[eval_single].use_background=true"
SET "GSEL=DefaultDQNAgent.actor.[greedy].$=DefaultDQNAgent.actor.[eval]>DefaultDQNAgent.actor.@greedy"
SET "GOFF=run.eval.[greedy].actor=greedy run.eval_schedule.[greedy].use_background=true"
SET "MGR="metrics.scalar.[21_eval/09_greedy_reward]=mean.episode_return $runner @session_end $eval.[greedy]" "metrics.scalar.[21_eval/11_greedy_ep_steps]=mean.episode_steps $runner @session_end $eval.[greedy]""
SET "MGS="metrics.scalar.[21_eval/31_greedy_s_reward]=mean.episode_return $runner @session_end $eval.[greedy_single]" "metrics.scalar.[21_eval/32_greedy_s_steps]=mean.episode_steps $runner @session_end $eval.[greedy_single]""
SET "MUS="metrics.scalar.[21_eval/33_eval_s_reward]=mean.episode_return $runner @session_end $eval.[eval_single]" "metrics.scalar.[21_eval/34_eval_s_steps]=mean.episode_steps $runner @session_end $eval.[eval_single]""
SET "MSK="metrics.scalar.[50_agent_skip/31_eval_decision_rate]=skip_session_decision_rate $eval.[eval] $actor @session_end" "metrics.scalar.[50_agent_skip/32_eval_length_mean]=skip_session_length_mean $eval.[eval] $actor @session_end""

SET "TWSES=DefaultDQNAgent.actor.[greedy_single].$=DefaultDQNAgent.actor.[greedy] DefaultDQNAgent.actor.[greedy_single].skip.max_length_mode=single run.eval.[greedy_single].actor=greedy_single run.eval.[greedy_single].eval_batch_size=1 run.eval.[greedy_single].eval_episodes=1 run.eval_schedule.[greedy_single].interval=1009 run.eval_schedule.[greedy_single].use_background=true DefaultDQNAgent.actor.[eval_single].$=DefaultDQNAgent.actor.[eval] DefaultDQNAgent.actor.[eval_single].skip.max_length_mode=single run.eval.[eval_single].actor=eval_single run.eval.[eval_single].eval_batch_size=1 run.eval.[eval_single].eval_episodes=1 run.eval_schedule.[eval_single].interval=200 run.eval_schedule.[eval_single].use_background=true"
SET "TWMET="metrics.scalar.[50_agent_skip/07_max_length]=$actor skip_max_length @learn $exp_step""
SET "TWVAR=DefaultDQNAgent.actor.[train].skip.max_length_mode=variable DefaultDQNAgent.actor.[train].skip.max_length.type=phased "DefaultDQNAgent.actor.[train].skip.max_length.phases=hold off" DefaultDQNAgent.actor.[train].skip.max_length.phase.[hold].type=constant DefaultDQNAgent.actor.[train].skip.max_length.phase.[hold].value=4 DefaultDQNAgent.actor.[train].skip.max_length.phase.[off].type=constant DefaultDQNAgent.actor.[train].skip.max_length.phase.[off].value=1 DefaultDQNAgent.actor.[train].skip.max_length.phase.[off].steps=1"
SET "TWWU=DefaultDQNAgent.actor.[train].skip.max_length.phase.[hold].steps=8192"
SET "TWE05=DefaultDQNAgent.actor.[train].skip.max_length.phase.[hold].steps=500000"
SET "TWUNI=DefaultDQNAgent.actor.[train].skip.policy.policy_type=EpsilonGreedy DefaultDQNAgent.actor.[train].skip.policy.spatial.epsilon.first=1 DefaultDQNAgent.actor.[train].skip.policy.spatial.epsilon.last=1"

call :use_ws LunarLander-01 LunarLander.txt
echo === 1. LunarLander eval only, merge seed 1 checkpoint: UQE and Greedy, full and single ===
call :find_done ll_ev_mrg_s1
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% %EVGS% %EVUS% %MGS% %MUS% %MSK% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261028-131118_t103_ll_mrg_s1\agent_close.anet" "app.run_name=run_{t}_ll_ev_mrg_s1"
echo === 2. LunarLander eval only, merge seed 2 checkpoint: UQE and Greedy, full and single ===
call :find_done ll_ev_mrg_s2
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% %EVGS% %EVUS% %MGS% %MUS% %MSK% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261028-143129_t103_ll_mrg_s2\agent_close.anet" "app.run_name=run_{t}_ll_ev_mrg_s2"
echo === 3. LunarLander eval only, merge seed 3 checkpoint: UQE and Greedy, full and single ===
call :find_done ll_ev_mrg_s3
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% %EVGS% %EVUS% %MGS% %MUS% %MSK% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261028-155145_t103_ll_mrg_s3\agent_close.anet" "app.run_name=run_{t}_ll_ev_mrg_s3"
echo === 4. LunarLander eval only, no skip seed 1 checkpoint: UQE and Greedy ===
call :find_done ll_ev_off_s1
if not defined DONE_DIR call :run_exe "%LLOFF%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% "%GSEL%" %GOFF% %MGR% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261025-230909_t98_ll_off_s1\agent_close.anet" "app.run_name=run_{t}_ll_ev_off_s1"
echo === 5. LunarLander eval only, no skip seed 2 checkpoint: UQE and Greedy ===
call :find_done ll_ev_off_s2
if not defined DONE_DIR call :run_exe "%LLOFF%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% "%GSEL%" %GOFF% %MGR% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261026-001350_t98_ll_off_s2\agent_close.anet" "app.run_name=run_{t}_ll_ev_off_s2"
echo === 6. LunarLander eval only, no skip seed 3 checkpoint: UQE and Greedy ===
call :find_done ll_ev_off_s3
if not defined DONE_DIR call :run_exe "%LLOFF%" "%LLAPP%" %EVONLY% %EVUQE% %EVGRD% "%GSEL%" %GOFF% %MGR% "DefaultDQNAgent.auto_load_file=%RUNS%\run_20261026-010415_t98_ll_off_s3\agent_close.anet" "app.run_name=run_{t}_ll_ev_off_s3"

echo === 7. LunarLander 2M seed 1: merge, train skip length uniform random ===
call :find_done ll_tw_uni_s1
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=1 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWUNI% "app.run_name=run_{t}_ll_tw_uni_s1"
echo === 8. LunarLander 2M seed 1: merge, train repeats only until 8192 experiences ===
call :find_done ll_tw_wu_s1
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=1 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWWU% "app.run_name=run_{t}_ll_tw_wu_s1"
echo === 9. LunarLander 2M seed 1: merge, train repeats only until 500000 experiences ===
call :find_done ll_tw_e05_s1
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=1 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWE05% "app.run_name=run_{t}_ll_tw_e05_s1"
echo === 10. LunarLander 2M seed 1: merge baseline ===
call :find_done ll_tw_base_s1
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=1 %TWSES% %MGS% %MUS% %MSK% %TWMET% "app.run_name=run_{t}_ll_tw_base_s1"
echo === 11. LunarLander 2M seed 2: merge, train repeats only until 8192 experiences ===
call :find_done ll_tw_wu_s2
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=2 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWWU% "app.run_name=run_{t}_ll_tw_wu_s2"
echo === 12. LunarLander 2M seed 2: merge, train repeats only until 500000 experiences ===
call :find_done ll_tw_e05_s2
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=2 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWE05% "app.run_name=run_{t}_ll_tw_e05_s2"
echo === 13. LunarLander 2M seed 2: merge baseline ===
call :find_done ll_tw_base_s2
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=2 %TWSES% %MGS% %MUS% %MSK% %TWMET% "app.run_name=run_{t}_ll_tw_base_s2"
echo === 14. LunarLander 2M seed 2: merge, train skip length uniform random ===
call :find_done ll_tw_uni_s2
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=2 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWUNI% "app.run_name=run_{t}_ll_tw_uni_s2"
echo === 15. LunarLander 2M seed 3: merge, train repeats only until 500000 experiences ===
call :find_done ll_tw_e05_s3
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=3 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWE05% "app.run_name=run_{t}_ll_tw_e05_s3"
echo === 16. LunarLander 2M seed 3: merge baseline ===
call :find_done ll_tw_base_s3
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=3 %TWSES% %MGS% %MUS% %MSK% %TWMET% "app.run_name=run_{t}_ll_tw_base_s3"
echo === 17. LunarLander 2M seed 3: merge, train skip length uniform random ===
call :find_done ll_tw_uni_s3
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=3 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWUNI% "app.run_name=run_{t}_ll_tw_uni_s3"
echo === 18. LunarLander 2M seed 3: merge, train repeats only until 8192 experiences ===
call :find_done ll_tw_wu_s3
if not defined DONE_DIR call :run_exe "%LLMRG%" "%LLAPP%" run.seed=3 %TWSES% %MGS% %MUS% %MSK% %TWMET% %TWVAR% %TWWU% "app.run_name=run_{t}_ll_tw_wu_s3"

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
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
