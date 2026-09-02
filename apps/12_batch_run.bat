@echo off
REM ============================================================================
REM  atari-2nd  -  BTR structure ladder on the V/A-ReLU base.   about 13.2h
REM ----------------------------------------------------------------------------
REM  TWO-BATCH ROTATION. This is bat B. Edit 14_batch_run_nn.bat while this one
REM  runs, never this file - cmd.exe re-reads a batch by byte offset between
REM  lines, so editing a live .bat makes it jump into garbage.
REM
REM  THE BASE MOVED TO V/A ReLU ONLY (run_20260902-041010)
REM    headrelu changed three activations. The decomposition settled it:
REM      V/A ReLU only   eval 25-50M 472.8 +- 12.2 | wall 3.48% | >=600  80
REM      tau  ReLU only  eval 25-50M 453.1 +- 45.6 | wall 1.71% | >=600  32
REM      headrelu (all)  eval 25-50M 478.4 / 451.9 | wall 3.65 / 2.60%
REM      base pair       eval 25-50M 396.3 / 428.7 | wall 1.18 / 1.52%
REM    V/A lands inside the headrelu replicate pair, so on SCORE they are equal.
REM    The mechanism is not. At matched update window 113k-129k:
REM      probe_dormant   V/A .027 | tau .123 | headrelu .053-.067 | base .039
REM      action_churn    V/A .011 | tau .039 | headrelu .011-.013 | base .026
REM      probe_srank     V/A 480  | tau 460  | headrelu 471-473   | base 479
REM    The tau ReLU carries all the dormancy cost and none of the churn benefit.
REM    V/A ReLU alone is healthier than the base and scores the same as all
REM    three. So every arm here keeps V/A ReLU and leaves tau on SiLU.
REM    V/A eval is also the steadiest measured: 451.2 467.1 481.8 481.6 482.4.
REM
REM  BLOCKS 1-3 ARE A LADDER, ONE STEP APART
REM    iqn_fusion binds main_feature * tau_embedding, so the fusion dimension IS
REM    the main_feature dimension. Moving the shared trunk into
REM    iqn_fusion.structure lets the fusion point move on its own.
REM      1  Pool6 > Flatten(2304) > Linear512 > LN > ReLU > *tau(512)  > V/A
REM      2  Pool6 > Flatten(2304) > *tau(2304) > Linear512 > LN > ReLU > V/A
REM      3  Pool6 > Flatten(2304) > *tau(2304) > V/A(2304->512)          = BTR
REM    Block 1 is the control - it isolates what pool6 alone does on this base
REM    (it was neutral on the old SiLU base: 1.42% inside the 1.18-1.52 pair).
REM    Block 2 moves ONLY the fusion point. Block 3 also drops LN512, so 3 vs 2
REM    is two variables, not one. Read 2 first.
REM
REM  READING CAVEAT FOR BLOCK 3
REM    body.output.[features] = iqn_fusion, so the plasticity probe measures a
REM    2304-dim representation in block 3 and 512-dim elsewhere. 44_probe_srank
REM    and 43_probe_feature_norm are NOT comparable across that step. churn and
REM    the score metrics are unaffected.
REM
REM  BLOCK 4 IS hard UPDATE AT THE OPERATING POINT
REM    C=500 only carries BTR's meaning at RR4 (their 1 grad = 64 env steps, so
REM    C=500 is every 32,000 exp steps). At RR1, 1 grad = 256 exp steps, so the
REM    env-time-matched value is C=125. RR4 is 4x the compute for a quarter of
REM    the result now (RR4 best 0.80% vs RR1 3.48%), so this replaces the RR4
REM    hard arm. 15_target_sync_age emits values only under hard update.
REM
REM  BLOCK 5 LIGHTS UP DEAD INSTRUMENTS
REM    39_agent_per/05_sample_actor_init_ratio, 50_actor_init_mass_ratio and
REM    52_actor_learner_pair_count read 0 in every Atari run so far because
REM    per_initial_priority_mode = max leaves the actor path inactive.
REM    actor_approx is the only mode that exercises them. CHECK 52 - under
REM    NDEBUG a non-finite hint falls back to max silently, which would mean
REM    measuring max while believing it is actor_approx.
REM
REM  WIRING CHECK: blocks 1-5 are all new profiles, and blocks 2-3 use a new
REM  branch and a new block. Block 0 dumps all five in 15 minutes. Confirm:
REM    1  main_feature has AtariPool6, ends ... AtariLinear512 > AtariLN512 > ReLU
REM    2  main_feature ENDS AT Flatten | tau_embedding has AtariIQNTauProj2304
REM       | iqn_fusion.structure = AtariLinear512 > AtariLN512 > ReLU
REM    3  same as 2 but iqn_fusion.structure EMPTY
REM    4  soft_update_tau 0, hard_update_interval 125
REM    5  per_initial_priority_mode actor_approx
REM  All five must show value_stream / adv_stream ending in ReLU and
REM  tau_embedding ending in SiLU.
REM
REM  JUDGE ON THE WALL, NOT ON eval1, against the calibrated spread.
REM    Same-config replicate pairs measured so far:
REM      base      1.18 vs 1.52 percent (192 vs 244 episodes)
REM      headrelu  2.60 vs 3.65 percent (393 vs 530 episodes)
REM    Anything inside the relevant pair is not a difference. Use
REM    42_env/10_game_score_mean pooled over 25-50M, count >= 432 and >= 600.
REM    Note >=600 is noisier than the wall rate: the headrelu pair split
REM    110 vs 54 on it.
REM
REM  ARM MUST STAY QUOTED. The chain contains '>', which cmd.exe would otherwise
REM  treat as output redirection - both in SET and at the call site.
REM ============================================================================

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"
REM SET "BUILD=Release"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-2nd

SET "FIX1=backend.$=backend.@non-deterministic"
SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex"

echo === 0. wiring check x5 - VERIFY ALL FIVE IN THE DUMPS (15 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_pool6>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_fuse2304>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btrstruct>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_hard125>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_aainit>run.@pl_check"

echo === 1. V/A ReLU + 6x6 pool - the control for blocks 2-3, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_pool6"

echo === 2. tau fusion moved to 2304, trunk kept - one variable, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_fuse2304"

echo === 3. BTR structure - no shared trunk, V/A from 2304, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrstruct"

echo === 4. hard update C=125 - env-time matched to BTR at RR1, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125"

echo === 5. actor_approx initial priority - Ape-X style, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_aainit"

echo === ALL DONE ===
pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %* %FIX1% %FIX2%
echo   %DATE% %TIME% END   %*
exit /b


:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
