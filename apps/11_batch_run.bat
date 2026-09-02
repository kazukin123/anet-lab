@echo off
REM ============================================================================
REM  atari-2nd  -  NN structure first.   about 15.4h
REM ----------------------------------------------------------------------------
REM  TWO-BATCH ROTATION. This is bat A. Edit 15_batch_run_nn_b.bat while this
REM  one runs, never this file - cmd.exe re-reads a batch by byte offset between
REM  lines, so editing a live .bat makes it jump into garbage.
REM
REM  WHY NN FIRST
REM    Every positive result in this campaign came from the network, none from
REM    the learner hyperparameters.
REM      LN512      RR1 wall rate 0.47 -> 1.32 percent
REM      head ReLU  wall rate 1.35 (pair mean) -> 3.65 percent
REM      6x6 pool   neutral
REM      SN + he    0.43 percent, below the unprotected control
REM      per_beta   neutral | PER fixed 0.3  0.75 percent | BTR exploration  eval declining
REM    Four architecture arms produced two large wins; four learner arms
REM    produced none.
REM
REM  THE BASE IS NOW head/fusion ReLU (run_20260901-115207)
REM    eval1 25-50M 478.4 +- 42.6 | 45-50M 490.3
REM    train 25-50M pooled: n 14531, mean 209.5, >=432 3.65%, >=600 110
REM    eval 25-50M distribution: NOTHING below 200 in 98 episodes, p10 381,
REM      p50 428, p90 699, one 864 (two racks cleared), >=432 43.9%
REM    still climbing at 50M: 35-40M 3.38% -> 40-45M 5.57% -> 45-50M 5.66%
REM    n=1. Block 1 doubles as its replicate - see below.
REM
REM  BLOCK 1 DOES TWO JOBS AT ONCE
REM    per_beta is now fixed at 0.2, so half_exp_step has no consumer left
REM    (per_beta_step was the only one, Atari.txt:194) and max_exp_step only
REM    sets the stop point. A 100M run's first 50M is therefore a strict prefix
REM    of the matching 50M run - same seed, same everything, later stop.
REM    So block 1 gives the headrelu REPLICATE at 50M and the extension to 100M
REM    in one 5.1h run.
REM
REM  BLOCKS 2 AND 3 SPLIT headrelu
REM    headrelu changed three activations at once. Block 2 changes only the IQN
REM    tau embedding, block 3 only the V/A hidden layers. If block 2 alone
REM    reproduces the win, the mechanism is IQN-specific: ReLU turns the cos
REM    embedding product from a smooth rescale into a gate.
REM
REM  BLOCK 4 IS RR4 ON PURPOSE
REM    C=500 only carries BTR's meaning at RR4. BTR's --rr 1 is one grad step
REM    per 64 env steps, so C=500 means "every 32,000 exp steps". Copying C=500
REM    to our RR1 (one grad step per 256 exp steps) would be 128,000 exp steps,
REM    a 4x longer lag. The env-time-matched RR1 version would be C=125.
REM    This is also the only arm where 15_target_sync_age emits values at all.
REM
REM  WIRING CHECK: blocks 2 and 3 are new profiles. Block 0 dumps both.
REM    block 2: tau_embedding ends in ReLU, value_stream/adv_stream stay SiLU
REM    block 3: value_stream/adv_stream end in ReLU, tau_embedding stays SiLU
REM  Block 1 reuses run.@rr1_ln_headrelu (verified) plus run.@to_100m, which
REM  only moves max_exp_step / half_exp_step. Block 4 was dumped on 08-31.
REM
REM  JUDGE ON THE WALL, NOT ON eval1, and against the calibrated spread.
REM    Same-config replicate pair measured 2026-09-01: eval 25-50M 396.3 vs
REM    428.7, wall 192 vs 244 episodes (1.18 vs 1.52 percent). Anything inside
REM    that is not a difference. Use 42_env/10_game_score_mean pooled over
REM    25-50M and count >= 432 (one Breakout screen) and >= 600.
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
SET "RR4=%A5%>run.@a5_rr4"

echo === 0. wiring check x2 - VERIFY hr_tau / hr_va IN THE DUMPS (6 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_ln_hr_tau>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_ln_hr_va>run.@pl_check"

echo === 1. head/fusion ReLU at 100M - replicate + extension (5.1h) ===
call:run_exe "run.$=%A5%>run.@rr1_ln_headrelu>run.@to_100m"

echo === 2. ReLU on the IQN tau embedding ONLY, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_ln_hr_tau"

echo === 3. ReLU on the V/A hidden layers ONLY, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_ln_hr_va"

echo === 4. RR4 + LN512 + BTR-faithful hard C=500, 50M (5.1h) ===
call:run_exe "run.$=%RR4%>run.@rr4_ln_hard"

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
