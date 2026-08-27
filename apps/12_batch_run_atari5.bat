@echo off
REM ============================================================================
REM  Atari-5 batch A/B
REM ----------------------------------------------------------------------------
REM  Atari-5 (battle_zone / double_dunk / name_this_game / phoenix / qbert) is a
REM  subset that predicts the 57-game median. Use it to check that a setting
REM  tuned on Breakout generalizes instead of overfitting to one game.
REM
REM  Breakout is a poor lone judge: reward_clip=true makes every brick worth +1
REM  to the learner while game_score counts 1/4/7, and the 432/864 wall creates a
REM  phase transition no other game has.
REM
REM  SETTINGS LIVE IN CONFIG, NOT HERE.
REM    Protocol / NN / budget / eval are held by the run.@a5* trunks in
REM    apps\runner\config\Atari.txt. This file only picks a trunk and loops games.
REM    Budget: change the @vars pair inside run.@a5 (per_beta_step follows).
REM
REM  HOW TO EDIT
REM    1. A/B AXIS block - one SET ARM / call pair per arm.
REM    2. :run_all_games - comment out games to shorten the sweep.
REM
REM  ARM MUST STAY QUOTED. The trunk chain contains '>', which cmd.exe would
REM  otherwise treat as output redirection - both in SET and at the call site.
REM
REM  COST (Atari-5 measured 2026-08-27: 5,335 - 5,905 steps/s at RR1, 20M in about 57 min/game)
REM    20M per run -> 5 games = about 4.8h   (measured)
REM    50M per run -> 5 games = about 12h    (current budget, extrapolated)
REM    RR4 arm is about 2.6x slower per run.
REM  20M was too short: battle_zone / name_this_game / qbert were all still climbing.
REM ============================================================================

cd /d "%~dp0runner"

SET EXE="bin\Release\AnetRLRunner.exe" --workspace atari-5
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" --workspace atari-5

SET "BASE=run.@v5_iqn_impala_x2>run.@a5"

REM ---- A/B AXIS --------------------------------------------------------------

REM STAGE 1: wiring check + per-game baseline (about 4.2h)
SET "ARM=run.$=%BASE%>run.@a5_apex"
call:run_all_games

REM STAGE 2: does RR1 beat RR4 outside Breakout? BTR uses the RR4 equivalent.
REM   Uncomment the two lines below (keep STAGE 1 as the first arm).
REM SET "ARM=run.$=%BASE%>run.@a5_apex>run.@a5_rr4"
REM call:run_all_games

REM LADDER A/B: settled on Breakout (block 20). Kept for cross-env recheck.
REM SET "ARM=run.$=%BASE%>run.@a5_base"
REM call:run_all_games

REM ----------------------------------------------------------------------------

pause
exit /b


REM Order: readable first, then by Atari-5 regression weight, with phoenix pinned last.
REM   qbert is the clearest signal; battle_zone + name_this_game carry 69 percent of the
REM   median estimate; double_dunk has the lowest weight (0.068) and stays below random at 20M.
REM
REM   PHOENIX MUST STAY LAST until 999_batchrun_fatal_error_handling is implemented.
REM   On 2026-08-27 it died with bad_alloc at 12.3M/20M after its episodes hit the 27,000
REM   step truncation cap. A modal error dialog then blocked the queue for 13m48s. Running
REM   it last means a repeat delays nothing.
:run_all_games
call:run_game qbert
call:run_game battle_zone
call:run_game name_this_game
call:run_game double_dunk
call:run_game phoenix
exit /b


:run_game
call:run_exe "%ARM%" E1.game=%1
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
