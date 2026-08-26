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
REM  COST (2,150 steps/s measured on Breakout, no GPU sharing)
REM    20M per run = about 2.6h  ->  5 games = about 13h  ->  2 arms = about 26h
REM    50M per run = about 6.5h  ->  5 games = about 32h  ->  2 arms = about 65h
REM ============================================================================

cd /d "%~dp0runner"

SET EXE="bin\Release\AnetRLRunner.exe" --workspace atari-5
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" --workspace atari-5

SET "BASE=run.@v5_iqn_impala_x2>run.@a5"

REM ---- A/B AXIS --------------------------------------------------------------

SET "ARM=run.$=%BASE%>run.@a5_base"
call:run_all_games

SET "ARM=run.$=%BASE%>run.@a5_apex"
call:run_all_games

REM ----------------------------------------------------------------------------

pause
exit /b


:run_all_games
call:run_game battle_zone
call:run_game double_dunk
call:run_game name_this_game
call:run_game phoenix
call:run_game qbert
exit /b


:run_game
call:run_exe "%ARM%" E1.game=%1
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
