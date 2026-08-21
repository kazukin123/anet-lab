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
REM  HOW TO EDIT
REM    1. BUDGET  - steps per run.
REM    2. A/B AXIS block - one SET LABEL / SET VARIANT / call trio per arm.
REM       VARIANT is passed verbatim as config overrides. Empty means baseline.
REM    3. GAMES - comment out lines in :run_all_games to shorten the sweep.
REM
REM  COST (Breakout, IMPALA, num_envs 128, about 2000 steps/s, no GPU sharing)
REM    50M per run = about 6.9h  ->  2 arms x 5 games = about 69h
REM    20M per run = about 2.8h  ->  2 arms x 5 games = about 28h
REM
REM  WARNING - BUDGET interacts with per_beta_step
REM    Atari.txt fixes per_beta_step at 50M, so beta reaches 1.0 only at 50M.
REM    On Breakout the takeoff happened right as beta hit 1.0 (46M onward).
REM    With a 20M budget beta only reaches 0.64, so the sweep never leaves the
REM    pre-takeoff regime and every arm looks flat. Either keep BUDGET at 50M,
REM    or scale per_beta_step with it and accept that you are then testing a
REM    different schedule than the Breakout baseline.
REM
REM  Numbers are written without commas on purpose - cmd.exe treats commas as
REM  argument separators in some contexts.
REM ============================================================================

cd /d "%~dp0runner"

REM ---- workspace config -------------------------------------------------------
REM  --workspace creates a missing workspace from config\_workspace_template.txt,
REM  whose default include is LunarLander. Without this block a fresh clone would
REM  run the whole sweep on the wrong env and never report an error.
SET WS=atari-5
SET WSCONF=workspaces\%WS%\config\_main.txt

if not exist "%WSCONF%" (
    if not exist "workspaces\%WS%\config" mkdir "workspaces\%WS%\config"
    >"%WSCONF%" echo $include ^<Atari.txt^>
    echo Created %WSCONF%
)

findstr /B /C:"$include <Atari.txt>" "%WSCONF%" >nul
if errorlevel 1 (
    echo ERROR: %WSCONF% does not select Atari.txt.
    echo Delete the file and rerun to regenerate it, or fix the include by hand.
    pause
    exit /b 1
)

SET EXE="bin\Release\AnetRLRunner.exe" --workspace atari-5 app.$=app.batchrun
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" --workspace atari-5 app.$=app.batchrun

SET BUDGET=app.batchrun.exp_exit_step=50000000
REM SET BUDGET=app.batchrun.exp_exit_step=20000000
REM SET BUDGET=app.batchrun.exp_exit_step=2000000

REM ---- A/B AXIS --------------------------------------------------------------

SET LABEL=base
SET VARIANT=
call:run_all_games

SET LABEL=tau30
SET VARIANT=X.learner.grad_clip_tau=30
call:run_all_games

REM  Other axes, for later. Keep exactly one thing different per arm.
REM    SET VARIANT=A.train_policy.eps_end=0.001
REM    SET VARIANT=X.learner.per_beta_end=0.55
REM    SET VARIANT=A.learner.per_beta_step=20000000
REM    SET VARIANT=net.branch.[main_feature].$=net.branch.AtariImpalaViT
REM    SET VARIANT=train.seed=2
REM    SET VARIANT=AtariEnv.$=AtariEnv.classic

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
call:run_exe app.run_name=run_{t}_a5-%1-%LABEL% E.game=%1 %BUDGET% %VARIANT%
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
