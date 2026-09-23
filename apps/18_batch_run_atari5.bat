@echo off
cd /d "%~dp0runner"

SET EXE="bin\Release\AnetRLRunner.exe" --workspace atari5-01
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe" --workspace atari5-01

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR1=run.@hard125>run.@munch"
SET "RF=run.@rfit>run.@a5_metrics"
SET "ARM=run.$=%A5%>%RR1%>%RF%>run.@cap2m>run.@eval2ch_r1>run.@to_100m>run.@batch"
SET "SEED=2"
call:run_all_games

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, 0 FAILED ===
pause
exit /b 0


:run_all_games
call:run_game qbert
call:run_game battle_zone
call:run_game name_this_game
call:run_game double_dunk
call:run_game phoenix
exit /b


:run_game
call:run_exe "%ARM%" E1.game=%1 run.seed=%SEED% "app.run_name=run_{t}_a5_%1_rr1_100m_seed%SEED%"
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
SET "RUN_EXIT_CODE=%ERRORLEVEL%"
if "%RUN_EXIT_CODE%"=="0" goto :run_succeeded
echo %DATE% %TIME% [ERROR] RUN FAILED exit_code=%RUN_EXIT_CODE% args=%*
SET /A FAILED_RUNS+=1
exit /b 0

:run_succeeded
SET /A SUCCEEDED_RUNS+=1
echo   %DATE% %TIME% END   %*
exit /b 0
