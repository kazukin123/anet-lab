@echo off

cd /d "%~dp0runner"

SET "BUILD=RelWithDebInfo"

if not exist "bin\%BUILD%\AnetRLRunner.exe" goto :no_exe
copy /Y "bin\%BUILD%\AnetRLRunner.exe" "bin\%BUILD%\AnetRLRunner_ab.exe" >nul
if errorlevel 1 goto :no_exe

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace atari-3rd

SET /A SUCCEEDED_RUNS=0
SET /A FAILED_RUNS=0

SET "FIX2=E1.game=breakout"

SET "A5=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base"
SET "RR4=run.@hard500>run.@rr4>run.@munch"
SET "RF=run.@rfit"
SET "NET=run.@btrnet>run.@btrsn"

echo === 1. RR4 + Pool4 trunk SN 20M (4h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrtrunkpool4>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunkpool4_20m"
echo === 2. RR4 + BTR net + SN, num_envs 128, 50M (8.5h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET%>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrfull_envs128_50m"
echo === 3. walltime preset: RR1 + kyu-net 100M (7-8h) ===
call :run_exe "run.$=%A5%>run.@hard125>run.@munch>%RF%>run.@to_100m>run.@btreval_r1" "app.run_name=run_{t}_rr1_va_hard125_100m"
echo === 4. RR4 + Pool3 trunk SN 20M (4h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>run.@btrtrunkpool3>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrtrunkpool3_20m"
echo === 5. RR4 + BTR net + SN + actor_approx 20M (3.5h) ===
call :run_exe "run.$=%A5%>%RR4%>%RF%>%NET%>run.@aainit>run.@envs64>run.@a5_20m>run.@btreval_r4" "app.run_name=run_{t}_rr4_btrfull_aainit_20m"

if "%FAILED_RUNS%"=="0" goto :all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED, %FAILED_RUNS% FAILED ===
pause
exit /b 1

:all_succeeded
echo === ALL DONE: %SUCCEEDED_RUNS% SUCCEEDED ===
pause
exit /b 0


:run_exe
echo %DATE% %TIME% START %*
%EXE% %* %FIX2%
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
