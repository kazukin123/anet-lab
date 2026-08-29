@echo off
REM ============================================================================
REM  Plasticity collapse assay - standing harness
REM ----------------------------------------------------------------------------
REM  THE ASSAY
REM    RR8 / 5M / breakout / old bundle. eval1 peaks in the 2.0-2.5M window and
REM    falls to about 31. 54 min per run. Every arm is that assay with one thing
REM    changed, so "did the collapse point move" is the read.
REM
REM    IT IS 4/5, NOT 5/5. One run out of five identical-config runs did not
REM    collapse. An n=1 arm cannot be judged - always pair a new arm with at
REM    least one replicate.
REM
REM  CURRENT OPEN ARM
REM    group + WD 0.3 was the only cell that recovered at 5M (peak 74, end 73)
REM    but it is n=1. The control has a 20 pct natural non-collapse rate, so it
REM    is not yet distinguishable. This batch replicates it against the control.
REM
REM  WHAT TO LOOK AT
REM    34_agent_plasticity/42_probe_dead_ratio  - the trough is the turning point
REM    34_agent_plasticity/61_weight_norm_feature - the floor lands in the same window
REM    Both land just after the eval1 peak. Compare within an arm over time;
REM    the LEVEL does not compare across arms.
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

SET EXE="bin\%BUILD%\AnetRLRunner_ab.exe" --workspace plasticity

SET "BASE=run.@breakout_rr1_100m>run.@plasticity>run.@plasticity_rr8"
SET "GN=%BASE%>run.@pl_gn"

echo === 1. control  (norm none, WD 0) ===
call:run_exe "run.$=%BASE%"

echo === 2. group + WD 0.3  x2 ===
call:run_exe "run.$=%GN%>run.@pl_gn_wd030"
call:run_exe "run.$=%GN%>run.@pl_gn_wd030"

echo === ALL DONE ===
pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b


:no_exe
echo *** bin\%BUILD%\AnetRLRunner.exe not found or copy failed. Nothing was run.
pause
exit /b 1
