@echo off
REM ============================================================================
REM  A1 delta + A3 normalization x weight_decay - overnight, 12 runs, about 11h
REM ----------------------------------------------------------------------------
REM  UNATTENDED. Start it and leave.
REM
REM  THE COLLAPSE ASSAY
REM    RR8 / 5M / breakout / old bundle collapses reliably: eval1 peaks in the
REM    2.0-2.5M window and falls to about 31. Three runs on 2026-08-28 all peaked
REM    in that same window. 54 min per run. Every arm is that assay with one
REM    thing changed, so "did the collapse point move" is the read.
REM
REM  WHY NORMALIZATION IS THE AXIS
REM    The Atari body currently has NO normalization anywhere - Conv2d, MaxPool,
REM    ResBlock(norm_type=none), Linear, ReLU. That matters, because after a
REM    GroupNorm the scale of the preceding conv weights divides out: (x-mu)/sigma
REM    is unchanged if you scale x. So with normalization, ||w|| growth is
REM    functionally inert and only moves the effective learning rate. Without it,
REM    ||w|| growth changes the function directly.
REM
REM    That predicts block 13's result exactly: weight_decay 0.1 cost -43.4 pct
REM    at RR4/50M, which is what you get when you shrink weights that the
REM    function actually depends on. So testing weight_decay ALONE tests it in
REM    the regime where it is expected to fail. The 2x2 separates the two.
REM
REM    measured on 2026-08-28: weight_norm_feature grows 1.47x over the RR8
REM    collapse while weight_norm_readout grows 1.10x. The question is whether
REM    that growth is a cause or a symptom.
REM
REM  THE 2x2   (norm=none, WD=0 is already in hand - runs 062103 / 094914 / 155424)
REM    none  + WD 0.1     connects to block 13
REM    group + WD 0       does normalization alone move the collapse?
REM    group + WD 0.1     the recipe from the literature
REM    plus a WD dose ladder under normalization: 0.01 / 0.03 / 0.3
REM
REM  ORDER
REM    Distinct cells first, then replicates, then the long durability arm. If it
REM    stops early, the cells that answer the question are the ones already done.
REM
REM  WIRING CHECK FIRST
REM    Run 1 is 100k exp, about 2 min. GroupNorm goes in by swapping the branch
REM    profile (net.branch.@AtariImpalaX2GN), which is the same mechanism
REM    run.@breakout_rr1_100m already uses - a typo fails loudly rather than
REM    silently building the wrong network. The check confirms it anyway.
REM    Verify with: inspect_run.py config <check run> --config-key '*Res32GN*'
REM
REM  WHAT TO LOOK AT
REM    A1  : rows 41/42 (delta 0.01) vs 46/47 (0.05) vs 48/49 (0.20). Any dip?
REM    A3  : 06_weight_norm_feature slope, and which window eval1 peaks in.
REM          WD=0/none peaks in the 40-50 pct window. A shift is the finding.
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

SET "BASE=run.@breakout_rr1_100m>run.@plasticity>run.@plasticity_rr8>run.@pl_delta"
SET "GN=%BASE%>run.@pl_gn"

echo === 0. GroupNorm wiring check (about 2 min) ===
call:run_exe "run.$=%GN%>run.@pl_check"

echo === 1. A1 delta  (also the none/WD=0 control) ===
call:run_exe "run.$=%BASE%>run.@pl_a1_wd0"

echo === 2. 2x2 distinct cells ===
call:run_exe "run.$=%GN%>run.@pl_gn_wd0"
call:run_exe "run.$=%GN%>run.@pl_gn_wd010"
call:run_exe "run.$=%BASE%>run.@pl_wd010"

echo === 3. WD dose ladder under normalization ===
call:run_exe "run.$=%GN%>run.@pl_gn_wd001"
call:run_exe "run.$=%GN%>run.@pl_gn_wd003"
call:run_exe "run.$=%GN%>run.@pl_gn_wd030"

echo === 4. replicates of the three cells ===
call:run_exe "run.$=%GN%>run.@pl_gn_wd0"
call:run_exe "run.$=%GN%>run.@pl_gn_wd010"
call:run_exe "run.$=%BASE%>run.@pl_wd010"

echo === 5. durability: group + WD 0.1 at 10M (about 1.8h) ===
call:run_exe "run.$=%GN%>run.@pl_gn_wd010>run.@pl_gn_10m"

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
