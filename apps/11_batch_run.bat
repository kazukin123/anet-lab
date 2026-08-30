@echo off
REM ============================================================================
REM  atari-2nd  -  SN screening, BTR-faithful scope
REM ----------------------------------------------------------------------------
REM  WHY THE SCOPE CHANGED  (2026-08-30, run_20260830-131655_c_spectral)
REM    The all-layers arm DIVERGED. q_max went 0.89 -> 12 -> 267 -> 2700 between
REM    exp 660k and 1014k, td_mean reached 7.9e4, grad_clip_ratio pinned at 1.0.
REM
REM    SN ITSELF WORKED. Over the same window:
REM        61 raw norm        49.4 -> 53.2   still growing, as designed
REM        63 effective norm  28.2 -> 26.5   FLAT - stage (1) finally braked
REM        65 sigma           1.88 -> 3.50   the clamp is engaging
REM    That is the three-point signature the PRD predicted, and it is the first
REM    time stage (1) has been stopped in any arm. The weights did not blow up;
REM    the value function did.
REM
REM    The likely cause is scope. SN pinned the feature's effective norm at ~27
REM    and stopped it growing, but gamma=0.997 needs q_max around 12 (measured
REM    on the control). 64_weight_norm_readout_effective equals 62 exactly, i.e.
REM    the readout is NOT under SN, so the whole burden of producing large Q
REM    lands on an unconstrained head fed by a frozen-scale feature.
REM    A second candidate: 'spectral' always divides, so layers with sigma < 1
REM    get AMPLIFIED. 65 reports the MAX sigma over the group, so per-layer
REM    sigma < 1 is invisible here. Separating the two needs per-layer sigma.
REM
REM  WHY NOT PUT SN ON THE HEAD
REM    The head is what sets the value scale. Projecting it to sigma=1 removes
REM    the freedom to represent q_max ~ 12 at all. It would remove the escape
REM    route rather than fix the pressure.
REM
REM  BTR-FAITHFUL SCOPE IS THE ANSWER
REM    BTR applies spectral_norm to the two convs inside each residual block and
REM    to nothing else - not the stem conv, not the downsample, not the final
REM    linear (verified in VIPTankz/BTR networks.py). The all-layers choice was
REM    ours, argued from the GroupNorm partial-application trap. That argument
REM    does not transfer: GroupNorm attacks stage (2) and has to cover the
REM    measurement point; SN attacks stage (1) and does not.
REM
REM  ARMS
REM    C2  spectral      residual convs only, init2=he    (BTR faithful)
REM    D2  spectral_cap  residual convs only, zero-init kept
REM    The all-layers cap arm (run.@pl_sncap) is defined in Atari.txt but is NOT
REM    in this batch. It would only tell us WHY the all-layers spectral arm
REM    diverged; finding a scope that works comes first.
REM
REM  WHAT TO LOOK AT
REM    First: 37_agent_qtd/11_q_max_mean must stay in the 0.5-8 band the control
REM    walks. If it leaves that band the arm is diverging, stop reading further.
REM    Then the three-point set 61 (raw, climbing) / 65 (sigma, climbing) /
REM    63 (effective, flat) - that is SN working.
REM    Then the assay readout: eval1 peak window and the fall from it.
REM    Control band from four runs: peak 240-324, end 113-139, fall -53 to -61%.
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

SET "ASSAY=run.@v5_iqn_impala_x2>run.@plasticity>run.@plasticity_rr8"
SET "M=%ASSAY%>run.@pl_snmetrics"

echo === 0. wiring check: BTR-scope branches (4 min) ===
call:run_exe "run.$=%M%>run.@pl_snr>run.@pl_check"
call:run_exe "run.$=%M%>run.@pl_capr>run.@pl_check"

echo === 1. FIRST LOOK: BTR-faithful spectral and cap (about 2.2h) ===
call:run_exe "run.$=%M%>run.@pl_snr"
call:run_exe "run.$=%M%>run.@pl_capr"

echo === 2. replicates of the BTR-faithful arms (about 2.2h) ===
call:run_exe "run.$=%M%>run.@pl_snr"
call:run_exe "run.$=%M%>run.@pl_capr"

echo === 3. F: LayerNorm replicate - r1 was n=1 and out of the control band ===
call:run_exe "run.$=%ASSAY%>run.@pl_ln512"

echo === 4. B redo: init2.mode key was wrong the first time (about 2.2h) ===
call:run_exe "run.$=%ASSAY%>run.@pl_he"
call:run_exe "run.$=%ASSAY%>run.@pl_he"

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
