@echo off

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

echo === 0. wiring check x4 (12 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft008>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btrflat>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btrstruct_taurelu>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_vit>run.@pl_check"

echo === 1. soft tau 0.008 - freshness vs stationarity, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008"

echo === 2. btrstruct replicate with eval2, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrstruct"

echo === 3. btrstruct without maxpool - fusion at 7744, 50M (2.6h+) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrflat"

echo === 4. btrstruct with tau ReLU - BTR faithful, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrstruct_taurelu"

echo === 5. ViT hybrid + LN512 on V/A ReLU base, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_vit"

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
