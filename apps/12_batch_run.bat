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
SET "EV=run.@evalN10"

echo === 0. wiring check x4 (12 min) ===
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft020>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft050>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btr_default>%EV%>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_soft008_btr>%EV%>run.@pl_check"

echo === 1. soft tau 0.02 - sweep, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft020>%EV%"

echo === 2. soft tau 0.05 - sweep upper end, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft050>%EV%"

echo === 3. BTR NN full match - btrstruct + tau ReLU + default init, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btr_default>%EV%"

echo === 4. soft tau 0.008 + BTR structure - does tau mask structure, 50M (2.9h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_soft008_btr>%EV%"

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
