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
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_a_relu>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_v_relu>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_leaky>run.@pl_check"
call:run_exe "run.$=%A5%>run.@a5_20m>run.@rr1_va_btrstruct_hard125>run.@pl_check"

echo === 1. adv_stream only ReLU, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_a_relu"

echo === 2. value_stream only ReLU, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_v_relu"

echo === 3. hard C=125 remeasure with eval2, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_hard125"

echo === 4. BTR structure + hard C=125, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_btrstruct_hard125"

echo === 5. V/A LeakyReLU, 50M (2.6h) ===
call:run_exe "run.$=%A5%>run.@rr1_va_leaky"

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
