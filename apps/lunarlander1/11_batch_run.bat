@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe "RainbowAgent.trunk.per_beta_step=5000"
call:run_exe "RainbowAgent.trunk.per_beta_step=10000"
call:run_exe "RainbowAgent.trunk.per_beta_step=20000"
call:run_exe "RainbowAgent.trunk.per_beta_step=40000"
call:run_exe "RainbowAgent.trunk.per_beta_step=80000"

REM call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
REM call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
REM call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
REM call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2 %3 %4
%EXE% %1 %2 %3 %4
echo %DATE% %TIME% END   %1 %2 %3 %4
exit /b
