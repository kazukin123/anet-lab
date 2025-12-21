@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  RainbowAgent.trunk.alpha=5e-4
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  RainbowAgent.trunk.alpha=1e-4
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  RainbowAgent.trunk.alpha=5e-5
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  RainbowAgent.trunk.alpha=1e-5

pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
