@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=1" "RainbowAgent.trunk.use_dueling_net=true"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=2" "RainbowAgent.trunk.use_dueling_net=true"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=3" "RainbowAgent.trunk.use_dueling_net=true"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=1" "RainbowAgent.trunk.use_dueling_net=false"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=2" "RainbowAgent.trunk.use_dueling_net=false"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "RainbowAgent.trunk.n_step=3" "RainbowAgent.trunk.use_dueling_net=false"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2 %3 %4
%EXE% %1 %2 %3 %4
echo %DATE% %TIME% END   %1 %2 %3 %4
exit /b
