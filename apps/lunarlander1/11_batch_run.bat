@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"
call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"
call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"
call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2 %3 %4
%EXE% %1 %2 %3 %4
echo %DATE% %TIME% END   %1 %2 %3 %4
exit /b
