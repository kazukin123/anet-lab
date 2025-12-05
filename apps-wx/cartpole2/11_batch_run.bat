@echo off
SET EXE="bin\Release\CartPoleRLGUI.exe"

call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"
call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"
call:run_exe "train.$=train.batchrun" "agent.$=agent_trunk"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2
%EXE% -t %1 -a %2
echo %DATE% %TIME% END   %1 %2
exit /b
