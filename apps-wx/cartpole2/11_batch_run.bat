@echo off
SET EXE="bin\Debug\CartPoleRLGUI.exe"

call:run_exe train.batchrun agent_asdqn_bool10
call:run_exe train.batchrun agent_asdqn_bool11

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2
%EXE% -t %1 -a %2
echo %DATE% %TIME% END   %1 %2
exit /b
