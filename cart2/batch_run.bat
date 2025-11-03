@echo off
SET EXE="..\\out\build\x64-Debug\cart2\CartPoleRLGUI.exe"

call:run_exe train.batchrun agent_asdqn_bool8
call:run_exe train.batchrun agent_asdqn_bool9
call:run_exe train.batchrun agent_asdqn_bool8
call:run_exe train.batchrun agent_asdqn_bool9


pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2
%EXE% -t %1 -a %2
echo %DATE% %TIME% END   %1 %2
exit /b
