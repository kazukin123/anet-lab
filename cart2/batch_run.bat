@echo off
SET EXE="..\\out\build\x64-Debug\cart2\CartPoleRLGUI.exe"

call:run_exe train.batchrun2 agent_rb4

call:run_exe train.batchrun agent_rb1
call:run_exe train.batchrun agent_rb2
call:run_exe train.batchrun agent_rb3
call:run_exe train.batchrun agent_rb5
call:run_exe train.batchrun agent_rb6
call:run_exe train.batchrun agent_rb7

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2
%EXE% -t %1 -a %2
echo %DATE% %TIME% END   %1 %2
exit /b
