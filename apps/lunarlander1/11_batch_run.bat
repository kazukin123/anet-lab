@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2 %3 %4
%EXE% %1 %2 %3 %4
echo %DATE% %TIME% END   %1 %2 %3 %4
exit /b
