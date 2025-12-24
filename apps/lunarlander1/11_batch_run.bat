@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun

pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
