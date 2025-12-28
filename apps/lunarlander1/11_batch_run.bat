@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun train.eval_interval=100 LunarLanderApp.use_image_log=true  LunarLanderApp.use_per_image_log=true  LunarLanderApp.run_name=run_{t}_ev100v
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun train.eval_interval=100 LunarLanderApp.use_image_log=false LunarLanderApp.use_per_image_log=false LunarLanderApp.run_name=run_{t}_ev100
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun train.eval_interval=0 LunarLanderApp.use_image_log=true    LunarLanderApp.use_per_image_log=true  LunarLanderApp.run_name=run_{t}_ev0v
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun train.eval_interval=0 LunarLanderApp.use_image_log=false   LunarLanderApp.use_per_image_log=false LunarLanderApp.run_name=run_{t}_ev0


REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  train.seed=
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_20251226-bs=64  train.seed=



pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
