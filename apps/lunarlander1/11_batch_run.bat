@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"


REM call:run_exe LunarLanderApp.run_name=run_{t}
call:run_exe LunarLanderApp.run_name=run_{t}-00
call:run_exe LunarLanderApp.run_name=run_{t}-01 A.learner.replay_ratio=1
call:run_exe LunarLanderApp.run_name=run_{t}-02 A.learner.replay_ratio=8
call:run_exe LunarLanderApp.run_name=run_{t}-03 A.qnet.nn_init_mode=3
call:run_exe LunarLanderApp.run_name=run_{t}-04 A.learner.grad_clip_tau=1
call:run_exe LunarLanderApp.run_name=run_{t}-05 A.learner.eps_decay_step=20000 A.learner.per_beta_step=20000
call:run_exe LunarLanderApp.run_name=run_{t}-06 A.learner.eps_decay_step=80000 A.learner.per_beta_step=80000


REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.use_image_log=true  LunarLanderApp.use_per_image_log=true  LunarLanderApp.run_name=run_{t}_ev100v
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun  train.seed=
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_20251226-bs=64  train.seed=



pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
