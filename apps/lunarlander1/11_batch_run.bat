@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-10_W-Q30    DefaultDQNAgent.trunk2.reward_scaler.target_q_scale=30.0
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-11_W-Q10    DefaultDQNAgent.trunk2.reward_scaler.target_q_scale=10.0
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-13_W-Q3    DefaultDQNAgent.trunk2.reward_scaler.target_q_scale=3.0
call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-14_W-Q100    DefaultDQNAgent.trunk2.reward_scaler.target_q_scale=100.0

REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-05_W-0.0003   DefaultDQNAgent.trunk2.reward_scaler.constant_scale=0.0003
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-05_W-0.0003-A DefaultDQNAgent.trunk2.reward_scaler.constant_scale=0.0003 DefaultDQNAgent.trunk2.learner.alpha=5e-3
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-06_W-0.01     DefaultDQNAgent.trunk2.reward_scaler.constant_scale=0.01
REM call:run_exe LunarLanderApp.$=LunarLanderApp.batchrun LunarLanderApp.run_name=run_RS-07_W-0.03     DefaultDQNAgent.trunk2.reward_scaler.constant_scale=0.03

pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
