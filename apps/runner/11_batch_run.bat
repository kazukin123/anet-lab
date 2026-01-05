@echo off
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe"

call:run_exe app.run_name=run_{t}-0
REM call:run_exe app.run_name=run_{t}-1_EPS   A.action_policy.policy_type=0
REM call:run_exe app.run_name=run_{t}-2_UQE   A.action_policy.policy_type=1
REM call:run_exe app.run_name=run_{t}-3_TS    A.action_policy.policy_type=2
REM call:run_exe app.run_name=run_{t}-4_TSf   A.action_policy.policy_type=2 A.action_policy.uqe_use_tail_mean=true
REM call:run_exe app.run_name=run_{t}-5_UQE5  A.action_policy.policy_type=1 A.action_policy.uqe_eps_max=0.05 A.action_policy.uqe_eps_min=0.05
REM call:run_exe app.run_name=run_{t}-6_TSA5  A.action_policy.policy_type=2 A.action_policy.uqe_eps_max=0.05 A.action_policy.uqe_eps_min=0.05
REM call:run_exe app.run_name=run_{t}-7_TSA5f A.action_policy.policy_type=2 A.action_policy.uqe_eps_max=0.05 A.action_policy.uqe_eps_min=0.05  A.action_policy.uqe_use_tail_mean=true
REM call:run_exe app.run_name=run_{t}-8_UQE1  A.action_policy.policy_type=1 A.action_policy.uqe_eps_max=0.01 A.action_policy.uqe_eps_min=0.01
REM call:run_exe app.run_name=run_{t}-9_TSA1  A.action_policy.policy_type=2 A.action_policy.uqe_eps_max=0.01 A.action_policy.uqe_eps_min=0.01


REM call:run_exe app.run_name=run_{t}-2_B4096 A.learner.replay_batch_size=4096
REM call:run_exe app.run_name=run_{t}-3_B256-rr2 A.learner.replay_batch_size=256 A.learner.replay_ratio=2
REM call:run_exe app.run_name=run_{t}-4_B256-rr1 A.learner.replay_batch_size=256 A.learner.replay_ratio=1
REM call:run_exe app.run_name=run_{t}-01_UQE  A.action_policy.policy_type=1
REM call:run_exe app.run_name=run_{t}-02_H128 A.qnet.nn_hidden1=128 A.qnet.nn_hidden2=128
REM call:run_exe app.run_name=run_{t}-03_B256 A.learner.replay_batch_size=256
REM call:run_exe app.run_name=run_{t}-04_B128 A.learner.replay_batch_size=128




pause
exit /b


:run_exe
echo %DATE% %TIME% START %*
%EXE% %*
echo   %DATE% %TIME% END   %*
exit /b
