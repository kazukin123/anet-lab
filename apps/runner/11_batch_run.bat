@echo off
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe" app.$=app.batchrun

REM ==========================================================
REM 2. Deep MLP (MLPでの深層化テスト)
REM ==========================================================
REM [構成] Flatten -> 120 -> 120 -> 120 -> 120 -> 84 -> Out
REM MLPを深くすると勾配消失などで学習効率が落ちるかを確認する対照実験。
call:run_exe app.run_name=run_{t}_02_MLP_Deep5 "net.body.structure=Flatten > Linear_120 > ReLU > Linear_120 > ReLU > Linear_120 > ReLU > Linear_120 > ReLU > Linear_84 > ReLU"

REM ==========================================================
REM 3. ResNet Shallow (機能確認)
REM ==========================================================
REM [構成] ResBlock(*3)
REM 新実装のCompositeブロックが正常に機能し、MLP並みに学習するかを確認。
REM ※ ResBlock, Linear_Proj, Linear_Out が定義済みである前提
call:run_exe app.run_name=run_{t}_03_ResNet_x3 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*3) > Linear_Out > ReLU"

REM ==========================================================
REM 4. ResNet Deep (本命: 深層化耐性)
REM ==========================================================
REM [構成] ResBlock(*10)
REM MLPでは厳しくなる深さでも、Skip Connectionのおかげで学習が進むか？
REM "実時間で遅い" 問題はあるものの、一晩かければ結果は出るはず。
call:run_exe app.run_name=run_{t}_04_ResNet_x10 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*10) > Linear_Out > ReLU"

REM ==========================================================
REM 5. ResNet Very Deep (ストレステスト)
REM ==========================================================
REM [構成] ResBlock(*20)
REM さらに深くして、エラーが出ないか、収束するかを確認。
call:run_exe app.run_name=run_{t}_05_ResNet_x20 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*20) > Linear_Out > ReLU"


REM --- 1. Baseline (Standard) ---
REM 定義 [nn.txt source: 24]: Linear_120 > ReLU > Linear_84 > ReLU
REM call:run_exe app.run_name=run_{t}_01_MLP_Base

REM --- 2. Wide MLP (Feature Rich) ---
REM 中間層の幅を広げて特徴抽出能力を向上
REM 構造: Linear_256 > ReLU > Linear_120 > ReLU
REM call:run_exe app.run_name=run_{t}_02_MLP_Wide "net.body.MLP.structure=Linear_256 > ReLU > Linear_120 > ReLU"

REM --- 3. Large MLP (Max Capacity) ---
REM [nn.txt source: 23] の Linear_512 を使用した最大構成
REM 構造: Linear_512 > ReLU > Linear_256 > ReLU
REM call:run_exe app.run_name=run_{t}_03_MLP_Large "net.body.MLP.structure=Linear_512 > ReLU > Linear_256 > ReLU"

REM --- 4. Deep MLP (3 Layers) ---
REM 層を深くして非線形な推論能力を強化
REM 構造: Linear_120 > ReLU > Linear_120 > ReLU > Linear_84 > ReLU
REM call:run_exe app.run_name=run_{t}_04_MLP_Deep "net.body.MLP.structure=Linear_120 > ReLU > Linear_120 > ReLU > Linear_84 > ReLU"

REM --- 5. Deep & Wide (Heavy) ---
REM 広さと深さを両立させた重厚長大モデル
REM 構造: Linear_256 > ReLU > Linear_256 > ReLU > Linear_120 > ReLU
REM call:run_exe app.run_name=run_{t}_05_MLP_DeepWide "net.body.MLP.structure=Linear_256 > ReLU > Linear_256 > ReLU > Linear_120 > ReLU"



REM call:run_exe app.run_name=run_{t}

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
