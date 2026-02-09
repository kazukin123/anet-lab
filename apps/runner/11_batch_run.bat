@echo off
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe" app.$=app.batchrun
REM SET EXE="bin\Release\AnetRLRunner.exe"

call:run_exe app.run_name=run_{t}_r1 X.learner.replay_ratio=1.0
call:run_exe app.run_name=run_{t}_r2 X.learner.replay_ratio=2.0
call:run_exe app.run_name=run_{t}_r4 X.learner.replay_ratio=4.0


REM call:run_exe app.run_name=run_{t}_b256 train.batch_size=256
REM call:run_exe app.run_name=run_{t}_b256_4064 net.body.$=net.body.SuikaNet4064 "DropMergeEnv.$=DropMergeEnv.baseline > DropMergeEnv.G4064 > E"
REM call:run_exe app.run_name=run_{t}_b512 train.batch_size=512


REM call:run_exe app.run_name=run_{t}_seed train.seed=7508636947373324265 

REM call:run_exe app.run_name=run_{t}
REM call:run_exe app.run_name=run_{t}_g20  E.grid_rows=20  E.grid_cols=20
REM call:run_exe app.run_name=run_{t}_g20w net.block.FC1.linear.out_features=2048 net.block.FC2.linear.out_features=1024
REM call:run_exe app.run_name=run_{t}_g30  E.grid_rows=30  E.grid_cols=30


REM call:run_exe app.run_name=run_{t}_g10  E.grid_rows=10  E.grid_cols=10
REM call:run_exe app.run_name=run_{t}_g30  E.grid_rows=30  E.grid_cols=30
REM call:run_exe app.run_name=run_{t}_g100 E.grid_rows=100 E.grid_cols=100

REM call:run_exe app.run_name=run_{t}_b128  A.learner.replay_batch_size=128
REM call:run_exe app.run_name=run_{t}_b512  A.learner.replay_batch_size=512

REM call:run_exe app.run_name=run_{t}-EPS   A.action_policy.policy_type=0
REM call:run_exe app.run_name=run_{t}-UQE   A.action_policy.policy_type=1
REM call:run_exe app.run_name=run_{t}-TS    A.action_policy.policy_type=2

REM call:run_exe app.run_name=run_{t}-A_5e-5 A.learner.alpha=5e-5
REM call:run_exe app.run_name=run_{t}-A_5e-3 A.learner.alpha=5e-3


REM call:run_exe app.run_name=run_{t}_B-16 train.batch_size=16 
REM call:run_exe app.run_name=run_{t}_B-32 train.batch_size=32
REM call:run_exe app.run_name=run_{t}_B-64 train.batch_size=64
REM call:run_exe app.run_name=run_{t}_B-128 train.batch_size=128
REM call:run_exe app.run_name=run_{t}_B-256 train.batch_size=256
REM call:run_exe app.run_name=run_{t}_B-512 train.batch_size=512
REM call:run_exe app.run_name=run_{t}_B-1024 train.batch_size=1024

REM call:run_exe app.run_name=run_{t}_s8      "A.stucker.stack_count=8"
REM call:run_exe app.run_name=run_{t}_s16     "A.stucker.stack_count=16"
REM call:run_exe app.run_name=run_{t}_s4_n64  "A.stucker.stack_count=4" "net.block.FC2.linear.out_features=64"
REM call:run_exe app.run_name=run_{t}_s4_n128 "A.stucker.stack_count=4" "net.block.FC2.linear.out_features=128"

REM ■好成績順上位3件は収束値はほぼ同じ。違いは立ち上がりの速さ(大差ではない)。
REM call:run_exe app.run_name=run_{t}_K4_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].conv.kernel_size=4"
REM call:run_exe app.run_name=run_{t}_K2_C512_L256  "net.block.[Conv1D_Conv1d].conv.out_channels=512" "net.block.[Conv1D_Linear].linear.out_features=256"
REM call:run_exe app.run_name=run_{t}_K2_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128"
REM call:run_exe app.run_name=run_{t}_K3_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].conv.kernel_size=3"
REM call:run_exe app.run_name=run_{t}_02_MLP_Wide "net.body.MLP.structure=Linear_256 > ReLU > Linear_120 > ReLU"


REM call:run_exe app.run_name=run_{t}_K3_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].conv.kernel_size=3"
REM call:run_exe app.run_name=run_{t}_K4_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].conv.kernel_size=4"
REM call:run_exe app.run_name=run_{t}_K2_C512_L256  "net.block.[Conv1D_Conv1d].conv.out_channels=512" "net.block.[Conv1D_Linear].linear.out_features=256"
REM call:run_exe app.run_name=run_{t}_MLP_256_128   "net.body.MLP.structure=Linear_256 > ReLU > Linear_128 > ReLU"

REM call:run_exe app.run_name=run_{t}_C128_LR2e3 "A.learner.alpha=2e-3"
REM call:run_exe app.run_name=run_{t}_C128_LR1e4 "A.learner.alpha=1e-4"

REM call:run_exe app.run_name=run_{t}_K2_C128_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=128"
REM call:run_exe app.run_name=run_{t}_K3_C128_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].conv.kernel_size=3"
REM call:run_exe app.run_name=run_{t}_K2_C256_L128  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128"
REM call:run_exe app.run_name=run_{t}_K2_C128_L64   "net.block.[Conv1D_Conv1d].conv.out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=64"
REM call:run_exe app.run_name=run_{t}_K2_C64_L64    "net.block.[Conv1D_Conv1d].conv.out_channels=64"  "net.block.[Conv1D_Linear].linear.out_features=64"
REM call:run_exe app.run_name=run_{t}_K2_C256_L256  "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=256"
REM call:run_exe app.run_name=run_{t}_K2_C128_L128he "net.block.[Conv1D_Conv1d].conv.out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=128" "net.block.[Conv1D_Conv1d].init.mode=2" "net.block.[Conv1D_Linear].init.mode=2" "A.head_init.mode=1"


REM call:run_exe app.run_name=run_{t}_K2_C64_L64   "net.block.[Conv1D_Conv1d].conv.out_channels=64"  "net.block.[Conv1D_Linear].linear.out_features=64"
REM call:run_exe app.run_name=run_{t}_K2_C256_L128 "net.block.[Conv1D_Conv1d].conv.out_channels=256" "net.block.[Conv1D_Linear].linear.out_features=128"
REM call:run_exe app.run_name=run_{t}_K2_C128_L256 "net.block.[Conv1D_Conv1d].conv.out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=256"
REM ↑ 容量倍増：Gapだけ広がってスコア落ちたら過学習

REM call:run_exe app.run_name=run_{t}_eps-10 "A.action_policy.uqe_eps_max=1.0" "A.action_policy.uqe_eps_decay_step=10000000"
REM call:run_exe app.run_name=run_{t}_eps-08 "A.action_policy.uqe_eps_max=0.8" "A.action_policy.uqe_eps_decay_step=10000000"
REM call:run_exe app.run_name=run_{t}_eps-04 "A.action_policy.uqe_eps_max=0.4" "A.action_policy.uqe_eps_decay_step=10000000"
REM call:run_exe app.run_name=run_{t}_eps-10a "A.action_policy.uqe_eps_max=1.0" "A.action_policy.uqe_eps_decay_step=5000000"
REM call:run_exe app.run_name=run_{t}_eps-100 "A.action_policy.uqe_eps_max=1.0" "A.action_policy.uqe_eps_min=0.0" "A.action_policy.uqe_eps_decay_step=10000000"

REM call:run_exe app.run_name=run_{t}_Base_K2_C64
REM call:run_exe app.run_name=run_{t}_K3_C64 "net.block.[Conv1D_Conv1d].kernel_size=3"
REM call:run_exe app.run_name=run_{t}_K2_C32 "net.block.[Conv1D_Conv1d].out_channels=32" "net.block.[Conv1D_Linear].linear.out_features=32"
REM call:run_exe app.run_name=run_{t}_K2_C128 "net.block.[Conv1D_Conv1d].out_channels=128" "net.block.[Conv1D_Linear].linear.out_features=128"

REM call:run_exe app.run_name=run_{t}_A_5e-5 "A.learner.alpha=5e-5"
REM call:run_exe app.run_name=run_{t}_A_5e-4 "A.learner.alpha=5e-4"
REM call:run_exe app.run_name=run_{t}_A_1e-3 "A.learner.alpha=1e-3"
REM call:run_exe app.run_name=run_{t}_A_5e-3 "A.learner.alpha=5e-3"
REM call:run_exe app.run_name=run_{t}_A_1e-2 "A.learner.alpha=1e-2"
REM call:run_exe app.run_name=run_{t}_A_1e-1 "A.learner.alpha=1e-1"

REM SET COMMON_ARGS="net.body.$=net.body.MLP2 net.block.FC1.init.mode=1 net.block.FC2.init.mode=1 A.head_init.mode=1"

REM call:run_exe app.run_name=run_{t}_Init_Mode0 %COMMON_ARGS% "net.block.FC1.init.mode=0 net.block.FC2.init.mode=0 A.head_init.mode=0"
REM call:run_exe app.run_name=run_{t}_Init_Mode1_Base %COMMON_ARGS% "net.block.FC1.init.mode=1 net.block.FC2.init.mode=1 A.head_init.mode=1"
REM call:run_exe app.run_name=run_{t}_Init_Mode2 %COMMON_ARGS% "net.block.FC1.init.mode=2 net.block.FC2.init.mode=2 A.head_init.mode=2"
REM call:run_exe app.run_name=run_{t}_Init_Mode3 %COMMON_ARGS% "net.block.FC1.init.mode=3 net.block.FC2.init.mode=3 A.head_init.mode=3"
REM call:run_exe app.run_name=run_{t}_Init_Mixed %COMMON_ARGS% "net.block.FC1.init.mode=2" "net.block.FC2.init.mode=2" "A.head_init.mode=1"

REM call:run_exe app.run_name=run_{t}_Gain_0.0 %COMMON_ARGS% "A.head_init.manual_gain=0.0"
REM call:run_exe app.run_name=run_{t}_Gain_0.01 %COMMON_ARGS% "A.head_init.manual_gain=0.01"
REM call:run_exe app.run_name=run_{t}_Gain_0.1 %COMMON_ARGS% "A.head_init.manual_gain=0.1"
REM call:run_exe app.run_name=run_{t}_Gain_1.0 %COMMON_ARGS% "A.head_init.manual_gain=1.0"
REM call:run_exe app.run_name=run_{t}_Gain_1.41 %COMMON_ARGS% "A.head_init.manual_gain=1.41"


REM ==========================================================
REM 2. Deep MLP (MLPでの深層化テスト)
REM ==========================================================
REM [構成] Flatten -> 120 -> 120 -> 120 -> 120 -> 84 -> Out
REM MLPを深くすると勾配消失などで学習効率が落ちるかを確認する対照実験。
REM call:run_exe app.run_name=run_{t}_02_MLP_Deep5 "net.body.structure=Flatten > Linear_120 > ReLU > Linear_120 > ReLU > Linear_120 > ReLU > Linear_120 > ReLU > Linear_84 > ReLU"

REM ==========================================================
REM 3. ResNet Shallow (機能確認)
REM ==========================================================
REM [構成] ResBlock(*3)
REM 新実装のCompositeブロックが正常に機能し、MLP並みに学習するかを確認。
REM ※ ResBlock, Linear_Proj, Linear_Out が定義済みである前提
REM call:run_exe app.run_name=run_{t}_03_ResNet_x3 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*3) > Linear_Out > ReLU"

REM ==========================================================
REM 4. ResNet Deep (本命: 深層化耐性)
REM ==========================================================
REM [構成] ResBlock(*10)
REM MLPでは厳しくなる深さでも、Skip Connectionのおかげで学習が進むか？
REM "実時間で遅い" 問題はあるものの、一晩かければ結果は出るはず。
REM call:run_exe app.run_name=run_{t}_04_ResNet_x10 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*10) > Linear_Out > ReLU"

REM ==========================================================
REM 5. ResNet Very Deep (ストレステスト)
REM ==========================================================
REM [構成] ResBlock(*20)
REM さらに深くして、エラーが出ないか、収束するかを確認。
REM call:run_exe app.run_name=run_{t}_05_ResNet_x20 "net.body.structure=Flatten > Linear_Proj > ReLU > ResBlock(*20) > Linear_Out > ReLU"


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
