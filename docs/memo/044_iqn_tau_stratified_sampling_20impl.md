# IQN tau sampling modes 実装メモ

## 概要

IQN の `sample_mode` を `random | fixed | stratified | systematic | antithetic` へ拡張する。`GenerateTaus` の既存2 overload、`TauRuleConfig`、Policy/Learnerの呼び出し構造と既定値は維持し、生成方式を `TauGenerator` 内へ閉じ込める。

## 主な変更

- 対象device上で正規化位置 `(B,K)` を作るfile-local helperを設け、共通範囲版とper-env下限版で共有する。
- `stratified` は `(B,K)`、`systematic` は `(B,1)`、`antithetic` は `(B,ceil(K/2))` の乱数を各1回で生成する。antitheticの奇数`K`末尾は独立サンプルとする。
- `DefaultDQNAgentConfig` の全8 tau ruleと直接呼び出し時の検証を同じ5 modeへ拡張し、未知値は指定値と許容値一覧を含めてfail-fastする。
- public interfaceは設定列挙値の追加だけとし、新field、互換alias、fallback、追加metric、Policy/Learner側のmode分岐は追加しない。
- PRD、domain glossary、DQN設計文書、Runnerの現行設定コメントを5 mode契約へ更新する。既存PRD 001とADR 0018は履歴として変更しない。

## テスト

- Public interface / surface: `GenerateTaus` の両overload、`DefaultDQNAgentConfig` の全tau rule、既存の解決済み設定経路。
- 優先 behavior: stratifiedの被覆と再現性、per-env写像とRNG消費、systematicの行別位相と等間隔、antitheticの偶数・奇数レイアウトと鏡映、設定受理と未知値エラー、CUDA上のdevice・shape・dtype、既存random/fixedと既定値の回帰。
- TDD 順序: `stratified`、`systematic`、`antithetic`、config/validation、CUDA・回帰の順に、1テスト追加、RED確認、最小実装、GREEN確認を繰り返す。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test -- -j1'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][iqn][tau]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][iqn][config]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][iqn][action_policy]"
git diff --check
```

## 前提

- Train PolicyとLearnerは`random`、Eval Policyとtarget Policyは`fixed`のままとする。
- 奇数`K`のantitheticは鏡映ペアの後ろへ独立サンプルを1点置く。全点平均の中点固定は偶数`K`だけの契約とする。
- 数式上はstratified/systematicが半開区間、antitheticが閉区間だが、float32保存値は新3 modeとも`tau_min <= tau <= tau_max`とする。丸めによる上端到達と極狭範囲での隣接同値を許容し、clampや再抽出は行わない。
- antitheticの分散低減は単調な分位関数に対する条件付き理論であり、単調性を強制しない現行IQNでは実験で評価する仮説とする。
- 比較関係は被覆軸 `random -> stratified -> systematic -> fixed` と対称性軸 `random -> antithetic` の2軸で扱う。
- `num_taus`、PER、ReplayBuffer、batch/replay ratio、NN、checkpoint、実験preset、IQN full-query tail metricsのfixed限定契約は変更しない。
