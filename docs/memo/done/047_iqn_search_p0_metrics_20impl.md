# PRD 047 IQN探索P0メトリクス 実装メモ

## 概要

`docs/memo/047_iqn_search_p0_metrics_10prd.md`を正本として、学習・行動選択・RNG・priorityを変えずに、DropMerge成果、IQNの`K/N/M`、初回Learner priority更新、PER健全性を観測可能にする。

## 主な変更

- `DropMergeEnv::GetScalar()`へepisode終了時だけ値を返す`ep_double_suika_created`と`ep_double_suika_achieved`を追加し、Train/Evalの`exp_step`軸へ前者raw・後者EMAで登録する。
- IQN+UQEの既存risk quantile、`uqe_values`、`full_q_values`を再利用し、`DQNActionInfo`から`iqn_policy_margin_mc_ratio`、`iqn_uqe_full_q_argmax_disagreement`、`action_full_q_margin.[i]`を公開する。診断値は1本のdetached packed Tensorへまとめ、1回だけlazy CPU materializeする。
- IQN loss内の既存`delta = y - z`を再利用し、current/target MC scale、priority MC ratio、初回更新時のratio・pairwise TD・cancellation・`N`正規化loss・初回行数を`BatchUpdateResult`へ追加する。
- 初回行はpriority sourceが`fixed_initial | max_initial | actor_initial`の行だけとする。PER OFFまたは初回行0件ではcountを`0`、初回系を`NaN`とする。PER OFFでも一般IQN診断は算出する。
- Learner診断はPER ONでは既存priority readbackへ同梱し、PER OFFでは固定長診断packだけを同じ非同期経路で回収する。追加forward、pairwise Tensor再構築、metric単位の`.item()`を行わない。
- `metrics.scalar.iqn_search_p0`を追加して必要なPER/throughput metricだけを選び、DropMerge/LunarLanderへ合成する。Learner/PER/IQN診断は`exp_step`・`interval: 100`、ノイズ系と初回系はEMA、`elapse_hour`はrawとする。
- full-query依存metricは既存full-query評価policyだけへ登録する。`action_full_q_margin`はAPI上任意index対応、P0設定は`[0]`のみとする。
- `CONTEXT.md`へ「初回Learner priority更新」を追加し、DQN設計、可観測性、Run分析ガイドへ式、NaN条件、h空間、`fixed`/`stratified`での解釈を反映する。ADRは追加しない。

## テスト

- Public interface / surface: `DropMergeEnv::Reset()` / `Step()` / `GetScalar()`、`DQNActionInfo::GetScalar()`、公開learner更新経路、Runner設定解決結果を対象にする。
- 優先 behavior: Double Suika終了契約、Policy手計算とfail-fast、Learnerの`N != M`・相殺率・初回source・PER OFF・NaN、非干渉、設定合成の順で確認する。
- TDD順序: 各behaviorについて1テストを追加してREDを確認し、最小実装でGREENにしてから次へ進む。refactorは関連テストがGREENの時だけ行う。
- CPUに加え、利用可能ならCUDA/BF16のdtype/device契約を確認する。診断追加前後でloss、raw priority、ReplayBuffer反映priority、action、RNG列、forward回数が不変であることを確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe
git diff --check
```

同一binary・seed・実行条件でP0 group OFF/ONの短時間paired smokeを直列実行し、安定区間の`exp_step_per_sec`低下が2%以内であることを確認する。超過時はreadback・同期設計を修正して再計測する。

## 前提

- 公開設定キー、checkpoint、TensorDict、既存scalarの名称・意味は変更しない。
- IQN loss、priority式、sampling、action、tau生成、探索条件、Run条件は変更しない。
- 新規componentファイル、config flag、互換alias、ADR、実探索Runは追加しない。
- 既存の未コミット変更を保持し、特に`apps/runner/config/DropMerge.txt`は対象行だけを局所編集する。
