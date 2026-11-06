# PRD 048 分位tail探索診断メトリクス 実装メモ

## 概要

`048_quantile_tail_metrics_10prd.md`を正本として、QR / IQNの既存quantile Tensorから5個の観測専用scalarを追加する。Policy score、loss、raw priority、Replay sampling、action、RNGは変更せず、新しい設定キー、checkpoint形式、Replay schema、ADRは追加しない。

## 主な変更

- DQN機能グループ内に、tau順quantileの最終次元からfloat32のupper / lower truncated stdとcrossing ratioを求める共通Tensor helperを追加する。値による再sortは行わず、`K < 2`は`NaN`とする。
- PolicyではQRの`q_quantiles`、IQNのfixed `full_q_quantiles`からper-action上下幅とglobal disagreement / crossingを作る。`DQNActionInfo::GetScalar()`は最終actionをgatherした4 scalarを初回参照時に一括CPU転送し、`WithAction()`後はpayloadを共有しつつcacheを作り直す。
- LearnerではQRをquantile index順、IQNを`current_taus`と同じpermutationで昇順化してsample-wise upper-tail幅を求める。PER有効時だけclip後raw priorityと同じpacked readbackへ同梱し、CPU上の平均順位からSpearman相関を計算する。
- raw priorityのpack先頭、clip件数、Replay更新順序、既存IQN diagnosticsを維持する。PER無効、batch不足、定数順位列は`NaN`とする。
- DropMergeへPolicy 4本を`eval2`、Learner 1本を`@learn`として追加し、すべてEMA `0.01`、`interval:100`とする。Train、`eval1`、LunarLander、raw版には登録しない。
- DQN設計、可観測性、Run分析ガイドへ式、h-space、`NaN`条件、crossingとPER sampling biasの解釈を反映する。`CONTEXT.md`と既存ADRは変更しない。

## 公開観測interface

- `DQNActionInfo::GetScalar()`: `policy_upper_truncated_std`、`policy_lower_truncated_std`、`lower_risk_full_q_argmax_disagreement`、`quantile_crossing_ratio`
- `BatchUpdateResult::GetScalar()`: `upper_tail_priority_spearman`
- 既存scalarの名称、値、欠損条件は変更しない。

## テスト

- Public interface / surface: `DQNActionInfo::GetScalar()`、QR / IQN Learnerの公開更新経路、`BatchUpdateResult::GetScalar()`、DropMergeの解決済みscalar設定。
- 優先 behavior: QR Policyの偶数・奇数・`K=2`数値契約、IQN fixed full queryと欠損時`NaN`、`WithAction()`、QR / IQN Learnerのclip後priority相関、tie順位、PER OFF、TBO、BF16、設定tag数とscope。
- TDD順序: QR Policyの公開scalarをtracer bulletとして1テストずつRED→GREENし、IQN Policy、QR Learner、IQN Learner、設定契約の順で縦に追加する。refactorはGREEN後だけ行う。
- 非干渉: action、loss、raw priority、Replay適用priority、RNG列、forward回数が変わらず、Learner readbackのwait boundaryが増えないことを確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner-test'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn]"
core\anet-core\bin\Debug\anet-core-test.exe
apps\runner\bin\Debug\AnetRLRunner-test.exe
git diff --check
```

## 前提

- ユーザー判断によりpaired throughput smokeと「低下2%以内」は今回の受入評価から除外する。追加forward・追加wait boundaryがないことはテストとコード構造で確認する。
- `DropMerge.txt`を含む既存の未コミット変更を保持し、関連箇所だけを局所編集する。
- DLTV、探索bonus、risk penalty、PER方式変更、Viewer変更、実験Runは対象外とする。
