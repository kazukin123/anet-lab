# RFD0191 の検証資材

[RFD0191](../../../../../docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) の受入と [実装メモ](../../../../../docs/memo/0191_rnd_raw_error_and_recent_learning_20impl.md) の実施記録を参照する。

## 既存挙動の比較

変更前の Release Runner とその共通設定を同じフォルダへ保存する。変更後は現用の共通設定と Release Runner を使う。LunarLander の既定連鎖に、RND 無効、累積、EMA、variance_feedback、誤差だけを集める待ちの 5 条件を適用する。決定的 CUDA、seed 1、torch 1 thread、100,000 経験。評価と描画、画像を止める。時刻と資源の 3 prefix を除いた、全 scalar の tag、step、値を照合する。後半では既定構成の旧 checkpoint を読み込む Run も行う。

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0191/compare_rnd.py before --baseline .scratch/prd191/before-ac5135ff --output .scratch/rfd0191/functional-recheck
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0191/compare_rnd.py after --baseline .scratch/prd191/before-ac5135ff --output .scratch/rfd0191/functional-recheck
```

## 補助学習の実時間

Atari の Montezuma、128 lane、replay ratio 8、Q の hard update 間隔 250 の同じ構成と実行体で、intake、一括 rollout、分散 rollout を開始順を変えて 3 巡する。32,768 から 49,152 経験の実時間を群ごとに集計する。Replay は計測に使う全経験を収める 131,072 件で確保する。別 Run で Tracy の `intrinsic_rollout_batch` を採取し、一括学習の 1 回分を読む。

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0191/measure_rollout.py measure --output .scratch/rfd0191/perf-recheck
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0191/measure_rollout.py profile --output .scratch/rfd0191/profile-recheck
```

## 単体・結合テスト

```powershell
out\test\Debug\anet-core-base-test.exe '[rfd0191],[intrinsic],[shipped_config]'
out\test\Debug\anet-algo-dqn-test.exe '[rfd0191],[intrinsic],[shipped_config]'
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0194/audit_config_messages.py --expected src/core/base/testdata/rfd0194/expected.jsonl
```
