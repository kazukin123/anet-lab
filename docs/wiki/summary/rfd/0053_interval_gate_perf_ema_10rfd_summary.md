---
title: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
page_kind: サマリ
material: "[[docs/memo/done/0053_interval_gate_perf_ema_10rfd.md|0053_interval_gate_perf_ema_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化

## 元の材料

[interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md) blob:a188d005c1d443ac7f68557a37ebffbe6bbfad20

## 要点

- 並列環境数などで step が一度に飛ぶと、倍数との完全一致で判定する処理が周期を取り逃す。
- IntervalGate は step を interval で割った商の増加から区間境界をまたいだことを検出する。
- 起動・計測側が同じ判定を使い、step の増分に依存する違いを減らす。
- 性能指標はサンプル件数に等しい重みを付ける代わりに、経過時間に応じた EMA で平滑化する。
- 並列度が違う Atari Run を用い、発火間隔とスループットの比較条件を確かめる。

## 内容

### Context / Problem Statement

Atari の num_envs A/B で、独立した 2 つの不具合が同時に露見した。どちらも 「刻みが 1 でないカウンタ」「長さが不揃いな窓」を、均一である前提の式で扱っている ことが根本にある。
eval は同期的にコストを発生させるため、この差がそのまま wall-clock 差になった。当初「n512 の方が 2.65 倍速い」と読めたが、eval 頻度を揃えると n64 1,198 steps/s ≒ n512 1,264 steps/s で差は消える。num_envs の性能差と読めていたものは、実体はこのバグだった。

### 現状の事実（コード確認済み）

RR と batch を固定している限り `learn_step = exp_step × RR / batch` であり、比例定数は num_envs によらない（本 Run では 1 更新 = 32 exp_step）。軸の選択は正しく、壊れているのは発火規則だけである。
`interval=500`、num_envs=64 では `LCM(2,500)=500` 更新 = 16,000 exp_step、実測 154 発火。指定どおりに効く構成と 4 倍ずれる構成が混在する。

### 契約

以降は `step / interval_ > last_bucket_` のときだけ発火する。1 回の呼び出しで複数バケットを跨いだ場合も発火は 1 回（catch-up しない。ADR 0028）。
`train_step_per_sec` も同一関数内なので同じ扱いになる。

### 実装範囲

perf 系は実時間依存で時刻注入の仕組みが無いため、`CalcPerformanceMetrics` 自体の値テストは書かない。テスト可能な核（α 算出と発火判定）を `util.hpp` 側に寄せることで担保する。

### 受け入れ基準

eval 発火の num_envs 非依存: 同一 `interval` で num_envs=64 と 512 の Run を回し、`51_eval1/*` サンプルの exp_step 間隔が両者で一致する（`interval × batch / replay_ratio`）。修正前は 4 倍ずれる。

### スコープ外

`interval` 指定なしの agent 系メトリクス（metrics.jsonl の約 96%）の間引き。今回のバグとは独立で、毎イベント記録は意図どおり。

### Further Notes

τ を config 化しなかったのは、`TrainRunner` に config 経路が存在しないため（コンストラクタは `(env, agent, notifier)` のみ、`RunnerBase` も同様）。設定化すると `RunnerFactory::CreateMainRunner` まで signature 変更が波及する。必要になった時点で `RunManager::Config` からの setter 注入を検討する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
