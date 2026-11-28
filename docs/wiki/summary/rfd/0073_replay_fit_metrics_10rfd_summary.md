---
title: replay 抽選履歴別の TD・損失メトリクス
page_kind: サマリ
material: "[[docs/memo/done/0073_replay_fit_metrics_10rfd.md|0073_replay_fit_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
updated: 2026-11-28
---
# replay 抽選履歴別の TD・損失メトリクス

## 元の材料

[replay 抽選履歴別の TD・損失メトリクス](../../../memo/done/0073_replay_fit_metrics_10rfd.md) blob:f9fbafee08af1d4f3943a283c4db38a56d76a171

## 要点

- ReplayBuffer の抽選履歴で遷移を群に分け、TD と損失を診断する。
- 一度も学習に使われていない遷移の群は、独立に用意した held-out データとは異なる。
- 各群と実際の PER バッチで同じ計算を適用し、IQN では既定32個の固定 midpoint を使う。
- 平均値だけでなく母数と平均年齢も残し、群の構成が変わった場合に分けて解釈できるようにする。
- 任意の測定として有効にし、既定では503回の Learner 更新ごとに診断する。

## 内容

### 群の作り方

測定時点の抽選可能な遷移を、未抽選と抽選済みに分ける。抽選済みには先読み済みや現在処理中の遷移も含むため、全件にすでに勾配更新がかかったとはいえない。両群について候補数と年齢を求める。

### 同じ条件で誤差を見る

通常 TD と分布型の QR・IQN の両方を扱う。診断は FP32 の評価モードで実行し、IQN の分位には固定 midpoint を使う。期待値の TD 誤差は PER に使われ、分位回帰の損失とは区別して記録する。

### PER 選択の影響と測定コスト

未抽選群と抽選済み群だけでなく、実際に PER が選んだ学習バッチとの違いも見る。期待値の TD 誤差と分布回帰の損失には別の意味があるので、比の分子・分母になる平均も残し、比だけが動いた場合にどちらが変化したかを調べられるようにする。

母数と平均年齢は同一の snapshot から取り、dummy は候補に数えない。指標の購読内容に応じて不要な処理を省き、通常学習や既存の可塑性 probe の乱数列に干渉しないようにする。学習側で分位をランダムに選ぶ設定でも、この診断の分位は両群と実バッチで統一する。

### 比較の読み方

比の変化は群ごとの平均値と組み合わせて読む。測定の間隔は Learner 更新単位であるため、同じ経験数の Run を比べるときは replay ratio による測定点数の差も確認する。他の probe の有無が双方のサンプル系列に影響しないことを検証する。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
