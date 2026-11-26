# DropMerge の TempoRL（scratch 第 1 弾）と、対照の新旧の差の探索履歴

## 元の材料

[DropMerge の TempoRL（scratch 第 1 弾）と、対照の新旧の差の探索履歴](../../../../../experiments/default-dqn/dropmerge/2026-11-01_temporl-and-control-gap.md) blob:738ca9d8bd4fcb63609ae8a744261fa80bcc7480

## 要点

- TempoRL（分離×合算、J=10）は Greedy の評価で無効の対照を下回った。
- 差は序盤ほど大きく、進むにつれて縮んだ。
- Train Actor は 20〜28% の step でしか判断しなかったが、Greedy の full はほぼ毎 step 判断した。
- 補助学習のみ（47f）は、10.8M で止めるまでに対照を上回っていた。
- 原因は特定できていない。 設定で学習に効く差は UQE の ε（旧は全 lane 1e-4、新は 0）だけである。本番モードで ε を 1e-4 にした Run は測っていない。
- 特徴ノルム・勾配ノルム・序盤のエントロピーの水準は、ビルドではなく seed で揃った。

## 内容

### この文書の読み方

評価は 5 種類ある。Eval1 は target ネットの UQE（τ0.85）、Eval2 は online ネットの UQE、Eval3 は target ネットの Greedy。学習中の評価は Eval1・Eval3 が 1 セッション 8 局（nzrep と 40_s2 は 1 局）、Eval2 が 1 局である。`53_evalg`（`[greedy]`）と `54_evalgs`（`[greedy_single]`）は Greedy の 128 局を 5M ごとに行い、`53_evalg` は Actor のスキップ設定のまま（TempoRL の Run では full）、`54_evalgs` は毎 step 判断する（single）。

TempoRL の無い対照では、`53_evalg` と `54_evalgs` は同じ方策の別のセッションである。この 2 つの差は 47b で平均 29（最大 120）、47c で平均 17（最大 55）だった。

### 探索ブロック 02: TempoRL（分離×合算）と無効

Greedy の 3 回平均（各 128 局、full / single）。

学習中の評価の Eval1 は 0–10M の 477 から 50–60M の 661 で、同じ窓の Eval3 は 535 から 1,024、47b の Eval1 は 631 から 1,235 だった。

### 探索ブロック 03: 補助学習のみ（47f、10.8M まで）

Greedy（128 局）。47f は全 Actor が single なので、`53_evalg` と `54_evalgs` は同じ方策の別のセッションである。

### 探索ブロック 04: 対照の新旧の差

Eval1（target ネットの UQE）も同じ形で、40〜50M は旧 1,499、1,399、1,470、新 1,153、1,244 だった。

### 探索ブロック 05: 決定的モードでの新旧の比較

Train 側で時間で変わる設定のうち、この設定で実際に効くのは PER の β（0.4 から 0.6 を 20M で）だけである。ε と UQE の τ は lane ごとの固定値を使うので、時間の減衰は効かない。β が止まる 20M を越えても差は出なかった。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [DropMerge](../../../../concepts/subjects/dropmerge.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
