# 出力の大きさで指定する max pooling（AdaptiveMaxPool2d）暫定 RFD

## 元の材料

[出力の大きさで指定する max pooling（AdaptiveMaxPool2d）暫定 RFD](../../../memo/0088_nn_adaptive_max_pool_10rfd.md) blob:863cbc747538a6012d11d73757719d9200f4bce1

## 要点

- 入力寸法から独立に出力の高さと幅を選ぶ AdaptiveMaxPool2d の暫定案を記す。
- DropMerge では DROP の行動が列に対応するため、縦だけを縮約して横方向の位置を残す用途がある。
- 通常の全面 Flatten や全体 pooling と異なる空間情報の残し方をネットワーク設定から選ぶ。
- 暫定の出力サイズ指定は、高さと幅を正の整数2個で表す。
- 平均版と長方形カーネルの通常 pooling は暫定案の対象から外し、詳細設計で採否を決める。

## 内容

### 目的と問題

今の NN は、盤面（46×58）を畳み込みで 12×15 に縮め、Transformer で 180 トークン（12 行 × 15 列）として処理する。Transformer を出た時点では、列の位置が残っている。

### 現状

出力の大きさは BTR と同じだが、窓は違う。11 → 6 の場合、`AdaptiveMaxPool2d` の窓は幅 2〜3 で隣と重なる。`AtariPool6` の窓は幅 1〜2 で重ならない。

### 提案の骨子

「入力の大きさのまま」を表す特別な値は持たない。その次元を残したいときは、入力の大きさをそのまま書く（DropMerge なら幅に 15 と書く）。
既存の `AtariPool6`・`AtariPool4` と、それを使う profile は変えない。過去の Run と比べられるようにするためである。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [Action queryとattentionの証拠](../../concepts/methods/action_query_attention.md)
