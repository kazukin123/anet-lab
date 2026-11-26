# 学習の同 seed 再現性のため `setDeterministicAlgorithms` をグローバル既定で有効化する

## 元の材料

[ADR 0006](../../../adr/0006-deterministic-algorithms.md) blob:a8984bfa96d08fa6a2efd40b6b617ff1d51953d2

## 要点

- 同じ seed でも学習結果が分かれ、推論だけの評価では再現した。
- 調査では省メモリ SDPA の backward の atomic 加算に原因を特定した。
- cuDNN の決定性設定は ATen の SDPA を対象にしないため、決定的アルゴリズムを全体で既定有効にする。
- 決定的な実装がない演算は既定でエラーにし、任意に警告へ切り替えられる。
- 実測では無効時より約11〜13%遅く、速度重視の条件では明示的に無効にできる。

## 内容

### 原因と代替案

従来の cuDNN 設定は畳み込みにしか効かない。math backend に固定する案と旧 MultiheadAttention へ戻す案は、速い backend を使う選択肢まで失う。

### 実行条件

初期化時に全 Run 共通で決定性を設定し、再現可能な演算を使う。実行環境によっては cuBLAS の workspace の指定も必要になる。「同じ seed で同じ結果」は決定性を有効にした条件の中で確かめる。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
