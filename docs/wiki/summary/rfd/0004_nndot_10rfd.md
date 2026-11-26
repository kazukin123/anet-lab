# anet::nn::Network GraphViz 仕様

## 元の材料

[anet::nn::Network GraphViz 仕様](../../../memo/done/0004_nndot_10rfd.md) blob:4cc87199e35a5cf700b3a1aaad8229484771c2f8

## 要点

- Network の入力から Branch・Block・Body output・Head までの接続を GraphViz で出力する。
- 構造版は接続関係を表示し、詳細版は設定に応じて TensorSpec やパラメータ数を加える。
- Agent の初期化時に構造を1回取得する。
- Network のグラフ生成の入口は MakeGraphViz にまとめる。
- 学習中の再生成や、同じ構造を持つ target network の重複出力を避ける。

## 内容

### Context

ダンプは Agent 初期化時に 1 回だけ行う。Network 構造は学習中に変化しないため、再生成や target network の重複出力は行わない。

### API / Config

`Network` 自体は `anet::graphviz::GraphViz` を継承しない。GraphViz 生成は `Network::MakeGraphViz(const NetworkGraphVizConfig&) const` に一本化する。

### Graph Structure

Branch へ入る edge label は `LabelData::SetText(bind_key)` を使う plain text とし、table label にはしない。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
