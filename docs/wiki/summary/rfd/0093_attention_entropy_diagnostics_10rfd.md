# TransformerEncoder の attention エントロピー診断 PRD

## 元の材料

[TransformerEncoder の attention エントロピー診断 PRD](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md) blob:045867f092651063ee1ff0433fdb2412cbc304bd

## 要点

- TransformerEncoder の各層で attention が何個のトークンに集中しているかを層平均で測る。
- 正規化エントロピーは一様なら1、一つのトークンに集中すれば0になる。
- 学習の forward と数値を保ち、購読していない場合は診断計算を行わない。
- 複数層の最小値一つにまとめず、層ごとに learn_step に沿って記録する。
- 一部の head の集中や、エントロピーと成績の因果関係は層平均の結果だけでは判断できない。

## 内容

### 目的と問題

Atari の ViT 混成（`run.@vitva`、`run.@vitbtr`）は、成績によらず常設のプロファイルである。
可塑性メトリクス（`34_agent_plasticity`）は final feature branch の出力の健全性（dormant、srank など）を測るが、attention の分布は見ない。

### 現状

TransformerEncoder は `CustomTransformerEncoderLayer` を層とする module で、設定キーは `tf.d_model`、`tf.nhead`、`tf.num_layers`、`tf.dim_feedforward`、`tf.norm_first`、`tf.activation`、`tf.use_sdpa` と dropout 系である（nn_modules.cpp の `TransformerEncoderModule`）。層は `layer_0` から番号を持ち、SN 診断の名前も `layer_0.q` のように 0 始まりである。
self-attention は `at::scaled_dot_product_attention` で計算していて（ADR 0004）、attention の重みを返さない。重みの保持器としての `MultiheadAttention` は残っているので、Q と K は同じ重みから計算し直せる。SN 有効時は `ForwardAttention` が実効重みを使う。
attention entropy collapse という語を作った Zhai ら（ICML 2023、σReparam）は、学習中に head ごとの attention エントロピーを追跡し、低エントロピーの状態を崩壊と呼んで学習の不安定化との対応を示した。head ごとの定義が文献の標準で、TQL の head 平均が変則である。

### ゴールと非ゴール

TransformerEncoder の層平均の attention エントロピーを、学習の `learn_step` 軸のスカラーとして記録し、集中度の推移と成績の時期を対応させて読めるようにする。
学習の forward の計算と数値（SDPA の経路）を変えない。指標を購読しない構成では計算も起きない。
attention マップの画像化（GUI や Viewer での表示）、head 単位の値の公開。

### 決定した設計

branch 名には `/` を使えない（Network 構築時に reject）。branch の鍵と module の鍵の名前空間を構造で分ける。衝突が起きたことはなく、規約を構造にするための番人。
Learner は `TransformerEncoder/attention_entropy/` を接頭辞で要求し、`GetAll` が返す列を実行順に連結して、全層まとめて 1 回 CPU へ転送して更新結果に載せる。平均・正規化を再度行わず、層ごとの CPU 同期を増やさない。

### 機構ごとの根拠

head 単位の値の公開（head 最小値、閾値以下 head 割合）。

### 受入基準

本節は診断機能の完了条件とする。長期 Run の分析、集中の発見、成績改善は機能の合否に含めない。
ON/OFF 等価性: 同じ seed・決定論の設定で、`@attention` の有無によらず学習由来の metrics checksum が一致する（ADR 0031 の非干渉性を検証）。比較対象を明記し、実時間などの性能値と追加診断値を除く。checkpoint ファイルの SHA は直列化の非決定性により一致しない場合があるため、等価性の判定には使わない。probe を eval mode・NoGrad で実行する前提で、学習用 RNG、ReplayBuffer の優先度・sampled-once 統計、正規化器や network の buffer 状態が probe 単独で変わらず、実行モードが復元されることも確認する。
層平均で集中の進行が見えなかった → 今回の観測では正則化へ進む根拠が得られなかったと記録する。一部 head の集中が存在しないとは結論しない。

### 文書同期

`core/anet-core/testdata/prd072/`: 固定入力・golden 17 本・manifest を保持する。今回の改名では再採取しない。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [Action queryとattentionの証拠](../../concepts/methods/action_query_attention.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
