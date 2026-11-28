---
title: RND による新しさの評価と内的報酬
page_kind: サマリ
material: "[[docs/design/212_rnd_lifelong_novelty.jp.md|212_rnd_lifelong_novelty.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# RND による新しさの評価と内的報酬

## 元の材料

[RND による新しさの評価と内的報酬](../../../design/212_rnd_lifelong_novelty.jp.md) blob:15e0599a079bdcf962f39332f82a0349422f4304

## 要点

- RND は固定した Network の出力を predictor に予測させ、誤差からあまり経験していない観測を採点する。
- 採点に使う観測、正規化、報酬への変換と倍率、predictor の補助学習を個別の段階として設定する。
- 待ちはすべて採点した観測数で決まる。whiten は batch の採点前、feedback の倍率は各 lane、predictor の学習は採点後の件数で判定する。
- 補助学習のサンプルの取り方は学習時の予測誤差に影響するため、報酬の倍率と区別して読む。
- 誤差・倍率・補助学習のメトリクス、Evaluation Trace、保存・再開する状態から各段階を追跡できる。

## 内容

### 2. 段階とモード

- RND の設定キーは `DefaultDQNAgent.intrinsic.` 配下で指定する。
- 統計 — スコープは`rnd.statistics`。モードは`mode = whiten / variance_feedback`。モードの下で選べるものはwhiten の観測正規化。feedback の観測 clip と倍率 clip。
- `observation` で `next_state` または `state` の生の観測を選ぶ。target の構造は `rnd.target.net.*`、両方の Head は `rnd.head.*`、predictor の optimizer は `rnd.learning_rate`、`rnd.weight_decay`、`rnd.adam_eps` で設定する。報酬の合成は `beta`、`use_done_mask`、`reward_scaler.*`、`value_mode` と `value.head.*` が決める。

### 3. 観測、誤差と倍率

- 正規化した観測を同じ target と predictor に入れる。output の次元を $d$ とすると、lane ごとに $e = \sum_{j=1}^{d}(g(x)_j-f(x)_j)^2/d$ を得る。
- whiten の観測正規化は成分ごとの平均を引き、`sqrt(var) + epsilon` で割る。既定の `epsilon` は 1.0、観測の clip は ±5。非カテゴリの byte 観測は浮動小数点に変換する。
- feedback の観測平均と誤差平均は 0、観測と誤差の分散に相当する値は 1、件数は 0 から始める。連続観測は更新前の平均を引いて分散で割り、clip した値を観測統計に戻す。

### 4. 待ちとスケーラの準備

- 件数は採点した観測数で、0 は開始から有効を表す。whiten では batch を採点する前の件数が指定値未満の間に待つ。feedback の倍率は lane ごとの件数で判定する。 `Learn` は、predictor の待ちが正で採点後の件数がその値以下の間、偽を返す。選んだモードが使う正の待ちは、始まりと終わりを INFO で一回ずつ記録する。
- 標準化の待ちが明ければ、predictor の待ちの間も誤差を統計に入れる。none と predictor の待ちを同時に使う場合、predictor の待ちの間は倍率が 1 なので、スケーラに入る値は 0 になる。対応する待ちがすべて明けると報酬が出る。none の待ちはスケーラの種類によらず同じ順序で働く。特に `return_std` では報酬の出始めまでに内的な収益の標準偏差を見積もれる。

### 5. predictor の補助学習

- `intake` は受け取った lane の中から引く。lane 数 $N$、比率 $r$、1 step の件数 $k$ として、intake ごとに credit に $Nr/k$ を足す。整数の credit があり、今の intake に未使用の組が残っている間、重複なしで $k$ 件ずつ引く。学習が有効なときは $k$ を lane 数以下にする。
- `LaneTimeTensorStore` が観測の型と device を保ち、lane と時間で必要な tensor を読み出す。学習はその時点の観測統計で正規化した全 $k$ 件を使う。待ちにより `Learn` が偽を返した step は消費し、成功回数 `intrinsic.learn_count` は増やさない。学習を無効にした Run では採点、蓄積、補助学習を休止する。

### 6. 名前のある組み合わせ

- 倍率 clip — 既定 RND（`run.@rnd`）は真・5。Taiga 条件（`run.@rnd_taiga`）は偽。Burda 条件に寄せる構成は偽。
- 観測 — 既定 RND（`run.@rnd`）はnext_state、done をマスク。Taiga 条件（`run.@rnd_taiga`）はstate。Burda 条件に寄せる構成はnext_state、done も使う。
- 既定の RND を使うには Env の Run 連鎖に `run.@rnd` を加える。Atari は β = 0.03、DropMerge は β = 0.01 を選ぶ。GridMaze と LunarLander は Agent の既定の β = 0.1 を使う。

### 7. 設定の一覧

- 無効なモードの値も型と値域を検証する。実際の採点や学習は選択したモードの値を使う。
- `rnd.statistics.mode` — 型・値域はwhiten / variance_feedback。既定はwhiten。変わるものは観測の統計と feedback の式。
- `rnd.statistics.whiten.obs_norm.*` で whiten 時の観測正規化を設定する。clip_range は有限の正の値で既定は5。
- `rnd.statistics.variance_feedback.obs_norm.clip_range` — 型・値域は有限で正。既定は5。変わるものはfeedback の正規化済み観測の上限。

### 9. 背景、実装と検証

- この追加報酬は、環境のゴールを変更するものではない。ただし、学習器が最適化する報酬は変わる。探索に熱中して外的な点数が下がることもあり得るため、ボーナスが出たことと、目的の課題が改善したことは別に確認する。
- predictor の学習が十分でなかったり、別の観測を学んで予測が変わったりすると、再訪問でも誤差が大きいことがある。
- 今回の「一生の新しさ」は、1つの estimator が続いている間の学習履歴を指す。エピソードが終わっても predictor や累積統計をリセットしない。同じ Agent の全 lane が1つの estimator を共有し、どこかの lane で学んだことが他の lane の新しさにも影響する。checkpoint から再開した場合は、その履歴を引き継ぐ。
- `34_agent_plasticity` の可塑性の指標は DQN の online の Network を測り、RND の target と predictor は測らない（§9.4）。

### その後の変更

checkpoint は新しい Run に引き継ぐためのモデルや学習状態を含む。RND predictor と累積統計にある新しさの履歴も引き継ぐ。保存するのは統計と補助学習の credit・乱数・成功回数、内的報酬スケーラなどで、rollout は空からため直す。誤差の式と待ち、および optimizer の options は引き継ぎ先の設定に従う。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
