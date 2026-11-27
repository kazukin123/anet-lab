---
title: RND による一生の新しさと内的報酬
page_kind: サマリ
material: "[[docs/memo/0116_rnd_lifelong_novelty_10rfd.md|0116_rnd_lifelong_novelty_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
updated: 2026-11-26
---
# RND による一生の新しさと内的報酬

## 元の材料

[RFD0116　RND による一生の新しさと内的報酬](../../../memo/0116_rnd_lifelong_novelty_10rfd.md) blob:2f6696f53d4b6395dd7c0117b19069c08502b271

## 要点

- 開発者の判断で、暫定の既定のまま実装へ渡した。実装計画に理由を記せば既定を変更でき、設計の採否は RFD0173 の再グリルで見直す。
- 固定された乱数ネットワークの出力を予測する誤差から、学習の履歴全体に対する観測の新しさを求める。
- 既定では次の観測を採点し、外的報酬をスケールした後で内的報酬を加える。
- 次の観測を採点する場合は done の遷移をマスクし、truncated は採点する。
- RND のネットワーク・optimizer・統計は専用の実装が所有し、Actor や Replay の契約は保つ。
- 内的報酬を切ったときは従来の系列と照合し、エピソード内の新しさや複数の方策とは別に RND 単独の効果を調べる。

## 内容

### 問題とゴール

anet の DQN 系 Agent には、観測の新しさに応じて Agent 自身が足す報酬（内的報酬）が無い。173 はエピソード内の新しさ（NGU の k 近傍、E3B）、RND による一生の新しさ、方策の組を 1 本で扱うが、採点点・Replay の契約・方策の表現が未決で、実装の単位にできない。
RND による一生の新しさの内的報酬を、`DefaultDQNAgent` の学習報酬に足せる。既定は無効で、無効のときは今と同じ学習系列（metrics checksum の一致）になる。
一生の新しさの口と RND の実装を、173 の最終形でもそのまま使える部品として作る。Replay・Actor・NetworkModel の契約は変えない。

### 起草時点の現状

採点と学習に使う観測は、その遷移で行動した観測 x_t。`base_intrinsic_agent.py` の `step` と `end_episode` が、新しい観測を記録する前の `self._observation` を渡す。終端の遷移も x_{T−1} で採点し、0 にしない。
誤差 e は 512 次元の MSE の平均。誤差の統計も同じ再帰で、z ＝ (e − mean) ÷ std は更新後の値で作る。内的報酬 ＝ 0.00005 × z。z は clip せず、負の値もそのまま入る。
predictor は Adam（lr 1e-4、eps 1e-5）で、eval でない agent step ごとに、その 1 件で 1 回更新する（Q の `update_period` 4・batch 32 とは別）。
`rainbow_rnd.gin` は `RNDRainbowAgent.clip_reward = False`。外的報酬は Dopamine の Runner が [−1, 1] に clip し（`clip_rewards` の既定 true）、外的＋内的の合計は clip しない。

### 起草時の暫定案

運び方は報酬へ畳む（仮）。 学習報酬 = スケール済み外的報酬 ＋ β × (α − 1)。採点する観測が `next_state`（既定）のとき、done の lane の遷移は内的報酬 0（truncated は採点する）。`state` のときは mask しない。β は `intrinsic.beta`（ProfiledValue、schedule 可）。
公開ヘッダ（`anet::rl::intrinsic`）に置くのは一生の新しさの口と生成の口だけ。RND の本体は専用の非公開ヘッダと cpp に閉じ、NetworkModel の外で自分の Network・optimizer・統計を所有する。 DQN は RND の型も `intrinsic.rnd.*` の葉も見ず、`intrinsic.enabled`・`intrinsic.beta` を読んで Builder に設定の接頭辞を渡す。
入力は facade に届く生の観測（最新 1 フレーム）で、どちらの観測を使うかは `intrinsic.observation ?= next_state`（`next_state`｜`state`）、統計の作り方は `intrinsic.rnd.statistics_mode ?= whiten`（`whiten`｜`variance_feedback`）で選ぶ。 `next_state` は遷移の次観測 x_{t+1}（今と同じ）、`state` は行動した観測 x_t（Taiga ら）。`whiten` は RND 専用の `ObservationNormalizer` で累積の平均・標準偏差により白色化して clip し、誤差の統計は Welford。採点は更新前の統計で行い、採点の後に更新する。`variance_feedback` は Taiga らの式で、観測は正規化後の値で統計を更新し（フィードバック）、分散に相当する値で割り、誤差の z は更新後の統計で作る。lane 順に 1 件ずつ更新する。
warm-up。 採点した観測の件数が `intrinsic.rnd.warmup_steps` に達するまでは統計だけ更新し α=1（内的報酬 0）を返し、`Learn` は何もしない。件数は intake で呼ぶなら exp step と同じで、既定 12,800 は N=128 の採点 100 回と同じ長さ。0 なら warm-up を置かない（Taiga らの条件）。

### 公開契約

名前空間 `anet::rl::intrinsic`、公開ヘッダ `include/anet/intrinsic.hpp`。anet-core の公開 API（`NetworkBuilder`・`Network`・`ObservationNormalizer`・`SeedMaker`・`Serializable`）だけに依存し、DQN の型を参照しない。
DQN が見るのは口と Builder だけで、RND の型・設定 DTO・葉の名前は非公開ヘッダ（`src/intrinsic_rnd.hpp`）に閉じる。テストは非公開ヘッダを含めて RND を直接試験する。
obs_m・obs_v は観測と同じ形（成分ごと）、err_m・err_v はスカラー。Taiga らのコードの変数名は `std` だが中身は上の再帰で、平方根を取らない。
warm-up 中は bonus 系が 0、error 系は値を持つ。無効のときは key が存在せず（`std::nullopt`）、群に tag を足さないので defs は変わらない。

### 候補を採る場合の受入

有効の checkpoint の読み書きが無効の checkpoint のバイト配置を変えない（無効で保存したファイルが着手前と同一）。
群は有効（`run.@rnd`）と無効の 2 つに、Atari では `run.@rnd > run.@rnd_taiga` を足した 3 つ。群を交互に（ABBA の要領の巡回で）3 巡以上回し、群ごとの実時間の中央値と Tracy の内訳（target forward・predictor forward・backward・統計・転送）を記録する。環境は Atari（B=64）と GridMaze。
Montezuma で `run.@rainbow_like > run.@rnd > run.@rnd_taiga` を回し、落ちない・群 56 の全指標が全点で有限（上の NaN の例外を除く）・`56_agent_in/04_bonus_min` の点のうち負の点が 9 割以上（z の負が報酬に入っている。warm-up 0 なので最初の点から数える）。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
