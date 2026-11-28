---
title: RND の生の予測誤差と最近の観測による補助学習
page_kind: サマリ
material: "[[docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md|0191_rnd_raw_error_and_recent_learning_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# RND の生の予測誤差と最近の観測による補助学習

## 元の材料

[RFD0191　RND の生の予測誤差と最近の観測による補助学習](../../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) blob:5a287f3df0375254df3abdb9ede4cf63617936dd

## 要点

- 従来の RND では誤差の尺度やクリップが先に決まり、元論文に近い生の予測誤差を内的報酬にできなかった。
- 誤差の計算、統計、報酬の生成、補助学習の各段階に沿って設定を整理する。
- 生の誤差を返す方式と、既存の標準化した方式を区別して選べるようにする。
- predictor は今の intake だけで学ぶ方式に加え、最近の観測をためて複数エポックに分けて学ぶ方式を持つ。
- RND の target と predictor の構造を Run の成果物で確認し、構成と学習結果を対応づける。

## 内容

### 目的と問題

predictor の補助学習を、論文の学び方にできない。 論文は、rollout ごとに集めた観測の 25% を残し、4 エポック、約 1,000 件ずつのミニバッチで学ぶ。anet は、今の intake の観測だけから、各観測を高々 1 回、`learn_batch_size` 件ずつ引く。1 step の件数は lane の数を超えられず、`learn_ratio` は 1 以下である。既定（`learn_ratio` 0.25、32 件）では 1 経験あたり 0.25 回の勾配で、論文の約 1 回の 4 分の 1 である。`learn_ratio` を 1 にすれば回数はそろうが、同じ観測の再利用と、lane の数を超えるバッチは作れない。

### 背景

論文（Burda ら、ICLR 2019）は、PPO の上に RND を載せる（付録 A.1）。
内的報酬。 次観測について、target と predictor の出力の差の二乗ノルムを内的報酬にする。内的報酬は、内的の収益の標準偏差の移動推定で割る。内的報酬の clip は無い（表 2）。
表 5 から計算すると、predictor は 1 経験あたり約 1 回の勾配を受け（残した 25% を 4 エポック）、Adam の 1 step は約 1,000 件になる（128 env × 128 step ÷ 4 ミニバッチ × 0.25）。
NGU は、actor が行動のときに内的報酬を計算し、報酬に足して Replay へ入れる。learner は優先度つきで引いた系列で Q を学び、その系列の末尾 5 時点だけで、行動予測と RND の predictor を学ぶ（付録 A）。

### 起草時点の現状

warm-up 明けが大きいのは、統計が 0 ばかりで、標準偏差が小さいからである。最初の intake では、統計は 0 が warm-up の件数ぶんと e が 128 個で、標準偏差は e × √(p(1 − p))、p = 128 ÷（warm-up の件数 ＋ 128）になる。
warm-up 0 の最初の intake は、統計に 128 個の値しか無い。標準偏差は lane の間の誤差のばらつきになる。Run の最初は全 lane がほぼ同じ画面なので、ばらつきが小さく、非常に大きな値になりうる。2 回めは約 2 である。
誤差が一定という前提なので、落ち着いたあとの値は目安である。大きな内的報酬が付いた遷移は Replay に残り、Q の学習でくり返し使われる。実害の大きさは測っていない。
n-step の待ち行列は、tensor ではなく小さな記録の列なので、同じ部品にはならない。

### 確定事項

割らない式の待ちの間の、内的報酬スケーラの統計。 割らない式では、待ち（`rnd.error_norm.none.warmup_steps`）の間も、推定器は式どおりの倍率（1 ＋ e）を返す。呼び出し側は、その値を内的報酬スケーラに通して統計を育て、スケーラを通したあとの値を 0 にする。論文の形は、待ちを rollout 1 回ぶん（128 lane で 16,384 件）にして使う。誤差を標準化する式と `variance_feedback` の動きは変えない。
設定の組み替え。 RND と補助学習の設定キーを、段階ごとのスコープに分ける。スコープは、統計（`rnd.statistics`）、誤差（`rnd.error_norm`）、predictor（`rnd.predictor`）、補助学習（`learn`）である。モードを持つ段階は、モードのキー（`mode`）と、モードの名前の名前空間（`<スコープ>.<モード>.<キー>`）で持つ。モードによって意味や適した値が変わるキーは、被ってもモードごとに持つ。今あるキーも、本 RFD で移す。
clip を、誤差のモードごとに持つ。 倍率の clip は、列挙のモードにせず、真偽のキー（`use_clipping`）と上限（`multiplier_max`）で持ち、誤差のモードごとの名前空間に置く。倍率だけのスコープは作らない。
統計と誤差のスコープ。 2 つは横並びのスコープにする。統計が `variance_feedback` のとき、誤差のスコープ（`rnd.error_norm.*`）は丸ごと使わない。統計のモードの 1 キーで切り替わる。

### 仕様

件数は、採点した観測の数である。whiten は batch を採点する前の件数で、`variance_feedback` は lane ごとの件数で判定する（今と同じ）。
新しく書ける組が 1 つある。predictor の待ちを短く、誤差の統計の待ちを長くすると、「最初は観測の統計だけを育て、次に predictor を学ばせ、最後に誤差の統計と内的報酬を始める」形になる。RFD0123 で先送りにした、2 ステージの warm-up に当たる。本 RFD は、この組を書けるようにするだけで、効果は確かめない。
`rnd.error_norm.mode = standardize` は、今の式のままである。
学習が無効（`learner.enabled = false`）の Run では、今と同じく、ためることも学ぶこともしない。

### 実装契約

行 12 の step 数は倍精度で計算し、最も近い整数との差が相対で 1e-6 以下なら、その整数として扱う。小数の表現の誤差を吸収するためで、値を丸めて動かすものではない。
黙った代替はしない。step 数を丸める、`batch_size` を lane の数に合わせて変える、といった扱いをしない。
口に足す操作を「待ちの間か」ではなく「呼び出し側が 0 にするか」で返すのは、呼び出し側が式の種類を知らずに済むようにするためである。標準化する式と `variance_feedback` は、推定器が自分で倍率 1 を返しており、呼び出し側が重ねて 0 にする必要が無い。
名前のある組み合わせ。 既定の RND、Taiga らの再現、Burda らの形を、段階ごとの選び方の表と、設定の行で示す。

### 成功の測り方と受入

読む指標。 評価側（Greedy と ε 0.01）の点、Montezuma なら 2500 点の割合と訪れた部屋の種類数。学習側では `56_agent_in/05_bonus_pos_ratio`（内的報酬が正になる割合）、`01_bonus_mean` の時間の変わり方、`12_error_std`、`31_predictor_loss`。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
