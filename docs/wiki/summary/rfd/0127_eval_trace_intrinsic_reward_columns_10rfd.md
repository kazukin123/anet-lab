---
title: Evaluation Trace で見る RND の内的報酬
page_kind: サマリ
material: "[[docs/memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md|0127_eval_trace_intrinsic_reward_columns_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# Evaluation Trace で見る RND の内的報酬

## 元の材料

[RFD0127　Evaluation Trace で見る RND の内的報酬](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) blob:1fb6b2a0643840aadec8d0ec5355d50f74ed296d

## 要点

- 学習中の集計だけでは、Greedy 評価で止まった画面の新しさを特定できなかった。
- 評価の各 step で観測の予測誤差、標準化した値、内的ボーナスを Trace の列として読めるようにする。
- 評価用の採点では重み、統計、観測件数を進めず、評価操作によって学習結果が変わるのを防ぐ。
- 評価時の値は、その時点の予測器から見た新しさを示す診断値として読む。
- 単一の画面や lane から値を取り出し、学習報酬と同じ尺度で読める場合と空欄になる場合を区別する。

## 内容

### 目的と問題

RND を有効にした Run で、評価の 1 step ごとに新しさを見る手段が無い。
内的報酬は、Agent が Train の遷移を受け取って Replay へ渡す入口でだけ計算する。評価の Runner はこの入口を呼ばないので、評価の step には内的報酬の値が無い。
学習側の値は群 `56_agent_in` の集計だけで、step ごと・画面ごとには見えない。

### 起草時点の現状

推定器を書き換えるのは、この入口と、checkpoint の読み込みだけである。読み込み（`LoadNetwork`）は Agent のコンストラクタからだけ呼ばれ、その時点に step 診断の部品はまだ無い。
`RndNoveltyEstimator::Estimate`（intrinsic_rnd.cpp）は、呼ぶたびに次を更新する。
`learner.enabled=false` でも、`intrinsic.enabled=true` なら推定器は作られ、checkpoint から predictor と統計を読む。採点と補助学習は休止する（default_dqn_agent.cpp の構築と `LoadNetwork`）。学習の入口は呼ばれないので、β は評価されない。
スペクトル正規化の内部状態は、学習モードで勾配が有効な forward でだけ更新する（nn_modules.cpp の `MakeEffectiveWeight`）。採点は勾配なしで forward するので、状態は動かない。

### 確定事項

学習用の採点は、Agent の学習の入口のままにする。Evaluation Trace の値は、同じ predictor と統計を読むだけの採点で、可視化用に別に計算する。学習用の採点を Actor 側へ移すか、方策が内的報酬を入力に取るかは、173 の問いとして開いたままにする。
写しを持たない Actor は、forward がすでに step ごとに読み取りロックを待っているので、同じ待ち方を足しても、その設定を選んだ利用者の想定を外れない。
DefaultDQN で `intrinsic.enabled=false` のとき、3 つの鍵は認識して NaN を返す。Rainbow は内的報酬の鍵を知らないので、列に書くと既存の規則どおり最初の読み出しで止まる。
「写しが無い」の NaN が出るのは、部品を作ってから Actor がまだ写し直していない間と、休んでいる間に写しを捨てたあと、Actor がまだ写し直していない間である。RunnerApp では、pane を開いたときの同期があるので、ほぼ出ない。Runner を直接使う構成（テスト、RFD0184 の評価 session）では、最初の同期までの間に出る。

### 仕様

`variance_feedback` の標準化した値は分散で割るので、whiten の値と単位が違う。方式の違う Run どうしで `rnd_norm` の列は比べられない。
誤差が有限でない lane は、エラーにせず NaN で返す。学習の `Estimate` が非有限の誤差で止まる動きは変えない。
鍵の一覧に `intrinsic.` で始まる鍵が無いときは、採点せず、推定器の写しも作らず、Actor にもつながず、Agent のロックも取らない。この一覧に無い鍵を読まれたときも、既知の鍵なら NaN を返す（一覧は計算するかどうかだけを決め、鍵の認識は変えない）。
表示をやり直したとき（`Reset`）は写し直さない。Actor の network もそのとき写し直さないためである。

### 実装契約

`LifelongNoveltyEstimator::Clone`: 同じ状態の推定器を作る。写しを採点しても、元の推定器は変わらない。呼び出し側は、元の推定器が書き換えられない状態（Agent の読み取りロックの中）で呼ぶ。
`LifelongNoveltyEstimatorFactory::LaneDiagnosticKeys`: 推定器を作らない構成でも、lane ごとの診断値の鍵の一覧を返す（既存の `InitialDiagnostics` と同じ考え方）。DefaultDQNAgent は、これに接頭辞 `intrinsic.` を付けた鍵と `intrinsic.bonus` を、step 診断の既知の鍵にする。
`ActorSyncFollower`: 同期に従うものの口。Actor は、source から写し直すとき、source を守るロックの中でこれを呼ぶ。呼ばれた側は、同じ source から自分の写しを写し直す。
別の Agent が作った Actor を渡すのは契約違反だが、検出しない。基底の `Actor` には、どの Agent が作ったかを確かめる口が無く、`dynamic_cast` は実装の規約で使えない。Run の中の Agent は 1 つなので、起きるのはテストの書き誤りに限られる。写しを持たない Actor と、DQN 系でない Actor は、`AttachSyncFollower` が偽を返すので、step ごとに自分の Agent の読み取りロックを取る形になり、別の Agent のロックの中で写すことは起きない。

### ほかの RFD との関係

RFD0123: 誤差の統計の取り方、warm-up、平均を引かない形、done の lane の扱いは、本 RFD の読むだけの採点がそのまま従う。
173: 問い 7（ボーナスの合成と尺度）と 11（保存する報酬）は開いたままである。Actor 側の採点や、方策が内的報酬を入力に取る形が入ると、Actor が出す値を読む列が別にできる。評価の Actor は学習用の統計を更新できないので、そのときも読むだけの採点の口と写しは使える。
RFD0121（外的と内的の価値の分解。暫定）: 内的の価値を別の Head で学ぶ形が入ると、`intrinsic.bonus` の列を内的の価値の列と並べて読める。本 RFD は 121 を待たない。
RFD0170（未実装）: `MakeAction` と Runner の step に引数を足す。本 RFD とは別の口だが、同じ基底とテスト用の Agent を触るので、実装の順によって書き換えが重なる。意味はぶつからない。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
