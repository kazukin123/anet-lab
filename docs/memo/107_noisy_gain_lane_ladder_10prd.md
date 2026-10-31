# 学習の Actor の lane ごとに NoisyNet のノイズの強さを変える（ノイズのラダー）PRD

> 2026-10-29 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 起点: qbert の評価（Greedy）で、同じ得点で終わる決まった失敗の道が、スキップの有無や動かし方によらず見つかった（[PRD106](106_temporl_skip_forward_noise_key_10prd.md) §2.2 の 7）。Greedy がたどる道の状態が学習のデータに少ないので、その道の上の誤りが直らない、という見立てから、学習の lane の一部をノイズの弱い（ノイズの無い）動き方にして、Greedy に近い道のデータを replay に入れる形を起こした。lane ごとの強さを、ε のラダーと同じく ProfiledValue の両端で書く案は、2026-10-29 のユーザーの発案。
> 改訂: 2026-10-31。qbert の TempoRL の調査を区切ったので、ステージ 1 の構成と比べる相手を、合算からスキップなしの 2 seed に置き直した（D6・§3 の比べ方と G2・§7・§9 の 6・§10）。暫定のまま。
> 関連: [PRD084](084_noisynet_10prd.md)（NoisyNet）、[PRD101](101_exploration_schedule_profiled_value_10prd.md)（探索と学習のスケジュール。ProfiledValue と spatial のラダー）、[PRD106](106_temporl_skip_forward_noise_key_10prd.md)（skip 側の NN 実行設定）、[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)・[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)（NoisyNet のノイズは呼び出し側が持つ状態と NN 実行設定で駆動し、呼び出し情報を forward に渡す）、[NoisyNet の観測（wiki）](../wiki/mechanisms/noisynet.md)、[Qbert（wiki）](../wiki/subjects/qbert.md)。

## 1. 目的と問題

qbert のサンプル効率優先の構成（BTR に寄せた構成）では、学習の Train Actor は 25M 以降 ε=0 で、NoisyNet のノイズだけで探索する。全 lane が同じ NN 実行設定（`act_sample`）で動き、ノイズの強さは lane によらず同じである（§2.1 の 1・2）。

一方、評価の Greedy（ノイズなし・τ 固定・ε=0）は、始まり方（noop 最大 30）の数だけの道をほぼそのまま繰り返し、その道のどこかに誤りがあれば、毎回同じ得点でゲームが終わる（§2.2 の 1）。この決まった失敗は、TempoRL の合算・分割、スキップなし、`single` の動かし方のどれにも見つかった。

見立て: 学習のデータは、ノイズで揺れる Train Actor が集めるので、Greedy がたどる道の状態は replay に少ない。その道の上の誤った判断は、経験されず、直す勾配がほとんど来ない。学習の lane の一部を、ノイズの弱い（または無い）動き方にすれば、Greedy に近い道のデータが replay に入り、その道の上の誤りが経験されて直る見込みがある。

今は、NoisyNet のノイズの強さを lane ごとに変える口が無い。ε には lane ごとのラダー（spatial）があり、効いた記録もある（§2.2 の 2）。

## 2. 現状（2026-10-29 に確認したこと）

### 2.1 実装と設定

1. Actor の方策には、ε・UQE の値を lane ごとに変えるラダー（spatial exploration）がある。`spatial.enabled` で選び、`spatial.scale = linear｜log` と、`spatial.epsilon.first/last`・`spatial.uqe.epsilon.first/last`・`spatial.uqe.tau.first/last` で lane の両端を指定する。両端は ProfiledValue で、`SpatialLadderSchedule` が両端を評価してラダーを作る。1 lane のときは first になる。ラダーを使わないときの値は `epsilon` などの ProfiledValue で、全 lane に同じ値を使う（[200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md) 60-62 行）。使わない側の値は評価せず保持し、値そのものの検査は両側に行う。使わない側の値に WARN は出さない（[PRD101](101_exploration_schedule_profiled_value_10prd.md) D6 と受入 6。設定の読み取りは [agent.hpp:447-462](../../core/anet-core/include/anet/agent.hpp)）。
2. NoisyNet のノイズは、NN の部品（Body の `Linear` と Head の射影）が、NN 実行設定（`nn_forward.[<name>]` の `noisy.mode`・`hold`・`share`・`count`）を準備時に解釈し、呼び出しごとに引く（[130 ニューラルネットワーク（設計文書）](../design/130_neural_networks.jp.md) 346-348 行）。Actor・Learner・共通の実行処理は、Noisy の有無で forward の挙動と設定の検証を切り替えない（[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) の Consequences）。Train Actor の設定 `act_sample` は `noisy.mode = sample`・`hold = call`・`share = batch`（1 回の forward で全 lane が同じノイズ）である（合算の Run の設定の写し [run_20261027-004945_a5_qb_aln_t98mrg_s1](../experiments/default-dqn/atari/config/run_20261027-004945_a5_qb_aln_t98mrg_s1.txt) 796-800 行）。
3. forward の呼び出し情報（`NetworkForwardCallInfo`）は、時計の値、行ごとのエピソード開始のマスク、行対応を持つ（[nn.hpp:139-155](../../core/anet-core/include/anet/nn.hpp)、[CONTEXT.md](../../CONTEXT.md) の「NN 呼び出し情報」）。行ごとの値を forward へ運ぶ口は、エピソード開始のマスクとしてすでにある。
4. 合算の構成の Train Actor は、`spatial.enabled = false`、ε は `phased`（1.0 から時定数 2M で 0.01 へ下げ、25M で 0 にする）である（同じ写しの 809・831・928・932-936 行）。skip 専用の方策（`skip.policy.*`）は書かれていないので、skip 側は Q 側と同じ方策を使う（`skip.policy.` で始まるキーが 1 つでもあれば skip 専用の方策を作り、無いか方策が Greedy なら共有する。[default_dqn_agent.hpp:158](../../core/anet-core/include/anet/default_dqn_agent.hpp)、[default_dqn_agent.cpp:575](../../core/anet-core/src/default_dqn_agent.cpp)）。skip 専用の方策の設定は、主の方策（`policy.*`）の設定を写し、書かれた `skip.policy.*` のキーだけで上書きして作る（[default_dqn_agent.hpp:486-487](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
5. Learner の forward は `learner.current`・`learner.target` の NN 実行設定で動き、Actor の設定とは別である。
6. NoisyNet の診断の指標（`47_agent_noisy`）は、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の `metrics.scalar.@noisy`（344-354 行）にあり、どれも Learner の更新の前に測る（[200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md) §9.7、529 行）。ε のラダーの指標（`epsilon_lane_min/max`）は、同じファイルの `metrics.scalar.@baseline` に `$actor` の値として `@learn $exp_step` で並ぶ（39-43 行）。

### 2.2 実測

1. Greedy の評価（各 200 ゲーム）で、同じ得点で終わる道の集中: 合算の `full` は面 5 で 45 本中 43 本が 14,550 点、分割の `full` は面 6 で 52 本中 51 本が 18,925 点（評価の seed 2 では 57 本すべて）、スキップなしの seed 1 は面 8 で 45 本が 25,750 点、seed 2 は面 5 で 32 本が 15,850 点・面 8 で 53 本が 26,050 点、分割の `single` はクリア数 8 で 60 本が 26,175 点（[PRD106](106_temporl_skip_forward_noise_key_10prd.md) §2.2 の 7）。同じ checkpoint の評価に skip 側の τ やノイズのランダムさを入れると、集中した面で終わるゲームの得点はばらける。Q 側の τ だけをランダムにした評価は、分割の面 6 の道からほとんど外れない（面 6 で終わった 35 本中 33 本が 18,925 点。同 §2.2 の 7）。
2. ε のラダー: qbert で ε のラダーを 1e-4〜0.05 へ下げて効いたのは、低 ε の lane の割合で、同じ ε での腕前ではなかった（[Qbert（wiki）](../wiki/subjects/qbert.md) の「分かっていること」。出典は [2026-09-25 の実験記録](../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md) の現時点のまとめ 12）。
3. NoisyNet と ε のラダー: Atari の実時間優先 50M で ε のラダーを NoisyNet に置き換えると 7 ゲームとも基準を下回り、Breakout の 2×2 では NoisyNet の有無の差がラダーありで +15.3、なしで +0.8 だった（[NoisyNet の観測（wiki）](../wiki/mechanisms/noisynet.md)。出典は [2026-10-11 の実験記録](../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)）。lane ごとに探索の強さを散らすことが効いている可能性を示すが、NoisyNet の強さのラダーを測った結果は無い。
4. Evaluation Trace の観察（3 ゲーム）: 最後でない残機を失っても `V_s` はほとんど変わらなかった（[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md) §1）。誤りを経験しても直す信号が弱い可能性があり、本書の効き目を弱めうる。

## 3. ゴール・非ゴール・仮説と成功の測り方

**用語**: 本書の「ノイズの倍率」は、NoisyNet の部品がノイズの項（σ とノイズの積）に掛ける、学習しない 0 以上の係数である。[CONTEXT.md](../../CONTEXT.md) の「ノイズスケール（σ）」は学習する係数で、「ノイズサンプル」は ε で、どちらもこの係数を言い表せない。設定キーとメトリクス名は `noisy_gain` とする（`noisy_scale` は「ノイズスケール（σ）」と同じ語に読め、「勾配ノイズスケール」の `_Avoid_` にも NoisyNet と紛れる語として挙がっている）。

**ゴール**

- G1: Train Actor の lane ごとに、ノイズの倍率を変えられる。倍率は ε と同じく ProfiledValue で、ラダーでは両端を ProfiledValue で書ける。倍率 0 の lane はノイズの無い forward と同じ値になる。既定（倍率 1）では今と同一。
- G2: qbert で、lane の一部をノイズの弱い（または無い）動き方にした学習が、Greedy の評価の決まった失敗を減らすかを確かめる（ステージ 1）。

**非ゴール**

- Learner の forward のノイズの強さを変えること。消える案は Learner にも倍率を足す案で、本当の制約ではないが、問いはデータを集める側の動き方である。
- 既定の変更（BTR に寄せた構成の既定を、ラダーありにするなど）。消える案はその変更で、本当の制約ではないが、ステージ 1 の結果が無いと決められない。
- τ の取り方を lane ごとに変えること（ノイズの無い lane の τ も固定にするなど）。消える案は τ のラダーを足す案で、本当の制約ではない（同じ形で足せる）が、まずノイズだけの効き目を見る（§9 の 3）。
- 残機を失うことを学習上の区切りにすること（`episodic_life`）。消える案は学習だけ `episodic_life` を真にする案で、本当の制約ではない（設定 1 つで変えられる）が、2026-10-29 にユーザーが外した。
- 残機を失ったときの報酬を足すこと。消える案はその報酬の案で、Atari のゲームの仕様を変えない方針による制約である。
- 評価の Actor の動き方を変えること（評価は今までどおり `mu_only`）。消える案は評価の Actor にも倍率を書く案で、本当の制約ではない（同じキーを評価の Actor にも書ける）が、Greedy の評価は比べる物差しなので変えない。

**仮説・比べ方・終わり方**

- 仮説: Greedy の決まった失敗は、Greedy の道の状態が replay に少ないために直らない。ノイズの弱い lane を混ぜれば、その道のデータが入り、決まった失敗が減る。
- 比べ方: ステージ 1 で、スキップなしの構成（`run_20261025-231207_a5_qb_aln_btreps_s1` と同じ）にノイズのラダーを足した qbert 50M を 1 本流し、スキップなしの seed 1（`run_20261025-231207_a5_qb_aln_btreps_s1`）・seed 2（`run_20261028-132638_a5_qb_aln_btreps_s2`）と比べる。
- 終わり方: §7 の読む値を記録したところで終える。1 seed なので合否ではない。

**成功の測り方**

- G1: §7 の受入 1〜4 が通る。
- G2: ステージ 1 の Run の 50M の checkpoint を Greedy で 200 ゲーム評価し、同じ得点で終わる道の集中（PRD106 D9 と同じ決め方）の数と、全体の 8 面突破率を、スキップなしの seed 1・seed 2 と並べて記録する。あわせて、学習側の読む値（§7）を記録する。

## 4. 確定事項（暫定の既定。§9 で見直す）

- **D1. ノイズの倍率は Actor の方策の値として持つ。** 時間の値 `policy.noisy_gain`（ProfiledValue）と、ラダーの両端 `policy.spatial.noisy_gain.first/last`（ProfiledValue）を足す。ラダーは今の `spatial.enabled` と `spatial.scale` に従い、ε・UQE と同じ `SpatialLadderSchedule` で lane に割り当てる。使わない側の値の扱いも ε・UQE と同じにする（評価せず保持し、値の検査は両側に行い、WARN は出さない。§2.1 の 1）。理由: lane ごとのラダーと、時間の ProfiledValue は、方策がすでに持つ仕組みで（§2.1 の 1）、exp_step で値を評価できるのも方策の側である。NN の部品は lane も exp_step も知らない（§2.1 の 2・3）。依存: §2.1 の 1。
- **D2. 倍率は、forward の呼び出し情報に行ごとの値として載せて、NoisyNet の部品へ渡す。** 部品はノイズの項（σ とノイズの積）に、その行の倍率を掛ける。倍率 0 の行は μ だけの値になり、倍率 1 の行は今と同じ値になる。ノイズの引き方（`hold`・`share`）は変えない（`share = batch` なら、全行が同じノイズを引き、行ごとに倍率だけが違う）。理由: 呼び出し情報は、行ごとのエピソード開始のマスクをすでに運んでいて（§2.1 の 3）、ノイズは呼び出し側が持つ状態と設定で駆動する、という既存の判断（ADR 0048・0051）の中に収まる。既存の ADR との関係: [ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) は、呼び出し情報を「時計・エピソード開始・行対応」を運ぶものとし、呼び出し情報を組み合わせる処理では抽選・State の更新・Tensor の計算をしないと決めている。本書は呼び出し情報に運ぶ値を 1 つ足す（拡張）。倍率を掛ける計算は NoisyNet の部品の本体で行い、呼び出し情報を組み合わせる処理には入れないので、後者の判断とは食い違わない。ADR にするかは §9 の 1。依存: §2.1 の 3。
- **D3. 倍率は 0 以上の有限の値で、上限は置かない。** 負・NaN・∞ は、設定を読む時点でエラーにする。検査は、使う側か使わない側かによらず、時間の値と両端のすべてに、定数なら値そのもの、スケジュールなら各段の値に行う（ε の値の検査と同じ時機。§2.1 の 1）。理由: 1 を超える倍率（ノイズを強める）も、探索を強めるラダーとして意味がある。値の検査を読む時点にそろえれば、使わない側の不正な値も見逃さない。依存: D1。
- **D4. Actor は、NN 実行設定と Network の中身によらず、倍率を呼び出し情報に載せる。倍率を使うのはノイズを引く NoisyNet の部品だけで、ノイズを引かない準備の部品（`noisy.mode = mu_only`）と、NoisyNet の無い Network では、倍率は値に効かない。DQN 側は、倍率が効くかどうかを見た検証も WARN もしない。** 理由: [ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) は、Actor・Learner・共通の実行処理が Noisy の有無で forward の挙動と設定の検証を切り替えないと決めている（§2.1 の 2）。効かない倍率に WARN を出すには、DQN 側が `noisy.mode` か NoisyNet の有無を見る必要があり、この判断に反する。`mu_only` の forward は、倍率がいくつでも μ だけの値で、倍率 1 のときと同じ値になる（置き換えではなく、掛ける相手のノイズの項が無い）。部品の側で WARN を出すかは §9 の 5。依存: D2、ADR 0048。
- **D5. 倍率は、その forward を動かす方策の値を使う。** Q 側の forward には Actor の方策（`policy.*`）の倍率を、skip 側の forward（PRD106）には skip 側の方策の倍率を載せる。skip 側が Q 側と方策を共有するとき（`skip.policy.*` が無いか、方策が Greedy。§2.1 の 4）は同じ倍率になり、skip 専用の方策があるときは、その方策の倍率を使う。skip 専用の方策の倍率は、ほかの方策の値と同じく主の方策から写り、`skip.policy.noisy_gain`・`skip.policy.spatial.noisy_gain.first/last` を書いたものだけが上書きされる（§2.1 の 4）。倍率が効くかは、forward ごとにその NN 実行設定で決まる（D4）。PRD106 の `shared` では Q 側と同じ NN 実行設定、`dedicated` では `skip.nn_forward_key` が指す NN 実行設定で、Q 側の設定とは独立に効く。理由: 倍率は方策の値である（D1）。skip 側の ε も、skip 側の方策の値を使っている。依存: D1、D4、PRD106 の実装（§10）。
- **D6. ステージ 1 の設定は、スキップなしの構成に、ラダーの両端 first = 1（今と同じ強さ）・last = 0（ノイズなし）・`spatial.scale = linear` を足す。** ε は今の `phased` を、ラダーの両端に同じスケジュールとして書いて保つ（`spatial.enabled = true` にすると ε もラダーの両端で決まるので、両端を同じにして全 lane を同じ ε にする。スキップなしの構成の Train Actor も、合算と同じく `spatial.enabled = false`・ε は `phased`。[実効設定の写し](../experiments/default-dqn/atari/config/run_20261025-231207_a5_qb_aln_btreps_s1.txt)）。skip は無効なので、skip 側の forward の倍率（D5）はステージ 1 では使わず、受入 2 の単体テストで確かめる。理由: lane の半分ほどがノイズの弱い動き方になり、ノイズの無い lane も含まれる。ε の扱いは今と同じにして、ノイズのラダーだけを比べる。2026-10-31 に合算の構成から置き直した（qbert の TempoRL の調査を区切ったので、比べる相手を TempoRL の無いスキップなしの 2 seed にする。決まった失敗の道はスキップなしにもある。§2.2 の 1）。依存: D1、§2.1 の 1、§2.2 の 1。

## 5. 実装契約

確定事項と本節には実装の語（内部の型、入れ物の型、添字の決め方）を書かない。公開 API 名・設定キー・メトリクス名・クラス一覧の名前は書く。

### 5.1 設定キー

`DefaultDQNAgent.actor.[<key>].` の下。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `policy.noisy_gain` | 0 以上の有限の ProfiledValue | 1（定数） | ラダーを使わないとき、全 lane のノイズの倍率 |
| `policy.spatial.noisy_gain.first` | 同上 | 1（定数） | ラダーの最初の lane の倍率 |
| `policy.spatial.noisy_gain.last` | 同上 | 1（定数） | ラダーの最後の lane の倍率 |

- `spatial.enabled = true` のときは両端のラダーを使い、`policy.noisy_gain` は評価せず保持する。`false` のときは `policy.noisy_gain` を使い、両端は評価せず保持する。使わない側の値に WARN は出さない。値の検査（D3）は両側に行う（ε・UQE と同じ規則。§2.1 の 1）。
- skip 専用の方策の `skip.policy.noisy_gain`・`skip.policy.spatial.noisy_gain.first/last` も同じ型・値域・規則で、skip 側の forward に使う（D5）。書かなければ、表の既定（1）ではなく、主の方策の同じキーの値が写る（ほかの `skip.policy.*` と同じ。§2.1 の 4）。表の既定は主の方策（`policy.*`）の既定である。
- `spatial.scale = log` の扱いは ε と同じ（`max(端, 1e-4)` の列を指定端へ伸縮し、0 の端は 0 のまま）。
- 既定は C++ 側に置き、`agent.txt` にも `?=` で書く（`[train]` と `@eval_base` の `policy.*` の下。PRD104・PRD106 の Actor の skip の設定と同じ置き方）。`skip.policy.*` の下には書かない（書くと skip 専用の方策ができて、skip 側の方策の共有が変わる。§2.1 の 4）。

設定例（コロン記法。ステージ 1。ε は今の `phased` を両端に同じく書く）:

```
run.@noisyladder : A2.actor.[train].policy.spatial.enabled = true
run.@noisyladder : A2.actor.[train].policy.spatial.scale = linear
run.@noisyladder : A2.actor.[train].policy.spatial.noisy_gain.first = 1
run.@noisyladder : A2.actor.[train].policy.spatial.noisy_gain.last = 0
run.@noisyladder : A2.actor.[train].policy.spatial.epsilon.first.$ = A2.actor.[train].policy.epsilon
run.@noisyladder : A2.actor.[train].policy.spatial.epsilon.last.$ = A2.actor.[train].policy.epsilon
```

（ε の両端に今の `phased` のスケジュールを写す書き方は、設定の記法で確かめる。§9 の 4）

### 5.2 判定の網羅

forward（Q 側と skip 側）ごとに、その forward の NN 実行設定と、その forward を動かす方策（D5）の設定で決まる。Q 側と skip 側は互いに独立に決まる。

| その forward の NN 実行設定 | 方策の `spatial.enabled` | 使う側の値 | 答え |
|---|---|---|---|
| ノイズを引く（`noisy.mode = sample`） | false | `noisy_gain` が 1 | 今と同一 |
| ノイズを引く | false | `noisy_gain` が 1 以外 | 全 lane に同じ倍率 |
| ノイズを引く | true | 両端が 1 | 今と同一 |
| ノイズを引く | true | 両端のどちらかが 1 以外 | lane ごとにラダーの倍率 |
| ノイズを引かない（`mu_only`、または NoisyNet なし） | どちらでも | どれでも | 今と同一（倍率は値に効かない。WARN なし。D4） |

- 使わない側の値は、どの行でも答えを変えない（評価せず、WARN も出さない。§5.1）。
- 時間の値・両端のどこか（使う側か使わない側か、定数か段の値かによらず）に負・NaN・∞ があれば、表のどの行にも進まず、設定を読む時点でエラーになる（D3）。

### 5.3 Actor と NoisyNet の部品

- Actor は、行動を選ぶたびに、その時点の exp_step で倍率（ラダーなら lane ごと）を評価し、lane の行ごとの倍率として呼び出し情報に載せる。skip 側の forward には、skip 側の方策の倍率を載せる（D5）。
- NoisyNet の部品は、ノイズを引く準備のとき、呼び出し情報に倍率があれば、ノイズの項に行の倍率を掛ける。無ければ今と同じ（倍率 1）。IQN の τ の行など、1 つの lane が複数の行になる forward では、その lane の倍率を各行に当てる（行対応はエピソード開始のマスクと同じ扱い）。
- 倍率がすべて 1 のときは、今と同じ値・同じ乱数の消費になる（受入 1）。
- Learner の forward（更新の forward と、`47_agent_noisy` の probe）には倍率を載せない（非ゴール）。

### 5.4 指標

- 方策の `GetScalar` で、評価した倍率の lane 平均 `noisy_gain` と、`noisy_gain_lane_min`・`noisy_gain_lane_max` を返す（ε の `epsilon`・`_lane_min`・`_lane_max` と同じ形。ラダーを使わないときは 3 つとも同じ値）。方策は NN 実行設定を知らないので、倍率が値に効かない forward（D4）でも、評価した倍率をそのまま返す。
- メトリクスの scalar の定義は、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の `metrics.scalar.@noisy` に、`$actor` の値として `@learn $exp_step` で足す（ε のラダーの指標と同じ取り方。§2.1 の 6）。tag は `47_agent_noisy/12_gain`・`47_agent_noisy/13_gain_lane_min`・`47_agent_noisy/14_gain_lane_max`（どれも 40 文字以内）。

### 5.5 移行対象と等価性

- 設定: `agent.txt` に既定の `?=`（§5.1）。`metrics_scalar.txt` の `metrics.scalar.@noisy` に 3 行（§5.4）。Env の設定は変えない。
- 文書:
  - [200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md) の spatial exploration の段落（60-62 行）に、ノイズの倍率とそのラダーを書く。
  - 同 §9.7 Noisy 診断（529 行。今は「σ の 8 統計と 3 つの probe 指標を更新前に測る」）に、Actor が返す倍率の 3 指標を足す。
  - [130 ニューラルネットワーク（設計文書）](../design/130_neural_networks.jp.md) の NoisyNet の段落（346-348 行）に、呼び出し情報の行ごとの倍率と、部品がノイズの項に掛けることを書く。
  - [CONTEXT.md](../../CONTEXT.md) の「NN 呼び出し情報」（今は「時計の今の値・episode_start・行対応だけを持つ」）に行ごとのノイズの倍率を足し、「ノイズの倍率」の項を「ノイズスケール（σ）」の近くに足す（§3 の用語）。
  - 英訳は翻訳の工程が追う。起票の時点では直さない（D2 は §9 の 1 で変わりうる）。
- テスト: 呼び出し情報を組むテストは、倍率を省けば今と同じに動くので直さない。
- 互換の仕組みは作らない（新しいキーで、既定が今と同一）。
- 変えないもの: 過去の Run の記録、実験記録、Learner の forward、評価の Actor の動き方。

### 5.6 性能

倍率の掛け算は、NoisyNet の層ごとに行 × 出力の要素数の積 1 回で、forward の費用に比べて小さい。forward の回数・順序は変えない。性能の受入は置かない。

### 5.7 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ActionPolicyConfig` | 変更 | `noisy_gain` と `spatial.noisy_gain.first/last` を持ち、読み取りと値の検査（D3）をする | Agent 共通の設定（agent.hpp） |
| `SpatialLadderSchedule` | 変更 | ノイズの倍率のラダーも、ε・UQE と同じ規則で作る | DQN 共通 component（dqn_based_agent） |
| `ActionPolicy`（ε-greedy・UQE などの方策） | 変更 | 倍率を評価し、`noisy_gain` と lane の最小・最大を返す | DQN 共通 component（dqn_based_agent） |
| `Actor`（DQN） | 変更 | Q 側と skip 側の forward の呼び出し情報に、それぞれの方策の lane の行ごとの倍率を載せる | DQN 共通 component（dqn_based_agent） |
| `NetworkForwardCallInfo` | 変更 | 行ごとのノイズの倍率を持てる（省けば倍率 1） | ニューラルネットワーク（nn） |
| `LinearProjectionModule`（NoisyNet の射影）と Body の Noisy `Linear` | 変更 | ノイズを引く準備のとき、ノイズの項に行の倍率を掛ける | ニューラルネットワーク（nn_modules） |

## 6. 細部（暫定で置いた既定）

1. 倍率のキーの名前は `noisy_gain`（理由: NoisyNet の `noisy.*` と並べて読め、「ノイズスケール（σ）」〔学習する係数〕とも、ε の「ノイズサンプル」〔CONTEXT.md〕とも紛れない。§3 の用語）。
2. 倍率はノイズの項（σ とノイズの積）に掛ける。σ そのもの（学習する parameter）は変えない（理由: 学習の parameter と、動き方の設定を分ける）。
3. ステージ 1 のラダーは first = 1・last = 0・linear（D6）。64 lane なら、倍率は 1, 0.984, …, 0 と並ぶ。

## 7. 検証と受入

### ステージ 1（学習。実装のあと）

- 群: スキップなしの構成（`run_20261025-231207_a5_qb_aln_btreps_s1` と同じ）に D6 の設定を足した qbert、サンプル効率優先、seed 1、50M。
- 読む値:
  1. 50M の checkpoint の Greedy の評価（200 ゲーム）: 全体の 8 面突破率と、同じ得点で終わる道の集中（PRD106 D9 の決め方。集中した面ごとに、最多の同じ得点の本数）。
  2. 学習側の 8 面突破率が初めて 50% を超える 2.5M の窓。
  3. 最初の 10M の学習側の平均点（PRD104 §2.2 と同じ 3 窓）と、5.05M・30〜45M の ε=0 の評価。
  4. `47_agent_noisy` の行動の食い違いの割合（`10_action_disagreement_ratio`）と、倍率の lane 平均・最小・最大（`12_gain`・`13_gain_lane_min`・`14_gain_lane_max`）。
- 比べる相手: スキップなしの seed 1（`run_20261025-231207_a5_qb_aln_btreps_s1`）・seed 2（`run_20261028-132638_a5_qb_aln_btreps_s2`）。ビルドはどちらも本書の前。
- 見込みありの読み: 同じ得点で終わる道の集中が、スキップなしの seed 1・seed 2 のどちらよりも小さく、全体の 8 面突破率が下がっていない。1 seed なので合否ではない。

### 受入（外から観測できるもの）

1. 既定で今と同一: 本書の前のビルドと設定一式を着手前に snapshot し、[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md) の 6 構成と、skip 有効の LunarLander に `run.@noisy` を足した構成（NoisyNet と skip の両方を通す。PRD106 の受入 1 と同じ構成）の metrics checksum が、前後のビルドで一致する（同じ短縮条件と比べ方）。
2. 単体テスト:
   - 部品ごと: Head の射影（`LinearProjectionModule`）と Body の Noisy `Linear` のそれぞれを、ノイズを引く準備で単体に呼ぶ（活性化を含まない、ノイズを持つ線形の層だけを呼ぶ）。同じ入力・同じノイズの引き方で、行の倍率を 0・0.5・1 にすると、倍率 0 の行は `mu_only` の準備の出力と一致し、倍率 1 の行は倍率を省いた出力と一致し、倍率 0.5 の行は両者のちょうど中間（各出力について、倍率 0 の値との差が倍率 1 のときの半分）になる。
   - Network: Body と Head の両方にノイズのある Network（ノイズの層の後ろに活性化を含む）で、lane ごとに倍率 0 と 1 を与えると、倍率 0 の lane の全行は `mu_only` の forward の値と、倍率 1 の lane の全行は倍率を省いた forward の値と一致する。IQN の τ の行がある forward でも、lane の倍率がその lane の全行に当たる（同じ一致で確かめる）。
   - ラダー: 両端 1・0、linear、4 lane で、lane の倍率が 1、2/3、1/3、0 になる。1 lane なら first。log の 0 の端は 0 のまま。
   - 検証: 負・NaN・∞ は、使う側でも使わない側でも、定数でも段の値でも、設定を読む時点でエラー。使わない側に 1 以外の値を書いても WARN は出ず、使う側の値で動く。`mu_only` の Actor に 1 以外の倍率を書いても構築でき、WARN は出ず、forward の値は倍率 1 のときと一致する。
   - skip 側: skip 側が方策を共有する Actor（`skip.policy.*` が無いか、方策が Greedy）では、skip 側の forward に Q 側と同じ lane の倍率が載る。方策が Greedy でなく、skip 専用の方策を持つ Actor では次のとおり。
     - `skip.policy.*` に倍率のキーを書かない（たとえば `skip.policy.tau_rule.sample_mode` だけを書く）とき、skip 側の forward には、主の方策から写った倍率（`policy.spatial.noisy_gain.*` のラダー）が載る。
     - `skip.policy.spatial.noisy_gain.*` を書いたとき、skip 側の forward にはその倍率が載り、Q 側の forward には `policy.*` の倍率が載る。
3. 指標: LunarLander に `run.@noisy`（Train Actor がノイズを引く構成。`metrics.scalar.@noisy` を有効にする）を足し、`[train]` に `spatial.enabled = true`・`spatial.scale = linear`・`spatial.noisy_gain.first = 1`・`spatial.noisy_gain.last = 0` を足した短い Run（2 lane 以上）で、`47_agent_noisy/12_gain` が 0.5、`13_gain_lane_min` が 0、`14_gain_lane_max` が 1 で記録される。
4. 設定の写し: 既定の設定の短い Run の config dump に、`[train]` の `policy.noisy_gain = 1` と、両端の `policy.spatial.noisy_gain.first/last = 1` が出る。`skip.policy.*` の倍率のキーは出ない（§5.1）。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| Learner の forward の倍率 | 範囲外 | 非ゴール。再開条件: 学習の目標値のノイズを lane とは別に弱めたい需要が出たとき |
| τ の取り方の lane ごとのラダー | 先送り | まずノイズだけの効き目を見る。再開条件: ステージ 1 で、ノイズの無い lane が Greedy の道をたどりきれていない（τ のランダムさで外れている）と分かったとき |
| ノイズの倍率を NN 実行設定（`nn_forward.[<name>].noisy.*`）に置く形 | 不採用 | NN 実行設定は行も exp_step も知らず、lane ごとのラダーや時間のスケジュールを持てない（§2.1 の 2・3） |
| lane の一部だけ別の NN 実行設定（`mu_only`）で forward する形 | 不採用 | forward が 2 回になり、lane を分けて結合する仕組みが要る。倍率 0 の行で同じ値を 1 回の forward で作れる（D2） |
| DQN 側で Noisy の有無を見て、効かない倍率に WARN を出す形 | 不採用 | ADR 0048 の「Noisy の有無で設定の検証を切り替えない」に反する（D4） |
| 名前 `noisy_scale` | 不採用 | 「ノイズスケール（σ）」と同じ語に読める（§3 の用語） |
| `episodic_life` を学習だけ真にする形 | 範囲外 | 2026-10-29 にユーザーが外した |
| 残機を失ったときの報酬 | 不採用 | Atari のゲームの仕様を変えない方針 |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | 倍率を forward へ渡す口（呼び出し情報に行ごとの値を足す形か、別の口か）と、それを ADR にするか | 呼び出し情報に足す（D2）。CONTEXT.md の「NN 呼び出し情報」を書き換えるので、グリルで決まったら ADR 0051 に注記して新しい ADR を起こすかを決める |
| 2 | ステージ 1 のラダーの形（first = 1・last = 0・linear か、last を 0.2 などにして完全なノイズなしを避けるか、log か） | first = 1・last = 0・linear（D6）。結果を見て形を変えた群を足すかを決める |
| 3 | ノイズの無い lane の τ（Train Actor は τ ランダム）を固定にするか | 固定にしない（非ゴール）。§8 の再開条件 |
| 4 | ε の `phased` のスケジュールを、ラダーの両端へ写す設定の書き方（`$` での写しが ProfiledValue の入れ子に効くか） | 実装計画で設定の記法を確かめる。効かなければ、両端に同じスケジュールを書き下す |
| 5 | ノイズを引かない準備の NoisyNet の部品が、1 以外の倍率を受け取ったときに、部品の側（`noisy.mode` を解釈する側。ADR 0048 に反しない）で WARN を 1 回出すか | 出さない（D4）。`mu_only` の評価の Actor に倍率を書く設定ミスが実際に起きたら足す |
| 6 | ステージ 1 をいつ流すか（メイン機の空き、NGU の準備との順） | 段取りを示して承認を得てから流す |

## 10. 依存と順序

- [PRD084](084_noisynet_10prd.md)（NoisyNet）と [PRD101](101_exploration_schedule_profiled_value_10prd.md)（ProfiledValue と spatial のラダー）の実装に乗る。
- D5（skip 側の forward の倍率）は [PRD106](106_temporl_skip_forward_noise_key_10prd.md) の実装と同じ forward に触れる。どちらを先に実装しても、skip 側の forward の呼び出し情報に skip 側の方策の倍率が載るようにする。
- ステージ 1 の比べる相手のスキップなしの 2 seed は、どちらも 50M まで完走している（終了 checkpoint の Greedy の評価は、[PRD104 段 0 の実験記録](../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md) の探索ブロック 01（seed 1）と 03（seed 2）にある）。

## 11. 参照

- [PRD084 NoisyNet](084_noisynet_10prd.md)、[PRD101 探索と学習のスケジュール](101_exploration_schedule_profiled_value_10prd.md)、[PRD106](106_temporl_skip_forward_noise_key_10prd.md)、[PRD104](104_temporl_skip_auxiliary_only_10prd.md)、[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md)
- [ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)
- [130 ニューラルネットワーク（設計文書）](../design/130_neural_networks.jp.md)、[200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md)
- [NoisyNet の観測（wiki）](../wiki/mechanisms/noisynet.md)、[Qbert（wiki）](../wiki/subjects/qbert.md)、[2026-09-25 の実験記録](../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、[2026-10-11 の実験記録](../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)
