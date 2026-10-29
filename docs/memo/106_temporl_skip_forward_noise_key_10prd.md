# TempoRL の skip 側の forward のノイズを Q 側と分けて指定する（`full` でだけノイズが効く仕組みの切り分け）PRD

> 2026-10-29 起票（暫定。質疑を省き、既定を置いた）。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 2026-10-29 再グリル: 暫定版の「確定事項」は会話で合意していなかったので、案として一つずつ確かめ直した。ゴールに G4（ステージ 0 の読みを受けた学習を条件付きで含める）を足した。設定の形は、別モデルの検討で「値を書き写す形」への反対を受け、モード値 `skip.nn_forward_mode = shared｜dedicated` ＋値キー `skip.nn_forward_key` に改めた（D1〜D3）。学習の引き金を「読みの表の行」から「skip 側を切ったときに戻る割合」の目安に変え、流すかはステージ 0 の数値を見てユーザーが決めることにした（D7）。分割は最初から 400 ゲームで評価する（§6 の 6）。用語を「行動の forward」から「Q 側」「skip 側」へ、仮説 A・B を仮説 1・2 へ改めた。受入 1 を、skip 有効と NoisyNet を含む構成で確かめる形に直した。§6 の細部を埋め、§9 の未決を閉じた。
> 起点: qbert の 50M の checkpoint を、評価時の行動の選び方だけを変えて評価した結果（[qbert の行動固定を評価時だけ外す比較](../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02）。分割の checkpoint は、評価時に NoisyNet のノイズや IQN の τ のランダムさを入れると `full`（固定あり）でだけ 8 面突破率が下がり、`single`（毎 step 判断）では下がらなかった。その仕組みを、Q 側と skip 側のノイズ・τ を別々に入れて切り分ける。
> 関連: [PRD098](098_temporl_skip_policy_10prd.md)（TempoRL 対応）、[PRD103](103_temporl_reference_alignment_10prd.md)（第二弾。§8 の T11 が τ の側の先送り）、[PRD104](104_temporl_skip_auxiliary_only_10prd.md)（補助学習のみモード。`single` と `max_length_mode`）、[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)（第三弾。割り込み (I)）、[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)・[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)（NN 実行設定キーと用途別の NN 実行オブジェクト）、[行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[NoisyNet の観測（wiki）](../wiki/mechanisms/noisynet.md)。

## 1. 目的と問題

本書では、TempoRL の Actor が判断の step で動かす 2 つの forward を次のように呼ぶ。

- **Q 側**: 行動価値関数の forward。どの行動を取るかを選ぶ。NN 実行設定は Actor の `nn_forward_key` で決まる。
- **skip 側**: スキップ価値の forward。選んだ行動を何 step 続けるか（スキップ長 j）を選ぶ。

TempoRL の Actor は、判断の step で行動とスキップ長 j を選び、`full` では選んだ行動を j step 続ける。qbert のサンプル効率優先の 50M の分割の checkpoint（seed 1）を、評価時の選び方を学習中の Train Actor に寄せて評価すると、`full` の 8 面突破率が下がった（§2.2 の 1）。同じ checkpoint の `single` は 4 条件とも同じ水準で、スキップなしの checkpoint は逆にノイズで上がった。

学習データを集める Train Actor は、`full`・NoisyNet のノイズあり・τ ランダムで動く。学習中の 8 面突破率が評価より低いことと、学習の中盤の遅れ（[PRD104](104_temporl_skip_auxiliary_only_10prd.md) §2.2）にも、この「`full` でノイズが効く」ことが関わっている可能性がある。仕組みの候補は 2 つある。

- **仮説 1（固定で誤りが長引く）**: Q 側のノイズで Greedy と違う行動を選ぶと、`full` ではその行動を、その行動に対して読んだスキップ長 j のあいだ続ける。毎 step 選び直す `single` やスキップなしでは、誤りは 1 step で終わる。
- **仮説 2（スキップ長の選択にノイズが入る）**: skip の Head にもノイズの重みがあり、skip 側の forward は Q 側と同じ NN 実行設定で動く（§2.1 の 1・2）。スキップ長どうしの価値の差は行動価値の差より小さく（§2.2 の 3）、ノイズで j の選択が入れ替わりやすい。

τ にも同じ 2 つの経路がある（Q 側で行動を選ぶときの τ と、skip 側でスキップ長を選ぶときの τ）。

今は skip 側の NN 実行設定を Q 側と別に指定できないので、仮説 1 と 2 を分けて測れない。

## 2. 現状（2026-10-29 に確認したこと）

### 2.1 実装と設定

1. Actor は Q 側と skip 側に別々の NN 実行オブジェクトを持ち、乱数の seed も別に派生させる。ただし、どちらも同じ NN 実行設定（Actor の `nn_forward_key`）で作る（[dqn_based_agent.cpp:2333-2338](../../core/anet-core/src/dqn_based_agent.cpp)）。
2. skip の Head の射影は、Q 側の Head の射影の設定（`head.projection`。NoisyNet の有無と σ0 を含む）を引き継ぐ（[default_dqn_agent.hpp:411](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。qbert の TempoRL の Run は `head.projection.noisy.enabled = true` なので、skip の Head もノイズの重みを持つ。skip の Body（`SkipFC1 > SkipReLU > SkipFC2 > SkipReLU`）の Linear にノイズは無い（[分割の Run の設定の写し](../experiments/default-dqn/atari/config/run_20261028-215839_a5_qb_aln_t103spl_s1.txt) 270-273・793・920 行）。射影は `skip.head.projection.*` で上書きできるが、`noisy.enabled` はノイズの重みの有無を決める構造の設定で（[nn_test.cpp:4428-4430](../../core/anet-core/src/nn_test.cpp)）、Agent 全体に効き、Learner も同じ skip の Network を使う。
3. skip の入力は、観測、Q 側の forward が公開する特徴量、選んだ行動である（[dqn_based_agent.cpp:2271-2281](../../core/anet-core/src/dqn_based_agent.cpp)）。qbert の構成の公開特徴量は `skip_context = iqn_fusion`（`main_feature * tau_embedding`）で、Q 側の NoisyNet の層（`value_stream`・`adv_stream` の `AtariHeadFC512` と Head の射影）より前にある（同じ写しの 294・606-609・1302 行）。したがって qbert の構成で skip の値にノイズが入る経路は、(a) skip の Head の射影のノイズと、(b) ノイズありの Q で選んだ行動（分離型ではその行動の行の値を読み、`full` ではその行動を続ける）の 2 つである。
4. τ: Actor が skip 専用の方策（`actor.[key].skip.policy.*` の葉）を持ち、Q 側の方策が Greedy 型でないときは、Q 側の forward に skip 用の τ の行を足して問い合わせ、スキップ長はその行だけで選ぶ（[dqn_based_agent.cpp:2425-2438・2521-2523](../../core/anet-core/src/dqn_based_agent.cpp)、[default_dqn_agent.cpp:575-578](../../core/anet-core/src/default_dqn_agent.cpp)、[default_dqn_agent.hpp:486-487](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。専用の方策が無いときは、Q 側の τ の行を共用する（[dqn_based_agent.cpp:2524-2526](../../core/anet-core/src/dqn_based_agent.cpp)）。共用では skip 側は Q 側と同じ τ の抽選を読み、専用の方策は skip 用の乱数で別に引く。ノイズは Q 側と skip 側で NN 実行オブジェクトと seed が別なので、両方にノイズを入れると独立した 2 つの抽選になる（1）。skip 専用の方策の `tau_rule` を `fixed` にすれば、Q 側の τ がランダムでもスキップ長の選択の τ だけを固定できる見込みだが、確かめていない（[PRD103](103_temporl_reference_alignment_10prd.md) §8 の T11）。専用の方策で UQE の楽観の τ を分けるテストはある（[dqn_based_agent_test.cpp:3068-3083](../../core/anet-core/src/dqn_based_agent_test.cpp)）が、`tau_rule.sample_mode` の固定とランダムを分けるテストは無い。専用の方策の τ の行は Q 側の forward に足されるので、Q 側の forward の τ の本数が増える（同テストの `last_taus.size(1) > 8`）。Atari の `[greedy]` の Actor の Q 側の方策は EpsilonGreedy（ε=0）なので、Greedy 型の除外には当たらない（同じ写しの 1308 行）。Atari の Train Actor は skip 専用の方策を持たず、Q 側の τ を共用する（[Atari.txt:276](../../apps/runner/config/Atari.txt) の `run.@skip` に `skip.policy` の葉が無い）。
5. Learner は skip の Network を `learner.current` の NN 実行設定で forward する（[dqn_based_agent.cpp:2660-2661](../../core/anet-core/src/dqn_based_agent.cpp)）。
6. 設定の読み取り: Actor の `nn_forward_key` は常に必須で、宣言されていない名前はエラーになる（[default_dqn_agent.hpp:163・568-581](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。Learner の 2 つのキーも、学習を止めた Run で必須である（同 214-218）。Actor ごとの skip の設定には PRD104 のモード値 `skip.max_length_mode`（`single｜full｜variable`）があり、キーが無ければ C++ 側の既定 `full` になる（[agent.hpp:170](../../core/anet-core/include/anet/agent.hpp)、[config.hpp:185-199](../../core/anet-core/include/anet/config.hpp)）。`agent.txt` も同じ値を `[train]` と `@eval_base` に `?=` で書く（[agent.txt:372・378](../../apps/runner/config/agent.txt)）。値キー `skip.max_length` は `variable` のときだけ必須で、ほかのモードで書かれていれば skip 有効のときに WARN を 1 回出す（[200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md) §6.5）。skip 専用の方策は、葉の有無で Q 側の方策との共用を決める（[default_dqn_agent.hpp:158](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
7. Actor の `nn_forward_key` を上書きする現用の設定・検証資材がある。[12_batch_run.bat:56](../../apps/12_batch_run.bat) の `NZACT`（DropMerge で `actor.[train].nn_forward_key=act_lane`。skip を有効にしない群で使う）。[prd098/run_validation.py:38-44](../../core/anet-core/testdata/prd098/run_validation.py) の DropMerge（同じく `act_lane`。skip を有効にする smoke と長い Run でも使う）と、同 58-69 行の長い Run で手で宣言する `[eval_target_greedy]`（`mu_only`）。[Atari.txt:494・508・522](../../apps/runner/config/Atari.txt) の `run.@nz_learning`・`@nz_reference`・`@nz_reference_off`（既定と同じ `act_sample`）。Atari の既定の連鎖は `run.@skip` を含み（[Atari.txt:40](../../apps/runner/config/Atari.txt)）、skip が有効である。
8. 既定の構成で NoisyNet と skip が両方有効な短い検証の構成は、PRD103 の検証資材の 6 構成（[README](../../core/anet-core/testdata/prd103/README.md)）に無い（skip 有効の GridMaze・LunarLander は NoisyNet 無効）。LunarLander には NoisyNet を有効にする `run.@noisy` がある（[LunarLander.txt:91-95](../../apps/runner/config/LunarLander.txt)）。
9. NN 実行設定 `act_sample` は `noisy.hold = call`（[agent.txt:133](../../apps/runner/config/agent.txt)）で、呼び出しの時計が進むたびにノイズを引き直す。`full` の Actor は、継続中の lane では提案された行動を前の行動で上書きする（[dqn_based_agent.cpp:2509](../../core/anet-core/src/dqn_based_agent.cpp)）。

### 2.2 実測

1. qbert のサンプル効率優先の 50M の終了 checkpoint（seed 1）を、`[greedy]` の評価 session で各 200 ゲーム評価した 8 面突破率（[実験記録](../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02）。「両方」が ε=0 の Train Actor と同じ選び方である。

   | checkpoint・動かし方 | Greedy（ノイズなし・τ 固定） | ノイズ | τ ランダム | 両方 |
   |---|---:|---:|---:|---:|
   | 分割・`full` | 66.0% | 41.5% | 50.5% | 37.5% |
   | 分割・`single` | 68.5% | 66.5% | 70.0% | 66.5% |
   | スキップなし | 50.5% | 61.5% | 66.0% | 60.0% |

2. 分割の `full` の判断率（0.186〜0.191）と平均スキップ長（5.24〜5.39）は、4 条件でほぼ同じだった（同じ記録、探索ブロック02）。ノイズは平均の長さを変えず、どの場面でどの j になるかを変えている可能性がある（確かめていない）。
3. 学習 Run の 45〜50M（各メトリクスの最後の 1 割の点の平均）: 分割（`run_20261028-215839_a5_qb_aln_t103spl_s1`）では、Learner の batch での行動価値の最大と 2 番目の差 `37_agent_qtd/17_q_gap` が 0.619、Train Actor の判断の lane でのスキップ価値の最大と 2 番目の差 `50_agent_skip/04_skip_q_gap` が 0.174 だった。両者は測る母集団が違う（Learner の batch と Actor の判断）ので、大きさの比べは目安にとどまる。Learner の probe でノイズが Q 側の選ぶ行動を変える割合 `47_agent_noisy/10_action_disagreement_ratio` は、分割 0.038、スキップなし（`run_20261025-231207_a5_qb_aln_btreps_s1`）0.019 だった。skip 側でノイズが j を変える割合は測っていない。
4. 合算の checkpoint（`run_20261027-004945_a5_qb_aln_t98mrg_s1`）は、Greedy の評価で `full` 71.5%、`single` 97.0% だった（同じ記録、探索ブロック01）。ノイズや τ のランダムさを入れた評価はしていない。
5. 中盤の遅れ（ε=0 の評価が 30M 以降に無効より遅れ、学習側の 8 面突破率が 50% を超える 2.5M の窓が 50M まで現れない）は、遮断・合算・重み 0.3 の 3 本で記録されている（[PRD104](104_temporl_skip_auxiliary_only_10prd.md) §2.2）。分割の学習 Run の経過は記録に無い。
6. 探索ブロック01 では、分割の評価 Run（2 session、計 400 ゲーム）の実時間が 7.5 分だった（起動・読込・終了を含む。同じ記録、探索ブロック01 の表）。これから、1 条件（200 ゲーム）の評価 Run は約 5 分（起動・読込・終了の分を含めて切り上げた）と見積もる。分割の分母（Greedy と両方の差）は、ノイズで 24.5 ポイント（66.0 − 41.5。ブロック02 の「ノイズ」）、τ で 15.5 ポイント（66.0 − 50.5）だった。

## 3. ゴール・非ゴール・仮説と成功の測り方

**ゴール**

- G1: Actor ごとに、skip 側の forward の NN 実行設定を Q 側と分けて指定できる。既定の Run は今と同一。
- G2: τ の側は新しい設定を作らず、既存の skip 専用の方策でスキップ長の選択の τ だけを固定・ランダムにできることを確かめる（PRD103 T11 の「未確認」を閉じる）。
- G3: 評価だけのステージ 0 で、`full` の 8 面突破率の低下が仮説 1 と 2 のどちらから来るか（τ の 2 経路も同じく）を読む。
- G4: ステージ 0 の数値を見て、Train Actor の skip 側のノイズ（スキップ長の τ）を切った学習で、中盤の遅れが変わるかを見る（流すかはユーザーが決める。D7）。

**非ゴール**

- Learner の skip の forward の NN 実行設定を分けること（§8）。消える案は Learner 側のキーを足す案で、本当の制約ではない（クリーンブレークで足せる）。問いが Actor の選択に入るノイズなので範囲から外す。
- 既定の変更（学習の Train Actor の skip 側のノイズを既定で切るなど）。消える案はその既定の変更で、本当の制約ではないが、G4 の結果が無いと決められない。
- skip 側のノイズで j が入れ替わる割合などの新しい診断指標（§8）。消える案は noisy probe の skip 版を足す案で、本当の制約ではない（足せる）が、ステージ 0 の評価で切り分けられる見込みなので、読みが「決めきれない」になるまで先送りする。
- 行動の固定の仕方そのものの変更（`single` は PRD104、割り込み (I) は 999 の第三弾）。消える案は Train Actor を `single` にする・割り込みを入れる案で、本当の制約ではないが、それぞれの PRD の関心なので本書では扱わない。
- スカラーのスキップ価値（PRD098 §8 の先送りのまま）。消える案はスキップ価値をスカラーで持つ案で、本当の制約ではないが、PRD098 §8 の再開条件がまだ揃っていない。

**仮説・比べ方・終わり方**

仮説は §1 の 1 と 2（τ の 2 経路を含む）。比べ方はステージ 0（§7）で、同じ checkpoint を、Q 側と skip 側のノイズ・τ を別々に入れた条件で評価し、8 面突破率の下がり方を比べる。ステージ 0 は、2 本の checkpoint で読み（§7 の読みの表の行、または「再現しない」「決めきれない」）と戻る割合（D7）を記録したところで終える。そのあと、ユーザーが数値を見て、ステージ 1 と τ の学習を流すかを決める。流した群は、PRD104 の段 2 と同じ読む値と見込みありの判定を記録したところで終える。流さないと決めたら、その判断と数値を記録して終える。

**成功の測り方**

- G1・G2: §7 の受入 1〜4 がすべて通る。
- G3: 2 本の checkpoint について §7 の h と k が計算でき、ノイズの読みと τ の読みの表の行（または「再現しない」「決めきれない」）が実験記録に残る。
- G4: 流した群の 6 つの読む値と見込みありの判定（§7）が実験記録に残る。流さない場合は、その判断と、判断に使った戻る割合が記録に残る。

## 4. 確定事項

- **D1. Actor ごとに、モード値 `actor.[key].skip.nn_forward_mode`（`shared｜dedicated`）と値キー `actor.[key].skip.nn_forward_key` を足す。`shared` では skip 側の forward を Q 側と同じ NN 実行設定（その Actor の `nn_forward_key`）で動かし、`dedicated` では `skip.nn_forward_key` の NN 実行設定で動かす。値キーは `dedicated` のときだけ読み、値は `nn_forward.[<name>]` で宣言済みの名前。** 2026-10-29 にユーザーが選んだ（別モデルの検討が、暫定版と再グリル前半の「Q 側と同じ値を各 Actor に書き写す形」に反対した）。理由: 「Q 側に従う／指定した設定を使う」という意図の分岐をモード値で明示する（[AGENTS.md](../../AGENTS.md) §設定値の扱いの「モード値として明示させる」）。`shared` なら Q 側の `nn_forward_key` を上書きしても skip 側が追従するので、2 つの葉を手で揃え続ける必要が無い（§2.1 の 7 の上書きを直さずに済む）。`dedicated` の値は、NN 実行設定をカタログ（`nn_forward`）から参照キーで選ぶ既存の形（[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[CONTEXT.md](../../CONTEXT.md) の「NN 実行設定」）で、skip 側はすでに独立の NN 実行オブジェクトと seed を持つ（§2.1 の 1）。モード値＋値キーは PRD104 の `skip.max_length_mode`＋`skip.max_length` と同じ形で、置き場も同じ `actor.[key].skip.*`。Actor ごとにするのは G1 の要求で、評価の session ごとに条件を変える（ステージ 0）ことと、Train Actor だけを変える（G4）ことの両方に要る。退けた案: Q 側と同じ値を各 Actor に書き写す形（Q 側だけ上書きすると skip 側が黙って既定で動く）。キーが無ければ Q 側と共用する形（未指定に意味を持たせる。既存の `skip.policy` はこの形だが、本書は揃えない。`skip.policy` をモード値へ移すのは本当の制約ではなくクリーンブレークでできるが、skip 専用の方策を使う今の設定（GridMaze の `run.@skip` など）の移行が要り、本書の問いには要らないので範囲外にする。§8）。モード値でノイズの有無そのものを選ぶ形（`shared｜off`。カタログの外に 2 つ目のノイズの指定を作る）。既存の `skip.head.projection.noisy.enabled = false`（構造の設定で、Agent 全体と Learner の skip まで変わり、学習済みの checkpoint を Actor ごとの条件で評価できない。§2.1 の 2。本当の制約）。
- **D2. モード値の既定は `shared` とし、C++ 側の既定と `agent.txt` の `?=` の両方に置く。** `agent.txt` は `skip.max_length_mode` と同じく `[train]` と `@eval_base` に書く（§6 の 1）。`$` で派生する Actor（`[eval]`・`[eval_target]`、Atari の `[greedy]`、`[eval_panel]` など）は派生元から受け取る（[agent.txt:373-374・430](../../apps/runner/config/agent.txt)、[Atari.txt:1280](../../apps/runner/config/Atari.txt)）。理由: 既定の Run を今と同一にする（G1。`shared` は今と同じ設定・同じ seed で skip 側を作る）。`max_length_mode`（C++ 既定 `full`、`agent.txt` にも `?=`。§2.1 の 6）と同じ扱いにし、手で組む設定（テスト・検証資材）を直さずに済ませる。キーが無いときの結果は、D1 で退けた「キーが無ければ Q 側と共用する形」と同じになる。違いは、Q 側に従うことを `shared` と書いて明示でき、`dedicated` との分岐がモード値として設定の写しに残ることで、空欄に別の意味を読ませる形にはならない（[AGENTS.md](../../AGENTS.md) §設定値の扱いの「意図の分岐はモード値として明示させる」）。依存: D1。
- **D3. 検証と必須:** モード値は skip の有効・無効によらず `shared｜dedicated` 以外ならエラー。`skip.nn_forward_key` は、書かれていれば、モードと skip の有効・無効によらず宣言済みの名前かを検証する。`skip.enabled = true` で `dedicated` の Actor にキーが無ければ、構築時にエラーにする（該当する Actor をすべて示す）。`skip.enabled = true` で `shared` の Actor にキーが書かれていれば、使わないことを Actor ごとに 1 回 WARN する。`skip.enabled = false` では、必須の検査も WARN もしない。理由: 2026-10-29 にユーザーが選んだ「skip 有効のときだけ必須」（skip を使わない構成に使わない設定を書かせない。[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の「意図された休止・無効化」の行）を、D1 の形に当てはめた。必須と WARN は PRD104 の `skip.max_length`（`variable` のときだけ skip 有効で必須、ほかのモードで書かれていれば skip 有効で WARN 1 回、skip 無効では必須と WARN の検査をしない。§2.1 の 6）と同じ規則にする。書かれた値そのものの検証（モード値の列挙、キーが宣言済みの名前か）は、`max_length` がモードによらず値の下限を検証するのと同じく、skip の有効・無効によらず行う。書かれた参照が未宣言の対象を指すのは意図しない状態だからである（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の「未宣言対象への参照」の行）。依存: D1・D2（既定ではキーを書かないので、WARN が既定の Run で出ることはない）。
- **D4. τ の側には新しい設定を作らない。** 既存の skip 専用の方策（`actor.[key].skip.policy.tau_rule`）で、スキップ長の選択の τ を Q 側と分ける（§2.1 の 4）。経路が働くことを受入 3 で確かめ、通らなければ、経路の不具合として本書の実装の中で直す（G2 と、τ も読む G3 の前提のため）。Q 側の方策が Greedy 型の Actor で skip 専用の方策を使わない今の規則（[default_dqn_agent.cpp:575](../../core/anet-core/src/default_dqn_agent.cpp)）は変えない。消える案はこの規則を変えて Greedy 型の `[eval_panel]` でも τ を分ける案で、本当の制約ではないが、ステージ 0 の `[greedy]` には要らない。理由: τ の分離は既にある経路で表せる（G2）。依存: Atari の `[greedy]` が EpsilonGreedy であること（§2.1 の 4）。
- **D5. Learner の skip の forward（`learner.current` の設定、§2.1 の 5）と、学習の既定は変えない。** 理由: どちらも合意した非ゴール。問いは Actor の選択に入るノイズで、Learner のノイズは NoisyNet の学習の一部である（§8 の先送り）。
- **D6. 切り分けは評価だけのステージ 0 で行う。checkpoint は分割と合算の 2 本、条件は §7 の 7 つ。** 2 本は 2026-10-29 にユーザーが選んだ。理由: 分割はノイズの害が分かっている（§2.2 の 1）。合算は G4 の学習の土台（D8）で、ノイズありの評価をまだしていない（§2.2 の 4）。遮断と重み 0.3 は、2 本の読みが分かれたときに足せる。条件はノイズ（τ）を「入れない／Q 側だけ／skip 側だけ／両方」に分けた組で、両方は探索ブロック02 の「ノイズ」「τ ランダム」と同じ設定であり、Train Actor の選び方に当たる（§2.1 の 4）。評価の量は §6 の 6（分割は最初から 400 ゲーム）。依存: D8。
- **D7. G4 の学習（ステージ 1 と τ の学習）は、ステージ 0 の数値を見てユーザーが流すかを決める。目安は「skip 側を切ったときに戻る割合」1 − h_Q（τ は 1 − k_Q）が 0.5 以上。** G4 を含めることと、数値を見て決めることは、2026-10-29 にユーザーが選んだ。どちらの checkpoint の値を重く見るかも、数値を見て決める。理由: ステージ 1 では Train Actor の選び方が「両方ノイズ（NB）」から「Q 側だけノイズ（NQ）」に変わるので、学習で効きうる量は r_NQ と r_NB の差である。退けた案: 読みの表で仮説 2 の行のときだけ流す形は、「2 つがそろったときに下がる」行（skip 側を切れば NQ まで戻る場合）を落とす。流すと決めたときは、段取りを示して承認を得てから流す。依存: D6・D8。
- **D8. 学習の群の土台は、合算の構成（§2.2 の 4 の Run と同じ構成）とする。読む値と見込みありの判定は PRD104 の段 2 と同じにする。** 理由: G4 は「中盤の遅れが変わるか」を見るもので、遅れは合算で記録されている（§2.2 の 5）。比べる相手と判定の基準値（無効の 8,207、37.5M）が、PRD104 で合算と無効について決まっている（[PRD104](104_temporl_skip_auxiliary_only_10prd.md) §6 の 13・§7 の段 2）。

## 5. 実装契約

確定事項と本節には実装の語（内部の型、入れ物の型、添字の決め方）を書かない。公開 API 名・設定キー・メトリクス名・クラス一覧の名前は書く。

### 5.1 設定キー

`DefaultDQNAgent.` の下。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `actor.[key].skip.nn_forward_mode` | `shared｜dedicated` | `shared`（C++ 側の既定と `agent.txt` の `?=`。D2） | `shared`: skip 側の forward をその Actor の `nn_forward_key` の NN 実行設定で動かす。`dedicated`: `skip.nn_forward_key` の NN 実行設定で動かす |
| `actor.[key].skip.nn_forward_key` | 文字列。`nn_forward.[<name>]` で宣言済みの `<name>` | なし（`dedicated` のときだけ読む） | `dedicated` の skip 側の forward が使う NN 実行設定 |

- モード値が `shared｜dedicated` 以外なら、skip の有効・無効によらず、構築時にキーと値と期待値を示して `ANET_SYSTEM_ERROR` で止める。
- `skip.nn_forward_key` が書かれていて、宣言されていない名前なら、モードと skip の有効・無効によらず、構築時にキーと値を示して止める（`nn_forward_key` と同じ扱い。[default_dqn_agent.hpp:576-579](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
- `skip.enabled = true` で、`dedicated` なのに `skip.nn_forward_key` が無い Actor があれば、構築時に該当する Actor をすべて示して止める（D3）。
- `skip.enabled = true` で、`shared` なのに `skip.nn_forward_key` が書かれている Actor があれば、使わないことを Actor ごとに 1 回 WARN する（D3）。
- 値を丸めたり既定へ置き換えたりしない。

設定例（コロン記法）:

```
# agent.txt: 既定（skip.max_length_mode の隣。§6 の 1）。既定では今と同一
DefaultDQNAgent.@baseline : actor.[train].skip.nn_forward_mode ?= shared
DefaultDQNAgent.actor.@eval_base : skip.nn_forward_mode ?= shared

# ステージ 0 の NQ: 評価の Actor で、Q 側だけノイズあり（skip 側はノイズなし）
DefaultDQNAgent.actor.[greedy].nn_forward_key=act_sample DefaultDQNAgent.actor.[greedy].skip.nn_forward_mode=dedicated DefaultDQNAgent.actor.[greedy].skip.nn_forward_key=mu_only

# ステージ 0 の TS: スキップ長の選択の τ だけをランダムにする（既存の skip 専用の方策。D4）
DefaultDQNAgent.actor.[greedy].skip.policy.tau_rule.sample_mode=random

# G4 の学習（ユーザーが流すと決めたとき）: Train Actor の skip 側だけノイズを切る
DefaultDQNAgent.actor.[train].skip.nn_forward_mode=dedicated DefaultDQNAgent.actor.[train].skip.nn_forward_key=mu_only
```

### 5.2 判定の網羅

| `skip.enabled` | `skip.nn_forward_mode` | `skip.nn_forward_key` | 答え |
|---|---|---|---|
| どちらでも | `shared｜dedicated` 以外 | — | エラー |
| どちらでも | どちらでも | 書かれていて未宣言 | エラー |
| false | `shared` か `dedicated` | 無い、または宣言済み | 構築できる。使わない（WARN なし） |
| true | `shared` | 無い | skip 側の forward を Q 側と同じ設定で作る（今と同一） |
| true | `shared` | 宣言済み | Q 側と同じ設定で作る。キーは使わず、Actor ごとに 1 回 WARN |
| true | `dedicated` | 無い（どれか 1 つの Actor でも） | エラー（該当する Actor をすべて示す） |
| true | `dedicated` | 宣言済み | その設定で skip 側の forward を作る。Q 側の `nn_forward_key` と同じ値なら今と同一 |

### 5.3 Actor

- skip 側の forward の NN 実行オブジェクトを、`shared` ならその Actor の `nn_forward_key` の NN 実行設定で、`dedicated` なら `skip.nn_forward_key` の NN 実行設定で作る。seed の派生（skip 用の名前付き seed）、時計の種別、forward の回数と順序は変えない。
- Q 側の forward は今までどおり `nn_forward_key` で作る。
- `shared` と、`dedicated` で `skip.nn_forward_key` が `nn_forward_key` と同じ値のときは、今と同一の挙動になる（同じ設定・同じ seed）。
- `single` でも skip 側の forward は今までどおり行う（PRD104 D2）。`single` ではスキップ長を選ばないので、skip 側の設定は選ぶ行動に効かない。

### 5.4 移行対象と等価性

- 設定: `apps/runner/config/agent.txt` に D2 の 2 行（§6 の 1）。Env の設定（`Atari.txt`・`LunarLander.txt`・`GridMaze.txt`・`DropMerge.txt`）は、`agent.txt` の既定と `$` の派生で受け取るので変えない。
- Actor の `nn_forward_key` を上書きする設定・検証資材（§2.1 の 7）は、既定の `shared` で skip 側が追従するので直さない。
- テスト: 設定を手で組むテストは、モード値が無くても C++ 側の既定 `shared` で今と同じに動くので直さない。skip の資源（`SkipActorResources`）を直接組んで Actor を作るテストは、skip 側の NN 実行設定を渡す形に合わせる。
- 設定の移行を検査する前の PRD の資材は直さない。[prd098/verify_config_migration.py](../../core/anet-core/testdata/prd098/verify_config_migration.py) と [prd103/verify_config_migration.py](../../core/anet-core/testdata/prd103/verify_config_migration.py) は、そのときに保存した解決結果どうしを比べるので、本書の変更に影響されない。[prd104/verify_config.py](../../core/anet-core/testdata/prd104/verify_config.py) は、PRD104 の着手前の設定と今の設定を解決し直し、PRD104 で増えた 2 つの葉（`learner.skip.target_window`・`skip.max_length_mode`）だけを許して比べる（14・32-35・40-41 行）。本書のあとで回すと D2 の葉を想定外の差として止まるが、これは PRD104 の受入の時点の確かめなので、許す葉を足さない（PRD103 も前の PRD の凍結した比較を直さず、自分の資材で受入を確かめた。[PRD103](103_temporl_reference_alignment_10prd.md) §5.7）。本書の設定の差は受入 4 で同じ方法で確かめる。
- 文書: [200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md) の 2 か所を直す（英訳は翻訳の工程が追う）。§7.3「DefaultDQN の Noisy 実行設定」は、参照キーの並びに Actor の skip 側の `skip.nn_forward_mode`・`skip.nn_forward_key` を足し、「欠落・未知参照を設定読み込み時に拒否する」の文に D3 の扱い（`dedicated` で skip 有効なら欠落を拒否し、未知参照は常に拒否する）を書く。§6.5「DefaultDQN の skip 方策」の Actor の段落には、skip 側の forward が `shared` では Q 側と同じ NN 実行設定で、`dedicated` では `skip.nn_forward_key` の NN 実行設定で動くことを書く。[CONTEXT.md](../../CONTEXT.md) の「NN 実行設定」は本書の起票と再グリルで直した。
- 互換の仕組みは残さない（`shared` はモード値として明示されたもので、互換の読み替えではない）。
- 変えないもの: 過去の Run の artifact と設定の写し、PRD103・PRD104 の検証資材が凍結した設定の写し（`baseline-config`・`current-config` など。凍結した実行体と組で使う。§10）、実験記録。

### 5.5 性能

forward の回数・順序を変えないので、性能の受入は置かない。`mu_only` の skip 側の forward はノイズを引かない分だけ軽くなりうるが、測らない。G4 の τ の群（§6 の 8）は、Train Actor に skip 専用の方策を持たせるので、Q 側の forward の τ の本数が増える（§2.1 の 4）。これは既存の経路の挙動で本書の変更ではなく、その群の実時間は比べに使わない。

### 5.6 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `DQNActorConfig` | 変更 | Actor ごとの skip 側の NN 実行設定のモード値とキーを持つ | Agent 共通の設定（agent.hpp） |
| `DefaultDQNAgentConfig` | 変更 | `skip.nn_forward_mode`・`skip.nn_forward_key` の読み取りと、D3 の検証・WARN | DQN 具象（default_dqn_agent） |
| `DefaultDQNAgent` | 変更 | Actor の生成時に、モード値に従って skip 側の NN 実行設定を選んで渡す | DQN 具象（default_dqn_agent） |
| `SkipActorResources` | 変更 | skip 側の NN 実行設定を持つ | DQN 共通 component（dqn_based_agent） |
| `Actor`（DQN） | 変更 | skip 側の NN 実行オブジェクトを、skip 用に渡された NN 実行設定で作る | DQN 共通 component（dqn_based_agent） |

## 6. 細部

2026-10-29 の再グリルで決めた。1〜3 は D1〜D3 の形に合わせて、6 は分割の評価量の選択を受けて直した。

1. 既定の置き場: `agent.txt` の `skip.max_length_mode` の 2 行の隣（`[train]` と `@eval_base`）（理由: 同じ Actor の skip の設定を並べて読める。派生の仕方も `max_length_mode` と同じになる）。
2. WARN: skip 有効で `shared` の Actor に `skip.nn_forward_key` が書かれていれば、Actor ごとに 1 回出す。skip 無効では出さない（理由: PRD104 の `max_length` と同じ規則。既定ではキーを書かないので、既定の Run では出ない）。
3. エラーと WARN の文言（`nn_forward_key` と `max_length` の既存の文言にそろえる）:
   - モード値が不正: `Invalid DefaultDQNAgent.actor.[<key>].skip.nn_forward_mode='<値>'; expected shared|dedicated.`
   - キーが無い: `Missing DefaultDQNAgent.actor.[<key>].skip.nn_forward_key; expected a declared nn_forward key when skip.nn_forward_mode=dedicated and skip.enabled=true.`（該当する Actor をすべて並べる）
   - 宣言されていない名前: `Invalid DefaultDQNAgent.actor.[<key>].skip.nn_forward_key='<値>'; expected a declared nn_forward key.`
   - 使わないキー: `DefaultDQNAgent.actor.[<key>].skip.nn_forward_key='<値>' is unused with skip.nn_forward_mode=shared; expected skip.nn_forward_mode=dedicated to use this value.`
4. ステージ 0 の評価は、探索ブロック02と同じ形（評価専用 Run、1 Run 1 条件、`[greedy]` の session、1 Run 200 ゲーム）。評価の seed は 1（分割は 1 と 2。§6 の 6）（理由: ブロック02の値と並べられる）。
5. ステージ 0 の読みは 8 面突破率（29,000 点以上かつ `ram_metric.[1] >= 8`）で行い、平均点・判断率・平均スキップ長も並べる（理由: ブロック02で差がはっきり出たのは突破率で、平均点の差は小さかった）。
6. h と k の読みのしきい値は 0.5。分割は最初から 7 条件すべてを評価の seed 1 と 2 で流し、400 ゲームで計算する（2026-10-29 にユーザーが選んだ。理由: 分母がノイズ 24.5・τ 15.5 ポイントと分かっていて（§2.2 の 6）、200 ゲームの突破率の差の標準誤差約 5 ポイントでは h の揺れが ±0.2、k の揺れが ±0.3 ほどあり、揺れる帯にほぼ確実に入るので、足す往復を省く）。合算は 200 ゲームで始める。分母（G と両方の差）が 10 ポイント未満なら、その checkpoint では h・k を作らない。合算の h か k が 0.3〜0.7 に入ったら、その 4 条件（ノイズなら G・NQ・NS・NB）を評価の seed 2 で 200 ゲームずつ足し、400 ゲームで計算し直す。400 ゲームの h か k が 0.3〜0.7 に入ったら、分割（最初から 400 ゲーム）も合算（足して 400 ゲーム）も、その読み（ノイズか τ）を「決めきれない」と記録し、読みの表の行は付けない。h・k と突破率の値はそのまま記録する（D7 の判断に使う）。
7. τ の 3 条件（TQ・TS・TB）は新しいキーを使わないので、実装の前のビルドでも流せる。その場合は同じビルドで G も流し、k はその G から計算する。h は実装のあとのビルドの G・NQ・NS・NB だけで計算し、ビルドをまたいで混ぜない（理由: 待たずに半分を進められ、ビルドの違いを読みに混ぜない）。
8. G4 の学習の群: 合算の構成に、ノイズは `DefaultDQNAgent.actor.[train].skip.nn_forward_mode=dedicated` と `DefaultDQNAgent.actor.[train].skip.nn_forward_key=mu_only`、τ は `DefaultDQNAgent.actor.[train].skip.policy.tau_rule.sample_mode=fixed` を足す。qbert、サンプル効率優先、seed 1、50M。両方流すときは別々の群にする（理由: どちらが効いたかを分けて読める。1 seed なので合否ではない）。τ の群は Train Actor に skip 専用の方策を持たせるので、Q 側の forward の τ の本数が増える（§5.5）。
9. ステージ 0 は、PRD104 のステージ 1・2 と同じ GPU で並べて流してよい。実時間の比べには使わない（理由: どちらも性能の受入を持たない）。
10. 仮説 1 が主因だったときの学習側の手（PRD104 の `single` か、999 の割り込み (I)）は本書では決めず、読みを記録して各 PRD の判断に渡す（理由: 行動の固定の仕方を変えるのは非ゴール）。

## 7. 検証と受入

### ステージ 0（評価だけ。実装のあと）

checkpoint は分割（`run_20261028-215839_a5_qb_aln_t103spl_s1`）と合算（`run_20261027-004945_a5_qb_aln_t98mrg_s1`）の 2 本で、どちらも `full`。各 checkpoint で次の 7 条件を流す（葉はすべて `DefaultDQNAgent.actor.[greedy].` の下。「既定」は葉を書かず、`skip.nn_forward_mode` は既定の `shared`、skip の τ は Q 側と共用する）。評価の量は §6 の 6（分割は 400 ゲーム、合算は 200 ゲームから）。

| 条件 | `nn_forward_key`（Q 側） | `skip.nn_forward_mode` | `skip.nn_forward_key` | `policy.tau_rule.sample_mode` | `skip.policy.tau_rule.sample_mode` | 見るもの |
|---|---|---|---|---|---|---|
| G | `mu_only` | 既定 | 既定 | `fixed` | 既定 | 基準（ブロック02の Greedy と同じ） |
| NQ | `act_sample` | `dedicated` | `mu_only` | `fixed` | 既定 | 仮説 1（Q 側のノイズだけ） |
| NS | `mu_only` | `dedicated` | `act_sample` | `fixed` | 既定 | 仮説 2（skip 側のノイズだけ） |
| NB | `act_sample` | 既定 | 既定 | `fixed` | 既定 | 両方（ブロック02の「ノイズ」と同じ。Train Actor の選び方） |
| TQ | `mu_only` | 既定 | 既定 | `random` | `fixed` | τ の仮説 1（Q 側の τ だけランダム） |
| TS | `mu_only` | 既定 | 既定 | `fixed` | `random` | τ の仮説 2（スキップ長の τ だけランダム） |
| TB | `mu_only` | 既定 | 既定 | `random` | 既定 | 両方（ブロック02の「τ ランダム」と同じ。Train Actor の選び方） |

8 面突破率を r と書き、checkpoint ごとに次を計算する。

- h_Q = (r_G − r_NQ) ÷ (r_G − r_NB)、h_S = (r_G − r_NS) ÷ (r_G − r_NB)
- k_Q = (r_G − r_TQ) ÷ (r_G − r_TB)、k_S = (r_G − r_TS) ÷ (r_G − r_TB)
- skip 側を切ったときに戻る割合: 1 − h_Q = (r_NQ − r_NB) ÷ (r_G − r_NB)、τ は 1 − k_Q = (r_TQ − r_TB) ÷ (r_G − r_TB)

ノイズの読み（h で決める。G3 の記録）:

| h_Q | h_S | 読み |
|---|---|---|
| 0.5 以上 | 0.5 未満 | Q 側のノイズで選んだ行動を固定で続けることが主因（仮説 1） |
| 0.5 未満 | 0.5 以上 | スキップ長の選択に入るノイズが主因（仮説 2） |
| 0.5 以上 | 0.5 以上 | どちらの経路でも下がる |
| 0.5 未満 | 0.5 未満 | 2 つがそろったときに下がる（単独では小さい） |

τ の読み（k で決める。G3 の記録）:

| k_Q | k_S | 読み |
|---|---|---|
| 0.5 以上 | 0.5 未満 | Q 側の τ の揺らぎを固定で続けることが主因 |
| 0.5 未満 | 0.5 以上 | スキップ長の選択に入る τ の揺らぎが主因 |
| 0.5 以上 | 0.5 以上 | どちらの経路でも下がる |
| 0.5 未満 | 0.5 未満 | 2 つがそろったときに下がる（単独では小さい） |

τ の「両方」（TB）は、Q 側と skip 側が同じ τ の抽選を読む（§2.1 の 4）。ノイズの「両方」（NB）は独立した 2 つの抽選で、TS の skip 側も Q 側とは別に引く。そのため、τ の読みの「2 つがそろったときに下がる」行には、同じ抽選を共有することの効き方が混ざりうる。k の読みでは、この違いを書き添える。

分母が 10 ポイント未満のときは「この checkpoint ではノイズ（τ）の害が再現しない」と記録し、h（k）を作らない。h か k が 0.3〜0.7 に入ったときは §6 の 6 による（合算は 200 ゲームなら seed 2 を足して計算し直す。400 ゲームでも入れば、分割も合算も「決めきれない」と記録して値を残す）。2 本の checkpoint で読みが分かれたら、両方をそのまま記録する。分割の G と NB は、ブロック02の 66.0%・41.5% と並べ、ビルドの違いで値がずれていないかも見る。τ の 3 条件を実装の前のビルドで流すときは §6 の 7 による。実装の前の結果は、受入 3（経路の確認）の前の結果として記録する。

ステージ 0 を終えたら、2 本の checkpoint の 7 条件の突破率、h・k、戻る割合（1 − h_Q、1 − k_Q）を並べてユーザーに示す。ユーザーが G4 の学習を流すか、どちらの checkpoint の値を重く見るかを決める（D7。目安は戻る割合が 0.5 以上）。あわせて、G4 の群の候補として、構造の設定 `DefaultDQNAgent.skip.head.projection.noisy.enabled=false` で skip の Head のノイズを Actor と Learner の両方から消す群もあることを示す（新しく学習するので使える。Learner の skip のノイズも消える点が §6 の 8 の群と違う）。

### G4 の学習（ユーザーが流すと決めたときだけ）

群は §6 の 8 のとおり（合算の構成に、ノイズは Train Actor の `skip.nn_forward_mode=dedicated`・`skip.nn_forward_key=mu_only`、τ は `skip.policy.tau_rule.sample_mode=fixed`。qbert、サンプル効率優先、seed 1、50M）。ユーザーがステージ 0 のあとで選んだ群を足すこともある。段取りを示して承認を得てから流す。比べる相手は合算（`run_20261027-004945_a5_qb_aln_t98mrg_s1`）とスキップなし（`run_20261025-231207_a5_qb_aln_btreps_s1`）で、どちらもビルドが違う。1 seed なので合否ではない。

読む値（[PRD104](104_temporl_skip_auxiliary_only_10prd.md) §7 の段 2 の「読む値」と同じ 6 つ。ここに全部書く）:

1. 最初の 10M の学習側の平均点（PRD104 §2.2 と同じ 3 窓）
2. 5.05M の ε=0 の評価
3. 学習側の 8 面突破率が初めて 50% を超える 2.5M の窓
4. 30〜45M の ε=0 の評価
5. Munchausen の log π の clip 率（無効と同じ帯に戻るか）
6. 眠ったユニットの率

見込みありの判定（PRD104 §7 の段 2 と同じ 2 条件。両方を満たすとき）: 最初の 10M の学習側の平均点が無効（8,207）以上で、かつ学習側の 8 面突破率が初めて 50% を超える 2.5M の窓の始まりが 37.5M 以前（37.5〜40M の窓まで含む）。

### 受入（外から観測できるもの）

1. 既定の設定で今と同一: 本書の前のビルドと設定一式を着手前に snapshot し、PRD103 の検証資材と同じ短縮条件と比べ方（[README](../../core/anet-core/testdata/prd103/README.md)。8 lane、seed 1、8,704 experience、FP32・決定論的 backend、評価停止。`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar 点の完全一致）で、次の 7 構成の metrics checksum が前後のビルドで一致する。前後のビルドへは、snapshot の同じ設定を渡す（新しいビルドではモード値が無く、C++ 側の既定 `shared` で動く）。
   - PRD103 の 6 構成: skip 無効の GridMaze・LunarLander・Atari BTR・DropMerge nzrep、skip 有効の GridMaze・LunarLander。
   - skip 有効の LunarLander に `run.@noisy` を足した構成（skip の Head がノイズの重みを持ち、skip 側の forward が `act_sample` で乱数を引く。§2.1 の 8）。
2. 単体テスト:
   - モード値が `shared｜dedicated` 以外なら、skip の有効・無効によらず、構築時にキーと値を含むエラーになる。
   - `skip.nn_forward_key` に宣言されていない名前が書かれていれば、モードと skip の有効・無効によらずエラーになる。
   - `skip.enabled = true` で、`dedicated` なのにキーの無い Actor があると、構築時に該当する Actor のキー名をすべて含むエラーになる。`skip.enabled = false` なら同じ設定で構築できる。
   - `skip.enabled = true` で、`shared` の Actor にキーが書かれていると、構築でき、その Actor について WARN が 1 回出る。`skip.enabled = false` なら出ない。
   - skip 側の forward の設定の選び方: テストの構成は、Q 側と skip の Head の射影の NoisyNet を有効（`head.projection.noisy.enabled = true`）にする。Actor の skip 側の forward を、同じ入力（同じ観測・同じ Q 側の公開特徴量・同じ行動。行動選択を通さず、入力を固定して渡す）で、呼び出しの時計を進めて 2 回呼ぶ。`shared` では、Q 側の `nn_forward_key` が `mu_only` なら 2 回の skip の値が一致し、`act_sample` なら一致しない（Q 側を上書きすると追従する）。`dedicated` では、Q 側の `nn_forward_key` によらず、`skip.nn_forward_key = mu_only` なら一致し、`act_sample` なら一致しない。
   - Q 側の forward の設定の選び方: 同じ構成で、τ の規則を `fixed` にし、Q 側の forward を同じ観測で、呼び出しの時計を進めて 2 回呼ぶ。`nn_forward_key = mu_only` なら 2 回の行動価値が一致し、`act_sample` なら一致しない。これが skip 側のモード値とキーによらず成り立つ。
3. τ の経路（PRD103 T11 の確認）: テストの構成は IQN で、`skip_context` を τ に依る出力（qbert の構成と同じく、τ の埋め込みと掛け合わせた出力）に結び、ノイズは Q 側・skip 側とも `mu_only` にする。Q 側の方策が EpsilonGreedy（ε=0）で `policy.tau_rule.sample_mode = random`、`skip.policy.tau_rule.sample_mode = fixed` の Actor で、同じ観測について行動選択を 2 回通す。2 回とも全 lane が判断の step になるようにし（`single` にするか、episode の開始として渡す。継続中の lane は行動を前の行動で上書きするため。§2.1 の 9）、2 回とも同じ行動を選んだ lane（少なくとも 1 lane あること）で比べると、skip の値（aux の `skip_q_values`）が一致し、行動価値は一致しない。逆の組（Q 側 `fixed`、skip 専用の方策 `random`）では、行動価値が一致し、同じ行動の lane の skip の値は一致しない。通らなければ D4 のとおり経路を直す。
4. 設定:
   - 既定の設定の短い Run の config dump に、`DefaultDQNAgent.actor.[train].skip.nn_forward_mode = shared`、`[eval]`・`[eval_target]` も `shared` が出て、`skip.nn_forward_key` は出ない。Atari の設定では `[greedy]` も `shared` になる。
   - 受入 1 の snapshot の設定と今の `apps/runner/config` を、4 env（GridMaze・LunarLander・Atari・DropMerge）× 既定・skip 有効の 8 入力で同じ resolver に通して全キー・文字列値を比べると（[prd104/verify_config.py](../../core/anet-core/testdata/prd104/verify_config.py) と同じ方法）、違いは宣言された Actor の `skip.nn_forward_mode = shared` の追加だけで、既存の値の変更と削除は無い。受入 1 は前の設定で checksum を比べるので、新しい `agent.txt` の既定が今と同一の挙動になることは、この比較と受入 2 の `shared` の選び方で確かめる。
5. ステージ 0 の各 Run が `53_evalg/episode` の trace を 200 件持ち、解決後の設定の葉がステージ 0 の表のとおりである。分割の各条件と、§6 の 6 で足した合算の条件は、評価の seed 1 と seed 2 の 2 Run（各 200 件）を合わせて 400 件で数える。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| Learner の skip の forward の NN 実行設定を分ける | 先送り | D5。再開条件: G4 の学習で Train Actor の skip 側のノイズを切っても学習が変わらず、Learner 側のノイズを疑う材料が出たとき |
| skip 側のノイズで j が入れ替わる割合の診断指標（noisy probe の skip 版） | 先送り | ステージ 0 の評価で切り分けられる見込み。再開条件: ステージ 0 の読みが「決めきれない」になったとき |
| Q 側と同じ値を `skip.nn_forward_key` に各 Actor で書き写す形 | 不採用 | Q 側の `nn_forward_key` だけを上書きすると、skip 側が黙って既定で動く。上書きのたびに 2 つの葉を揃える必要がある（D1） |
| キーが無ければ Q 側と共用する形（既存の `skip.policy` と同じ形） | 不採用 | 未指定に意味を持たせる（D1、[AGENTS.md](../../AGENTS.md) §設定値の扱い） |
| モード値でノイズの有無そのものを選ぶ形（例: `skip.noise_mode = shared ／ off`） | 不採用 | NN 実行設定のカタログの外に、2 つ目のノイズの指定を作ることになる。`dedicated` でカタログのキーを選ぶ形で表せる（D1） |
| `skip.head.projection.noisy.enabled = false` で skip の Head のノイズを消す形（ステージ 0 の条件として） | 不採用 | 構造の設定で、Agent 全体と Learner の skip まで変わり、学習済みの checkpoint を Actor ごとの条件で評価できない（D1、§2.1 の 2）。新しく学習する G4 では、群の候補として示す（§7） |
| 学習の引き金を、読みの表で仮説 2 の行のときだけにする形 | 不採用 | 「2 つがそろったときに下がる」行（skip 側を切れば戻る場合）を落とす（D7） |
| skip 専用の方策（`skip.policy`）の共用の決め方も、モード値にそろえる | 範囲外 | 今は葉の有無で共用を決める（§2.1 の 6）。本当の制約ではなく、クリーンブレークでモード値へ移せるが、skip 専用の方策を使う今の設定（GridMaze の `run.@skip` など）の移行が要り、本書の問いには要らない。再開条件: skip 専用の方策の共用の規則を見直すとき |
| Q 側の方策が Greedy 型の Actor で、skip 専用の方策の葉を使わない今の規則を変える | 範囲外 | PRD103 T12 の関心。ステージ 0 の `[greedy]` は EpsilonGreedy なので当たらない（D4）。再開条件: Greedy 型の評価 slot で τ を分けたいとき |
| 既定の変更（Train Actor の skip 側のノイズを切る） | 範囲外 | G4 の学習の結果を見て、別に決める |
| 仮説 1 が主因のときの学習側の手 | 範囲外 | 行動の固定の仕方の変更は非ゴール。PRD104 の `single` と 999 の割り込み (I) の判断に渡す（§6 の 10） |

## 9. 未決

なし（2026-10-29 の再グリルで、暫定版の 6 件はすべて決めた。§6・D6・D7）。

## 10. 依存と順序

- [PRD098](098_temporl_skip_policy_10prd.md)・[PRD103](103_temporl_reference_alignment_10prd.md)・[PRD104](104_temporl_skip_auxiliary_only_10prd.md) の実装に乗る（2026-10-29 までにコミット済み）。
- 受入 1 の snapshot は、本書の実装に着手する前に取る（前のビルドと設定一式）。
- PRD104 のステージ 1・2 は、準備のときに写した実行体と設定（`current-config`）で動き、実行の前に実行体のハッシュを照合する（[run_experiments.py:62-65・199-204](../../core/anet-core/testdata/prd104/run_experiments.py)）。本書の実装は既定（`shared`）で今と同一なので、凍結した組も、凍結した設定の写しを新しい実行体で読む場合も、挙動は変わらない。ステージ 0 の評価の設定は、今の `apps/runner/config` から作る。
- ステージ 0 の τ の 3 条件は、実装を待たずに流せる（§6 の 7）。
- G4 の学習は、ステージ 0 の数値を見てユーザーが決めたあとに、段取りを示して承認を得てから流す（D7）。
- 第三弾の割り込み (I)・PRD104 の `single` とは独立（本書は切り分けだけを行い、行動の固定の仕方を変えない）。

## 11. 参照

- [PRD098 TempoRL 対応](098_temporl_skip_policy_10prd.md)、[PRD103 TempoRL 第二弾](103_temporl_reference_alignment_10prd.md)、[PRD104 補助学習のみモード](104_temporl_skip_auxiliary_only_10prd.md)、[999 TempoRL の後続](999_temporl_skip_followups_10prd.md)
- [ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)
- [200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md)、[130 ニューラルネットワーク（設計文書）](../design/130_neural_networks.jp.md)
- [qbert の行動固定を評価時だけ外す比較（実験記録）](../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)
- [PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md)、[PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md)
- [行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[NoisyNet の観測（wiki）](../wiki/mechanisms/noisynet.md)、[324 DQN 系列（wiki）](../wiki/categories/324_dqn.md)
