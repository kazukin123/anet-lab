# RFD0209: PPO 系（方策ベースの Agent の系統）と、方策ベースの学習に要る core の共通部品

> 2026-11-28 番号を 0202 から 0209 へ振り直した（RFD0204）。 旧い番号のコミット: 66296d56・3ae08c1e。
> 2026-11-25 起票（暫定）。質疑を省き、決めきれない点も、起点の発言から方向が出ている点も、§12 の未決表に「暫定の既定」として置いた。確定した決定は持たない。
> 起点: 2026-11-23・11-25 の開発者の発言（§13.1 に原文）。方向は、DQN 系に続く 2 本目の系統を立てること、最初は離散行動の PPO だけにすること、algo と同時に core の共通部品を育てること、方策と価値の Network を設定だけで共有にも別にもできること、CPU と CUDA の両方で動かすこと、PPO の評価に要るメトリクスを全部入れること。
> 2026-11-25 実装の入口のレビュー（1 往復目）の指摘で改訂。評価の間隔の書き方と ADR 0028 の関係（§6.6）、KL でやめた rollout が返す件数（§7.4・A8）、行動ごとの鍵の受入（A16）、AMP の差し込み位置（§11）、勾配の clip の上限の名前の理由（U30）、「方策の世代」を既存の語で言い換えられないこと（§5）、学習率の既定と時間の値を読む step（§7.4）、KL の閾値の根拠（U23）、材料 M1 の使い先（§13.3）、無い・開けないファイルの扱い（§7.9）、hidden のテストの学習の長さ（A12）、表のセルの縦棒（§7.5）を直した。あわせて、活性化の module `Tanh` を core に足す契約（§7.3・A19）と、移行対象の配線・検証資材・文書（§8）を足した。グリルはしておらず、暫定の印は残る。
> 2026-11-25 実装で分かった点で改訂。出荷の Actor カタログに `[eval_sample]` と `[eval_panel]` を置くこと（§7.4）、経験の形の検査（§7.4）、Head の名前に区切りの「.」を使えないこと（§7.3）、設定の既定の検査が持つ env の設定の一覧への追加（§8）、設定の既定の検査の受入の言い方（A18。基準コミットで既に出る指摘を除く）を足した。グリルはしておらず、暫定の印は残る。
> 2026-11-25 実装のあとのレビューで改訂。2 往復目のあとに直した文言（§6.6 の等間隔になる条件、§7.1 の float32 での正規化、U20 の言い回し）を、この行に記録する。コードにある検査のうち、列挙に無かったもの（§7.1 の浮動小数でない logits、§7.3 の受け取るキーが空・factory が無い組、§7.4 の別の種類の device にある Actor の値）を足し、§8 の用語集の項目の表記を用語集の題に合わせ、§8 の設計文書の追随に 100 を足した。あわせて、用語集の「rollout」の定義を 2 文にした。グリルはしておらず、暫定の印は残る。

## 1. 解く問題

anet-lab の Agent は、DefaultDQN・Rainbow・MuZero（試作）・ImageCls の 4 つで、強化学習の Agent はどれも価値を学んで行動を選ぶ。方策を直接学ぶ Agent が無い。

1. 方策ベースの Agent を足す置き場と、系統としての形が無い。StampFly の計画（§13.3 の材料 M3）は、離散の PPO を最初の段に置いている。
2. core の学習の部品は、Replay を使う価値ベースの学習に合わせて育ってきた。方策ベースの学習に要る部品（行動の分布、advantage の推定、ためてから学ぶ経験の器、方策と価値の 2 つの Head）のうち、どれが core にあり、どれが DQN 系の中にあり、どれが無いかが、整理されていない。
3. DQN 系の中には、ほかの系統でも要る部品がある。系統どうしは依存できない（[011](../design/011_source_layout.jp.md)）ので、PPO 系からは使えない。

## 2. 現状（2026-11-25 にコードで確かめたこと）

### 2.1 学習の契約と Runner

- `Learner::UpdateFromBatch(step, 1 step ぶんの経験)` は、0 件以上の更新結果を返す（[rl.hpp:797-801](../../src/core/base/include/anet/rl.hpp#L797)）。Runner は 1 env step ごとに呼ぶ（Serial は [trainer.cpp:601](../../src/core/base/src/trainer.cpp#L601)、Pipeline は [trainer.cpp:727](../../src/core/base/src/trainer.cpp#L727)）。`learn_step` は返った件数だけ進む（[trainer.cpp:625](../../src/core/base/src/trainer.cpp#L625)・[711](../../src/core/base/src/trainer.cpp#L711)）。
- 更新結果が空なら `LearnEvent` は出ない（[trainer.cpp:605-610](../../src/core/base/src/trainer.cpp#L605)）。定期評価は `LearnEvent` の `learn_step` でだけ発火する（[observers.cpp:620-634](../../src/core/base/src/observers.cpp#L620)）。
- 1 回の `LearnEvent` に更新結果が複数あると、LEARN 軸のメトリクスは 1 件ずつ `learn_step` を進めて記録し、ほかの軸では有限な値の平均を 1 点として記録する（[observers.cpp:913-960](../../src/core/base/src/observers.cpp#L913)）。
- Pipeline の Train Runner は、前の step の学習の完了を待ってから行動を選び、そのあとで直前の経験の学習を投入して、Env の step と並行に走らせる（[trainer.cpp:690-735](../../src/core/base/src/trainer.cpp#L690)）。行動選択と学習は重ならない。T step の経験で更新するとき、T 番目の経験の学習は T+1 番目の行動選択のあとに始まるので、次のまとまりの最初の 1 step だけ、更新の前の Network で行動が選ばれる。
- 出荷の共通設定は `run.train.runner_type ?= pipeline`（[common.txt:3](../../apps/runner/config/common.txt#L3)）。Agent は Runner の種類を受け取らない（`AgentFactory::CreateAgent` の引数に無い。[rl.hpp:1374-1388](../../src/core/base/include/anet/rl.hpp#L1374)）。
- Train の Runner は Actor の `Sync()` を呼ばない。呼ぶのは EvalRunner だけである（[trainer.cpp:250-254](../../src/core/base/src/trainer.cpp#L250)。`rg -n "actor_->Sync\(\)" src/core/base/src/trainer.cpp` は 1 件）。

### 2.2 経験と行動の情報

- Runner が Learner に渡す経験は CPU 上にあり、`state`・`action`（`BatchActionInfo`）・`reward`・`next_state` を持つ（[rl.hpp:650-682](../../src/core/base/include/anet/rl.hpp#L650)）。`next_state` は Reset の前の観測で、`done`（終端）と `truncated`（打ち切り）は別の欄である（[rl.hpp:458-462](../../src/core/base/include/anet/rl.hpp#L458)・[631-635](../../src/core/base/include/anet/rl.hpp#L631)）。
- `BatchActionInfo` は、行動のほかに、Actor が足した値を `info`（`TensorDict`）で運べる（[rl.hpp:548-603](../../src/core/base/include/anet/rl.hpp#L548)）。行動は device 上の値のまま持て、CPU で要るときに 1 回だけ写す。

### 2.3 NN

- Head は、Body の出力を「Head が受け取るキー → branch 名」の対応で受け取る（`NetworkConfig::output_keys`。[nn.hpp:55](../../src/core/base/include/anet/nn.hpp#L55)）。設定では `<接頭辞>.body.output.[<キー>] = <branch 名>` と書く（例: [CartPole.txt:43-44](../../apps/runner/config/CartPole.txt#L43)）。2 つのキーを同じ branch に向ければ共有、別の branch に向ければ別になる。
- core の汎用 Head は `PassThroughHead` と `LinearHead` で、`LinearHead` は `features` のキー 1 つを読み、指定した名前で 1 つ出力する（[nn_head.hpp:278-328](../../src/core/base/include/anet/nn_head.hpp#L278)、[nn_head.cpp:125-134](../../src/core/base/src/nn_head.cpp#L125)）。2 つの Head を 1 つの Network に並べる汎用の factory は core に無い。DQN 系には、外的と内的の Q を並べる `DecomposedHeadFactory` がある（[dqn_heads.hpp:32-57](../../src/algo/dqn/src/dqn_heads.hpp#L32)）。
- 活性化の module は `ReLU`・`GELU`・`SiLU`・`Mish`・`LeakyReLU` の 5 つで、`Tanh` は無い（[nn_modules.cpp:3348-3353](../../src/core/base/src/nn_modules.cpp#L3348)。`rg -n 'repo.Register\(' src/core/base/src/nn_modules.cpp` の 29 件に `Tanh` が無い）。NN の module の登録口は core の内部にあり、基盤の外からは足せない（[010](../design/010_framework_overview.jp.md) §4.3 の拡張点の表）。

### 2.4 ためてから学ぶ器

- core の `LaneTimeTensorStore` は、同じキー・形・device の lane 一式を時刻の順にため、(lane, 時刻) の組でまとめて読む（[tensor_store.hpp:15-33](../../src/core/base/include/anet/tensor_store.hpp#L15)）。たまっている範囲の全体を複製せずに読む口は無い。使っているのは DQN 系の RND の補助学習だけである（[default_dqn_agent.cpp:110-200](../../src/algo/dqn/src/default_dqn_agent.cpp#L110)）。

### 2.5 観測

- 観測を Network の入力（Perception）へ加工する Forwarder の並びは core にあり、使うのは DefaultDQN だけである（[135](../design/135_forwarder_and_perception.jp.md) §3）。並びの出力に stack 軸を補い、開始の印を除く仕上げ（`CompletePerception`）と、spec の照合（`RequirePerceptionSpecs`）は DQN 系の中にある（[dqn_learner.cpp:30-104](../../src/algo/dqn/src/dqn_learner.cpp#L30)）。

### 2.6 CartPole（anet の実装）

- `CartPoleEnv.limit_step`（既定 200）に届くと `truncated` になる。報酬は、step ごとに 1 から姿勢と位置の罰を引いた値で、倒れるか範囲を出ると −1（終端）、打ち切りの step は +1 である。角度の上限は 90 度である（[CartPoleEnv.hpp:12](../../src/env/cartpole/src/CartPoleEnv.hpp#L12)、[CartPoleEnv.cpp:20](../../src/env/cartpole/src/CartPoleEnv.cpp#L20)・[CartPoleEnv.cpp:207-218](../../src/env/cartpole/src/CartPoleEnv.cpp#L207)）。Gymnasium の CartPole-v1 とは、上限の step、報酬、角度の上限が違う。
- 評価の RunMode では、開始の状態が毎回同じ（全部 0）である（[CartPoleEnv.cpp:86-99](../../src/env/cartpole/src/CartPoleEnv.cpp#L86)）。Greedy の評価は、同じ重みなら何本回しても同じ 1 本になる。

### 2.7 参照実装で確かめた式と既定値（2026-11-25 に原文を取得）

CleanRL の `ppo.py` と Stable-Baselines3（SB3）の `ppo.py`・`on_policy_algorithm.py`・`buffers.py` の、既定ブランチの先頭を読んだ（出典は §15）。取得の道具が本文を抜き出して返すので、空白と改行は原文と違うことがある。

| 項目 | CleanRL `ppo.py` | SB3 |
|---|---|---|
| 既定値 | env 4 本、収集 128 step、ミニバッチ 4 分割、4 epoch、lr 2.5e-4（線形に 0 へ）、γ 0.99、λ 0.95、clip 0.2、value の clip あり、entropy 0.01、value 0.5、勾配 clip 0.5、target KL 無し、総 500,000 step | 収集 2048 step、ミニバッチ 64 件、10 epoch、lr 3e-4、γ 0.99、λ 0.95、clip 0.2、value の clip 無し、entropy 0.0、value 0.5、勾配 clip 0.5、target KL 無し |
| GAE | `delta = r + γ·nextvalues·nextnonterminal − values[t]`、`lastgaelam = delta + γ·λ·nextnonterminal·lastgaelam`、`returns = advantages + values` | 同じ式 |
| 打ち切り | 終端と同じに扱う（`next_done = terminations or truncations`）。bootstrap しない | 打ち切りの step の報酬に `γ·V(終端の観測)` を足す |
| 比と KL | `ratio = exp(newlogprob − oldlogprob)`、`approx_kl = mean((ratio − 1) − logratio)` | 同じ式 |
| 方策の損失 | `mean(max(−A·ratio, −A·clamp(ratio, 1−ε, 1+ε)))` | 同じ（`min` の符号反転） |
| value の損失 | clip あり: `0.5·mean(max((v − R)², (v_old + clamp(v − v_old, −ε, ε) − R)²))`。clip なし: `0.5·mean((v − R)²)` | `mse_loss(R, v)`（0.5 倍なし） |
| advantage の標準化 | ミニバッチごとに `(A − mean)/(std + 1e-8)` | 同じ式。件数が 1 のときは行わない |
| 合計 | `pg_loss − ent_coef·entropy + vf_coef·v_loss` | 同じ形 |
| KL で止める | epoch の終わりに、最後のミニバッチの `approx_kl > target_kl` なら残りの epoch をやめる | ミニバッチごとに、`approx_kl > 1.5·target_kl` なら optimizer の step の前でやめる |
| optimizer | Adam、eps 1e-5 | Adam、eps 1e-5 |
| 初期化と Network | 直交初期化。隠れ層は √2、方策の出力は 0.01、価値の出力は 1、bias は 0。方策と価値は別の MLP（64-64、Tanh） | — |
| explained variance | `1 − Var(R − v)/Var(R)`。`Var(R)` が 0 なら NaN | （今回は取得していない） |

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

| # | ゴール |
|---|---|
| G1 | 方策ベースの Agent の系統「PPO 系」を、DQN 系に続く 2 本目の系統として立てる。置き場は `src/algo/ppo`、ライブラリは `anet-algo-ppo`、初期化は `anet::rl::algo::InitPpo()`、Agent の class id は `PPOAgent` |
| G2 | 離散行動の PPO が、Runner の今の契約のまま学習できる |
| G3 | PPO が使う共通部品を core に置き、core・PPO 系・DQN 系の分担と、あとで core へ上げる部品を表にする（§10） |
| G4 | 方策と価値の Network を、設定だけで、共有層ありにも完全に別にもできる |
| G5 | CPU と CUDA の両方で動き、device 間の同期の転送を数えて最小にする（§7.6） |
| G6 | PPO の学習の診断に要るメトリクスを、既定のプロファイル 1 つでそろえる（§7.5） |
| G7 | あとで足すもの（連続行動、Actor と Critic の別の入力、RNN、RND、A2C・PPG・V-trace、SAC）の差し込み位置と、作る条件を書く（§11） |

### 3.2 非ゴール

| # | 非ゴール | 退けた案と、本当の制約か |
|---|---|---|
| NG1 | DQN 系（`src/algo/dqn/`）のコードと動きを変えない | 退けた案: DQN 系の部品を core へ移して PPO と共用する。制約ではない（クリーンブレークで移せる）。今回は開発者の了承が無いので移さず、候補と確かめ方を §10 に書く（U17） |
| NG2 | Runner・Observer・`rl.hpp` の今ある口を変えない | 退けた案: Runner に rollout の境界を知らせる口や、評価を exp step で駆動する口を足す。制約ではない。今の契約で書ける（P1・P2）ので足さない（U5・U8） |
| NG3 | 連続行動を扱わない | 退けた案: Gaussian の分布と連続の Head を同時に作る。制約ではない。行動の受け渡し（Env 基底・経験・Viewer）の横断の改修で、別の関心である（M3 の RFD B） |
| NG4 | Forwarder の並び（FrameStack・Normalize）を PPO 系に結線しない | 退けた案: DefaultDQN と同じ結線を PPO 系にも書く。制約ではない。仕上げの部品が DQN 系の中にあり（§2.5）、同じものを重ねて書くことになるので、今回は結線しない（U11） |
| NG5 | 報酬の正規化、観測の正規化、AMP、可塑性・勾配ノイズの診断、方策の確率の GUI 表示を入れない | 退けた案: DQN 系と同じ診断と表示をそろえる。制約ではない。CartPole の受入に要らず、どれも独立に足せる（§11） |
| NG6 | 正式な Run（番号付きの bat）と実験記録を作らない | 退けた案: 受入の Run を RFD の実装の中で流す。Run の起動は開発者の了承が要る（§13.2） |

### 3.3 成功の測り方

- 実装の正しさ: §9 の受入のうち、テストで確かめる項（A1〜A12、A14〜A19）。
- 学習の成立: CartPole の Run で、評価（Greedy）の `mean.episode_steps` が `CartPoleEnv.limit_step` に届く（§9 の A13。測り方は U19）。
- 共通部品の整理: §10 の表の各行に、置き場、行数、移す条件、無回帰の確かめ方がある。

### 3.4 導入する機構と、ゴールへの寄与

| 機構・概念 | 寄与するゴール |
|---|---|
| PPO 系のライブラリと `PPOAgent` | G1・G2 |
| 行動の分布（`CategoricalDistribution`） | G2・G3 |
| advantage と value の目標の推定（`ComputeGae`） | G2・G3 |
| Head の合成（`CompositeHeadFactory`） | G3・G4 |
| 活性化の module の種類 `Tanh` | G2（参照実装と同じ形の MLP を設定で組む） |
| `LaneTimeTensorStore` の全体の読み出し | G2・G5 |
| rollout（学ぶ前にためる経験のまとまり）と、複数 epoch・ミニバッチの更新 | G2 |
| 方策の世代（Network の重みが更新された回数）と、収集に使った方策の古さ | G2（写しを持つ Actor の誤用を止める）・G6 |
| メトリクスのプロファイル `metrics.scalar.@ppo` | G6 |

## 4. 前提

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 既存部品の働き | `Learner::UpdateFromBatch` は 0 件以上の更新結果を返せる。Learner が中で経験をためて、たまったときにまとめて返す形は、契約を変えずに書ける | §2.1 | 対象外 |
| P2 | 既存部品の働き | 定期評価は、更新結果が空でない `LearnEvent` の `learn_step` でだけ発火する。1 回の `LearnEvent` に複数の更新結果があっても、LEARN 軸は 1 件ずつ、ほかの軸は平均で記録される | §2.1 | 対象外 |
| P3 | 既存部品の働き | Pipeline の Train Runner では、行動選択と学習は重ならず、まとまりの最初の 1 step だけ、更新の前の Network で行動が選ばれる | §2.1 | 対象外 |
| P4 | 事実 | Agent は Runner の種類を受け取らない。Train の Runner は Actor の `Sync()` を呼ばない | §2.1 | 対象外 |
| P5 | 既存部品の働き | 経験は CPU 上にあり、`next_state` は Reset の前の観測で、終端と打ち切りは別の欄である | §2.2 | 対象外 |
| P6 | 既存部品の働き | Actor は、行動のほかの値を `BatchActionInfo` の `info` で Learner へ運べる。行動は device 上のまま持てる | §2.2 | 対象外 |
| P7 | 既存部品の働き | Head が受け取るキーを branch へ対応づける設定で、共有と別を切り替えられる | §2.3 | 対象外 |
| P8 | 事実 | 2 つの Head を並べる汎用の factory は core に無く、DQN 系の中にある | §2.3 | 対象外 |
| P9 | 既存部品の働き | `LaneTimeTensorStore` は core にあり、全体を複製せずに読む口が無い | §2.4 | 対象外 |
| P10 | 事実 | Forwarder の並びの仕上げの部品は DQN 系の中にある | §2.5 | 対象外 |
| P11 | 守る線 | 系統のライブラリは、上位と兄弟のライブラリへ依存しない。系統の公開ヘッダは初期化関数だけを置く | [011](../design/011_source_layout.jp.md)、[AGENTS.md](../../AGENTS.md)「基本方針」 | 対象外 |
| P12 | 守る線 | 汎用層の型・関数は、特定の Agent・Env・アルゴリズムの名前（鍵、欄、モード）を持たない | [AGENTS.md](../../AGENTS.md)「汎用機構と利用側の責任境界」 | 対象外 |
| P13 | 守る線 | 毎 step・毎回の更新の経路に、device 間の同期の転送を足すときは、頻度と 1 回あたりの時間を見積もる。要素のループの中には置かない | [impl.md](../quality/impl.md)「計測」の規約 | 対象外 |
| P14 | 守る線 | 購読の無い診断は計算しない | [110](../design/110_agents_and_learning.jp.md) §7.5 | 対象外 |
| P15 | 守る線 | 共有する Network は通常 eval にし、Learner の学習の区間だけ train にする。Actor の行動選択の中で eval・train を切り替えない | [RFD0134](0134_network_lock_audit_10rfd.md)「Current Contract」 | 対象外 |
| P16 | 過去の判断 | Perception は、渡す先の Network ごとにあり、「Frame をもとに Perception を作る」依存はフレームワークの前提にしない。DefaultDQN は Perception そのものを運ばず、保存した Frame から作り直す | [RFD0161](0161_observation_pipeline_10rfd.md) の D1・D2 | 対象外 |
| P17 | 過去の判断 | 方策と価値の Network は、単純な環境では別のほうがよいと報告されている。画像では計算を節約するため共有が使われる | §13.3 の材料 M1（A.4・B.3） | 対象外 |
| P18 | 事実 | 参照実装の式と既定値 | §2.7 | 対象外 |
| P19 | 事実 | anet の CartPole は Gymnasium の CartPole-v1 と、上限の step・報酬・角度の上限が違う。評価の開始の状態は毎回同じである | §2.6 | 対象外 |
| P20 | 過去の判断 | メトリクスの群の番号は、agent の群を 60〜89 に置く向きで見直す暫定の RFD があり、78・79・89 が空いている | [RFD0108](0108_metrics_tag_numbering_10rfd.md) §5.1 | 対象外 |
| P21 | 用語 | 用語集の「rollout（補助学習）」は、補助学習で学ぶ前にためる、続いた intake のまとまりである。「epoch」はデータセットを一巡する単位である | [CONTEXT.md](../../CONTEXT.md)「rollout（補助学習）」「epoch」 | 対象外 |
| P22 | 事実 | 開発者の発言（方向と好み） | 開発者の発言（2026-11-23・11-25）。§13.1 | 対象外 |
| P23 | 過去の判断 | StampFly の計画は、離散の PPO の受入を「CartPole 離散で 500」とし、Actor と Critic に別の入力を渡す構造を最初から入れるよう求めている | §13.3 の材料 M3 | 対象外 |
| P24 | 事実 | Run の起動、DQN 系の変更、git の操作は、開発者の了承が無い | §13.2 | 対象外 |
| P25 | 過去の判断 | Perception の stack の対象のキーは、1 枚でも必ず stack 軸を持つ。長さ 1 の軸は、Forwarder の仕組みではなく、使う側が足す。Network の定義は、先頭の module で軸の扱いを書く | [ADR 0086](../adr/0086-perception-stack-target-keys-always-carry-stack-axis.md)、[ADR 0096](../adr/0096-stack-axis-and-chain-validity-are-ensured-by-the-caller-not-the-forwarder-mechanism.md) | 対象外 |
| P26 | 過去の判断 | 強制行動は `MakeAction` の入力で、Actor が方策の選択のあとに反映し、Actor の出力が実行された行動になる | [ADR 0083](../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md) | 対象外 |
| P27 | 過去の判断 | Learner が更新する Network は Agent が、Actor だけが使う写しは Actor が所有する。2 つ目の Head は、補助の Network ではなく、同じ Network の合成の Head に置く（DQN 系の価値の分解） | [ADR 0013](../adr/0013-actor-network-resource-policy.md)、[ADR 0074](../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md) | 対象外 |
| P28 | 過去の判断 | `interval` の発火は、間隔の区切りをまたいだ最初のイベントで 1 回で、間隔を刻みの倍数にする決まりは置かない | [ADR 0028](../adr/0028-interval-fires-on-bucket-crossing.md) | 対象外 |
| P29 | 事実 | NN の活性化の module は `ReLU`・`GELU`・`SiLU`・`Mish`・`LeakyReLU` の 5 つで、`Tanh` が無い。module の登録口は core の内部にあり、系統のライブラリからは足せない | §2.3 | 対象外 |

## 5. 用語

- **PPO 系**: 方策を表す Network を、方策勾配で学ぶ Agent の実装の括り。用語集の「Agent の系統」の 1 つで、最初の Agent は `PPOAgent` である。A2C・PPG のように rollout と advantage の推定を分け合う手法は、この系統に足す（§11）。
- **rollout**: 学ぶ前にためる、続いた intake のまとまり。長さは intake の数で数える。用語集の「rollout（補助学習）」と同じ概念で、PPO 系では方策と価値の学習がこの単位でためる。用語集の項目を、補助学習に限らない定義へ広げる（§8）。
- **epoch**: 1 つの rollout の経験を一巡する単位。用語集の「epoch」（データセットを一巡する単位）で、データセットが rollout の経験に当たる。
- **方策の世代**: Learner が 1 つの rollout の更新を終えるたびに 1 進む、Network の重みの番号。Actor は、行動を選んだときの世代を行動の情報に載せる。用語集に、Network の重みの版を数える語は無い。「Train Actor network snapshot」は DQN 系の Actor が持つ写しそのものを指して番号を持たず、「Metricsキャッシュ世代」は Metrics のキャッシュの版で、対象が別である。既存の語では言い換えられないので、この語を置く。
- **方策の古さ**（`policy_lag`）: Learner が rollout を更新するときの方策の世代から、その経験の行動を選んだ世代を引いた値。単位は rollout の更新の回数で、Serial の Runner と共有の Network なら 0 である。用語集の「遷移の年齢」（Replay の中の位置の差）と、DQN 系の target の Network の古さとは別である。
- **advantage**: その行動の価値が、状態の価値をどれだけ上回るかの推定（GAE の出力）。用語集の「アドバンテージストリーム」（Dueling の Network の流れ）とは別である。
- **value の目標**（`value_target`）: value の学習の目標で、advantage に収集時の value を足した値。次の観測の value（bootstrap）を含む。用語集の「target return」（bootstrap を含まない N-step の割引和）と「episode return」（episode の報酬の総和）とは別である。参照実装は `returns` と呼ぶ。
- 設定キーの `ratio`（`ratio_clip_range`）は、新旧の方策の確率の比を指す。DQN 系の `replay_ratio`（更新の回数の比）とは別である。
- 「policy churn」は DQN 系の指標の名前（expected Q の greedy 行動の変化）なので、PPO 系の更新の大きさには使わず、`approx_kl`・`ratio_clip_fraction` と呼ぶ。

## 6. 設計の概要

未決表の暫定の既定を前提に書く。前提にした行は括弧で名指す。

### 6.1 部品の分担（U1・U2・U3）

```text
core（src/core/base）
  distribution    行動の分布（今回は Categorical）            ← 新しい機能グループ
  advantage       advantage と value の目標の推定（今回は GAE） ← 新しい機能グループ
  nn_head         CompositeHeadFactory（複数の Head を 1 つの Network に並べる）
  nn_modules      活性化の module の種類 Tanh を足す
  tensor_store    LaneTimeTensorStore に全体の読み出しを足す

PPO 系（src/algo/ppo）
  ppo             初期化（Agent の登録）
  ppo_config      設定の読み取りと検証
  ppo_network_model   方策と価値の Network の組み立て、方策の世代、保存
  ppo_math        PPO の損失と統計（純粋な計算）
  ppo_actor       行動選択（抽選・Greedy）
  ppo_learner     rollout の蓄積、GAE、複数 epoch の更新、更新結果
  ppo_agent       PPOAgent と factory
```

### 6.2 1 step の流れ

```text
Actor::MakeAction（毎 step）
  観測を device へ移し、各キーに長さ 1 の stack 軸を足す（これを Network の入力とする。U11）
  Network を 1 回通す → logits と value
  分布から行動を抽選する（評価の Actor は確率が最大の行動を選べる）
  行動の情報に載せる: 選んだ行動の log 確率、value、Network の入力、方策の世代

Learner::UpdateFromBatch（毎 step）
  行動の情報から、Network の入力・行動・log 確率・value を device 上の器へためる
  報酬・終端・打ち切りは CPU 上の器へためる
  打ち切りの lane があれば、Reset の前の観測の value を計算してためる（U6）
  rollout の長さに届くまでは、空の結果を返す
  届いたら:
    最後の次の観測の value を計算する
    報酬と終端の印を device へ 1 回で送り、GAE で advantage と value の目標を作る
    epoch ごとに並べ替え、ミニバッチごとに損失 → 勾配 → clip → optimizer の step
    方策の世代を 1 進め、実行した optimizer の step の数だけ更新結果を返す（1 step に 1 件）
```

### 6.3 方策と価値の Network（U4）

Network は 1 つで、Head が `policy_feature` と `value_feature` の 2 つのキーを受け取る。共有か別かは、2 つのキーを向ける branch で決まる。

```text
# 完全に別（CartPole の出荷の設定。P17）
PPOAgent.net.branch.[policy_body].bind = vector
PPOAgent.net.branch.[policy_body].structure = Flatten > PPO_FC1 > Tanh > PPO_FC2 > Tanh
PPOAgent.net.branch.[value_body].bind = vector
PPOAgent.net.branch.[value_body].structure = Flatten > PPO_FC1 > Tanh > PPO_FC2 > Tanh
PPOAgent.net.body.output.[policy_feature] = policy_body
PPOAgent.net.body.output.[value_feature] = value_body

# 共有層あり（2 つのキーを同じ branch に向ける）
PPOAgent.net.branch.[shared_body].bind = vector
PPOAgent.net.branch.[shared_body].structure = Flatten > PPO_FC1 > Tanh > PPO_FC2 > Tanh
PPOAgent.net.body.output.[policy_feature] = shared_body
PPOAgent.net.body.output.[value_feature] = shared_body
```

幹だけを共有し、その先を分ける形（`shared_body` を `bind` する branch を 2 本足す）も、同じ設定の書き方で組める。Actor と Critic に別の入力を渡す形は、2 つの branch の `bind` に別の観測キーを書けば組める（§11）。

### 6.4 打ち切りと rollout の終わりの扱い（U6）

GAE は、各 step について「次の観測の value」と「終端か」「episode の終わりか」を受け取る（§7.2）。Learner は、次の観測の value を次の表で決める。行は上から順に当て、最初に条件を満たした行で扱う。

| # | 条件 | 次の観測の value | 終端の印 | episode の終わりの印 |
|---|---|---|---|---|
| 1 | その step の `next_state.done` が真 | 0 | 真 | 真 |
| 2 | `next_state.truncated` が真で、`learner.truncation_mode = bootstrap` | Reset の前の観測（`next_state.obs`）の value | 偽 | 真 |
| 3 | `next_state.truncated` が真で、`learner.truncation_mode = terminal` | 0 | 真 | 真 |
| 4 | rollout の最後の step | 次の観測（`next_state.obs`）の value | 偽 | 偽 |
| 5 | そのほか | 次の step でためた value | 偽 | 偽 |

`done` と `truncated` が両方真の step は行 1 に当たる。

### 6.5 Runner と Actor の同期（U5・U7）

- Serial と Pipeline のどちらの Train Runner でも動く。Pipeline では、まとまりの最初の 1 step の経験だけ、方策の古さが 1 になる（P3）。記録した log 確率は行動を選んだ方策の値なので、比の計算は成り立つ。出荷の CartPole の設定は `serial` にする。
- Train の Actor は、Learner と Network を共有する（`clone_model = false`）。評価の Actor は写しを持ち、EvalRunner の `Sync()` で写し直す。
- 写しを持つ Actor を Train に使うと、Runner が `Sync()` を呼ばない（P4）ので、重みが古いままになる。Learner は、rollout の経験の方策の古さの最大が 2 以上なら、`ANET_SYSTEM_ERROR` で止める（Pipeline の 1 step 遅れは 1 までなので当たらない）。

### 6.6 評価（U8・U9）

- 評価は、rollout の更新が返る `LearnEvent` でだけ発火しうる。`run.eval_schedule.[tag].interval` は `learn_step`（optimizer の step）で数え、発火は、今の仕組みのとおり、間隔の区切りをまたいだ最初の `LearnEvent` で 1 回である（P28）。`learn_step` は rollout の更新 1 回でまとめて進む（§7.4）ので、それより小さい間隔や、その倍数でない間隔を書いても、発火は rollout の更新 1 回につき多くても 1 回で、区切りをまたいだ最初の更新で起きる。間隔を倍数で書く決まりは置かない。`learner.kl_stop.mode = none` で、評価を rollout の更新の数で等間隔にしたいときは、`epochs × minibatch_count` の倍数を書けばよい（`epoch` で途中でやめる rollout があると、`learn_step` の進みが rollout ごとに変わるので、等間隔にはならない）。
- 出荷の CartPole の設定は、Greedy の評価（`[eval]`）と、抽選の評価（`[eval_sample]`）の 2 本を持つ。`[eval_target]` は、target の Network が無いので休止にする（`interval = 0`）。

## 7. 実装契約

未決表の暫定の既定を前提に書く。

### 7.1 行動の分布（core・`anet/distribution.hpp`）

- `CategoricalDistribution` は、行ごとの logits（行 × 行動数）から作る。行動数は 2 以上である。
- 口は、行動の抽選（`Sample`。乱数の生成器を受け取る）、確率が最大の行動（`GetMode`）、行動の log 確率（`ComputeLogProb`）、entropy（`ComputeEntropy`）、全行動の確率（`GetProbs`）と log 確率（`GetLogProbs`）である。
- 計算は logits と同じ device で行い、結果も同じ device に返す。値を CPU へ読み戻さない。logits は float32 にしてから正規化する（半精度の logits でも、確率の和が 1 からずれないようにする）。
- 構築時に、logits が浮動小数でない、rank が 2 でない、行動数が 2 未満のときは `ANET_SYSTEM_ERROR` で止める。`ComputeLogProb` は、行動の行数が logits と違う、または整数の型でないときに止める。

### 7.2 advantage の推定（core・`anet/advantage.hpp`）

- `ComputeGae` は、時刻 × lane の形の、報酬・value・次の観測の value・終端の印・episode の終わりの印と、γ・λ を受け取り、同じ形の advantage と value の目標を返す。

  ```text
  delta[t]     = reward[t] + γ · next_value[t] · (1 − terminated[t]) − value[t]
  advantage[t] = delta[t] + γ · λ · (1 − episode_end[t]) · advantage[t+1]      （最後の時刻の次は 0）
  value_target[t] = advantage[t] + value[t]
  ```

- 計算は入力と同じ device で行い、勾配を作らない。値を CPU へ読み戻さない。
- 入力の形が互いに違う、rank が 2 でない、γ・λ が [0, 1] に無いときは `ANET_SYSTEM_ERROR` で止める。
- 「GAE」は推定の方法の名前で、特定の Agent の名前ではない（P12）。同じ機能グループに、あとで V-trace・TD(λ) を足す（§11）。

### 7.3 Head の合成と、活性化の `Tanh`（core・`anet/nn_head.hpp`、`nn_modules`）

- `CompositeHeadFactory` は、名前・Head が受け取るキー・Head の factory の組を 1 つ以上受け取り、1 つの Head を作る。
- 作った Head は、組ごとに、受け取ったキーの値を `features` として中の Head に渡し、出力を 1 つの `TensorDict` にまとめる。中の Head は、組の名前で登録する（parameter の名前は `head.<名前>.…` になる）。
- 構築時に、組が 0 個、名前が空か重複、名前が区切りの「.」を含む（名前は module の登録名になる）、受け取るキーが空、Head の factory が無い、受け取るキーが Body の出力に無い、中の Head どうしの出力のキーが重複、のどれかなら `ANET_SYSTEM_ERROR` で止める。
- NN の module の種類に `Tanh` を足す。`net.block.[<名前>].type = Tanh` で選び、入力の各要素に tanh を当てる。形は変えない。固有の設定キーは持たない。

### 7.4 設定キー

接頭辞は `PPOAgent` である。型・値域に反したときは、構築時に `ANET_SYSTEM_ERROR` で止め、メッセージに完全キー・指定値・期待する範囲を入れる。時間の値は、用語集の「時間の値」（定数か、step に対する schedule）で、step は exp step である。

| 設定キー | 型と値域 | 出荷の既定 | 意味 |
|---|---|---|---|
| `rollout.length` | 整数、1 以上 | 128 | rollout の長さ（intake の数） |
| `learner.epochs` | 整数、1 以上 | 4 | 1 つの rollout を一巡する回数 |
| `learner.minibatch_count` | 整数、1 以上 | 4 | 1 epoch のミニバッチの数 |
| `learner.gamma` | 実数、[0, 1] | 0.99 | 割引率 |
| `learner.gae_lambda` | 実数、[0, 1] | 0.95 | GAE の λ |
| `learner.learning_rate` | 時間の値、0 以上 | `@baseline` は 2.5e-4 の定数。CartPole の設定は、2.5e-4 から 0 へ、exp step 500,000 で線形 | 学習率 |
| `learner.adam_eps` | 実数、0 より大きい | 1e-5 | optimizer の eps |
| `learner.weight_decay` | 実数、0 以上 | 0 | 重み減衰 |
| `learner.use_fused_optimizer` | 真偽 | true | fused の AdamW を使うか |
| `learner.ratio_clip_range` | 時間の値、0 より大きい | 0.2 | 比の clip の幅 ε |
| `learner.use_value_clip` | 真偽 | true | value の損失で、収集時の value からの変化を clip するか |
| `learner.value_clip_range` | 実数、0 より大きい | 0.2 | value の変化の clip の幅 |
| `learner.entropy_coef` | 時間の値、0 以上 | 0.01 | entropy の項の係数 |
| `learner.value_coef` | 実数、0 以上 | 0.5 | value の損失の係数 |
| `learner.use_grad_clip` | 真偽 | true | 勾配の全体のノルムを clip するか |
| `learner.grad_clip_max_norm` | 実数、0 より大きい | 0.5 | 勾配のノルムの上限（名前は U30） |
| `learner.advantage_norm_mode` | `none`・`minibatch` | `minibatch` | advantage の標準化 |
| `learner.truncation_mode` | `bootstrap`・`terminal` | `bootstrap` | 打ち切りの扱い（§6.4） |
| `learner.kl_stop.mode` | `none`・`epoch` | `none` | KL で残りの epoch をやめるか |
| `learner.kl_stop.epoch.threshold` | 実数、0 より大きい | 0.015 | `epoch` のとき、epoch の最後のミニバッチの `approx_kl` がこれを超えたらやめる |
| `head.policy.init.*` | 初期化の設定（`WeightInitConfig`） | 直交、gain 0.01 | 方策の Head の初期化 |
| `head.value.init.*` | 初期化の設定 | 直交、gain 1.0 | 価値の Head の初期化 |
| `actor.[<Actor キー>].action_mode` | `sample`・`greedy` | `[train]`・`[eval_sample]` は `sample`、`[eval]`・`[eval_panel]` は `greedy` | 行動の選び方（抽選か、確率が最大の行動か） |
| `actor.[<Actor キー>].clone_model` | 真偽 | Train は false、評価は true | Network の写しを持つか |
| `net.*` | Network の設定（[130](../design/130_neural_networks.jp.md)） | env の設定が書く | Body の配線。`body.output.[policy_feature]` と `[value_feature]` が要る |
| `nn_viz.*` | 図の設定 | 既存と同じ | Network の図 |
| `auto_load_file` | 文字列 | 空 | 起動時に読む保存ファイル |

- 使わない側の設定キー（`use_value_clip = false` のときの `value_clip_range`、`use_grad_clip = false` のときの `grad_clip_max_norm`、`kl_stop.mode = none` のときの `kl_stop.epoch.threshold`）も読み、値域を検証する。通知は出さない。
- ほかの設定と突き合わせる検査は次の 2 つで、どちらも構築時に `ANET_SYSTEM_ERROR` で止める。
  - `rollout.length × lane 数` が `learner.minibatch_count` で割り切れない。
  - `learner.advantage_norm_mode = minibatch` で、1 ミニバッチの件数（`rollout.length × lane 数 ÷ learner.minibatch_count`）が 2 未満。
- 値を丸めたり、既定へ置き換えたりしない。
- 1 つの rollout の更新が返す更新結果は、実行した optimizer の step 1 回につき 1 件である。`learner.kl_stop.mode = none` なら `epochs × minibatch_count` 件になる。`epoch` で残りの epoch をやめたら、実行した epoch の数 × `minibatch_count` 件になる（やめる判定は epoch の終わりにだけ行うので、epoch の途中では切れない）。`learn_step` は、返した件数だけ進む。
- 時間の値（`learner.learning_rate`・`learner.ratio_clip_range`・`learner.entropy_coef`）は、rollout が埋まった intake に Runner が渡した exp step で 1 回読み、その rollout の全部の step に同じ値を使う。
- Env の行動が離散でない、行動数が 2 未満のときは、構築時に止める。
- Learner が受け取った経験の行動の情報に、PPO 系の Actor が載せる値が無いときは、その intake で止める。報酬・終端と打ち切りの印・log 確率・value が、lane ごとに 1 つの値（`[lane 数]`）でない経験と、Actor が載せた値が Agent の device と別の種類の device（CPU と CUDA）にある経験も、その intake で止める。
- 出荷の Actor カタログは `[train]`・`[eval]`・`[eval_sample]`・`[eval_panel]` の 4 つである。`[eval_panel]` は、共通の設定（`common.txt`）が EvalPanel の Actor に `eval_panel` を引くので置き、`[eval]` と同じ設定にする。

### 7.5 メトリクス

更新結果の `GetScalar` の鍵と、出荷のプロファイル `metrics.scalar.@ppo`（`metrics_scalar.txt`）の tag を次のとおりにする。プロファイルは Run プロファイルの既定の連鎖には入れず、PPO の env の設定が `metrics.scalar.$` で選ぶ。step の軸は exp step で、1 つの rollout のミニバッチの値は平均の 1 点になる（P2）。

「単位」が「ミニバッチ」の鍵は、ミニバッチごとの値である。「rollout」の鍵は、その rollout の全部の更新結果が同じ値を持つ。成立しない値は NaN を返す（括弧に条件）。

| tag | 鍵 | 単位 | 中身 |
|---|---|---|---|
| `78_agent_ppo/01_loss_total` | `loss_total` | ミニバッチ | 合計の損失 |
| `78_agent_ppo/02_loss_policy` | `loss_policy` | ミニバッチ | 方策の損失（clip 後） |
| `78_agent_ppo/03_loss_value` | `loss_value` | ミニバッチ | value の損失（係数を掛ける前） |
| `78_agent_ppo/11_approx_kl` | `approx_kl` | ミニバッチ | `mean((ratio − 1) − log ratio)` |
| `78_agent_ppo/12_kl_first_epoch` | `kl_first_epoch` | rollout | 最初の epoch の `approx_kl` の平均 |
| `78_agent_ppo/13_kl_last_epoch` | `kl_last_epoch` | rollout | 実行した最後の epoch の `approx_kl` の平均 |
| `78_agent_ppo/14_epochs_run` | `epochs_run` | rollout | 実行した epoch の数 |
| `78_agent_ppo/15_ratio_clip_fraction` | `ratio_clip_fraction` | ミニバッチ | `ratio − 1` の絶対値が clip の幅を超えた割合 |
| `78_agent_ppo/16_value_clip_fraction` | `value_clip_fraction` | ミニバッチ | value の変化が clip の幅を超えた割合（value の clip が無効なら NaN） |
| `78_agent_ppo/17_ratio_mean` | `ratio_mean` | ミニバッチ | 比の平均 |
| `78_agent_ppo/18_ratio_std` | `ratio_std` | ミニバッチ | 比の標準偏差 |
| `78_agent_ppo/19_ratio_max` | `ratio_max` | ミニバッチ | 比の最大 |
| `78_agent_ppo/20_first_ratio_deviation` | `first_ratio_deviation` | rollout | 最初のミニバッチの、`ratio − 1` の絶対値の最大。方策の古さが 0 なら 0 に近い |
| `78_agent_ppo/31_explained_variance` | `explained_variance` | rollout | `1 − Var(value_target − value)/Var(value_target)`（分母が 0 なら NaN） |
| `78_agent_ppo/32_value_mean` | `value_mean` | rollout | 収集時の value の平均 |
| `78_agent_ppo/33_value_std` | `value_std` | rollout | 収集時の value の標準偏差 |
| `78_agent_ppo/34_value_target_mean` | `value_target_mean` | rollout | value の目標の平均 |
| `78_agent_ppo/35_value_target_std` | `value_target_std` | rollout | value の目標の標準偏差 |
| `78_agent_ppo/36_value_error_std` | `value_error_std` | rollout | `value_target − value` の標準偏差 |
| `78_agent_ppo/37_bootstrap_value_mean` | `bootstrap_value_mean` | rollout | rollout の最後の次の観測の value の平均 |
| `78_agent_ppo/41_advantage_mean` | `advantage_mean` | rollout | 標準化の前の advantage の平均 |
| `78_agent_ppo/42_advantage_std` | `advantage_std` | rollout | 標準化の前の advantage の標準偏差 |
| `78_agent_ppo/43_advantage_norm_abs_max` | `advantage_norm_abs_max` | ミニバッチ | 損失に使った advantage の絶対値の最大 |
| `78_agent_ppo/51_grad_norm` | `grad_norm` | ミニバッチ | clip の前の勾配の全体のノルム |
| `78_agent_ppo/52_grad_norm_shared` | `grad_norm_shared` | ミニバッチ | 方策と価値の両方の Head へ届く parameter の勾配のノルム（無ければ NaN） |
| `78_agent_ppo/53_grad_norm_policy` | `grad_norm_policy` | ミニバッチ | 方策の Head にだけ届く parameter と方策の Head の勾配のノルム |
| `78_agent_ppo/54_grad_norm_value` | `grad_norm_value` | ミニバッチ | 価値の Head にだけ届く parameter と価値の Head の勾配のノルム |
| `78_agent_ppo/55_grad_clipped` | `grad_clipped` | ミニバッチ | 勾配が clip されたら 1、されなければ 0（勾配の clip が無効なら NaN） |
| `78_agent_ppo/61_learning_rate` | `learning_rate` | rollout | その rollout に使った学習率 |
| `78_agent_ppo/62_ratio_clip_range` | `ratio_clip_range` | rollout | その rollout に使った比の clip の幅 |
| `78_agent_ppo/63_entropy_coef` | `entropy_coef` | rollout | その rollout に使った entropy の係数 |
| `79_agent_policy/01_entropy` | `entropy` | ミニバッチ | 方策の entropy の平均 |
| `79_agent_policy/02_entropy_ratio` | `entropy_ratio` | ミニバッチ | entropy を log（行動数）で割った値 |
| `79_agent_policy/03_max_prob_mean` | `max_prob_mean` | ミニバッチ | 確率が最大の行動の確率の平均 |
| `79_agent_policy/31_terminated_ratio` | `terminated_ratio` | rollout | rollout の遷移のうち終端の割合 |
| `79_agent_policy/32_truncated_ratio` | `truncated_ratio` | rollout | rollout の遷移のうち打ち切りの割合 |
| `79_agent_policy/33_episode_end_count` | `episode_end_count` | rollout | rollout の中で終わった episode の数 |
| `79_agent_policy/34_policy_lag` | `policy_lag` | rollout | 方策の古さの平均 |

- 行動ごとの鍵 `action_prob_<n>`（行動 n の確率の平均）と `action_freq_<n>`（ミニバッチの行動のうち行動 n の割合）は、ミニバッチごとの値で、n が 0 以上、行動数未満のときに値を返す。n が行動数以上なら NaN を返す。n が数字の並びでない鍵は、表に無い鍵と同じく、値なしを返す。tag は env の設定が書く（CartPole は `79_agent_policy/10_action_prob_0`・`11_action_prob_1`・`20_action_freq_0`・`21_action_freq_1`）。
- 成績は、今ある Runner の鍵を使う。`21_eval/11_return_greedy`・`12_return_sample`（`mean.episode_return`）、`21_eval/21_ep_steps_greedy`・`22_ep_steps_sample`（`mean.episode_steps`）、学習側の `20_eps`・`10_train`、`90_perf`（`exp_step_per_sec` など）をプロファイルに入れる。
- 統計は、Train の LEARN の購読に、上の表と行動ごとの鍵のどれかがあるときだけ計算する。勾配の内訳（`grad_norm_shared`・`grad_norm_policy`・`grad_norm_value`）は、その鍵の購読があるときだけ計算する（P14）。
- 時間は、`PPOAgent` の Actor の行動選択、Learner の蓄積・GAE・更新に `ANET_PROFILE_SCOPE` の計測範囲を置く。

### 7.6 device と同期の転送

agent の device が CUDA のとき、PPO 系が行う device 間の同期の転送は次のとおりである。CPU のときは、どれも起きない。T は rollout の長さ、K は epoch の数（`learner.epochs`）、M は 1 epoch のミニバッチの数（`learner.minibatch_count`）である。

| 経路 | 向き | 頻度 | 中身 | 避けられるか |
|---|---|---|---|---|
| Actor の行動選択 | CPU → device | 毎 step に 1 回 | 観測（lane 数ぶん） | 避けられない（Env が CPU） |
| Env の step | device → CPU | 毎 step に 1 回 | 行動（lane 数ぶん）。`BatchActionInfo` が写す | 避けられない |
| 打ち切りの value | CPU → device | 打ち切りの lane がある step だけ | Reset の前の観測と、打ち切りの印 | `truncation_mode = terminal` なら無い |
| rollout の終わり | CPU → device | rollout に 1 回 | 報酬・終端・打ち切りの印（T × lane 数）をまとめた 1 本と、最後の次の観測 | 避けられない |
| 更新結果の読み出し | device → CPU | rollout に 1 回 | 実行した step の件数ぶん（多くても K × M 件）の統計をまとめた 1 本。購読が無ければ無い | 購読しなければ無い |
| KL で止める判定 | device → CPU | `kl_stop.mode = epoch` のとき、epoch に 1 回 | `approx_kl` 1 個 | `none` なら無い |

- lane や行を回すループの中に、同期の転送を置かない（P13）。
- Learner は、経験の観測を device へ送り直さない。Actor が Network へ渡した入力を、行動の情報で受け取る（U10）。
- 並べ替えの番号と抽選は、device の上で作る。

### 7.7 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `anet::rl::CategoricalDistribution` | 新規 | 離散行動の分布。抽選・最大確率の行動・log 確率・entropy | core `distribution` |
| `anet::rl::GaeResult` | 新規 | advantage と value の目標の組 | core `advantage` |
| `anet::rl::ComputeGae`（関数） | 新規 | GAE で advantage と value の目標を作る | core `advantage` |
| `anet::nn::CompositeHeadEntry` | 新規 | 名前・Head が受け取るキー・factory の組 | core `nn_head` |
| `anet::nn::CompositeHeadFactory` | 新規 | 複数の Head を 1 つの Head にまとめる | core `nn_head` |
| NN の module の種類 `Tanh`（module とその factory。core の内部） | 新規 | 入力の各要素に tanh を当てる活性化 | core `nn_modules` |
| `anet::LaneTimeTensorStore` | 変更 | たまっている範囲の全体を、時刻 × lane の形で複製せずに読む口 `GetAll` を足す | core `tensor_store` |
| `anet::rl::algo::InitPpo`（関数） | 新規 | PPO 系の Agent の登録 | PPO 系 `ppo`（公開ヘッダ `anet/algo/ppo.hpp`） |
| `anet::rl::ppo::PPOAgentConfig` | 新規 | `PPOAgent` の設定の読み取りと検証 | PPO 系 `ppo_config` |
| `anet::rl::ppo::ActorConfig`・`LearnerConfig` | 新規 | Actor カタログの 1 項目と、Learner の設定の値 | PPO 系 `ppo_config` |
| `anet::rl::ppo::NetworkModel` | 新規 | 方策と価値の Network の組み立て、写し、保存と読込、方策の世代、勾配の内訳に使う parameter の組 | PPO 系 `ppo_network_model` |
| `anet::rl::ppo::Actor` | 新規 | 行動選択と、行動の情報への値の載せ方 | PPO 系 `ppo_actor` |
| `anet::rl::ppo::Learner` | 新規 | rollout の蓄積、GAE、複数 epoch の更新 | PPO 系 `ppo_learner` |
| `anet::rl::ppo::UpdateResult` | 新規 | 更新結果。統計をまとめて持ち、`GetScalar` で 1 回だけ読み戻す | PPO 系 `ppo_learner` |
| `anet::rl::ppo::PPOAgent` | 新規 | Actor と Learner の生成、資源の所有、保存 | PPO 系 `ppo_agent` |
| `anet::rl::ppo::PPOAgentFactory` | 新規 | 設定から `PPOAgent` を作る。class id は `PPOAgent` | PPO 系 `ppo_agent` |
| PPO の損失と統計の関数群（`ComputeClippedPolicyLoss`・`ComputeValueLoss`・`ComputeApproxKl`・`ComputeExplainedVariance` など） | 新規 | 入力の tensor から損失と統計を作る純粋な計算 | PPO 系 `ppo_math` |

### 7.8 機能グループの見込み

| ファイル（ライブラリ） | 関心 | 今回入るもの | あとで入る見込み | 入れないもの | 近い既存の機能グループと、そこへ入れない理由 | 機能カテゴリ | 設計文書 |
|---|---|---|---|---|---|---|---|
| `anet/distribution.hpp`・`distribution.cpp`（core、公開） | 行動の分布 | `CategoricalDistribution` | 対角の Gaussian、tanh で縮めた Gaussian、分布の共通の口（連続行動の RFD）、合法手のマスク | Head の作り方、行動の受け渡し | `rl`（契約のヘッダで、計算の部品を置かない）。`nn_head`（Head は分布の入力を出すだけ） | 321 Agent 共通基盤（既存の行の `match` に `distribution` を足す） | 新設 `docs/design/115_on_policy_learning.jp.md`（以下 115） |
| `anet/advantage.hpp`・`advantage.cpp`（core、公開） | advantage と value の目標の推定 | `ComputeGae`・`GaeResult` | V-trace、TD(λ)（§11） | 損失、rollout の器 | `replay_experience`（Replay の N-step の組み立てで、Replay の窓を前提にする） | 318 強化学習 共通基盤（既存の行の `match` に `advantage` を足す） | 新設 115 |
| `ppo.cpp`・`anet/algo/ppo.hpp`（PPO 系） | 系統の初期化 | `InitPpo` | 系統に足す Agent の登録 | — | — | 326 PPO 系列（行を新しく立てる。U18） | 新設 `docs/design/230_ppo_agents.jp.md`（以下 230） |
| `ppo_config.hpp`・`.cpp`（PPO 系、内部） | 設定 | 設定の型と検証 | 足す機能の設定 | — | — | 326 | 230 |
| `ppo_network_model.hpp`・`.cpp`（PPO 系、内部） | Network の持ち主 | `NetworkModel` | 連続行動の Head、2 つの価値の Head | 損失 | — | 326 | 230 |
| `ppo_math.hpp`・`.cpp`（PPO 系、内部） | 損失と統計の計算 | 損失と統計の関数 | A2C・PPG の損失 | 状態を持つ処理 | core `advantage`（推定は系統の外でも使える。損失は PPO 系の中身） | 326 | 230 |
| `ppo_actor.hpp`・`.cpp`（PPO 系、内部） | 行動選択 | `Actor` | 再帰状態 | 学習 | — | 326 | 230 |
| `ppo_learner.hpp`・`.cpp`（PPO 系、内部） | 学習 | `Learner`・`UpdateResult` | 補助の相（PPG）、内的の advantage | 行動選択 | — | 326 | 230 |
| `ppo_agent.hpp`・`.cpp`（PPO 系、内部） | Agent と factory | `PPOAgent`・`PPOAgentFactory` | 系統の 2 つ目の Agent（共通の組み立てを分けるときは `ppo_assembler` を立てる） | — | — | 326 | 230 |

DQN 系にある `dqn_assembler`（2 つの Agent が分け合う部品の組み立て）と `dqn_heads`（専用の Head）に当たるファイルは、今回は立てない。PPO 系の Agent が 1 つで、専用の Head が無い（core の `LinearHeadFactory` と `CompositeHeadFactory` で足りる）ためである。2 つ目の Agent か、専用の Head（連続行動の標準偏差など）を足すときに立てる。

### 7.9 保存

`PPOAgent` は、形式の名前（`PPOAgent-1`）、設定の文字列、Network、optimizer を保存する。`auto_load_file` があれば、起動時に同じ形式から Network と optimizer を読む。形式の名前が違うファイルと、無い・開けないファイルは、構築時に `ANET_SYSTEM_ERROR` で止める。

### 7.10 排他

- 共有する Network は通常 eval にする。Learner は、損失の forward から optimizer の step までの区間だけ、Agent の排他のロックを持って train にする（P15）。
- Network を共有する Actor は、forward の間、共有のロックを持つ。写しを持つ Actor は、`Sync()` の間だけ共有のロックを持って重みを写す。
- Actor は、行動選択の中で eval・train を切り替えない。

## 8. 移行対象

今ある契約は変えない（NG1・NG2）。足すものと、あわせて直すものを挙げる。

| 種類 | 対象 | 内容 |
|---|---|---|
| core のコード | `src/core/base/include/anet/distribution.hpp`・`advantage.hpp`、`src/core/base/src/distribution.cpp`・`advantage.cpp` | 新設 |
| core のコード | `anet/nn_head.hpp`・`nn_head.cpp`、`nn_modules.cpp`、`anet/tensor_store.hpp`・`tensor_store.cpp` | §7.3 と `GetAll` を足す |
| core のテスト | `distribution_test.cpp`・`advantage_test.cpp`（新設）、`nn_test.cpp`・`tensor_store_test.cpp`（追加） | §9 |
| PPO 系 | `src/algo/ppo/`（`CMakeLists.txt`、`include/anet/algo/ppo.hpp`、`src/` の 7 つの機能グループとテスト） | 新設 |
| ビルド | ルートの `CMakeLists.txt`、`src/apps/runner/CMakeLists.txt` | `src/algo/ppo` の追加と、Runner へのリンク |
| Runner | `src/apps/runner/src/RunnerApp.cpp` | `InitPpo()` の呼び出し |
| 検証資材 | `src/core/base/testdata/rfd0182/check_stage3_dependencies.py` | 系統の一覧に `anet-algo-ppo` を足す（PPO 系が兄弟の系統に依存しないことを、同じ検査で確かめる） |
| 検証資材 | `src/core/base/testdata/rfd0194/expected.jsonl` | PPO 系の、設定キーを示す診断で期待一覧に差が出たら、RFD0194 の規則で足す |
| 検証資材 | `src/core/base/testdata/rfd0072/migrate.py` | env の設定ファイルの一覧に `CartPole_ppo.txt` を足す（設定の既定の検査 `check_default_leaves.py` が、新しい env の設定の行を分類できるようにする） |
| 設定 | `apps/runner/config/agent.txt` | `PPOAgent.@baseline` の既定と Actor カタログ |
| 設定 | `apps/runner/config/nn.txt` | `Tanh` の block と、PPO の MLP の block（直交初期化）。module の種類の一覧のコメント |
| 設定 | `apps/runner/config/metrics_scalar.txt` | `metrics.scalar.@ppo` |
| 設定 | `apps/runner/config/CartPole_ppo.txt` | 新設（CartPole を `PPOAgent` で回す env の設定） |
| 設計文書 | `docs/design/115_on_policy_learning.jp.md` | 新設（core の、方策ベースの学習の共通部品） |
| 設計文書 | `docs/design/230_ppo_agents.jp.md` | 新設（PPO 系 Agent） |
| 設計文書 | `README.jp.md`（文書の一覧）、`010`（§5.2 の実装の一覧）、`011`（置き場の表）、`110`（§2.4 登録済み Agent、§4 コードマップ）、`130`（Head の節、module の種類の表）、`150`（`LaneTimeTensorStore` の記述、Agent ごとの適用範囲の表）、`100`（NN の設定の読込の接頭辞を持つ Agent Factory の列挙） | 追随 |
| 用語集 | `CONTEXT.md` | 「rollout（補助学習）」を「rollout」に広げ、「PPO系エージェント」を足す。「Agent の系統」の例に PPO 系を足す |
| 分類 | `reports/tools/categories.json` | 326 の行と、321・318 の `match` |
| wiki | `docs/wiki/` | トピック別まとめへの書き戻しと `log.md` |

- 英語版の設計文書（`*.en.md`）は、今回は作らず、直さない。日本語版を直したあとの英訳は `anet-translate-docs` で行う。
- 旧い契約のための仕組み（別名、互換の分岐）は足さない。過去の記録（Run の記録、ADR、実験記録）は書き換えない。
- 用語集の「rollout（補助学習）」という題を引く現用の文書は、用語集のほかに無い（2026-11-25 に `git grep -n -i "rollout（補助学習）"` を、`docs/memo`・`docs/adr`・`docs/experiments` を除いたリポジトリ全体に引いて 1 件。その 1 件が用語集の題である）。
- RFD の課題（`rfd` でこの RFD を指す課題）は、起票の時点では結び付けていない。課題の起票は開発者の了承が要るので、候補として報告に挙げる。

## 9. 受入

| # | 確かめること | 確かめ方 |
|---|---|---|
| A1 | Debug のビルドが通る | `cmake --build --preset x64-Debug` が終了値 0 |
| A2 | 分布の計算が正しい | `anet-core-base-test.exe "[distribution]"` が通る。log 確率と entropy が手計算の値と合い、抽選の頻度が確率に合い（行動 3 つ、10 万回、差 0.01 以内）、不正な形で止まる。CPU と、CUDA が使える環境では CUDA でも通る |
| A3 | GAE の計算が正しい | `anet-core-base-test.exe "[advantage]"` が通る。手計算の 3 step の例、終端、episode の終わり（打ち切りで次の観測の value を使う）、λ = 0（1 step の TD）、λ = 1（割引の和）で値が合い、不正な形と範囲で止まる |
| A4 | Head の合成が正しい | `anet-core-base-test.exe "[composite_head]"` が通る。2 つの Head の出力が 1 つにまとまり、受け取るキーごとに別の branch の値が届き、重複と不足で止まる |
| A5 | 器の全体の読み出しが正しい | `anet-core-base-test.exe "[tensor_store]"` が通る。`GetAll` が、ためた時刻の数 × lane 数の形で、ためた値を返す |
| A6 | PPO 系のテストが CPU と CUDA で通る | `anet-algo-ppo-test.exe "[ppo]"` が通る。device を受けるテストは CPU と CUDA の両方で走り、CUDA が使えない環境では CUDA の分を飛ばしたと出る |
| A7 | 方策と価値の Network が、設定だけで共有にも別にもなる | A6 の中のテスト。共有の設定では、方策と価値の両方の Head へ届く parameter があり、別の設定では 1 つも無い。別の設定で、`value_feature` に向けた branch だけが読む観測キーの値を変えても、logits が変わらない |
| A8 | rollout と更新の進み方が正しい | A6 の中のテスト。rollout の長さに届くまで更新結果が空で、`learner.kl_stop.mode = none` では、届いた intake で `epochs × minibatch_count` 件が返る。`epoch` で、最初の epoch の終わりに閾値を超える設定（閾値を小さく、学習率を大きくする）では、`minibatch_count` 件が返り、`epochs_run` が 1 になる。割り切れない組み合わせは構築で止まる |
| A9 | 更新の前の最初のミニバッチで、比が 1 になる | A6 の中のテスト。共有の Network で集めた rollout で、`first_ratio_deviation` が 1e-5 以下 |
| A10 | 打ち切りの扱いが設定どおりになる | A6 の中のテスト。`bootstrap` では打ち切りの step の value の目標に次の観測の value が入り、`terminal` では入らない |
| A11 | 写しを持つ Actor を Train に使うと止まる | A6 の中のテスト。方策の古さの最大が 2 以上の rollout で `ANET_SYSTEM_ERROR` |
| A12 | 学習が進む | `anet-algo-ppo-test.exe "[ppo_learn]"`（hidden。CPU）が通る。テストの中の小さな課題（一本道を右へ進むと報酬。打ち切りあり）で、テストが固定した長さ（rollout の数。実装計画に書く）の学習のあと、seed 3 つのすべてで、最後の 20 個の rollout の中で終わった episode の return の平均が、最適な方策の return の 9 割以上になる |
| A13 | CartPole で学習が成立する | Run で確かめる（開発者の了承のあと。§13.2）。`CartPole_ppo.txt`、`CartPoleEnv.limit_step = 500`、seed 3 つ、exp step 500,000。3 seed のすべてで、Greedy の評価（`21_eval/21_ep_steps_greedy`）が 500 に届き、exp step 400,000 から 500,000 の評価の点の 8 割以上が 500 である。この Run が流れないまま実装を終えるときは、未実施として実装計画に書く |
| A14 | DQN 系を変えていない | `git diff --stat 9310b087 -- src/algo/dqn` の出力が空（基準コミットは起票の時点の HEAD）。`anet-algo-dqn-test.exe` が通る |
| A15 | 既定スイートが通る | `anet-core-base-test.exe`・`anet-algo-dqn-test.exe`・`anet-algo-muzero-test.exe`・`anet-algo-imagecls-test.exe`・`anet-algo-ppo-test.exe` を引数なしで走らせて、失敗が無い（基準コミットで既に落ちるテストは、実装計画に名前を書いて除く） |
| A16 | メトリクスの鍵がそろっている | A6 の中のテスト。§7.5 の表の鍵のすべてで、更新結果の `GetScalar` が値か NaN を返す。行動ごとの鍵（`action_prob_<n>`・`action_freq_<n>`）は、行動数 2 の課題で、n = 0・1 が値、n = 2 が NaN を返す。表にも行動ごとの鍵にも当たらない鍵（n が数字の並びでないものを含む）は、値なしを返す。`metrics.scalar.@ppo` の、更新結果を読む行の鍵が、どれも表にある（`[shipped_config]`） |
| A17 | 設定の不正が止まる | A6 の中のテスト。§7.4 の表の値域の外の値、未知の列挙値、未定義の Actor キー、離散でない行動で、完全キーと指定値を含むメッセージで止まる |
| A18 | 設定ファイルの検査が通る | `anet-core-base-test.exe "[shipped_config]"` と `anet-algo-ppo-test.exe "[shipped_config]"` が通る。`check_default_leaves.py` の指摘が、基準コミットから増えない（基準コミットで既に出る指摘は、実装計画に書いて除く） |
| A19 | `Tanh` の module が使える | `anet-core-base-test.exe "[nn]"` が通る。設定で `type = Tanh` の block を並べた Network が、入力に tanh を当てた値を返す |

## 10. DQN 系から core へ上げる候補（今回は移さない）

DQN 系の中にあり、ほかの系統でも要る部品である。今回は DQN 系を変えない（NG1）ので移さず、PPO 系は、要る分を core の新しい部品（§7.1〜§7.3）と自分の実装で持つ。移すかどうかと順は U17 で決める。行数は 2026-11-25 の作業ツリーで数えた概算である。

| # | 部品（今の場所） | 行数 | PPO 系の今回の扱い | 移す条件 | 移したときに DQN 系で変わる所 |
|---|---|---|---|---|---|
| C1 | rollout をためて、epoch ごとに並べ替え、ミニバッチで読む（`IntrinsicLearnRolloutBatcher`。[default_dqn_agent.cpp:110-200](../../src/algo/dqn/src/default_dqn_agent.cpp#L110)） | 約 90 | `ppo_learner` に、device 上の並べ替えで書く | 3 つ目の利用者が出たとき、または RND の補助学習を PPO 系で使うとき | RND の補助学習（`intrinsic.learn.mode = rollout`）が core の部品を呼ぶ。乱数の消費の順が変わると、同じ seed の結果が変わる |
| C2 | Actor の Network の写しと同期（`CopySourceNetwork`・`UpdateSnapshot`・`Sync`。[dqn_actor.cpp:1436-1487](../../src/algo/dqn/src/dqn_actor.cpp#L1436)。ImageCls にも同じ形） | 約 55 | `ppo_actor` に、`Sync()` での写し直しだけを書く | 写しを持つ Train の Actor を PPO 系で使うとき（rollout ごとの同期が要る） | `dqn::Actor` と `ImageClsActor` の同期が core の部品に置き換わる |
| C3 | optimizer の組み立てと、学習率の書き戻し（`BuildOptimizerOptions`・`SetupOptimizer`。[dqn_learner.cpp:874-930](../../src/algo/dqn/src/dqn_learner.cpp#L874)。ImageCls にも同じ形） | 約 60 | `ppo_learner` に、1 つの group で書く | parameter の group を分けた学習率を PPO 系で使うとき | group の分け方（Noisy の σ、価値の分解）を呼ぶ側が渡す形になる |
| C4 | 購読ごとの診断の間引き（`DiagnosticDemandTable` ほか。[dqn_math.hpp:29-111](../../src/algo/dqn/src/dqn_math.hpp#L29)） | 約 85 | 使わない。購読の有無だけで統計の計算を切り替える（§7.5） | 鍵ごとの間隔で計算を間引く診断（可塑性など）を PPO 系に足すとき | include と名前空間。エラーの文言の「DQN diagnostic」 |
| C5 | 2 つの Head を並べる（`DecomposedHeadFactory`。[dqn_heads.hpp:32-57](../../src/algo/dqn/src/dqn_heads.hpp#L32)） | 約 25（宣言） | core の `CompositeHeadFactory`（§7.3）を使う | DQN 系を core の部品の上に載せ直すとき | 出力の鍵の接頭辞と parameter の名前（optimizer の group 分けが使う） |
| C6 | Perception の仕上げと spec の照合、重ねる設定の解決（`CompletePerception`・`RequirePerceptionSpecs`。[dqn_learner.cpp:30-104](../../src/algo/dqn/src/dqn_learner.cpp#L30)。`ResolveReplayFrameStack`。[default_dqn_agent.cpp:64-106](../../src/algo/dqn/src/default_dqn_agent.cpp#L64)） | 約 120 | 使わない。Forwarder の並びを結線しない（NG4） | PPO 系に Forwarder の並びを結線するとき（U11） | DefaultDQN の結線が core の補助を呼ぶ |
| C7 | 設定の検証の補助（`ValidateChoice`・`ValidateRange` ほか。[default_dqn_agent.hpp:126-168](../../src/algo/dqn/src/default_dqn_agent.hpp#L126)） | 約 45 | `ppo_config` に、要る分を書く | 3 つ目の Agent の設定が同じ検証を書くとき | DefaultDQN の設定の検証が core の関数を呼ぶ |
| C8 | RND の段取り（`IntrinsicRewardCoordinator`。[default_dqn_agent.cpp:204-714](../../src/algo/dqn/src/default_dqn_agent.cpp#L204)） | 約 510 | 使わない | PPO 系に RND を足すとき（§11） | 設定の型、報酬の積み方、保存の形式 |

無回帰の確かめ方（どの行にも当てる）:

1. `anet-algo-dqn-test.exe`（`[dqn]`・`[config]`・`[determinism]` を含む）と `anet-core-base-test.exe` が通る。
2. 設定の基準（`src/core/base/testdata/rfd0072/`）と `[shipped_config]` が通る。
3. 同じ seed の Run の全点の照合（`src/core/base/testdata/rfd0097/compare_metrics.py`）。移す前の Release の Runner を保存し、移したあとと、GridMaze・LunarLander・Q*bert・DropMerge で比べる。LunarLander・DropMerge・Atari は `backend.$=backend.@deterministic` で比べる。手順は [RFD0161 の実装計画](0161_observation_pipeline_20impl.md)の「検証」による。C1・C8 は RND を有効にした構成を加える。
4. ライブラリの依存の検査（`src/core/base/testdata/rfd0182/check_stage3_dependencies.py`）。

## 11. あとで足すものの差し込み位置

| 足すもの | 差し込み位置 | 作る条件 | 先送りの理由 |
|---|---|---|---|
| 連続行動（Gaussian、tanh で縮める、log 確率の補正） | core `distribution` に分布と共通の口。`ppo_network_model` に連続の Head。行動の受け渡しは Env 基底と経験 | 連続行動の RFD（M3 の RFD B） | 別の関心（行動の受け渡しの横断の改修） |
| Actor と Critic の別の入力 | Network の 2 つの branch の `bind` に別の観測キーを書く（今回の形で組める。A7 が、`value_feature` に向けた branch だけの入力が logits に効かないことを確かめる）。真値の観測キーの宣言と、`policy_feature` に向けた branch に真値が入らないことの構築時の検査は Env の側 | 真値の観測を出す Env（SimStampFlyEnv）の RFD | 別の関心（観測キーの定義と Env） |
| Forwarder の並び（FrameStack・Normalize） | `ppo_actor` の Network の入力を作る 1 か所と、`ppo_learner` の次の観測の入力を作る 1 か所 | Atari か連続行動を PPO 系で回すとき（U11） | 別の関心（§10 の C6 の移行と合わせる） |
| RNN | `ppo_actor` が再帰状態を持ち、行動の情報で運ぶ。`ppo_learner` のミニバッチを lane 単位にする | RNN の module が NN に入ったあと | 別の関心 |
| RND（外的と内的の 2 つの価値の Head、2 つの advantage） | `CompositeHeadFactory` に価値の Head を 1 つ足す。`ComputeGae` を 2 回呼ぶ。推定器は core の `intrinsic` | Montezuma を PPO 系で回すとき | 実験の結果が要る（DQN 系の RND の評価が進行中） |
| A2C | `ppo_learner` の損失の選択（clip なし、1 epoch、1 ミニバッチ） | 比べたいとき | 別の関心 |
| PPG | `ppo_learner` に補助の相、`CompositeHeadFactory` に補助の価値の Head | 共有の Network で方策と価値の釣り合いが問題になったとき | 実験の結果が要る |
| V-trace・TD(λ) | core `advantage` に関数を足す | 非同期の収集か、off-policy の補正が要るとき | 別の関心 |
| SAC | 別の系統（Replay と Q を使う。分布は core `distribution` を分け合う。U13） | 連続行動のあと | 別の関心 |
| 報酬の正規化 | `ppo_learner` の蓄積の前に core の `RewardScaler` | 報酬の尺度が大きい Env を回すとき | 別の関心 |
| AMP（半精度の forward） | `ppo_actor` と `ppo_learner` の Network の forward を、core の `Autocast` で囲む。分布は float32 で正規化する（§7.1） | 画像の Env を PPO 系で回し、更新の時間が Run の長さを決めるようになったとき | 別の関心 |
| 写しを持つ Train の Actor の、rollout ごとの同期 | `ppo_actor` が方策の世代を見て写し直す（§10 の C2） | Actor を別の device で回すとき | 別の関心 |
| 可塑性・勾配ノイズの診断 | `ppo_learner`（core の `ComputePlasticityMetrics`・`MeasureGradientNoise`） | PPO 系の学習の停滞を調べるとき。[RFD0092](done/0092_gradient_noise_scale_10rfd.md) §8 の 5 の再開条件（対象の Learner が具体化）は本 RFD で満たす | 別の関心 |
| 方策の確率の GUI 表示 | Actor が `info` に確率を載せ、Runner の panel が読む | GUI で方策を見たいとき | 別の関心（毎 step の device から CPU への転送が増える） |

## 12. 未決表

暫定の起票なので、決定はすべてここに置く。「暫定の既定」は、再グリルで見直す前提で AI が置いたものである。

| # | 事項 | 暫定の既定 | 理由と、よりどころにした前提 |
|---|---|---|---|
| U1 | 系統の名前と置き場 | 系統は「PPO 系」。`src/algo/ppo`・`anet-algo-ppo`・`anet::rl::ppo`・`InitPpo()`。Agent の class id と設定の接頭辞は `PPOAgent` | 開発者の発言 1（P22）。置き場は P11。class id は `DefaultDQNAgent`・`MuZeroAgent` と同じ形 |
| U2 | core に置く共通部品の範囲 | 分布、advantage の推定、Head の合成、器の全体の読み出しの 4 つと、活性化の module `Tanh`。損失、rollout の並べ替え、optimizer の組み立ては PPO 系に置く | 開発者の発言 2（P22）。4 つは、特定の Agent の名前を持たずに書け（P12）、参照した 6 つのフレームワークのうち 3 つ以上が系統をまたいで分け合う部品である（M1 の A.11）。残りは 2 つ目の利用者が出たときに上げる（§10）。`Tanh` は、参照実装と同じ MLP を設定で組むのに要り、系統のライブラリからは足せない（P29） |
| U3 | PPO 系のファイルの分け方 | 7 つの機能グループ（§7.8）。`ppo_assembler`・`ppo_heads` は立てない | 開発者の発言 6（P22）。DQN 系の役割の分け方に合わせ、中身が無いファイルは作らない |
| U4 | 方策と価値の Network の形 | 1 つの Network で、Head が受け取る 2 つのキーを branch へ向ける設定で、共有と別を選ぶ。出荷の CartPole は別 | 開発者の発言 3（P22）。今ある設定の書き方で書ける（P7）。DQN 系の価値の分解も、2 つ目の Head を同じ Network の合成の Head に置いている（P27）。Network を 2 つ持つ案は、写し・保存・同期・optimizer が 2 組になり、共有の形を別の仕組みで作ることになる。出荷を別にするのは P17 |
| U5 | Runner の種類 | Serial と Pipeline の両方で動かす。Pipeline の 1 step の遅れは許し、`policy_lag` で見えるようにする。出荷の CartPole は `serial` | P3・P4。Agent は Runner の種類を知らないので、構築で止められない。Runner に待ち合わせの口を足す案は NG2 |
| U6 | 打ち切りの扱い | `learner.truncation_mode = bootstrap`（次の観測の value を使う）を既定にし、`terminal`（終端と同じ）も選べる | P5・P18。打ち切りを終端と同じに扱うと、value の目標が偏る。CleanRL は元の実装に合わせて `terminal` に当たる扱いを残しているので、比べられるようにする |
| U7 | 写しを持つ Actor を Train に使ったときの扱い | 方策の古さの最大が 2 以上なら、その rollout の更新の前に止める | P4。Agent は、どの Actor が Train かを知らない。重みが古いまま学ぶ構成を、黙って通さない |
| U8 | 評価の発火 | 今の仕組みのまま（rollout の更新の `LearnEvent` で発火）。間隔は `learn_step` で数え、区切りをまたいだ最初の更新で発火する | P2・P28。評価を exp step で駆動する口を足す案は NG2 |
| U9 | 評価の Actor | `[eval]` は Greedy、`[eval_sample]` は抽選。`[eval_target]` は休止 | 起動元の案（§13.2 のメトリクスの「成績」）。Greedy は方策の中心、抽選は学習中の方策そのものを見る |
| U10 | rollout に入れる観測 | Actor が Network へ渡した入力を、行動の情報で Learner へ運び、device 上にためる。Learner は経験の観測を使わない（次の観測の value だけ、経験から作る） | P6・P13・P16。観測を 2 回 device へ送らずに済む。Actor が作った入力そのものをためるので、収集時の log 確率と、更新の前の再計算が一致する。DefaultDQN の「Perception そのものは運ばない」（P16）は、Frame を Replay に保存する構成の決定で、PPO 系は Replay を持たない |
| U11 | Forwarder の並びの結線 | 今回は結線しない。Network の入力は、device へ移した観測の各キーに、長さ 1 の stack 軸を足したもの | P10・P25。仕上げの部品が DQN 系の中にあり、今回は移せない（NG1）。CartPole は重ねも正規化も要らない。長さ 1 の軸を足すのは、あとで並びを結線して重ねる枚数を増やしても、Network の入力の形の意味が変わらないようにするためである |
| U12 | 用語「rollout」 | 用語集の「rollout（補助学習）」を「rollout」に広げ、PPO 系の rollout と同じ語にする | P21。どちらも「学ぶ前にためる、続いた intake のまとまり」で、別の語にすると同じ概念に名前が 2 つになる |
| U13 | SAC をこの系統に入れるか | 入れない。別の系統にする | SAC は Replay と Q の Network を使い、rollout と advantage の推定を使わない（M1 の C.6）。分け合うのは分布だけで、分布は core にある |
| U14 | 行動の分布の共通の口（抽象の基底） | 今回は作らない。`CategoricalDistribution` を具象のまま使う | 実装が 1 つしか無い。連続行動を足すときに、entropy を持たない分布（tanh で縮めた Gaussian）を含めて口を決める |
| U15 | メトリクスの群の番号と名前 | `78_agent_ppo`（更新）と `79_agent_policy`（方策と収集）。群の中は十の位で分類する | P20。agent の帯で空いている番号を使う。群を 2 つにとどめ、中の番号で分類する |
| U16 | メトリクスのプロファイルの置き場と連鎖 | `metrics_scalar.txt` に `metrics.scalar.@ppo` を置き、PPO の env の設定が選ぶ。`@baseline` には足さない | `@baseline` は DQN 系の鍵を含み、PPO 系で選ぶと、見つからない鍵の WARN が並ぶ |
| U17 | DQN 系の部品を core へ移すか | 今回は移さない。§10 の表を残す | P24。開発者の発言 2 は「DQN を移す」とは言っていない |
| U18 | 機能カテゴリと Topic | `326_ppo`（326 AGENT個別実装 PPO系列）の行を立て、Topic は 19（323_AGENT個別実装）を当てる。PPO 系列の Topic Issue は起こさない | Topic Issue を起こすのは人間の操作である。コミットの対象名の一覧に `PPO` を足すかも、あわせて決める |
| U19 | CartPole の受入の測り方 | `CartPoleEnv.limit_step = 500`、seed 3 つ、exp step 500,000、Greedy の評価の `mean.episode_steps` が 500 に届き、最後の 100,000 step の評価の点の 8 割以上が 500 | P19・P23。anet の CartPole は報酬に罰があるので、収益ではなく step の数で測る。予算は CleanRL の既定（P18） |
| U20 | 出荷の既定値 | CleanRL の `ppo.py` の既定（§2.7）。CartPole は lane 8、rollout 128、ミニバッチ 4、4 epoch。学習率を線形に 0 へ下げる schedule は、`@baseline` には置かず、CartPole の設定が書く | P18。公開の実装で CartPole の結果がある組み合わせ。lane は、評価の間隔を取りやすい 8 にする。schedule の終わりの step は Run の長さで決まるので、Agent の共通の既定には置けない |
| U21 | 時間の値にする係数 | 学習率、比の clip の幅、entropy の係数。γ・λ・value の係数は定数 | 参照実装が schedule にするのは学習率と clip の幅（SB3）で、entropy の係数は RLlib が schedule にする（M1 の A.6） |
| U22 | 更新結果の読み出しの手段 | 統計を 1 本にまとめ、最初の `GetScalar` で 1 回だけ CPU へ写す。`HostReadback` には載せない | 読むのは更新の直後の同じスレッドで、あいだに進める処理が無い。P13 のとおり、載せても同期の転送と同じである |
| U23 | KL で止める形と、閾値の既定 | `none` と `epoch`（CleanRL の形）。SB3 の形（ミニバッチごと、1.5 倍）は入れない。`learner.kl_stop.epoch.threshold` の出荷の既定は 0.015 | P18。ミニバッチごとの判定は、ミニバッチごとに device から CPU への転送が要る。CleanRL と SB3 は閾値の既定を持たない（§2.7）ので、0.015 は、OpenAI Spinning Up の PPO が方策の更新をやめる値（`target_kl` の既定 0.01 の 1.5 倍。§15）に合わせた |
| U24 | 設計文書の置き場 | core の部品は新設の 115、PPO 系は新設の 230 | 起動元の指示（§13.2）。115 は 110（Agent と学習）の次、230 は 220 の次の空き |
| U25 | 保存の範囲 | Network と optimizer を保存し、`auto_load_file` で読む。Runner の checkpoint 引き継ぎで `PPOAgent` を続ける使い方は、受入で確かめない | ImageCls と同じ範囲。退けた案は、引き継ぎの受入を入れること。制約ではなく、CartPole の受入に要らないので入れない |
| U26 | ADR | 起こさない | 暫定の起票で、決定が確定していない。再グリルで U4・U10 が確定したときに、ADR の 3 条件を当てる |
| U27 | 担当 | 起票・実装とも、この RFD を起こした枠（Claude Code、Fable 5.2）。再グリルは別の枠 | 開発者の発言 4・9（P22） |
| U28 | 強制行動 | Actor は、方策の選択のあとに強制行動を反映し、実行された行動の log 確率を記録する | P26 |
| U29 | 実装のステージ分け | 分けない | 開発者は分け目を求めていない |
| U30 | 勾配の clip の上限の設定キーの名前 | `learner.grad_clip_max_norm` | 同じ役の既存のキーは、DQN 系の `learner.grad_clip_tau` と、ImageCls の `grad_clip_max_norm` の 2 つがある。`tau` は、soft update の係数、方策温度、taus（τ サンプル）にも使う語なので、何の値かが名前から読める ImageCls の語にそろえる（[impl.md](../quality/impl.md) の「名前の付け方」の 2・3）。対になる `learner.use_grad_clip` は、DQN 系と同じ名前である |

## 13. グリルの材料

### 13.1 開発者の発言（原文）

2026-11-23:

1. 「DQN系（価値ベース）みたいに、方策ベースのAgent実装として今後いろいろと育てていきたい。とりあえずPPOって名前にしといて後で考えれば十分かな？PPO系、って感じ？まずはPPO離散のみ、ただし今後の拡張性を最大限に考慮した形。」
2. 「PPO実装は、algo側だけでなく、フレームワーク側で用意すべき共通部品の整理の意味合いもあります。使う側を作るのではなく、coreとalgoの両方を同時に育てます。」
3. 「Networkは設定次第で別でも共有層ありでも、どちらも出来ように。Network自体をわけるかブランチを分けるかだけど、なんとなくブランチ構成の設定だければ分ければ良いんじゃないかな」
4. 「Fableで基礎設計だけして細かいところはお任せしたい感じ。眠いからビルドも任せたいな」

2026-11-25:

5. 「アルゴリズムの学習も兼ねるので処理コメントは多めが良いです。」
6. 「ソースファイルを分ける単位はDQNと同じくらい。」
7. 「GPUCpU両対応、CPU同期は必要最小限。」
8. 「メトリクスはPPOのアルゴリズム評価として必要なものを全て入れる。」
9. 「仕事で昼休み以外ほぼ動けないので止めずに進めて欲しい」（問い「PPO の RFD ができたあと、RFD で止めるか、止まらず進めるか」への答え）

### 13.2 起動元のセッションが置いた案と範囲（開発者の了承は無い）

- 了承のある範囲: RFD のあと、止まらずに実装・ビルド・テスト・レビューまで進めること。全ビルドをセッションが回すこと。
- 了承の無い範囲: Runner で Run を起動すること（試運転を含む）。DQN 系（`src/algo/dqn/`）の動きや構成を変えること。git の操作。
- 受入の案: CartPole 離散で 500（M3 から借りた）。測り方は RFD で決める。
- 拡張性の取り方の案: あとで足すものは差し込み位置と作る条件を書き、実装は離散の PPO と、それに要る共通部品だけにする。
- コメントの案（発言 5）: 実装のコメントに、式と変数の対応、その段階が要る理由、参照実装と変えた所を書く。
- メトリクスの候補（発言 8）: 損失、更新の大きさ（approx KL、clip の割合、比の統計）、価値（explained variance ほか）、advantage、方策（entropy、行動ごとの確率と頻度）、勾配（内訳と clip）、schedule の実効値、収集（終端と打ち切り、方策の古さ）、成績（Greedy と抽選の評価）、時間。
- 設計文書の指示: 新しい系統と共通部品には、設計文書を新設する。

### 13.3 材料

- M1: 既存の RL フレームワーク（SB3・TorchRL・RLlib・Acme・CleanRL・Tianshou ほか）の共通部品の切り方の調査（2026-11-23。起動元の作業用の置き場にあり、リポジトリには無い）。本 RFD が使った事実は、P17（U4 の理由が引く）と、U2・U13・U21 の理由に書いた。参照実装の式と既定値（§2.7）は、M1 からではなく、原文から取り直した。
- M2: anet の core と algo の下調べ（2026-11-23。同上）。本 RFD が使った事実は §2 に、コードの位置つきで書き直した。
- M3: StampFly の計画（`docs/StampFly.md`。2026-11-25 の時点で、main の作業ツリーにある未追跡の文書）の §5.1.0（真値と Observation の分離）と §7.3（RFD A・B の分割）。離散の PPO の受入「CartPole 離散で 500」と、Actor と Critic の別の入力を最初から入れる要求は、ここによる。

### 13.4 再グリルで先に見る点

- U4（Network の形）と U10（rollout に入れる観測）は、あとの拡張の形を決める。
- U2・U17（core の範囲と DQN 系の移行）は、開発者の発言 2 の読み方に依る。
- U19（受入の測り方）は、Run の段取りの承認と合わせる。
- 欠けた前提: なし（測定や試作の結果が要る前提は置いていない）。

## 14. 依存と順序

- [RFD0200](0200_forwarder_public_interface_rework_10rfd.md)（Forwarder の公開の口の作り直し）: 実装済み。本 RFD は Forwarder を使わない（NG4）ので、触る箇所が重ならない。
- [RFD0142](0142_agent_prev_action_obs_10rfd.md)（直前の行動を観測に足す）: まだ実装していない。`BatchActionInfo` の `info` を使う点が同じだが、鍵が別で、DQN 系の中の変更なので、どちらが先でも当たらない。
- [RFD0108](0108_metrics_tag_numbering_10rfd.md)（メトリクスの tag の番号）: まだ実装していない。本 RFD の群（78・79）は、同 RFD の割り当ての空きに置いた。同 RFD を先に実装しても、あとに実装しても、番号は当たらない。
- [RFD0201](0201_top_episode_memory_sil_lower_bound_10rfd.md)（上位エピソード記憶と SIL）: まだ実装していない。DQN 系と Replay の変更で、触る箇所が重ならない。
- 本 RFD の中の順: core の部品（§7.1〜§7.3、`GetAll`）→ PPO 系 → 設定 → 設計文書。

## 15. 参照

- CleanRL `ppo.py`: <https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py>（2026-11-25 に取得）
- Stable-Baselines3 `ppo.py`・`on_policy_algorithm.py`・`buffers.py`: <https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3>（2026-11-25 に取得）
- OpenAI Spinning Up `ppo.py`（PyTorch 版）: <https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py>（2026-11-25 に取得。`target_kl=0.01`、`if kl > 1.5 * target_kl:` で方策の更新をやめる）
- Schulman ら「Proximal Policy Optimization Algorithms」（2017）、「High-Dimensional Continuous Control Using Generalized Advantage Estimation」（2016）
- Huang ら「The 37 Implementation Details of Proximal Policy Optimization」（ICLR Blog Track 2022）
- [011 ソース配置とライブラリの境界](../design/011_source_layout.jp.md)、[110 Agent と学習](../design/110_agents_and_learning.jp.md)、[130 ニューラルネットワーク](../design/130_neural_networks.jp.md)、[135 Forwarder と Perception](../design/135_forwarder_and_perception.jp.md)
- [RFD0134](0134_network_lock_audit_10rfd.md)、[RFD0161](0161_observation_pipeline_10rfd.md)、[RFD0182](0182_source_layout_10rfd.md)、[RFD0092](done/0092_gradient_noise_scale_10rfd.md)
