# RFD0128: DQN 系 Agent まわりのリファクタリング 5 件をまとめて片づける（価値の分解の前に） RFD

> 2026-11-12 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §8 の未決表に置いた。§5〜§7 は §8 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: 2026-11-12 の [PRD121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)（価値の分解）のグリル。開発者は、PRD121 と同じ場所を触るリファクタリングの課題 5 件を、PRD121 の実装より先に片づけると選び（PRD121 の D18）、5 件を 1 本の暫定 PRD にまとめるよう指示した。直し方の中身について、会話で出た方向は無い。
> 題材: §9 の課題 5 件。
> 関連: [PRD121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)・[PRD173](0173_intrinsic_reward_novelty_10rfd.md)（同じ場所へ変更を足す PRD）、[PRD125](0125_dqn_learner_update_skeleton_10rfd.md)・[PRD122](0122_small_refactor_and_test_gap_fixes_10rfd.md)（挙動を変えない直しと、短い Run の前後比較の前例）、[PRD127](0127_eval_trace_intrinsic_reward_columns_10rfd.md)（同じファイルを触る。実装済み）、[PRD059](done/0059_config_concept_tree_alignment_10rfd.md)（D22 で RainbowAgent を残すと決めた）・[ADR 0030](../adr/0030-nn-config-tree-under-agent-prefix.md)、[PRD165](0165_nn_head_projection_separation_10rfd.md)（Head の射影の分離）、[PRD170](0170_nn_trace_capture_unification_10rfd.md)（Learner の購読の見直し案）、[PRD161](0161_observation_pipeline_10rfd.md)（観測パイプライン。同じコンストラクタを触る）、[PRD182](0182_source_layout_10rfd.md)（ソースの配置）、[リファクタリング観点](../quality/refactor.md)。

> 2026-11-13 改訂: 起票とは別の枠でグリルを行って暫定の印を外した。起票時の未決表の 21 行をすべて仕分け、確定事項（§5）・非ゴール（§1.3）・現状（§2）・クラス一覧（§6.5）・未決表（§8）へ移した（対応は §5.2）。§5〜§7 は書き直し、節の中の番号は付け直した（起票の行にある節の番号は当時のもの）。起票のあとに分かった前提を反映した。(1) PRD127 は実装済みで、内的報酬のメンバが 7 つになり、Evaluation Trace の採点がボーナスの式を学習の入口とは別に書いている（§2.1）。(2) RainbowAgent は、Rainbow 相当まで設定の柔軟性を意図して落とした実装で、DQN 系の共通部品の責務分担を確かめるためだけに残しており、Run では使わない（開発者の説明。D2）。(3) 名前付きの seed は作る順に依らない（§2.5）。起票時の未決 11 の理由「seed の派生の順に触る」を直した。(4) `ExperienceSamples::To` の stream guard は効き目が無い（§2.2）。(5) 接頭辞の下を写して記録するキーの形は 4 か所とも同じで、違うのは写す先の形だった（§2.4）。起票時の未決 15 と §2.4 の記述を直した。(6) 方策と τ の検証の関数が DefaultDQN のヘッダーにあり、Rainbow から呼べない（§2.3）。(7) prd116 の検証資材は内的報酬のメンバを名指していなかったので、移行対象から外した（§6.6）。開発者に聞いて決めたのは、Rainbow の食い違い 5 点の扱い（D4）、細部 10 件を推奨の既定で埋めること（D4・D9・D15〜D19 に書いた）、probe 用の乱数を使う側の Learner が作って持つこと（D13）、設定のエラー文の書き方をそろえること（D6）である。D13 と D6 は、別モデル（Fable）の検討の論点を受けて聞いた。同じ検討から次を直した。(8) D1 の「残りの 3 件は互いに独立」は誤りで、24d885 を 72d640 の前へ移した。(9) D4 の (d) の理由（`nn_viz` への依存）は誤りだった。(10) Rainbow は共通部品へ渡す入力の多くを今は渡していない（§2.3）。共通部品はそれを任意の入力として受け、それでも Rainbow で変わる点を §6.2 に足した（D3）。Rainbow の制限を、設定の DTO を直接渡す経路でも守る（D4）。(11) 内的報酬のクラスはボーナスを返し、報酬へ足すのは Agent の側にした（D11）。(12) 診断の共通の読み出しを毎更新に値を出す群まで広げ、指標を名前で指して、並びと参照が食い違えばビルドが止まる形にする（止め方は実装計画で決める。D12・D19）。§7 の 9 の検索語に当たらない直書きがあった（§2.5）。(13) PRD121 の足す先を §6.7 の表で確かめる（D24）。(14) Rainbow の学習と、学習を止めた構成の特性テストを足した（§6.3）。工程の終わりに、§1.3 の 3 行の記録先（課題 f9790d への追記、課題 6f954f の新設、wiki への見送り）を書き、§8 の U2 に ERROR の行が出たときの扱いを足した。同じ日に別の枠で起票された PRD129 との関係を §10 に足した。
> 2026-11-14 改訂: §1.3 の表の 2 行で、課題に記録した先を直した。「ボーナスの診断の `.item<float>()` 6 回を 1 回の転送にまとめること」の行は、課題 f9790d の「対応内容」から「対応方針」へ、「Learner の購読を「開始前の宣言」と「呼び出しごとの要求」に分ける見直し」の行は、課題 24d885 の「対応内容」から「対応方針」へ直した。課題への追記の置き場を中身で決める規則（anet-create-issue の「起票のあとの追記」）に合わせて、課題の側で行を移したためである。決定と範囲は変えていない。

> 2026-11-14 改訂: 実装前の特性テストで U2 を確認した。CPU の転送で actions・target_returns・next_state.terminals のそれぞれを未定義にすると例外になり、ERROR のログは出なかった。§6.2 の変更前の挙動と §8 の U2 に結果を反映した。D7 の未定義保持への変更と、そのほかの契約は変えていない。

> 2026-11-14 改訂: D6 の変更に伴い期待値を変える既存のテストは、skip forward mode と policy churn 設定の 2 本とする。policy churn 設定も旧表記の value=0 を検査していたため、§2.4・D6・§6.2・§6.6 の移行対象の記載を補った。エラー文をそろえる契約と、意図して変える範囲は変えていない。

> 2026-11-14 改訂: 追加レビューで見つかった §2.2 の U2 の未確認表記を実測結果へそろえ、D7 の理由も確認と記録を済ませた表記へ直した。決定と実装の契約は変えていない。

## 1. 目的と問題

### 1.1 問題

2026-11-10〜11 のリファクタリング候補の抽出で、DQN 系の Agent まわりに、重複と散らばりが 5 件見つかった（§2）。どれも、価値の分解（RFD0121）が変更を足す場所にある。

| 場所 | RFD0121 が足すもの | 今のままだと |
|---|---|---|
| 取り込みの内的報酬の扱い | 内的報酬用の報酬スケーラ、成分での受け渡し、保存する状態、Evaluation Trace のボーナスの列の式 | 7 つのメンバと、学習の入口・保存・読み込み・値の読み出し・Evaluation Trace の採点に散った扱いが、さらに膨らむ。ボーナスの式は 2 か所を直す |
| 標本のフィールドの並び | 成分ごとの収益 | フィールドを 1 つ足すたびに 4〜5 か所を直す |
| Head の選択と Agent の構築 | 合成 Head | DefaultDQNAgent と RainbowAgent に写した分岐の、食い違いが増える |
| 設定の読み取り | 設定キー 10 個ほど | 534 行の 1 つの関数に、読む行と検証する行が離れて足される |
| 診断指標の群 | Head ごとの損失などのメトリクス | 群を足すたびに 12 か所前後を直し、指標を番号で指す箇所が増える |

### 1.2 ゴール

- **G1**: §9 の課題 5 件が片付く。DefaultDQNAgent の学習の挙動（メトリクス、Agent が記録する設定、ログの文言と回数、checkpoint の形式、乱数の消費の順）は変えない。意図して変える点は、§6.2 に挙げたものに限る。
- **G2**: RFD0121 の実装が、取り込み・標本・Head の選択・設定の読み取り・診断の群のそれぞれで、1 か所へ足せば済む形になる。
- **G3**: 挙動を変えていないことを、実装前に保存した実行体との前後比較で、ステージごとに確かめられる。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 理由 | 本当の制約か | 作る条件 |
|---|---|---|---|
| RFD0121 の機能（価値の分解、報酬の成分、報酬スケーラの種類）を足すこと | 機能の追加と、挙動を変えない直しを 1 本に混ぜると、前後の比較で差が出たときに原因を分けにくい（RFD0121 の D18） | 制約ではない | RFD0121 が本 RFD のあとに足す |
| RainbowAgent を消すこと | DQN 系の共通部品の責務分担を確かめる 2 つ目の利用者として残す（D2。[RFD0059](done/0059_config_concept_tree_alignment_10rfd.md) の D22） | 制約ではない（意図した保持） | — |
| ボーナスの診断の `.item<float>()` 6 回を 1 回の転送にまとめること | 性能の変更で、構造の直しとは別の関心。速さを測ってから決める（課題 08d557 の対応方針） | 制約ではない | RND の時間の内訳（[課題 f9790d](../management/issues/issue-20261112-f9790d.md)）と合わせて測ってから決める。この案は、課題 f9790d の対応方針に材料として記録した |
| Learner の購読を「開始前の宣言」と「呼び出しごとの要求」に分ける見直し | 購読の口を変える別の関心で、G1・G2 に要らない（[RFD0170](0170_nn_trace_capture_unification_10rfd.md) §8 の 1） | 制約ではない | 購読の口を変える要求が出たとき。材料は課題 24d885 の対応方針と RFD0170 §8 の 1 にある。24d885 が解決すると材料が埋もれるので、この案は [課題 6f954f](../management/issues/issue-20261113-6f954f.md) に記録した |
| Head の射影の分離 | 別の関心で、[RFD0165](0165_nn_head_projection_separation_10rfd.md) が持つ | 制約ではない | RFD0165 の実装 |
| ソースの配置（ファイルの分け方）を変えること | 配置は別の課題が扱う（[リファクタリング観点](../quality/refactor.md) の「候補にしないもの」、[RFD0182](0182_source_layout_10rfd.md)） | 制約ではない | RFD0182 の実装 |
| RFD0165・RFD0161 が入るのを待ってから始めること | どちらも未実装の暫定 RFD で、待つと RFD0121 の待ちが延びる | 制約ではない（順序の選択） | — |
| ImageClsAgent が持つ同じ形の診断の要求を、共通の型に載せること | 題材の 5 件の範囲の外で、前後比較の組が ImageCls を通らない（D22） | 制約ではない | ImageCls に診断の群を足すとき。課題にはせず、再開の条件を wiki の [321 Agent 共通基盤](../wiki/categories/321_agent_core.md) の「選ばなかった形」に残した（2026-11-13 に開発者が見送りを選んだ） |

### 1.4 足す部品とゴールへの寄与

新しい機構は足さない。足す部品は、散った処理を寄せる先の 3 つである（§6.5）。

| 部品 | 寄せる処理 | ゴールへの寄与 |
|---|---|---|
| 内的報酬の扱いをまとめるクラス | DefaultDQNAgent に散った内的報酬の処理と、Evaluation Trace の採点 | G1（08d557）、G2（取り込み） |
| DQN 系の共通部品の、構築の部品 | DefaultDQNAgent と RainbowAgent が写している Head の選択、方策と Actor の生成、Learner の生成、値の振り分け | G1（72d640）、G2（Head の選択） |
| 診断指標の群の共通の型 | 6 つの群が同じ形で書いている購読の解析・間隔・発火・要求と、毎更新に値を出す群も含めた値の読み出し | G1（24d885）、G2（診断） |

### 1.5 成功の測り方

- §7 の受入 1〜3 が、5 つのステージのそれぞれの終わりで一致する。
- §6.7 の表で、RFD0121 が足すものごとに、本 RFD のあとの足す先が 1 か所ずつ挙がり、その場所が実装のあとのコードにある（§7 の 13）。RFD0121 の実装計画は、その足す先を使う。
- RainbowAgent の構築が、DQN 系の共通部品と Rainbow の制限（D2）だけで書かれている（§7 の 7）。
- 先送りしたもの（§1.3 の `.item<float>()`、購読の見直し、ImageClsAgent）は、どれも別の関心として範囲から外した。作る条件は §1.3 の表に書いた。

## 2. 現状（2026-11-12〜13 にコードで確認）

行番号は 2026-11-13 の作業ツリーの位置である。同じファイルを触る作業が並行しうるので、実装時は関数の名前と近くの記述で探し直す。

### 2.1 内的報酬の扱いが散っている（08d557）

- 内的報酬だけのメンバが、`DefaultDQNAgent` に 7 つ並ぶ（[default_dqn_agent.hpp:728-734](../../src/algo/dqn/src/default_dqn_agent.hpp)）。推定器、β、直近の β（RFD0127 で足した）、補助学習の credit、成功回数、lane の抽選の乱数、診断値である。
- `DefaultDQNAgent::UpdateFromBatch`（[default_dqn_agent.cpp:838](../../src/algo/dqn/src/default_dqn_agent.cpp)。内的報酬は 851〜907）の中に、ボーナスの計算、診断値の公開、credit の加算、lane の抽選、補助学習のループが並ぶ。
- 推定器の診断値を写すループが、構築・読み込み・更新の 3 か所にある。
- 保存（`Save`、[同 334](../../src/algo/dqn/src/default_dqn_agent.cpp)）と読み込み（`LoadNetwork`、[同 385](../../src/algo/dqn/src/default_dqn_agent.cpp)）で、書く順と読む順を手で合わせている。
- lane の抽選の乱数だけ標準ライブラリの `std::mt19937_64` で、同じクラスのほかの乱数は `anet::RandomGenerator` である。
- RFD0127 の実装（コミット 72699639）で、Evaluation Trace 用の写し（推定器の複製と直近の β。Actor が network を写し直すロックの中で作る）と採点が、同じファイルの無名の名前空間に足された（[同 693-830](../../src/algo/dqn/src/default_dqn_agent.cpp)）。採点はボーナスの式（β×(α−1)、終端の lane は 0）を、学習の入口（[同 855-861](../../src/algo/dqn/src/default_dqn_agent.cpp)）とは別に書いている（[同 801-817](../../src/algo/dqn/src/default_dqn_agent.cpp)）。RFD0121 はこの 2 つの式をどちらも変える（[RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md) の §8.4 の「127 との調整」）。
- 補助学習の刻みと抽選は、推定器ではなく外側の Agent（facade）が持つと設計文書が定めている（[200_dqn_agents.jp.md:404](../design/200_dqn_agents.jp.md)）。

### 2.2 標本のフィールドの並びが分かれている（288879）

- `ExperienceSamples::To`（[rl.cpp:773](../../src/core/base/src/rl.cpp)）は、CUDA へ送る経路（777〜810）と CPU の経路（812〜840）で、全フィールドの並びを 2 回書いている。
- 2 つの並びは 3 点で食い違う。
  - 未定義の tensor の扱い。CUDA の経路は未定義を素通しする `anet::To`（[tensor_util.hpp:103](../../src/core/base/include/anet/tensor_util.hpp)）を使う。CPU の経路は `actions`・`target_returns`・`next_state.terminals` に素の `.to()` を呼び、`n_steps`・`is_weights` だけを場合分けする。
  - `replay_item_keys`・`per_priority_sources` を CPU に置く書き方が違う。
  - CUDA の経路だけ stream guard を置く。
- stream guard は効き目が無い。今の device の今の stream を取り（libtorch 2.11 の `getCurrentCUDAStream()` は、device を渡さなければ今の device の今の stream を返す。`c10/cuda/CUDAStream.h` の 220〜226 行）、それを `CUDAStreamGuard` に渡している（`CUDAStreamGuard` は、渡した stream の device を今の device にし、その device の今の stream を渡した stream にする。`c10/cuda/CUDAGuard.h` の 148〜151 行）。どちらも元の値と同じなので、何も変わらない。
- 今の作り手（Replay）は `actions`・`target_returns`・`next_state.terminals` を必ず作る（[replay_buffer_impl.cpp:1488-1490](../../src/core/base/src/replay_buffer_impl.cpp)）。2026-11-14 の実装前の特性テストで、CPU の経路へこの3フィールドを1件ずつ未定義にして渡すと、いずれも例外になり、ERROR のログは出ないことを確認した（§8 の U2）。
- `ForEachTensor`（[rl.hpp:889](../../src/core/base/include/anet/rl.hpp)）も全フィールドを並べる。`ToString`（[rl.cpp:843](../../src/core/base/src/rl.cpp)）は一部だけを並べ、`window` の 6 つと `sampled_rewards`・`sampled_window_rewards` が無い。`ToString` を呼ぶのは、コメントアウトした 2 行だけである（[dqn_based_agent.cpp:4076・4080](../../src/core/base/src/dqn_based_agent.cpp)）。
- 学習の毎回の転送の経路である。

### 2.3 Agent の構築と、RainbowAgent の写し（72d640）

- `DefaultDQNAgent` のコンストラクタ（[default_dqn_agent.cpp:53](../../src/algo/dqn/src/default_dqn_agent.cpp)）の 1 つの関数に、seed、内的報酬、スケーラ、Head の Factory の選択、forward 設定、spec、skip の定義、モデル、構造図、target の方策、Learner、読み込みが並ぶ。
- Head の選択（[同 146-175](../../src/algo/dqn/src/default_dqn_agent.cpp)）は、分布の表し方 3 値と Dueling の有無の組を、入れ子の分岐で書く。Learner の構築 3 つ（[同 261-290](../../src/algo/dqn/src/default_dqn_agent.cpp)）は、型を除いて引数が同じである。
- `RainbowAgent`（[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp)）が、Head の選択（62〜83）、Learner の構築（103〜111）、方策と Actor の生成（172〜189）、値の振り分け（131〜170）、更新を写している。課題は、5 点の食い違いを挙げている。(a) 方策と τ の検証を呼ばない。(b) 方策の種類を見ずに、常に ε-greedy を作る。(c) 設定の記録の仕方が違う。(d) `net.body` と構造図の記録が無い。(e) `UpdateFromBatch` の公開範囲が違う。
- RainbowAgent は、Rainbow 相当まで設定の柔軟性を意図して落とした実装である。原論文の 6 つの拡張のうち、分布表現は未実装の C51 の代わりに QR を使い（`use_qr` の真偽で QR か通常の Q を選ぶ。IQN は選べない）、NoisyNet の代わりに ε-greedy を使う（[200_dqn_agents.jp.md:94](../design/200_dqn_agents.jp.md)）。Head の射影の設定を渡さないので Noisy も選べない。報酬スケーラ・観測の正規化・stacker・`nn_viz` を持たず、target の方策は Greedy に固定である。
- RainbowAgent は、DQN 系の共通部品を 2 つ目の Agent から使い、責務の分け方がおかしくないかを確かめるためだけに残している。Run では使わない（2026-11-13 の開発者の説明）。[RFD0059](done/0059_config_concept_tree_alignment_10rfd.md) の D22 も「Agent 柔軟性の実証として保持」と決め、[ADR 0030](../adr/0030-nn-config-tree-under-agent-prefix.md) がそれを前提にしている。構築するのは Factory の登録（[init.cpp:175](../../src/core/base/src/init.cpp)）とテストだけで、出荷の設定で選ぶ行はコメントである（[common.txt:29](../../apps/runner/config/common.txt)）。
- (b) は、Rainbow が Actor の `policy` を設定から読み（[rainbow_agent.hpp:45](../../src/algo/dqn/src/rainbow_agent.hpp)）、`policy_type` を見ずに ε-greedy を作る（[rainbow_agent.cpp:178](../../src/algo/dqn/src/rainbow_agent.cpp)）ことによる。`policy_type = UQE` も綴り違いも、何も言わずに ε-greedy で動く。[AGENTS.md の「設定値の扱い」](../../AGENTS.md)（指定値を別の値に置き換えない、不正値は ERROR で落とす）に反した状態である。DefaultDQN は ε-greedy を `EpsilonGreedy` と数字の別名 `0` で受け付ける（[default_dqn_agent.cpp:302](../../src/algo/dqn/src/default_dqn_agent.cpp)）。数字の別名の受け付け方が関数ごとに違うことは、[課題 17fea4](../management/issues/issue-20261111-17fea4.md) が扱う。Rainbow のテストは、設定の DTO を直接作り、Actor の方策に既定（`EpsilonGreedy`）を使う（[dqn_based_agent_test.cpp:758](../../src/core/base/src/dqn_based_agent_test.cpp) ほか。2026-11-13 に `RainbowAgent` と `policy_type`・`tau_rule` で引いて確かめた）。
- (a) の検証の関数 `ValidateActionPolicy`・`ValidateTauRule` は、DefaultDQN のヘッダーにある（[default_dqn_agent.hpp:30-71](../../src/algo/dqn/src/default_dqn_agent.hpp)）。検証する型 `ActionPolicyConfig`・`TauRuleConfig` は DQN 系の共通のヘッダーにある（[agent.hpp:90-103](../../src/core/base/include/anet/agent.hpp)）。Rainbow は、DefaultDQN に依存しない限り、この検証を呼べない。検証の中身は、どれもその型の中の値だけで正誤が決まる（`policy_type` の値、`tau_rule` の件数と取り方、UQE・Thompson と型の中の `quantile_mode` の組、`full_distribution_query` と方策の組）。
- (c) の違い: DefaultDQN は `MetricsLogger::Log(const Config&)` で、metrics の json の記録と、Run の `config/DefaultDQNAgent.txt` の両方を書く（[metrics_logger.cpp:665](../../src/core/base/src/metrics_logger.cpp)）。Rainbow は json を渡す口で書くので、json の記録だけになり、そのあと `Flush()` を呼ぶ。
- (d) の記録: DefaultDQN は、`net.body` を Network の設定から（[default_dqn_agent.cpp:71](../../src/algo/dqn/src/default_dqn_agent.cpp)）、`net.structure` を既定の可視化の設定から（同 233〜234）、`net.detail` を `nn_viz` から（同 235〜236）書く。`nn_viz` を使うのは `net.detail` だけである。
- Rainbow の値の振り分けは、ReplayBuffer の接頭辞を `"replaybuffer."` と直書きし、DefaultDQN は `ReplayBuffer::kKeyPrefix` を使う。
- RainbowAgent は、DefaultDQN が同じ処理へ渡している入力の多くを、今は渡していない（2026-11-13 に確認）。Learner へ、観測の正規化・stacker の設定（コードの名前は `stucker`）・learner の seed・probe 用の乱数・forward の資源を渡さず、Learner の生成のログに Munchausen の接尾が無い（[rainbow_agent.cpp:104-112](../../src/algo/dqn/src/rainbow_agent.cpp)。DefaultDQN は [default_dqn_agent.cpp:261-290](../../src/algo/dqn/src/default_dqn_agent.cpp)）。target の方策は、引数の既定（lane 数 0・CPU。[dqn_based_agent.hpp:642-645](../../src/core/base/src/dqn_based_agent.hpp)）で作る（[rainbow_agent.cpp:101](../../src/algo/dqn/src/rainbow_agent.cpp)）。DefaultDQN は Agent の lane 数と device で作る（[default_dqn_agent.cpp:251](../../src/algo/dqn/src/default_dqn_agent.cpp)）。値の振り分けで答えるのは schedule の鍵・`per_beta`・`replaybuffer.` だけで、DefaultDQN が答える内的報酬・`target_policy.`・`plasticity_probe_`・`noisy_`・報酬スケーラ・観測の正規化の鍵には答えない（[rainbow_agent.cpp:133-151](../../src/algo/dqn/src/rainbow_agent.cpp)、[default_dqn_agent.cpp:504-540](../../src/algo/dqn/src/default_dqn_agent.cpp)）。Actor は観測の正規化と stacker を持たず、既定の forward の設定で作る（[rainbow_agent.cpp:190-191](../../src/algo/dqn/src/rainbow_agent.cpp)）。購読の設定（`ConfigureScalarMetricSubscriptions`）を持たないので、診断の購読は Learner に届かない。
- `RainbowAgentConfig` は、分位の設定の検証を、設定の読み取りとコンストラクタの両方から呼ぶ（[rainbow_agent.hpp:84](../../src/algo/dqn/src/rainbow_agent.hpp)、[rainbow_agent.cpp:35](../../src/algo/dqn/src/rainbow_agent.cpp)）。設定の DTO を直接渡す経路でも検証を通すためである。DefaultDQN も、DTO を直接渡す経路のための検査をコンストラクタに持つ（[default_dqn_agent.cpp:61-65](../../src/algo/dqn/src/default_dqn_agent.cpp)）。
- RainbowAgent のテストは、設定の読み取り、構築、Actor と Evaluation Trace の生成までで、学習を回して値を確かめるものは無い（[dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp) の 4773・5516・5543・5566・8753・11531・11547 行付近。2026-11-13 に `RainbowAgent` で引いて確かめた）。

### 2.4 設定の読み取りが 1 つの関数に並ぶ（e33797）

- `DefaultDQNAgentConfig` のコンストラクタ（[default_dqn_agent.hpp:116](../../src/algo/dqn/src/default_dqn_agent.hpp) から）に、読み取り・検証・既定値の導出・記録が並ぶ。
- 「値が決まった語のどれかであること」と「数値が範囲に入ること」の検証が、同じ形で繰り返される。エラー文の書き方は、語で 4 通り、数値の範囲で 2 通りある（2026-11-13 に数えた）。語は、`'; expected A, B, or C.`（候補が 2 つなら `A or B`）が 6 か所、最後の「,」の無い `'; expected A, B or C.` が 1 か所（`learner.skip.feature_gradient`）、縦棒の `'; expected A|B|C.` が 3 か所（`learner.skip.target_window`、Actor の `skip.nn_forward_mode`・`skip.max_length_mode`）、「;」も「.」も無い `' expected one of: A, B, C` が 1 か所（`quantile_mode`）である。数値の範囲は、`; expected <範囲>.` が 4 か所、「;」も「.」も無い ` expected <範囲>` が 13 か所（probe の `batch_size` など）である。文言を確かめている既存のテストは 2 本（skip forward mode の縦棒と policy churn 設定の value=0。[dqn_based_agent_test.cpp:3270・7290](../../src/core/base/src/dqn_based_agent_test.cpp)）で、文言を引く文書は無い。
- 「接頭辞の下の設定キーをそのまま写して記録する」処理が 4 か所に手書きされている（intrinsic.rnd [同 141-148](../../src/algo/dqn/src/default_dqn_agent.hpp)、head.projection 155〜163、nn_forward 195〜202、skip.net と skip.head.projection 473〜483）。記録するキーの形は 4 か所とも同じで、Agent の接頭辞を除いた相対のキーである。違うのは写す先の形で、受け取る側の口に合わせている（skip.net は `NetworkConfig` が完全なキーと接頭辞を受け取る、Head の射影は末端のキーを受け取る、RND は設定全体と接頭辞を受け取る）。基底の `Config::RecordConfigValue`（[config.hpp:244](../../src/core/base/include/anet/config.hpp)）が、記録の 2 行と同じ本体を持つ。
- `net.block.` と `net.config_profile.` で始まるキーを skip.net の写し先へ写す処理（[同 487-489](../../src/algo/dqn/src/default_dqn_agent.hpp)）には、理由のコメントが無く、記録もしない。

### 2.5 診断指標の群が同じ形を繰り返す（24d885）

- 「指標の番号と間隔」の組を表す型が、群ごとに 6 つある（[dqn_based_agent.hpp:961・974・987・996・1042・1081](../../src/core/base/src/dqn_based_agent.hpp)）。可塑性は 3 つのチャネルと、最小の間隔で先に振り落とす判定を持つ（同 996〜1028）。attention は指標ではなく層で要求を持つ（同 987）。
- キー配列から番号を引く関数、発火した指標だけ要求の旗を立てるループ、読み出しの形も、群ごとに繰り返す。
- `Learner::ConfigureScalarMetricSubscriptions`（[dqn_based_agent.cpp:2750](../../src/core/base/src/dqn_based_agent.cpp)）は、購読の振り分け・設定値の検証・WARN・Resource の確保を 1 つの関数に並べる。Agent 側の `DefaultDQNAgent::ConfigureScalarMetricSubscriptions`（[default_dqn_agent.cpp:542](../../src/algo/dqn/src/default_dqn_agent.cpp)）も、replay fit・noisy・attention の購読を見て probe 用の乱数を作る。
- probe 用の乱数の渡し口は 2 つに分かれる。可塑性と policy churn は構築時に作って Learner のコンストラクタへ（[default_dqn_agent.cpp:93-96](../../src/algo/dqn/src/default_dqn_agent.cpp)）、replay fit・noisy・attention は購読されたときに作って購読の設定の引数へ渡す。どの乱数も名前付きの seed から作る。名前付きの seed は、基の seed と名前だけで決まり、作る順に依らない（[random.cpp:190](../../src/core/base/src/random.cpp)）。
- probe 用の乱数を使うのは Learner だけで、Agent は作って渡すだけである（[default_dqn_agent.cpp:93-96・565-599](../../src/algo/dqn/src/default_dqn_agent.cpp)）。Learner は、自分の seed（`learner` の名前の seed）を持つ乱数の持ち主として作られる（[dqn_based_agent.cpp:2632-2638](../../src/core/base/src/dqn_based_agent.cpp)）。設計文書は、probe 用の乱数を Agent が名前付きの seed から作って所有し、使う部品は参照だけを持つと書く（[110_agents_and_learning.jp.md:48](../design/110_agents_and_learning.jp.md)、[150_replay_buffer.jp.md:333・345](../design/150_replay_buffer.jp.md)、[200_dqn_agents.jp.md:594・763](../design/200_dqn_agents.jp.md)）。一方で、Agent の所有は Learner の配下に置いてもよいとも書く（110 の 49 行）。200 の 717・797 行と [ADR 0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md) の「caller-owned」は、ReplayBuffer から見た呼び出し側（Learner）が乱数を持つという意味で、Agent の所有を求めていない。
- Agent の購読の設定は、replay fit・noisy・attention の購読を Learner とは別に読み直し、乱数を作るかを決める（[default_dqn_agent.cpp:547-599](../../src/algo/dqn/src/default_dqn_agent.cpp)）。Learner の購読の設定も、同じ一覧を読んで要求を登録する（[dqn_based_agent.cpp:2766-2826](../../src/core/base/src/dqn_based_agent.cpp)）。同じ判断が 2 か所に書かれている。乱数が渡らないときの扱いは群ごとに違い、policy churn は購読を黙って捨て（同 2797）、noisy は購読の設定で ERROR（同 2863）、可塑性と attention は測る時点で ERROR（同 3127・3089）、replay fit は null のまま渡す（同 2953）。2 か所の判断が食い違うと、Run の途中で止まるか、黙って購読を捨てる。
- Learner の購読の設定は、冒頭で群の状態を作り直す（同 2755〜2762）。Agent は購読されたときの乱数を 1 回だけ作って持ち続ける（[default_dqn_agent.cpp:565・574・592](../../src/algo/dqn/src/default_dqn_agent.cpp)）。今の Run では、購読の設定は学習の前に 1 回だけ呼ばれる（[trainer.cpp:927](../../src/core/base/src/trainer.cpp)）。probe 用の乱数は checkpoint に入らない。probe は、学習の乱数・PER・正規化の統計・network の状態に触れない（[150_replay_buffer.jp.md:331](../design/150_replay_buffer.jp.md)、[200_dqn_agents.jp.md:749・763](../design/200_dqn_agents.jp.md)）。
- 学習を止めた構成（`learner.enabled = false`）の分岐が、内的報酬（[default_dqn_agent.cpp:509・851](../../src/algo/dqn/src/default_dqn_agent.cpp)）と、診断の群（[dqn_based_agent.cpp:2774・2791・2861](../../src/core/base/src/dqn_based_agent.cpp)。policy churn と可塑性はこの条件を見ない）にある。
- 毎更新に値を出す群（skip・Munchausen）は、間隔の gate を持たず、キー配列と 1 次元の pack で値を運び、群ごとの関数で読み出す（[dqn_based_agent.hpp:34-41](../../src/core/base/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp:5395-5446](../../src/core/base/src/dqn_based_agent.cpp)）。RFD0121 が足すメトリクス（Head ごとの損失など）も、毎更新に出す値である（[RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md) の §8.4）。
- 指標を番号で直書きしている。schedule の鍵の 3〜5 番（[dqn_based_agent.cpp:1241-1243・2714-2719](../../src/core/base/src/dqn_based_agent.cpp)）、勾配ノイズの raw などの振り分け（同 2780〜2782）、noisy の 8 の境（同 2861・2888・3039〜3076、[default_dqn_agent.cpp:572](../../src/algo/dqn/src/default_dqn_agent.cpp)。`.first(8)`・`.subspan(8)`・`metric = 8`・`metric - 8` を含む）、replay fit の `request`・`values` の番号（同 2917〜2991）、policy churn（[dqn_based_agent.hpp:1060・1068・1078](../../src/core/base/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp:3256](../../src/core/base/src/dqn_based_agent.cpp)。1060 行は「最後の 1 つは Q を使わない指標」という並びに依る `request.size() - 1`）、IQN 診断の pack の 0〜6 番（[dqn_based_agent.cpp:5238-5256](../../src/core/base/src/dqn_based_agent.cpp)）、weight norm の pack の 6・7 番（同 5494・5496）である。テストも、キー配列の先頭の範囲を番号で取る（[dqn_based_agent_test.cpp:5199・5263・7571・8122](../../src/core/base/src/dqn_based_agent_test.cpp)）。配列へ 1 つ挿すと、離れた場所の番号が黙って別の指標を指す。
- `BatchUpdateResult::GetScalar` の鍵の全部を 1 枚の表にする形は、名前を解析して受ける群が表に載らないので退けられている（[RFD0122](0122_small_refactor_and_test_gap_fixes_10rfd.md) の D6）。
- ImageClsAgent も、同じ形の要求を可塑性の 1 群だけ持つ（[image_cls_agent.hpp:300](../../src/algo/imagecls/src/image_cls_agent.hpp)）。

### 2.6 ほかの RFD の状況（2026-11-13）

- [RFD0127](0127_eval_trace_intrinsic_reward_columns_10rfd.md) は実装・コミット済みである（72699639）。起票時の未決 17（RFD0127 の実装を待つ）の前提は満たされた。
- [RFD0161](0161_observation_pipeline_10rfd.md)（観測パイプライン）は、別の枠でグリルが始まっている。DefaultDQNAgent のコンストラクタの、stack 軸の spec を作る部分（[default_dqn_agent.cpp:183-201](../../src/algo/dqn/src/default_dqn_agent.cpp)）を触る。
- [RFD0182](0182_source_layout_10rfd.md)（ソースの配置）は暫定・未着手で、DQN 系の共通部品を `dqn_based` のライブラリに、DefaultDQN と Rainbow をそれぞれ別のライブラリに置く案を持つ。

## 3. 用語

新しい語は足さない。[CONTEXT.md](../../CONTEXT.md) を変えない。本書の「DQN 系の共通部品」は、設計文書の「共通 DQN component」（`anet::rl::dqn` の部品。[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の §2.2）を指す。

## 4. 規約と過去の判断

- 挙動を変えない直しでは、hot path の速さが変わらないことを Run のブレ幅で確かめる。乱数の呼び出し順に触るなら、同 seed の再現性を確かめる。テストの無い箇所は、特性テストを先に足す（[リファクタリング観点](../quality/refactor.md) の「ガードレール」）。
- 「挙動を変えない」には、ログの文言と WARN の出る回数を含める。変わる直しは、変わることを書く（同「運用ルール」）。
- 学習器や Agent を変えるときは、乱数の派生と消費の順を変えず、同 seed・決定的モードの短い Run のメトリクスを、実装の前に保存した実行体と比べる（[324 DQN](../wiki/categories/324_dqn.md) の「変えるときの条件」）。
- 短い Run の前後比較は、RFD0122 と RFD0125 が同じファイルで行った。変更前の実行体と設定を保存し、seed 1・決定的モードで流して、時間と資源の tag を除く全 scalar の点を比べる（[RFD0125 の検証資材](../../src/core/base/testdata/rfd0125/README.md)）。
- ある型の設定の値域や整合のように、汎用の部品の局所的な情報だけで正誤が決まる検証は、個別の実装に書かず、その型の共通処理として置いて利用側から呼ぶ。前例は `WeightInitConfig` の `ValidateWeightInitConfig`（[AGENTS.md](../../AGENTS.md) の「汎用機構と利用側の責任境界」）。
- 汎用層の型・関数・分岐・検証は、特定の Agent の名前を持たない（同）。
- 人が指定した設定値を別の値へ置き換えない。不正値は `ANET_SYSTEM_ERROR` で落とし、エラー文にキー・指定値・期待値を入れる（[AGENTS.md](../../AGENTS.md) の「設定値の扱い」）。
- RainbowAgent は Agent 柔軟性の実証として残す（[RFD0059](done/0059_config_concept_tree_alignment_10rfd.md) の D22）。
- Evaluation Trace の内的報酬の写しは、Actor が network を写し直すロックの中でだけ作る（[RFD0127](0127_eval_trace_intrinsic_reward_columns_10rfd.md) の D4）。
- もとからある死にコードは、消さずに報告する（[AGENTS.md](../../AGENTS.md) の「LLM コーディング規律」）。課題が名指して要否を確かめる対象にしたもの（stream guard）は、これに当たらない。

性能の区分（リファクタリング観点の「ガードレール」）は次のとおり。

| 課題 | 区分 | 確かめ方 |
|---|---|---|
| 08d557 | 中立（step ごとの取り込みの経路）。lane の抽選の乱数に触る | 速さをブレ幅で確かめる（D18）。同 seed の前後比較（D16） |
| 288879 | 中立（学習の毎回の転送） | 速さをブレ幅で確かめる（D18） |
| 72d640 | 関わらない（構築時） | 前後比較（D16） |
| e33797 | 関わらない（設定の読み込み） | Agent が記録する設定の比較（D16） |
| 24d885 | 中立（更新ごとの要求の組み立て） | 速さをブレ幅で確かめる（D18） |

## 5. 確定事項

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | **束ね方とステージ。** 5 件を 1 本の RFD で扱い、課題ごとの 5 つのステージに分ける。頭に、実装前の実行体と設定を保存するステージ 0 を置く。各ステージの終わりに §7 の 1〜3 の前後比較を通し、ステージごとにコミットできる。順は §6.1 の表のとおりで、08d557（内的報酬）と 24d885（診断）を 72d640（構築）より前に置く | 1 本にするのは起票時の指示。ステージに分けるのは、前後比較で差が出たときに原因を絞るため（G3）。08d557 を 72d640 より先にするのは、構築の関数から内的報酬の部分を先に抜いておくため。24d885 を 72d640 より先にするのは、24d885 が Learner の構築のときに受け取るもの（probe 用の乱数。D13）を変え、72d640 がその Learner の生成を共通部品へ移すためで、逆の順だと共通部品の口を作り直す。起票時は「残りの 3 件は互いに独立」としていたが、別モデルの検討でこの依存が分かった。288879 と e33797 は起票時の順のままにする（e33797 が移す検証の関数（D5）は、72d640 で Rainbow から呼ぶので、先にあるほうがよい）。依存: D5・D13 |
| D2 | **RainbowAgent の役割と制限。** RainbowAgent は残す。役割は、DQN 系の共通部品を 2 つ目の Agent から使い、責務の分け方がおかしくないかを確かめることで、Run では使わない。Rainbow 相当まで設定の柔軟性を意図して落とす。制限は次のとおり。分布表現は QR か通常の Q（IQN は持たない）、Actor の方策は ε-greedy だけ、Head の射影の設定を持たない（Noisy を選べない）、報酬スケーラ・観測の正規化・stacker・`nn_viz` を持たない、target の方策は Greedy に固定。この役割と制限を設計文書に書く（§6.6） | 開発者の説明（2026-11-13）。[RFD0059](done/0059_config_concept_tree_alignment_10rfd.md) の D22 と同じ向きで、開き直す条件（前提の変化、矛盾する証拠）は見当たらない。依存: なし |
| D3 | **写しの直し方。** DefaultDQNAgent と RainbowAgent が同じ処理を写している箇所（Head の Factory の選択、方策の生成、Actor の生成の共通部分、Learner の生成、Learner と ReplayBuffer への値の振り分け）は、DQN 系の共通部品に欠けている責務と読み、共通部品へ移す。共通部品は、具象 Agent の設定の型（`DefaultDQNAgentConfig`・`RainbowAgentConfig`）を受け取らず、DQN 系の共通の型だけを受け取る。2 つの Agent は、共通部品に自分の制限（D2）と構成を渡して組み立てる。共通部品は、Agent が持たない部品（観測の正規化、stacker の設定、learner の seed、forward の資源、Head の射影の設定、ログの Munchausen の接尾）を任意の入力として受け、RainbowAgent は今渡していないものを渡さない。ただし probe 用の乱数の基の seed は、どの Learner も乱数を持つので（D13）、RainbowAgent も渡す。共通部品を通すことで Rainbow で変わる点（target の方策の lane 数と device、値の振り分けで答える鍵）は、§6.2 に挙げる | D2 の役割から、写しは「2 つの Agent の共有」ではなく「共通部品の欠け」として直すのが役割どおりである。具象 Agent の型を受け取らないのは、汎用層が特定の Agent の名前を持たない規約による（§4）。任意の入力にするのは、Agent の構成の違いを受けるためで、Rainbow だけの切り替え（§5.1）ではない。必須にすると、Rainbow が今渡していない learner の seed などを渡すことになり、Rainbow の乱数の列が変わる（§2.3）。target の方策と値の振り分けは、共通部品を通す以上は Rainbow でも DefaultDQN と同じになり、Run で使わないので影響が無い（D2）。probe 用の seed を渡しても、Rainbow は購読の設定を持たず probe を動かさないので、乱数の列は変わらない（§2.3）。依存: D2・D13 |
| D4 | **Rainbow の食い違い 5 点。** (a) Rainbow も、方策と τ の検証（D5）を呼ぶ。(b) Rainbow は ε-greedy に絞ったまま、受け付けない `policy_type` を ERROR にする。受け付けるのは `EpsilonGreedy` と、その数字の別名 `0` の 2 つで、エラー文は `Invalid RainbowAgent.actor.[<key>].policy.policy_type='<値>'; expected EpsilonGreedy (RainbowAgent supports epsilon-greedy only).` とする。この検査は (a) の共通の検証より先に行う。(a) と (b) は `RainbowAgentConfig` の 1 つの関数にまとめ、設定の読み取りと RainbowAgent のコンストラクタの両方から呼ぶ。(c) 設定の記録を `MetricsLogger::Log(const Config&)` に揃え、直後の `Flush()` は消す。(d) 今のまま（`net.body` と構造図を記録しない）。(e) `UpdateFromBatch` を private に揃える | 開発者の選択（制限は ERROR で明示し、ずれは揃える）。(b) は D2 の制限で、今の「読んでいるのに使わない」は設定値の扱いの規約に反する（§2.3）。受け付ける値は DefaultDQN が ε-greedy として作る値と同じにし、数字の別名の整理は別の課題に任せる（§2.3）。エラー文が数字の別名 `0` を挙げないのは、DefaultDQN の方策の検証のエラー文（[default_dqn_agent.hpp:49-51](../../src/algo/dqn/src/default_dqn_agent.hpp)）と同じにしたため。共通の検証より先にするのは、共通の検証のエラー文が Rainbow の受け付けない方策も候補に挙げるため。コンストラクタからも呼ぶのは、方策の生成を共通部品へ移すと（D3）、設定の DTO を直接渡す経路で `policy_type = UQE` の Rainbow が UQE を作ってしまうため。前例は分位の設定の検証である（§2.3）。(c)・(e) は意図がコードから読めないずれ。`Flush()` を呼ぶ意図も読めない。(d) は、構造図の記録が D3 の共通部品に入らず、Rainbow は Run で使わないので記録を読む人がいないため。起票時の理由「Rainbow が持たない `nn_viz` に依存する」は誤りで、`nn_viz` を使うのは `net.detail` だけだった（§2.3）。依存: D2・D3・D5 |
| D5 | **方策と τ の検証の置き場。** `ValidateActionPolicy`・`ValidateTauRule` を、検証する型と同じ DQN 系の共通のヘッダーへ移す。中身・エラー文・呼ぶ時機（各 Agent の設定の読み取りの中）は変えない | 検証の中身はどれも型の中の値だけで正誤が決まり、汎用の部品の局所的な情報で決まる検証は型の共通処理に置く規約と、`ValidateWeightInitConfig` の前例による（§2.3・§4）。依存: なし |
| D6 | **設定のエラー文の書き方。** `DefaultDQNAgentConfig` の、語の検証と数値の範囲の検証のエラー文を、2 つの形にそろえる。語は `Invalid <key>='<値>'; expected A, B, or C.`（候補が 2 つなら `A or B`）、数値の範囲は `Invalid <key>=<値>; expected <範囲>.` とする。共通の関数は、語の検証と数値の範囲の検証で 1 つずつにする。キーの文字（`DefaultDQNAgent.` を付けるか）は今のまま、呼ぶ側が渡す。D5 で移す方策と τ の検証の文言は変えない | 開発者の選択（2026-11-13、別モデルの検討のあと）。速さと正しさは変わらず、変わるのは不正な設定のときにだけ出る文面（語 5 か所、数値の範囲 13 か所）と、既存のテスト 2 本の期待値である（§2.4）。書き方を保つと、共通の関数が書き方の数（6 つ）だけ要り、新しい検証を足すときも書き方を選ぶことになる。正しい設定の Run には出ないので、前後比較（§7 の 3）に影響しない。変わる点は §6.2 に挙げる（G1 の例外）。D5 の文言を変えないのは、D5 が置き場だけを移す決定だからである。依存: なし |
| D7 | **`ExperienceSamples::To` の未定義の tensor。** CPU の経路も CUDA の経路も、未定義の tensor をそのまま返す | どのフィールドを必須とするかは標本を使う側（Learner）が決め、転送の側は決めない（汎用機構と利用側の責任境界、§4）。今の作り手は必須のフィールドを必ず作るので、今の Run の入力では何も変わらない（§2.2）。CPU の経路の変更前の挙動は §8 の U2 で確認し、変わる点を §6.2 に記録した。依存: なし |
| D8 | **`ExperienceSamples::To` の stream guard。** 消す | 効き目が無い（§2.2）。課題 288879 が、要否を確かめる対象として名指している。CUDA の構成の前後比較（D16）で値が変わらないことも確かめる。hot path の回数の変化は、学習の転送 1 回（更新 1 回）につき、今の stream の取得と設定し直し・戻しの軽い呼び出しが無くなるだけで、増えるものは無い。速さは §7 の 8 で確かめる。依存: なし |
| D9 | **`ExperienceSamples::ToString`。** 転送と同じフィールドの並びから作り、今は出していない `window` の 6 つと `sampled_rewards`・`sampled_window_rewards` も出す | 呼ぶ箇所はコメントアウトした 2 行だけで、どの Run のログにも出ない（§2.2）。並びを 1 つにする課題の対応方針による。依存: なし |
| D10 | **lane の抽選の乱数の種類。** 変えない | 変えると、checkpoint の形式と乱数の列が変わる（G1）。依存: なし |
| D11 | **内的報酬のクラスの範囲と口。** 内的報酬の扱いを 1 つのクラスへ移し、Agent が持つ。クラスが持つのは、学習の入口での採点とボーナスの計算・診断値の公開・補助学習（credit と lane の抽選を含む）、checkpoint への保存と読み込み、`intrinsic.` で始まる値の読み出し、Evaluation Trace の採点と写し（推定器の複製と β）である。学習の入口では、クラスはボーナス（β を掛け、終端をマスクした値）を返し、報酬へ足すのは Agent の 1 行にする。ボーナスの比率の分母に使う外的報酬は、Agent から受け取る。ボーナスの式は、倍率と終端の印から「マスクした（倍率 − 1）× β」を作る関数 1 つにし、学習の入口と Trace の採点の両方が使う。Trace の採点だけが持つ処理（β が未成立のときと、誤差が有限でない lane を NaN にする）は Trace の側に残す。Evaluation Trace の採点が使うのは、Agent ではなく内的報酬のクラスとする。写しを作るときと、写しを持たない Actor が読むときのロックの取り方は、RFD0127 の D4 のまま変えない。保存の形式と、乱数の消費の順は変えない | RFD0121 は学習のボーナスと Trace のボーナスの列の両方を変えるので、式が 2 か所だと G2 を満たせない（§2.1）。ボーナスを返して足すのを Agent に置くのは、RFD0121 が β を掛けない内的の成分を別に Learner へ渡すため（RFD0121 の D3・D8）で、クラスの中で足すと RFD0121 で口を作り直すことになる（別モデルの検討で分かった）。2 つの式は、推定器を更新するか、マスクを複製するか書き換えるか、NaN の扱いでも違う（[default_dqn_agent.cpp:801-817・855-861](../../src/algo/dqn/src/default_dqn_agent.cpp)）ので、共有するのは両方に共通の部分だけにした。credit と lane の抽選を推定器へ寄せないのは、設計文書が外側の Agent に置くと定めているため（§2.1）。依存: なし |
| D12 | **診断の群の共通の型と読み出し。** 間隔の gate を持つ 6 つの群は、購読の解析・間隔・発火・要求を 1 つの共通の型で持つ。値の読み出しは、gate を持つ 6 群に、毎更新に値を出す群（skip・Munchausen）と、IQN 診断・weight norm の値を加えて、「群の指標の並びと値を受け、値が無ければ NaN を返す」共通の処理 1 つにする。値が、その更新で成立した指標だけにある群も、毎更新の値をまとめて運ぶ群も、同じ処理で読む。ただし `BatchUpdateResult::GetScalar` の鍵を 1 枚の表にはしない。指標と、指標の範囲の境（noisy の統計と probe の境など）は、値を作る側も読む側も名前で指し、番号で指さない。指標の並びと、名前で指す参照が食い違えば、ビルドが止まるようにする。共通の型に載らない群（§8 の U1）も同じである。指標の並びを変えたときに、ほかの場所の参照が黙って別の指標を指す形を無くす（確かめ方は §7 の 9）。メトリクスの名前と WARN の文言・回数は変えない。共通の型に載らない群の扱いは §8 の U1 による | 課題 24d885 の対応方針と事象（IQN 診断・weight norm の値も挙げている）。読み出しを毎更新に値を出す群まで広げるのは、RFD0121 が足すメトリクスが毎更新に出す値で（§2.5）、gate を持つ 6 群だけを共通にすると、G2 の「診断の群」が 1 か所にならないため（別モデルの検討で分かった）。食い違いをビルドで止めるのは、指標に名前を付けるだけでは、並びを変えたときに名前が黙って別の指標を指す形が残るため。止め方は実装計画で決める。1 枚の表にしないのは、名前を解析して受ける群が載らないため（[RFD0122](0122_small_refactor_and_test_gap_fixes_10rfd.md) の D6）。依存: なし |
| D13 | **probe 用の乱数の持ち主。** 可塑性・policy churn・replay fit・noisy・attention の probe 用の乱数 5 つは、使う側の Learner が、自分の構築時に作って持つ。seed は、Agent の基の seed を Learner の構築のときに受け取り、今と同じ名前（`plasticity_probe`・`policy_churn_probe`・`replay_fit_probe`・`noisy_probe.sample`・`attention_entropy_probe`）から作る。乱数は、購読を設定し直しても作り直さない。Agent は probe 用の乱数を作らず、購読を見て乱数を作るかを決める処理も持たない。noisy の probe の forward の設定は今どおり Agent が作るが、購読を見ずに構築時に作り（Network に Noisy の parameter があるときだけ）、Learner の構築時に渡す。どの Learner も乱数を必ず持つので、乱数が渡らないときの群ごとの扱い（4 通り）は無くなる。RainbowAgent も自分の基の seed を渡す。replay fit と Thompson の方策の組の検査は、今どおり購読の設定で行う | 開発者の選択（2026-11-13、別モデルの検討のあと）。使うのは Learner だけなので、使う側が作って持つのが単純である（§2.5）。起票時の既定（購読されたときに Agent が作る）は、Agent と Learner が同じ購読の判断を別々に書き、食い違うと Run の途中で止まるか、黙って購読を捨てる（§2.5）。Agent の基の seed から今と同じ名前で作るので、乱数の列は変わらない（名前付きの seed は作る順に依らず、作るだけでは消費しない。probe 用の乱数は checkpoint に入らない。§2.5）。作り直さないのは、購読を設定し直したときに乱数の列が頭に戻らないようにするため（購読の設定は、冒頭で群の状態を作り直す）（今は Agent が 1 回だけ作って持ち続ける。§2.5）。noisy の forward の設定を構築時に作るのは、Agent に購読の判断を残さないため。Learner 自身の seed から作る形を退けた理由は §5.1。所有の形を変えるので、設計文書を直す（D20、§6.6）。起票時の未決 11 の理由（seed の派生の順に触る）は誤りだった。依存: なし |
| D14 | **接頭辞の下を写して記録する関数。** 記録は 1 つの関数にまとめ、基底の `Config::RecordConfigValue` を通す。写す先の形は、受け取る側の口のまま保つ。`net.block.`・`net.config_profile.` を skip.net の写し先へ写す処理には、理由のコメントを足す | 記録するキーの形は 4 か所とも同じで、違うのは受け取る側の口に合わせた写す先の形である（§2.4）。受け取る側の口を変えると、直す範囲が `NetworkConfig` と Head の射影に広がる（G1）。依存: なし |
| D15 | **特性テスト。** 直す前に足し、直す前と直したあとの両方で通す。ただし、§6.2 の意図して変える点のテストと、probe 用の乱数を Learner へ直接渡すテスト（D13 で渡し方が変わる）は、直したあとの形と期待値を実装計画に書いて差し替える。RFD0129 がステージの切れ目で入ったときの扱いは §10 による。対象は §6.3 のとおり。既にあるテストで覆えるものは足さない | テストの無い箇所は特性テストを先に足す（§4）。Run の構成で通らない組（Rainbow の学習、学習を止めた構成など）を、テストで覆う。依存: D2・D4・D7・D11・D12・D13 |
| D16 | **前後比較の組と比べ方。** [RFD0125 の検証資材](../../src/core/base/testdata/rfd0125/README.md) の 31 組を基にし、RND の 2 組と、診断の 6 群をすべて購読する組を足す（§6.4）。比べるのは、メトリクス・Agent が記録する設定・ログの WARN と ERROR（§7 の 1〜3） | ステージごとに直す経路（CPU と CUDA の転送、分布 3 種の Head と Learner、内的報酬、診断の 6 群）を Run で通すため。比べ方は RFD0125 と起票時の受入のまま。依存: D1 |
| D17 | **checkpoint の確かめ。** 実装前の実行体で保存した RND 有効の checkpoint を、新しい実行体で読んで続けた Run と、実装前の実行体で続けた Run を比べる（§7 の 4） | ステージ 3 で、保存と読み込みの順を書いた処理を移すため。依存: D11 |
| D18 | **速さの確かめ。** ステージ 1・3・4 のあとに、RFD0125 の `performance` の手順で、変更後の中央値が基準の最大値以下であることを確かめる。ステージ 3 だけは同じ構成に `run.@rnd` を重ねる（§7 の 8） | RFD0125 と同じ基準（§4 の性能の区分）。ステージ 3 は、RND を有効にしないと直した経路を通らない。依存: D1 |
| D19 | **番号の直書きの確かめ。** §7 の 9 の検索語を、3 つのファイルと、指標の範囲を番号で取るテストで引いて 0 件にする。加えて、指標の並びと参照が食い違うとビルドが止まることを、並びを入れ替えて確かめる | 起票時の検索語は replay fit と policy churn の形だけで、課題 24d885 が挙げた直書きのうち schedule の鍵・勾配ノイズ・noisy の境を覆えなかった。別モデルの検討で、noisy の範囲の取り方、policy churn の「最後の 1 つ」、IQN 診断と weight norm の値の読み出し、テストの範囲の取り方も当たらないと分かった（§2.5）。検索だけでは、番号を名前に置き換えるだけで 0 件になり、D12 の目的を確かめられないので、ビルドで止まることも確かめる。作ってすぐ読む同じ関数の中で値をまとめる箇所（[dqn_based_agent.cpp:83-89](../../src/core/base/src/dqn_based_agent.cpp)）と、文字列の長さの計算（同 1158）は当たらない形にした。依存: D12 |
| D20 | **設計文書。** 新しく作らない。既存の設計文書の該当する節を、変更に合わせて直す（§6.6）。RainbowAgent の役割と制限（D2）と、probe 用の乱数の持ち主（D13）も、設計文書に書く | 新しい仕組みを足さない（G1）。設計文書と乖離する変更は同じ変更の中で直す（[AGENTS.md](../../AGENTS.md) の「AI エージェントの作業ルール」）。依存: D2・D13 |
| D21 | **ADR。** 起こさない | 戻しにくく、文脈が無いと意外で、本当に案を比べて選んだ、の 3 条件を満たす判断が無い。RainbowAgent は RFD0059 の D22 のとおり残す。依存: D2 |
| D22 | **ImageClsAgent の同じ形の要求。** 触らない。診断の群の共通の型は DQN 系の共通部品に置く | 合意した範囲は題材の 5 件で、前後比較の組が ImageCls を通らない（§1.3）。依存: D12 |
| D23 | **意図して変える点。** §6.2 に挙げる。DefaultDQNAgent の学習の挙動を変えるものは `ExperienceSamples::To` の CPU の経路の 1 点だけで、今の Run の入力では値が変わらない。DefaultDQNAgent の設定のエラー文は D6 のとおり変わる（不正な設定のときだけ出る）。RainbowAgent の挙動は 5 点（D4 の (a)・(b)・(c) と、D3 の target の方策・値の振り分け）変わる | D3・D4・D6・D7 の帰結。G1 は DefaultDQNAgent の挙動を変えないとした。依存: D3・D4・D6・D7 |
| D24 | **G2 の確かめ。** RFD0121 が足すものごとに、本 RFD のあとの足す先を §6.7 の表に書く。§1.1 の表の場所に当たる行は、どれも 1 か所を指す形にする | G2 は、RFD0121 の実装で初めて分かる形にせず、確定した RFD0121 の本文と今のうちに突き合わせる。§7 に G2 を確かめる受入が無かった（別モデルの検討で分かった）。依存: D3・D11・D12・D14 |

### 5.1 選ばなかった形

| 選ばなかった形 | 裁定 | 理由 |
|---|---|---|
| RainbowAgent を消す | 不採用 | DQN 系の共通部品の責務分担を確かめる 2 つ目の利用者として残す（D2、RFD0059 の D22） |
| Rainbow の食い違い 5 点を、すべて DefaultDQN に揃える | 不採用 | Rainbow でも UQE・Thompson などが効くようになり、Rainbow 相当という制限（D2）が外れる |
| Rainbow の食い違い 5 点を、今の差のまま保ち、共通部品の引数で切り替える | 不採用 | (a)・(b) の黙った置き換えが残り、共通部品が Rainbow のためだけの切り替えを持つ |
| 2 つの Agent が共有する関数を、具象 Agent の設定の型ごと受け取る形で作る | 不採用 | 共通部品が具象 Agent の名前を持ち、Rainbow で責務分担を確かめる役割が成り立たない（D3） |
| 共通部品が、DefaultDQN の渡す入力をすべて必須で受ける | 不採用 | Rainbow が今渡していない learner の seed などを渡すことになり、Rainbow の乱数の列と制限が変わる（D3） |
| Rainbow の制限の検査を、設定の読み取りだけで行う | 不採用 | 設定の DTO を直接渡す経路で制限が外れ、共通部品が UQE などを作る（D4） |
| 構造図の記録を共通部品へ寄せ、Rainbow でも `net.body` と `net.structure` を記録する | 不採用 | Rainbow は Run で使わず、記録を読む人がいない（D4 の (d)） |
| 方策と τ の検証の関数を DefaultDQN のヘッダーに残し、Rainbow から DefaultDQN のヘッダーを読む | 不採用 | Rainbow が DefaultDQN に依存し、型の検証を型の共通処理に置く規約に反する（D5） |
| 方策の検証を、方策の部品の構築時に移す | 不採用 | 検証の時機が設定の読み取りから Actor の生成へ遅れ、使わない Actor のカタログの誤りを検出しなくなる（G1） |
| `ExperienceSamples::To` で、必須のフィールドが未定義なら止める | 不採用 | どのフィールドが必須かは標本を使う側が決める（D7） |
| `ExperienceSamples::To` の経路ごとの未定義の扱いを、引数で切り替えて保つ | 不採用 | 片方だけ直した跡と読める食い違いを、引数として残す（D7） |
| stream guard を、CUDA のときだけ同じ範囲で作って残す | 不採用 | 効き目が無い（D8） |
| 設定のエラー文の書き方を保ち、書き方ごとに共通の関数を分ける | 不採用 | 共通の関数が書き方の数（6 つ）だけ要り、新しい検証を足すときも書き方を選ぶことになる。そろえて変わるのは、不正な設定のときの文面だけである（D6） |
| lane の抽選の乱数を、同じクラスのほかの乱数と同じ種類にそろえる | 不採用 | checkpoint の形式と乱数の列が変わる（D10） |
| 補助学習の credit と lane の抽選を、推定器の側へ寄せる | 不採用 | 設計文書が外側の Agent に置くと定めている（D11） |
| Evaluation Trace の採点を、内的報酬のクラスの外に残す | 不採用 | ボーナスの式が 2 か所に残り、RFD0121 が 2 か所を直す（D11） |
| 内的報酬のクラスが、ボーナスを報酬へ足すところまで持つ | 不採用 | RFD0121 が β を掛けない内的の成分を別に渡すので、口を作り直すことになる（D11） |
| 共通の型を作らず、指標を名前で指す形にだけ直す | 不採用 | 群を足すたびに 12 か所前後を直す形が残る（D12）。共通の型に載らない群だけはこの形にする。その群でも、並びと参照が食い違えばビルドが止まる形にする（§8 の U1） |
| 指標に名前を付けるだけで、並びとの食い違いをビルドで止めない | 不採用 | 並びを変えたときに、名前が黙って別の指標を指す形が残る（D12・D19） |
| 診断の共通の読み出しを、gate を持つ 6 群だけにする | 不採用 | RFD0121 が足す、毎更新に出すメトリクスが載らず、G2 を満たせない（D12） |
| RFD0121 の待ちから、診断のステージ（24d885）を外す | 不採用 | 読み出しを毎更新に値を出す群まで広げれば、RFD0121 のメトリクスも 1 か所に足せる。順は RFD0121 の D18 のまま（D12） |
| probe 用の乱数の渡し口を 2 つのまま残す | 不採用 | 課題 24d885 の事象が残る（D13） |
| probe 用の乱数を、購読されたときに Agent が作り、購読の設定のときに渡す（起票時の既定） | 不採用 | Agent と Learner が同じ購読の判断を別々に書き、食い違うと Run の途中で止まるか、黙って購読を捨てる（D13） |
| probe 用の乱数を、Agent が構築時に作って Learner へ渡す | 不採用 | 使うのは Learner だけで、使う側が作って持つほうが単純である（D13） |
| probe 用の乱数を、Learner が購読されたときに群の型の中で作る | 不採用 | 購読を設定し直すと、乱数の列が頭に戻る置き場になる（D13） |
| probe 用の乱数を、Learner 自身の seed（`learner`）から作る | 不採用 | 同じ seed でも probe の診断の値が変わり、診断を作り直すステージの前後比較を一致で確かめられない。切り替えたいときは、本 RFD のあとに別の段で行い、変わるのが probe の tag だけであることを確かめる（D13） |
| 写す先の形を 1 つにそろえる | 不採用 | 受け取る側の口（`NetworkConfig`、Head の射影、RND）を変えることになる（D14） |
| 診断の群の共通の型を Agent の共通基盤に置き、ImageClsAgent も載せる | 範囲外 | 題材の 5 件の外。再開条件: ImageCls に診断の群を足すとき（D22） |
| ボーナスの診断の `.item<float>()` を 1 回の転送にまとめる | 範囲外 | 性能の変更で別の関心。再開条件: RND の時間の内訳と合わせて測るとき（§1.3） |
| Learner の購読を「開始前の宣言」と「呼び出しごとの要求」に分ける | 範囲外 | 購読の口を変える別の関心。再開条件: 購読の口を変える要求が出たとき（§1.3） |
| RFD0165・RFD0161 を先に入れる | 不採用 | RFD0121 の待ちが延びる（§1.3） |
| 課題ごとに RFD を分ける、規模の小さい 2 件だけを先に出す | 不採用 | 起票時の指示（D1） |

### 5.2 起票時の未決表との対応

| 起票時の行 | 行き先 | 仕分け |
|---|---|---|
| 1 束ね方とステージ | D1 | 決定の組み合わせから導いた。順は、別モデルの検討のあとに D13 から直した |
| 2 RainbowAgent を残すか | D2 | RFD0059 の D22 と開発者の説明 |
| 3 Rainbow の食い違い 5 点 | D4 | 開発者に聞いた |
| 4 エラー文 | D6 | 開発者に聞いた（別モデルの検討のあと） |
| 5 `To` の CPU の経路の未定義 | D7、§8 の U2 | 規約から導いた。今の挙動は試して確かめる |
| 6 stream guard | D8 | 事実（§2.2）から導いた |
| 7 `.item<float>()` 6 回 | §1.3 の非ゴール | 再グリルの初めに、ゴールと非ゴールとして合意した |
| 8 lane の抽選の乱数 | D10 | G1 から導いた |
| 9 内的報酬のクラスの口 | D11 | G2 と事実（§2.1）、RFD0121 の D3・D8 から導いた |
| 10 診断の群の共通の型 | D12、§8 の U1 | G2 と課題 24d885 から導いた。載るかは実装計画で試して確かめる |
| 11 probe 用の乱数の渡し口 | D13 | 開発者に聞いた（別モデルの検討のあと）。起票時の理由を直した |
| 12 Learner の購読の見直し | §1.3 の非ゴール | 再グリルの初めに、ゴールと非ゴールとして合意した |
| 13 確かめ方 | D16〜D19・D24、§8 の U3 | 細部として推奨の既定で埋めた。D19 と D24 は別モデルの検討のあとに直した・足した |
| 14 特性テスト | D15 | 規約から導き、対象は細部として推奨の既定で埋めた |
| 15 写して記録する関数 | D14 | 事実（§2.4）から導いた。起票時の記述を直した |
| 16 RFD0165 との順序 | §1.3 の非ゴール | 再グリルの初めに、ゴールと非ゴールとして合意した |
| 17 RFD0127 との順序 | §2.6 | 事実（RFD0127 は実装済み） |
| 18 設計文書 | D20 | G1 から導いた |
| 19 ADR | D21 | ADR を起こす条件から導いた |
| 20 新しいクラスの名前 | §6.5 のクラス一覧 | クラス一覧は起草で書く |
| 21 ImageClsAgent | D22 | 合意した範囲（題材の 5 件）から導いた |

## 6. 実装契約

### 6.1 ステージと直す箇所

| ステージ | 課題 | 直すこと |
|---|---|---|
| 0 | — | 実装前の実行体と設定を保存する（§6.4）。前後比較の組が直す経路を通ることと、診断の 6 群が点を出すことを確かめる（§8 の U3）。§6.3 の特性テストを足し、実装前の実行体で通す |
| 1 | 288879 | `ExperienceSamples` のフィールドの並びを 1 つにし、`To`・`ForEachTensor`・`ToString` が同じ並びを使う。`To` は、D7・D8 を当てて CPU と CUDA で 1 つの経路にし、「device へ送る」「CPU に置く」の 2 つの扱いだけを使い、未定義はそのまま返す（D7）。stream guard を消す（D8）。`ToString` は全フィールドを出す（D9） |
| 2 | e33797 | `DefaultDQNAgentConfig` のコンストラクタを、関心ごとの private の関数に分ける（内的報酬、Head、Actor のカタログ、learner、skip、target_policy の既定値、組み合わせの検証など）。語の検証と数値の範囲の検証を、キーと候補（または範囲）を受ける共通の関数 1 つずつに寄せ、エラー文を D6 の 2 つの形にそろえる。写して記録する処理を 1 つにする（D14）。方策と τ の検証を共通のヘッダーへ移す（D5） |
| 3 | 08d557 | 内的報酬の扱いを 1 つのクラスへ移す（D11）。Agent の側は、「有効なら、学習の入口でクラスからボーナスを受け取り、報酬へ足す（1 行）」「保存と読み込みを委ねる」「`intrinsic.` で始まる値の読み出しを委ねる」「Evaluation Trace の採点と写しを委ねる」だけにする |
| 4 | 24d885 | 診断の群の共通の型を足し、gate を持つ 6 つの群が 1 つずつ持つ（D12、§8 の U1）。値の読み出しを、毎更新に値を出す群も含めて共通の処理 1 つにする（D12）。Learner の購読の設定を、振り分け・検証・Resource の確保に分ける。probe 用の乱数を Learner が構築時に作って持つ形にし、Agent から乱数の作成と購読の判断を除く（D13）。指標を番号で指す箇所を無くし、並びと参照が食い違えばビルドが止まる形にする（D12・D19） |
| 5 | 72d640 | Head の Factory の選択、方策の生成、Actor の生成の共通部分、Learner の生成、値の振り分けを、DQN 系の共通部品へ移し、2 つの Agent から呼ぶ。Agent が持たない部品は任意の入力にする（D3）。Head の Factory の選択は、呼ぶたびに Head の設定を受け取る（§6.7 の合成 Head の行）。DefaultDQNAgent のコンストラクタは、「Head の Factory を選ぶ」「入力の spec を作る」「Learner を作る」「構造図を記録する」に分ける。Rainbow の食い違い 5 点を D4 のとおりにする |

ステージを通して守ること。

- ログの文言と順を変えない（§6.2 の表の 2・3・5 の ERROR を除く）。Head の選択と Learner の生成のログ（`Network Head: …`、`Initialized …Learner …`）は、2 つの Agent で今と同じ文言・同じ位置に出す。
- Agent の値の振り分けは、鍵を引く順を変えない（DefaultDQN は内的報酬、`target_policy.`、schedule の鍵、`per_beta`、probe と noisy と ReplayBuffer、報酬スケーラ、観測の正規化の順）。
- 乱数を作る名前と、乱数を消費する順を変えない。probe 用の乱数は、購読を設定し直しても作り直さない（D13）。
- 公開ヘッダの口（`ExperienceSamples` の利用側から見た口、Agent の公開の関数）を変えない。例外は、RainbowAgent の `UpdateFromBatch` を private にすること（D4 の (e)）だけである。`ValidateActionPolicy`・`ValidateTauRule` は名前と引数を変えずに置き場だけを移す。

### 6.2 変えないものと、意図して変える点

変えないもの: 設定キー、メトリクスの名前、TensorDict の鍵、checkpoint の形式、ログと WARN と ERROR の文言・順・回数（下の表の 2・3・5 を除く）、seed の作り方、乱数の消費の順、forward の回数と順、Agent が記録する設定（Run の `config/DefaultDQNAgent.txt` と metrics の json の記録）、公開ヘッダの口（RainbowAgent の `UpdateFromBatch` を private にする D4 の (e) を除く）。

意図して変える点は次のとおりである（D23）。

| # | 変える点 | 今 | 変えたあと | 影響する Run |
|---|---|---|---|---|
| 1 | `ExperienceSamples::To` の CPU の経路で、`actions`・`target_returns`・`next_state.terminals` が未定義のとき（D7） | 3 フィールドとも例外になる。ERROR のログは出ない（§8 の U2） | 未定義のまま返す | なし（今の作り手は必ず作る） |
| 2 | RainbowAgent が、方策と τ の設定の不正値を受けたとき（D4 の (a)） | そのまま構築する | 構築時に ERROR（D5 の検証のエラー文） | なし（Run で使わない） |
| 3 | RainbowAgent の Actor の `policy_type` が `EpsilonGreedy`・`0` 以外のとき（D4 の (b)） | 何も言わずに ε-greedy で動く | 構築時に ERROR（D4 のエラー文） | なし（同上） |
| 4 | RainbowAgent の設定の記録（D4 の (c)） | metrics の json の記録だけを書き、`Flush()` を呼ぶ | json の記録に加えて Run の `config/RainbowAgent.txt` を書き、`Flush()` を呼ばない | なし（同上） |
| 5 | DefaultDQNAgent の設定の、語の検証と数値の範囲の検証のエラー文（D6） | 書き方が 6 通り（§2.4） | 2 つの形。変わる文言は語 5 か所、数値の範囲 13 か所 | なし（不正な設定のときだけ出る）。期待値を変える既存のテストは 2 本（[dqn_based_agent_test.cpp:3270・7290](../../src/core/base/src/dqn_based_agent_test.cpp)） |
| 6 | RainbowAgent の target の方策（D3） | lane 数 0・CPU で作る | Agent の lane 数と device で作る | なし（Run で使わない） |
| 7 | RainbowAgent の値の振り分け（D3） | `target_policy.`・`plasticity_probe_`・`noisy_` の鍵には答えない | 共通の振り分けを通して答える | なし（同上） |

外から見える挙動を変えないが、形が変わるもの: `ExperienceSamples::ToString` が全フィールドを出す（D9。呼ぶ箇所はコメントだけ）、stream guard を消す（D8。効き目が無い）、RainbowAgent の `UpdateFromBatch` を private にする（D4 の (e)）、probe 用の乱数を Learner が構築時に作って持ち、乱数が渡らないときの群ごとの扱いを無くす（D13。乱数の列は変わらない。DefaultDQN は今も乱数を必ず渡しているので、Run の挙動は変わらない）。

### 6.3 特性テスト

直す前に次があるかを実装計画で確かめ、無ければステージ 0 で足す（D15）。直す前と直したあとの両方で通す。§6.2 の意図して変える点のテストと、probe 用の乱数を Learner へ直接渡しているテスト（D13 で渡し方が変わる。§6.6）だけは、直したあとの形と期待値を実装計画に書いて差し替える。RFD0129 がステージの切れ目で入ったときは、§10 のとおり、RFD0129 で挙動が変わるテストの期待値を RFD0129 のあとの挙動に直し、以降のステージはそれで前後を確かめる。

- `ExperienceSamples::To` の、CPU と CUDA の device それぞれでの、定義済み・未定義の tensor の扱い（CUDA は使える環境で）。
- 内的報酬の状態（credit、乱数の状態、回数、診断値）の、保存と読み込みの往復。
- ボーナスの値が、学習の入口と Evaluation Trace の採点で同じになること。1 lane で、写しを取った直後の同じ観測と同じ β を使い、終端の lane を含める（同じ状態の推定器では、更新しない採点と更新する採点の倍率が一致する。[intrinsic_test.cpp:2431-2433](../../src/core/base/src/intrinsic_test.cpp)）。
- RainbowAgent の構築と、食い違い (a)・(b) の挙動（設定から読む経路と、設定の DTO を直接渡す経路）。
- RainbowAgent の学習。CPU・固定 seed で数回更新し、損失と重みの checksum を直す前後で比べる（D2 の役割を、構築だけでなく学習の経路でも働かせる。今は学習を回すテストが無い。§2.3）。
- Head の選択の 6 通り（分布の表し方 3 種 × Dueling の有無）と、Learner の 3 種の選択。選ばれた Head と Learner の種類と、ログの文言。
- 診断の 6 群の、購読と間隔ごとの要求（どの learn step でどの指標の要求が立つか）。
- 学習を止めた構成（`learner.enabled = false`）で、診断の 6 群と `intrinsic.` の鍵が、それぞれ NaN を返すか未知の鍵になるかと、WARN の回数（§2.5）。

### 6.4 前後比較の組と比べ方

- 条件は [RFD0125 の検証資材](../../src/core/base/testdata/rfd0125/README.md) の短縮条件のまま（seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、決定的 backend、FP32。GridMaze・LunarLander は CPU、Atari qbert・DropMerge は CUDA）。
- 組は RFD0125 の 31 組（既定の 4 組と、LunarLander の分布 3 種 × 9 組。replay fit・Munchausen・skip・IQN の診断を含む）に、次の 3 組を足す。
  - LunarLander に `run.@rnd` を重ねた組。
  - Atari qbert に `run.@rnd` を重ねた組。
  - 診断の 6 群をすべて購読する組。DropMerge に Noisy の設定と、`metrics.scalar` の `@full`・`@replay_fit`・`@gradnoise`・`@noisy`・`@attention` を重ねる。6 群のどれかが点を出さなければ、出す構成を 1 組足す（§8 の U3）。
- 実行体と設定の保存、比較の手順は、`core/anet-core/testdata/rfd0128/README.md` に書く（§6.6）。

### 6.5 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| ExperienceSamples | 変更 | フィールドの並びを 1 つにし、転送・走査・文字列化が同じ並びを使う | RL 共通基盤（rl） |
| ActionPolicyConfig | 変更 | 自分の値の検証（`ValidateActionPolicy`・`ValidateTauRule`）を、型と同じヘッダーに持つ | DQN 系の共通部品（agent） |
| DefaultDQNAgentConfig | 変更 | 読み取りを関心ごとの関数に分ける。語の検証と数値の範囲の検証を共通の関数 1 つずつに寄せ、エラー文を 2 つの形にそろえる。接頭辞の下の記録を共通の関数に寄せる | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgentConfig | 変更 | Actor の方策を ε-greedy に絞る検査と、方策と τ の共通の検証を 1 つの関数にまとめ、設定の読み取りと RainbowAgent のコンストラクタの両方から呼ぶ | RainbowAgent（rainbow_agent） |
| IntrinsicRewardCoordinator | 新規 | 学習の入口での採点とボーナスの計算（報酬へは足さずに返す）、補助学習の credit と lane の抽選、診断値、保存と読み込み、`intrinsic.` の値の読み出し、Evaluation Trace の採点と写しを持つ。ボーナスの式は 1 つの関数にする | DefaultDQNAgent（default_dqn_agent） |
| DefaultDQNAgent | 変更 | 内的報酬の扱いを委ね、受け取ったボーナスを報酬へ足す。probe 用の乱数を持たない。構築を、役割ごとの private の関数に分け、共通部品を呼ぶ | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgent | 変更 | Head の Factory の選択、方策と Actor の生成、Learner の生成、値の振り分けを、共通部品から呼ぶ。自分の基の seed を Learner へ渡す。`UpdateFromBatch` を private にする | RainbowAgent（rainbow_agent） |
| DqnComponentAssembler | 新規 | DQN 系の Agent が共通部品を組み立てるときの、Head の Factory の選択（呼ぶたびに Head の設定を受け取る）、方策の生成、Actor の生成の共通部分、Learner の生成、Learner と ReplayBuffer への値の振り分けを持つ。具象 Agent の設定の型を受け取らず、Agent が持たない部品を任意の入力として受ける | DQN 系の共通部品（dqn_based_agent） |
| DiagnosticDemandTable | 新規 | 診断指標の群 1 つぶんの、購読の解析・間隔・発火した要求を持つ。群の指標を名前で指せるようにし、並びとの食い違いをビルドで止める | DQN 系の共通部品（dqn_based_agent） |
| dqn::Learner | 変更 | 6 つの群が共通の型を持つ。購読の設定を、振り分け・検証・Resource の確保に分ける。probe 用の乱数 5 つを、受け取った Agent の基の seed から構築時に作って持つ | DQN 系の共通部品（dqn_based_agent） |
| dqn::BatchUpdateResult | 変更 | 群の値の読み出しを、毎更新に値を出す群も含めて共通の処理 1 つにする | DQN 系の共通部品（dqn_based_agent） |

### 6.6 移行対象

経路の検索（2026-11-13、`git grep -l -i -F`、追跡しているファイルの全体）で、直す関数・型・メンバを名指す箇所を確かめた。検索語と、当たったファイルの数は、`ExperienceSamples::To` 12、`ForEachTensor` 13、`DefaultDQNAgentConfig` 44、`ValidateActionPolicy` 4、`ValidateTauRule` 3、`intrinsic_learn_` 4、`lifelong_novelty_` 37（RFD0116 のファイル名 `0116_rnd_lifelong_novelty_` に当たるものを含む）、`last_intrinsic_beta_` 2、`IntrinsicSnapshot` 4、`RainbowAgent` 650、`ConfigureScalarMetricSubscriptions` 27、`DemandEntry` 4、`plasticity_probe_random` 4、`policy_churn_probe_random` 4、`kLearnerScheduleMetricKeys` 6 である。別モデルの検討のあとの改訂で、`probe_random` 7、`kSkipUpdateMetricKeys` 3、`kMunchausenMetricKeys` 4、`iqn_diagnostics` 6、`plasticity_weight_norms` 7、`caller-owned` 10、`expected one of` 12 を足して引いた。過去の RFD・実装計画・ADR・課題・wiki の課題の表・実験記録・過去の検証資材（書き換えない）を除くと、現用のものは次のとおり。

- **変えるコード**: [rl.hpp](../../src/core/base/include/anet/rl.hpp)・[rl.cpp](../../src/core/base/src/rl.cpp)・[agent.hpp](../../src/core/base/include/anet/agent.hpp)・[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp)・[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp)・[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp)・[dqn_based_agent.hpp](../../src/core/base/src/dqn_based_agent.hpp)・[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp)。共通部品をどのファイルに置くか（[dqn_based_heads.hpp](../../src/algo/dqn/src/dqn_based_heads.hpp) を含む）は実装計画で決める。
- **検索に当たるが変えないコード**: 公開の口を変えないので、呼ぶ側は変えない。[init.cpp](../../src/core/base/src/init.cpp)（RainbowAgent の登録）、[muzero_rb.cpp](../../src/algo/muzero/src/muzero_rb.cpp)（MuZero の標本の `To`）、[trainer.cpp](../../src/core/base/src/trainer.cpp)（Agent の購読の設定を呼ぶ）、[rl.hpp](../../src/core/base/include/anet/rl.hpp) の Agent の購読の設定の宣言。[tensor_util.hpp](../../src/core/base/include/anet/tensor_util.hpp)・[transfer.hpp](../../src/core/base/include/anet/transfer.hpp) の `ForEachTensor` は別の型の関数である。[image_cls_agent.hpp](../../src/algo/imagecls/src/image_cls_agent.hpp)・[image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) は範囲の外（D22）。
- **テスト**: [dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp)・[intrinsic_test.cpp](../../src/core/base/src/intrinsic_test.cpp)・[replay_buffer_test.cpp](../../src/core/base/src/replay_buffer_test.cpp) に、§6.3 の特性テストを足す。Learner を直接作るテストと、Learner の購読の設定を直接呼ぶテスト（[dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp)・[dqn_munchausen_test.cpp](../../src/algo/dqn/src/dqn_munchausen_test.cpp)）は、Learner が probe 用の乱数を自分で作る形（D13）に合わせ、Agent の基の seed を渡す形に直す。Learner を直接作る箇所は 27 か所（dqn_based_agent_test.cpp に 14、dqn_munchausen_test.cpp に 13。2026-11-13 に `TDLearner`・`QRLearner`・`IQNLearner` で引いた）、乱数を購読の設定へ直接渡しているテストは 7 か所ある（[dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp) の 9469・9608・9789・10231・10585・10640・10763 行付近）。乱数に依る期待値が変わるものは、直したあとの値を実装計画に書く。指標の範囲を番号で取るテスト（同 5199・5263・7571・8122 行）は、指標を名前で指す形に直す（D12）。診断の値の読み出しを確かめるテスト（[dqn_based_agent_scalar_test.cpp](../../src/algo/dqn/src/dqn_based_agent_scalar_test.cpp)）は、共通の読み出し（D12）に合わせて呼び方を直す。断定は変えない。設定のエラー文の期待値を変える既存のテストは 2 本（3270 行付近の skip forward mode と、7290 行付近の policy churn 設定。D6）。RainbowAgent を構築するテストは、Actor の方策に既定（`EpsilonGreedy`）を使い、不正な τ も使わないので（§2.3）、§6.2 の 2・3 で期待値の変わるテストは無い。既存の断定の期待値は、§6.2 の点と、probe 用の乱数の渡し方の直しと、RFD0129 がステージの切れ目で入ったときの §10 の直しを除いて変えない。[config_baseline_test.cpp](../../src/core/base/src/config_baseline_test.cpp)・[nn_test.cpp](../../src/core/base/src/nn_test.cpp)・[image_cls_agent_test.cpp](../../src/algo/imagecls/src/image_cls_agent_test.cpp)・[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) は、検索に当たるが変えない（設定の型と Agent の購読の設定の、公開の口を使う側）。
- **設定**: 変えない。[agent.txt](../../apps/runner/config/agent.txt)（RainbowAgent の `@baseline` は ε-greedy だけを書く）・[common.txt](../../apps/runner/config/common.txt) と、`core/anet-core/testdata/rfd0072/` の解決済みの設定の基準は、検索に当たるが変わらない。
- **設計文書**: クラスの構成と、メンバや関数を名指している節を直す。en は `anet-translate-docs` で作る。
  - [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md): §2.1 の Rainbow の行（41・94 行目）と §3 の Rainbow の説明（280 行目）に、D2 の役割と制限を書く。§3〜§5 のクラス図と部品の説明、§6.2（取り込みと学習更新の、内的報酬の扱い）、§8.1。
  - [010_framework_overview.jp.md](../design/010_framework_overview.jp.md)（183 行目）・[110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md)（61 行目）の RainbowAgent の行に、役割（DQN 系の共通部品の責務分担を確かめる 2 つ目の Agent。Run では使わない）を書く。110 の購読の設定を名指す箇所も直す。
  - probe 用の乱数の持ち主（D13）: [110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) の 48 行目（「probe 等の機能専用 RNG も Agent が named seed から所有し」）を、使う部品が Agent の基の seed から作って持つ形に直す。[150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) の 333 行目（Agent が作り、Learner が非所有参照で渡す）と 345 行目（購読が必要になるまで生成しない）、[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の 594 行目（外側 Agent が probe 用 RNG の lifetime を所有する）と 763 行目（Agent 所有の専用 RNG）も直す。200 の 717・797 行目と [ADR 0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md) の「caller-owned」は、ReplayBuffer から見た呼び出し側の所有なので変えない（§2.5）。
  - [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md)（`ForEachTensor` を名指す箇所）。
  - [212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md)（用語の表の facade の行（53 行目）を、内的報酬のクラスに合わせる。§3.8 ほか、`DefaultDQNAgent::UpdateFromBatch` を facade として名指す箇所）。
  - [100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) は、RainbowAgent の名前を挙げるだけなので変えない。
- **wiki**: [324 DQN](../wiki/categories/324_dqn.md)・[321 Agent 共通基盤](../wiki/categories/321_agent_core.md)・[318 強化学習 共通基盤](../wiki/categories/318_rl_core.md) の「選ばなかった形」「変えるときの条件」へ、本書の §5.1 と §6.1・§6.2 から書き戻す（RFD づくりで行う）。
- **検証資材**: `core/anet-core/testdata/rfd0128/README.md` を新設する（実装前の実行体と設定の保存、§6.4 の組、前後比較と速さの手順）。RFD0125 の検証の道具を呼んで使い、200 の「テストと拡張時の確認事項」の章から参照する。`rfd0084`・`rfd0092`・`rfd0093`・`rfd0116`・`rfd0125`・`rfd0127` の資材は、過去の検証の記録なので変えない（rfd0116 の README は内的報酬のメンバを名指していない）。
- **互換層**: 作らない。旧い置き場の検証の関数への転送や別名も置かない。過去の Run・ADR・実験記録は書き換えない。

### 6.7 RFD0121 の足す先（G2 の確かめ。D24）

[RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md) が足すものを、本 RFD のあとの足す先と並べる。§1.1 の表の場所に当たる行は、どれも 1 か所を指す。実装計画の完了の時点で、行ごとに、実装のあとのコードの場所（関数の名前）を書いて確かめる（§7 の 13）。

| §1.1 の場所 | RFD0121 が足すもの | 出典（RFD0121） | 本 RFD のあとの足す先 | ステージ |
|---|---|---|---|---|
| 取り込み | 内的報酬用の報酬スケーラ、β を掛けない内的の成分を Learner へ渡すこと、学習の β の受け渡し | §6.5・§8.1 | 内的報酬のクラスが返す値の定義と、Agent がボーナスを足す 1 行（D11） | 3 |
| 取り込み | 外的報酬用と内的報酬用の 2 つの報酬スケーラの状態の保存と読み込み | §8.5 | Agent の保存と読み込みの並び（`Save`・`LoadNetwork`）の 1 か所。内的報酬用のスケーラを内的報酬のクラスに持たせるなら、クラスの保存と読み込み（D11） | 3 |
| 取り込み | Evaluation Trace のボーナスの列の式（RFD0127 との調整） | §8.4 | ボーナスの式の関数 1 つ（D11） | 3 |
| 取り込み | `intrinsic.reward_scaler.scale` の値の読み出し | §8.4 | 内的報酬のクラスの `intrinsic.` の値の読み出し（D11） | 3 |
| 標本 | 成分ごとの n-step の収益と、抽選時の step の報酬 | §8.2 | `ExperienceSamples` のフィールドの並び 1 か所（§6.1 のステージ 1） | 1 |
| Head の選択 | 合成 Head（既存の Head の生成部品を 2 回使う） | D2・§8.6 | 共通部品の Head の Factory の選択。呼ぶたびに Head の設定を受け取るので、`head.*` と `intrinsic.value.head.*` で 1 回ずつ呼べる（D3） | 5 |
| 設定の読み取り | 設定キーと判定の表（`intrinsic.value.head.*`・`intrinsic.reward_scaler.*` の一式を含む） | §8.1 | `DefaultDQNAgentConfig` の関心ごとの関数（内的報酬・Head・learner）。語と数値の範囲の検証は共通の関数（D6）、接頭辞の下を写して記録するのは共通の関数（接頭辞は呼ぶ側が渡す。D14） | 2 |
| 診断の群 | 学習のメトリクス（Head ごとの損失、選んだ行動の Q の平均、勾配の大きさ、`intrinsic_q_share`） | §8.4 | 共通の読み出しへ、群の指標の並びと値を 1 組足す（D12） | 4 |
| §1.1 の表の外 | forward の結果に合わせた Q を足す関数、損失ごとの最適化、Actor の hint の 5 列、行動の選択への効き方の診断、Replay の Push の口 | §6.3・§6.8・§8.2・§8.3 | 本 RFD は変えない。Actor（`dqn::Actor`）は DQN 系の共通部品で、2 つの Agent が同じ型を使う | — |

## 7. 受入

前後比較（1〜3）は、各ステージの終わりに、§6.4 のすべての組で通す。比べる相手は、ステージ 0 で保存した実装前の実行体である（RFD0129 がステージの切れ目で入ったときは、§10 のとおり保存し直したもの。4 と 8 の「実装前の実行体」も同じ）。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | メトリクスの前後比較 | 実装前の実行体と変更後の実行体を、同じ凍結設定で流す。metrics の全 scalar の点を tag・step・値で比べ、すべて一致する。除くのは、時間と資源の tag（`90_perf/`・`91_resource/`・`92_resource/` で始まるもの。実時間で変わるため）だけである。NaN どうしは一致とみなす（RFD0084 の正規化） |
| 2 | Agent が記録する設定の前後比較 | 同じ Run の組で、Run の `config/DefaultDQNAgent.txt` の全行が、並びも含めて一致する。metrics の `DefaultDQNAgent` の json の記録も一致する |
| 3 | ログの前後比較 | 同じ Run の組で、Run フォルダの `<Run 名>.log` のうち WARN と ERROR の行の、文言と順が一致する。行頭の時刻と、Run 名・Run フォルダのパスは置き換えてから比べる（Run ごとに変わるため） |
| 4 | checkpoint | 実装前の実行体が、LunarLander の `run.@rnd` の組の Run の終わりに保存した checkpoint を、新しい実行体が読める。それを読んで同じ短縮条件で 8,704 experience 続けた Run の metrics を、実装前の実行体で同じ checkpoint から同じだけ続けた Run と、1 と同じ比べ方で比べ、一致する |
| 5 | 特性テスト | §6.3 の特性テストが、直す前と直したあとの両方で通る。§6.2 の意図して変える点のテストと、probe 用の乱数を Learner へ直接渡すテスト（D13）だけは、直したあとの形と期待値が実装計画に書かれている。RFD0129 がステージの切れ目で入ったときは、§10 のとおり期待値を直したテストで、以降のステージの前後を確かめる |
| 6 | 標本の並び | 全フィールドに値を入れた標本で、`To`（CPU）、`To`（CUDA。使える環境で）、`ForEachTensor`、`ToString` が、同じフィールドの集合を扱う。`To` は、CPU と CUDA のどちらの device へ送るときも、未定義のフィールドを未定義のまま返す |
| 7 | RainbowAgent | RainbowAgent を構築するテストが通る。設定（ConfigData）から読むとき、`policy_type` が `UQE` と `Greedyy` の Actor を持つ設定は D4 のエラー文の ERROR になり、`tau_rule.num_taus = 0` の Actor を持つ設定は D5 の検証の ERROR になる。設定の DTO を直接作り、`policy_type` が `UQE` の Actor を渡して RainbowAgent を構築しても、D4 のエラー文の ERROR になる。表の下の検索語の 7a〜7d が、どれも 0 件である |
| 8 | 速さ | ステージ 1・3・4 のそれぞれのあと、RFD0125 の `performance` の手順（Release、Atari qbert・IQN・128 lane・batch 256、捨て Run のあとに実装前と変更後を交互に 3 巡以上）で、更新 1 回あたりの実時間の、変更後の中央値が、実装前の最大値以下である。ステージ 3 だけは、同じ構成に `run.@rnd` を重ねる |
| 9 | 番号の直書き | 表の下の検索語の 9a〜9l が、どれも 0 件である。名前で指す指標を持つ群で、指標の並びの隣り合う 2 つを入れ替えると、ビルドが止まる（実装計画で、共通の型に載る群から 1 つと、§8 の U1 で載らないと決めた群のそれぞれを選んで一度ずつ試し、結果を書く） |
| 10 | 内的報酬の移し替え | 表の下の検索語の 10a が 0 件である（内的報酬だけのメンバが、内的報酬のクラスへ移っている）。10b で引いた行のうち、内的報酬の倍率から 1 を引く行（ボーナスの式）が 1 行である。実効ホライズンのログの `std::abs(g - 1.0f)` は数えない |
| 11 | 設定の記録とエラー文 | 表の下の検索語の 11a が 0 件である（記録は `RecordConfigValue` を通す）。11b と 11c が 0 件である（語と数値の範囲のエラー文が D6 の 2 つの形にそろっている） |
| 12 | 既存のテスト | `anet-core-test` の既定のスイートがすべて通る。§6.2 の意図して変える点と、probe 用の乱数の渡し方の直し（D13。§6.6）と、RFD0129 がステージの切れ目で入ったときに §10 のとおり直したもののほかに、期待値を変えたテストが無い |
| 13 | RFD0121 の足す先 | §6.7 の表の、§1.1 の表の場所に当たる行のそれぞれについて、本 RFD のあとの足す先が実装のあとのコードに 1 か所ある。実装計画の完了の時点で、行ごとに関数の名前を書いて確かめる |
| 14 | probe 用の乱数の持ち主 | 表の下の検索語の 14 が 0 件である（Agent が probe 用の乱数を持たない）。`Learner::ConfigureScalarMetricSubscriptions` の引数が購読の一覧だけである。§6.4 の組で、受入 1 が一致する（probe の診断の値が変わらない） |

検索語は、表の中に書くと縦棒が表の区切りと紛れるので、ここにまとめる。1 行が 1 つの検索語で、`#` の行は対象のファイルと種類を示す。正規表現は ERE（`git grep -E -e <検索語>`）で、固定文字列は `git grep -F -e <検索語>` で引く（`-e` は、`- 1.0f)` のように `-` で始まる検索語を引数と取り違えないため）。

```text
# 7a〜7b: core/anet-core/src/rainbow_agent.cpp と core/anet-core/src/default_dqn_agent.cpp。正規表現
(Linear|Dueling|Quantile|QuantileDueling|IQN|IQNDueling)HeadFactory
(TD|QR|IQN)Learner>
# 7c〜7d: core/anet-core/src/rainbow_agent.cpp。固定文字列
"replaybuffer."
ActionPolicy>(
# 9a〜9k: core/anet-core/src/dqn_based_agent.hpp・core/anet-core/src/dqn_based_agent.cpp・core/anet-core/src/default_dqn_agent.cpp。正規表現
request\[[0-9]+\]
kLearnerScheduleMetricKeys\[[0-9]+\]
values\[[0-9]+\] =
i == [0-9]+ \|\|
metric [<>]=? [0-9]+
metric = [1-9][0-9]*
metric - [0-9]+
\.first\([0-9]+\)
\.subspan\([0-9]+\)
request\.size\(\) - [0-9]+
\[[0-9]+\]\.item<
# 9l: core/anet-core/src/dqn_based_agent_test.cpp。正規表現
Keys\)\.(first|subspan)\([0-9]+\)
# 10a: core/anet-core/include/anet/default_dqn_agent.hpp の class DefaultDQNAgent の宣言の範囲。正規表現
lifelong_novelty_|intrinsic_beta_|intrinsic_learn_|intrinsic_diagnostics_
# 10b: core/anet-core/src/default_dqn_agent.cpp と、内的報酬のクラスを置いたファイル。固定文字列
- 1.0f)
# 11a: core/anet-core/include/anet/default_dqn_agent.hpp。固定文字列
my_config_json_[
# 11b: 同じファイル。正規表現
' expected one of|" expected (integer|finite|>)
# 11c: 同じファイル。固定文字列
<< "|" <<
# 14: core/anet-core/include/anet/default_dqn_agent.hpp と core/anet-core/src/default_dqn_agent.cpp。固定文字列
probe_random
```

## 8. 決めること（未決表）

| # | 事項 | 区分 | 確かめ方 | 結果ごとの分かれ方 |
|---|---|---|---|---|
| U1 | 診断の 6 群が、共通の型に載るか | 試して確かめる | 実装計画で、群ごとに、購読の解析・間隔・発火・要求の旗の持ち方を今の形と突き合わせる。載らない候補は、可塑性（3 つのチャネルと、最小の間隔で先に振り落とす判定）と attention（層ごとの要求） | 載る群は共通の型を 1 つずつ持つ。載らない群は今の形を残し、指標を名前で指して、並びと参照が食い違えばビルドが止まる形にだけ直す（D12）。どちらでも §7 の 9 を満たす |
| U2 | `ExperienceSamples::To` の CPU の経路で、未定義の `actions`・`target_returns`・`next_state.terminals` を渡したときの今の挙動 | 確認済み | 2026-11-14、ステージ 0 の特性テストで 3 フィールドを 1 件ずつ未定義にして確認した。結果は検証資材の README に記録した | 3 フィールドとも例外になり、ERROR のログは出なかった。§6.2 の表の 1 に変更前の挙動を書いた。D7 のとおり未定義のまま返す形へ変える。ERROR のログの除外は追加しない |
| U3 | 診断の 6 群をすべて購読する組が、1 組で足りるか | 試して確かめる | ステージ 0 で、§6.4 の組を実装前の実行体で流し、6 群のそれぞれに有限の点があるかを見る | 足りれば 1 組。点を出さない群があれば、その群が点を出す構成を 1 組足し、検証資材の README に書く |

## 9. issue の対応表

| 課題 | 題 | area | 根拠の位置（2026-11-13） | 確定事項 | 受入 |
|---|---|---|---|---|---|
| [issue-20261111-08d557](../management/issues/issue-20261111-08d557.md) | 内的報酬の扱いが DefaultDQNAgent の 5 つの関数と 6 つのメンバに散っている | 324_dqn | [default_dqn_agent.hpp:728-734](../../src/algo/dqn/src/default_dqn_agent.hpp)、[default_dqn_agent.cpp:838](../../src/algo/dqn/src/default_dqn_agent.cpp) | D10、D11、D15、D17 | 1、4、5、8、10 |
| [issue-20261111-288879](../management/issues/issue-20261111-288879.md) | ExperienceSamples のフィールドの並びが分かれ、未定義の tensor の扱いが食い違っている | 318_rl_core | [rl.cpp:773](../../src/core/base/src/rl.cpp)、[rl.hpp:889](../../src/core/base/include/anet/rl.hpp) | D7、D8、D9、D15 | 1、5、6、8 |
| [issue-20261111-72d640](../management/issues/issue-20261111-72d640.md) | DefaultDQNAgent のコンストラクタが長く、RainbowAgent が写して 5 点で食い違っている | 324_dqn | [default_dqn_agent.cpp:53](../../src/algo/dqn/src/default_dqn_agent.cpp)、[rainbow_agent.cpp:62-83](../../src/algo/dqn/src/rainbow_agent.cpp) | D2、D3、D4、D5、D15 | 1、3、5、7 |
| [issue-20261111-e33797](../management/issues/issue-20261111-e33797.md) | DefaultDQNAgentConfig のコンストラクタが 534 行あり、検証と記録が手書きで繰り返されている | 324_dqn | [default_dqn_agent.hpp:116](../../src/algo/dqn/src/default_dqn_agent.hpp) | D5、D6、D14 | 2、3、11、12 |
| [issue-20261111-24d885](../management/issues/issue-20261111-24d885.md) | 診断指標の群を 1 つ足すたびに 12 か所前後へ足し、添字も数値で直書きしている | 321_agent_core | [dqn_based_agent.hpp:961](../../src/core/base/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp:2750](../../src/core/base/src/dqn_based_agent.cpp) | D12、D13、D19 | 1、5、8、9、14 |

## 10. ほかの RFD との関係

- [RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)（価値の分解）は、本 RFD のあとに実装する（RFD0121 の D18）。RFD0121 の課題は、§9 の 5 件を待つ。RFD0121 の「127 との調整」（Trace のボーナスの列の式）は、D11 で 1 か所にした式を直せば済む。RFD0121 が足すものごとの足す先は §6.7 に並べた。
- [RFD0173](0173_intrinsic_reward_novelty_10rfd.md)（内的報酬と方策の組）も、内的報酬の扱いと設定を広げる。本 RFD の直しのあとに足す。
- [RFD0127](0127_eval_trace_intrinsic_reward_columns_10rfd.md) は実装済み（§2.6）。本 RFD は、その実装のあとのコードを直す。写しのロックの取り方（RFD0127 の D4）は変えない。
- [RFD0129](0129_eval_trace_bonus_beta_before_learning_10rfd.md)（Evaluation Trace のボーナスの β を、学習の入口を通る前から用意する。未実装）は、D11 で内的報酬のクラスへ移す箇所（構築時の β、学習を止めた構成での学習の入口）を変える。どちらを先に実装してもよい（RFD0129 の D8）。RFD0129 は、本 RFD のステージの切れ目（あるステージの前後比較を通したあと。本 RFD を始める前も含む）にだけ入れ、ステージの途中には入れない。ステージの途中で入れると、そのステージの前後比較で、本 RFD の直しによる差と RFD0129 による差を分けられないため（G3、D1）。切れ目で入れたときは、次のステージへ進む前に、その時点のコードで実装前の実行体と設定を保存し直し（§6.4。以降のステージの前後比較はこれと比べる）、§6.3 の特性テストのうち RFD0129 で挙動が変わるもの（学習を止めた構成など）の期待値を、RFD0129 のあとの挙動に直す。β の評価の置き場は、ステージ 3 より前に入ったなら、ステージ 3 で内的報酬のクラスへ移す。ステージ 3 のあとに入ったなら、RFD0129 がクラスの中に置く（RFD0129 の D8）。
- [RFD0170](0170_nn_trace_capture_unification_10rfd.md) の再グリルで出た、Learner の購読の見直し案は入れない（§1.3）。
- [RFD0161](0161_observation_pipeline_10rfd.md)（観測パイプライン）と [RFD0165](0165_nn_head_projection_separation_10rfd.md)（Head の射影の分離）は待たない（§1.3）。RFD0161 は、§6.1 のステージ 5 で分けた「入力の spec を作る」関数を触ることになる。
- [RFD0182](0182_source_layout_10rfd.md)（ソースの配置）は、DQN 系の共通部品を `dqn_based` のライブラリに置く案を持つ。D3 で共通部品へ移した処理は、その置き場に乗る。
