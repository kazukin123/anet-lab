# PRD122: 小さなリファクタリングとテストの空白をまとめて片づける

> 2026-11-10 起票（暫定。質疑を省き、既定を §4 の未決表に置いた）。行番号は起票時点の作業ツリーの位置で、実装時は近傍の記述で探し直す。
> 起点: 2026-11-10 の課題の整理の枠。ユーザーが、type がリファクタリングで手間の小さい課題 8 件を 1 本の PRD に束ねるよう指示した。AI は、ビルドと単体テストで閉じる core の 6 件と、Runner を動かして確かめる GUI の 2 件で確かめ方が違うことを示し、人の手間（グリル・計画の承認・レビュー・コミットの回数）で比べて、1 本の PRD の中で実装を 2 つのステージに分ける形を出した。ユーザーは、その形で暫定の PRD を起こすよう指示した。グリルと実装は別の枠で行う。
> 題材: [issue-20261103-d4aa77](../management/issues/issue-20261103-d4aa77.md)、[issue-20261103-d337bd](../management/issues/issue-20261103-d337bd.md)、[issue-20261103-60f11c](../management/issues/issue-20261103-60f11c.md)、[issue-20261103-03a1b6](../management/issues/issue-20261103-03a1b6.md)、[issue-20261103-741251](../management/issues/issue-20261103-741251.md)、[issue-20261103-c1e48b](../management/issues/issue-20261103-c1e48b.md)、[issue-20261103-58bf49](../management/issues/issue-20261103-58bf49.md)、[issue-20261103-e088eb](../management/issues/issue-20261103-e088eb.md)。
> 関連: [PRD119](119_small_doc_comment_config_fixes_10prd.md)（小さな直しを 1 本に束ねた前例）、[PRD117](117_warn_policy_and_unread_config_10prd.md)（scaler の WARN と `post_process_type` の構築時の検証）、[PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md)（QValuePanel の Q と手動の行動の時点）、[PRD070](done/070_device_spec_auto_10prd.md)（device の一括検査）、[PRD035](done/035_approx_actor_priority_per_10prd.md)（初期優先度の completer）、[PRD105](done/105_eval_trace_arrow_key_forced_action_10prd.md)（矢印キーと強制の行動）。

## 1. 問題

### 1.1 既定で動く経路の検査と計算に、テストが無い

次の 3 か所は、壊してもどのテストも落ちない。

- ReplayBuffer のコンストラクタは、PER では priority store と初期優先度の completer を対で持ち、uniform ではどちらも持たないことを検査している。片方だけを渡したときに止まることを確かめるテストが無い（d4aa77）。
- 報酬の scaler の `use_auto_post_scale`、観測の正規化の robust update と SymLog の後処理は、DefaultDQNAgent の既定のプロファイルで有効だが、テストはどれも通らない（d337bd）。
- MuZero の Replay の、エピソード終端での価値ターゲットの打ち切りと、unroll 窓の mask は、学習ターゲットそのものを決めるが、テストが無い（60f11c）。

式や不等号を取り違えても気づけず、既定の設定で動く Run の学習が黙って変わる。

### 1.2 dqn_based_agent.hpp の形が崩れている

- core のヘッダで、`dqn_based_agent.hpp` だけがインクルードガードを持たず、先頭のファイル名のコメントが旧名のままである（03a1b6）。
- `BatchUpdateResult::GetScalar` が、ヘッダのクラス定義の中に 241 行・key の比較 43 個の if の連鎖として書かれている（741251）。診断値を 1 つ足すたびに、このヘッダを読み込む 8 本の .cpp が作り直しになり、長い連鎖で重複や届かない分岐を見落とす。

### 1.3 device の既定値を二重に持っている

`InitRL` の device の検査が、`agent.device`・`env.device`・`run.eval_device` の既定値を、それぞれを持つ設定のクラスとは別に書いている（c1e48b）。片方だけを変えると、検査が見る値と実際に使う値が食い違う。

### 1.4 GUI の長い関数と、規約に反する型の照会

- `QValuePanel::Update` が約 310 行あり、統計の計算、Grid の更新、HeatMap 用のデータの加工を 1 つの関数で続けて行う（58bf49）。表示のモードを足すときに、Grid の差分更新を壊しても気づきにくい。
- `Panel::OnKey` が、キーの多重の転送を防ぐ判定に `dynamic_cast` を 3 回使う。規約は production のコードでの `dynamic_cast` を禁じていて、品質確認の機械検査が gui.cpp を変えるたびに 3 件を出す（e088eb）。

## 2. 現状（2026-11-10 に確認したこと）

### 2.1 題材ごとの事実

| 題材 | 事実 |
| --- | --- |
| d4aa77 | [replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の 1530 行のコンストラクタが、1555〜1557 行で `priority_store_` と `initial_priority_completer_` の片方だけがあると `ANET_SYSTEM_ERROR` にする。クラスは [replay_buffer_impl.hpp](../../core/anet-core/src/replay_buffer_impl.hpp) の 449 行の `DefaultReplayBuffer` で、生成は replay_buffer_impl.cpp の 2629 行のファクトリが行う。[replay_buffer_test.cpp](../../core/anet-core/src/replay_buffer_test.cpp) の 1804 行のテストは `CreateInitialPriorityCompleter` が PER のときだけ completer を作ることまでを見て、コンストラクタに片方だけを渡す経路は通らない |
| d337bd | [scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp) の 892〜894 行は、`use_auto_post_scale` のとき post_scale を `reference_q_std * (1 - gamma)` にする。449〜505 行の robust update は、統計の件数が `robust_warmup_count` に届くまでは無条件に更新し、そのあとは平均からの偏差が `robust_std_threshold * std` を超える要素を 1 つでも含む行を、統計の更新から外す。392〜405 行の SymLog は `scale * sign(x/scale) * log1p(|x/scale|)` で、`scale` は `post_process_threshold` が正ならその値、そうでなければ 1 である。[agent.txt](../../apps/runner/config/agent.txt) は 2 行で `DefaultDQNAgent.$ = @baseline` とし、`@baseline` の 265 行で `use_auto_post_scale ?= true`、276・278・279 行で robust update と SymLog（`post_process_type ?= 1`）を有効にしている。[scaler_test.cpp](../../core/anet-core/src/scaler_test.cpp) のテスト 3 本（6・24・49 行）は、どれも `use_auto_post_scale = false` で、robust update と SymLog を通らない |
| 60f11c | [muzero_rb.cpp](../../core/anet-core/src/muzero_rb.cpp) の 66 行の `MuZeroReplayBuilder::ProcessQueue` は、98〜121 行で、最新の step が done か truncated のとき、残りの各 step の価値ターゲットを終端までの割引報酬で作ってキューを空にする。truncated のときは最後の step の `root_value` で残りを補う（113〜115 行）。170 行の `MuZeroUniformSampler::Sample` は、`unroll_steps + 1` の窓に行動・報酬・価値・policy・mask を詰め、done の step の次からと、lane の記録の末尾の先を、mask 0・価値 0 で埋める（222〜250 行）。287 行の `MuZeroReplayBuffer::Push` は、batch の経験を lane ごとのキューへ渡す。クラスは [muzero_rb.hpp](../../core/anet-core/src/muzero_rb.hpp) の 58〜116 行にある。これらを直接使うテストは [prd101_muzero_test.cpp](../../core/anet-core/src/prd101_muzero_test.cpp) の 4 行の 1 本だけで、終端の無い 4 step の n-step の割引を見るだけである。MuZeroAgent は試作の Agent である（[325](../wiki/categories/325_muzero.md)） |
| 03a1b6 | [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) の 1 行は `// rainbow_agent_impl.hpp` で、`#pragma once` も `#ifndef` のガードも無い。`core/anet-core/src` と `core/anet-core/include/anet` のヘッダの先頭 15 行を調べると、ガードの無いヘッダはこのファイルだけだった。このヘッダを読み込むのは、`default_dqn_agent.cpp`・`dqn_based_agent.cpp`・`rainbow_agent.cpp` と、テスト 5 本（`dqn_based_agent_test.cpp`・`dqn_based_test.cpp`・`dqn_munchausen_test.cpp`・`intrinsic_test.cpp`・`prd101_ladder_test.cpp`）の 8 本である |
| 741251 | dqn_based_agent.hpp の 219 行の `class BatchUpdateResult` の中で、`GetScalar` が 285〜525 行の 241 行を占め、`key == "` の比較が 43 個ある。先頭で attention entropy の key を名前から層の番号に読み、そのあとに損失・Q・PER・勾配・可塑性・policy churn・noisy・replay fit などの群が続く。値が成り立たない既知の key には NaN を、未知の key には `std::nullopt` を返す（[実装 品質確認観点](../quality/impl.md) の「GetScalar」、項番 42）。テストから `GetScalar("<key>")` を呼ぶ行は 8 ファイルに 466 行あるが、Env の結果や ImageCls など別の型の呼び出しも含み、この 43 個の key のすべてを通るかは確かめていない |
| c1e48b | [init.cpp](../../core/anet-core/src/init.cpp) の 95〜98 行が、`{"agent.device", "auto"}`・`{"env.device", "cpu"}`・`{"run.eval_device", "auto"}` の表を持ち、キーが無ければ表の既定で `ParseDevice` にかけ、`auto` でない cuda を明示した cuda として集める。既定値を持つ設定のクラスは、[agent.hpp](../../core/anet-core/include/anet/agent.hpp) の 394 行の `DefaultAgentFactoryConfig`（prefix `agent`、`auto`）、[env.hpp](../../core/anet-core/include/anet/env.hpp) の 348 行の `BatchEnvBuilderConfig`（prefix `env`、`cpu`）、[trainer.cpp](../../core/anet-core/src/trainer.cpp) の 808 行の `RunManager::Config`（prefix `run`、`auto`。trainer.cpp の中で定義した型）である。表の既定 3 つは、どれも `ParseDevice` で失敗せず、明示した cuda にも数えられない。PRD070 の D5 は「cuda を指定している全キー」を一括で報告する決定である（[PRD070](done/070_device_spec_auto_10prd.md) の 68 行）。検査のテストは [init_test.cpp](../../core/anet-core/src/init_test.cpp) の 29・44 行（`[prd070]`）と、空の設定で呼ぶ 192 行にある。`ConfigData` はキーの有無を問う `Has` を持つ（[config.hpp](../../core/anet-core/include/anet/config.hpp) の 70 行） |
| 58bf49 | [QValuePanel.cpp](../../apps/runner/src/QValuePanel.cpp) の `QValuePanel::Update` は 420〜728 行の約 310 行で、438 行から統計量の計算、449 行から Grid の更新、522 行から HeatMap のデータの加工（Advantage の変換、Histogram 化、色付けのための正規化）が続く。Grid の更新は、選択行のハイライトを選択が変わったときだけ塗り直し、列の幅は値が広がったときだけ当て直す（毎 step 全行を塗り直さないための差分更新）。表示のモード（Histogram・Advantage・Log）の判定は関数の先頭で取り、Grid と HeatMap の両方の分岐で使う。QValuePanel を直接見るテストは無く、`EvalTracePanel_test.cpp` と `RunnerFrame_test.cpp` が Runner の中の部品として触るだけである |
| e088eb | [gui.cpp](../../core/anet-core/src/gui.cpp) の 75 行で `Panel` が `wxEVT_CHAR_HOOK` を `Panel::OnKey`（89 行）に結び、95・100・108 行の `dynamic_cast` で、発生元が別の `Panel` か、発生元から自分までの親に `Panel` があるかを調べる。あれば `Skip` だけをして、無ければ（自分が最も下の `Panel` なら）`ForwardedKeyEvent` に包んで転送する。転送したキーは `RunnerFrame::OnKey` が解釈する（[PRD922](922_input_mapping_joystick_10prd.md) の 18〜19 行）。規約は [実装 品質確認観点](../quality/impl.md) の 230 行（production のコードで `dynamic_cast` を使わない）で、機械検査は同じ文書の項番 2（45 行）、取り出し口を足すときの項番 58（256 行）である |

### 2.2 規約と過去の判断

- リファクタリングのガードレール: テストの無い箇所は、特性テストを先に足す。hot path（Sample・Push・Learn、step ごとの経路）に触るときは、性能が変わらないことをブレ幅の基準で確かめる（[リファクタリング観点](../quality/refactor.md) の 53〜57 行）。
- `GetScalar` の戻り値: 未知の key だけを `std::nullopt`、既知の key で値が成り立たないときは NaN にする（[実装 品質確認観点](../quality/impl.md) の「GetScalar」）。
- .cpp の補助: 外で使わない補助のクラスは、持ち主のクラスの private な入れ子か .cpp の無名名前空間に置く（実装 品質確認観点の 233 行、項番 60）。.cpp の名前空間とリンケージは [PRD079](079_cpp_internal_linkage_convention_10prd.md) の D1〜D4 による（同 221 行）。
- Runner のキーとクリック: Trace モードでないときの矢印キー、テンキー、QValuePanel のクリック、Ctrl、Space の、行動の割り当てと step の進み方を変えない（[110 RunnerApp の地図](../wiki/apps/110_runner_app.md) の「変えるときの条件」、出典は PRD105 §2 の NG5）。Train View が表示中の振る舞いを変えるときは、GUI のテスト `[eval_steps_menu]`・`[prd105_ui_keys]`・`[prd109_ui]` が通ることを確かめる（同、出典は PRD110 §7 の受入 1）。
- 起動の順: InitRL を含む、表示より前の起動の段の順序を変えない（同、出典は PRD109 §2 の NG4・D4）。
- 課題の状態: `解決` にするのは開発者で、AI は対応内容と AI の判定（`ai_judge`）を書く（[anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: 題材の 8 件の課題をすべて片づける。
- G2: 外から見える振る舞いを変えない。設定のキーと既定値、メトリクスの key と値、画面の表示と操作、ログの文言は、直す前と同じにする。変わるのは、テストが増えることと、コードの置き場と形だけである。
- G3: 足したテストが、課題の「影響」に書いた壊し方で落ちる。

### 3.2 非ゴール

| 退ける案 | 理由 | 本当の制約か |
| --- | --- | --- |
| scaler の `Unknown squash_type` の WARN と、`post_process_type` の値域の構築時の検証を一緒に直す | [PRD117](117_warn_policy_and_unread_config_10prd.md) の D7 と §5.5 の範囲である。本 PRD は今の振る舞いをテストで固めるだけにする | 制約ではない（別の関心。PRD117 の WARN の線引きと一緒に決める） |
| テストで見つかった食い違い（MuZero の打ち切りの式など）を本 PRD で直す | 振る舞いを変えると G2 に反し、直し方の判断が要る | 制約ではない。見つかったら起票の候補にする（§4 の 12） |
| `GetScalar` の key の名前を変える・整理する | 設定の `metrics.scalar` が key の名前で参照している。名前を変えると設定とテストの移行が要る | 制約ではない（別の関心。メトリクスの名前の整理は [PRD108](108_metrics_tag_numbering_10prd.md) の範囲） |
| QValuePanel の Q と手動の行動の時点のずれを一緒に直す | [PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md) の範囲で、表示の振る舞いを変える | 制約ではない（別の関心） |
| `Panel` のキーの転送の経路（`wxEVT_CHAR_HOOK` の止め方、`RunnerFrame::OnKey` の解釈）を作り替える | 振る舞いを変えるので G2 に反する。キーの割り当ての設定化は [PRD922](922_input_mapping_joystick_10prd.md) の範囲である | 制約ではない（別の関心） |
| ほかのヘッダの先頭のコメントを一括で点検する | ガードの無いヘッダは dqn_based_agent.hpp だけだった（§2.1）。先頭のコメントの食い違いは、見つかった 1 件だけを直す | 制約ではない。見つかれば別の課題にできる |

### 3.3 成功の測り方

- 題材の 8 件を、実装のあとに解決にできる（§6 の受入がすべて通る）。
- 足したテストが、課題の「影響」に書いた壊し方（検査を消す、式を変える、不等号を逆にする、mask を 1 に戻すなど）で落ちる。実装の工程で一時的にコードを変えて確かめ、結果を実装記録に残す（§6 の A1〜A3・A5）。
- 品質確認の機械検査（[実装 品質確認観点](../quality/impl.md) の項番 2）が、gui.cpp で 0 件になる。

## 4. 未決と暫定の既定

暫定の起票なので、確定した決定は無い。どの行もグリルで見直す。実装契約（§5）と受入（§6）は、この表の暫定の既定を前提に書き、前提にした行の番号を名指す。

| # | 事項 | 暫定の既定 | 理由と依存 |
| --- | --- | --- | --- |
| 1 | 束ね方と実装の順 | 1 本の PRD にし、実装を 2 つのステージに分ける。ステージ 1 は core の 6 件（d4aa77・d337bd・60f11c・03a1b6・741251・c1e48b）、ステージ 2 は GUI の 2 件（58bf49・e088eb）。ステージ 1 だけでコミットでき、その 6 件の課題はステージ 2 を待たずに解決にできる | ステージ 1 はビルドと単体テストで閉じ、ステージ 2 は Runner を動かして人が確かめる項を持つ（§2.1）。人の手間（グリル・計画の承認・レビュー・コミットの回数）は 1 本のほうが少なく、ステージに分ければ GUI の確認を待たずに core の 6 件を閉じられる（起点の会話の方向）。依存: なし |
| 2 | d4aa77 のテストの形 | `DefaultReplayBuffer` のコンストラクタへ、priority store と completer の 4 通りの組み合わせを直接渡す。両方ありと両方なしは構築でき、片方だけの 2 通りは `ANET_SYSTEM_ERROR` で止まり、文言に `must be configured together` が入る | 不変条件は「対で持つ」なので、4 通りで網羅できる（§2.1）。ファクトリを通すと片方だけの組み合わせを作れない。依存: なし |
| 3 | d337bd のテストの範囲 | 3 本を足す。(a) `use_auto_post_scale = true` で、gamma と `reference_q_std` から手で計算した post_scale が報酬の変換に効く。(b) robust update で、warmup の件数までは外れ値も統計に入り、そのあとは閾値を超える要素を含む行だけが統計から外れる。(c) SymLog で、`post_process_threshold` が正のときと 0 のときの出力を手で計算した値と比べる | 既定のプロファイルで有効な 3 つの経路を、課題の対応方針どおりに通す（§2.1）。(b) の warmup の境目は、境目の前後の 1 件ずつで見る（件数の端の値）。依存: PRD117 が `post_process_type` の検証を足しても、有効な値（SymLog）だけを使うので影響しない |
| 4 | 60f11c のテストの範囲 | 短いエピソードを手で作り、次を見る。(a) done で終わるキューの打ち切り（残りの各 step の価値ターゲットが終端までの割引報酬）。(b) truncated で終わるキューの、`root_value` による補い。(c) 窓が終端をまたぐときの mask と価値（終端の step までは mask 1、その次から mask 0・価値 0）。(d) 窓が lane の記録の末尾を越えるとき。(e) `Push` が lane ごとのキューへ分けること | 学習ターゲットを決める箇所を、課題の対応方針どおりに並べる（§2.1）。サンプラーの行動の埋め草は乱数なので値を見ず、mask だけを見る。依存: なし |
| 5 | 03a1b6 の直し方 | 先頭に `#pragma once` を足し、先頭のコメントを `// dqn_based_agent.hpp` にする | core のほかのヘッダはすべてガードを持つ（§2.1）。依存: なし |
| 6 | 741251 の分け方 | `GetScalar` の本体を dqn_based_agent.cpp へ移し、群ごとの private な関数に分ける。各関数は、自分の群の key なら値（成り立たなければ NaN）を、自分の群の key でなければ「当たらない」を返し、`GetScalar` は群を順に引いて、どれにも当たらなければ `std::nullopt` を返す。key から値への 1 枚の表にはしない | attention entropy の key は名前から層の番号を読み、PER の ESS などは引くときに計算するので、1 枚の表に載らない key がある（§2.1）。群ごとの関数なら、計算の要る key と関係する値をまとめて置ける。依存: 7 |
| 7 | 741251 の特性テスト | 移す前に、43 個の key のすべてと未知の key 1 つについて、値を決めて作った `BatchUpdateResult` の戻り値（値、NaN、`std::nullopt` の別）を固定するテストを足し、移したあとも同じテストがそのまま通る | テストの無い箇所は特性テストを先に足す（§2.2）。既存のテストは key ごとに散らばっていて、43 個を網羅するか分からない。依存: なし |
| 8 | 741251 の性能 | 性能の受入は置かない | `GetScalar` を呼ぶ側と回数は変えず、比較の連鎖が群ごとの関数の呼び出しに替わるだけである。メトリクスの購読の間隔で呼ばれ、学習の 1 step の計算の経路（Sample・Push・Learn の計算そのもの）ではない。測るなら、短い Run の throughput を前後でラウンドロビンで比べる（3 巡以上・群ごとの中央値）。依存: 6 |
| 9 | c1e48b の直し方 | `InitRL` は、設定にあるキーだけを検査し、無いキーは検査しない（`Has` で問う）。init.cpp の既定値の表を消す | 表の既定 3 つはどれも検査で失敗せず、明示した cuda にも数えられないので、無いキーを飛ばしても今の結果は変わらない（§2.1）。PRD070 の D5 が報告するのは「指定している」キーで、書かれていないキーは対象ではない。既定値を持つクラスの定数を引く案は、`RunManager::Config` が trainer.cpp の中の型なので、型を外へ出す変更が要る。依存: なし |
| 10 | 58bf49 の分け方 | `Update` を、統計の計算、Grid の更新、HeatMap のデータの加工の 3 つの private な関数に分け、`Update` はそれを順に呼ぶ。表示のモードの判定は `Update` の先頭で 1 回取り、引数で渡す。Grid の差分更新（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）はそのまま Grid の関数に残す | 課題の対応方針どおりの 3 段である（§2.1）。モードの判定を 1 回にするのは、今と同じく Grid と HeatMap で同じ判定を使うため。依存: なし |
| 11 | 58bf49 の性能の確かめ方 | 実時間の受入は置かない。差分更新の 2 つの条件（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）が分けたあとの Grid の関数に残ることを、その行を実装記録に書いて確かめる | `Update` は画面の描画の timer で呼ばれ、学習の経路ではない。差分更新の効き目は、毎 step 全行を塗り直すかどうかで決まり、関数を分けるだけでは変わらない。実時間を測るなら、同じ checkpoint の評価で `Update` の区間の時間を前後で比べる（3 巡以上・中央値）。依存: 10 |
| 12 | テストで今の振る舞いの誤りが見つかったとき | 本 PRD では直さない。テストは今の振る舞いを固め、誤りと思われる点は工程末に起票の候補として示す | 振る舞いを変えると G2 に反し、直し方の判断が要る（§3.2）。依存: 3・4・7 |
| 13 | e088eb の判定の仕方 | `Panel` が、生きている `Panel` の集まりに自分を登録し（構築で足し、破棄で外す）、`OnKey` は、発生元と、発生元から自分までの親が、その集まりに入っているかで判定する。キーを転送するかどうかの結果は今と同じにする | 型を照会せずに「`Panel` かどうか」を答えられ、取り出し口（`AsPanel` など）を wxWindow に足せない（wx の型は変えられない）。イベントに印を付ける案は、`wxKeyEvent` に利用者の印を付ける口が無く、別の型の照会か、イベントの中身の流用になる。依存: なし |
| 14 | e088eb の確かめ方 | 機械検査（gui.cpp と gui.hpp の `dynamic_cast` が 0 件）、GUI のテスト `[prd105_ui_keys]`・`[eval_steps_menu]`・`[prd109_ui]` の合格、人が Runner で確かめる項（Live モードで矢印キー・テンキー・Space・Ctrl を 1 回押すと 1 回だけ効く、Evaluation Trace の pane にフォーカスがあるときも同じ、QValuePanel のクリックで行動が 1 回だけ選ばれる） | キーの多重の転送は、子の Panel の中にフォーカスがあるときに起きうる。GUI のテストはキーの経路の一部しか通らないので、人の確かめを残す（§2.2）。依存: 13 |
## 5. 実装契約

### 5.1 ステージと直す箇所

| ステージ | 題材 | ファイルと位置 | 直し方 | 未決表 |
| --- | --- | --- | --- | --- |
| 1 | d4aa77 | core/anet-core/src/replay_buffer_test.cpp | コンストラクタの 4 通りの組み合わせのテストを足す | 2 |
| 1 | d337bd | core/anet-core/src/scaler_test.cpp | 3 本のテストを足す | 3 |
| 1 | 60f11c | core/anet-core/src の MuZero の Replay のテスト（置き場は実装計画で決める） | 終端・truncated・mask・`Push` のテストを足す | 4 |
| 1 | 03a1b6 | core/anet-core/src/dqn_based_agent.hpp の 1 行 | `#pragma once` を足し、先頭のコメントを直す | 5 |
| 1 | 741251 | dqn_based_agent.hpp の 285〜525 行、dqn_based_agent.cpp、特性テスト | 特性テストを先に足し、本体を .cpp へ移して群ごとに分ける | 6・7・8 |
| 1 | c1e48b | core/anet-core/src/init.cpp の 95〜98 行、init_test.cpp | 既定値の表を消し、設定にあるキーだけを検査する。キーの無い設定のテストを足す | 9 |
| 2 | 58bf49 | apps/runner/src/QValuePanel.cpp の 420〜728 行、QValuePanel.hpp | `Update` を 3 つの関数に分ける | 10・11 |
| 2 | e088eb | core/anet-core/src/gui.cpp の 75〜125 行、core/anet-core/include/anet/gui.hpp | `dynamic_cast` を、生きている Panel の集まりで判定する形に替える | 13・14 |

ステージ 1 の 6 件は互いに独立で、どの順に入れてもよい。741251 は、特性テスト（未決表の 7）を、本体を移すより先に入れる。

### 5.2 移行対象と、触らないもの

振る舞いを変えないので、設定・サンプル・設計文書に移すものは無い。移行対象は、§5.1 の表のファイルと、題材の 8 件の課題の記録（対応内容と AI の判定）である。

変える経路を書いている箇所を検索した結果は次のとおりで、直すものは無い。

- `InitRL` の device の検査: `git grep -n -i -e "InitRL" -e "explicit_cuda" -- docs/design AGENTS.md CONTEXT.md` の当たりは、設計文書 110 の日英の 4 行（Agent の factory を `InitRL()` で登録する記述）だけで、device の既定値の記述ではない。
- キーの転送の判定: `git grep -n -e "OnKey" -e "多重発火" -e "ForwardedKeyEvent" -e "EVT_FORWARDED_KEY"` を `docs`・`AGENTS.md` に当て、過去の PRD（`docs/memo/done`）と課題の記録を除いた当たりは、PRD922・PRD923・PRD944 の転送の経路の記述と、wiki の 319 の課題の行である。経路は変えないので直さない。
- `GetScalar`: `git grep -n "GetScalar" -- docs/design AGENTS.md CONTEXT.md` の 40 件は、戻り値の意味、key の名前と意味、値の作り方、汎用層の責任境界の記述で、本体をどこに置くか（ヘッダの中か .cpp か）を書くものは無い。
- 旧名のコメント: `git grep -n "rainbow_agent_impl"`（課題の記録を除く）の当たりは、dqn_based_agent.hpp の 1 行だけである。

触らないもの:

- `GetScalar` の key の名前と、それを参照する `metrics.scalar` の設定（§3.2）。
- PRD117 が扱う scaler の WARN と `post_process_type` の検証（§3.2）。
- QValuePanel の `CreateData`（315〜397 行。PRD948 の範囲）と、`RunnerFrame::OnKey` の解釈（PRD922 の範囲）。
- 過去の PRD・実装記録・課題の単票の題と事象に書かれた当時の行番号。

### 5.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
| --- | --- | --- | --- |
| `BatchUpdateResult`（`anet::rl::dqn`） | 変更 | `GetScalar` の宣言だけをヘッダに残し、本体を群ごとの private な関数に分けて .cpp に置く | dqn_based_agent |
| `QValuePanel` | 変更 | `Update` を、統計の計算・Grid の更新・HeatMap のデータの加工の 3 つの private な関数に分ける | QValuePanel |
| `Panel` | 変更 | 生きている `Panel` の集まりへの登録と、それによるキーの転送の判定を持つ | gui |

`DefaultReplayBuffer`・scaler・MuZero の Replay のクラスと、`InitRL` の関数の宣言は変えない。

### 5.4 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-d4aa77](../management/issues/issue-20261103-d4aa77.md) | DefaultReplayBufferのコンストラクタの、priority storeとcompleterを対で持つ不変条件にテストが無い | 318_rl_core | core/anet-core/src/replay_buffer_impl.cpp:1555、core/anet-core/src/replay_buffer_test.cpp:1804 | 未決 1・2 | A1・A9・A10 |
| [issue-20261103-d337bd](../management/issues/issue-20261103-d337bd.md) | 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効） | 314_tensor | core/anet-core/src/scaler_impl.cpp:892、core/anet-core/src/scaler_impl.cpp:449、core/anet-core/src/scaler_impl.cpp:392、core/anet-core/src/scaler_test.cpp:12 | 未決 1・3・12 | A2・A9・A10 |
| [issue-20261103-60f11c](../management/issues/issue-20261103-60f11c.md) | MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い | 318_rl_core | core/anet-core/src/muzero_rb.cpp:98、core/anet-core/src/muzero_rb.cpp:222、core/anet-core/src/muzero_rb.cpp:287、core/anet-core/src/prd101_muzero_test.cpp:4 | 未決 1・4・12 | A3・A9・A10 |
| [issue-20261103-03a1b6](../management/issues/issue-20261103-03a1b6.md) | dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い | 321_agent_core | core/anet-core/src/dqn_based_agent.hpp:1 | 未決 1・5 | A4・A9・A10 |
| [issue-20261103-741251](../management/issues/issue-20261103-741251.md) | BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖 | 321_agent_core | core/anet-core/src/dqn_based_agent.hpp:285 | 未決 1・6・7・8・12 | A5・A9・A10 |
| [issue-20261103-c1e48b](../management/issues/issue-20261103-c1e48b.md) | InitRLがagent.device・env.device・run.eval_deviceの既定値を、所有するConfigとは別に持っている | 312_util | core/anet-core/src/init.cpp:95、core/anet-core/include/anet/agent.hpp:394、core/anet-core/include/anet/env.hpp:348、core/anet-core/src/trainer.cpp:808 | 未決 1・9 | A6・A9・A10 |
| [issue-20261103-58bf49](../management/issues/issue-20261103-58bf49.md) | QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う | 110_runner_app | apps/runner/src/QValuePanel.cpp:420 | 未決 1・10・11 | A7・A9・A10 |
| [issue-20261103-e088eb](../management/issues/issue-20261103-e088eb.md) | Panel::OnKeyがproductionコードでdynamic_castを3回使う | 319_gui | core/anet-core/src/gui.cpp:95、docs/quality/impl.md:230 | 未決 1・13・14 | A8・A9・A10 |

## 6. 検証と受入

A1〜A6 はステージ 1、A7・A8 はステージ 2 の受入である。A9・A10 は、ステージごとに、そのステージの題材について確かめる。

### A1. ReplayBuffer のコンストラクタ（d4aa77）

- replay_buffer_test.cpp に、priority store と completer の 4 通りの組み合わせでコンストラクタを呼ぶテストがあり、両方ありと両方なしは構築でき、片方だけの 2 通りは例外を投げ、例外の文言に `must be configured together` が入る（未決表の 2）。
- 実装の工程で、replay_buffer_impl.cpp の 1555〜1557 行の検査を一時的に消すと、このテストが落ちる。結果を実装記録に書く。

### A2. scaler の 3 つの経路（d337bd）

- scaler_test.cpp に、`use_auto_post_scale = true`、robust update、SymLog のテストが 1 本以上ずつあり、期待値はテストの中で gamma・`reference_q_std`・閾値・scale から計算した値である（未決表の 3）。テストの gamma は 0.5 以外の値にし、SymLog は `post_process_threshold` を 1 以外の正の値にする場合を含める（下の一時的な変更で値が変わるようにするため）。
- 実装の工程で、次の 3 つを 1 つずつ一時的に変えると、対応するテストが落ちる。post_scale の式の `(1 - gamma)` を `gamma` にする。robust update の `deviation > threshold` を `deviation < threshold` にする。SymLog の `* scale` を外す。結果を実装記録に書く。

### A3. MuZero の Replay（60f11c）

- 未決表の 4 の (a)〜(e) の各場面のテストがあり、価値ターゲットと mask の期待値を、テストの中で報酬と割引率から計算した値と比べる。
- 実装の工程で、次の 2 つを 1 つずつ一時的に変える。muzero_rb.cpp の `steps_to_done` を 1 減らす。終端のあとの `seq_masks[k]` を 1 にする。どちらの変更でも、対応するテストが落ちる。結果を実装記録に書く。

### A4. インクルードガード（03a1b6）

`git grep -n "#pragma once" -- core/anet-core/src/dqn_based_agent.hpp` が 1 件で、`git grep -n "rainbow_agent_impl" -- core/anet-core/src/dqn_based_agent.hpp` が 0 件である。

### A5. `GetScalar` の置き場（741251）

- 特性テスト（未決表の 7）が、43 個の key と未知の key 1 つを引き、値・NaN・`std::nullopt` の別を確かめる。本体を移す前の作業ツリーと、移したあとの作業ツリーの両方で通る。
- dqn_based_agent.hpp の `class BatchUpdateResult` の定義（219 行の `class BatchUpdateResult` から、対応する `};` まで）の中で、`GetScalar` が宣言だけ（`;` で終わる行）であり、`git grep -c 'key == "' -- core/anet-core/src/dqn_based_agent.hpp` が、直す前の数から 43 を引いた数である。
- 実装の工程で、移したあとの群の関数のどれか 1 つで、key の 1 つの綴りを一時的に変えると、特性テストが落ちる。結果を実装記録に書く。

### A6. device の既定値（c1e48b）

- `git grep -n -E '\{"(agent\.device|env\.device|run\.eval_device)", "' -- core/anet-core/src/init.cpp` が 0 件である。
- init_test.cpp の `[prd070]` のテストが、そのまま通る。
- 3 つのキーのどれも書かれていない設定で `InitRL` が止まらず、明示した cuda として何も報告しないことを確かめるテストがある（未決表の 9）。

### A7. QValuePanel の関数の分け方（58bf49）

- QValuePanel.cpp に、統計の計算・Grid の更新・HeatMap のデータの加工の 3 つの関数の定義があり、`QValuePanel::Update` の本体がそれらを呼ぶ（未決表の 10）。名前はクラス一覧と実装計画による。
- 人が Runner で確かめる: 同じ checkpoint で評価を流し、Q-Values の表示を、モードの組み合わせ（Histogram・Advantage・Log の on と off）ごとに、直す前と見比べて同じである。行の選択とクリックで、ハイライトと選ばれる行動が直す前と同じである。
- 実装記録に、Grid の差分更新の 2 つの条件（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）が分けたあとの Grid の関数に残っていることと、その行を書く（未決表の 11）。

### A8. キーの転送（e088eb）

- `git grep -n "dynamic_cast" -- core/anet-core/src/gui.cpp core/anet-core/include/anet/gui.hpp` が 0 件である。
- GUI のテスト `[prd105_ui_keys]`・`[eval_steps_menu]`・`[prd109_ui]` が通る。
- 人が Runner で確かめる（未決表の 14）: Live モードで、矢印キー・テンキー・Space・Ctrl を 1 回押すと、行動の割り当てと step の進みが 1 回分だけ起きる。Evaluation Trace の pane と QValuePanel にフォーカスを置いた状態でも同じである。QValuePanel のクリックで、行動が 1 回だけ選ばれる。

### A9. ビルドと既存のテスト

[AGENTS.md](../../AGENTS.md) の手順で Debug ビルドを通し、既定のテストスイートと `"[shipped_config]"` を流す。ステージ 2 では、A8 の GUI のテストも流す。既知の失敗（PRD093 の attention のテスト 1 件）のほかに、直す前に通っていたテストが落ちない。結果と、流さなかったものがあればその理由を実装記録に書く。

### A10. 課題の記録

そのステージの題材の課題の対応内容に、直した内容と場所が書かれ、AI の判定が `残件無し` である。状態を `解決` にするのは開発者である（§2.2）。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| `GetScalar` を key から値への 1 枚の表にする | 不採用（暫定。未決表の 6） | 名前から番号を読む key と、引くときに計算する key が表に載らない |
| `InitRL` が既定値を持つクラスの定数を引く | 不採用（暫定。未決表の 9） | `RunManager::Config` を trainer.cpp の外へ出す変更が要り、無いキーを飛ばせば今の結果のまま二重持ちが消える |
| キーの転送済みをイベントの印で表す | 不採用（暫定。未決表の 13） | `wxKeyEvent` に利用者の印を付ける口が無い |
| テストで見つかった今の振る舞いの誤りを直す | 範囲外（未決表の 12） | 振る舞いを変えると G2 に反する。見つかったら起票の候補にする |
| scaler の WARN と `post_process_type` の検証 | 範囲外（別の関心） | PRD117 の範囲（§3.2） |

## 8. 依存と順序

- 先に要る PRD は無い。
- [PRD117](117_warn_policy_and_unread_config_10prd.md) は scaler_impl.cpp の 420 行の WARN と 505 行付近の DEBUG の出力を変え、`post_process_type` の構築時の検証を足す。本 PRD の scaler のテストは有効な値だけを使い、ログを見ないので、どちらを先に入れてもよい。同じファイルの同じ関数（`NormalizeAndUpdateStats`）に触るので、並べて実装するときは片方を先に入れる。
- [PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md) は QValuePanel.cpp の `CreateData` に触る。本 PRD は `Update` だけを変えるので、どちらを先に入れてもよい。
- [PRD922](922_input_mapping_joystick_10prd.md) は `RunnerFrame::OnKey` の解釈を作り替える。本 PRD は `Panel::OnKey` の判定だけを変えるので、どちらを先に入れてもよい。

## 9. 参照

- 規約: [リファクタリング観点](../quality/refactor.md) の「ガードレール」、[実装 品質確認観点](../quality/impl.md) の「C++ コード」「GetScalar」と項番 2・42・58・60、[PRD079](079_cpp_internal_linkage_convention_10prd.md)。
- 地図: [110 RunnerApp](../wiki/apps/110_runner_app.md)、[312 ユーティリティ](../wiki/categories/312_util.md)、[314 テンソル処理](../wiki/categories/314_tensor.md)、[318 強化学習 共通基盤](../wiki/categories/318_rl_core.md)、[319 GUI 共通基盤](../wiki/categories/319_gui.md)、[321 Agent 共通基盤](../wiki/categories/321_agent_core.md)。
- 課題の記録: [anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)。
