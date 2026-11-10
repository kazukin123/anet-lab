# PRD122: 小さなリファクタリングとテストの空白をまとめて片づける

> 2026-11-10 起票（暫定。質疑を省き、既定を §4 の未決表に置いた）。行番号は起票時点の作業ツリーの位置で、実装時は近傍の記述で探し直す。
> 2026-11-10 改訂（グリル）: グリルを行い、暫定の印を外した。未決表の 14 行を §4 の確定事項 D1〜D15 へ移し、未決は無くなった（§8）。暫定の既定から変えた点は次のとおり。(1) e088eb は、生きている Panel の集まりで判定する形をやめ、転送した Panel が親への伝播を止める形にした（D13）。(2) 58bf49 は、ヒートマップの加工を画面の部品に依らない関数に切り出して単体テストを足す（D10・D11）。(3) 741251 の特性テストの範囲を、直接比べる 43 個から、認識する key の全部に広げ、`GetScalar` だけが呼ぶ補助 4 つも .cpp へ移す（D6・D7）。(4) ステージ 1 に、短い Run の前後比較を足した（D8、受入 A7）。(5) e088eb に特性テストを足した（D14）。(6) SymLog のテストで、閾値 0 の読み替えを固めない（D3）。(7) §2.1 の事実を 4 点直した（741251 の key の数、58bf49 の表の更新とモード、e088eb の既存の GUI テストが通る経路、d337bd の robust update の不等号）。§2.3（`wxEVT_CHAR_HOOK` の流れ）を足した。(8) G2 にデバッグログの例外を書き、非ゴールの「キーの転送の経路」の行を狭めた（§3）。(9) グリルのあと、別モデル（Opus）の検討を受けて直した。741251 の直接比べる 43 個の key を、群ごとの関数ではなく 1 つの表にまとめる形にした（D6）。robust update の受入で書き換える行、`GetScalar` の止まる経路と key ごとの戻り値、ヒートマップの加工の契約（平滑化の重み、契約の外の入力）とテストの場合分け、device の検査の前提、テストの実行体が既定のスイートに入らないことの扱い、退けた案の表を直した。(10) レビューの指摘を受けて、A6 の検索の文字列（キーと既定値の対だけに当たる形）と、D7・A5 の、止まる経路のうちテストで通さなくてよいものの条件を直した。あわせて、A5 の .cpp の件数の受入を 43 個の名前での検索に替え、A5 の宣言の検索を「直す前より 1 多い」に替え、A6 にキーの有無を問う検索の項を足し、D8 の理由に表への変更を、D14 に設計文書へ書く名前を、§5.3 に地図への書き戻しの段落を、§5.4 に `State` の既存の用例を足した。(11) 工程の終わりに、見つかった不具合の疑い 2 件を課題に記録し、GUI のテストの実行体が既定のテストに入らない件を既存の課題へ追記して、§7 にその ID を書いた。
> 起点: 2026-11-10 の課題の整理の枠。ユーザーが、type がリファクタリングで手間の小さい課題 8 件を 1 本の PRD に束ねるよう指示した。AI は、ビルドと単体テストで閉じる core の 6 件と、Runner を動かして確かめる GUI の 2 件で確かめ方が違うことを示し、人の手間（グリル・計画の承認・レビュー・コミットの回数）で比べて、1 本の PRD の中で実装を 2 つのステージに分ける形を出した。ユーザーは、その形で暫定の PRD を起こすよう指示した。グリルと実装は別の枠で行う。
> 題材: [issue-20261103-d4aa77](../management/issues/issue-20261103-d4aa77.md)、[issue-20261103-d337bd](../management/issues/issue-20261103-d337bd.md)、[issue-20261103-60f11c](../management/issues/issue-20261103-60f11c.md)、[issue-20261103-03a1b6](../management/issues/issue-20261103-03a1b6.md)、[issue-20261103-741251](../management/issues/issue-20261103-741251.md)、[issue-20261103-c1e48b](../management/issues/issue-20261103-c1e48b.md)、[issue-20261103-58bf49](../management/issues/issue-20261103-58bf49.md)、[issue-20261103-e088eb](../management/issues/issue-20261103-e088eb.md)。
> 関連: [PRD119](119_small_doc_comment_config_fixes_10prd.md)（小さな直しを 1 本に束ねた前例）、[PRD117](117_warn_policy_and_unread_config_10prd.md)（scaler の WARN と `post_process_type` の構築時の検証）、[PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md)（QValuePanel の Q と手動の行動の時点）、[PRD070](done/070_device_spec_auto_10prd.md)（device の一括検査）、[PRD035](done/035_approx_actor_priority_per_10prd.md)（初期優先度の completer）、[PRD105](done/105_eval_trace_arrow_key_forced_action_10prd.md)（矢印キーと強制の行動）、[PRD094](done/094_evalpanel_step_history_10prd.md)（読み口の整理の前後比較）、[PRD109](done/109_runner_startup_background_init_10prd.md)（短い Run の前後比較の前例）。

## 1. 問題

### 1.1 既定で動く経路の検査と計算に、テストが無い

次の 3 か所は、壊してもどのテストも落ちない。

- ReplayBuffer のコンストラクタは、PER では priority store と初期優先度の completer を対で持ち、uniform ではどちらも持たないことを検査している。片方だけを渡したときに止まることを確かめるテストが無い（d4aa77）。
- 報酬の scaler の `use_auto_post_scale`、観測の正規化の robust update と SymLog の後処理は、DefaultDQNAgent の既定のプロファイルで有効だが、テストはどれも通らない（d337bd）。
- MuZero の Replay の、エピソード終端での価値ターゲットの打ち切りと、unroll 窓の mask は、学習ターゲットそのものを決めるが、テストが無い（60f11c）。

式や不等号を取り違えても気づけず、既定の設定で動く Run の学習が黙って変わる。

### 1.2 dqn_based_agent.hpp の形が崩れている

- core のヘッダで、`dqn_based_agent.hpp` だけがインクルードガードを持たず、先頭のファイル名のコメントが旧名のままである（03a1b6）。
- `BatchUpdateResult::GetScalar` が、ヘッダのクラス定義の中に 241 行の if の連鎖として書かれている（741251）。診断値を 1 つ足すたびに、このヘッダを読み込む 8 本の .cpp が作り直しになり、長い連鎖で重複や届かない分岐を見落とす。

### 1.3 device の既定値を二重に持っている

`InitRL` の device の検査が、`agent.device`・`env.device`・`run.eval_device` の既定値を、それぞれを持つ設定のクラスとは別に書いている（c1e48b）。片方だけを変えると、検査が見る値と実際に使う値が食い違う。

### 1.4 GUI の長い関数と、規約に反する型の照会

- `QValuePanel::Update` が約 310 行あり、統計の計算、Grid の更新、HeatMap 用のデータの加工を 1 つの関数で続けて行う（58bf49）。加工は画面の部品を直接読むので、画面を作らないと動かせず、テストが無い。表示のモードを足すときに壊しても気づきにくい。
- `Panel::OnKey` が、キーの多重の転送を防ぐ判定に `dynamic_cast` を 3 回使う。規約は production のコードでの `dynamic_cast` を禁じていて、品質確認の機械検査が gui.cpp を変えるたびに 3 件を出す（e088eb）。

## 2. 現状（2026-11-10 に確認したこと）

### 2.1 題材ごとの事実

| 題材 | 事実 |
| --- | --- |
| d4aa77 | [replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の 1530 行のコンストラクタが、1555〜1557 行で `priority_store_` と `initial_priority_completer_` の片方だけがあると `ANET_SYSTEM_ERROR` にする。クラスは [replay_buffer_impl.hpp](../../core/anet-core/src/replay_buffer_impl.hpp) の 449 行の `DefaultReplayBuffer` で、生成は replay_buffer_impl.cpp の 2629 行のファクトリが行う。ファクトリは片方だけの組み合わせを作らない。[replay_buffer_test.cpp](../../core/anet-core/src/replay_buffer_test.cpp) の 1804 行のテストは `CreateInitialPriorityCompleter` が PER のときだけ completer を作ることまでを見て、コンストラクタに片方だけを渡す経路は通らない |
| d337bd | [scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp) の 892〜894 行は、`use_auto_post_scale` のとき post_scale を `reference_q_std * (1 - gamma)` にする。449〜551 行の robust update は、統計の件数が `robust_warmup_count` に届くまでは無条件に更新する（463 行）。届いたあとは、全部の要素の偏差が `robust_std_threshold * std` 以下ならそのまま更新する（469・476 行。全部が正常なときの近道）。そうでなければ、偏差が閾値を超える要素（495 行の `deviation > threshold`）を 1 つでも含む行（501 行の `.any(1)`）を外し、残りの行だけで更新する（502・546 行）。統計の件数は、与えた行の数だけ増える（187〜217 行）。外した行があると、WARN を出す（505〜541 行）。392〜405 行の SymLog は `scale * sign(x/scale) * log1p(|x/scale|)` で、`scale` は `post_process_threshold` が正ならその値、そうでなければ 1 である。共通の検証 `ValidateObservationNormalizerConfig`（667 行）は `post_process_threshold` が 0 以下の設定を不正とするが（678 行）、呼ぶのは RND の経路（[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp) の 267 行）だけで、DefaultDQNAgent の観測の正規化は通らない。[agent.txt](../../apps/runner/config/agent.txt) は 2 行で `DefaultDQNAgent.$ = @baseline` とし、`@baseline` の 265 行で `use_auto_post_scale ?= true`、276・278・279 行で robust update と SymLog（`post_process_type ?= 1`）を有効にしている。[scaler_test.cpp](../../core/anet-core/src/scaler_test.cpp) のテスト 3 本（6・24・49 行）は、どれも `use_auto_post_scale = false` で、robust update と SymLog を通らない |
| 60f11c | [muzero_rb.cpp](../../core/anet-core/src/muzero_rb.cpp) の 66 行の `MuZeroReplayBuilder::ProcessQueue` は、98〜121 行で、最新の step が done か truncated のとき、残りの各 step の価値ターゲットを終端までの割引報酬で作ってキューを空にする。truncated のときは最後の step の `root_value` で残りを補う（113〜115 行）。170 行の `MuZeroUniformSampler::Sample` は、`unroll_steps + 1` の窓に行動・報酬・価値・policy・mask を詰め、done の step の次からと、lane の記録の末尾の先を、mask 0・価値 0 で埋める（222〜250 行）。埋める行動は乱数で選ぶ（234 行）。287 行の `MuZeroReplayBuffer::Push` は、batch の経験を lane ごとのキューへ渡す。クラスは [muzero_rb.hpp](../../core/anet-core/src/muzero_rb.hpp) の 58〜116 行にある。これらを直接使うテストは [prd101_muzero_test.cpp](../../core/anet-core/src/prd101_muzero_test.cpp) の 4 行の 1 本だけで、終端の無い 4 step の n-step の割引を見るだけである。MuZeroAgent は試作の Agent である（[325](../wiki/categories/325_muzero.md)） |
| 03a1b6 | [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) の 1 行は `// rainbow_agent_impl.hpp` で、`#pragma once` も `#ifndef` のガードも無い。`core/anet-core/src` と `core/anet-core/include/anet` のヘッダは 56 本あり、55 本が `#pragma once` を持つ（例: [gui.hpp](../../core/anet-core/include/anet/gui.hpp) の 3 行）。`#ifndef` のガードを使うヘッダは無く、ガードの無いヘッダはこのファイルだけである。このヘッダを読み込むのは、`default_dqn_agent.cpp`・`dqn_based_agent.cpp`・`rainbow_agent.cpp` と、テスト 5 本（`dqn_based_agent_test.cpp`・`dqn_based_test.cpp`・`dqn_munchausen_test.cpp`・`intrinsic_test.cpp`・`prd101_ladder_test.cpp`）の 8 本である |
| 741251 | dqn_based_agent.hpp の 219 行の `class BatchUpdateResult` の中で、`GetScalar` が 285〜525 行の 241 行を占める。key は 2 通りの形で受けている。(a) 文字列を直接比べる key が 43 個ある（損失・Q・IQN の診断・PER など）。(b) 名前の解析か key の配列で受ける群が 8 つある: attention entropy（287 行）、skip（293 行）、勾配ノイズ（299 行）、Noisy（302 行）、replay fit（305 行）、Munchausen（308 行）、policy churn（318 行）、可塑性（329 行。537〜594 行の `GetPlasticityScalar` が、自分の群の key でなければ「当たらない」を返す）。値が成り立たない既知の key には NaN を、未知の key には `std::nullopt` を返す（[実装 品質確認観点](../quality/impl.md) の「GetScalar」、項番 42）。ただし、値が無いときに NaN でない値を返す key がある。`per_sample_initial_count` は 0 を返し（441〜445 行）、`per_actor_learner_` で始まる 5 個（466〜470 行）と `upper_tail_priority_spearman`（421〜423 行）は、持っている値をそのまま返す。attention entropy の key は `attention_entropy.[n]` の形で、n が 0 以上の整数なら受け、層の数を超える n には NaN を返す（288〜290 行、[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の 5295〜5309 行）。`ANET_SYSTEM_ERROR` で止まる経路は次のとおり。`attention_entropy.` で始まるが形の崩れた key（dqn_based_agent.cpp の 5307 行）、Munchausen の診断値の形（311 行）、policy churn の件数（322 行）、可塑性の重みノルムの件数（553 行）、スペクトル正規化の番人の 3 通り（`ValidateSpectralNormSentinel` の 603・612・616 行）、可塑性の指標の控えに値が無いとき（591 行）。`GetScalar` だけが呼ぶ補助が 4 つ、同じクラスの中にある（`GetPlasticityScalar`、596 行の `ValidateSpectralNormSentinel`、620 行の `TransQToCpu`、626 行の `TransRealQToCpu`）。ヘッダの `key == "` は 50 件で、内訳は `GetScalar` の 43、`GetPlasticityScalar` の 6、`GetTensor` の 1 である。`GetScalar` は基底 `anet::Module` の純粋仮想関数の上書きで（[common.hpp](../../core/anet-core/include/anet/common.hpp) の 65 行、[rl.hpp](../../core/anet-core/include/anet/rl.hpp) の 608 行）、呼ぶ側は基底の型越しに呼ぶ。テストから `GetScalar("<key>")` を呼ぶ行は 8 ファイルに 466 行あるが、Env の結果や ImageCls など別の型の呼び出しも含み、認識する key のすべてを通るかは確かめていない |
| c1e48b | [init.cpp](../../core/anet-core/src/init.cpp) の 95〜98 行が、`{"agent.device", "auto"}`・`{"env.device", "cpu"}`・`{"run.eval_device", "auto"}` の表を持ち、キーが無ければ表の既定で `ParseDevice` にかけ、`auto` でない cuda を明示した cuda として集める。既定値を持つ設定のクラスは、[agent.hpp](../../core/anet-core/include/anet/agent.hpp) の 394 行の `DefaultAgentFactoryConfig`（prefix `agent`、`auto`）、[env.hpp](../../core/anet-core/include/anet/env.hpp) の 348 行の `BatchEnvBuilderConfig`（prefix `env`、`cpu`）、[trainer.cpp](../../core/anet-core/src/trainer.cpp) の 808 行の `RunManager::Config`（prefix `run`、`auto`。trainer.cpp の中で定義した型）である。表の既定 3 つは、どれも `ParseDevice` で失敗せず、明示した cuda にも数えられない。PRD070 の D5 は「cuda を指定している全キー」を一括で報告する決定である（[PRD070](done/070_device_spec_auto_10prd.md) の 68 行）。検査のテストは [init_test.cpp](../../core/anet-core/src/init_test.cpp) の 128・143 行（`[prd070]`）と、空の設定で呼ぶ 192 行にある。143 行のテストは CUDA が使える機械では skip になる（145 行）。環境変数 `CUDA_VISIBLE_DEVICES` を `-1` にして `[prd070]` を流すと、3 件とも skip されずに通る（2026-11-10 に Debug の実行体で確かめた）。`ConfigData` はキーの有無を問う `Has` を持つ（[config.hpp](../../core/anet-core/include/anet/config.hpp) の 70 行） |
| 58bf49 | [QValuePanel.cpp](../../apps/runner/src/QValuePanel.cpp) の `QValuePanel::Update` は 420〜727 行の約 310 行で、1 フレームに 1 回呼ばれる（411 行の予約）。438 行から全体の平均と標準偏差の計算、449 行から Grid の更新、522 行から HeatMap のデータの加工（Advantage の引き算、Histogram 化、色付けのための正規化）、717 行から描画が続く。Grid の更新は、選択行のハイライトを選択が変わったときだけ塗り直し（496 行）、列の幅は値が広がったときだけ当て直す（519 行。毎 step 全行を塗り直さないための差分更新）。Grid の Advantage の引き算は無効化されていて（474 行）、Grid は表示のモードを使わない。全体の平均と標準偏差を使うのは加工だけである。455〜466 行にコメントアウトされた古い処理が残る。加工は、チェックボックス（Histogram・Advantage・Log・Auto Range）、設定（`hist_bins`・`hist_range_k`・`use_hist_smooth`・`smooth_radius`。28〜32 行）、Histogram の累積範囲（`accumulated_min_`・`accumulated_max_`）をメンバから直接読み、累積範囲を書き換える。Log と Auto Range が効くのは、Histogram が on のときだけである（687・555 行）。平滑化の重みは、コンストラクタが半径から 1 回だけ作ってメンバに持ち（137〜146 行）、加工が読む（638 行）。平滑化が有効で `smooth_radius` が 0 以下のときは、重みを作らないまま加工が重みを読む（138・627〜638 行。空の重みの範囲外を読む不具合で、振る舞いが決まっていない）。加工の前に、共有の値を 1 回写している（429 行）。QValuePanel を直接見るテストは無く、`EvalTracePanel_test.cpp` と `RunnerFrame_test.cpp` が Runner の中の部品として触るだけである。Runner の GUI のテストは、実行体 `AnetRLRunner-gui-test` にまとまっている（[CMakeLists.txt](../../apps/runner/CMakeLists.txt) の 148 行）。この実行体は `EXCLUDE_FROM_ALL` で、既定のビルドにも既定のテストスイートにも入らない。wx を立ち上げるのはこの実行体だけで（[EvalTracePanel_test.cpp](../../apps/runner/src/EvalTracePanel_test.cpp) の 1357 行）、core のテストの実行体 `anet-core-test` は wx を立ち上げず、window を作るテストを持たない |
| e088eb | [gui.cpp](../../core/anet-core/src/gui.cpp) の 75 行で `Panel` が `wxEVT_CHAR_HOOK` を `Panel::OnKey`（89 行）に結び、95・100・108 行の `dynamic_cast` で、発生元が別の `Panel` か、発生元から自分までの親に `Panel` があるかを調べる。あれば `Skip` だけをして、無ければ（自分が最も下の `Panel` なら）`ForwardedKeyEvent`（種類は `EVT_FORWARDED_KEY`。以下「転送イベント」）に包んで転送し（123〜127 行）、`Skip` を呼ぶ。92 行で、受けるたびにデバッグログ（`KeyCode=`）を出す。転送したキーは `RunnerFrame::OnKey` が解釈する（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) の 887・1416 行）。`Panel` は入れ子になる（Q-Values のヒートマップの `QValueHeatMapPanel` は `QValuePanel` の中にある。[QValuePanel.hpp](../../apps/runner/src/QValuePanel.hpp) の 18・60 行）。`OnKey` を上書きする派生クラスは無い。GUI のテスト `[prd105_ui_keys]` は、転送後の `ForwardedKeyEvent` を作って frame へ直接送るので（[EvalTracePanel_test.cpp](../../apps/runner/src/EvalTracePanel_test.cpp) の 1181〜1188 行）、`Panel::OnKey` の判定を通らない。判定を通るテストは無い。規約は [実装 品質確認観点](../quality/impl.md) の 230 行（production のコードで `dynamic_cast` を使わない）で、機械検査は同じ文書の項番 2（45 行）である |

### 2.2 規約と過去の判断

- リファクタリングのガードレール: テストの無い箇所は、特性テスト（今の振る舞いをそのまま固定するテスト）を先に足す。hot path（Sample・Push・Learn、step ごとの経路）に触るときは、性能が変わらないことをブレ幅の基準で確かめる（[リファクタリング観点](../quality/refactor.md) の 53〜57 行）。
- `GetScalar` の戻り値: 未知の key だけを `std::nullopt`、既知の key で値が成り立たないときは NaN にする（[実装 品質確認観点](../quality/impl.md) の「GetScalar」）。
- 読み口の整理の確かめ方: 読み口と配線を整理するときは、metrics の出力・定義・WARN・fail-fast の文言と時機を変えず、同じ seed・決定的モードの短い Run で前後を比べる（[314 テンソル処理の地図](../wiki/categories/314_tensor.md)・[318 強化学習 共通基盤の地図](../wiki/categories/318_rl_core.md) の「変えるときの条件」、出典は [PRD094](done/094_evalpanel_step_history_10prd.md) の D3）。
- 短い Run の前後比較の手順: [PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md) の「段0・smoke」が、変更前の実行体を保存して、4 環境の短い Run の全 scalar の点を比べる手順を持つ。PRD105 と PRD109 が同じ手順で受入を取った（[PRD109](done/109_runner_startup_background_init_10prd.md) の受入 7）。
- 設定値の扱い: 特定の値に別の意味を持たせる読み替えを足さない（[AGENTS.md](../../AGENTS.md) の「設定値の扱い」）。
- 既存の死にコード: 自分の変更で使われなくなったものだけを片づけ、もとからある死にコードは消さずに報告する（AGENTS.md の「LLM コーディング規律」）。
- 暗黙のグローバル状態を増やさない（AGENTS.md の「基本方針」、実装 品質確認観点の項番 35）。
- .cpp の補助: 外で使わない補助のクラスは、持ち主のクラスの private な入れ子か .cpp の無名名前空間に置く（実装 品質確認観点の 233 行、項番 60）。.cpp の名前空間とリンケージは [PRD079](079_cpp_internal_linkage_convention_10prd.md) の D1〜D4 による（同 221 行）。
- Runner のキーとクリック: Trace モードでないときの矢印キー、テンキー、QValuePanel のクリック、Ctrl、Space の、行動の割り当てと step の進み方を変えない（[110 RunnerApp の地図](../wiki/apps/110_runner_app.md) の「変えるときの条件」、出典は PRD105 §2 の NG5）。Train View が表示中の振る舞いを変えるときは、GUI のテスト `[eval_steps_menu]`・`[prd105_ui_keys]`・`[prd109_ui]` が通ることを確かめる（同、出典は PRD110 §7 の受入 1）。
- 起動の順: InitRL を含む、表示より前の起動の段の順序を変えない（同、出典は PRD109 §2 の NG4・D4）。
- 課題の状態: `解決` にするのは開発者で、AI は対応内容と AI の判定（`ai_judge`）を書く（[anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)）。

### 2.3 `wxEVT_CHAR_HOOK` の流れ（wxWidgets 3.3.3 のソースで確かめたこと）

`wxEVT_CHAR_HOOK` は、キーが押されたときに、通常のキーイベントより先に届くイベントである。行番号は wxWidgets 3.3.3 のソースの位置で、リポジトリの外にあるのでリンクは付けない。

1. wx はキーが押されると、フォーカスのある window を発生元にして `wxEVT_CHAR_HOOK` を作り、その window へ送る（src/msw/window.cpp の 7458〜7477 行。発生元の設定は 6500 行）。
2. `wxEVT_CHAR_HOOK` は、受けた window が処理しなければ親の window へ順に渡る（include/wx/event.h の 2388〜2392 行が伝播を有効にし、src/common/wincmn.cpp の 3510〜3532 行が親へ渡す）。
3. 親へ渡すのは、イベントの伝播が止められていないときだけである（wincmn.cpp の 3514 行）。`StopPropagation` は伝播を止める（event.h の 1033〜1038 行）。伝播を止めても、アプリケーションの受け口へは渡る（wincmn.cpp の 3532 行）。
4. どの受け手も処理しなければ（`Skip` を呼べば）、wx は未処理として、標準のキーの処理（通常のキーイベントの生成、Tab での移動など）へ進む（window.cpp の 7477〜7496 行）。
5. wx の中で `wxEVT_CHAR_HOOK` を受けるのは、ダイアログ（src/common/dlgcmn.cpp の 109 行）、印刷プレビューの frame（src/common/prntbase.cpp の 1709 行）と、自分に来たキーを処理するコントロール（ボタン、リスト、ツリー、文字入力の補完。src/msw/button.cpp の 62 行、src/msw/listctrl.cpp の 221 行、src/common/treebase.cpp の 171 行、src/msw/textentry.cpp の 443 行）である。`Panel` の親になる frame、AUI の枠、wxPanel は受けない。
6. GTK でも、`wxEVT_CHAR_HOOK` は window のイベントの処理へ送られ、2〜4 と同じ伝播の経路を通る（src/gtk/window.cpp の 1269〜1283 行）。Runner の主な環境は Windows である。

リポジトリの中で `wxEVT_CHAR_HOOK` を受けるのは 3 か所である（`git grep -n "wxEVT_CHAR_HOOK" -- core apps` のうちテストを除く当たり）。

| 位置 | 受け手 | Panel の親か |
| --- | --- | --- |
| [gui.cpp](../../core/anet-core/src/gui.cpp) の 75 行 | `Panel` 自身 | — |
| [EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp) の 228 行 | Evaluation Trace の表 | 親ではない（表は発生元の側にある） |
| [LogPanel.cpp](../../apps/runner/src/LogPanel.cpp) の 298 行 | Logs の pane | 親ではない（中に `Panel` を持たない） |

`Panel` より上の window で `wxEVT_CHAR_HOOK` を受ける部品は無い。`Panel` を中に置くダイアログも無い。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: 題材の 8 件の課題をすべて片づける。
- G2: 外から見える振る舞いを変えない。設定のキーと既定値、メトリクスの key と値、画面の表示と操作、ログの文言は、直す前と同じにする。変わるのは、テストが増えることと、コードの置き場と形である。例外は 1 つで、`Panel::OnKey` のデバッグログ（`KeyCode=` の行。デバッグログを有効にしたビルドを、デバッガを付けて動かしているときだけ出る。[log.hpp](../../core/anet-core/include/anet/log.hpp) の 320〜331 行）が、入れ子の上位の `Panel` の分だけ出なくなる（D13）。
- G3: 足したテストが、課題の「影響」に書いた壊し方で落ちる。

### 3.2 非ゴール

| 退ける案 | 理由 | 本当の制約か |
| --- | --- | --- |
| scaler の `Unknown squash_type` の WARN と、DefaultDQNAgent の経路での `post_process_type` の値域の構築時の検証を一緒に直す | [PRD117](117_warn_policy_and_unread_config_10prd.md) の D7 と §5.5 の範囲である。本 PRD は今の振る舞いをテストで固めるだけにする | 制約ではない（別の関心。PRD117 の WARN の線引きと一緒に決める） |
| テストで見つかった食い違い（MuZero の打ち切りの式など）を本 PRD で直す | 振る舞いを変えると G2 に反し、直し方の判断が要る | 制約ではない。見つかったら起票の候補にする（D15） |
| `GetScalar` の key の名前を変える・整理する | 設定の `metrics.scalar` が key の名前で参照している。名前を変えると設定とテストの移行が要る | 制約ではない（別の関心。メトリクスの名前の整理は [PRD108](108_metrics_tag_numbering_10prd.md) の範囲） |
| QValuePanel の Q と手動の行動の時点のずれを一緒に直す | [PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md) の範囲で、表示の振る舞いを変える | 制約ではない（別の関心） |
| 転送イベントの中身（種類・ID・発生元の window・キーの情報）と、`RunnerFrame::OnKey` の解釈を作り替える | 行動の割り当てと step の進み方が変わり、G2 に反する。キーの割り当ての設定化は [PRD922](922_input_mapping_joystick_10prd.md) の範囲である | 制約ではない（別の関心）。`wxEVT_CHAR_HOOK` の親への伝播は、外から見える振る舞いを変えないので変える（D13） |
| `InitRL` の同じ塊に触る課題（[issue-20261103-78df9c](../management/issues/issue-20261103-78df9c.md): 同じ fatal が 2 回ログに出る、[issue-20261103-9b314a](../management/issues/issue-20261103-9b314a.md): エラーが指定値を書かない）を一緒に直す | どちらもログとエラーの文言を変えるので、G2 に反する | 制約ではない（別の関心。振る舞いを変える直しとして別に扱う） |
| ほかのヘッダの先頭のコメントを一括で点検する | ガードの無いヘッダは dqn_based_agent.hpp だけだった（§2.1）。先頭のコメントの食い違いは、見つかった 1 件だけを直す | 制約ではない。見つかれば別の課題にできる |

### 3.3 成功の測り方

- 題材の 8 件を、実装のあとに解決にできる（§6 の受入がすべて通る）。
- 足したテストが、課題の「影響」に書いた壊し方（検査を消す、式を変える、不等号を逆にする、mask を 1 に戻すなど）で落ちる。実装の工程で一時的にコードを変えて確かめ、結果を実装記録に残す（§6 の A1〜A3・A5・A8・A9）。
- ステージ 1 の前後で、短い Run の全 scalar の点が一致する（A7）。
- 品質確認の機械検査（[実装 品質確認観点](../quality/impl.md) の項番 2）が、gui.cpp で 0 件になる。

### 3.4 足すものと、ゴールへの寄与

| 足すもの | 寄与 |
| --- | --- |
| 題材 3 件のテスト（D2〜D4） | G1（d4aa77・d337bd・60f11c）、G3 |
| `GetScalar` と `Panel::OnKey` の特性テスト（D7・D14） | G2 の確かめ。直す前と後で同じテストが通る |
| 短い Run の前後比較（D8） | G2 の確かめ。メトリクスの値が変わらないことを直接見る |
| ヒートマップの加工の関数と、その入出力の型、単体テスト（D10・D11） | G1（58bf49）。課題の「影響」（表示のモードを足すときに壊しても気づきにくい）を、テストで気づける形にする |
| 転送した `Panel` が伝播を止める形（D13） | G1（e088eb） |

先送りするものは無い。

## 4. 確定事項

2026-11-10 のグリルで決めた。D1・D6・D10・D11・D13 はユーザーが案を比べて選び、ほかは AI が示した推奨の既定を、ユーザーがまとめて採った。グリルのあとに別モデル（Opus）の検討を受け、D6 を選び直し、D3・D4・D7・D9・D11・D14 の範囲と前提を補った。

| # | 決定 | 理由と前提 |
| --- | --- | --- |
| D1 | **束ね方と実装の順。** 1 本の PRD にし、実装を 2 つのステージに分ける。ステージ 1 は core の 6 件（d4aa77・d337bd・60f11c・03a1b6・741251・c1e48b）、ステージ 2 は GUI の 2 件（58bf49・e088eb）。ステージ 1 だけでコミットでき、その 6 件の課題はステージ 2 を待たずに解決にできる。e088eb は core のファイル（gui.cpp）を直すが、Runner で人が確かめる項を持つので、ステージ 2 に置く | ステージ 1 はビルド・単体テスト・短い Run で閉じ、ステージ 2 は人が Runner で見る項を持つ（§2.1）。確かめ方の違うものを 1 つのコミットに混ぜないほうが、手戻りの範囲が小さい。PRD を 2 本に分ける形より、グリルと計画の承認が 1 回ずつ少ない。ステージに分けない形は、GUI の確かめで止まると core の 6 件も止まる。依存: なし |
| D2 | **d4aa77 のテスト。** `DefaultReplayBuffer` のコンストラクタへ priority store と completer の片方だけを渡す 2 通りが `ANET_SYSTEM_ERROR` で止まり、文言に `must be configured together` が入る。両方ありと両方なしは構築できる（この 2 通りはファクトリ経由でもよい） | 不変条件は「対で持つ」なので、4 通りで尽きる。ファクトリは片方だけの組み合わせを作らないので、片方だけの 2 通りはコンストラクタへ直接渡す（§2.1）。依存: なし |
| D3 | **d337bd のテスト。** 3 つを足す。(a) `use_auto_post_scale = true` で、gamma（0.5 以外）と `reference_q_std` から手で計算した post_scale が、報酬の変換に効く。(b) robust update で、統計の件数が warmup の件数に届く前は外れ値も統計に入り、届いたあとは閾値を超える要素を含む行だけが統計から外れる。warmup の境目は、前後の 1 件ずつで見る。(c) SymLog で、`post_process_threshold` に 1 以外の正の値を使い、出力を手で計算した値と比べる。`post_process_threshold` が 0 以下のときに 1 と読み替える動きは断定しない | 既定のプロファイルで有効な 3 つの経路を、課題の対応方針どおりに通す（§2.1）。gamma を 0.5 以外にするのは、`(1 - gamma)` を `gamma` に取り違えると値が変わる入力にするため。境目の前後を見るのは、不等号の取り違えが件数の端に出るため。閾値 0 の読み替えを固めないのは、特定の値に別の意味を持たせる動きで（§2.2）、共通の検証は 0 以下を不正としているため（§2.1）。固めると、PRD117 が DQN の経路に検証を足すときに直すテストが増える。依存: PRD117 とは先後を問わない（§9） |
| D4 | **60f11c のテスト。** 短いエピソードを手で作り、5 場面を見る。(a) done で終わるキューの打ち切り（残りの各 step の価値ターゲットが、終端までの割引報酬になる）。(b) truncated で終わるキューの、`root_value` による補い。補う値は、打ち切った次の状態の価値ではなく、最後の step 自身の `root_value` である（muzero_rb.cpp の 113〜115 行。n-step の経路は n 歩先の step の値を使う。83 行）。誤りの疑いとして、D15 のとおり今の値を断定する。疑いは課題に記録した（§7）。(c) 窓が終端をまたぐときの mask と価値（終端の step までは mask 1、その次から mask 0・価値 0）。(d) 窓が lane の記録の末尾を越えるとき。(e) `Push` が lane ごとのキューへ分けること。乱数で埋める行動は値を見ず、mask だけを見る | 学習ターゲットを決める分岐を、1 回ずつ通す（§2.1）。埋める行動は乱数で選ぶので、値は期待値にならない。依存: なし |
| D5 | **03a1b6 の直し方。** 先頭に `#pragma once` を足し、先頭のコメントを `// dqn_based_agent.hpp` にする | core のヘッダ 56 本のうち 55 本が `#pragma once` で、`#ifndef` のガードを使うヘッダは無い（§2.1）。依存: なし |
| D6 | **741251 の分け方。** `GetScalar` の本体と、`GetScalar` だけが呼ぶ補助 4 つ（`GetPlasticityScalar`・`ValidateSpectralNormSentinel`・`TransQToCpu`・`TransRealQToCpu`）の本体を、dqn_based_agent.cpp へ移す。名前の解析か key の配列で受ける 8 つの群は、群ごとの関数にして、今の並びの順で先に引く。各関数は、自分の群の key なら今と同じ値を、自分の群の key でなければ「当たらない」を返す。どの群にも当たらなければ、直接比べていた 43 個の key を、key の名前と値の作り方を 1 行ずつ並べた 1 つの表から引く。表にも無ければ、未知の key として `std::nullopt` を返す。表に同じ key を 2 行書いたら、ビルドかテストで止まるようにする（手段は実装計画で決める）。key の名前、返す値、止まる条件と文言は変えない。ヘッダのクラスの定義には、`GetScalar` の宣言だけを残す | ユーザーが、43 個も群ごとの関数に分けて中を if の連鎖のままにする形と比べて選んだ（別モデルの検討で出た案）。課題の「影響」のうち、key の綴りの重複や届かない分岐の見落としは、if の連鎖を群に分けても機械では見つからない。表にすれば重複を止められ、key が 1 か所に並ぶ（`GetScalar` の鍵を名前つきの定数にする規約に合う。[実装 品質確認観点](../quality/impl.md) の 237 行）。値の作り方を行に持たせるので、引くときに計算する key（PER の一部など）も表に載る。名前を解析して受ける群は、名前から番号を読むので表に載らず、今と同じ群ごとの形のままにする（可塑性の `GetPlasticityScalar` が既にこの形である。§2.1）。43 個は今も 8 つの群のあとに比べているので、群を先に引けば順は変わらない。補助も移すのは、可塑性の key を足すときもヘッダを触らずに済ませるためで、課題の「影響」（診断値を足すたびに 8 本が作り直しになる）と同じ関心である。群の順を保つのは、先に当たった群が返す作りなので、順を変えると結果が変わりうるため。依存: D7（特性テストを先に入れる） |
| D7 | **741251 の特性テスト。** 本体を移す前に、次を固定するテストを足す。`GetScalar` が認識する key の全部（直接比べる 43 個と、8 つの群の key）について、値がある状態と無い状態の、今の戻り値。期待値は key ごとに書く（値が無いときに NaN でない値を返す key がある。§2.1）。attention entropy は番号を無限に受けるので、代表（0 番、層の数を超える番号）で引く。未知の key 1 つの戻り値。`ANET_SYSTEM_ERROR` で止まる経路（§2.1 の一覧）の全部。通さなくてよいのは、本体を変えない限り、テストの手段（公開のメンバへの値の代入、fixture、テスト用の派生クラス）のどれでも到達できない経路だけで、その経路と到達できない理由を実装記録に書く。準備が大きいこと（Network の構築など）と、通常の学習の経路では起きないことは、通さない理由にしない。形の崩れた attention の key、Munchausen の診断値の形、policy churn の件数、可塑性の重みノルムの件数、スペクトル正規化の番人の「Network が無い」は、公開のメンバへの代入と key の文字列だけで作れるので、必ず通す。本体を移したあとも、同じテストがそのまま通る | テストの無い箇所は特性テストを先に足す（§2.2）。既存のテストは key ごとに散らばっていて、全部を通るか分からない（§2.1）。43 個だけでなく全部を引くのは、群の順や境目を誤ると、どの key も影響を受けうるため。止まる経路を入れるのは、移す本体（補助を含む）の中にあるため。依存: なし |
| D8 | **ステージ 1 で振る舞いが変わらないことの確かめ方。** 性能の受入は置かない。ステージ 1 の前後で、PRD098 の検証資材の短い Run（4 環境、seed 1、決定的モード）を流し、時間と資源の tag を除く全 scalar の点が一致することを受入にする（A7） | `GetScalar` は仮想関数の上書きで、呼ぶ側は基底の型越しに呼ぶ（§2.1）。本体がヘッダにあっても呼び出しは展開されていないので、.cpp へ移しても呼び出しの形は変わらない。直接比べる 43 個の引き方は if の連鎖から表（D6）に変わるが、key の文字列を比べる回数は今と同じ程度である。短い Run の比較は、読み口を整理するときの条件である（§2.2）。`InitRL` の直し（D9）とヘッダの直し（D5）も、同じ Run が通るので一緒に確かめられる。依存: D5・D6・D9 |
| D9 | **c1e48b の直し方。** `InitRL` は、設定に書かれているキーだけを検査し、書かれていないキーは検査しない。init.cpp の既定値の表を消す。検査する 3 つのキーの名前は `InitRL` に残す | 表の既定 3 つはどれも検査で失敗せず、明示した cuda にも数えられないので、書かれていないキーを飛ばしても結果は変わらない（§2.1）。PRD070 の D5 が報告するのは「指定している」キーで、書かれていないキーは対象ではない。検査が書かれた値だけを見れば、持ち主のクラスが既定値を変えても、検査が見る値と使う値は食い違わない。既定値を持つクラスの定数を引く形は、`RunManager::Config` が trainer.cpp の中の型なので、型を外へ出す変更が要る。前提: 持ち主のクラスの既定値に、明示の cuda を置かない（今は `auto`・`cpu`・`auto`）。置くと、書かれていないキーの cuda は一括の報告に載らず、止まるのが NN を作る時点に遅れるので、そのときはこの検査を見直す。依存: なし |
| D10 | **58bf49 の分け方。** `QValuePanel::Update` を、Grid の更新と、ヒートマップの加工と、描画に分ける。ヒートマップの加工（Advantage の引き算、Histogram 化、色付けのための正規化）は、画面の部品に依らない関数 `ComputeQValueHeatMap` に切り出す（契約は §5.2）。全体の平均と標準偏差の計算は、加工の中に置く。Grid の更新は、中身を変えずに `Update` とは別の関数へ移す。差分更新の 2 つの条件（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）、無効化された Advantage の引き算、コメントアウトは、そのまま残す。表示のモードとチェックボックスは `Update` で 1 回読み、加工の入力として渡す | ユーザーが 3 案（関数に分けるだけ、加工を切り出して単体テストを足す、表の更新も GUI のテストで固める）を比べて選んだ。関数を分けるだけでは、加工を壊したときに気づく手段が人の目のままである。加工は配列の計算なので、入力と出力の口を作ればテストできる（§2.1）。統計を加工の側に置くのは、使うのが加工だけだから（§2.1）。Grid の死にコードを残すのは、もとからある死にコードは消さずに報告する決まりによる（§2.2）。消すと、移しただけであることが差分から読めなくなる。依存: D11 |
| D11 | **58bf49 の加工のテスト。** `ComputeQValueHeatMap` の単体テストを足し、Runner の GUI のテストの実行体 `AnetRLRunner-gui-test` に置く。通す場合は次のとおり。Histogram が off のときの、Advantage の on と off。Histogram が on のときの、Advantage・Log・Auto Range の組み合わせの全部。平滑化の on と off。累積範囲が広がる場合と縮む場合。値が全部同じとき。期待値は、手で計算した値にする。あわせて、切り出す前のコードへ一時的に書き出しを入れて、テストと同じ入力に対する加工後の値を控え、単体テストの期待値と一致することを実装記録に残す | 加工の分岐を 1 回ずつ通す。Log と Auto Range は、Histogram が on のときだけ効く（§2.1）。切り出す前のコードには関数の口が無いので、テストを先に書けない（§2.1）。切り出す前の値を控えるのは、特性テストを先に足す決まり（§2.2）に代えて、切り出しで振る舞いが変わっていないことを示すため。この実行体は既定のテストスイートに入らないので（§2.1）、QValuePanel の加工を変えたときに流すことを、地図 110 の「変えるときの条件」に書く。加工を core へ移して既定のスイートで流す形と、Runner を起動して表の更新を読む GUI のテストは採らない（§7）。依存: D10 |
| D12 | **58bf49 の性能。** 実時間の受入は置かない。加工の前に値を写す回数を、今の 1 回から増やさない | `Update` は 1 フレームに 1 回の描画の経路で、学習の経路ではない（§2.1）。差分更新の効き目は Grid の関数に残る（D10）。依存: D10 |
| D13 | **e088eb の判定の置き換え。** `Panel::OnKey` は、転送イベントを出したあと、`Skip` を呼んで標準のキーの処理を続けさせ、`StopPropagation` で親の window への伝播を止める。発生元と親の `Panel` を調べる判定と、3 つの `dynamic_cast` を消す。転送イベントの中身（種類・ID・発生元の window・キーの情報）と回数、`RunnerFrame::OnKey` の解釈は変えない。変わることは 2 つである。入れ子の上位の `Panel` に `wxEVT_CHAR_HOOK` が届かなくなり、その分のデバッグログが出なくなる。`Panel` より上の window は `wxEVT_CHAR_HOOK` を受けられず、キーを受けたい部品は転送イベントを使う | ユーザーが、生きている `Panel` の集まりで判定する形と比べて選んだ。多重の転送を防ぐ仕組みが「転送した者が止める」の 1 つになり、判定・状態・型の扱いが要らなくなる。集まりの形は、ファイルの中に可変の状態が 1 つ増え（§2.2 のグローバル状態の決まりに触れる）、発生元を window として扱う変換が残る。前提は、`Panel` より上で `wxEVT_CHAR_HOOK` を受ける部品が無いことと、伝播を止めても標準のキーの処理が続くことで、どちらもソースで確かめた（§2.3）。依存: D14（特性テストを先に入れる） |
| D14 | **e088eb の確かめ方と文書。** 直す前に、特性テストを足す。window を作るテストなので、`AnetRLRunner-gui-test` に置く（core のテストの実行体は wx を立ち上げない。§2.1）。入れ子の `Panel` の中（発生元が中の window のときと、内側の `Panel` 自身のとき）、`Panel` の直下、`Panel` の外の 3 場面で、発生元の window へ `wxEVT_CHAR_HOOK` を送り、転送イベントの回数、転送元の `Panel`、未処理のまま返ることを確かめる。直したあとも同じテストがそのまま通る。あわせて、機械検査（gui.cpp と gui.hpp の `dynamic_cast` が 0 件）、既存の GUI のテスト 3 本、人が Runner で確かめる項を受入にする（A9）。設計文書 160 の `anet::rl::gui::Panel` の行に、キーの転送と伝播の停止、`Panel` より上では転送イベントを使うことを足す。受けるイベントは `wxEVT_CHAR_HOOK`、転送イベントは `EVT_FORWARDED_KEY` と、名前で書く | 判定を通るテストが今は無い（§2.1）。特性テストは、利用者の入力が通る経路（`wxEVT_CHAR_HOOK`）で送る（[PRD 品質確認観点](../quality/prd.md) の「受入」の規約）。フォーカスと実際のキーの入力はテストから作れないので、人が確かめる項を残す。文書に足すのは、D13 の新しい前提を、あとから部品を足す人が読める場所に置くため。特性テストの実行体は既定のテストスイートに入らないので（§2.1）、`Panel` のキーの転送を変えたときに流すことを、地図 319 の「変えるときの条件」に書く。依存: D13 |
| D15 | **テストで今の振る舞いの誤りが見つかったとき。** 本 PRD では直さない。テストは今の値のまま断定し、誤りの疑いをテストのコメントに書き、工程の終わりに起票の候補として示す | 直すと G2 に反し、直し方の判断が要る（§3.2）。断定しないと、その分岐が守られないまま残る。D3 の閾値 0 は、誤りの疑いではなく、固めない入力として扱う。依存: D3・D4・D7・D11 |

## 5. 実装契約

### 5.1 ステージと直す箇所

| ステージ | 題材 | ファイルと位置 | 直し方 | 確定事項 |
| --- | --- | --- | --- | --- |
| 1 | d4aa77 | core/anet-core/src/replay_buffer_test.cpp | コンストラクタの組み合わせのテストを足す | D2 |
| 1 | d337bd | core/anet-core/src/scaler_test.cpp | 3 つのテストを足す | D3 |
| 1 | 60f11c | core/anet-core/src の MuZero の Replay のテスト（置き場は実装計画で決める） | 5 場面のテストを足す | D4 |
| 1 | 03a1b6 | core/anet-core/src/dqn_based_agent.hpp の 1 行 | `#pragma once` を足し、先頭のコメントを直す | D5 |
| 1 | 741251 | dqn_based_agent.hpp の 285〜630 行、dqn_based_agent.cpp、特性テスト（置き場は実装計画で決める） | 特性テストを先に足し、本体と補助を .cpp へ移して、8 つの群の関数と、43 個の key の表に分ける | D6・D7・D8 |
| 1 | c1e48b | core/anet-core/src/init.cpp の 95〜111 行、init_test.cpp | 既定値の表を消し、設定にあるキーだけを検査する。書かれたキーだけが検査されることのテストを足す | D9 |
| 2 | 58bf49 | apps/runner/src/QValuePanel.cpp の 420〜727 行、QValuePanel.hpp、加工の単体テスト（Runner の GUI のテストの実行体に足す。ファイルは実装計画で決める） | `Update` を分け、加工を `ComputeQValueHeatMap` に切り出して単体テストを足す | D10・D11・D12 |
| 2 | e088eb | core/anet-core/src/gui.cpp の 89〜129 行、特性テスト（`AnetRLRunner-gui-test` に足す。ファイルは実装計画で決める）、docs/design/160_applications_and_tools.jp.md と .en.md | 特性テストを先に足し、判定を消して伝播を止める。設計文書に 1 行足す | D13・D14 |

ステージ 1 の 6 件は互いに独立で、どの順に入れてもよい。741251 は特性テスト（D7）を、e088eb は特性テスト（D14）を、本体を直すより先に入れる。ステージ 1 に入る前に、短い Run の比較（A7）に使う変更前の実行体を保存する。

### 5.2 `ComputeQValueHeatMap` の契約

入力と出力だけで結果が決まり、画面の部品とメンバを読まない。

| 向き | 中身 | 型（§5.4） |
| --- | --- | --- |
| 入力 | Q の格子値と、その幅・高さ | `QValueData`（既存） |
| 入力 | 表示のモード（Histogram・Advantage・Log）、Auto Range の on と off、Histogram の設定（bin の数、範囲の係数、平滑化の有無と半径） | `QValueHeatMapRequest` |
| 入力と出力 | Histogram の累積範囲（最小と最大）。呼ぶたびに更新して返す。範囲のリセットは、今と同じく `QValuePanel` が行う | `QValueHeatMapRangeState` |
| 出力 | 0〜1 に正規化した格子値と、その幅・高さ。ガイド線の位置（0〜1。出さないときは負の値） | `QValueHeatMapResult` |

計算の中身（Advantage の引き算、範囲の広げ方と縮め方、bin への振り分けと平滑化、両端の bin を除いた最大値での正規化、Log の変換、値が全部同じときの扱い）は、今の `Update` の 522〜715 行と同じにする。`QValuePanel` は、結果を今と同じ手順で絵にして、ヒートマップの部品へ渡す。

平滑化の重みは、今のコンストラクタと同じ式（137〜146 行）で半径から作る。重みを作る場所（`QValuePanel` が作って指定に入れるか、加工の中で作るか）は、実装計画で決める。平滑化が有効で半径が 0 以下の入力は、今は振る舞いが決まっていないので（§2.1）、加工の契約の外とし、テストで使わない。この不具合は本 PRD で直さない（D15）。課題に記録した（§7）。加工の前に値を写す回数は、今と同じ 1 回にする（D12）。

### 5.3 移行対象と、触らないもの

移行対象は次のとおり。設定とサンプルに移すものは無い。

- §5.1 の表のコードとテスト。
- 設計文書 160 の日英: `anet::rl::gui::Panel` の行（[160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) の 32 行、[160_applications_and_tools.en.md](../design/160_applications_and_tools.en.md) の 33 行）に、D14 の内容を足す。
- 題材の 8 件の課題の記録（対応内容と AI の判定）。

地図（[110](../wiki/apps/110_runner_app.md)・[312](../wiki/categories/312_util.md)・[319](../wiki/categories/319_gui.md)・[321](../wiki/categories/321_agent_core.md)）の「成立の経緯」「選ばなかった形」「変えるときの条件」は、この改訂と同時に書き戻したので、実装の移行対象ではない。

変える経路を書いている箇所を検索した結果は次のとおり。

- キーの転送: `git grep -n -i -E "OnKey|CHAR_HOOK|ForwardedKeyEvent|EVT_FORWARDED_KEY|多重発火|多重の転送|キーの転送|転送された"` を `docs`・`AGENTS.md`・`CONTEXT.md`・`.agents`・`.claude` に当て、過去の PRD（`docs/memo/done`）、課題の記録、本 PRD を除いた当たりは 11 行である。ユーザーガイド 020 の 470・529 行と設計文書 160 の 131 行は、転送された矢印の使い方の記述で、転送の中身と解釈を変えないので直さない。PRD922 の 18・19・312・347 行、PRD923 の 19 行、PRD944 の 61 行は、転送と `RunnerFrame::OnKey` の経路の記述で、D13 のあとも成り立つので直さない。PRD954 の 62 行と wiki の 319 の課題の行は、別の語に当たっただけである。直すのは、上に挙げた設計文書 160 の `Panel` の行だけである。
- `InitRL` の device の検査: `git grep -n -i -E "InitRL|explicit_cuda|明示.{0,6}cuda|一括.{0,10}(報告|検査)|device.{0,20}既定" -- docs/design AGENTS.md CONTEXT.md` の当たりは 9 行で、Agent の factory を `InitRL()` で登録する記述（設計文書 110 の日英の 4 行）と、3 つのキーの既定値の記述（ユーザーガイド 020 の 61・63・64 行、設計文書 100 の 327 行、設計文書 110 の 189 行）である。既定値は持ち主のクラスのもので変わらないので、直すものは無い。
- `GetScalar`: `git grep -n "GetScalar" -- docs/design AGENTS.md CONTEXT.md` の 40 件は、戻り値の意味、key の名前と意味、値の作り方、汎用層の責任境界の記述で、本体をどこに置くか（ヘッダの中か .cpp か）を書くものは無い。
- `GetScalar` の補助: `git grep -n -E "rainbow_agent_impl|GetPlasticityScalar|ValidateSpectralNormSentinel|TransQToCpu"`（課題の記録と本 PRD を除く）の当たりは、dqn_based_agent.hpp の中と、ImageCls の同名の別の関数（image_cls_agent.hpp）、過去の PRD065 の 1 行である。直すのは dqn_based_agent.hpp と dqn_based_agent.cpp だけである。
- QValuePanel の加工: `git grep -n -i -E "QValuePanel::Update|QValueData|hist_bins|hist_range_k|smooth_radius" -- docs/design AGENTS.md CONTEXT.md` は 0 件である。

触らないもの:

- `GetScalar` の key の名前と、それを参照する `metrics.scalar` の設定（§3.2）。
- PRD117 が扱う scaler の WARN と、DefaultDQNAgent の経路での `post_process_type` の検証（§3.2）。
- QValuePanel の `CreateData`（317〜399 行。PRD948 の範囲）と設定のキー、`RunnerFrame::OnKey` の解釈（PRD922 の範囲）。
- `Panel::OnMouse` と転送イベントの型。
- 過去の PRD・実装記録・課題の単票の題と事象に書かれた当時の行番号。

### 5.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
| --- | --- | --- | --- |
| `BatchUpdateResult`（`anet::rl::dqn`） | 変更 | `GetScalar` の宣言だけをヘッダに残し、本体と、`GetScalar` だけが呼ぶ補助の本体を .cpp に置く | dqn_based_agent |
| `QValuePanel` | 変更 | `Update` を、Grid の更新、加工の呼び出し、描画に分ける。加工の入力を作り、累積範囲を持つ | QValuePanel |
| `QValueHeatMapRequest` | 新規 | ヒートマップの加工への指定（表示のモード、Auto Range、Histogram の設定）を運ぶ | QValuePanel |
| `QValueHeatMapRangeState` | 新規 | Histogram の累積範囲（最小と最大）を、呼び出しをまたいで持つ | QValuePanel |
| `QValueHeatMapResult` | 新規 | 加工の結果（正規化した格子値、幅・高さ、ガイド線の位置）を運ぶ | QValuePanel |
| `Panel`（`anet::rl::gui`） | 変更 | `wxEVT_CHAR_HOOK` を受けて転送イベントを出したあと、親への伝播を止める。発生元と親の型の照会を持たない | gui |

`ComputeQValueHeatMap` は関数で、置き場は QValuePanel である。新しい 3 つの型は、テストから使うので QValuePanel.hpp に置く（既存の `QValueData` と同じ置き方）。名前は、既存の語に合わせた。`Request` は「何を計算するかの指定」（`PlasticityMetricRequest`）、`Result` は「計算の結果」（`OptimizerStepResult`）、`State` は「呼び出しをまたいで持つ値」（`GradientNoiseState`・`PolicyChurnState`）、`Compute` は「入力から結果を計算する」（`ComputePlasticityMetrics`）である。用語集（[CONTEXT.md](../../CONTEXT.md)）に当たる語は無く、足す用語も無い。

`DefaultReplayBuffer`・scaler・MuZero の Replay のクラスと、`InitRL` の関数の宣言は変えない。

### 5.5 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-d4aa77](../management/issues/issue-20261103-d4aa77.md) | DefaultReplayBufferのコンストラクタの、priority storeとcompleterを対で持つ不変条件にテストが無い | 318_rl_core | core/anet-core/src/replay_buffer_impl.cpp:1555、core/anet-core/src/replay_buffer_test.cpp:1804 | D1・D2 | A1・A10・A11 |
| [issue-20261103-d337bd](../management/issues/issue-20261103-d337bd.md) | 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効） | 314_tensor | core/anet-core/src/scaler_impl.cpp:892、core/anet-core/src/scaler_impl.cpp:449、core/anet-core/src/scaler_impl.cpp:392、core/anet-core/src/scaler_test.cpp:12 | D1・D3・D15 | A2・A10・A11 |
| [issue-20261103-60f11c](../management/issues/issue-20261103-60f11c.md) | MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い | 318_rl_core | core/anet-core/src/muzero_rb.cpp:98、core/anet-core/src/muzero_rb.cpp:222、core/anet-core/src/muzero_rb.cpp:287、core/anet-core/src/prd101_muzero_test.cpp:4 | D1・D4・D15 | A3・A10・A11 |
| [issue-20261103-03a1b6](../management/issues/issue-20261103-03a1b6.md) | dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い | 321_agent_core | core/anet-core/src/dqn_based_agent.hpp:1 | D1・D5・D8 | A4・A7・A10・A11 |
| [issue-20261103-741251](../management/issues/issue-20261103-741251.md) | BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖 | 321_agent_core | core/anet-core/src/dqn_based_agent.hpp:285、core/anet-core/src/dqn_based_agent.hpp:537 | D1・D6・D7・D8・D15 | A5・A7・A10・A11 |
| [issue-20261103-c1e48b](../management/issues/issue-20261103-c1e48b.md) | InitRLがagent.device・env.device・run.eval_deviceの既定値を、所有するConfigとは別に持っている | 312_util | core/anet-core/src/init.cpp:95、core/anet-core/include/anet/agent.hpp:394、core/anet-core/include/anet/env.hpp:348、core/anet-core/src/trainer.cpp:808 | D1・D8・D9 | A6・A7・A10・A11 |
| [issue-20261103-58bf49](../management/issues/issue-20261103-58bf49.md) | QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う | 110_runner_app | apps/runner/src/QValuePanel.cpp:420 | D1・D10・D11・D12・D15 | A8・A10・A11 |
| [issue-20261103-e088eb](../management/issues/issue-20261103-e088eb.md) | Panel::OnKeyがproductionコードでdynamic_castを3回使う | 319_gui | core/anet-core/src/gui.cpp:95、docs/quality/impl.md:230 | D1・D13・D14 | A9・A10・A11 |

## 6. 検証と受入

A1〜A7 はステージ 1、A8・A9 はステージ 2 の受入である。A10・A11 は、ステージごとに、そのステージの題材について確かめる。「実装の工程で一時的に変える」項は、変えた行、落ちたテストの名前、元に戻したことを実装記録に書く。

### A1. ReplayBuffer のコンストラクタ（d4aa77）

- replay_buffer_test.cpp に、コンストラクタへ priority store と completer の片方だけを渡す 2 通りのテストがあり、どちらも例外を投げ、例外の文言に `must be configured together` が入る。両方ありと両方なしで構築できることを確かめるテストがある（D2）。
- 実装の工程で、replay_buffer_impl.cpp の 1555〜1557 行の検査を一時的に消すと、片方だけの 2 通りのテストが落ちる。

### A2. scaler の 3 つの経路（d337bd）

- scaler_test.cpp に、`use_auto_post_scale = true`、robust update、SymLog のテストが 1 本以上ずつあり、期待値はテストの中で gamma・`reference_q_std`・閾値・scale から計算した値である（D3）。gamma は 0.5 以外、SymLog の `post_process_threshold` は 1 以外の正の値である。robust update のテストは、warmup の件数の境目の前後の 1 件ずつを含む。
- `git grep -n "post_process_threshold" -- core/anet-core/src/scaler_test.cpp` の当たりに、0 以下の値を設定する行が無い（D3）。
- robust update のテストは、warmup の件数の境目を、1 行ずつ与えて見る（統計の件数は、与えた行の数だけ増える。§2.1）。
- 実装の工程で、次の 4 つを 1 つずつ一時的に変えると、対応するテストが落ちる。post_scale の式の `(1.0f - gamma)` を `gamma` にする（scaler_impl.cpp の 894 行）。robust update の、行を外す判定の `deviation > threshold` を `deviation < threshold` にする（495 行）。warmup の判定の `<` を `<=` にする（463 行）。SymLog の `* scale` を外す（404 行）。469 行の `deviation <= threshold` は、全部が正常なときの近道の判定で、向きを変えても統計は変わらないので、書き換えの対象にしない。

### A3. MuZero の Replay（60f11c）

- D4 の (a)〜(e) の各場面のテストがあり、価値ターゲットと mask の期待値を、テストの中で報酬と割引率から計算した値と比べる。
- 実装の工程で、次の 2 つを 1 つずつ一時的に変える。muzero_rb.cpp の 104 行の `steps_to_done` を 1 減らす。232 行の、終端のあとの `seq_masks[k]` を 1 にする。どちらの変更でも、対応するテストが落ちる。

### A4. インクルードガード（03a1b6）

`git grep -n "#pragma once" -- core/anet-core/src/dqn_based_agent.hpp` が 1 件で、`git grep -n "rainbow_agent_impl" -- core/anet-core/src/dqn_based_agent.hpp` が 0 件である。

### A5. `GetScalar` の置き場（741251）

- 特性テスト（D7）が、認識する key の全部を、値がある状態と無い状態で引き、key ごとの今の戻り値（値、NaN、0 などの決まった値、`std::nullopt` の別）を確かめる。未知の key 1 つも確かめる。テストが引く key の集まりは、直接比べる 43 個を並べたものと、8 つの群の key（key の配列を持つ群は配列の全部、名前を解析する群は解析が受ける名前の全部。attention entropy は 0 番と、層の数を超える番号）である。本体を移す前の作業ツリーと、移したあとの作業ツリーの両方で通る。
- 特性テストが、`ANET_SYSTEM_ERROR` で止まる経路（§2.1 の 741251 の行の一覧の 8 つ）を通り、止まることを確かめる。次の 5 つは必ず通る。形の崩れた attention の key、Munchausen の診断値の形、policy churn の件数、可塑性の重みノルムの件数、スペクトル正規化の番人の「Network が無い」（603 行）。残りの 3 つ（番人の 612・616 行、可塑性の指標の控えの 591 行）も通す。通さなくてよいのは、本体を変えない限り、公開のメンバへの値の代入・fixture・テスト用の派生クラスのどれでも到達できない場合だけで、そのときは、経路と、3 つの手段のそれぞれで到達できない理由が実装記録に書かれている。準備が大きいことと、通常の学習の経路で起きないことは、理由にならない（D7）。
- `git grep -c 'key == "' -- core/anet-core/src/dqn_based_agent.hpp` が 1 である（直す前は 50。残る 1 は `GetTensor` の `max_q`）。
- `git grep -c -E "GetScalar\(.*\) const override;" -- core/anet-core/src/dqn_based_agent.hpp` が、直す前の数より 1 多い（2026-11-10 の作業ツリーでは 4 が 5 になる。今の 4 は同じヘッダのほかのクラスの宣言で、増える 1 は、宣言だけになった `BatchUpdateResult` の `GetScalar` である）。
- 直接比べていた 43 個の key のどれも、dqn_based_agent.hpp と dqn_based_agent.cpp に `key == "<名前>"` の形で残っていない（表の行になる。D6）。43 個の名前は次のとおりで、それぞれを `git grep -n -F 'key == "<名前>"' -- core/anet-core/src/dqn_based_agent.hpp core/anet-core/src/dqn_based_agent.cpp` で引いて、どれも 0 件である。`loss`、`td_mean`、`td_std`、`grad_norm`、`grad_clip_ratio`、`q_max_max`、`q_max_mean`、`q_max_std`、`q_sa_mean`、`q_max_real_max`、`q_max_real_mean`、`q_max_real_std`、`q_sa_real_mean`、`q_std`、`q_gap`、`q_gap_rel`、`iqn_current_mc_scale`、`iqn_target_mc_scale`、`iqn_priority_mc_ratio`、`iqn_first_priority_mc_ratio`、`iqn_first_pair_abs_td`、`iqn_first_cancellation_ratio`、`iqn_first_quantile_loss_norm`、`upper_tail_priority_spearman`、`per_td_error_abs_max`、`per_prio_clip_ratio`、`per_sample_initial_ratio`、`per_sample_initial_count`、`per_sample_fixed_initial_ratio`、`per_sample_max_initial_ratio`、`per_sample_actor_initial_ratio`、`per_priority_update_stale_ratio`、`per_actor_learner_pair_count`、`per_actor_learner_positive_pair_ratio`、`per_actor_learner_ratio_median`、`per_actor_learner_log_ratio_mean`、`per_actor_learner_spearman`、`per_prio_max`、`per_batch_prio_mean`、`per_is_weight_mean`、`per_prio_cv`、`per_is_ess_ratio`、`per_prio_ess_ratio`。
- 実装の工程で、次の 2 つを 1 つずつ一時的に変える。表の行の 1 つで key の綴りを変えると、特性テストが落ちる。表に、既にある key と同じ key の行を 1 行足すと、ビルドかテストが落ちる（D6）。

### A6. device の既定値（c1e48b）

- `git grep -n -E '"(agent\.device|env\.device|run\.eval_device)", *"(auto|cpu)"' -- core/anet-core/src/init.cpp` が 0 件である（キーの名前と既定値の対に当たる。今は 97 行の 1 行が当たる。D9 が残すキーの名前の並びは、名前の次に既定値の文字列が続かないので当たらない）。
- `git grep -n "config_data.Has(" -- core/anet-core/src/init.cpp` が 1 件以上である（device の検査が、キーが書かれているかを問う。今は 0 件）。
- 環境変数 `CUDA_VISIBLE_DEVICES` を `-1` にして `core\anet-core\bin\Debug\anet-core-test.exe "[prd070]"` を流すと、skip が 0 件で、すべて通る（§2.1）。CUDA が見える状態で流したときの skip を、合格に数えない。
- init_test.cpp に、3 つのキーのうち 1 つだけを不正な書き方で置いた設定で `InitRL` が止まり、文言にそのキーの名前が入ることを確かめるテストがある。3 つのキーのどれも書かれていない設定で `InitRL` が止まらないことを確かめるテスト（既存の 192 行）が通る（D9）。

### A7. 短い Run の前後比較（ステージ 1）

- Run は、環境ごとに変更前 1 本と変更後 1 本である。条件は [PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md) の「段0・smoke」の短縮条件（GridMaze・LunarLander・Atari qbert・DropMerge、seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、FP32、決定的 backend）で、変更前はステージ 1 に入る前に保存した Debug の実行体、変更後はステージ 1 を終えた Debug の実行体を使う。
- 4 環境とも、`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar の点が、tag・step・値とも一致する（D8）。除く 3 つは時間と資源の値で、Run ごとに変わる。
- 検証資材の手順が今の設定で動かないときは、同じ条件と同じ比べ方（全 scalar の点を tag・step・値で突き合わせる）を満たす手順を実装計画に書き、それで確かめる。比較を省いて合格にしない。

### A8. QValuePanel の関数の分け方と加工のテスト（58bf49）

- QValuePanel.cpp に `ComputeQValueHeatMap` の定義があり、`QValuePanel::Update` がそれを呼ぶ。Grid の更新が `Update` とは別の関数にある（D10）。
- `ComputeQValueHeatMap` の単体テストが Runner の GUI のテストの実行体（`AnetRLRunner-gui-test`）にあり、D11 に挙げた場合のすべてを通る。期待値は手で計算した値である。
- 実装記録に、加工の前に値を写す箇所が 1 つであることと、その行が書かれている（D12）。
- その単体テストは、`QValuePanel` も wx の window も作らずに `ComputeQValueHeatMap` を呼ぶ（加工が画面の部品に依らないことを、呼べることで確かめる。D10）。
- 実装記録に、切り出す前のコードで控えた加工後の値（モードの組み合わせごとに 1 つ以上の入力。入力は単体テストと同じ）と、単体テストの期待値が一致したことが書かれている（D11）。
- 実装の工程で、次の 2 つを 1 つずつ一時的に変えると、単体テストが落ちる。Advantage の引き算を外す。Histogram の色付けで、両端の bin を除く条件を外す。
- 実装記録に、Grid の差分更新の 2 つの条件（選択が変わった行だけを塗り直す、値が広がった列だけ幅を当て直す）が分けたあとの Grid の関数に残っていることと、その行が書かれている（D10・D12）。
- 人が Runner で確かめる: 評価を流し、Q-Values の表示が出て、Histogram・Advantage・Log・Auto Range の切替で絵が変わる。行の選択とクリックで、ハイライトと選ばれる行動が直す前と同じである。

### A9. キーの転送（e088eb）

- 特性テスト（D14）が次を確かめる。イベントは、発生元の window に、発生元をその window にした `wxEVT_CHAR_HOOK` を、window のイベントの処理へ送る形で渡す。(a) 入れ子の `Panel` の中の window が発生元のとき、転送イベントは 1 回で、転送元は内側の `Panel` である。(b) 発生元が内側の `Panel` 自身のときも同じである。(c) `Panel` の直下の window が発生元のとき、転送イベントは 1 回で、転送元はその `Panel` である。(d) `Panel` の外の window が発生元のとき、転送イベントは 0 回である。(e) どの場面でも、`wxEVT_CHAR_HOOK` は未処理のまま返る。直す前の作業ツリーと、直したあとの作業ツリーの両方で通る。
- `git grep -n "dynamic_cast" -- core/anet-core/src/gui.cpp core/anet-core/include/anet/gui.hpp` が 0 件である。
- GUI のテスト `[prd105_ui_keys]`・`[eval_steps_menu]`・`[prd109_ui]` が通る（それぞれ別のプロセスで流す。[PRD109 の検証資材](../../core/anet-core/testdata/prd109/README.md)）。
- 設計文書 160 の日英の `anet::rl::gui::Panel` の行に、キーの転送と伝播の停止、`Panel` より上では転送イベントを使うことが書かれている。`git grep -n "EVT_FORWARDED_KEY" -- docs/design/160_applications_and_tools.jp.md docs/design/160_applications_and_tools.en.md` が、それぞれ 1 件以上である。
- 実装の工程で、`StopPropagation` の呼び出しを一時的に外すと、特性テストの (a) か (b) が落ちる（転送イベントが 2 回になる）。
- 人が Runner で確かめる: Live モードで、矢印キー・テンキー・Space・Ctrl を 1 回押すと、行動の割り当てと step の進みが 1 回分だけ起きる。Q-Values のヒートマップと Env の View にフォーカスを置いた状態でも同じである。Evaluation Trace の表と Logs にフォーカスがあるときのキーの効き方が、直す前と同じである。QValuePanel のクリックで、行動が 1 回だけ選ばれる。Tab でのフォーカスの移動が、直す前と同じである。

### A10. ビルドと既存のテスト

[AGENTS.md](../../AGENTS.md) の手順で Debug ビルドを通し、既定のテストスイートと `"[shipped_config]"` を流す。ステージ 2 では、Runner の GUI のテストの実行体のうち隠しタグでないテストと、A9 の GUI のテストも流す。既知の失敗（PRD093 の attention のテスト 1 件）のほかに、直す前に通っていたテストが落ちない。結果と、流さなかったものがあればその理由を実装記録に書く。

### A11. 課題の記録

そのステージの題材の課題の対応内容に、直した内容と場所が書かれ、AI の判定が `残件無し` である。状態を `解決` にするのは開発者である（§2.2）。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| 生きている `Panel` の集まりを持ち、発生元と親がその集まりに入っているかで判定する | 不採用（D13） | ファイルの中に可変の状態が 1 つ増え、発生元を window として扱う変換が残る。親をたどる処理も残る |
| `Panel` かどうかを wx の型情報（`wxDynamicCast`）で判定する | 不採用（D13） | 機械検査は通るが、型の照会であることは変わらない。課題が挙げた「`Panel` を継承しない部品がキーを転送するときに判定を書き足す」が残る |
| `Panel::OnKey` の `dynamic_cast` を規約の例外にして残す | 不採用（D13） | 課題が片づかない |
| 転送済みであることを、キーのイベントの側の印で表す | 不採用（D13） | キーのイベントに利用者の印を置く欄が無く、印を外の状態に持つことになる |
| `RunnerFrame` が `wxEVT_CHAR_HOOK` を 1 回だけ受けて転送する | 不採用（D13） | `Panel` の外にフォーカスがあるときのキーも転送してしまい、G2 に反する。避けるには、発生元が `Panel` の中かを調べる型の照会が要る |
| priority store と completer を 1 つの組にして、片方だけの状態を作れなくする | 不採用（D2） | 検査とテストは要らなくなるが、`DefaultReplayBuffer` のコンストラクタとファクトリの形が変わる。課題はテストの空白で、変更が見合わない |
| ヒートマップの加工を core の GUI 共通へ移し、既定のテストスイートで流す | 不採用（D11） | 加工と `QValueData` を使うのは Runner の Q-Values の表示だけである。テストを既定のスイートに入れるために、アプリケーションの部品を core へ置くことになる |
| QValuePanel の、平滑化が有効で半径が 0 以下のときに空の重みを読む不具合を直す | 範囲外（D15） | 振る舞いを変える直しである。[issue-20261110-3f4e63](../management/issues/issue-20261110-3f4e63.md) に記録した |
| MuZero の truncated の補いの値を直す | 範囲外（D15） | 学習ターゲットが変わる。[issue-20261110-d21e5f](../management/issues/issue-20261110-d21e5f.md) に記録した（D4） |
| Runner の GUI のテストの実行体を、既定のビルドと標準のテスト手順に入れる | 範囲外（別の関心） | ビルドとテストの自動化の関心である。[issue-20261103-61d771](../management/issues/issue-20261103-61d771.md)（PRD906 の課題）に追記した。再開条件: PRD906 のグリル |
| `GetScalar` の key の全部を 1 枚の表にする | 不採用（D6） | 名前を解析して受ける群（attention entropy など）は、名前から番号を読むので表に載らない。表にするのは、直接比べていた 43 個だけにする |
| 直接比べていた 43 個も群ごとの関数に分け、中を if の連鎖のままにする | 不採用（D6） | key の重複や届かない分岐が、機械では見つからないまま残る。key が文字列の直書きのまま散らばる |
| `InitRL` が既定値を持つクラスの定数を引く | 不採用（D9） | `RunManager::Config` を trainer.cpp の外へ出す変更が要り、書かれていないキーを飛ばせば今の結果のまま二重持ちが消える |
| `QValuePanel::Update` を、メンバを直接読む private な関数に分けるだけにする | 不採用（D10） | 加工を壊したときに気づく手段が、人の目のままになる |
| Runner を起動して Grid のセルとハイライトを読む GUI のテスト | 不採用（D11） | Grid の更新は中身を変えずに関数へ移すだけで、Runner を起動するテストの保守に見合わない |
| QValuePanel の Grid の死にコード（無効化された引き算、コメントアウト）を消す | 範囲外（別の関心） | もとからある死にコードは消さずに報告する決まり（§2.2）。再開条件: 片づけを別の課題として起こすとき |
| テストで見つかった今の振る舞いの誤りを直す | 範囲外（D15） | 振る舞いを変えると G2 に反する。見つかったら起票の候補にする |
| scaler の WARN と、DefaultDQNAgent の経路での `post_process_type` の検証 | 範囲外（別の関心） | PRD117 の範囲（§3.2） |
| `InitRL` の fatal の二重のログと、エラーの文言 | 範囲外（別の関心） | ログと文言を変える直しで、G2 に反する（§3.2）。再開条件: 課題 78df9c・9b314a に取り組むとき |

## 8. 未決

なし。暫定の起票の未決表（14 行）は、2026-11-10 のグリルですべて §4 の確定事項へ移した。

| 暫定の未決表の行 | 移した先 |
| --- | --- |
| 1（束ね方と実装の順） | D1 |
| 2（d4aa77 のテストの形） | D2 |
| 3（d337bd のテストの範囲） | D3 |
| 4（60f11c のテストの範囲） | D4 |
| 5（03a1b6 の直し方） | D5 |
| 6（741251 の分け方） | D6 |
| 7（741251 の特性テスト） | D7 |
| 8（741251 の性能） | D8 |
| 9（c1e48b の直し方） | D9 |
| 10（58bf49 の分け方） | D10・D11 |
| 11（58bf49 の性能の確かめ方） | D12 |
| 12（テストで誤りが見つかったとき） | D15 |
| 13（e088eb の判定の仕方） | D13 |
| 14（e088eb の確かめ方） | D14 |

## 9. 依存と順序

- 先に要る PRD は無い。
- [PRD117](117_warn_policy_and_unread_config_10prd.md) は scaler_impl.cpp の 420 行の WARN と 505 行付近の DEBUG の出力を変え、DefaultDQNAgent の経路に `post_process_type` の構築時の検証を足す。本 PRD の scaler のテストは有効な値だけを使い、ログを見ないので、どちらを先に入れてもよい（D3）。同じファイルの同じ関数（`NormalizeAndUpdateStats`）に触るので、並べて実装するときは片方を先に入れる。
- [PRD948](948_qvaluepanel_manual_action_state_alignment_10prd.md) は QValuePanel.cpp の `CreateData` に触る。本 PRD は `Update` だけを変えるので、どちらを先に入れてもよい。
- [PRD922](922_input_mapping_joystick_10prd.md) は `RunnerFrame::OnKey` の解釈を作り替える。本 PRD は `Panel::OnKey` の伝播だけを変え、転送イベントの中身は変えないので、どちらを先に入れてもよい。PRD922 が `Panel` より上の window でキーを受ける部品を足すときは、転送イベントを使う（D13）。
- [PRD108](108_metrics_tag_numbering_10prd.md) がメトリクスの key の名前を変えるときは、本 PRD の `GetScalar` の表（D6）と特性テスト（D7）の key の並びも直す。

## 10. 参照

- 規約: [リファクタリング観点](../quality/refactor.md) の「ガードレール」、[実装 品質確認観点](../quality/impl.md) の「C++ コード」「GetScalar」と項番 2・35・42・60、[PRD 品質確認観点](../quality/prd.md) の「受入」、[PRD079](079_cpp_internal_linkage_convention_10prd.md)。
- 地図: [110 RunnerApp](../wiki/apps/110_runner_app.md)、[312 ユーティリティ](../wiki/categories/312_util.md)、[314 テンソル処理](../wiki/categories/314_tensor.md)、[318 強化学習 共通基盤](../wiki/categories/318_rl_core.md)、[319 GUI 共通基盤](../wiki/categories/319_gui.md)、[321 Agent 共通基盤](../wiki/categories/321_agent_core.md)。
- 検証資材: [PRD098](../../core/anet-core/testdata/prd098/README.md)（短い Run の前後比較）、[PRD105](../../core/anet-core/testdata/prd105/README.md)・[PRD109](../../core/anet-core/testdata/prd109/README.md)（GUI のテストの流し方）。
- 課題の記録: [anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)。
