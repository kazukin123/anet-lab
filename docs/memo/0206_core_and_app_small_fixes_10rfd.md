# RFD0206: 基盤とアプリの小さい直し 12 件（設定の値域と数の読み取り、Replay の n-step と先読み、行動の device の控え、テストと検査の道具、名前の整理、bat の題）

> 2026-11-27 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §4 の未決表に置いた。§5〜§7 は §4 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: 2026-11-27 の会話。開発者は、仕様の調整がほぼ要らない小さい課題を集め、寝ている間に実装させたいと述べた。課題の台帳から、対応の方針が 1 つに決まっている課題 34 件を選び、置き場ごとに 3 本（RFD0205 は文書、本 RFD は基盤とアプリ、RFD0207 は DQN 系）に分けることを開発者が選んだ（§9）。
> 課題: [issue-20261123-003c85](../management/issues/issue-20261123-003c85.md)・[issue-20261123-0c9793](../management/issues/issue-20261123-0c9793.md)・[issue-20261120-1979f6](../management/issues/issue-20261120-1979f6.md)・[issue-20261123-446e6f](../management/issues/issue-20261123-446e6f.md)・[issue-20261123-7671ff](../management/issues/issue-20261123-7671ff.md)・[issue-20261123-ff9a9a](../management/issues/issue-20261123-ff9a9a.md)・[issue-20261123-9a0e78](../management/issues/issue-20261123-9a0e78.md)・[issue-20261122-4adabb](../management/issues/issue-20261122-4adabb.md)・[issue-20261123-27b23f](../management/issues/issue-20261123-27b23f.md)・[issue-20261123-2709db](../management/issues/issue-20261123-2709db.md)・[issue-20261123-4df9cc](../management/issues/issue-20261123-4df9cc.md)・[issue-20261123-26f763](../management/issues/issue-20261123-26f763.md)（いずれも題材）。

## 1. 目的と問題

### 1.1 問題

2026-11-20〜23 の監査と実装の検証で、基盤（`src/core/base`）、Runner（`src/apps/runner`）、ImageCls（`src/algo/imagecls`・`src/env/imagecls`）、検査の道具と bat に、小さい欠けが 12 件見つかった（§3）。種類は 5 つである。

- 設定の値域や数の読み取りを確かめず、不正な値で黙って動くか、設定キーの無い文言で止まる（003c85・0c9793・1979f6・446e6f）。
- 例外や device の扱いの誤りで、条件がそろうと止まる（7671ff・ff9a9a）。
- テストと検査の道具が、今の配置や初期化の順で動かない（9a0e78・4adabb）。
- 名前と中身が合わない、使われない引数と重複した記録がある、基盤がアプリの名前を持つ（27b23f・2709db・4df9cc）。
- bat の題の規約から 2 本が外れている（26f763）。

どの課題も、課題の記録に直し方が 1 つ書いてあるか、規約から直し方が決まる。決めきれない細部は §4 に既定を置いた。

### 1.2 ゴール

- G1. 題材の課題 12 件がすべて片付く（各課題の事象が残らない）。
- G2. 正しい設定の Run の結果を変えない。§7 の A15 の 6 構成で、時間と資源を除く全 scalar が実装の前後で一致する。わざと変える点は 27b23f（ImageCls の構築の INFO とメトリクスの設定の記録が 1 つずつ減る）だけで、6 構成には現れない（U10）。
- G3. 不正な設定（n_step・ImageCls の 4 キー・QValuePanel の hist_bins・メトリクスの宣言の数の属性）は、Run の構築の間に、完全キーと指定値と期待値を示して止まる（[AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」「設定値の扱い」。P1・P2）。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| メトリクスの graph の節の読み取りを、scalar・trace と同じ字句の解析へ寄せる（[issue-20261103-78c643](../management/issues/issue-20261103-78c643.md)。446e6f の課題が「一緒に直すと手間が減る」とする） | 字句の解析の作り直しは別の関心で、受け付ける語が変わりうる。本 RFD は数の読み取りだけを直す | 制約ではない。78c643 で行える |
| graph の宣言の未知の語の扱いと、`interval`・`episode_interval` の値域（[issue-20261125-29c3b2](../management/issues/issue-20261125-29c3b2.md)） | 止めるか既定の間隔を文書に書くかの判断が残る（課題に「決める」とある） | 制約ではない。29c3b2 で決める |
| RunManager のコンストラクタの分割と、名前の予約の検査を 1 つにまとめること（[issue-20261111-4a6b6f](../management/issues/issue-20261111-4a6b6f.md)。4df9cc の課題が「一緒に直すと手間が減る」とする） | 277 行の関数の分け方は別の関心で、分け方の判断が要る | 制約ではない。4a6b6f で行える |
| 先読みの Replay で、同期の前にアプリを閉じ、デストラクタが worker の例外を最初に受け取る場合の扱い（U7） | 例外を呼び出し元へ返せない場面で、ログに残して続けるか止めるかの判断が要る。課題の事象（一度投げたあとの残り）とは別の場面 | 制約ではない。起票の候補（U7） |
| テストの実行体で、wx のエラーがダイアログを出さない形にする（9a0e78 の課題が「あわせて決める」とする。U8） | 実行体ごとのログの出し先の設計で、9a0e78 の事象（初期化の抜け）を直せば知られた場面は無くなる | 制約ではない。起票の候補（U8） |
| `learner.use_n_step = false` のときの `learner.n_step` の値域の検証（U16） | DQN 系の設定の検証の置き場（`src/algo/dqn`）で、本 RFD の置き場の外 | 制約ではない。起票の候補（U16） |
| bootstrap.bat を今のビルドの手順（cmake のプリセット）に合わせて書き直すこと | 26f763 の事象は題の行だけで、bat の中身は別の関心 | 制約ではない |

### 1.4 機構とゴールの対応

新しい機構と概念は足さない。変えるものとゴールの対応は次のとおり。

| 変えるもの | ゴール |
|---|---|
| 値域と数の読み取りの検査（U2〜U5） | G1・G3 |
| 行動の device の控えの鍵、先読みの待ちの片づけ（U6・U7） | G1・G2 |
| テストの初期化、検査の道具の確かめ（U8・U9） | G1 |
| 名前・引数・記録の整理（U10〜U12） | G1・G2 |
| bat の題（U13） | G1 |
| 文書の追随（U14） | G1 |

### 1.5 成功の測り方

- §7 の受入が全部通り、題材の 12 件の AI の判定が「残件無し」になる。
- 6 構成の比較（A15）で、時間と資源を除く全 scalar の点が一致する（G2）。

## 2. 前提

| 番号 | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 守る線 | Run の構築が終わるまでに判定できる設定の違反は `ANET_SYSTEM_ERROR` で止め、実行時へ先送りしない。使われない側の設定キーも、そのキーだけで決まる値域を検証する。動作不能な値を黙って置き換えない | [AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」（424 行）・「設定値の扱い」（454 行） | 対象外 |
| P2 | 守る線 | エラーやログが示す設定キーは完全キーにし、指定値と期待値を添える | [RFD0194](0194_config_key_full_names_in_messages_10rfd.md)、[AGENTS.md](../../AGENTS.md) の「設定値の扱い」の最後の箇条 | 対象外 |
| P3 | 過去の判断 | 読み口と配線の整理では、metrics の出力・定義レコード・WARN・fail-fast の文言と時機を変えない | [RFD0094](done/0094_evalpanel_step_history_10rfd.md) の §4.1 D3、[318 のトピック別まとめ](../wiki/concepts/framework/318_rl_core.md) の「変えるときの条件」 | 対象外 |
| P4 | 既存部品の働き | RFD0103 の検証資材は、8 lane・batch 16・warmup 256・容量 4096・RR1・seed 1・8704 experience・FP32・決定論的 backend・評価停止の短い Run を、保存した変更前の実行体と今の実行体で走らせ、`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar を比べる。RFD0192 の A13 は 6 構成（GridMaze・LunarLander の skip 無効と有効、Atari・DropMerge の skip 無効）でこれを使った | [RFD0103 の検証資材の README](../../src/core/base/testdata/rfd0103/README.md) の 9〜11 行、[RFD0192](0192_concurrency_and_boundary_defect_fixes_10rfd.md) の §7 A13、[RFD0192 の実装記録](0192_concurrency_and_boundary_defect_fixes_20impl.md) の 29・74 行 | 対象外 |
| P5 | 事実 | `ANET_ASSERT` は `ANET_ENABLE_ASSERT` が 0 のビルドでは中身を評価しない。`ANET_ENABLE_ASSERT` は `ANET_ENABLE_DEBUGINFO` のビルドで既定 1 になる | [diag.hpp](../../src/core/base/include/anet/diag.hpp) の 8〜12・22〜44 行 | 対象外 |
| P6 | 事実 | `ANET_SYSTEM_ERROR` が呼ぶ `ThrowError` は戻らない関数（`[[noreturn]]`）である | [diag.hpp](../../src/core/base/include/anet/diag.hpp) の 72 行 | 対象外 |
| P7 | 守る線 | `.bat` は CP932 と CRLF を保ち、`@echo off` の次の行に `title %~nx0` を置く | [AGENTS.md](../../AGENTS.md) の「改行と文字コード」（209〜214 行） | 対象外 |
| P8 | 事実 | Runner の GUI のテストの実行体 `AnetRLRunner-gui-test` は全体の build に入らない（`EXCLUDE_FROM_ALL`）。QValuePanel と RunnerApp のテストはこの実行体にある | [src/apps/runner/CMakeLists.txt](../../src/apps/runner/CMakeLists.txt) の 150 行 | 対象外 |
| P9 | 過去の判断 | `BatchActionInfo` の控え（キャッシュ）には守り（排他）を足さず、呼んでよい時点をコメントに書く | [RFD0192](0192_concurrency_and_boundary_defect_fixes_10rfd.md) の §5 D7 | 対象外 |
| P10 | 守る線 | 名前を変えるときは、旧い名前の別名や互換の分岐を残さない | [AGENTS.md](../../AGENTS.md) の「クリーンブレーク方針」（359 行） | 対象外 |
| P11 | 既存部品の働き | 先読みの Replay の Push の待ちを片づける 2 つの関数は、待ちの結果を受け取ってから列から外す順で書かれている。受け取りが例外を投げ直すと、受け取り済みの待ちが列の先頭に残る | [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) の 2336〜2352 行、課題 ff9a9a の「事象」 | 対象外 |

## 3. 現状（2026-11-27 に確かめた）

行番号は 2026-11-27 の作業ツリー（HEAD 48c98707）による。課題の記録の行番号と違うものは、ここが新しい。

### 3.1 n_step が 0 以下の Replay（003c85）

- `CreateReplayBuffer`（[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 2523 行）は、容量・`per_initial_priority`・`per_initial_priority_mode` を検査し（2531〜2547 行）、n_step は検査しない。必要な容量は `max(1, n_step, ...)` で計算する（2529〜2530 行）。n-step の列の制御は 2557 行で `config.n_step` から作る。
- n_step が 0 のとき、`NStepQueueController::ExtractSequences`（[replay_experience.cpp](../../src/core/base/src/replay_experience.cpp) 55 行から）は空の系列を出し、受け手は空の系列を飛ばす（replay_buffer_impl.cpp 1676 行）。Size は 0 のままになり、学習は始まらない（課題の「事象」）。
- DQN は `rep_config.n_step = use_n_step ? n_step : 1`（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 995 行）とし、完全キーの `n_step` を 987 行で写してから、1031 行で `CreateReplayBuffer` を呼ぶ。設定の読み取り（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 489 行）に値域の検査は無い。`n_step > 0` を確かめるのは step 診断の部品（[dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 92〜94 行。`use_n_step` が真のときだけ）である。
- `CreateReplayBuffer` の呼び出しは、検索語 `CreateReplayBuffer(`、範囲 `src` で、宣言（replay_buffer.hpp 94 行）と定義のほかに 4 件（dqn_learner.cpp 1031 行、dqn_learner_test.cpp 2241 行、replay_buffer_test.cpp 356・1182・4589 行）。

### 3.2 ImageCls の 4 キーの値域（0c9793）

- `ImageClsAgentConfig::Validate`（[image_cls_agent.hpp](../../src/algo/imagecls/src/image_cls_agent.hpp) 106〜117 行）が検証するのは、mixup の 4 つと `learn_log_interval` だけである。`grad_clip_max_norm`・`label_smoothing`・`weight_decay`・`learning_rate` は読む（61〜66 行）が検証しない。`learning_rate` の下限は置いていない（61 行は値だけ）。DQN の `learning_rate` は下限 0 を持つ（[dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 288 行）。
- 学習は `clip_grad_norm_` に `grad_clip_max_norm` をそのまま渡し（[image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) 408 行）、損失は `label_smoothing` で作り（395 行）、学習率は step ごとに予定の値を入れる（412〜414 行）。
- ProfiledValue は、下限を持つ設定では、予定の各区間の値・始点・終点を構築時に検証する（[schedule.hpp](../../src/core/base/include/anet/schedule.hpp) の `ValidateProfiledValueBounds` 110 行・`ValidateProfiledValuePhaseConfig` 116 行。各区間への呼び出しは 211 行）。この検証は、設定を読み終えたとき（324 行）と ProfiledValue を作るとき（362 行）に走る。
- 出荷設定は `label_smoothing = 0.1`・`grad_clip_max_norm = 1.0`（[ImageCls.txt](../../apps/runner/config/ImageCls.txt) 123・124 行）、学習率は warmup（0 から 1e-3）と cosine（1e-3 から 1e-5）の phased（108〜117 行）で、下限 0 を満たす。
- 範囲外の値を試すテストは、mixup と `learn_log_interval` のもの（[image_cls_agent_test.cpp](../../src/algo/imagecls/src/image_cls_agent_test.cpp) 695 行の TEST_CASE）だけである。

### 3.3 QValuePanel の hist_bins（1979f6）

- `QValuePanel::Config`（[QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) 28〜48 行）は `hist_bins` を読む（39 行）が検証しない。同じ Config の `smooth_radius` は 1 以上を検証し、`QValuePanel.smooth_radius=<値> must be an integer >= 1.` の形で止まる（43〜46 行）。
- 描画の加工は `n_bins = request.hist_bins`（471 行）から `n_actions * n_bins` の領域を作り、ビン番号を `n_bins - 1` で頭打ちにして書く（543〜571 行）。0 では領域が空のまま書きに行く。1 では、中央のピークの探索が両端を飛ばして何も見ず、全体の最大値へ代わる（592〜609 行）ので、壊れない。
- 出荷設定は `QValuePanel.hist_bins ?= 510`（[common.txt](../../apps/runner/config/common.txt) 67 行）。

### 3.4 メトリクスの宣言の数の属性（446e6f）

- scalar の節は `interval` を先頭の数だけ読む整数の変換、`ema_alpha`・`clip` を同じく先頭の数だけ読む小数の変換で読む（[observers.cpp](../../src/core/base/src/observers.cpp) 1435〜1437 行）。graph の節の `interval`・`step_interval`（1666 行）と `episode_interval`・`eps_interval`（1668 行）も同じである。範囲 `src/core/base/src` で、この 2 種の変換を使う箇所は observers.cpp の 5 か所のほか、config.cpp と nn_impl.cpp にある（検索語 `std::sto`）。本 RFD はメトリクスの宣言の 5 か所だけを扱う。
- scalar の `interval` は、読んだあとに 1 以上を確かめる（observers.cpp 796〜803 行）。
- 全体を読み切ったかを確かめる読み方は、ほかの箇所にある（[metrics.cpp](../../src/core/base/src/metrics.cpp) 117・160 行、[rl.cpp](../../src/core/base/src/rl.cpp) 521 行、[tensor_util.cpp](../../src/core/base/src/tensor_util.cpp) 26 行の `from_chars`）。
- 出荷設定と検証資材が書く属性の値は、`apps/runner/config` と `src` を `(interval|step_interval|episode_interval|eps_interval|ema_alpha|clip):` で引いた結果、整数（`100`・`503`・`5000` など）と小数（`0.01`・`0.001`・`0.0002` など）だけである。ほかに当たったのは、コメントの書式の説明（metrics_scalar.txt 6・7・11 行）、Python の検証資材が書式に埋める `{LANES}`・`{lanes}`（埋めてから渡す）、trace の不正な宣言のテスト（[observers_test.cpp](../../src/core/base/src/observers_test.cpp) 1699 行の `interval:no`。trace の節は `interval` の属性を受け付けないので、本 RFD の読み取りを通らない）である。

### 3.5 行動の device の控え（7671ff）

- `BatchActionInfo::GetAction(device)`（[rl.cpp](../../src/core/base/src/rl.cpp) 531 行）は、index の無い cuda を `req_dev = cuda:0` に直し（552〜555 行）、控えの鍵に元の `device`（index なし）を記録する（560 行）。2 回目の呼び出しは鍵と `req_dev` を比べ（563 行）、cuda と cuda:0 の違いで "GPU device mismatch" で止まる（565 行）。
- 呼び出しの例は、[env.cpp](../../src/core/base/src/env.cpp) 774 行と、dqn_learner.cpp 2241・2605・2720 行の `GetAction(device_)`。出荷設定の既定は `env.device ?= cpu`（common.txt 56 行）である。

### 3.6 先読みの Replay の待ちの片づけ（ff9a9a）

- `PruneCompletedPushesLocked`（[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 2336〜2344 行）と `WaitForQueuedPushesLocked`（2346〜2352 行）は、先頭の待ちの結果を受け取ってから列から外す（P11）。
- 次に列に触るのは、Sample の中の片づけ（2417 行）、`WaitForQueuedPushesLocked` の呼び出し（2460 行ほか）、デストラクタ（2377〜2382 行）から呼ぶ `StopPrefetch`（2354 行）である。
- `use_rb_prefetch = true` は、Atari.txt 813 行、DropMerge.txt 216 行、GridMaze.txt 96 行、LunarLander.txt 196 行にある。
- 先読みの Replay のテストは、[replay_buffer_test.cpp](../../src/core/base/src/replay_buffer_test.cpp) の 3290〜3523 行の 9 件と 4779 行の 1 件で、どれも Push が成功する場合である。

### 3.7 imagecls の Env のテストの初期化（9a0e78）

- [ImageClsEnv_test.cpp](../../src/env/imagecls/src/ImageClsEnv_test.cpp) の TEST_CASE は 21 件。テスト用のデータを作る `TinyImageClsDataset` を使うのに `EnsureWxImageSupport()`（193 行）を呼ばないのは、305 行の「ImageDatasetManager keeps manifest construction failures sticky」だけである（ファイルの全体を、TEST_CASE ごとに「`EnsureWxImageSupport();` より前に `TinyImageClsDataset` を作るか」で引いた結果）。
- `TinyImageClsDataset` は PNG を `SaveFile` で書き（126 行）、画像の口が初期化されていないと失敗する。乱数の並びで先に回ると wx のダイアログで止まる（課題の「事象」）。

### 3.8 既定の値の宣言の検査（4adabb）

- [check_default_leaves.py](../../src/core/base/testdata/rfd0072/check_default_leaves.py) の検索先は、27 行で `apps` と `src` になっている。移転前の `core` を `src` に直したのは、RFD0161 のコミット 125075dc（2026-11-23）である。リポジトリの根は 11 行で、ファイルの位置から 5 段上と求める。
- 課題の記録には、この直しも、直したあとに検査を回した結果も書かれていない。

### 3.9 ImageCls の Agent の構築（27b23f）

- `ImageClsAgentConfig` のコンストラクタは 2 つ目の引数 `config_prefix` を取り（image_cls_agent.hpp 58 行）、空でなければ上書きの接頭辞の下の Actor の一覧も読む（80〜84 行）。呼び出しは、検索語 `ImageClsAgentConfig(`、範囲 `src` で、本番の [image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) 775 行（`ImageClsAgentConfig config(config_data)` の形）と、テストの 17 件（image_cls_agent_test.cpp 13 件、imagecls_config_baseline_test.cpp 4 件。宣言の行を除く）で、どれも引数 1 つである。
- `ImageClsAgent` のコンストラクタは、INFO（585 行の `ImageClsAgent config=` と 638 行の `ImageClsAgent initialized. config=`）と、メトリクスの設定の記録（586 行と 637 行）を 2 回ずつ行う。文言 `ImageClsAgent initialized` を引く箇所は、範囲をリポジトリ全体（課題記録を除く）にして 638 行だけである。

### 3.10 観測の検証の関数（2709db）

- `StateSpec::ValidateObservation`（宣言 [rl.hpp](../../src/core/base/include/anet/rl.hpp) 229 行、定義 [rl.cpp](../../src/core/base/src/rl.cpp) 188〜250 行付近）は bool を返し、失敗の 6 経路は `ANET_SYSTEM_ERROR` のあとに届かない `return false` を置く（P6）。隣の `AssertSanity`（rl.hpp 226 行）は戻り値を持たない。
- 呼び出しは [trainer.cpp](../../src/core/base/src/trainer.cpp) 539・680 行の `ANET_ASSERT(...)` の中の 2 件と、Env のテストの 23 件（AtariEnv_test 7、DropMergeEnv_test 14、LunarLanderEnv_test 2。検索語 `ValidateObservation(`、範囲 `src` の `_test.cpp`）。設計文書（`docs/design`）と `CONTEXT.md` に、この名前は無い。

### 3.11 基盤が持つ Runner の名前（4df9cc）

- `RunManager` のコンストラクタ（[trainer.cpp](../../src/core/base/src/trainer.cpp) 924 行）は、`validate_planned_name("EvalPanel", "EvalPanel")`（959 行）で名前 "EvalPanel" を予約する。その名前で Runner を作るのはアプリで、[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) 553 行の `CreateEvalRunner("EvalPanel", ...)` である（[EvalPanel.cpp](../../src/apps/runner/src/EvalPanel.cpp) 147 行の "EvalPanel" は Observer の名前で、Runner の名前ではない）。
- `RunManager` を作る本番の箇所は [RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 445 行の 1 か所。テストでは、検索語 `make_shared<...RunManager>` と `RunManager <名前>(` の形で、trainer_test.cpp 30 件、EvalTracePanel_test.cpp・RunnerApp_test.cpp・RunnerFrame_test.cpp・GridMazeEnv_test.cpp 各 1 件がある。
- 予約名で止まることは、[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) 1118〜1143 行の TEST_CASE（SECTION "EvalPanel"）が確かめる。文言は `Duplicate Env name 'EvalPanel' within Run: existing_owner='EvalPanel', ...` である。
- 設計 100 は、予約名 `EvalPanel` を RunManager が検証すると書く（[100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) 208・219 行、英訳も 208・219 行）。

### 3.12 bat の題（26f763）

- 追跡している `.bat` は 21 本（`git ls-files '*.bat'`）。2 行目が `title %~nx0` でないのは、[apps/10_run.bat](../../apps/10_run.bat)（1 行目 `@echo off`、2 行目 `cd /d "%~dp0runner"`）と [bootstrap.bat](../../bootstrap.bat)（`@echo off` も無く、1 行目が `REM rmdir /s /q build`）の 2 本だけである。どちらも作業ツリーは CRLF である（`git ls-files --eol`）。

## 4. 未決表

すべて暫定の既定である。課題の対応方針をそのまま既定にした行は、理由にその旨を書く。

| 番号 | 論点 | 暫定の既定 | 理由と依存 |
|---|---|---|---|
| U1 | 実装の順と段階 | 段階に分けない。RFD0205（文書）のあとに本 RFD、本 RFD のあとに RFD0207（DQN 系）を実装する前提で書く。本 RFD のコードに手を入れる前に、A15 の比較に使う変更前の実行体（HEAD の Debug の Runner と DLL）を保存する | 開発者が置き場ごとの 3 本と順を選んだ（§9）。段階の分け目は求められていない。比較の基準は実装の前に保存したものに限る（P4、[318](../wiki/concepts/framework/318_rl_core.md) の「変えるときの条件」の RFD0084 の行） |
| U2 | n_step の値域（003c85） | `CreateReplayBuffer` で n_step が 1 以上かを確かめ、外れたら完全キー（`config.full_keys` の `n_step`）・指定値・期待値（`>= 1`）を示して `ANET_SYSTEM_ERROR` で止める。文言は同じ関数の `per_initial_priority` の検査の形にそろえる | 課題の対応方針どおり。DQN は完全キーを既に写している（§3.1）。Replay の構築は Run の構築の中なので、構築時に止まる（P1）。公開 API と抽選の分布は変えず、止める検査を足すだけ（[318](../wiki/concepts/framework/318_rl_core.md) の「変えるときの条件」の RFD0078 の行） |
| U3 | ImageCls の 4 キー（0c9793） | `Validate` に足す。`grad_clip_max_norm` は有限で 0 より大きい、`label_smoothing` は有限で 0 以上 1 以下、`weight_decay` は有限で 0 以上。外れたら完全キー・指定値・期待値を示して止める（今の `Validate` の文言 `Invalid config <key>=<value>. Expected ...` の形）。`learning_rate` は下限 0 を置き、ProfiledValue の構築時の検証に任せる | 課題の対応方針どおり。NaN を外す「有限」は、今の `ValidateNonNegative`・`ValidateUnitInterval` が NaN を通すため、比べる形だけでは足りないので足した。ImageCls の勾配の clip は切り替えを持たず常に当てるので、0 以下に意味を持たせない（P1） |
| U4 | hist_bins（1979f6） | `QValuePanel::Config` で `hist_bins` が 1 以上かを確かめ、外れたら `smooth_radius` と同じ形（`QValuePanel.hist_bins=<値> must be an integer >= 1.`）で止める | 課題の方針は「未定」。1 では描画の加工が壊れない（§3.3）ので、下限は 1。同じ Config の隣の検査の形にそろえる（P1・P2） |
| U5 | 数の属性の読み取り（446e6f） | scalar の `interval`・`ema_alpha`・`clip` と、graph の `interval`・`step_interval`・`episode_interval`・`eps_interval` を、値の全体を読み切る変換で読む。空、読み残しがある、型の範囲を超える、のどれかなら、`config_key`・`config_value`・属性の名前・属性の値・期待（`integer` か `number`）を示して `ANET_SYSTEM_ERROR` で止める（同じ節の `Unknown attribute key. config_key=...` の形）。値域の検査は今のまま（scalar の `interval` の 1 以上だけ） | 課題の対応方針どおり。正しい値の解釈は変わらない（§3.4 の出荷設定の値は、どちらの変換でも同じ値になる）。graph の値域は 29c3b2 の判断が残るので足さない（§1.3） |
| U6 | 行動の device の控え（7671ff） | 控えの鍵に、index を補ったあとの device（`req_dev`）を記録する 1 か所の直し。控えに守りは足さない | 課題の対応方針どおり。P9 の判断に触れない。毎 step の経路だが、比べる値が変わるだけで処理の回数は変わらない |
| U7 | 先読みの待ちの片づけ（ff9a9a） | 2 つの関数とも、先頭の待ちを列から取り出してから結果を受け取る順に変える。投入の順と、1 つ先までの先読みの契約は変えない。worker の Push を失敗させる特性テストを先に足す。同期の前にアプリを閉じ、デストラクタが worker の例外を最初に受け取る場合は、本 RFD では扱わない（起票の候補） | 課題の対応方針どおり。デストラクタの場面は、例外を呼び出し元へ返せず、ログに残して続けるか止めるかの判断が要る（P1 の worker の例外の箇条） |
| U8 | テストの初期化（9a0e78） | 305 行のテストの先頭で、ほかのテストと同じく `EnsureWxImageSupport()` を呼ぶ。テスト用のデータを作る型の中で呼ぶ形はとらない。wx のエラーのダイアログを出さない形は、本 RFD では扱わない（起票の候補） | 課題の方針は「未定」で、案としてこの 2 つを比べるとある。同じファイルのほかの 20 件と同じ形にするのが一番小さい。型の中で呼ぶには、関数の定義の位置を前へ動かすことになる（§3.7） |
| U9 | 既定の値の宣言の検査（4adabb） | コードは直さない（§3.8 のとおり直っている）。リポジトリの根で標準の手順を回し、`apps` と `src` が検査されることを確かめて、課題の対応内容に直したコミットと結果を書く。検査が報告する違反の件数は記録するだけで、本 RFD では直さない | 課題の対応方針の「探索先を今の配置に合わせる」は 125075dc で済んでいる。違反があれば、それは設定の中身の問題で、この課題の事象（検査が止まる）とは別 |
| U10 | ImageCls の構築（27b23f） | `ImageClsAgentConfig` のコンストラクタを引数 1 つにし、上書きの接頭辞の下の Actor の一覧を読む分岐を消す。構築時の記録は、冒頭の INFO（`ImageClsAgent config=`）とメトリクスの設定の記録を 1 回ずつ残し、末尾の 2 行（メトリクスの記録と `ImageClsAgent initialized. config=` の INFO）を消す。ImageCls の Run では INFO の行とマスタの json の行が 1 つずつ減る | 課題の対応方針どおり。「`ImageClsAgent initialized.` を残すか」は、文言を引く箇所がほかに無い（§3.9）ので残さない。DefaultDQN と同じく冒頭で書く形にそろう |
| U11 | 観測の検証の関数（2709db） | `StateSpec::ValidateObservation` を、戻り値を持たない `StateSpec::AssertObservation` に改める（引数は今と同じ。§5.3）。trainer.cpp の 2 か所は、`ANET_ENABLE_ASSERT` が真のビルドでだけ呼ぶ形にし、今の `ANET_ASSERT` と同じく Release では呼ばない。Env のテストの 23 件は「例外を投げない」を確かめる形に替える。旧い名前の別名は残さない | 課題の対応方針どおり。名前は隣の `AssertSanity` にそろえた。呼ぶ条件は P5 のとおり今と同じ。P10 |
| U12 | 予約する名前（4df9cc） | `RunManager` のコンストラクタに、アプリが予約する Env name の並びを渡す引数を足す（既定は空。§5.3）。基盤の "EvalPanel" の文字を消す。予約の検査は今の `validate_planned_name` に、名前を持ち主として渡す（文言 `existing_owner='EvalPanel'` を保つ）。Runner は名前を 1 か所の定数に置き、予約（RunnerApp.cpp 445 行）と `CreateEvalRunner`（RunnerFrame.cpp 553 行）の両方がそれを使う。trainer_test の予約の TEST_CASE は、名前を渡して作る形に替える | 課題の対応方針どおり。既定を空にすると、予約を確かめないテストの 34 件を変えずに済む。空の並びは「予約が無い」という素直な意味で、特別な値ではない。文言と時機を変えない（P3） |
| U13 | bat の題（26f763） | 2 本に AGENTS.md の形を足す。apps/10_run.bat は 2 行目に `title %~nx0` を入れる。bootstrap.bat は先頭に `@echo off` と `title %~nx0` の 2 行を入れる（コマンドの表示は無くなる）。CP932 と CRLF を保つ | 課題の方針は「未定」（足すか、規約の対象から外すか）。規約は全部の `.bat` に求め、外す理由が課題に無い（P7）。規約の形は `@echo off` の次の行なので、bootstrap.bat には `@echo off` も要る |
| U14 | 文書の追随 | 設計 150 の §7.1 の表の `n_step` の行（[jp](../design/150_replay_buffer.jp.md) 276 行、[en](../design/150_replay_buffer.en.md) 277 行）に「1 以上。0 以下は構築で止まる」を足す。設計 100 の 208・219 行（[jp](../design/100_runtime_and_configuration.jp.md)・[en](../design/100_runtime_and_configuration.en.md)）を、予約名 `EvalPanel` を基盤が持つ書き方から、アプリが渡す予約名（Runner は `EvalPanel`）を検証する書き方に直す | 変える契約を書いている現用の文書（§6.4 の検索）。英訳は、和文を直したあとに翻訳の skill（`anet-translate-docs`）でファイルを指定して追随させ、手で部分的に直さない（[RFD0205](0205_design_doc_and_glossary_alignment_10rfd.md) の U11 と同じ。英訳は元の和文の blob を持ち、手で直すと blob と中身がずれる） |
| U15 | 振る舞いが変わらないことの確かめ方 | A15 の 6 構成の比較を、本 RFD の全部の変更を入れた最後に 1 回行う。ImageCls の変更（U3・U10）は 6 構成に現れないので、ImageCls のテストで確かめる | 課題が名指す確かめ方（RFD0122 §4 D8 の 4 環境の比較、RFD0094 の checksum）と同じ目的で、最近の同じ種類の RFD（RFD0192 の A13）が使った形（P4）。課題ごとに Run を分けると、同じ比較を何度も回すことになる |
| U16 | `use_n_step = false` のときの `learner.n_step` の値域 | 本 RFD では扱わない（起票の候補） | AGENTS.md は使われない側の設定キーも値域を検証するとする（P1）が、置き場は DQN 系の設定の検証で、本 RFD の置き場の外である。U2 の検査は、`use_n_step` が真のときに DQN から届く値を止める |

## 5. 仕様

### 5.1 値域の判定

各キーは独立に、下の表の該当する 1 行で判定する（行の条件は重ならない）。型に反する値（整数のキーに `1.5`、数のキーに `abc` など）は、今の設定の読み取り（`ANET_READ_CONFIG`）が型の変換の失敗として止める。数の属性の読み取り（U5）では、§5.2 が止める。

| キー | 型 | 受け付ける範囲 | 端の値の扱い | 外れたとき |
|---|---|---|---|---|
| Replay の `n_step`（DQN では `learner.n_step` の完全キー。`use_n_step` が真のとき） | 整数 | 1 以上 | 1 は受け付け、0 と -1 は止める | `ANET_SYSTEM_ERROR`（U2） |
| `ImageClsAgent.grad_clip_max_norm` | 小数 | 有限で 0 より大きい | 0・負・NaN・無限は止める | `ANET_SYSTEM_ERROR`（U3） |
| `ImageClsAgent.label_smoothing` | 小数 | 有限で 0 以上 1 以下 | 0 と 1 は受け付け、-0.1・1.1・NaN は止める | `ANET_SYSTEM_ERROR`（U3） |
| `ImageClsAgent.weight_decay` | 小数 | 有限で 0 以上 | 0 は受け付け、負・NaN・無限は止める | `ANET_SYSTEM_ERROR`（U3） |
| `ImageClsAgent.learning_rate`（予定の各区間の値・始点・終点） | 小数 | 0 以上 | 0 は受け付け、負は止める | ProfiledValue の構築時の検証（U3） |
| `QValuePanel.hist_bins` | 整数 | 1 以上 | 1 は受け付け、0 と -1 は止める | `ANET_SYSTEM_ERROR`（U4） |

`use_n_step` が偽のとき、DQN は Replay へ 1 を渡すので、表の 1 行目には当たらない（U16）。

### 5.2 数の属性の読み取り

対象は U5 の 7 つの属性名である。値の文字列について、上から順に当て、最初に当たった行で扱う。

| 条件 | 扱い |
|---|---|
| 空 | 止める |
| 整数の属性で、先頭から末尾までが 1 つの整数として読め、`int` の範囲に入る | その値 |
| 小数の属性で、先頭から末尾までが 1 つの数として読め、`float` の範囲に入る | その値 |
| 上のどれにも当たらない（読み残しがある、数で始まらない、範囲を超える） | 止める |

止めるときの文言は U5 のとおり。読めた値に対する値域の検査（scalar の `interval` の 1 以上）は今の箇所で今のまま行う。

### 5.3 公開ヘッダの宣言の草案（暫定）

```cpp
// src/core/base/include/anet/rl.hpp（StateSpec の中。U11）
/// 観測が spec に合うかを確かめる。合わなければ ANET_SYSTEM_ERROR で止める。
void AssertObservation(const anet::TensorDict& obs, bool is_batched = true) const;

// src/core/base/include/anet/trainer.hpp（RunManager の中。U12）
/// reserved_env_names: アプリがあとで CreateEvalRunner で作る Runner の Env name。
/// train・configured Eval tag と同じく、最初の BatchEnv を作る前に重なりを確かめる。
RunManager(const ConfigData& config_data, std::vector<std::string> reserved_env_names = {});
```

`AssertObservation` は、同じ型に並ぶ `AssertSanity`（spec 自身の整合を確かめる）と対になり、こちらは渡された観測を確かめる。`reserved_env_names` の `env_names` は [CONTEXT.md](../../CONTEXT.md) の「Env name」（320 行）の意味で使う。

## 6. 実装契約

### 6.1 変えるもの

| 課題 | ファイル | 変えること | 未決表 |
|---|---|---|---|
| 003c85 | replay_buffer_impl.cpp（`CreateReplayBuffer`） | n_step の検査を足す | U2 |
| 0c9793 | image_cls_agent.hpp（`ImageClsAgentConfig`） | 3 キーの検証と、`learning_rate` の下限 | U3 |
| 1979f6 | QValuePanel.cpp（`QValuePanel::Config`） | `hist_bins` の検証 | U4 |
| 446e6f | observers.cpp（`ObserverFactory` の scalar と graph の節） | 数の属性の読み取り | U5 |
| 7671ff | rl.cpp（`BatchActionInfo::GetAction`） | 控えの鍵 | U6 |
| ff9a9a | replay_buffer_impl.cpp（`PrefetchingReplayBuffer::State` の 2 関数） | 待ちを外してから結果を受け取る | U7 |
| 9a0e78 | ImageClsEnv_test.cpp | 305 行のテストの初期化 | U8 |
| 4adabb | （コードは変えない）課題の対応内容 | 確かめた結果の記録 | U9 |
| 27b23f | image_cls_agent.hpp・image_cls_agent.cpp | コンストラクタの引数、記録の重複 | U10 |
| 2709db | rl.hpp・rl.cpp・trainer.cpp、Env のテスト 3 ファイル | 名前と戻り値、呼ぶ条件、テストの形 | U11 |
| 4df9cc | trainer.hpp・trainer.cpp、RunnerApp.cpp・RunnerFrame.cpp（と名前の定数を置く Runner のファイル）、trainer_test.cpp | 予約する名前を外から渡す | U12 |
| 26f763 | apps/10_run.bat・bootstrap.bat | 題の行 | U13 |
| （文書） | 設計 150・100 の jp・en | U14 の行 | U14 |

足すテストは、どれもタグ `[rfd0206]` を付ける（A1〜A12 で引く）。置き場は実装計画で決める。

### 6.2 変えないもの（本当の制約か）

| 変えないもの | 理由 | 本当の制約か |
|---|---|---|
| 正しい設定の Run の metrics・定義レコード・WARN・fail-fast の文言と時機（27b23f の ImageCls の構築の記録を除く） | P3。G2 | 制約（この RFD のゴール） |
| `BatchActionInfo` の控えに守りを足さないこと | P9 | 制約ではない。RFD0192 の判断を変えるなら別の RFD |
| 先読みの投入の順と、1 つ先までの先読み | 課題の対応方針。抽選の順と乱数に触れないため | 制約（G2） |
| graph の宣言の値域と未知の語の扱い | §1.3 | 制約ではない。29c3b2 |
| `NetworkGraphVizConfig` などの、ImageCls の `nn_viz.*` の読み取り | 27b23f の事象の外 | 制約ではない |
| EvalPanel.cpp 147 行の Observer の名前 "EvalPanel" | Runner の名前ではない（§3.11） | 制約ではない。別の関心 |

### 6.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

| 名前 | 区分 | 責務（1 行） | 置き場（機能グループ） |
|---|---|---|---|
| `StateSpec` | 変更 | 観測の確かめを `AssertObservation`（戻り値なし）にする | rl（`anet/rl.hpp`・`rl.cpp`） |
| `RunManager` | 変更 | アプリが予約する Env name を構築の入力で受ける | trainer（`anet/trainer.hpp`・`trainer.cpp`） |
| `ImageClsAgentConfig` | 変更 | 引数 1 つで作り、3 キーの値域と学習率の下限を確かめる | imagecls の Agent（`image_cls_agent.hpp`） |
| `ImageClsAgent` | 変更 | 構築時の設定の記録を 1 回にする | imagecls の Agent（`image_cls_agent.hpp`・`.cpp`） |
| `QValuePanel::Config` | 変更 | `hist_bins` を確かめる | Runner の QValuePanel（`QValuePanel.cpp`） |
| `ObserverFactory` | 変更 | メトリクスの宣言の数の属性を全体で読む | observers（`observers.cpp`） |
| `BatchActionInfo` | 変更 | 控えの鍵を index を補った device にする | rl（`rl.cpp`） |
| `PrefetchingReplayBuffer` | 変更 | Push の待ちを外してから結果を受け取る | replay（`replay_buffer_impl.cpp`） |

新しい機能グループは立てない。

### 6.4 移行対象

検索は、範囲をリポジトリ全体（`git grep`、大文字小文字を区別しない）にした。

| 検索語 | 当たった数 | 移すもの |
|---|---|---|
| `ValidateObservation`（後ろに `NormalizerConfig`・`s` が続くものを除く） | コードは宣言 1・定義 1・trainer.cpp 2・Env のテスト 23（`ValidateObservation(` の形）。文書は `docs/memo/done` の RFD0032・RFD0040 の RFD と実装記録の 3 件と、課題記録（2709db と課題の一覧）。設計文書と `CONTEXT.md` には無い | コードの 27 か所を U11 のとおり直す。過去の RFD・実装記録と課題記録は過去の記録なので直さない |
| `"EvalPanel"` | trainer.cpp 959 行（予約）、RunnerFrame.cpp 553 行（作る）、EvalPanel.cpp 147 行（Observer の名前）、テスト 13 件 | 予約と作る 2 か所を U12 のとおり。テストは、予約を確かめる trainer_test の TEST_CASE だけ名前を渡す形に替える。ほかのテストの "EvalPanel" は Runner の名前の文字としてそのまま残る |
| `予約名`・`reserved name`・`EvalPanelのname`（設計文書） | 設計 100 の jp・en の 219 行と、図の 208 行 | U14 |
| `n_step`（設計 150 の §7.1 の表） | jp 276 行・en 277 行 | U14 |
| `hist_bins` | QValuePanel.cpp・QValuePanel.hpp・common.txt 67 行・QValuePanel_test.cpp 25 行（5 を渡す）・RFD0072 の検証資材の基準の json（設定の写し） | コードの Config だけを直す。テストの 5 と出荷設定の 510 は範囲に入るので変えない。基準の json は設定の写しで、値は変わらない |
| `ImageClsAgent initialized` | image_cls_agent.cpp 638 行だけ（課題記録を除く） | U10 で消す |
| `ImageClsAgentConfig(` | 本番 1・テスト 17（どれも引数 1 つ） | 呼び出しは変えない |
| `check_default_leaves` | 設計 020 の jp・en（今のパスで書いてある）、動線マップ、設計 100、過去の RFD と課題記録 | 直すものは無い（U9） |
| `title %~nx0` | AGENTS.md 214 行と bat 19 本 | 規約は変えない。2 本を足す（U13） |

設定キーの名前とメトリクスの名前は変えず、正しい設定のときのログの文言も変えない（27b23f の消す 1 行を除く。不正な値のときの文言は U2〜U5 で足すか改める）ので、設定・サンプルの移行は無い。新しく止める値（§5.1・§5.2）を使う出荷設定とテストは無い（§3.1〜§3.4 の検索）。

### 6.5 ほかの RFD との実装の順序

- 3 本の順は U1 のとおり。本 RFD と RFD0207 は同じファイルを触らない見込みである（本 RFD は `src/algo/dqn` を変えない。003c85 の完全キーは dqn_learner.cpp が既に写している）。RFD0205 は文書だけで、本 RFD が直す設計 150・100 を触らない見込みである。重なったら、あとから入る側が前の変更の上で直す。
- まだ実装していない RFD のうち、同じファイルを触るものを `docs/memo` の RFD の本文で引いた（検索語は `ObserverFactory`・`PrefetchingReplayBuffer`・`CreateReplayBuffer`・`QValuePanel`・`ValidateObservation`・`BatchActionInfo::GetAction`・`check_default_leaves`、実装済みかは各 RFD を指す課題の状態で見た）。ObserverFactory は RFD0081・RFD0169・RFD0184、先読みの Replay は RFD0171・RFD0201、QValuePanel は RFD0081・RFD0113・RFD0146・RFD0155・RFD0159・RFD0160・RFD0176・RFD0183、既定の値の宣言の検査は RFD0118・RFD0155・RFD0181 が触る。本 RFD の変更はどれも数行の局所の直しで、それらの RFD の契約に触れないので、どちらが先でもよい。あとから入る側は、本 RFD の直し（数の属性の読み取り、待ちの片づけの順、`hist_bins` の検証）を残したまま書き換える。段階を持つ RFD の途中で本 RFD が入っても、取り直す基準は無い。

### 6.6 置き場を変えるときの扱い

追跡しているファイルやフォルダを移す・改名する・分ける・消すことはしない（関数の名前を 1 つ変えるだけ）。[RFD レビュー観点](../quality/rfd.md) の「契約の変更」の表の行には当たらない。

## 7. 受入

テストの実行体は Debug（`out/test/Debug/` と `apps/runner/bin/Debug/`）を使う。「既知の失敗」は、本 RFD のコードに手を入れる前に同じ実行体を回して記録した失敗を指す。

| 番号 | 確かめること | 確かめ方 |
|---|---|---|
| A1 | ビルド | AGENTS.md の「AI エージェントでのビルド注意事項」の形で `cmake --build --preset x64-Debug` と、`--target AnetRLRunner-gui-test`（P8）が終了値 0 |
| A2 | n_step（003c85） | 足したテスト（`anet-core-base-test.exe "[rfd0206]"`）で、完全キーを持たせた `ReplayBufferConfig` の n_step が 0 と -1 のとき `CreateReplayBuffer` が止まり、文言に完全キーと指定値と `>= 1` が入る。1 では作れる |
| A3 | ImageCls の値域（0c9793） | 足したテスト（`anet-algo-imagecls-test.exe "[rfd0206]"`）で、§5.1 の ImageCls の 4 行の「止める」値ごとに、Agent の構築が終わる前に止まり、文言に `ImageClsAgent.` から始まる完全キーが入る。「受け付ける」端の値（label_smoothing 0 と 1、weight_decay 0、学習率の予定の始点 0）では止まらない |
| A4 | hist_bins（1979f6） | 足したテスト（`AnetRLRunner-gui-test.exe "[rfd0206]"`）で、`QValuePanel.hist_bins` が 0 と -1 のとき QValuePanel の構築が止まり、文言に `QValuePanel.hist_bins` と指定値が入る。1 と 510 では作れる |
| A5 | 数の属性（446e6f） | 足したテスト（`anet-core-base-test.exe "[rfd0206]"`）で、scalar の宣言の `interval:1O0`・`interval:2.5`・`ema_alpha:0.1x`・`clip:abc`・`interval:99999999999` と、graph の宣言の `interval:1O0`・`episode_interval:x` のそれぞれで `ObserverFactory` の構築が止まり、文言に宣言の設定キー（`metrics.scalar.[...]` か `metrics.graph.[...]`）と属性の値が入る。`interval:100`・`ema_alpha:0.01`・`clip:1` の宣言は作れる |
| A6 | 行動の device（7671ff） | 足したテスト（`anet-core-base-test.exe "[rfd0206]"`）で、CPU の行動を持つ `BatchActionInfo` に index の無い cuda で 2 回 `GetAction` を呼び、2 回とも止まらず同じ値が返る。続けて `cuda:0` で呼んでも止まらない。CUDA が使えない機械では SKIP し、SKIP したことを実装記録に書く |
| A7 | 先読みの待ち（ff9a9a） | 足した特性テスト（`anet-core-base-test.exe "[rfd0206]"`）で、worker の Push が例外になる先読みの Replay について、最初の同期が元の例外を投げ、2 回目の同期が例外を投げず、Replay を壊してもテストのプロセスが続く。直す前のコードでこのテストが落ちたことを実装記録に書く |
| A8 | テストの初期化（9a0e78） | `anet-env-imagecls-test.exe` を、(1) 305 行のテストの名前だけ、(2) `--order decl`、(3) `--order rand --rng-seed 1094244747` の 3 通りで回し、3 通りとも終了値 0 で、それぞれ 10 分以内に終わる |
| A9 | 既定の値の宣言の検査（4adabb） | リポジトリの根で `.\.venv\Scripts\python.exe -B src/core/base/testdata/rfd0072/check_default_leaves.py` を回し、`Assignments: ` で始まる行が出て、終了値が 0 か 1 である（`RuntimeError` で止まらない）。出力の `.scratch/rfd0072-default-leaf/default-leaf-audit.json` の `file` に、`apps/` で始まるものと `src/` で始まるものの両方がある。`check_default_leaves_test.py` が通る。出た違反の件数を課題の対応内容に書く |
| A10 | ImageCls の構築（27b23f） | `src/algo/imagecls/src/image_cls_agent.hpp` で `ImageClsAgentConfig(` のコンストラクタの宣言の引数が 1 つで、`MakeOverrideConfigKey("actor")` が 0 件。`src/algo/imagecls/src/image_cls_agent.cpp` で `MetricsLogger::Instance()->Log(config` が 1 件、`ImageClsAgent initialized` が 0 件。ImageCls のテスト（`anet-algo-imagecls-test.exe`）が期待値を変えずに既知の失敗を除いて通る |
| A11 | 観測の検証の関数（2709db） | 範囲 `src` で `ValidateObservation(` が 0 件（`ValidateObservationNormalizerConfig(`・`ValidateObservations(` はこの文字列を含まないので当たらない）。`AssertObservation(` が rl.hpp 1・rl.cpp 1・trainer.cpp 2・Env のテスト 23。`anet-env-atari-test.exe`・`anet-env-dropmerge-test.exe`・`anet-env-lunarlander-test.exe` が既知の失敗を除いて通る |
| A12 | 予約する名前（4df9cc） | `src/core/base` で `"EvalPanel"` が `_test.cpp` の外に 0 件。trainer_test の予約の TEST_CASE が、名前を渡して作り、文言 `Duplicate Env name 'EvalPanel' within Run: existing_owner='EvalPanel'` を確かめて通る。`AnetRLRunner-gui-test.exe` の RunnerApp・RunnerFrame・EvalTracePanel のテストが既知の失敗を除いて通る |
| A13 | bat の題（26f763） | `git ls-files '*.bat'` の 21 本すべてで、1 行目が `@echo off`、2 行目が `title %~nx0`。`git ls-files --eol apps/10_run.bat bootstrap.bat` が `w/crlf` |
| A14 | 文書（U14） | 設計 150 の jp・en の §7.1 の表の `n_step` の行に下限 1 がある。設計 100 の jp・en の 208・219 行に、基盤の固定の予約名としての `EvalPanel` の書き方が無く、アプリが渡す予約名の書き方がある。設計 150・100 の英訳の 1 行目の `translated-from` の blob が、直したあとの和文の `git hash-object` と一致し、`progress:done` である |
| A15 | 正しい設定の Run の結果（G2・U15） | RFD0103 の検証資材（[run_validation.py](../../src/core/base/testdata/rfd0103/run_validation.py)）と同じ 6 構成・同じ短縮条件（P4）で、U1 で保存した実装の前の実行体と、本 RFD の全部を入れた実行体を走らせる。`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar の tag・step・値が、6 構成とも一致する。除くのは、走らせるたびに変わる実時間と資源の値だからである。値は、検証資材が正規化した記録の表記で比べる（NaN は同じ表記どうしで一致とする）。比較の入出力は `.scratch/rfd0206/` に置く |
| A16 | 既存のテスト | `anet-core-base-test`・`anet-algo-dqn-test`・`anet-algo-imagecls-test`・`anet-algo-muzero-test`・`anet-env-*-test`（6 本）・`AnetRLRunner-test`・`AnetRLRunner-gui-test` が、既知の失敗のほかに失敗を出さない |

## 8. issue の対応表

| issue | 題（要点） | area | 根拠の位置 | 直す | 受入 |
|---|---|---|---|---|---|
| [issue-20261123-003c85](../management/issues/issue-20261123-003c85.md) | Replay が n_step 0 以下で作れ、学習が黙って始まらない | 318_rl_core・324_dqn | [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 2523〜2557 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 987・995 行 | U2・U14・U16 | A2・A14・A15・A16 |
| [issue-20261123-0c9793](../management/issues/issue-20261123-0c9793.md) | ImageCls の grad_clip・label_smoothing・learning_rate・weight_decay の値域を確かめない | 324_dqn | [image_cls_agent.hpp](../../src/algo/imagecls/src/image_cls_agent.hpp) 61〜66・106〜117 行、[image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) 395・408・412 行 | U3 | A3・A16 |
| [issue-20261120-1979f6](../management/issues/issue-20261120-1979f6.md) | QValuePanel の hist_bins=0 が検証を通る | 110_runner_app | [QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) 39・471・543〜571 行 | U4 | A4・A16 |
| [issue-20261123-446e6f](../management/issues/issue-20261123-446e6f.md) | メトリクスの宣言の数の属性を先頭の数だけ読む | 318_rl_core・317_metrics | [observers.cpp](../../src/core/base/src/observers.cpp) 1435〜1437・1666・1668 行 | U5 | A5・A15・A16 |
| [issue-20261123-7671ff](../management/issues/issue-20261123-7671ff.md) | GetAction(device) が index の無い cuda を控えの鍵にする | 318_rl_core・320_env_core | [rl.cpp](../../src/core/base/src/rl.cpp) 552〜565 行 | U6 | A6・A15・A16 |
| [issue-20261123-ff9a9a](../management/issues/issue-20261123-ff9a9a.md) | 先読みの Push が例外になると、受け取り済みの待ちが残る | 318_rl_core | [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 2336〜2382 行 | U7 | A7・A15・A16 |
| [issue-20261123-9a0e78](../management/issues/issue-20261123-9a0e78.md) | imagecls の Env のテストの 1 件が画像の口を初期化しない | 322_env_individual | [ImageClsEnv_test.cpp](../../src/env/imagecls/src/ImageClsEnv_test.cpp) 126・193・305 行 | U8 | A8・A16 |
| [issue-20261122-4adabb](../management/issues/issue-20261122-4adabb.md) | RFD0072 の既定の値の宣言の検査が、移転前の core を探して止まる | 311_dev_tools・316_config | [check_default_leaves.py](../../src/core/base/testdata/rfd0072/check_default_leaves.py) 11・27 行 | U9 | A9 |
| [issue-20261123-27b23f](../management/issues/issue-20261123-27b23f.md) | ImageCls の Agent に渡されない上書きの接頭辞の引数があり、設定を 2 回ずつ記録する | 324_dqn | [image_cls_agent.hpp](../../src/algo/imagecls/src/image_cls_agent.hpp) 58・80〜84 行、[image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) 585・586・637・638 行 | U10 | A10・A16 |
| [issue-20261123-2709db](../management/issues/issue-20261123-2709db.md) | ValidateObservation は bool を返すが false を返さない | 318_rl_core | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 229 行、[trainer.cpp](../../src/core/base/src/trainer.cpp) 539・680 行 | U11 | A11・A15・A16 |
| [issue-20261123-4df9cc](../management/issues/issue-20261123-4df9cc.md) | 基盤の RunManager がアプリの Runner の名前 "EvalPanel" を予約する | 318_rl_core・110_runner_app | [trainer.cpp](../../src/core/base/src/trainer.cpp) 959 行、[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) 553 行、[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 445 行 | U12・U14 | A12・A14・A15・A16 |
| [issue-20261123-26f763](../management/issues/issue-20261123-26f763.md) | 2 本の bat に題の行が無い | なし | [apps/10_run.bat](../../apps/10_run.bat) 1〜2 行、[bootstrap.bat](../../bootstrap.bat) 1 行 | U13 | A13 |

## 9. グリルの材料

起票の会話（2026-11-27）の開発者の発言。決定としては扱わない。

1. 「仕様調整がほぼ要らない小さいのをいっぱい集めたPRDを作りたい」（Codex に夜のうちに実装させたい、という前置きのあと）
2. 「34件全部で一つのPRD？A~Dで4つのPRD？寝てる間に回したくて。バラPRDでも順番に実装するようにCodexさんに言えば大丈夫かな」
3. 「3本で」（AI が、直しの種類ごとの 4 本ではなく、触るファイルが重ならない置き場ごとの 3 本（文書・基盤とアプリ・DQN 系）を勧めたのに対して）

AI が会話で示した既定（課題の方針が「未定」だった 3 件）: 1979f6 は hist_bins を 1 以上と確かめて完全キーで止める、9a0e78 はテストの先頭で `EnsureWxImageSupport()` を呼ぶ、26f763 は 2 本の bat に `title %~nx0` を足す。開発者はこの 3 件を含めて起票に進めた。

## 10. 関連

- 設計 [100](../design/100_runtime_and_configuration.jp.md)・[150](../design/150_replay_buffer.jp.md)・[160](../design/160_applications_and_tools.jp.md)
- [RFD0072](done/0072_config_selection_final_value_10rfd.md)（既定の値の宣言の検査）、[RFD0094](done/0094_evalpanel_step_history_10rfd.md)、[RFD0103](done/0103_temporl_reference_alignment_10rfd.md)、[RFD0122](0122_small_refactor_and_test_gap_fixes_10rfd.md)、[RFD0161](0161_observation_pipeline_10rfd.md)、[RFD0192](0192_concurrency_and_boundary_defect_fixes_10rfd.md)、[RFD0194](0194_config_key_full_names_in_messages_10rfd.md)
- トピック別まとめ [318](../wiki/concepts/framework/318_rl_core.md)・[317](../wiki/concepts/framework/317_metrics.md)・[324](../wiki/concepts/framework/324_dqn.md)・[322](../wiki/concepts/framework/322_env_individual.md)・[311](../wiki/concepts/framework/311_dev_tools.md)・[316](../wiki/concepts/framework/316_config.md)・[110 Runner](../wiki/concepts/apps/110_runner_app.md)
