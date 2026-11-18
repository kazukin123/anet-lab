# ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）PRD

> 2026-11-19 改訂: レビューの指摘を受けて直した。受入の検索（§6）を、範囲・除くもの・新しい名前との重なりの扱いが決まる形にし、S1-7 を足した。過去の記録のリンクの扱いを決め直した（D23・C4。ステージ 2 で無くなる 3 本を指すリンクは直さない。課題記録・変更の記録・`reports/` の下の記録を過去の記録に入れた。英訳と生成物は、リンクだけを直す）。`.gitattributes` と、生成物の作り直しを起動する側を §5.2・§5.7 に足した。退けた案を §4.3 に 3 つ足し、非ゴールの最後の行と D7 に理由を書き、§5.6 に用語を足した理由を書いた。§2.4 の `bin/` の区分と、§5.4 の Runner の依存の行を直した。検証資材の文書（`testdata` の `.md`）を過去の記録として扱うことにし（D23・§4.3・§5.7・§6）、生成物の確かめを共通の受入 C5 へ移した。§5.2 に、生成物の作り直しと英訳の追随を人に頼む時機を、§5.3 に、テストで分けるファイルを書いた。検証資材のうち、過去の Run の記録の写しは中身を変えないと決めた（D23・§4.3・§5.7・§6）。レビューのあと、起票のために数え直して、テストの中の代役の Agent の数を 9 から 8 に直した（§2.3・D14・§7。Agent を作らない代役の factory 1 つを数えていた）。
> 2026-11-18 改訂: グリルを行い、暫定の印を外した。起票時の未決表（D1〜D23）と §4 の暫定の既定をすべて仕分け、確定事項（§4 の D1〜D27）、現状（§2）、実装契約（§5）、受入（§6）へ振り分けた。振り分けは §4.2 に書いた。ゴールに DQN 系の中のファイルの分け方を足し、[PRD169](169_outside_core_extension_layout_10prd.md)（公開ヘッダの規約）を本書へ統合した。2026-11-17 に足した §12（再グリルの材料）の発言・事実・案は、§1・§2・§4 へ移して §12 を消した。題も内容に合わせて改めた。ADR [0078](../adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md)・[0079](../adr/0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public.md)・[0080](../adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md) を起こした。
> 2026-11-18 改訂: 起草のあと、別モデル（Opus）の検討の論点を反映した。事実を §2.2〜§2.4 に足し、D6・D7・D8・D12・D18・D19・D20・D21 を補い、§5.2〜§5.6・§5.8 と受入（C3・S1-1・S1-3・S2-1・S3-2・S3-5・S4-1）を直した。§5.3 の型の割り振りのうち 2 つを、依存の向きに合わせて改めた。data を移す手順（人が改名で行う）と、data が無いときの扱いを §5.5 に足し、退けた案を §4.3 に 2 つ足した。
> 2026-11-17 改訂: 再グリルの材料として、2026-11-17 のユーザーの発言、同じ日に確かめた事実、発言と事実を受けて AI が整理した未合意の案、起票時の本文と合わなくなる点を §12 に足した。決めることを §8 の D17〜D23 に足し、D1・D2・D3・D4・D6・D8・D10・D11・D16 の代案の欄に §12 を指す注記を足した。§2 と §10 に §12.2 を指す行を、§11 に参照を足した。質疑はしておらず、暫定のままである。§2〜§9 の本文と §8 の暫定の既定は変えていない。
> 2026-11-17 番号を 957 から 182 へ振り直した（PRD131）。
> 2026-11-03 番号を 999 から 957 へ振り直した（PRD114）。

> 2026-10-24 起票。暫定の既定を置き、決めることを §8 の表に残した。グリル前で、最終簡素化パスは経ていない。行番号・行数・参照数は起票時点のもので、実装時は近傍のシンボル名・ファイル名で探し直す。
> 起点: 2026-10-23 の会話。具象 Agent 4 つと系統ごとの共通基盤が `core/anet-core/src` に Env 以外のすべてと同居している。Env は `core/envs/<env>/` の独立ライブラリで、`Init<Env>()` を app が呼ぶ。Agent も同じ形にして責務の境界を CMake で強制したい（分散の不利より分ける利得が大きい）。wxWidgets の配置を参考に見て、ソースの root が `core/` である今の形が不自然だと気づいた。
> 起点（2026-11-17）: 直接のきっかけは、DQN のアルゴリズムの実装コードが、育つにつれて徐々に汚くなってきたこと。core の部分と DQN でソースを分ければ、ソースファイルの配置の柔軟性が上がり、結果としてクラスの責務を適正にできるのではないか。ソースが分散して一覧性が下がるので、過度に分割せず、適正な頃合いを目指したい。流れは `src` →（生成）→ `apps`［実行ファイル類］→（利用）→ `data`。
> 関連: [PRD169](169_outside_core_extension_layout_10prd.md)（公開ヘッダと登録口の規約。本書へ統合した）、[PRD181](181_config_file_layout_10prd.md)（設定ファイルの配置。別の PRD のまま）、[PRD185](185_release_packaging_cmake_10prd.md)（リリースの組み立てを CMake へ移す。順序は §9）、[PRD125](125_dqn_learner_update_skeleton_10prd.md)・[PRD128](128_dqn_agent_refactor_bundle_10prd.md)（`dqn_based_agent.cpp` をファイルに分けることを本書の範囲としている）、[080 動線マップ](done/080_harness_map_10prd.md)・[095 LLM-Wiki](done/095_llm_wiki_10prd.md)（パスを持つ生成物。作り直しが要る）、[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)（リリースの zip は配置の写し）。

## 1. 目的と問題

ソースの置き場を 1 つの規則で説明できる形にし、Agent を基盤の外のライブラリへ出して、DQN 系の実装に置き場を与える。あわせて、ライブラリとテストが読んでよい範囲を決め、依存の向きを機械が守る形にする。

今は次の問題がある。

1. DQN 系の実装が、基盤の 1 フォルダと 1 ファイルに同居したまま育っている。`dqn_based_agent.cpp` は 6,350 行で、24 日で 28% 増えた（§2.1）。置き場が無いので、責務を分けにくい。
2. 基盤と Agent の境界を機械が守っていない。基盤が具象 Agent を登録し、Agent は基盤の内部ヘッダを読む（§2.2）。公開ヘッダに何を置くかの規約も無い。
3. テストが境界をまたいでいる。基盤と Env のテストが具象 Agent を使い、系統のテストが基盤の内部の型で Network を手で組む（§2.3）。
4. トップの階層から、ビルドの入力・実行時の配置・実行時にできるものの別が読めない。`apps/runner` に、ソース、設定、実行体、Run フォルダが同居する（§2.4）。

## 2. 現状（2026-11-17〜18 の調査）

行数は、追跡しているファイルの作業ツリーの内容を数えた。行番号は調査時点のもので、実装時は名前で探し直す。

### 2.1 DQN 系の実装の大きさ

| 事実 | 確かめ方 |
|---|---|
| [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) は 6,350 行、[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) は 1,425 行、`dqn_based_agent_test.cpp` は 12,969 行。2026-10-24 は 4,966・1,404・9,567 行で、24 日で実装が 28% 増えた | 行数を数えた |
| 実装の持ち主ごとの行数は、`Learner`（メソッド 57 個）2,385、共通の計算の関数 18 個 822、方策（`UQEActionPolicy`・`EpsilonGreedyActionPolicy`・`SpatialLadderSchedule`）574、`Actor` 378、`DQNActionInfo` 376、`BatchUpdateResult` 371、`NetworkModel` 289、分位の Learner（`QuantileLearnerBase`・`IQNLearner`）378、step 診断と初期優先度の推定 263、`DqnComponentAssembler` 179 | 行頭から始まる定義を、持ち主（`Class::`）ごとに次の定義までの行数で数えた |
| ヘッダには、名前空間の直下に 57 の型がある。うち 13 は診断の指標の型である（290 行） | 字下げが 4 以下の `class`・`struct`・`enum` の宣言を数えた |
| `dqn_based_agent_test.cpp` には 265 ケースがある | 行頭の `TEST_CASE` を数えた |
| 系統ごとの量（テストを除く／テスト）は、DQN 系が 8 ファイル 11,261 行／5 ファイル 15,624 行、MuZero 系が 6 ファイル 2,206 行／1 ファイル 128 行、ImageCls が 2 ファイル 1,155 行／1 ファイル 1,797 行、内的報酬が 4 ファイル 1,130 行／1 ファイル 3,088 行 | `core/anet-core/src` と `core/anet-core/include` の追跡ファイルのうち、名前に `dqn`・`rainbow`・`intrinsic`・`muzero`・`image_cls`・`prd101` を含むもの（28 ファイル 36,389 行）を数えた |
| 基盤の CMake は `src/` の下を再帰で拾う。フォルダを切ってファイルを分けることは、今の CMake のままできる。ライブラリに分けて得られるのは、置き方の自由ではなく、依存の向きを機械が止めることである | [core/anet-core/CMakeLists.txt:16〜18](../../core/anet-core/CMakeLists.txt) |

### 2.2 基盤と Agent の境界

| 事実 | 確かめ方 |
|---|---|
| 系統の実装が読む基盤の内部ヘッダは 4 本である。`nn_heads.hpp`（6 ファイル）、`nn_impl.hpp`（1 ファイル）、`replay_buffer_impl.hpp`（1 ファイル）、`scaler_impl.hpp`（1 ファイル。内的報酬だけ） | 系統のファイル 23 本の include を、基盤の内部ヘッダ 9 本と突き合わせた |
| DQN の Head（[dqn_based_heads.hpp](../../core/anet-core/src/dqn_based_heads.hpp)・`dqn_based_heads.cpp`、Head 8 種）は、`nn_impl.hpp` の型を使う。系統の実装が使う行数は、`NetworkForwardContext` 35、`NetworkHead` 27、`HeadGraphVizInfo` 20、`NetworkHeadFactory` 17、`WeightInitializer` 13、`LinearProjectionModule` 12、`NetworkModuleExecutor` 8、`NetworkHeadExecutor` 4、`NetworkModule` 1、`MakePreparedHeadExecutor` 1 である。Body を組む型（`NetworkBlock`・`NetworkStruct`・`NetworkBranch`・`NetworkBody`）、module の登録、SN と SDPA の補助は、系統の実装からは 0 行である | 型の名前ごとに、系統の実装 12 ファイルを単語で検索した |
| 上の型を公開ヘッダへ出すには、定義を一緒に出す必要があるものが 3 つある。`WeightInitializer` が呼ぶ `GetNonlinearityType`（[nn_impl.hpp:21](../../core/anet-core/src/nn_impl.hpp)）と、`MakePreparedHeadExecutor` が使う `PreparedHeadExecutor`・`PreparedModuleChildren` である。`LinearProjectionModule` は `NetworkModule` を継承するので、`NetworkModule` も公開ヘッダに見える。定数は、`kKey_DefaultOutput` を 3 つの系統が使い、`kKey_Taus` は DQN だけが使う（基盤の製品コードは 0 行） | 別モデルの検討が挙げ、`nn_impl.hpp`・`nn_heads.hpp` の宣言と、使う側を検索して確かめた |
| 汎用の Head（`nn_heads.hpp` の `LinearHeadFactory`・`PassThroughHead`）は、Head の実装でないファイル 4 本が計 6 行で使う（`dqn_based_agent.cpp` 1、`intrinsic_rnd.cpp` 3、`image_cls_agent.cpp` 1、`muzero_based_agent.cpp` 1） | 同上 |
| 2026-10-24 の本文が内部ヘッダのものとした Replay の生成の 4 つ（`CreateReplayBuffer`・`ReplayBufferConfig`・`PrefetchingReplayBuffer`・`CreateDecisionWindowRule`）は、公開ヘッダ `replay_buffer.hpp` にある。DQN が `replay_buffer_impl.hpp` だけにある名前を使うのは、少なくとも 9 つ（`ReplayExperience`・`ExperienceQueue`・`CreateNStepQueueController`・`CreateDefaultExperienceBuilder`・`QueueRecord`・`ExperienceQueueController`・`ReplayExperienceBuilder`・`ReplayInitialPriorityHintRow`・`ExperienceSequence`）で、step 診断の部品（[dqn_based_agent.cpp:633〜804](../../core/anet-core/src/dqn_based_agent.cpp)）に集まる | 内部ヘッダで宣言された class・struct の名前と `Create` で始まる関数の名前だけを調べた。あとの 2 つは、別モデルの検討が挙げ、step 診断の部品の宣言で確かめた |
| 基盤のうち、系統のファイルでないものが系統のヘッダを読むのは、`init.cpp`（登録の 4 行。[init.cpp:175〜178](../../core/anet-core/src/init.cpp)）と、基盤のテストである。`apps/` と `core/envs/` からは 0 件 | `core` と `apps` の追跡ファイルで、系統の公開ヘッダと内部ヘッダの include を検索した（14 行） |
| 基盤の公開ヘッダ [agent.hpp:66〜432](../../core/anet-core/include/anet/agent.hpp) は、名前空間 `anet::rl::dqn` の設定の型である（`NetworkModelConfig`・`ActionPolicyConfig`・`DQNActorConfig`・`SkipConfig`・`MunchausenConfig`・`LearnerConfig` など、約 370 行）。基盤の製品コード、Runner、Env が `dqn::` を使うのは、`init.cpp` の登録の 2 行だけである | 別モデルの検討が挙げた。`core` と `apps` を `dqn::` で検索し、系統のファイルとテストを除いて確かめた |
| Env は、各ライブラリの `anet::rl::env::Init<Env>()` を Runner が呼ぶ（[RunnerApp.cpp:393〜399](../../apps/runner/src/RunnerApp.cpp)）。自己登録マクロは、`ANET_REGISTER_AGENT_FACTORY` が定義とコメントアウトされた呼び出し 2 行、`ANET_REGISTER_VIEW_CREATOR` が定義だけである | `core` と `apps` を `ANET_REGISTER_` で検索した（4 行） |
| Rainbow のファイルに DefaultDQN の名前は 0 行、DefaultDQN のファイルに Rainbow の名前は 0 行である。共通部品のファイルは具象のヘッダを include しない。共通部品のエラー文 4 か所は、`DefaultDQNAgent.` の接頭辞を固定で持つ（[dqn_based_agent.cpp:904](../../core/anet-core/src/dqn_based_agent.cpp) など） | 名前で検索した |
| 内的報酬（RND）の公開の口 [intrinsic.hpp](../../core/anet-core/include/anet/intrinsic.hpp) は名前空間 `anet::rl::intrinsic` で、RND のファイルに DQN の名前は 0 行である。読むのは、内的報酬のファイル自身とそのテストを除くと `default_dqn_agent.cpp` だけである | `intrinsic` で始まるヘッダの include を検索した（5 ファイル 6 行） |
| 公開の `Network` のコンストラクタは、内部の型 `NetworkBody` を引数に取る（[nn.hpp:201](../../core/anet-core/include/anet/nn.hpp)） | 公開ヘッダを読んだ |

### 2.3 テスト

| 事実 | 確かめ方 |
|---|---|
| 基盤と Env のテスト 7 本で、374 ケースのうち 57 ケースが具象 Agent に触れる。`intrinsic_test` 34／68、`config_baseline_test` 11／22、`config_test` 6／59、`nn_test` 3／143、`trainer_test` 1／49、`metrics_logger_test` 1／16、`GridMazeEnv_test` 1／17 | ケースごとに、具象 Agent の名前とヘッダの名前を検索した |
| `intrinsic_test` は、内的報酬の内部ヘッダ（`intrinsic_rnd.hpp`）の型を 15 行で使う。DefaultDQN と組み合わせるケースの中にも、内部の設定の型を使うものがある | `RndNoveltyEstimator`・`intrinsic_rnd` を検索した |
| 系統へ移す見込みのテストは、検証資材をリポジトリの根からのパスで読む（`config_baseline_test` 5 行、`dqn_based_agent_test` 1 行）。出荷設定（`apps/runner/config`）も、同じくパスで読む | `testdata` を検索した |
| `GridMazeEnv_test` の 1 ケース（[GridMazeEnv_test.cpp:617](../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp)、タグ `prd102_runner`）は、出荷設定を読んで `RunManager` で Run を組み、GridMaze の seed が Run をまたいで共有されることを確かめる。DefaultDQN は Run を作るために要るだけである。ほかの Env のテストは、Run を組まない | ケースを読んだ。Env のテストを `RunManager`・`Trainer`・`InitRL` で検索した（GridMaze の 3 行だけ） |
| 系統のテスト 5 本は、基盤の公開しない側の名前を約 130 行で使う。用途は 4 つで、小さな Network を手で組む（Body を組む型 4 つ、60 行）、Head や module を単体で動かす実行状態を作る（`NetworkForwardState`、8 行）、標準の module を用意する（`InitNN` 59 行、`NetworkModuleRepository` 1 行）、基盤と同じ計算を比べる相手として作る（Replay の部品 3 つ、[dqn_based_agent_test.cpp:12112](../../core/anet-core/src/dqn_based_agent_test.cpp) の 1 ケース）である。基盤の内部の状態を覗いて確かめるものは無い | 名前ごとに、系統のテストを単語で検索して中身を読んだ |
| テストが自分で定義する module と Head は 13 クラスある | `NetworkModule`・`NetworkHead`・`NetworkHeadFactory` を継承する宣言を数えた |
| 代役の Agent は、テストの中にすでに 8 個ある（[trainer_test.cpp:358](../../core/anet-core/src/trainer_test.cpp)、[observers_test.cpp:289](../../core/anet-core/src/observers_test.cpp)、[RunnerApp_test.cpp:57](../../apps/runner/src/RunnerApp_test.cpp) など）。ほかに、Agent を作らない代役の factory が 1 つある（`config_inventory_test.cpp`）。Agent は設定の名前で選ばれる（[agent.cpp:105](../../core/anet-core/src/agent.cpp)） | `Agent` と `AgentFactory` を継承する宣言を、テストのファイルで検索した |
| テスト用のヘッダは、すでに 2 本ある。[test_util.hpp](../../core/anet-core/include/anet/test_util.hpp)（358 行、名前空間 `anet::test`）と `catch_test.hpp` で、製品の公開ヘッダと同じ場所にある。読んでいるのはテストのファイルだけである | 2 本の include を検索した（製品のコードは 0） |

### 2.4 置き場とビルドの出力

| 事実 | 確かめ方 |
|---|---|
| `apps/runner` には、ビルドの入力（`src/`・`CMakeLists.txt`）、手で書いて実行時に読むもの（`config/`・`tools/`）、ビルドの出力（`bin/`）、実行時にできるもの（`workspaces/`・`appdata/`・`logs/`）が同居する | フォルダを見た |
| Runner は、実行体の 2 つ上のフォルダを根にして、`config`・`appdata`・`workspaces` を引く。`appdata` は、根にそのフォルダがあるときだけ使い、無ければ利用者の設定フォルダを使う。bat は作業ディレクトリを `apps/runner` に移してから起動する。Metrics Viewer は作業ディレクトリからの相対で `workspaces` を引く | [app_util.cpp:202〜232・337](../../core/anet-core/src/app_util.cpp)、[10_run.bat](../../apps/10_run.bat)、[application.properties:18](../../apps/metrics-viewer/src/main/resources/application.properties) |
| Run の置き場を相対で書いたときの基準は、2 か所で決まる。workspace の解決は、相対の値を根からの相対で設定へ入れる（[app_util.cpp:382〜384](../../core/anet-core/src/app_util.cpp)）。メトリクスの記録は、実行体の根を基準にして、その相対をつなぐ（[RunnerApp.cpp:382](../../apps/runner/src/RunnerApp.cpp)）。設定のファイルを直接渡す起動では、既定の `runs` が実行体の根の下になる | 別モデルの検討が挙げ、コードを読んで確かめた |
| 道具は、自分の位置から根を決める。`resolve_workspace.bat` は 1 つ上、`optuna_common.py` は 3 つ上、`inspect_run.py` は 2 つ上である。道具を動かすと、この計算が変わる | [resolve_workspace.bat:7](../../apps/runner/tools/resolve_workspace.bat)。`optuna_common.py`・`inspect_run.py` を `parents` で検索した |
| workspace は 18 個、Run フォルダは合わせて 683 個ある（開発機）。追跡ファイルは、workspace ごとの設定 7 本とタグセット 1 本だけである。`apps/runner/logs` の中身は Metrics Viewer のログだけで、Runner は logs フォルダを使っていない。dump を書くコードはリポジトリに無い | フォルダと追跡ファイルを数えた。`MiniDumpWriteDump`・`.dmp` を検索した（0 行） |
| 道具は 2 か所にある。`apps/runner/tools` に追跡ファイル 11 本、`viewers/metrics-tools` に 16 本。bat 4 本（`21_metrics_viewer.bat`・`31_tb_bridge.bat`・`41_mlflow_bridge.bat`・`42_start_mlflow.bat`）が `viewers/metrics-tools` を指す | `apps/*.bat` を `viewers` で検索した（4 ファイル 7 行） |
| リリースの zip の中身は `apps`・`docs`・`licenses` と LICENSE・NOTICE である。`apps/runner` からは `bin/Release`・`config`・`workspaces`・`tools` を個別に写し、`apps/metrics-viewer` からは jar だけを写す。`core/` と `viewers/` は入らないので、上の bat 4 本は zip の中で指す先が無い。`apps/README.md` はこの 4 本を「開発用」と書く | [release.yml:203〜256](../../.github/workflows/release.yml)、[apps/README.md:26〜36](../../apps/README.md) |
| 基盤の `.lib` は `core/anet-core/lib/<構成>/` に、テスト実行体は各ライブラリの `bin/<構成>/` に出る。Env の `.lib` はビルドの木（`out/build/<preset>/`）に出る。テスト実行体の置き場ごとに、依存 DLL の写しが約 2.6 GB ある（基盤と Env で 6〜7 組）。`.lib` のパスを書く追跡ファイルは無い | 各 CMakeLists の出力先の指定、フォルダの大きさ、`.lib` のパスの検索（0 行） |
| Runner のテスト実行体（`AnetRLRunner-gui-test`）は、Runner の `bin` に出る。Runner のソースを一緒にコンパイルし、実行体の位置から config を引く | [apps/runner/CMakeLists.txt:148〜158](../../apps/runner/CMakeLists.txt) |
| Git が無視するフォルダ（各ライブラリの `bin`・`lib`、`viewers/metrics-tools/.venv`、`apps/metrics-viewer/target`）は、`git mv` では動かない | `.gitignore` と bat を読んだ |
| ログと例外の文には、ソースのファイル名と行番号が付く | [log.cpp:45〜48](../../core/anet-core/src/log.cpp) |
| Env の名前は、include（`anet/env/<Env>.hpp` の 6 本）、名前空間（`anet::rl::env`、313 行）、設定キー（`env.…`）が単数で、フォルダ（`core/envs/`）だけが複数である | 名前で検索した |

### 2.5 動かすものと参照の数

動かす追跡ファイルは、`core/anet-core` 352（うち `testdata` 210）、`core/envs` 58、`apps/runner/src` 34、`apps/metrics-viewer` 110、`viewers/metrics-tools` 16、`apps/runner/tools` 11、`apps/runner/workspaces` 8 である。

旧いパスと名前を含む行の数（括弧はファイルの数）は次のとおり。`/` と `\` の両方の書き方を数えた。

| 検索語 | 規約と用語 | 設計文書（和文） | 工程・品質・skill | wiki | ビルドと CI | bat・道具・設定 | PRD と実装計画 | 検証資材 |
|---|---|---|---|---|---|---|---|---|
| `core/anet-core` | 5(1) | 150(15) | 51(10) | 98(12) | 9(5) | 912(49) | 1,730(251) | 291(70) |
| `core/envs` | 1(1) | 26(4) | 0 | 32(2) | 12(7) | 140(7) | 212(62) | 6(5) |
| `apps/runner/src` | 0 | 8(3) | 1(1) | 18(3) | 1(1) | 105(13) | 230(44) | 14(6) |
| `apps/metrics-viewer` | 2(2) | 29(4) | 1(1) | 109(2) | 6(1) | 190(6) | 94(26) | 12(5) |
| `viewers/` | 16(2) | 9(3) | 7(1) | 12(2) | 1(1) | 39(11) | 79(27) | 3(3) |
| `apps/runner/tools` | 7(2) | 6(2) | 3(1) | 19(2) | 1(1) | 43(16) | 41(20) | 10(3) |
| `apps/runner/workspaces` | 8(4) | 7(4) | 0 | 8(1) | 4(2) | 24(8) | 32(17) | 12(12) |
| `anet-core-test` | 6(2) | 5(2) | 2(1) | 4(3) | 10(1) | 37(27) | 504(172) | 61(44) |

## 3. ゴール・非ゴール・成功の測り方

ゴール:

1. DQN 系の中を、役割の単位のファイルに分ける。
2. Agent を基盤の外のライブラリにする。基盤は具象 Agent を知らず、Agent は基盤の公開ヘッダだけを読む。テストも、この境界を素通りしない。
3. トップの階層を、置き場の規則 1 つで説明できる形にする。
4. 挙動は変えない。ただし最後のステージ（data）だけ、実行時のデータの置き場の決め方が変わる。学習の結果は、どのステージでも変えない。

非ゴール:

| 外すもの | 本当の制約か |
|---|---|
| クラスの責務の作り直し（`Learner` を分ける、など） | 制約ではない。別の関心で、置き場ができたあとの別の PRD で扱う |
| 設定キー、メトリクス名、checkpoint、名前空間（`anet::rl::dqn` など）の変更 | 制約ではない。配置と関係が無いので変えない |
| 基盤そのものを分けること（NN・RL・GUI など） | 制約ではない。別の関心 |
| 設定ファイルの配置 | 制約ではない。PRD181 が扱う |
| リリースの組み立て方 | 制約ではない。PRD185 が扱う。本書は zip の中のパスだけを変える |
| Runner の実行体の名前（`AnetRLRunner`）と、トップの `apps/` という名前 | bat、文書、走っているプロセスの名前、zip の中のパスに出るので動かさない |
| ルート直下の個人のファイル、`third_party/`、`docs/`、`reports/`、`licenses/` の整理 | 制約ではない。置き場の規則（D2）が扱うのは、このリポジトリで書くソースと、その出力と、実行時のデータである。外部のライブラリ、文書、個人のファイルは別の関心で、動かす理由が今は無い |

成功の測り方:

- DQN 系の最大のファイル（テストを除く）が、6,350 行から約 3,500 行以下になる。
- 「基盤の外で何を書けるか」に、設計文書 010 の拡張点の表の 1 行で答えられる。
- 依存の検査（§6 の S3-1〜S3-5）が違反 0 で通り、DQN だけのテスト実行体を作って回せる。
- テスト実行体の DLL の写しが、6〜7 組から 1 組になる。

## 4. 確定事項

### 4.1 決定

| # | 決定 | 理由と前提 |
|---|---|---|
| D1 | 本書の範囲に、DQN 系の中のファイルの分け方を入れる | 起点（2026-11-17）のきっかけに直接効く。PRD125 と PRD128 が「ファイルに分けるのは本書の範囲」と送っている |
| D2 | 置き場の規則は 1 つにする。ビルドの入力は `src`、実行時の配置（配る形）は `apps`、実行時にできるものは `data`、配らないビルドの出力は `out` に置く | ゴール 3。規則が 1 つなら、新しいファイルの置き場を表を引かずに決められる。日常はビルドが `apps` へ直接出し、install を挟まない。今の作りと ADR 0017（開発と配布で同じ配置を使う）を保つ |
| D3 | 層とフォルダは `src/core/base`、`src/env/<名前>`、`src/algo/<系統>`、`src/apps/<app>` にする。フォルダ名は英小文字で、Env は `lunarlander`・`dropmerge`・`atari`・`gridmaze`・`cartpole`・`imagecls`（末尾の数字を外す）、系統は `dqn`・`muzero`・`imagecls` | どの層も、ライブラリが 3 段目に来て形がそろう。`env` は include・名前空間・設定キーがすでに単数（§2.4）。Windows と Git では、大文字小文字だけの違いが事故になりやすい。末尾の数字は、版の意味が今は無い。`apps` だけ複数なのは、トップの `apps/` と対にするため。D2 に依存 |
| D4 | ライブラリのターゲットは `anet-<層>-<名前>` にする。基盤は `anet-core-base`、系統は `anet-algo-dqn`・`anet-algo-muzero`・`anet-algo-imagecls`、Env は `anet-env-<名前>`。テスト実行体は、ターゲット名に `-test` を付ける。Runner の実行体 `AnetRLRunner` は変えない。文章の中の「anet-core」は、`src/core` の層を指す語として残す | 基盤だけ新しい形にすると、Env の `GridMazeEnv` などと形が食い違う。Env の名前を変える参照は CMake の 11 か所ほどで少ない。D3 に依存 |
| D5 | Agent のライブラリは、系統ごとの 3 本にする。共通部品、DefaultDQN、Rainbow は `dqn` に入れる | 変更の多い DQN の内部ヘッダ（型 57）を公開面にしなくて済む。テストの割り振りが要らない。系統どうしの独立は機械が守る。DefaultDQN と Rainbow のあいだ、共通部品から具象への向きは、今 0 行で（§2.2）、検索の検査で保つ。RainbowAgent を共通部品の 2 つ目の利用者として保つ PRD128 の決定は変えない |
| D6 | DQN が使う基盤の内部の部品は、公開する。対象は、Head を書くための型、汎用の Head、N-step の経験の組み立てである。DQN の Head と step 診断の部品は、DQN のライブラリへ出す。公開する型が要る定義（§2.2）も一緒に出す。DQN だけが使う定数 `kKey_Taus` は DQN のライブラリへ移す。Body を組む型、module の登録、SN と SDPA の補助は内部に残す | 基盤に DQN の Head を残すと、基盤が `q`・`q_dist`・`skip_q` などの名前と Dueling・分位・IQN の計算を持ち、[AGENTS.md](../../AGENTS.md) の「汎用機構と利用側の責任境界」に反する。step 診断を基盤に置くと、[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) の「n step の窓は Agent が持つ」と食い違う。内部に残すものは、系統の実装が使っていない（§2.2）。D5 に依存 |
| D7 | 登録は、各ライブラリが公開する初期化関数を、Runner とテスト実行体が呼ぶ形にする。系統は `anet/algo/<系統>.hpp` に `anet::rl::algo::InitDqn()`・`InitMuZero()`・`InitImageCls()` を置く。このヘッダの名前は、フォルダ名と同じ英小文字にする。Env のヘッダ（`anet/env/GridMaze.hpp` など）は今の名前のままにする。`InitRL` の登録 4 行は外す。自己登録マクロ 2 つは消す | ゴール 2（基盤は具象 Agent を知らない）。Env の今の形（`anet/env/<Env>.hpp` と `anet::rl::env::Init<Env>()`）と対にする。自己登録マクロは使われていない（§2.2）。Env のヘッダの名前を英小文字にそろえる案は退けた（§4.3）。D3・D5 に依存 |
| D8 | 具象 Agent のヘッダ 4 本（`default_dqn_agent.hpp`・`rainbow_agent.hpp`・`muzero_proto_agent.hpp`・`image_cls_agent.hpp`）は、系統のライブラリの内部へ移す。基盤の公開ヘッダ `agent.hpp` にある DQN の設定の型（名前空間 `anet::rl::dqn`）も、DQN のライブラリの内部へ移す。系統の公開ヘッダは、初期化関数のヘッダだけにする | 系統の外から読んでいたのは、登録とテストだけである（§2.2）。DQN の設定の型を基盤の公開ヘッダに残すと、基盤が Munchausen・skip・分位の名前を持ち続け、D6 の理由と食い違う。D7・D12 に依存 |
| D9 | 内的報酬（RND）は基盤に残す | 公開の口にも実装にも DQN の名前が無い（§2.2）。DefaultDQN と組み合わせるテストは DQN のテストへ移す（D12） |
| D10 | PRD169 を本書に統合する。規約 4 つ（§5.6）、拡張点の表、用語、自己登録マクロの削除を本書が行う | PRD169 が先送りした 2 件（Agent の外部登録口、Head の部品の公開）は、D6・D7 で本書が行う。残る規約と表は本書の結果そのもので、2 本に分けると同じ段落と同じ表を 2 回書くことになる |
| D11 | アプリのソースは `src/apps/runner`・`src/apps/metrics-viewer` へ移す。`apps/` には bin・config・道具・bat だけを残す。実行体の出力先（`apps/runner/bin/<構成>`）と bat の作業ディレクトリ（`apps/runner`）は変えない | D2 の規則どおり。`apps` のままだと、規則が「ライブラリのソースは `src`、アプリはソースも実行時の配置も `apps`」の 2 つになる |
| D12 | テストの参照範囲と依存の向きを、次の 4 つにする。(1) 単体テストは対象のライブラリの中に置き、そのライブラリの内部まで読める。(2) ほかのライブラリは、公開ヘッダとテスト用の口だけを通して使う。(3) テストは、上位や兄弟のライブラリに依存しない。(4) 組み合わせを確かめるテストは D14 による。データのファイル（出荷設定、検証資材）は、この範囲の対象にしない。どのライブラリのテストも、リポジトリの根からのパスで読んでよい | 基盤が外に見せる面を絞る D6 を、テストが素通りしないようにする。ほかのライブラリの `src/` を相対パスで読む箇所が無くなり、同じリポジトリにあることに頼らない。[実装 品質確認観点](../quality/impl.md)の項番 46（テストは内部の構造を直接確かめない）とも合う |
| D13 | テスト用の口は、基盤の `include/anet/test/` に置き、名前空間は `anet::test` にする。補助の実装は基盤の `src/test/` に置き、基盤とは別のターゲット（テスト用のライブラリ）に入れる。ヘッダの先頭にガードを置き、テスト用のライブラリにリンクした側だけが読めるようにする。中身は、組み立ての補助と、読むだけの口に限る。製品の処理は置かない。テスト用の口は、要るライブラリだけが持つ | 補助の実装が基盤のフォルダの中にあるので、ほかのフォルダの `src/` を読む箇所が残らない。ガードで、製品のコードが読もうとするとコンパイルが止まる。今のテスト用のヘッダがすでに `anet::test` である（§2.3）。D12 に依存 |
| D14 | 組み合わせを確かめるテストは、相手のライブラリが主題でないなら、テスト用の口に置く代役の Agent を使い、主題のライブラリのテストに置く。組み合わせそのものが主題なら、`src/apps/` の下の専用のアプリに置く。このアプリは、最初のケースが出たときに作る | テストを主題のそばに置けて、兄弟のライブラリに依存しない。今あるのは GridMaze の 1 ケースで、主題は Env の seed である（§2.3）。代役の Agent はテストにすでに 8 個あり、Agent は設定の名前で選ばれる。D12・D13 に依存 |
| D15 | DQN の共通部品は、役割で 5 つに分ける。学習、行動、Network、組み立て、共通の計算である。`Learner` は学習のファイルに丸ごと入る。ヘッダとテストも同じ単位で分ける。関数の本体は変えない | 起点（2026-11-17）の「過度に分割せず」に沿う選択である。部品の単位（12 ファイル）と型ごと（40 以上）は退けた（§4.3）。D1 に依存 |
| D16 | ステージの順は、トップ階層の付け替え、DQN のファイル分割、ライブラリの分離、data にする | 最初に全部のフォルダがそろい、途中で `src/` と `core/` がトップに並ぶ期間が無い。data は置き場の決め方が変わるので、最後に置いて受入を分ける |
| D17 | 実行時のデータ（workspaces・appdata・Metrics Viewer のログ）は、トップの `data/` へ移す。名前は `data` にする。dump は範囲外とする | D2 の規則どおり。`apps/` と `src/` から Run が無くなり、再帰の検索やコピーが安全になる。`apps/` を入れ替えても Run が残る。dump を書くコードはリポジトリに無い（§2.4） |
| D18 | `data` の根は、今の config と同じく、実行体の位置から固定の相対で決める（配置の根の `data/`）。探して見つける形や、見つからないときの黙った代替は作らない。appdata は今の規則のまま移す（`data/appdata` があるときだけ使い、無ければ利用者の設定フォルダ）。追跡している 8 本（workspace ごとの設定 7 本とタグセット）は、workspaces と一緒に移す。`data` の根を返す口は 1 つにし、workspace の解決、相対で書いた Run の置き場の基準、appdata が同じ口を使う。設定のファイルを直接渡す起動でも、相対の Run の置き場は `data` の根からの相対にする（既定は `data/runs`）。Metrics Viewer と道具も、それぞれの位置から固定の相対で `data` に至る（§5.5） | 今の決め方（§2.4）と形をそろえる。今は Run の置き場の基準が 2 か所にあり、片方だけ変えると旧い置き場に Run ができる。[AGENTS.md](../../AGENTS.md) の「設定値の扱い」（黙った代替を作らない）。D17 に依存 |
| D19 | 道具は `apps/tools/` の 1 か所に集める（`apps/runner/tools` の 11 本と `viewers/metrics-tools` の 16 本）。リリースの zip には `apps/tools` を丸ごと入れる。bat が使う Python の環境は、リポジトリの根の `.venv` にする | D2 の規則どおり（道具は、手で書いて実行時にそのまま読むもの）。トップの `viewers/` が無くなる。分析の道具も zip に入り、今は指す先が無い bat 4 本が zip でも動く。選り分けの規則を持たずに済む。Python の環境は、[AGENTS.md](../../AGENTS.md) の「Python 補助ツールの実行」が根の `.venv` を定めている。道具のフォルダに作ると、実行時にできるものが `apps` に入る |
| D20 | ビルドの出力先を変える。`.lib` はビルドの木、テスト実行体は `out/test/<構成>/` の 1 か所、Metrics Viewer の jar は `apps/metrics-viewer/bin/metrics-viewer.jar`、Maven の中間物は `out/` の下にする。Runner のテスト実行体だけは、Runner の `bin` のままにする | D2 の規則（`src/` はビルドの入力だけ）。Runner のテスト実行体は、実行体の位置から config を引く（§2.4）。`.lib` は人が使わない中間物で、パスを書く箇所が無い（§2.4）。テスト実行体を 1 か所にすれば、DLL の写しが 1 組で済む。jar は Runner の `bin` とそろえる。zip の中のパスが 1 つ変わる |
| D21 | testdata は、今の `prdNNN`（37 フォルダ）を `src/core/base/testdata` へまとめて移す。これから作る資材は、対象のライブラリの下に置く。検証資材は、入力のファイルも、それを確かめるスクリプトも、テストの入力として `src` の下に置く | 過去の PRD の検証資材を、ライブラリごとに振り分け直す得が無い |
| D22 | 互換層を置かない。旧パスの symlink・junction・転送ヘッダを作らず、参照は同じ変更で移す。旧い置き場（`apps/runner/workspaces`・`apps/runner/appdata`）は読まず、あっても知らせない。各マシンの移動は人が行う | [AGENTS.md](../../AGENTS.md) のクリーンブレーク方針。Run フォルダの操作は人が行う決まり |
| D23 | 過去の記録は、移る先が同じ文書であるリンクの、丸括弧の中のパスだけを直す。本文、課題記録の `scope` の欄、コードの枠の中は当時のまま残す。過去の記録に当たるのは、PRD と実装計画、ADR、実験記録、課題記録、`reports/` の下の記録（調査レポート、統計、定期作業の記録。`reports/tools` は除く）、wiki の作業の記録、変更の記録（`CHANGELOG.md`）のすでにある項である。ステージ 2 で分かれて無くなる 3 本（`dqn_based_agent.cpp`・`dqn_based_agent.hpp`・`dqn_based_agent_test.cpp`）を指すリンクは、同じ文書の新しいパスが無いので直さず、切れたまま残す。検証資材（`testdata`）の文書（`.md`。README と検証の記録）は、手順と当時の結果が同じ文書に混ざるので、過去の記録として扱い、リンクのほかに、先頭の注記 1 行だけを足す（パスと名前が本書で変わったことと、本文は当時のままであること）。検証資材のうち、過去の Run の記録をそのまま写したファイル（調査の時点では、`prd092` の `config` の下にある、Run の実効 config の写し 11 本）は、過去の記録として扱い、中身を変えない。現用の文書・設定と、検証資材のうち実行するもの（スクリプト）と、手で書いた入力のファイルは、本文も直す。英訳（`.en.md`）と生成物（動線マップ、architecture atlas）は、リンクの丸括弧の中のパスだけを実装する側が直し、本文は、人が翻訳の skill と生成の skill を起動して追随させる | [memo README](README.md) の「リンクの注意」は、同じ文書の新しいパスへ直すことだけを、書き換えに当たらないとする。課題記録の事象は書いた時点の観測で、行番号も当時のものである。検証資材のスクリプトは、あとの PRD の検証でも使う道具である。検証資材の文書は、設計文書から手順として参照されるが、日付のある結果の節を持ち、文ごとには分けられない。Run の記録の写しは、逐語で保存して hash を記録した当時の記録で、書き換えると写しでなくなる。翻訳と生成の skill は、人の起動でだけ動く |
| D24 | 挙動を変えていないことは、着手前に保存した実行体と比べて確かめる。DQN 系は PRD128 の比べ方（全 scalar の点、設定の記録、ログの WARN と ERROR の行）を使う | PRD128 §6.2・§6.4・§7 で確かめ方が定まっている |
| D25 | 受入の検査（§6）は、手で回すコマンドと検証資材として書く。CI には載せない。Observer と ReplayBuffer の差し替え口、NN module の登録口の公開は作らない | CI の整備は別の PRD の範囲。差し替え口と登録口は要求が無い |
| D26 | 設計文書を 1 本新設し、置き場の規則、ライブラリの境界、テストの参照範囲、テスト用の口の現在形を書く。拡張点の表は 010 に置く | 新しい仕組みには、現在形を書く場所が要る |
| D27 | ADR を 3 本起こす。置き場の規則（ADR 0078）、Agent を系統ごとのライブラリにして Head を外で書ける拡張点にすること（ADR 0079）、テストの参照範囲（ADR 0080） | どれも戻しにくく、背景を知らないと意外で、案を比べて選んだ |

### 4.2 起票時の未決表の振り分け

起票時と 2026-11-17 の未決表（D1〜D23）と、起票時の §4 の暫定の既定の行き先である。左の番号は旧い番号で、右が本書の今の場所である。

| 旧い行 | 行き先 |
|---|---|
| D1（基盤のフォルダ名）・D2（Env の数字）・D4（フォルダ名）・D21（階層の名前） | D3・D4 |
| D3（Agent の単位） | D5（5 本から 3 本へ変えた） |
| D5（公開ヘッダの include パス）・D7（Init 関数） | D7・D8 |
| D6（公開面の出し方） | D6 |
| D8（テスト実行体）・D9（testdata） | D12・D20・D21 |
| D10（`viewers/metrics-tools`） | D19 |
| D11（段の順） | D16 |
| D12（英訳） | D23 |
| D13（台帳 `findings.jsonl` のパス） | 外した。台帳は追跡ファイルに無い（ファイル名の検索で 0 件） |
| D14（用語のパス）・D20（PRD169 との進め方） | D10・§5.6 |
| D15（生成物と作業領域） | 非ゴール（§3） |
| D16（リリースの同梱物） | §2.4 の事実と D19・D20・§5.5 |
| D17（DQN の中のファイルの分け方） | D1・D15 |
| D18（アプリのソース） | D11 |
| D19（実行時のデータ）・D22（logs と dump） | D17・D18 |
| D23（内的報酬の置き場） | D9（DQN 系から基盤へ変えた） |
| §4 の 8（互換層を置かない） | D22 |
| §4 の 7（文書と生成物） | D23・§5.7 |

### 4.3 退けた案

| 案 | 退けた理由 |
|---|---|
| Agent のライブラリを具象ごとの 5 本にする（起票時の既定） | 変更の多い内部ヘッダが公開面になり、12,969 行のテストを 3 つに割り振る必要がある。足される強制は、今 0 行の依存に対するものである |
| Agent 全体を 1 本のライブラリにする | MuZero や ImageCls が DQN の部品を読むことを、機械が止めない |
| DQN の Head と step 診断を基盤に残す | D6 の理由のとおり、汎用層の規約と ADR 0058 に反する |
| 系統のライブラリが基盤の `src/` を include パスに足して内部ヘッダを読む | ゴール 2 を満たさない |
| PRD169 を分けたまま、どちらかを先に確定する | 同じ段落と同じ表を 2 本の PRD が書く |
| アプリのソースを `apps/` に残す | 置き場の規則が 2 つになる |
| config をソースの側に置き、ビルドで `apps/` へ写す | config を直してすぐ走らせる今の回し方に、写しの手順が挟まる |
| data を別の PRD に分ける、または作らない | ユーザーの選択で本書に入れた。受入は最後のステージとして分ける |
| 道具を 2 か所のまま、`viewers/` だけ `apps/` の下へ移す | 探す場所が 2 つ残る。workspace を解決する bat は、どちらの道具からも使われる |
| テストを境界の対象にしない、テスト実行体を分けない | テストが、基盤の公開面を素通りして何にでも依存できる |
| テストの向きだけ守り、系統のテストが基盤の `src/` を直接読む | 同じリポジトリにあることに頼る。テスト用の口を作る手間だけで避けられる |
| テスト用の口を、別のライブラリのフォルダ（`src/core/testing`）や、基盤の中の 3 つ目のフォルダにする | 前者は、テスト用のライブラリが別のフォルダの `src/` を読む箇所が残る。後者は、ライブラリの形が 3 つになる |
| 組み合わせのテストを `src/tests/` に置く | テストは各フォルダの中、という形から 1 つだけ外に出る |
| 組み合わせのテストを Runner のテストに置く | Runner 自身のテスト（画面の操作）とは別物が混ざる |
| Env のテストと系統のテストが、互いのライブラリを使ってよいとする | 置き場が「どちらが主題か」で割れる。相手が壊れると、使う側のテスト実行体が作れない |
| DQN のファイルを部品の単位（12 ファイル）や型ごと（40 以上）に分ける | ユーザーの選択で、役割の 5 つにした。起点の「過度に分割せず」に沿う |
| ステージを、DQN のファイル分割から始める | ユーザーの選択で、トップ階層から始める。途中で `src/` と `core/` がトップに並ぶ期間ができる |
| data の移動を、トップ階層の付け替えと同じステージにまとめる | いちばん重い変更と workspaces の移動が同時になり、最初のステージを入れられる時機が、Run の止まっているときに限られる。まとめずに道具の根の計算を 2 回直す手間は、道具 3 本の数行である |
| data の移動を、単独の最初のステージにする | ユーザーの選択で、最後のままにした |
| `.lib` を `src/core/lib/<構成>/` や `out/lib/<構成>/` に出す | 前者は `src` にビルドの出力が入り、規則に例外が要る。どちらも、`.lib` のパスを人が使う場面が無い |
| フォルダとターゲットの名前を `envs`（複数）にする | include・名前空間・設定キーが単数で、フォルダだけが食い違う |
| Env の公開ヘッダ 6 本（`anet/env/GridMaze.hpp` など）の名前も、フォルダ名と同じ英小文字にそろえる | 制約ではない。そろえても本書のゴールに効かず、58 か所の参照を書き換えるだけになるので、本書では変えない。大文字小文字だけを変える改名は、Windows と Git で事故になりやすい（D3）。そろえるかどうかは別に決める |
| 無くなるファイルを指す過去の記録のリンクを、分けたあとのファイルへ付け替える | 別の文書への付け替えで、過去の記録の書き換えに当たる。5 組のどれを指すかも、1 件ずつ読まないと決まらない |
| 未解決の課題記録は現用として、`scope` の欄と本文のパスも直す | ステージ 2・3 でファイルが分かれて移ると、行き先を 1 件ずつ読まないと決まらない（旧いパスを持つ未解決の課題は約 80 件）。事象は書いた時点の観測で、再確認のときも、書き換えずに追記する決まりである |
| 検証資材の文書（README）と、過去の Run の記録の写しも現用として、本文のパスと名前を直す | 日付のある結果の節や、変更したファイルの一覧、Run の実効 config の逐語の写しが書き換わり、過去の記録の書き換えに当たる。無くなる 3 本の名前を書く文は、分けたあとのファイルへ言い直せない。文書を手順として読むときの読み替えは、先頭の注記で示す |

## 5. 実装契約

### 5.1 最後の形

```text
apps/                         実行時の配置（配る形）
├ *.bat
├ runner/        bin/<構成>/、config/
├ metrics-viewer/ bin/
└ tools/         Python・bat・ps1 の道具
data/                         実行時にできるもの
├ workspaces/<workspace>/     config/、runs/
├ appdata/
└ logs/
src/                          ビルドの入力
├ core/base/     include/anet/、include/anet/test/、src/、src/test/、testdata/
├ env/<名前>/    include/anet/env/、src/
├ algo/<系統>/   include/anet/algo/、src/
└ apps/          runner/、metrics-viewer/
out/                          配らないビルドの出力
├ build/<preset>/             ビルドの木（.lib を含む）
└ test/<構成>/                テスト実行体と DLL
third_party/、docs/、reports/、licenses/   変えない
```

`core/` と `viewers/` はトップから無くなる。

### 5.2 ステージ 1: トップ階層の付け替え

- フォルダを移す。`core/anet-core` を `src/core/base` へ、`core/envs/<名前><数字>` を `src/env/<名前>` へ、`apps/runner` の `src/` と `CMakeLists.txt` を `src/apps/runner` へ、`apps/metrics-viewer` の Maven のプロジェクト一式を `src/apps/metrics-viewer` へ、`apps/runner/tools` と `viewers/metrics-tools` を `apps/tools` へ移す。移動は `git mv` で行い、履歴を保つ。
- ターゲットの名前を D4 のとおりに変える。系統のターゲットはステージ 3 で作る。
- 出力先を D20 のとおりに変える。Runner の実行体の出力先は変えない。
- bat、Metrics Viewer を起動する bat、道具どうしの参照、`.gitignore`、`.gitattributes`（行末の指定。移したファイルが、着手前と同じ指定を持つようにする）、Doxygen の設定、ファイルを分類する設定（`reports/tools/categories.json`）、CI とリリースのワークフローを、新しいパスに合わせる。
- リリースの zip は、`apps/tools` を丸ごと入れ、jar を新しいパスで入れる。`data/` はステージ 4 で足す。
- 文書の参照を D23 のとおりに直す。生成物のうち、wiki のトピック別まとめの「ファイル」の節は、実装する側が道具（`reports/tools/wiki_support.py maps`）で作り直す。動線マップ（`docs/agents`）と architecture atlas（`docs/archify`）は、生成の skill（`anet-harness-map`・`anet-archify-atlas`）が人の起動でだけ動くので、実装する側は作り直さず、各ステージの完了報告で人に起動を頼む。設計文書の英訳の追随（翻訳の skill）も、和文を直したステージの完了報告で人に頼む。
- 道具が workspace を引く根の計算（§2.4）を、新しい位置に合わせる。data はまだ `apps/runner` の下にあるので、ステージ 4 でもう一度直す。
- テスト実行体の DLL の写しは、実行体ごとに写さず、1 つの手順にまとめる。同じフォルダへ同時に写さないためである。実行体の位置から引く口を確かめるテスト（`app_util_test`）は、新しい出力先で成り立つことを確かめる。
- 各マシンでの片付けを、変更の記録（`CHANGELOG.md`）に書く。Git が無視する旧いフォルダ（`core/` の下の `bin`・`lib`、`viewers/metrics-tools/.venv`、`apps/metrics-viewer/target`）は `git mv` で動かないので、人が消す。
- このステージはリポジトリ全体に触れるので、ほかの作業の変更が作業ツリーに残っていない時機に入れる。

### 5.3 ステージ 2: DQN のファイル分割

`dqn_based_agent.hpp`・`dqn_based_agent.cpp` を、次の 5 組に分ける。

| ファイル（`.hpp`・`.cpp`） | 役割 | 入るもの |
|---|---|---|
| `dqn_based_learner` | 学習 | `Learner`、`QuantileLearnerBase`、`TDLearner`、`QRLearner`、`IQNLearner`、`BatchUpdateResult` と学習の結果の型、学習だけが使う診断の指標の型 |
| `dqn_based_actor` | 行動 | `Actor`、`ActionPolicy` と 3 つの方策、`SpatialLadderSchedule`、`DQNActionInfo`、Actor Q ヒントの型 |
| `dqn_based_network_model` | Network | `NetworkModel`、`SkipNetworkDefinition` |
| `dqn_based_assembler` | 組み立て | `DqnComponentAssembler`、step 診断の部品 |
| `dqn_based_math` | 共通の計算 | τ の配置、価値の分解の逆変換、統計などの関数、初期優先度の推定、2 つ以上の役割が使う型（診断の購読の表、指標のキーなど） |

- 上に名前の無い型は、使う役割が 1 つならそのファイルに、2 つ以上なら共通の計算のファイルに入れる。
- 依存の向きは、共通の計算、Network、行動、学習、組み立ての順にする。あとのものが前のものを読み、逆には読まない。初期優先度の推定は学習と step 診断が使い、診断の購読の表と指標のキーは行動・学習・step 診断が使うので、共通の計算に置く。
- 関数と型の本体は変えない。変えるのは、置き場、include、名前空間の開閉、リンケージの指定（ファイルの中だけで使う指定を、2 つ以上のファイルが使うものから外すこと）と、それに伴う宣言だけである。
- テスト（`dqn_based_agent_test.cpp` の 265 ケース）も、同じ 5 つの役割と、具象 Agent と設定を確かめるケースの単位でファイルに分ける。ケースの本文は変えない。DQN 系のほかのテストのファイルは分けない。
- `dqn_based_heads.hpp`・`dqn_based_heads.cpp` は分けない。
- このステージは、同じファイルに触れる実装途中の PRD が無い時機に入れる（§9）。

### 5.4 ステージ 3: ライブラリの分離

次の順で進める。実装計画で、さらに分けてよい。

1. **テストの整理。** 上位や兄弟のライブラリに依存する 57 ケース（§2.3）を、D12・D14 に合わせる。行き先は次の規則で決め、ケースごとの表を実装計画に書く。

   | ケース | 行き先 |
   |---|---|
   | `intrinsic_test` の 34 ケース（DefaultDQN と組み合わせる確認） | DQN のテスト。内的報酬の内部の型を使うケースは、Agent を代役に替えて基盤に残すか、テスト用の口に読む口を足す |
   | `config_baseline_test` の 11 ケース（出荷設定の基準） | 系統ごとのテストに分ける |
   | `config_test` 6・`nn_test` 3・`trainer_test` 1・`metrics_logger_test` 1 | 具象 Agent が主題なら系統のテストへ移す。Agent が Run や設定を作るために要るだけなら、代役の Agent に替えて基盤に残す |
   | `GridMazeEnv_test` の 1 ケース | 代役の Agent に替えて、Env のテストに残す |

2. **基盤の公開面。** D6 の部品を、それが要る定義（§2.2）と一緒に公開ヘッダへ分ける。公開ヘッダは、内部ヘッダを読まない。
3. **テスト用の口。** D13 のとおりに作る。今の `test_util.hpp`・`catch_test.hpp` を移し、§2.3 の 4 つの用途の補助と、代役の Agent を足す。系統のテストが基盤の内部の名前を使う約 130 行を、補助を使う形に直す。`src/test/` の下の実装は、基盤のライブラリの対象から外し、テスト用のライブラリにだけ入れる。テスト用のライブラリは、基盤のプリコンパイル済みヘッダを使い回さない。ガードの定義が食い違うためである。
4. **系統のライブラリ。** `src/algo/dqn`・`src/algo/muzero`・`src/algo/imagecls` を作り、系統のファイルとテストを移す。D7 の初期化関数を足し、`InitRL` の登録を外し、Runner が 3 つの初期化関数を呼ぶ。D8 のとおり、具象 Agent のヘッダと、`agent.hpp` の DQN の設定の型を内部へ移す。自己登録マクロ 2 つを消す。
5. **規約と文書。** §5.6 を行う。

ライブラリとテスト実行体の依存は、次の形にする。

| ターゲット | 依存してよい相手 |
|---|---|
| 基盤のライブラリ | 外部のライブラリだけ |
| テスト用のライブラリ | 基盤 |
| 系統のライブラリ、Env のライブラリ | 基盤（公開ヘッダだけ） |
| 基盤のテスト実行体 | 基盤、テスト用のライブラリ |
| 系統のテスト実行体、Env のテスト実行体 | 自分のライブラリ、基盤、テスト用のライブラリ |
| Runner の実行体 | 基盤、系統のライブラリ、Env のライブラリ。テスト用のライブラリには依存しない |
| Runner のテスト実行体 | 上に加えて、テスト用のライブラリ |

### 5.5 ステージ 4: data

- `data` の根を返す口を 1 つ足し、workspace の解決、相対で書いた Run の置き場の基準、appdata をその口に合わせる（D18）。設定のファイルを直接渡す起動も、同じ基準を使う。
- `data` の根には、種類ごとに次の固定の相対で至る。

  | 種類 | `data` の根の決め方 |
  |---|---|
  | Runner（実行体） | 実行体の 2 つ上（`apps/runner`）の、さらに 2 つ上にある `data` |
  | Metrics Viewer（jar と bat） | bat が作業ディレクトリを `apps/runner` に移し、設定の既定値が、そこからの相対（`../../data/workspaces`・`../../data/logs`）で引く |
  | 道具（スクリプト） | 自分の位置（`apps/tools`）の 2 つ上にある `data` |

- 道具（workspace を解決する bat、`inspect_run.py`、`optuna_common.py` など）、Metrics Viewer の設定の既定値とログの出力先、bat の中の workspace のパスを、新しい置き場に合わせる。
- 追跡している 8 本を `data/workspaces/` へ移し、`.gitignore` を合わせる。
- リリースの zip のトップに `data/` を足し、workspace ごとの設定を入れる。タグセットの作業ファイルは今と同じく入れない。
- 用語集の「ワークスペース」の定義（既定の置き場）、[AGENTS.md](../../AGENTS.md) の Run の規約、ユーザーガイドを、新しい置き場に合わせる。
- 各マシンでの移し方を、変更の記録（`CHANGELOG.md`）に書く。手順は、走っている Run を止める、`apps/runner/workspaces` の中身を `data/workspaces` へ移す、`apps/runner/appdata` を `data/appdata` へ移す、最後に開いた workspace を選び直す、の 4 つである。固定した古い実行体で流す bat は、その実行体の根から見た置き場を使い続けるので、引数の workspace を絶対パスで渡す。
- workspaces と appdata を動かすのは人である。AI は動かさず、消さない。移動は、同じドライブの中でのフォルダの改名だけで行い、コピーと削除をしない。
- 移すための道具を、本書の検証資材（`testdata` の `prd182`）に置く。リリースには入れない。道具は、何をどこへ動かすかを表示するだけの実行と、移す前後で workspace と Run フォルダの数を比べる確認を持つ。移す先に同じ名前があれば、上書きせずに止まる。
- `data/` や `data/workspaces/` が無いときは、今の `workspaces` が無いときと同じに扱う。workspace の一覧は空になり、workspace を新しく作るときにフォルダを作る。
- appdata の置き場は、次を上から順に当てて決める。(1) `data/appdata` がフォルダとしてあれば、そこを使う。(2) 無ければ、利用者の設定フォルダの下を使う。(3) 利用者の設定フォルダも取れなければ、今と同じく構築で止める。
- このステージは、走っている Run が無いときに入れる。

### 5.6 規約・拡張点の表・用語

PRD169 から来る分である。

- [AGENTS.md](../../AGENTS.md) の基本方針の「公開ヘッダへの変更は慎重に行う」を、次の 4 つに置き換える。
  1. 公開ヘッダに置くのは、基盤の外で書けると宣言した拡張点の interface・部品・登録口の一式である。
  2. それ以外は内部ヘッダに置く。公開ヘッダは内部ヘッダを読まない。内部ヘッダを読めるのは、そのライブラリ自身と、その単体テストと、そのライブラリのテスト用の口の実装だけである。
  3. 登録は、ライブラリが公開する初期化関数を app が呼ぶ。自己登録マクロは使わない。
  4. 拡張点を「書ける」に変えるときは、interface・部品・登録口を同じ変更で公開し、拡張点の表と設計文書を同時に直す。
- 同じ節に、D2 の置き場の規則と、D12 のテストの参照範囲を足す。
- 設計文書 010 に拡張点の表を置く。列は「拡張点／interface／部品／登録口／基盤の外で書けるか／実例」で、行は Env・View・Agent・NN Head・NN module・Observer・ReplayBuffer の 7 つである。Agent と NN Head は「書ける」、NN module と ReplayBuffer の差し替えは「書けない」、Observer は「コードで付けるなら書ける」とする。NN module の行には、基底の型は公開ヘッダに見えるが、登録口が無いので書けない、と書く。
- 設計文書 110・120・130 の「テストと拡張時の確認事項」に、表を指す 1 行を足す。
- 用語集（[CONTEXT.md](../../CONTEXT.md) の「ソース構成」）に「公開ヘッダ」「内部ヘッダ」「テスト用の口」「Agent の系統」を足した（本書の起草で足した）。どれも、用語集の既存の語では言い換えられない。「公開ヘッダ」と「内部ヘッダ」は、ヘッダを読んでよい範囲で分ける語で、用語集にはヘッダを指す語が無かった（`AGENTS.md` は「公開ヘッダ」を定義なしで使っていた）。「テスト用の口」は、読める範囲がその 2 語のどちらとも違う（テスト実行体だけが使える）ので、2 語では言い分けられない。「Agent の系統」は、共通部品を分け合う Agent の実装の括りで、ライブラリの単位である。用語集には、この括りを指す語が無かった。用語集の「実行系統」は Env の RunMode の区分で、別の概念なので、「Agent の」を付けて言い分ける。PRD169 が挙げた「anet-core の外」は足さない。文章の中の「anet-core」は層の名前として残り（D4）、語として定義しなくても通じるためである。

### 5.7 移行対象

| 種類 | 対象 | 扱い |
|---|---|---|
| ビルド | ルートと各ライブラリの CMakeLists、`CMakePresets.json`、Maven の設定 | 新しいパス・ターゲット名・出力先にする |
| CI とリリース | `.github/workflows/release.yml`、`.github/scripts/push_review.py`、`docs/Doxyfile`。ほかのワークフローは、ターゲット名と実行体のパスを書く行があれば直す | 新しいパスにする |
| bat と道具 | `apps/*.bat` の 16 本、道具の 27 本 | 新しいパスにする |
| 規約と用語 | `AGENTS.md`、`CONTEXT.md`、`README.md`、`apps/README.md` | 本文を直す。§5.6 の置き換えを含む |
| 設計文書（和文） | 旧いパスを含む 16 本（200・160・100・140・120・212・150・220・130・040・110・010・020・210・030・214） | 本文を直す。040 のビルドとテストの手順、200 のコードマップ、160 の道具とアプリの配置、020・030 の起動と分析の手順、110 の Agent の登録の記述（`InitRL` へ登録する、の 2 か所）を含む |
| 設計文書の新設 | ソース配置とライブラリの境界（番号は `docs/design/README.jp.md` の表で空きを取る） | 新設する。英訳は翻訳の skill |
| 設計文書（英訳） | `docs/design/*.en.md` | リンクの丸括弧の中のパスだけを実装する側が直す。本文は、人が翻訳の skill を起動して追随させる |
| 工程・品質・skill | `docs/process`・`docs/quality`・`.claude`・`.agents`・`.codex` のうち旧いパスを含むもの | 本文を直す。`docs/quality/impl.md` の、機能グループの単位でファイルを作る規約の例（`dqn_based_agent.*`）は、ステージ 2 で、分けたあとのファイルの例に替える |
| wiki | `docs/wiki/README.md` の置き場の記述、トピック別まとめ | 規約の記述を直し、機械的に作る節を道具で作り直す。書く部分の節にある旧いパスと名前（約 10 行）は、文を直す。成立の経緯の行は、無くなるファイルの名前を書かずに言い直す |
| 生成物 | 動線マップ（`docs/agents`）、architecture atlas（`docs/archify`） | リンクの丸括弧の中のパスだけを実装する側が直す。作り直しは、人が skill を起動して行う（§5.2） |
| Git の設定 | `.gitignore`、`.gitattributes` | 新しいパスにする |
| 分類の設定 | `reports/tools/categories.json` と、それを読む道具 | 新しいパスにする |
| 検証資材 | `testdata` のスクリプト（`.py`・`.ps1`・`.cjs`）と、手で書いた入力のファイル | 新しいパスと実行体の名前にする。ステージ 2 で分かれるファイルを名指すスクリプト（PRD125 の 2 本）は、分けたあとのファイルを指すように直す |
| 検証資材の文書 | `testdata` の `.md`（README と検証の記録。旧いパスと名前を持つものは 33 本） | 過去の記録として扱う。リンクの丸括弧の中のパスを直し、旧いパスと名前を持つ文書の先頭に、注記を 1 行足す（D23） |
| 検証資材の中の、過去の Run の記録の写し | `testdata` の `prd092` の `config` の下の 11 本（Run の実効 config の逐語の写し。うち 7 本が、旧い data のパスを絶対パスで持つ） | 過去の記録として扱い、中身を変えない（D23）。フォルダと一緒に移すだけである |
| テスト | 具象 Agent を使う 57 ケース、基盤の内部の名前を使う約 130 行、テスト用のヘッダの include（`.cpp`・`.hpp` の追跡ファイルで、`anet/test_util.hpp` が 28 本、`anet/catch_test.hpp` が 49 本）、workspace の解決を確かめるテスト（タグ `[workspace]`）、道具のテスト | D12〜D14・D18 に合わせる |
| 過去の記録 | PRD と実装計画（`docs/memo`）、ADR、実験記録、課題記録（`docs/management`）、`reports/` の下（`reports/tools` を除く）、`docs/wiki/log.md`、`CHANGELOG.md` のすでにある項 | 移る先が同じ文書であるリンクの、丸括弧の中のパスだけを直す（D23）。ステージ 2 で無くなる 3 本を指すリンクは直さない |

### 5.8 クラスとファイルの一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
|---|---|---|---|
| `anet::test::StubAgent`・`StubAgentFactory` | 新規 | 何もしない代役の Agent と、その登録 | 基盤のテスト用の口（`anet/test/agent.hpp`） |
| `anet/test/nn.hpp` の補助の関数 | 新規 | 小さな Network を組む、実行状態を作る、標準の module を登録する | 基盤のテスト用の口 |
| `anet/test/replay.hpp` の補助の関数 | 新規 | Replay と同じ初期優先度の計算を、比べる相手として返す | 基盤のテスト用の口 |
| `anet/test/util.hpp`・`anet/test/catch.hpp` | 変更 | 今の `test_util.hpp`・`catch_test.hpp` を移す | 基盤のテスト用の口 |
| `anet/nn_head.hpp` | 新規 | Head を書くための型（§2.2 の 10 個と、それが要る 3 つの定義）と、汎用の Head の公開ヘッダ | 基盤の公開ヘッダ |
| `anet/replay_experience.hpp` | 新規 | N-step の経験の組み立て（§2.2 の 9 つ）の公開ヘッダ | 基盤の公開ヘッダ |
| `dqn_based_config.hpp` | 新規 | `agent.hpp` から移す DQN の設定の型 | DQN のライブラリの内部 |
| `anet/algo/dqn.hpp`・`anet/algo/muzero.hpp`・`anet/algo/imagecls.hpp` | 新規 | 系統の初期化関数 | 系統の公開ヘッダ |
| `ANET_REGISTER_AGENT_FACTORY`・`ANET_REGISTER_VIEW_CREATOR` | 削除 | 自己登録マクロ | `agent.hpp`・`gui.hpp` |
| workspaces を移す道具 | 新規 | 動かすものの表示、改名、前後の数の確認 | 本書の検証資材（`testdata` の `prd182`） |
| テスト用のライブラリ（ターゲット `anet-core-testing`）と、ガードの定義 `ANET_TEST_SUPPORT` | 新規 | テスト用の口の実装をまとめ、リンクした側にだけガードの定義を配る | 基盤の CMake |

## 6. 受入基準

どのステージにも、次の共通の受入を当てる。

| # | 受入 |
|---|---|
| C1 | 3 つの構成（`x64-Debug`・`x64-RelWithDebInfo`・`x64-Release`）で build が通る |
| C2 | その時点のすべてのテスト実行体が、既定のケースで通る。[AGENTS.md](../../AGENTS.md) の「検証」の標準テスト手順も、新しいパスで通る |
| C3 | 着手前に保存した実行体と、PRD128 §6.4 の前後比較の組（PRD125 の短縮条件の 31 組と、PRD128 が足した 3 組。seed 1 本、決定的な backend で、GridMaze・LunarLander・Atari・DropMerge を含む）を比べ、全 scalar の点、設定の記録、ログの WARN と ERROR の行が完全に一致する。比べ方は PRD128 §6.2・§6.4・§7 による。ログの行は、ソースのファイル名と行番号の部分も、置き換えてから比べる。ファイルを移して分けると、挙動が同じでもこの部分が変わるためである（§2.4）。Run 名と Run フォルダのパスは、PRD128 §7 の 3 がすでに置き換えてから比べているので、ステージ 4 で置き場が変わっても比べ方は同じである |
| C4 | docs 全体のリンク切れ検査（[memo README](README.md) の「リンクの注意」のスクリプト）で、着手前に同じ検査を回して記録した一覧から、増えた行が無い。ただしステージ 2 からは、分かれて無くなる 3 本（D23）を指す行だけは増えてよい。増えた行の指す先が、この 3 本だけであることを確かめる |
| C5 | 人が skill（`anet-harness-map`・`anet-archify-atlas`）を起動して動線マップと architecture atlas を作り直したあと、`docs/agents` と `docs/archify` の追跡ファイルを、そのステージまでの検索の文字列（旧いパス、旧いファイル名、旧いヘッダの名前、旧い data のパス）で検索して 0 行である。人の起動が済むまでは、この項目だけを未了として残し、次のステージへ進んでよい。あとのステージのあとに作り直せば、前のステージの分も満たす |

ステージごとの受入は次のとおり。検索する文字列は、表の下の枠に書く。

| # | 受入 |
|---|---|
| S1-1 | `core/`・`viewers/`・`apps/runner/src`・`apps/runner/tools`・`apps/metrics-viewer` の下と、`apps/runner/CMakeLists.txt` に、追跡ファイルが無い（`git ls-files` の出力が空）。Git が無視するフォルダの片付けは、§5.2 の手順で人が行う |
| S1-2 | 現用の範囲を「旧いパス」の文字列で検索して 0 行である。新しい名前と重なる文字列の数え方は、枠の中に書く |
| S1-3 | クリーンな作業ツリーで build した直後に、`out/test/<構成>/` にテスト実行体と DLL があり、`src/` の下に Git が無視するファイルができていない（`git status --porcelain --ignored src` の出力が空）。Runner のテスト実行体だけは、Runner の `bin` にある |
| S1-4 | リリースのワークフローを手動で実行して作った zip で、中のパスの一覧が、着手前と比べて次の差だけである。`apps/runner/tools/` の下が消えて `apps/tools/` の下が増える。jar のパスが `apps/metrics-viewer/bin/metrics-viewer.jar` になる |
| S1-5 | `tmp` を含む名前の短い Run が、`apps/10_run.bat` から起動して完走し、終了値が 0 である |
| S1-6 | `reports/tools/wiki_support.py maps` が終了値 0 で終わる（トピック別まとめの「ファイル」の節に旧いパスが無いことは、S1-2 の検索で確かめる） |
| S1-7 | `.gitattributes` に、`core/` と `viewers/` で始まる行が無い。着手前に行末の指定を持っていた追跡ファイルは、移したあとも同じ指定を持つ（`git ls-files --eol` の属性の列を、移す前後の同じファイルの対で比べて、差が無い） |
| S2-1 | 分ける前の `dqn_based_agent.hpp`・`dqn_based_agent.cpp` にあった定義の本文の集まりと、新しい 5 組のファイルの定義の本文の集まりが一致する。include、名前空間の開閉、リンケージの指定とそれに伴う宣言、ファイルの先頭のコメントは比べない。検証資材に置く道具で確かめ、差が 0 件である |
| S2-2 | DQN のテストのケースの数が、分ける前後で同じである。ケースの本文の集まりも、S2-1 と同じ道具で一致する |
| S2-3 | 現用の範囲を「旧いファイル名」の文字列で検索して 0 行である |
| S3-1 | ターゲットの依存が §5.4 の表のとおりである。`cmake --graphviz` の出力を検証資材の道具で検査し、違反が 0 件である |
| S3-2 | 基盤の `src/` を include パスに持つターゲットが、基盤のライブラリ、基盤のテスト実行体、テスト用のライブラリだけである。各 CMakeLists の include パスの指定を検証資材の道具で検査し、違反が 0 件である。あわせて、公開ヘッダと、系統・Env のライブラリのファイル（テストを含む）を、「内部ヘッダ」の文字列と `../` を含む include で検索して 0 行である |
| S3-3 | `include/anet/test/` の下のすべてのヘッダが、先頭にガード（`ANET_TEST_SUPPORT`）を持つ。テストでないファイル（名前が `_test.cpp` で終わらず、`src/test/` と `include/anet/test/` の下でもない `.cpp`・`.hpp`）に、`anet/test/` の include が 0 行である |
| S3-4 | Rainbow のファイルに DefaultDQN の名前が 0 行、DefaultDQN のファイルに Rainbow の名前が 0 行である。共通部品のファイルが、具象 Agent のヘッダを include しない。「DefaultDQN と Rainbow の検査」の文字列で検索して確かめる |
| S3-5 | `InitRL` の中に Agent の登録が無い。Runner が系統の 3 つの初期化関数を呼ぶ。`src` と `apps` を `ANET_REGISTER_` で検索して 0 行である。基盤の公開ヘッダ（`src/core/base/include`）を `namespace dqn` と `dqn::` で検索して 0 行である |
| S3-6 | テストのケースの数の合計が、ステージの前後で同じである。57 ケースの行き先が、実装計画の表のとおりである |
| S3-7 | [AGENTS.md](../../AGENTS.md) に §5.6 の 4 つの規約があり、「公開ヘッダへの変更は慎重に行う」の行が無い。設計文書 010 に拡張点の表（7 行）があり、110・120・130 に表を指す行がある。新設の設計文書がある |
| S3-8 | DQN のテスト実行体のターゲット（`anet-algo-dqn-test`）だけを指定して build でき、できた実行体で DQN のテストが通る |
| S3-9 | 現用の範囲を「旧いヘッダの名前」の文字列で検索して 0 行である |
| S4-1 | `tmp` を含む名前の短い Run が、`data/workspaces/<workspace>/runs/` に Run フォルダを作る。設定のファイルを直接渡す起動で、相対の Run の置き場が `data/` の下に解決される。どちらでも、`apps/` の下に Run フォルダができない |
| S4-2 | `apps/runner/workspaces/<名前>` にだけフォルダがあるとき、その名前を相対で指定しても、そのフォルダは使われない。workspace の解決のテスト（タグ `[workspace]`）で確かめる |
| S4-3 | 道具のテスト（workspace の解決、Optuna、`inspect_run`）が通る。Metrics Viewer が `data/workspaces` の Run を一覧に出すことを、人が確かめる |
| S4-4 | リリースのワークフローを手動で実行して作った zip に、 `data/workspaces/<workspace>/config/_main.txt` が 7 本入り、`apps/runner/workspaces` の下とタグセットの作業ファイルが入らない |
| S4-5 | 現用の範囲を「旧い data のパス」の文字列で検索して 0 行である |

現用の範囲は、追跡ファイルのうち、次の表のものを除く全部である。Git が無視するもの（Run フォルダ、ビルドの出力、`.claude/worktrees` の下のリポジトリの写し）と、テキストでないファイルは、追跡ファイルの検索に当たらないので入らない。

| 除くもの | 理由 |
|---|---|
| 過去の記録（D23）。`docs/memo`、`docs/adr`、`docs/experiments`、`docs/management`、`docs/wiki/log.md`、`reports/` の下のうち `reports/tools` 以外、`CHANGELOG.md` | 当時のパスと名前を残すため |
| 生成物。`docs/agents`、`docs/archify` | 人が skill を起動して作り直す。C5 で別に確かめる |
| 設計文書の英訳（`docs/design` の `.en.md`） | 人が翻訳の skill を起動して追随させる。本書の受入では、リンク切れ（C4）だけを確かめる |
| 検証資材の文書（`testdata` の `.md`） | 手順と当時の結果が混ざる検証の記録で、当時のパスと名前を残すため（D23） |
| 検証資材の中の、過去の Run の記録の写し（`testdata` の `prd092` の `config` の下） | 逐語で保存した当時の記録で、中身を変えないため（D23） |
| 本書の検証資材（`testdata` の `prd182`） | 移す前の名前と置き場を扱う道具と、着手前に取った記録を持つため |
| S4-2 のケースを持つテストのファイル。「旧い data のパス」の検索でだけ除く | 旧い置き場を読まないことを確かめるために、旧いパスを書くため |

```text
検索は git grep -I -F で、追跡ファイルを引く。大文字小文字を区別する。

旧いパス（固定文字列。/ と \ の両方）:
  core/anet-core   core/envs   apps/runner/src   apps/runner/CMakeLists.txt   apps/runner/tools
  apps/metrics-viewer/src   apps/metrics-viewer/pom.xml   apps/metrics-viewer/mvnw   apps/metrics-viewer/build.bat
  viewers/   metrics-viewer/target   anet-core-test   Env-test
  数える前に、当たった行から、新しい正しい名前（次の 3 つ。/ と \ の両方）を取り除く。
  取り除いたあとにも上の文字列が残る行だけを数える。新しい名前が、旧い文字列を中に含むためである。
    src/apps/runner/   src/apps/metrics-viewer/   anet-core-testing
旧いファイル名（固定文字列。拡張子を付けずに引く。dqn_based_agent.* という書き方も拾うためである）:
  dqn_based_agent
内部ヘッダ（固定文字列。基盤の src/ の下（src/test/ を含む）にあるヘッダの名前の全部から、pch.hpp を除いたもの。
調査の時点では次の 9 本。pch.hpp は、どのライブラリも同じ名前で自分のものを持つので、文字列では引かない。
ほかのライブラリの pch.hpp を読むことは、S3-2 の include パスの検査と ../ の検索で止まる）:
  nn_impl.hpp   nn_heads.hpp   replay_buffer_impl.hpp   scaler_impl.hpp   intrinsic_rnd.hpp
  app_util_impl.hpp   config_impl.hpp   metrics_impl.hpp   metrics_logger_impl.hpp
旧いヘッダの名前（固定文字列）:
  anet/test_util.hpp   anet/catch_test.hpp   anet/default_dqn_agent.hpp   anet/rainbow_agent.hpp
  anet/muzero_proto_agent.hpp   anet/image_cls_agent.hpp
DefaultDQN と Rainbow の検査（固定文字列。テストのファイルは除く）:
  名前が rainbow_agent で始まるファイルで:      default_dqn   DefaultDQN
  名前が default_dqn_agent で始まるファイルで:  rainbow   Rainbow
  名前が dqn_based_ で始まるファイルで:         default_dqn_agent   rainbow_agent
旧い data のパス（固定文字列。/ と \ の両方）:
  apps/runner/workspaces   apps/runner/appdata   apps/runner/logs
```

## 7. 先送りしたものと、作る条件

| 先送りするもの | 理由 | 作る条件 |
|---|---|---|
| 組み合わせそのものを確かめるテストのための、`src/apps/` の下の専用のアプリ | 別の関心（そのテストを足す PRD が作る）。今は入れるケースが無い | 組み合わせそのものが主題のテストを、最初に足すとき |
| `Learner` などのクラスの責務の作り直し | 別の関心 | 置き場ができたあとの、責務を直す PRD |
| 基盤そのものの分割 | 別の関心 | 基盤の中で、DQN と同じ肥大が問題になったとき |
| テストごとに書かれている代役の Agent 8 個を、テスト用の口の代役に寄せること | 別の関心 | テスト用の口ができたあと |
| 受入の検査を CI に載せること | 別の関心 | CI の整備の PRD |
| NN module の登録口の公開、Observer と ReplayBuffer の差し替え口 | 別の関心 | 基盤の外で module を書く、または差し替えると決めたとき |

## 8. 未決

| # | 決まっていないこと | 確かめ方と、結果ごとの分かれ方 |
|---|---|---|
| U1 | GridMaze のケース（§2.3）は出荷設定をそのまま読む。代役の Agent に替えると、DefaultDQN 向けの設定キーが読まれない設定キーになる。警告で済むか、構築で止まるかが分かっていない | ステージ 3 の実装で、代役に替えて回す。警告で済むなら、そのまま残す。構築で止まるなら、DefaultDQN 向けの設定キーを外して読む形に直す |

## 9. 依存と順序

- **PRD169。** 本書に統合した（D10）。PRD169 の本文は当時の記録として残す。
- **[PRD185](185_release_packaging_cmake_10prd.md)（リリースの組み立てを CMake へ移す）。** 本書のステージ 1 と 4 のあとに実装するのを基本にする。包む中身を、新しい配置で 1 度だけ書くためである。PRD185 が先に実装された場合は、本書のステージ 1 と 4 が、包む中身のパスを直す。同書の「zip の中のパスを変えない」という条件は、同書の移行の前後を比べる条件なので、本書が決めた差（S1-4・S4-4）は、その比較の前か後に分けて入れる。
- **DQN のファイルに触れる PRD。** `docs/memo` 直下の PRD のうち、本文に `dqn_based_agent` を持つものは 68 本ある（実装済みを含む）。ステージ 2 は、そのうち実装途中のものが無い時機に入れる。ステージ 2 のあと、まだ実装していない PRD が指す行番号とファイル名は、型と関数の名前で探し直す。
- **ステージの切れ目と途中。** ステージの切れ目では、ほかの PRD の実装を入れてよい。ステージの途中に、同じファイルに触れるほかの PRD の変更が入ったときは、そのステージの基準（保存した実行体、リンク切れの一覧、定義の本文の集まり）を取り直す。
- **[PRD181](181_config_file_layout_10prd.md)（設定ファイルの配置）。** 順序は独立である。同書は Agent の単位を「具象 Agent」と書くが、ソースの単位は系統になった（D5）。
- **これから足す Env と系統。** 新しい Env は `src/env/` の下、新しい系統（PPO など）は `src/algo/` の下に、D3・D4・D7 の形で置く。

## 10. 参照

- [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[実装 品質確認観点](../quality/impl.md)
- [010 フレームワーク概要](../design/010_framework_overview.jp.md)、[040 開発環境](../design/040_development_environment.jp.md)、[200 DQN 系 Agent](../design/200_dqn_agents.jp.md)
- [ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)、[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)、[ADR 0067](../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)、[ADR 0078](../adr/0078-top-level-layout-separates-build-inputs-runtime-layout-and-runtime-data.md)、[ADR 0079](../adr/0079-agents-live-in-per-family-libraries-outside-core-and-head-authoring-parts-are-public.md)、[ADR 0080](../adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md)
- [PRD169](169_outside_core_extension_layout_10prd.md)、[PRD181](181_config_file_layout_10prd.md)、[PRD185](185_release_packaging_cmake_10prd.md)、[PRD125](125_dqn_learner_update_skeleton_10prd.md)、[PRD128](128_dqn_agent_refactor_bundle_10prd.md)
- [CMakeLists.txt](../../CMakeLists.txt)、[core/anet-core/CMakeLists.txt](../../core/anet-core/CMakeLists.txt)、[apps/runner/CMakeLists.txt](../../apps/runner/CMakeLists.txt)、[release.yml](../../.github/workflows/release.yml)、[app_util.cpp](../../core/anet-core/src/app_util.cpp)
- 配置の事例（2026-11-18 に確かめた）: [Pitchfork Layout](https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs)、[Unreal Engine の Directory Structure](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-directory-structure)、[CMake の install](https://cmake.org/cmake/help/latest/command/install.html)
- テストの参照範囲の事例（同上）: [Rust の Test Organization](https://doc.rust-lang.org/book/ch11-03-test-organization.html)、[Go の testing](https://pkg.go.dev/testing)、[Bazel の testonly と visibility](https://bazel.build/reference/be/common-definitions)、[Gradle の test fixtures](https://docs.gradle.org/current/userguide/java_testing.html)、[Chromium の checkdeps](https://chromium.googlesource.com/chromium/src/+/HEAD/buildtools/checkdeps/README.md)
