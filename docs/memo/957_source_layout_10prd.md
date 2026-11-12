# ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける）暫定 PRD

> 2026-11-03 番号を 999 から 957 へ振り直した（PRD114）。

> 2026-10-24 起票。暫定の既定を置き、決めることを §8 の表に残した。グリル前で、最終簡素化パスは経ていない。行番号・行数・参照数は起票時点のもので、実装時は近傍のシンボル名・ファイル名で探し直す。
> 起点: 2026-10-23 の会話。具象 Agent 4 つと系統ごとの共通基盤が `core/anet-core/src` に Env 以外のすべてと同居している。Env は `core/envs/<env>/` の独立ライブラリで、`Init<Env>()` を app が呼ぶ。Agent も同じ形にして責務の境界を CMake で強制したい（分散の不利より分ける利得が大きい）。wxWidgets の配置を参考に見て、ソースの root が `core/` である今の形が不自然だと気づいた。
> 関連: [999 outside-core 拡張の配置](945_outside_core_extension_layout_10prd.md)（公開ヘッダと登録口の規約 P1〜P4。本書の段 2 は、同書 §7 で先送りした「Agent の外部登録口」「NN Head 部品の公開」の起動条件を満たす）、[999 設定ファイル配置](907_config_file_layout_10prd.md)（設定も具象 Agent 別。同じ単位）、[080 動線マップ](done/080_harness_map_10prd.md)・[095 LLM-Wiki](done/095_llm_wiki_10prd.md)（パスを持つ地図。再生成が要る）。

## 1. 目的と問題

ソースの置き場を「どの層（基盤・Env・Agent・アプリ）のどの具象か」でパスから読めるようにし、層の境界を CMake のターゲット境界で強制する。

今は次の状態である。

- トップが `core/` で、その下に基盤（`anet-core`）と Env（`envs/`）が並び、Agent は基盤の中にある。ソースの root は `src/` が慣例で、基盤と Env と Agent が同じ段に並ぶ形が読みやすい。
- 具象 Agent 4 つ（DefaultDQN、Rainbow、MuZero 試作、ImageCls）と系統ごとの共通基盤（DQN 系 7,318 行、MuZero 系 1,888 行）が `core/anet-core/src` の 82,900 行（テスト込み）の中にある。
- 具象は anet-core の内部ヘッダに依存している（§2.3）。このままでは外に出せない。
- Env は分離済み（6 ライブラリ、各 553〜4,019 行、`Init<Env>()` を app が呼ぶ）。Agent だけ形が違う。
- パスの参照が多い。docs に 1,281 か所（209 ファイル）、AGENTS.md 4、ルートの CMakeLists 7、testdata の README とスクリプト約 160、台帳 `findings.jsonl` 29、`.claude` の skill 3、Doxyfile 3。移すなら一括置換と検査が要る。

## 2. 現状（2026-10-24 の調査）

### 2.1 トップ階層

| パス | 内容 |
|---|---|
| `apps/` | bat 16 本、`runner/`（C++ 7,072 行、`config/`、`workspaces/`、`bin/`、`tools/`）、`metrics-viewer/`（Java）、`Servers/` |
| `core/` | `anet-core/`（基盤）、`envs/`（6 ライブラリ） |
| `viewers/metrics-tools/` | Python の viewer と TensorBoard・MLflow bridge |
| `third_party/` | box2d、catch2、nlohmann、nvtx3、runtime_dlls、tracy |
| `docs/`、`reports/`、`papers/`、`licenses/` | 文書・報告・論文・ライセンス |
| `.scratch/`、`.claude/`、`.agents/`、`.codex/` | 作業領域とハーネス |
| `out/`、`tmp/`、`.vs/`、`.venv/`、`subagent-reviews/` | 生成物・作業（範囲外。§8 D15） |

### 2.2 ライブラリと行数

| ライブラリ | 行数 | 形 |
|---|---|---|
| `core/anet-core` | 82,900（include/anet の公開ヘッダ 40 本、src の実装・内部ヘッダ・テスト、`testdata/`） | 1 ライブラリ。`src/**` を GLOB_RECURSE、`src/*test.cpp` を 1 本の `anet-core-test` に |
| `core/envs/atari1` | 3,520 | 各 Env は `CMakeLists.txt` ＋ `include/anet/env/` ＋ `src/`、test 実行体 1 本、`anet-core` へ link |
| `core/envs/dropmerge1` | 4,019 | 同上 |
| `core/envs/imagecls1` | 2,736 | 同上 |
| `core/envs/lunarlander1` | 2,211 | 同上 |
| `core/envs/gridmaze1` | 687 | 同上 |
| `core/envs/cartpole2` | 553 | 同上 |

ルートの [CMakeLists.txt:229〜238](../../CMakeLists.txt) が `core/anet-core`、Env 6 つ、`apps/runner` を `add_subdirectory` する。[apps/runner/CMakeLists.txt:46〜61](../../apps/runner/CMakeLists.txt) は `anet-core` と Env のライブラリを列挙して link する。

### 2.3 Agent のファイルと依存

| ファイル | 行数 | 公開／内部 | 依存する内部ヘッダ |
|---|---|---|---|
| `src/dqn_based_agent.cpp` / `.hpp` | 4,966 / 1,404 | 内部 | `replay_buffer_impl.hpp`（`CreateReplayBuffer`、`ReplayBufferConfig`、`PrefetchingReplayBuffer`、`CreateDecisionWindowRule`）、`nn_heads.hpp` |
| `src/dqn_based_heads.cpp` / `.hpp` | 948 / — | 内部 | `nn_heads.hpp`・`nn_impl.hpp`（`ProjectionHead`、`LinearProjectionModule`、`HeadFactoryBase`、`HeadGraphVizInfo`、`GetFeature`） |
| `src/default_dqn_agent.cpp` ＋ `include/anet/default_dqn_agent.hpp` | 700 ＋ 633 | 公開（ヘッダ） | `dqn_based_agent.hpp`、`dqn_based_heads.hpp`、`nn_heads.hpp` |
| `src/rainbow_agent.cpp` ＋ `include/anet/rainbow_agent.hpp` | 244 ＋ — | 公開（ヘッダ） | 同上 |
| `src/muzero_based_agent.cpp` / `.hpp`、`src/muzero_rb.cpp` / `.hpp` | 1,148 / 258、334 / 148 | 内部 | `nn_heads.hpp` |
| `src/muzero_proto_agent.cpp` ＋ `include/anet/muzero_proto_agent.hpp` | 97 ＋ — | 公開（ヘッダ） | `muzero_based_agent.hpp`、`muzero_rb.hpp` |
| `src/image_cls_agent.cpp` ＋ `include/anet/image_cls_agent.hpp` | — | 公開（ヘッダ） | `nn_heads.hpp` |
| テスト | `dqn_based_agent_test.cpp` 9,567、`dqn_based_test.cpp` 1,193、`dqn_munchausen_test.cpp` 968、`image_cls_agent_test.cpp` 1,751。MuZero のテストは無い | — | — |

[999 outside-core §2.2](945_outside_core_extension_layout_10prd.md) の調べでは、公開ヘッダ 40 本のうち anet-core の外から届くのは 29 本で、具象 Agent 4 本の公開ヘッダは外から 0 である。

### 2.4 登録と初期化

- Agent: [init.cpp:173〜176](../../core/anet-core/src/init.cpp) の `RegisterAgentFactory<…>()` 4 行が `InitRL` の中で、Rainbow・DefaultDQN・MuZero・ImageCls を `AgentRepository` に登録する。
- Env: `anet::rl::env::InitLunarLander()` などを [RunnerApp.cpp:371〜377](../../apps/runner/src/RunnerApp.cpp) が呼ぶ。
- `ANET_REGISTER_AGENT_FACTORY`（[agent.hpp:368](../../core/anet-core/include/anet/agent.hpp)）の自己登録マクロは outside-core の P3 で削除予定。

### 2.5 参考にした配置（wxWidgets）

公開ヘッダは `include/wx/` に 1 本化、実装は `src/<common|generic|port>/`、`samples/`・`tests/`・`utils/`・`docs/`・`build/` がトップにある。1 ライブラリ＋ポート構成で、anet（基盤・Env・Agent の複数ライブラリ）とは単位が違う。写すのはトップ階層の考え方（ソースは `src/`、実行体と道具は別）だけにする。

## 3. ゴールと非ゴール

ゴール:

1. トップは `src/`（ライブラリ）、`apps/`（実行体）、`third_party/`、`docs/`、`reports/` で読める。
2. Agent は Env と同じ具象別のライブラリになり、`Init*()` を app が呼ぶ。
3. anet-core は Agent に依存しない。Agent は anet-core の公開ヘッダと自分のライブラリだけに依存する。
4. 挙動は変えない（4 env の metrics checksum 一致）。

非ゴール: 挙動の変更、名前空間の変更（`anet::rl::dqn` 等）、公開ヘッダ全体の整理（届かない 11 本は outside-core の範囲）、StampFly（別 repo 案）、metrics-viewer（Java）の中身、`.scratch`・`reports` の整理。

## 4. 提案の骨子（暫定の既定）

1. トップ階層。

   | 今 | 後 | 内容 |
   |---|---|---|
   | `core/anet-core` | `src/core` | 基盤ライブラリ。ターゲット名 `anet-core` は維持 |
   | `core/envs/<env>N` | `src/envs/<env>` | Env ライブラリ。末尾の数字を外す（atari、cartpole、dropmerge、gridmaze、imagecls、lunarlander） |
   | （新設） | `src/agents/<agent>` | Agent ライブラリ（2） |
   | `apps/runner`、`apps/metrics-viewer` | そのまま | 実行体 |
   | `viewers/metrics-tools` | `apps/metrics-tools` | Python の viewer と bridge（§8 D10） |
   | `third_party`、`docs`、`reports`、`papers`、`licenses` | そのまま | |
   | `out`、`tmp`、`.vs`、`.venv`、`subagent-reviews` | 範囲外 | 生成物・作業領域（§8 D15） |

2. Agent ライブラリ（5 本）。各ライブラリは Env と同じ形（`CMakeLists.txt`、`include/anet/agent/`、`src/`、test 実行体）。

   | ライブラリ | 中身 | 依存 |
   |---|---|---|
   | `src/agents/dqn_based` | Actor・Learner・方策・Head・NetworkModel の DQN 系共通。今の内部ヘッダ `dqn_based_agent.hpp`・`dqn_based_heads.hpp` を `include/anet/agent/dqn_based/` に公開 | anet-core |
   | `src/agents/default_dqn` | DefaultDQNAgent と factory | dqn_based |
   | `src/agents/rainbow` | RainbowAgent と factory | dqn_based |
   | `src/agents/muzero` | muzero_based、muzero_rb、proto agent と factory | anet-core |
   | `src/agents/image_cls` | ImageClsAgent と factory | anet-core |

3. anet-core の公開面。具象が使う内部の 2 系統を公開ヘッダへ出す。(a) Head 部品: `nn_heads.hpp` の `NetworkHead`・`LinearHead`・`LinearHeadFactory`・`PreparedHeadExecutor` と、`nn_impl.hpp` の `ProjectionHead`・`LinearProjectionModule`・`HeadFactoryBase`・`HeadGraphVizInfo`・`GetFeature`。(b) Replay の生成: `replay_buffer_impl.hpp` の `CreateReplayBuffer`・`ReplayBufferConfig`・`PrefetchingReplayBuffer`・`CreateDecisionWindowRule`。outside-core の P1・P4 の形で、拡張点の表の行を「書ける」に変える。それ以外の内部ヘッダは触らない。
4. 登録。各 Agent ライブラリが `anet::rl::agent::InitDefaultDQN()` 等を公開し、RunnerApp が Env と同じ並びで呼ぶ（P3）。`InitRL` から 4 行を外す。テスト実行体も必要な Init を呼ぶ。
5. テスト。DQN 系のテストは `dqn_based`・具象のライブラリの test 実行体へ。`anet-core-test` は基盤だけ。`testdata/` は各ライブラリに置く（`src/core/testdata/prdNNN` は残す）。
6. 段階。段 1 はトップ階層の付け替え（`core/` → `src/`、Env の数字の除去、`viewers` の移動、CMake と include path と docs の参照の一括置換、地図の再生成）。段 2 は Agent の分離（公開面 → 5 ライブラリ → Init → テスト分割）。各段で build・全テスト・4 env の checksum・リンク検査を通す。段 1 を先にするのは Agent のファイルを 2 度動かさないため（§8 D11）。
7. 文書と地図。[010 §4.2](../design/010_framework_overview.jp.md) のコードマップ、AGENTS.md（4 か所）、CONTEXT.md（outside-core §8 の 3 の用語「公開ヘッダ」「anet-core の外」が持つパス）、Doxyfile、`.claude` の 3 skill、docs の 1,281 参照（スクリプトで置換してリンク検査）、archify の地図（anet-archify-atlas）、動線マップ（anet-harness-map）、wiki の実装地図の「ファイル」節（anet-lint-wiki）、台帳 `findings.jsonl` の 29 パス（§8 D13）、testdata の README 約 160（置換）。
8. 互換層は作らない。旧パスの symlink・junction・include の転送ヘッダは置かない。`git mv` で履歴を保つ。

## 5. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定（暫定） |
|---|---|---|
| トップを `src/` に | `core/` の下に基盤と Env が並び Agent は基盤の中。層がパスから読めない。wx の配置と比べて気づいた | 作る（段 1） |
| Agent を具象別ライブラリに | Env は分離済みで形が決まっている。Agent は内部ヘッダ 2 系統に依存したまま 11,253 行（テスト除く）が基盤に同居 | 作る（段 2） |
| DQN 系の共通ライブラリ `dqn_based` | DefaultDQN と Rainbow が 7,318 行を共有する。具象ごとに複製しない | 作る |
| 公開面の切り出し 2 系統 | 具象の include を数えた結果（§2.3）。それ以外の内部ヘッダは使っていない | 2 系統だけ |
| Init 関数 | Env の形（P3）。自己登録マクロは静的ライブラリで落ちる | Env と同じ |
| 段 1 を先に | Agent のファイルを `core/agents/` → `src/agents/` と 2 度動かさない | 段 1 → 段 2 |
| 互換層を置かない | 旧パスが残ると地図と検査が二重になる | 置かない |

## 6. 受入基準

1. 3 構成（Debug・RelWithDebInfo・Release）で build が通り、全テストが通る。
2. 4 env（GridMaze・LunarLander・Atari・DropMerge）の段 0 checksum が移行前後で一致する（PRD098 と同じ手順）。
3. CMake の依存グラフで、`anet-core` は Agent ライブラリへ link しない。Agent ライブラリは `anet-core` の公開ヘッダと自ライブラリ（dqn_based）だけを include する。公開ヘッダから内部ヘッダの include は 0（P2）。
4. 旧パスが残らない（`core/`、`viewers/` が無い）。symlink・転送ヘッダが無い。
5. 文書のリンク検査が 0 件。docs・AGENTS.md・CONTEXT.md・Doxyfile・`.claude`・testdata の README・台帳のパスが新しい。
6. 地図 3 種（archify、動線マップ、wiki の実装地図）が再生成済み。
7. 行数の移動量（ファイルごとの前後の場所）を実装メモに記録する。

## 7. 先送りした段階と、作る条件

- 公開ヘッダ 11 本の整理（届かないもの）: outside-core §8 の 4 の裁定に従う。
- `tests/` のトップ集約: 条件は、ライブラリをまたぐテストが要るとき。
- `include/anet/` のトップ 1 本化（wx の形）: 条件は、ライブラリが増えて PUBLIC include dir の列挙が煩雑になったとき。
- `apps/runner/config` の配置: [999 設定ファイル配置](907_config_file_layout_10prd.md)。
- StampFly の PPO の置き場: 別 repo 案の裁定後。
- `papers/`・`licenses/` の扱い: 範囲外。

## 8. 決めること（暫定の既定）

| # | 決めること | 暫定の既定 | 代案と退けた理由 |
|---|---|---|---|
| D1 | 基盤のフォルダ名 | `src/core`（ターゲット名は `anet-core` のまま） | `src/anet-core`: パスに repo 名が重なる |
| D2 | Env の末尾の数字 | 外す（`gridmaze1` → `gridmaze`） | 残す: 版の意味が今は無く、Agent 側と揃わない |
| D3 | Agent の単位 | DQN 系 1 ＋ 具象 4 の 5 本 | 系統ごとに 3 本（具象をまとめる）: 具象 1 つに 1 パスの規則から外れる |
| D4 | Agent のフォルダ名 | snake_case（`default_dqn`） | クラス名（`DefaultDQNAgent`）: Env のフォルダが snake_case |
| D5 | 公開ヘッダの include パス | `include/anet/agent/<name>.hpp`（Env の `anet/env/` と対） | 今の `anet/default_dqn_agent.hpp` のまま: 所属が分からない |
| D6 | 公開面の出し方 | 基盤の部品（Head 部品・Replay 生成）は anet-core の公開ヘッダへ。DQN 系の部品（`dqn_based_agent.hpp`）は `dqn_based` ライブラリの include へ | 全部 `dqn_based` に抱える: MuZero・ImageCls も Head 部品を使う |
| D7 | Init 関数の名前と呼び手 | `anet::rl::agent::Init<Agent>()`。RunnerApp と各テスト実行体 | `InitRL` の中で呼ぶ: anet-core が Agent に依存する |
| D8 | テスト実行体 | ライブラリごと（Env と同じ） | `anet-core-test` に残す: 基盤が Agent に依存する |
| D9 | `testdata/` | 各ライブラリの下。`src/core/testdata/prdNNN` は残す | トップ `tests/`: §7 |
| D10 | `viewers/metrics-tools` | `apps/metrics-tools` | 残す: 実行体が 2 か所に分かれる |
| D11 | 段の順 | 段 1（階層）→ 段 2（Agent） | 段 2 を先: Agent を 2 度動かす |
| D12 | 英訳 `.en.md` | 和文の置換後に translate skill で追従 | 手で直す: 量が多い |
| D13 | 台帳 `findings.jsonl` の 29 パス | 置換する（指紋は内容で付くので変わらない。要確認） | 注記だけ: 台帳の導出ビューが切れる |
| D14 | outside-core の用語のパス | 「公開ヘッダ」= `src/core/include/anet/`、「anet-core の外」= `src/envs`・`src/agents`・`apps` に改める | 据え置き: 定義が旧パスを指す |
| D15 | `out`・`tmp`・`.vs`・`.venv`・`subagent-reviews` | 範囲外（生成物・作業領域） | 片付ける: 別の整理 |
| D16 | リリースの同梱物（043）の参照パス | 要確認。`apps/runner` 配下なら影響なし | — |

## 9. 依存と順序

1. 段 1 は独立で始められる。段 2 は [999 outside-core](945_outside_core_extension_layout_10prd.md) の P1〜P4 がグリルで確定してから。
2. [999 設定ファイル配置](907_config_file_layout_10prd.md)と「具象 Agent」の単位をそろえる。順序は独立。
3. 地図の再生成に [095](done/095_llm_wiki_10prd.md) の点検ツールと [080](done/080_harness_map_10prd.md) の skill を使う。

## 10. 再確認ポイント

- 行数・参照数（2026-10-24）。init.cpp・CMakeLists・RunnerApp の行番号。
- `findings.jsonl` の指紋がパスを含むか（D13）。
- リリースの同梱物の作り方が参照するパス（D16）。
- wxWidgets の配置の読み（§2.5）。

## 11. 参照

- [010 §4.2 コードマップ](../design/010_framework_overview.jp.md)
- [999 outside-core 拡張の配置](945_outside_core_extension_layout_10prd.md)
- [999 設定ファイル配置](907_config_file_layout_10prd.md)
- [CMakeLists.txt](../../CMakeLists.txt)、[apps/runner/CMakeLists.txt](../../apps/runner/CMakeLists.txt)、[core/envs/gridmaze1/CMakeLists.txt](../../core/envs/gridmaze1/CMakeLists.txt)
- [init.cpp](../../core/anet-core/src/init.cpp)、[RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp)、[agent.hpp](../../core/anet-core/include/anet/agent.hpp)
- [080 動線マップ](done/080_harness_map_10prd.md)、[095 LLM-Wiki](done/095_llm_wiki_10prd.md)
