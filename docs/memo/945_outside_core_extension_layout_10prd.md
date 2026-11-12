# anet-core の外で実装する拡張点とコード配置の規約 暫定 PRD

> 2026-11-03 番号を 999 から 945 へ振り直した（PRD114）。

> 2026-10-10 起票。暫定の既定を置き、決めることを §8 の表に残した。グリル前で、最終簡素化パスは経ていない。行番号は起票時点のもので、実装時は近傍のシンボル名で再検索する。
> 起点: [084 の Forward 準備リファクタリング](done/084_noisynet_23impl.md)のレビューで、Head の準備 helper（`PreparedModuleChildren<N>`）を `nn.hpp` と `nn_impl.hpp` のどちらに置くかが問われた。調べると、Head の interface は公開ヘッダにあるが、Head を組む部品と Agent への差し込み口は anet-core の中にしか無く、「anet-core の外で Head を書く」は扉だけがある状態だった。同じ問いは Head に限らず、Agent・NN module・Observer・ReplayBuffer にも立つ。置き場の規約が無いので、問われるたびに CMake・include・登録口を走査して答えている。
> 関連: [084_noisynet_10prd.md](done/084_noisynet_10prd.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[937_nn_head_projection_separation_10prd.md](937_nn_head_projection_separation_10prd.md)（Head の責務分担。本 PRD は置き場だけを扱い、責務は触らない）、[908_config_root_ownership_alignment_10prd.md](908_config_root_ownership_alignment_10prd.md)（設定の置き場の規約。本 PRD はコードの置き場で対になる）、[079](079_cpp_internal_linkage_convention_10prd.md)（規約を AGENTS.md の 1 段落に置いた前例）、[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md)（外部ビルド木の参照。外から取り込む側の境界）、[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)と[043](done/043_release_packaging_10prd.md)（リリースは実行用パッケージだけ）、[docs/StampFly.md](../StampFly.md)（次に増える Env と Agent の置き場）。

## 1. 目的と問題

anet-core の外（同じリポジトリの別ターゲット）で何を実装できるかを、拡張点ごとに 1 行で答えられるようにする。そのために、公開ヘッダと内部ヘッダの置き場の規約を 1 つ決め、今の配置をその規約に照らして表にし、ずれているところを最小限で直す。

今は次の状態で、問われるたびに調べ直している。

- 規約が無い。[AGENTS.md](../../AGENTS.md) の基本方針は「公開ヘッダへの変更は慎重に行う」の 1 行で、何を公開に置くかの基準が無い。[010 §4.2](../design/010_framework_overview.jp.md) はディレクトリ名を挙げるだけである。
- 公開と利用が一致していない。公開ヘッダ 40 本のうち、anet-core の外（`apps/runner`、`core/envs`）から include で届くのは 29 本で、11 本はどこからも届かない（§2.2）。NN module と Head は interface・部品・登録口が内部にある。
- 拡張点ごとに事情が違う。Env と View は公開ヘッダだけで外に実装でき、実例が 6 本ある。Agent は interface も部品も公開だが、登録は `InitRL()` の編集で、外からは足せない。NN module と Head は部品も登録口も内部にある（§2.1）。

## 2. 現状（2026-10-10 の調査、NN 公開面は 2026-10-11 の変更を反映）

### 2.1 拡張点ごとの置き場

「外で書けるか」は、interface・部品・登録口の 3 つが公開ヘッダに揃っているかで決まる。

| 拡張点 | interface | 部品 | 登録口 | 外で書けるか | 実例 |
|---|---|---|---|---|---|
| Env | 公開（[rl.hpp](../../core/anet-core/include/anet/rl.hpp) の `SingleDiscreteEnv`／`BatchEnv`、[env.hpp](../../core/anet-core/include/anet/env.hpp)） | 不要 | 公開（`EnvRepository`、`RegistEnvFactory`） | 書ける | `core/envs/` の 6 本。公開ヘッダだけを include し、各ライブラリが `Init<Env>()` を公開して [RunnerApp.cpp:371](../../apps/runner/src/RunnerApp.cpp) が呼ぶ。テストも公開の `test_util.hpp`／`catch_test.hpp` で書いている |
| View | 公開（[gui.hpp](../../core/anet-core/include/anet/gui.hpp) の `ViewCreator`） | 不要 | 公開（`ViewRepository`、`RegisterViewCreator`） | 書ける | Env ライブラリが `Init<Env>()` の中で登録する（[LunarLander.cpp:12](../../core/envs/lunarlander1/src/LunarLander.cpp)） |
| Agent | 公開（[agent.hpp](../../core/anet-core/include/anet/agent.hpp) の `Agent`／`AgentFactory`、rl.hpp の `Actor`／`Learner`） | 公開: NN の構築と forward（nn.hpp）、ReplayBuffer の生成（replay_buffer.hpp）、stacker・transfer・scaler・schedule・probe。内部: `ActionPolicy`・DQN 系 `Actor`／`Learner`（`dqn_based_agent.hpp`、41 型）、正規化の実装（`scaler_impl.hpp`） | 公開の `AgentRepository` があるが、登録は [init.cpp:173](../../core/anet-core/src/init.cpp) の `InitRL()` に固定。[110 §8](../design/110_agents_and_learning.jp.md) の手順 1 も「`InitRL()` へ登録する」 | 書けるが、登録に anet-core の編集が要る | 無し。具象 Agent 4 本（DefaultDQN、Rainbow、ImageCls、MuZero 試作）は anet-core の中で、ヘッダは公開だが外からの include は 0 |
| NN module | 内部（[nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp) の `NetworkModuleExecutor`／`NetworkModule`） | 内部（`WeightInitializer`、`LinearProjectionModule`、`SpectralNormState`、箱の内部型 `NetworkForwardState`） | 内部（`NetworkModuleRepository`、登録は [nn_modules.cpp:3267](../../core/anet-core/src/nn_modules.cpp) の `InitNN()`） | 書けない | — |
| NN Head | 内部（[nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp) の `NetworkHead`／`NetworkHeadExecutor`／`NetworkHeadFactory`。nn.hppには必要な前方宣言だけ） | 内部（NN module と同じ部品） | 無し。DefaultDQN は [default_dqn_agent.cpp:92](../../core/anet-core/src/default_dqn_agent.cpp) の switch で anet-core 内の factory から選ぶ | 書けない。interfaceも内部ヘッダにあり、Agentに差し込めない | 独自 Head はテストの 7 本だけで、全部 anet-core の中 |
| Observer | 公開（rl.hpp の `TrainObserver` 等、`Notifier::Attach`） | 不要 | 無し。設定からの生成は [observers.hpp:466](../../core/anet-core/include/anet/observers.hpp) の `ObserverFactory` に固定 | コードで attach する形なら書ける。設定からは生成できない | 無し |
| ReplayBuffer | 公開（rl.hpp）。生成は [replay_buffer.hpp:78](../../core/anet-core/include/anet/replay_buffer.hpp) の `CreateReplayBuffer` | 内部（`replay_buffer_impl.hpp`、20 型） | 無し | 差し替えられない | — |

境界は [CMakeLists.txt:46](../../core/anet-core/CMakeLists.txt) が決めている。`include/` が PUBLIC、`src/` が PRIVATE で、`src/` を見られるのは anet-core 自身と `anet-core-test` だけである。公開ヘッダが内部ヘッダを include している箇所は無く、`apps/` と `core/envs/` が `src/` を相対パスで include している箇所も無い。

### 2.2 公開ヘッダの到達状況

| 区分 | 本数 | ヘッダ |
|---|---|---|
| 外から直接 include | 22 | gui、env、rl、config、profile、observers、log、trainer、str_util、random、test_util、catch_test、metrics_logger、thread、heat_map、exception、diag、app_util、tensor_util、scaler、init、image |
| 推移的にだけ届く | 7 | common、graphviz、json_util、probe、serialize、tensor_check、util |
| 届かない | 11 | agent、default_dqn_agent、rainbow_agent、image_cls_agent、muzero_proto_agent、replay_buffer、stacker、transfer、schedule、nn、nn_util |

届かない 11 本は、Agent を外で書くときに要る部品、具象 Agent、NN 利用側のヘッダである。設計文書からのリンクは 1 本あたり 6〜25 文書に及ぶ（agent.hpp 25、image_cls_agent.hpp 16、replay_buffer.hpp 16、default_dqn_agent.hpp 15、muzero_proto_agent.hpp 13、transfer.hpp 12）。

### 2.3 登録の仕組み

- Env と View は、各ライブラリが公開する `Init<Env>()` を app が明示的に呼ぶ（[LunarLander.hpp](../../core/envs/lunarlander1/include/anet/env/LunarLander.hpp)）。静的ライブラリでは、参照されないオブジェクトの静的初期化子はリンクされないので、自己登録では落ちる。Agent 用の自己登録マクロ `ANET_REGISTER_AGENT_FACTORY` は [agent.hpp:332](../../core/anet-core/include/anet/agent.hpp) にあるが、呼び出しは 2 か所ともコメントアウトされている。
- Agent は `InitRL()` が 4 本を登録し、NN module は `InitNN()` が登録する。どちらも anet-core の中の固定の列挙である。

### 2.4 次に増えるもの

[docs/StampFly.md](../StampFly.md) では、SimStampFlyEnv は `core/envs/stampfly1`（Env の拡張点。今の規約で書ける）、PPO は PRD A で anet-core の中に置く。anet-core の外に Agent や NN の部品を置く計画は、起票時点では無い。

## 3. ゴールと非ゴール

ゴール:

1. 公開ヘッダと内部ヘッダの置き場の規約を 1 つ決め、正本を 1 か所に置く。
2. 拡張点ごとに「anet-core の外で書けるか」「要る interface・部品・登録口」「実例」を表にして、設計文書に載せる。
3. 今の配置のうち、規約とずれていて直す価値のあるものを最小限で直す。

非ゴール（§7 に、作る条件を書いた）:

- 別リポジトリ向けのライブラリ配布と ABI の安定。リリースは実行用パッケージだけで（[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)、[043](done/043_release_packaging_10prd.md)）、`install` 規則も無い。
- プラグインの動的読み込み。
- Agent・Head・Module の設計や責務の変更（[937_nn_head_projection_separation_10prd.md](937_nn_head_projection_separation_10prd.md) の範囲）。
- NN module と Head の部品・登録口の公開、Agent の外部登録口（先送り）。
- Observer と ReplayBuffer の差し替え口。

## 4. 提案の骨子（暫定の既定）

1. 用語。「公開ヘッダ」は `core/anet-core/include/anet/`、「内部ヘッダ」は `core/anet-core/src/` の `.hpp`。「anet-core の外」は同じリポジトリの anet-core 以外のターゲット（`core/envs/*`、`apps/*`、将来の別ライブラリ）。`core/envs` も `core/` の下にあるので、「core の外」とは書かず「anet-core の外」と書く。
2. 規約（P1〜P4）。

   | # | 規約 | 根拠・前例 |
   |---|---|---|
   | P1 | 公開ヘッダに置くのは、anet-core の外で実装すると宣言した拡張点の interface と、その実装に要る部品と登録口の一式。宣言は §2.1 の表（010 へ移す）で行う | Env・View の 6 本の実例。部品が揃わない公開（変更前の Head の interface だけ公開）は、扉だけを作る |
   | P2 | 外で実装しないと決めた型と、その部品は内部ヘッダに置く。公開ヘッダから内部ヘッダを include しない | CMake の PUBLIC／PRIVATE の区分。現状も違反 0 |
   | P3 | 外からの登録は、Env と同じ「ライブラリが公開する `Init` 関数を app が呼ぶ」形にする。自己登録マクロは使わない | 静的ライブラリで落ちる（§2.3）。コメントアウトされた 2 か所 |
   | P4 | 拡張点を「書ける」に変えるときは、interface・部品・登録口を同じ変更で公開し、表の行と設計文書の §8（拡張時の確認事項）を同時に直す | 片方だけ公開して食い違う形を避ける |

3. 正本の置き場。P1〜P4 は AGENTS.md「基本方針」の「公開ヘッダへの変更は慎重に行う」を置き換える 1 段落にし、拡張点の表は [010 §4.2](../design/010_framework_overview.jp.md) に置く。110・120・130 の §8 には、表を指す 1 行を足す。
4. 今回直す範囲。
   - 文書: 規約の段落、拡張点の表、各 §8 の 1 行、用語 3 語（§8 の 3）。
   - コード: `ANET_REGISTER_AGENT_FACTORY` のマクロ定義とコメントアウトされた呼び出し 2 か所の削除（P3）。`PreparedModuleChildren<N>` は内部ヘッダ `nn_impl.hpp`（P2 のとおりで、084 の整理に含める）。
   - 動かさないもの: 届かない公開ヘッダ 11 本は据え置く（§8 の 4）。
5. 「書ける」に変える 2 件（Agent の外部登録、NN 部品の公開）は、§7 の条件が満たされたときに P4 の形で行う。

## 5. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定（暫定） |
|---|---|---|
| 置き場の規約 1 段落 | 2026-10-10 の Head helper の置き場の問いに、規約が無く、CMake・include・登録口の走査で答えた。AGENTS.md は「慎重に」の 1 行だけ | 作る |
| 拡張点の表 | 公開 40 本のうち届くのは 29 本。NN module と Head の interface・部品・登録口は内部。具象 Agent 4 本は公開だが外から 0。拡張点ごとに事情が違い、1 行で答えられない | 作る |
| 用語 3 語 | 010 と AGENTS.md が「公開ヘッダ」の語を使っているが、CONTEXT.md に無い。「anet-core の外」も定義が無い | 作る（§8 の 3） |
| 自己登録マクロの削除 | 呼び出し 2 か所がコメントアウトされたまま残り、P3 と逆の形を示している | 作る（§8 の 8） |
| 届かない公開ヘッダ 11 本の移動 | 「公開＝外で使う」に揃う利益だけで、動かすと設計文書のリンク 6〜25 本／ヘッダの張り替えになる。Agent を外で書く道を閉じる副作用もある | 据え置き（§8 の 4） |
| Agent の外部登録口 | anet-core の外に Agent を置く計画が無い（StampFly の PPO は中） | 先送り（§7 の 1） |
| NN module・Head の部品と登録口の公開 | 外で NoisyNet・SN・箱に参加する Head や module の要求が無い。独自 Head の実例はテストの 7 本だけ | 先送り（§7 の 2） |
| Observer・ReplayBuffer の差し替え口 | 要求が無い。Observer はコードで attach できる | 対象外 |

## 6. 受入基準

1. AGENTS.md の基本方針に P1〜P4 の段落があり、「公開ヘッダへの変更は慎重に行う」の行が無い。`grep -n "公開ヘッダ" AGENTS.md` で確かめる。
2. 010 §4.2 に拡張点の表があり、列は「拡張点／interface／部品／登録口／anet-core の外で書けるか／実例」で、§2.1 の 7 行がある。
3. 110・120・130 の §8 に、010 の表を指す 1 行がある。
4. 公開ヘッダが内部ヘッダを include していない。次の出力が空である。

   ```bash
   grep -rn "#include \"[a-z_]*_impl.hpp\"\|#include \"dqn_based\|#include \"nn_heads\|#include \"muzero_based\|#include \"muzero_rb\"" core/anet-core/include/anet/
   ```

5. `apps/` と `core/envs/` が `src/` を include していない。次の出力が空である。

   ```bash
   grep -rn "anet-core/src\|\.\./src/" apps/runner/src core/envs --include=*.cpp --include=*.hpp
   ```

6. 「書ける」と宣言した拡張点（Env、View）には、公開ヘッダだけで実装した実例（`core/envs/` の 6 本）が表に挙がっている。
7. `ANET_REGISTER_AGENT_FACTORY` が `core/` と `apps/` のどこにも無い（`grep -rn ANET_REGISTER_AGENT_FACTORY core apps` が空）。
8. 文書のリンク切れが 0（[docs/quality/prd.md](../quality/prd.md) の項番 4 と同じ検査を、変えた設計文書にも当てる）。
9. 配置を動かした場合は、全ビルドと全テストが緑（AGENTS.md §検証の形）。文書とマクロ削除だけなら、`anet-core` のビルドと `anet-core-test` の全実行で足りる。

成功の測り方: 次に「anet-core の外で X を書けるか」の問いが出たとき、010 の表の 1 行で答えが決まり、CMake・include・登録口の走査が要らないこと。今回の Head の問いでは、答えるまでにヘッダ 6 本、CMake 2 本、登録口 3 か所を読んだ。

## 7. 先送りした段階と、作る条件

1. Agent の外部登録口。作る条件: anet-core の外に Agent のターゲットを置く決定が出たとき（StampFly の PPO が外に出る、または 2 本目の Agent 置き場が要るとき）。形: `InitRL()` の固定の列挙はそのまま残し、外のライブラリは Env と同じく `Init<Agent>()` を公開して app が呼ぶ。内部の `ActionPolicy`・DQN 系 `Actor`／`Learner` は公開しない（外の Agent は rl.hpp の `Actor`／`Learner` を自分で実装する）。
2. NN module と Head の部品と登録口の公開。作る条件: anet-core の外で、NoisyNet・SN・箱に参加する Head か module を書くと決めたとき。形: `nn.hpp` に足さず、部品の作者向けの公開ヘッダを 1 本分ける。中身は `NetworkModuleExecutor`・`NetworkModule`・`LinearProjectionModule`・`PreparedModuleChildren<N>`・箱のアクセス・`WeightInitializer` と、module の登録口。Head の登録口（DefaultDQN の switch を registry にする）も同じ変更で足す。
3. 届かない公開ヘッダ 11 本の内部化。作る条件: 1 を「作らない」と決めたとき（Agent を外で書く道を閉じるなら、部品を公開しておく理由が無くなる）。

## 8. 決めること（暫定の既定）

| # | 決めること | 暫定の既定 | ほかの案 |
|---|---|---|---|
| 1 | 「anet-core の外」の範囲 | 同じリポジトリの別ターゲット | 別リポジトリも含める（配布と ABI が要るので非ゴール） |
| 2 | 規約の正本 | P1〜P4 は AGENTS.md 基本方針の 1 段落、表は 010 §4.2、各設計文書 §8 は表を指す 1 行 | 各設計文書の §8 に分散して書く。CONTEXT.md に書く（用語集に規約は置かない） |
| 3 | 用語 | CONTEXT.md に「公開ヘッダ」「内部ヘッダ」「anet-core の外」を足す。各 1〜2 文 | 足さない（AGENTS.md と 010 で通じる） |
| 4 | 届かない公開ヘッダ 11 本 | 据え置き | 具象 Agent 4 本（default_dqn、rainbow、image_cls、muzero_proto）だけ `src/` へ移す。Config 型を参照する文書とテストの張り替えが要る |
| 5 | Agent の外部登録口 | 先送り（§7 の 1） | 今回 `Init<Agent>()` の形だけ先に作る |
| 6 | NN 部品の公開 | 先送り（§7 の 2） | 今回 helper と `LinearProjectionModule` だけ公開する（P4 に反する） |
| 7 | 新しい公開ヘッダの名前（6 を作るとき） | 未定。`nn.hpp` の「公開 Config・Network・Builder」とは分ける | `nn_parts.hpp`、`nn_modules.hpp` |
| 8 | 自己登録マクロ | `ANET_REGISTER_AGENT_FACTORY` の定義とコメントアウトされた呼び出し 2 か所を削除。View 側の同型マクロ（gui.hpp）は使用箇所を確かめ、未使用なら同じく削除 | 残す（P3 と逆の形が残る） |
| 9 | Observer・ReplayBuffer | 対象外 | Observer の設定からの生成に registry を足す |
| 10 | 受入の検査の自動化 | しない。§6 の 2 コマンドを手順として書く | [906_cicd_improvement_10prd.md](906_cicd_improvement_10prd.md) の定点ジョブに乗せる |
| 11 | 各設計文書 §8 の 1 行 | 足す。110「Agent は anet-core の中に置き、`InitRL()` へ登録する」、120「Env は公開ヘッダだけで書ける」、130「module と Head の部品は内部ヘッダにあり、外では書けない」 | 010 の表だけにする |

用語（#3）の案:

- 公開ヘッダ: `core/anet-core/include/anet/` にあり、anet-core の外のターゲットが include できるヘッダ。
- 内部ヘッダ: `core/anet-core/src/` にあり、anet-core とそのテストだけが include できるヘッダ。
- anet-core の外: 同じリポジトリで anet-core 以外のターゲット。`core/envs/*` と `apps/*` を含む。

## 9. 依存と順序

- 文書とマクロ削除だけなら、084 のコミットの前後どちらでも行える。
- 084 の整理（`PreparedModuleChildren<N>` の `nn_impl.hpp` への移動）は本 PRD を待たない。P2 のとおりで、本 PRD は置き場を変えない。
- [908_config_root_ownership_alignment_10prd.md](908_config_root_ownership_alignment_10prd.md) と正本の置き方を揃える（設定の規約は CONTEXT.md か 020、コードの規約は AGENTS.md と 010）。
- StampFly の SimStampFlyEnv は Env の拡張点で、本 PRD の規約で書ける。PPO（PRD A）は anet-core の中に置く前提で、本 PRD の §7 の 1 とは独立に進められる。

## 10. 再確認ポイント

- 最終簡素化パスは経ていない。グリルで §5 の裁定と §8 を詰める。
- §2.2 の本数は include の文字列検索で数えたもので、推移的な到達は公開ヘッダ同士の include だけを辿った。
- 「静的ライブラリで自己登録が落ちる」は一般的なリンカの挙動と、コメントアウトされた呼び出しから推定した。削除の前に、ビルドの `/WHOLEARCHIVE` 相当の設定が無いことを CMake で確かめる。

## 11. 参照

- [AGENTS.md](../../AGENTS.md) の「基本方針」。
- [010 §4.2](../design/010_framework_overview.jp.md)、[110 §8](../design/110_agents_and_learning.jp.md)、[120 §8](../design/120_environments.jp.md)、[130 §4](../design/130_neural_networks.jp.md)。
- [core/anet-core/CMakeLists.txt](../../core/anet-core/CMakeLists.txt) の `target_include_directories`。
- [ownership_guideline.md](../ownership_guideline.md)（Agent の変数の所有。置き場の規約とは別）。
