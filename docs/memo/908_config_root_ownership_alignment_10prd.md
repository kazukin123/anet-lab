# 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) 暫定 PRD

> 2026-11-03 番号を 999 から 908 へ振り直した（PRD114）。

> 起点: 2026-09-16。`common.txt` の trainer 設定が `train.` root にあるが、読んでいるのは `RunManager::Config` であり、`train` は Run の中の 1 Runner 名でしかない。Run 名も `app.run_name` にあるが、Run の identity であって app の属性ではない。「設定の階層は実際のオブジェクト階層、または消費クラスの略称に合わせたい」という方針で、root 全部を同じ物差しで棚卸しする。
> 関連: [061](done/061_eval_slot_policy_override_10prd.md) P1(`train.` → `run.` 改名は決定済み。本 PRD はその根拠を規約へ一般化し、残りの root を同じ規約で棚卸しする)、[070](done/070_device_spec_auto_10prd.md)(device キー統一。`train.eval_device_type` を両 PRD が触る)、[done/059](done/059_config_concept_tree_alignment_10prd.md)(概念ツリー整合。上書き層 2 軸、mode の無印仕様化、RUN_BUDGET の app 配下決定)、[ADR 0030](../adr/0030-nn-config-tree-under-agent-prefix.md)(net 読み口を Agent 配下へ移した=所有が階層に現れる先例)、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)(スロットが所有する上書き層 / 参照するカタログ)、[CONTEXT.md](../../CONTEXT.md)「プロファイル」「カタログ」「上書き層」「Run プロファイル」「ワークスペース」。
> 本書は候補の棚卸しと規約案であり、裁定はグリルで行う。行番号は 2026-09-16 時点で、実装時は近傍のシンボル名・キー名で再検索する。

## 1. 問題

### 1.1 起点の 2 件

**`train.*` の読み手は RunManager である。** [trainer.cpp:765](../../core/anet-core/src/trainer.cpp) の `RunManager::Config` は `config_prefix = "train"` で `seed` / `num_envs` / `main_runner_type` / `eval_device_type` / `eval_device_index` を読み、同 ctor が `MakeSubConfigData("train.eval")` / `("train.eval_schedule")` で configured eval の定義とスケジュールを読む。コード自身に `/// @todo config_prefixをrunに変更` が 2 箇所残っている。`train` は BatchEnv name `train`(主 Train Runner)の名前であり、eval の定義まで `train.` 配下に入っているのは概念と合わない。

**`app.run_name` の実体は Run の identity である。** [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) の `RunnerApp::Config`(prefix `app`)が `run_name` を読んで `MetricsLoggerConfig::run_name_tmpl` へ転記し、[metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) の `MetricsLogger` ctor が `{t}` を展開して Run フォルダ名にする。Run プロファイルには次のように書かれており、Run の名前を app 経由で指定している形になっている。

```properties
run.@a5_base : app.run_name = run_{t}_a5_${E1.game}_base
```

### 1.2 一般化: root の命名に規約がない

現用 root を「読み口(消費 Config の default_prefix)」「所有者(生成・保持するオブジェクト)」「フレームワーク概念」で並べると、ずれは 2 件に留まらない。

| root | 読み口 | 所有者 | 概念(設計文書の章) | 判定 |
|---|---|---|---|---|
| `app` | `RunnerApp::Config("app")`。ほかに `app.show_error_dialog`([RunnerApp.cpp:337](../../apps/runner/src/RunnerApp.cpp))の直読みと、`app.eval_trace_panel.*` / `app.log_panel.*`（LogPanel が直読み。2026-10-31 に `app.log_level` を `app.log_panel.level` へ改名）| process | アプリ(160) | `run_name` が Run の identity(§3.2)。`metrics_logger.*` / `train_panel.*` / `eval_panel.*` / `eval_trace_panel.*` / `log_panel.*` は app が所有する部品の設定を入れ子にしており規約どおり |
| `app.online` / `app.batchrun` | 読み口なし。`app.$ = app.online > P1` で `app.*` へ複製 | – | 実行モード | 059 D25 で「選択の源」として仕様化済み |
| `train` | `RunManager::Config("train")` + `train.eval` / `train.eval_schedule` の sub config | RunnerApp | Run(100) | **root 名が概念(Run)と一致しない**。061 P1 で `run` へ(§3.1) |
| `env` | `BatchEnvBuilderConfig("env")`([env.hpp:350](../../core/anet-core/include/anet/env.hpp)) | RunManager | Env スロット(120) | `class_id` + 実行方式(worker)+ device。規約どおり |
| `<Env>`(`AtariEnv` 等) | `<Env>::Config(config_data, "AtariEnv", override_prefix)`。override は `train.eval.[tag].env` | BatchEnvBuilder | 具象 Env | 規約どおり(`class_id` の値 = 具象 prefix) |
| `agent` | `DefaultAgentFactoryConfig("agent")`([agent.hpp:330](../../core/anet-core/include/anet/agent.hpp)) | RunManager | Agent スロット(110) | 規約どおり |
| `<Agent>`(`DefaultDQNAgent` 等)、`<Agent>.net` | `DefaultDQNAgent::Config`、`NetworkConfig("<Agent>.net")` | DefaultAgentFactory | 具象 Agent と所有 NN(130) | 規約どおり(ADR 0030) |
| `net.block.[*]` / `net.branch.@*` / `net.body.@*` / `net.@qr` | `NetworkConfig` が `<Agent>.net` へ merge | – | NN の共有カタログ・プロファイル(130) | 規約どおり |
| `backend` | `BackendConfig("backend")`([init.hpp:26](../../core/anet-core/include/anet/init.hpp)) | RunnerApp(`InitRL`) | libtorch backend | 規約どおり |
| `metrics.scalar/graph/trace.[tag]`、`metrics.image.[tag]`、`metrics.image.interval` | [observers.cpp:1174-1177](../../core/anet-core/src/observers.cpp)、[image.cpp:693-694](../../core/anet-core/src/image.cpp) の literal prefix | RunManager(Observer 構築) | メトリクス定義(140) | 規約どおり |
| `image.thg.@…` / `image.phm.@…` / `image.shm.@…` / `image.@conv2d` | 読み口なし。`metrics.image.[tag].$ = image.phm.@per-prio > …` のチェーン元 | – | image provider のプロファイル | **所有者 `metrics.image` の外に別 root**(§3.6) |
| `ImageDataset.[key]` | [ImageData.hpp](../../core/envs/imagecls1/src/ImageData.hpp) | ImageDatasetManager(process) | データセットカタログ | 境界例。差し替わらない部品が class 名 root(§3.7) |
| `QValuePanel` / `Conv2dPanel` | `Config("QValuePanel")` / `Config("Conv2dPanel")`([QValuePanel.cpp:34](../../apps/runner/src/QValuePanel.cpp)、[Conv2dPanel.cpp:18](../../apps/runner/src/Conv2dPanel.cpp)) | RunnerFrame | GUI パネル(160) | **同じ GUI パネルでも `app.train_panel` / `app.eval_panel` は app 配下、こちらは root 直下の class 名。2 流儀が混在**(§3.5) |
| `A1` `A2` `A3` / `E1` / `M1` `M2` / `P1` | 読み口なし | – | 上書き層 | 059 D20 改で規約化済み |
| `@vars` | 読み口なし | – | 定数置き場(命名慣習) | 規約どおり |
| `run.$` / `run.@<name>` | resolver([config_impl.cpp:83-84](../../core/anet-core/src/config_impl.cpp) `kNamedTrunkOwner = "run"`) | – | Run プロファイル | `run.` root は既に Run の意味で使われている。素の `run.*` は通常キーとして通る(059 §2.2) |

Module Config の dump は prefix 名がそのままファイル名になる(`config/<prefix>.txt`。[metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) `Log(const Config&)` が `GetConfigPrefix()` を tag にする)。root を変えると dump のファイル名も変わる。

## 2. 規約案(裁定対象)

起点の問いは「オブジェクト階層か、消費クラスの略称か」である。2 案を単独で当てると次の問題が出る。

- **案 O(所有=オブジェクト階層をそのまま写す)**: 止め方がないと RunManager が所有する `env` / `agent` も `app.run.env.class_id` のように app 配下へ沈む。深くなるだけで概念は増えない。
- **案 C(消費 Config クラスの略称を root にする)**: 全読み口が root 直下に並ぶフラットな体系になる。`net` を Agent 配下へ移した ADR 0030 と、スロット(`train.eval.[tag]`)が `env.*` 上書き層を所有する ADR 0038 に逆行する。読み口が 2 つある設定(具象 Env の既定 prefix とスロットの override prefix)を表せない。

現に ADR 0030 / 0038 / 059 が採っている形は両者の折衷であり、それを規約として明文化する。

| # | 規約 | 根拠・既存の先例 |
|---|---|---|
| R1 | **root はフレームワークの主要概念で固定列挙する。** `app` / `run` / `env` / `agent` / `net` / `metrics` / `backend`、`class_id` で差し替わる具象の class 名、上書き層、`@vars`。root は設計文書の章(100 / 110 / 120 / 130 / 140 / 160)に対応し、C++ の所有チェーンを root まで遡らない(`app.run.…` にはしない) | 059 の 3 層モデル(カタログ / プロファイル / Run プロファイル)は root の集合を前提にしている |
| R2 | **root 内の階層は所有関係を写す。** `<Agent>.net`、`run.eval.[tag].env`、`app.eval_panel` | ADR 0030、ADR 0038 |
| R3 | **読み口(消費 Config の default_prefix)は所有階層のノードに一致させる。** 読み口が概念と別の root を指す状態(`train`)を作らない。読む C++ クラスが所有者と別でもよい(MetricsLogger の設定を app が読んで渡す等) | ADR 0030 の「読み口の移動」 |
| R4 | **root 名の綴り**: フレームワーク概念は小文字の役割名、`class_id` で差し替わる具象は class 名そのまま | 059「class_id の値 = 具象 config prefix」契約 |
| R5 | **プロファイル・カタログの置き場所は所有者 prefix 配下**(`AtariEnv.@v5_noop30`、`DefaultDQNAgent.@iqn`、`metrics.scalar.@baseline`) | 059 §4.1 |
| R6 | **root 変更は Module Config dump のファイル名変更を伴う**ので、改名は同一変更内で dump 名・inspect_run の期待値も移す | §1.2 末尾 |

R4 は「差し替わらない部品(GUI パネル、ImageDataset)が class 名 root を持つ」現状と衝突する。ここが Q3 / Q4 の裁定点になる。

## 3. 棚卸し結果と候補

### 3.1 `train.` → `run.`(061 P1 で決定済み。再掲と補足)

| 旧 | 新(061 §7.1) |
|---|---|
| `train.seed` | `run.seed` |
| `train.num_envs` | `run.train.num_envs` |
| `train.main_runner_type` | `run.train.runner_type` |
| `train.eval_device_type` / `_index` | `run.eval_device_type` / `_index` |
| `train.eval.[tag].*` | `run.eval.[tag].*` |
| `train.eval_schedule.[tag].*` | `run.eval_schedule.[tag].*` |

補足:

- `run.train.*` は Runner 名 `train` のスロットで、`run.eval.[tag].*` と対になる。061 P3 の `run.train.env_key` / `run.train.actor` も同じ場所に生まれる。`ImageClsEnv.train.*`(Env 内の Train ImageDataSource)は Env 配下の別階層なので衝突しない。
- 070 は `train.eval_device_type` / `_index` を文字列 1 キー `train.eval_device` にする(070 D1)。両方を通した最終綴りは `run.eval_device`。実装順で中間状態が変わるだけなので、順序は Q7 で決める。
- `run.` root は Run プロファイル(`run.$` / `run.@<name>`)と同居するが、resolver が特別扱いするのはその 2 形だけで、素の `run.*` は通常キーとして通る(061 §7.1、059 §2.2)。`run.@x : run.seed = 2` は完全キーの追記として今と同じ意味。
- `train.eval.[eval_panel].*`(EvalPanel が鏡写しする configured eval tag)も機械的に `run.eval.[eval_panel].*` になり、`app.eval_panel.eval_config_tag = eval_panel` の名前参照は root をまたぐ参照のまま成立する。

### 3.2 `app.run_name` → `run.name`(新規候補)

- **理由**: Run の identity であり、所有者は Run(R2)。Run プロファイルに書いたとき、`run.@a5_base : run.name = run_{t}_a5_${E1.game}_base` と Run 自身の属性として読める。CONTEXT「Run プロファイル」は `run.seed` を反復軸として除外しているが、`run.name` は逆にプロファイル固有の値なので含めて自然。
- **読み口の置き場(Q2-b)**: `MetricsLogger::Init` は `RunManager` より先に走るので、`RunManager::Config` に `name` を足すだけでは間に合わない。候補は (a) RunnerApp が `config_data.Get("run.name")` で直読みして `MetricsLoggerConfig` へ渡し、`RunManager::Config` も `name` を読んで dump に残す、(b) prefix `run` の小さな Config を app 側に置く。(b) は `RunManager::Config("run")` と prefix が重なり、`MetricsLogger::Log(const Config&)` が同じ tag `run` で 2 回 dump して `config/run.txt` が後勝ちになる。同一 prefix の Config 二重化は避ける。
- **影響(2026-09-16 実測、literal 出現数)**: config 29 行(8 ファイル。Run プロファイル行を含む)、bat 11 行(`apps/11_batch_run.bat` / `12_batch_run.bat` の `"app.run_name=…"` CLI。bat はランチャーなので上書き)、C++ は `RunnerApp::Config` の `run_name` 1 経路、テスト 17、Python 4([dropmerge_optuna.py:376](../../apps/runner/tools/dropmerge_optuna.py) の生成行、[check_default_leaves.py:94](../../core/anet-core/testdata/prd072/check_default_leaves.py) の optuna 例外登録)、docs 10([020](../design/020_user_guide_run.jp.md) §2.3、[optuna.md](../design/optuna.md)、[AGENTS.md](../../AGENTS.md)「Run 命名・後片付けルール」「実験記録と実効 config の保存ルール」)。
- **同乗(Q2-c)**: 061 P1 の `train.` → `run.` は 8 config ファイルと bat・docs を機械走査する。同じ走査に `app.run_name` を載せれば走査は 1 回で済む。

### 3.3 `app.runs_dir` は据え置き

- workspace 機構が注入し([app_util.cpp:532](../../core/anet-core/src/app_util.cpp))、`ValidateWorkspaceRunsDir` が注入値との完全一致を要求する。020 §2.3 は「設定や CLI からの変更は禁止」、CONTEXT「ワークスペース」の _Avoid_ は「runs_dir(出力先設定であって箱ではない)」。
- Run の置き場を決めるのは workspace(設定の外、059 D2)で、Run の属性ではない。`run.name` と対にして `run.dir` へ寄せると、Run プロファイルから書き換えられるように見える(禁止事項)。R2 でも app 側の出力先なので `app.runs_dir` のまま。

### 3.4 `app.metrics_logger.*` は据え置き、表記ずれだけ直す

- MetricsLogger は app が所有する process singleton で、R2 のとおり `app.` 配下が正しい。
- [metrics_logger.cpp:94-98](../../core/anet-core/src/metrics_logger.cpp) のエラー文言が root 抜きの `metrics_logger.video_codec` になっている。利用者が検索する実キーは `app.metrics_logger.video_codec`。
- [RunnerApp.cpp:111-127](../../apps/runner/src/RunnerApp.cpp) の `WarnDeprecatedRunsDirConfig` は旧キー `app.metrics_logger.runs_dir` / `root_dir` 専用の WARN tripwire。AGENTS.md クリーンブレーク方針(旧キー専用の WARN・tripwire は残さない)に反するので削除候補。

### 3.5 GUI パネル: `QValuePanel.*` / `Conv2dPanel.*` と `app.train_panel.*` / `app.eval_panel.*`

4 パネルとも RunnerFrame の子で差し替わらないが、置き場所が 2 流儀に割れている。

| 案 | 内容 | 得失 |
|---|---|---|
| **a** | `app.qvalue_panel.*` / `app.conv2d_panel.*` へ移し、`app.train_panel` / `app.eval_panel` と揃える | R2 / R4 に合う。`app.online.qvalue_panel.row_height = 24` のようにモード別の値も `app.$` の複製に乗る。影響は config 28 行(ほぼ `common.txt`)、C++ 2 クラスの prefix 文字列、docs 1 |
| b | `TrainPanel.*` / `EvalPanel.*` を root へ出し、class 名流儀に揃える | 現行の `app.online.eval_panel.fps` / `app.batchrun.eval_panel.fps` のようなモード別値が `app.$` の複製から外れ、mode 機構を失う |

推奨は a。2026-10-31 に足した LogPanel の設定は、この推奨に合わせて `app.log_panel.*` に置いた（`app.eval_trace_panel.*` と同じ形）。`Conv2dPanel.conv2d.*` の `conv2d` 段は `Conv2dVisualizerConfig` 構造体の入れ子で、image provider 側の `image.@conv2d.conv2d.*` と同じ形なので残す。

### 3.6 `image.` root → `metrics.image.@…`(R5 の適用)

- `image.thg.@per-prio` / `image.phm.@visit-maxq` / `image.shm.@qmax` / `image.@conv2d` は `metrics.image.[tag].$ = image.phm.@per-prio > image.phm.@s02` のチェーン元で、読み口を持たないプロファイルである。R5 に照らすと置き場所は所有者配下の `metrics.image.@…`。
- ただし現用の参照は [CartPole.txt:69](../../apps/runner/config/CartPole.txt) の 1 行だけで、DropMerge の参照は全てコメント。中間段 `thg` / `phm` / `shm`(provider type 別のグループ名)を残すかも含めて、image metrics の整理と同時に扱う(Q4)。

### 3.7 見送り(規約内と判定したもの)

- `env.*` / `agent.*`: スロット(`class_id` + 実行方式・device)。R1 により root は概念単位なので、RunManager 所有でも `run.env` にはしない。
- `app.online` / `app.batchrun`: 059 D25 で無印の「選択の源」として仕様化済み。
- `app.exp_pause_step` / `app.exp_exit_step` / `app.train_pause_step` / `app.train_exit_step` / `app.train_auto_start`: RunnerApp が Runner thread を pause / exit させる監督動作([RunnerApp.cpp:425-460](../../apps/runner/src/RunnerApp.cpp))。予算値は Run の性質にも見えるが、pause(online)か exit(batchrun)かは app の挙動であり、059 D11 / S1 が RUN_BUDGET を app 配下と裁定している。据え置き(蒸し返すなら Q5)。
- `run.eval.[tag].eval_batch_size` / `eval_episodes`: `eval.[tag]` 配下で `eval_` が冗長だが、[060](done/060_eval_batch_episodes_10prd.md) の決定で metrics・inspect_run のテスト参照が多い。`clone_model` は 061 P2 で Actor 設定へ移る。削るなら P1 の走査に同乗させる以外に安い機会がない(Q6)。
- `ImageDataset.[key]`: process 横断のカタログで、差し替わらない部品の class 名 root(R4 の境界例)。ImageCls 以外の利用者が出るまで据え置き。
- `@vars`、上書き層、`net.*` 共有定義、`backend`、`metrics.*`: 規約どおり。

### 3.8 Head の設定: `<Agent>.head.*` → `<Agent>.net.head.*`（候補）

[130](../design/130_neural_networks.jp.md) では Network は Body と Head をまとめた単位だが、Head の設定（init・projection・dueling、skip の action_input）は Agent が読み、factory を `BuildNetwork` に別引数で渡している（[dqn_based_agent.cpp:631](../../core/anet-core/src/dqn_based_agent.cpp)・688）。R2 なら Head は `net` の下、R3 なら読み口は NetworkConfig であるべきで、今は両方を満たしていない。[PRD098](098_temporl_skip_policy_10prd.md) の設定見直しでは Agent 直下の `head.*` に統一し（案 C）、`net.head.*` は取らなかった（2026-10-24）。

候補の形: NetworkConfig が `head` の子木を opaque に運び（`nn_forward.[key]` の先例、[200 §7](../design/200_dqn_agents.jp.md)）、Agent の HeadFactory だけが解釈する。読み口は NetworkConfig 1 つになり、`net.head.*`・`skip.net.head.*` と書ける。`quantile_mode` は Learner と方策にも効くので Agent 直下に残す。

着手条件: 案 C が入ったあとで、[999 nn_head_projection_separation](937_nn_head_projection_separation_10prd.md) か [999 outside-core §7-2](945_outside_core_extension_layout_10prd.md) の Head の継ぎ目を触るとき。設定だけ先に動かさない（読み口の変更と同じ変更で行う）。

## 4. 未決 Q

- **Q1 規約の採否**: §2 の R1〜R6。採るなら正本を CONTEXT.md(用語「設定 root」)と 020 §3 のどちらに置くか。AGENTS.md は「設定ファイルの代入演算子」と同じ粒度で 1 段落を足すか。
- **Q2 `run.name`**: (a) 採用するか。(b) 読み口の置き場(§3.2。同一 prefix の Config 二重化は dump が衝突する)。(c) 061 P1 と同じ変更に同乗させるか。
- **Q3 GUI パネル**: §3.5 の案 a / b。
- **Q4 `image.` root**: §3.6 を image metrics の整理まで遅らせるか、休眠中の今のうちに R5 へ揃えるか。
- **Q5 RUN_BUDGET**: 059 の app 配下裁定を維持するか。
- **Q6 `eval_batch_size` / `eval_episodes`**: `eval_` を削るか。P1 同乗のみ。
- **Q7 順序**: 070(device キー)と 061 P1 の順。先に 070 なら `train.eval_device` を経て P1 で `run.eval_device`。先に P1 なら 070 の D1 / D6 の表記を `run.` で書き直す。
- **Q8 Head の設定の置き場**: §3.8 の候補（NetworkConfig が `head` を opaque に運ぶ形で `net.head.*`）を採るか。採るなら時期（Head の継ぎ目を触る PRD に同乗させる）。

## 5. 着手条件

（2026-10-31 追記: 061 P1 は実装済みのため、下の「P1 の走査に同乗する」段取りはもう使えない。`app.run_name` と GUI パネル（案 a）は、裁定後に単独の変更として行う。）

単独では着手しない。061 P1 の実装依頼を出す前にグリルで Q1〜Q7 を裁定し、`train.` → `run.` の機械走査と同じ変更で `app.run_name`(採用時)と GUI パネル(案 a 採用時)を処理する。R1〜R6 を採る場合は、061 P1 の受入(全テスト緑、smoke Run の `config_data.txt` がキー名置換を除いて一致)に「dump ファイル名(`config/run.txt` 等)と inspect_run の期待値が新 root に揃っている」を足す。

## 付録: 棚卸し中に見つけた表記ずれ(本 PRD の対象外)

- `RunnerApp::Config::run_name` の C++ 既定値が `"run_{%t}"`([RunnerApp.cpp:56](../../apps/runner/src/RunnerApp.cpp))。`MetricsLogger::CreateRunName` は `{t}` しか展開しない。`common.txt` が `app.run_name ?= run_{t}` を常に供給するため露見していない。
- `Conv2dPanel.conv2d.mergin_x` / `mergin_y`([common.txt](../../apps/runner/config/common.txt)、[metrics_image.txt](../../apps/runner/config/metrics_image.txt) の `image.@conv2d.conv2d.mergin_*`)に対し、読み口は `conv2d.margin_x` / `margin_y`([Conv2dPanel.cpp:19-20](../../apps/runner/src/Conv2dPanel.cpp))。設定行は NoCare で、C++ 既定値が使われている。
- [dropmerge_optuna.py:375](../../apps/runner/tools/dropmerge_optuna.py) が生成する `app.$ = app.batchrun > P` は、059 の上書き層 rename(`P` → `P1`)から取り残されている。未定義 prefix は no-op なので実害はない。
- `metrics_logger.cpp` のエラー文言の root 抜き(§3.4)。
