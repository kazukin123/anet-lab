# PRD117: WARN の意味の線引きと、読まれない設定の検出 PRD

> 2026-11-07 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、`default_dqn_agent.cpp`・`default_dqn_agent.hpp`・`intrinsic_rnd.cpp`・`nn.txt` は PRD116 の未コミットの変更を含む作業ツリーの位置である。実装時は近傍の記述で探し直す。
> 起点: 2026-11-07 の会話。ユーザーは [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md)（`net.block` の未知の葉が黙って無視される）と [issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md)（Atari の正常な Run で毎回 WARN が出る）を題材に、「出なくてよい WARN は出ず、出るべき WARN が出るように全体を見直したい。使われない設定はこれまで NoCare だったが、UI の工夫で WARN に実際の意味を持たせられるようになった今、この点も見直す時期だ」として、調査と方針案の整理、暫定 PRD の起票を求めた。UI の工夫は、Logs pane の未確認の警告の件数と、Panels toolbar の Logs の印である（[160](../design/160_applications_and_tools.jp.md) §3.5）。
> 関連: [PRD052](done/052_eval_schedule_separation_10prd.md)（dormant tag の WARN、旧キーの NoCare）、[PRD059](done/059_config_concept_tree_alignment_10prd.md)（カタログ・プロファイル・上書き層）、[PRD072](done/072_config_selection_final_value_10prd.md)（既定葉 `?=`）、[PRD081](081_evalpanel_episode_trace_10prd.md)（dormant tag の WARN を scalar の参照に限る）、[PRD101](done/101_exploration_schedule_profiled_value_10prd.md)（眠る bare の検査の先送り）、[PRD116](116_rnd_lifelong_novelty_10prd.md)（RND の Network が意図と別の形で組まれた件）。

## 1. 問題

### 1.1 WARN が正常な Run で出続け、意味を失っている

Logs pane は表示した WARN 以上の記録を「未確認の警告」として数え、Panels toolbar の Logs に印と件数を出す（[160](../design/160_applications_and_tools.jp.md) §3.5 の 153 行）。ところが、意図どおりに組んだ Atari の Run は毎回 WARN を 1 件出す（§2.1）。印が常に付けば、利用者は印を見なくなり、本物の WARN も読み飛ばされる。

### 1.2 書いた設定が読まれなくても、黙って通る

設定基盤は、書かれた葉が読まれたかを見ない（§2.4）。[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針も、旧キーについて「設定基盤が未知キーを一律検出しない場合は NoCare」としている（81・89 行）。2026-11-06〜07 の Atari の RND の Run は、block の直下に書いた `out_channels` などが読まれず、1 Network 105,006,592 パラメータ（意図は 854,240）で動いた（[PRD116](116_rnd_lifelong_novelty_10prd.md) §2.1）。設定の綴り違い（`mergin_x`、[issue-20261103-66100f](../management/issues/issue-20261103-66100f.md)）も同じく黙って通っている。

### 1.3 WARN を出す基準が 1 つにまとまっていない

[AGENTS.md](../../AGENTS.md) の Fail-Fast の表は、意図された休止に「必要なら対象ごと 1 回だけ WARN」を許し（187 行）、汎用機構の節は書き忘れの可能性に「必要に応じて一度だけ WARN してよい」とする（156 行）。意図した状態と、意図と違う可能性のある状態が、同じ WARN で出る。WARN だけで流している未知の値や、止めるべき失敗も残っている（§2.3）。

## 2. 現状（2026-11-07 に確認したこと）

### 2.1 実際の Run の WARN

`apps/runner/workspaces/*/runs/run_*` の Run のログ 616 本から `[W]` の行を集め、数値と引用符の中を伏せて種類ごとに数えた（集計の手順は §7 の A2 と同じ見方）。

| 文言（要点） | 出た Run 数 | 出た期間 | 今のコード |
|---|---|---|---|
| `ActionPolicy spatial log scale received non-positive boundary value. Clamping to 1e-4 …` | 213 | 2026-07-27〜11-02 | 今のコードに無い（ProfiledValue 化で消えた） |
| `Skipping metrics for unscheduled eval tag. tag='eval_target'. …` | 145 | 2026-09-17〜11-07 | [trainer.cpp](../../core/anet-core/src/trainer.cpp) 866 行 |
| `ActionPolicy spatial exploration is enabled with num_envs < 32. num_envs=1` | 79 | 2026-09-18〜11-07 | [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) 1525 行 |
| `net.config_profile.[…] is defined but not used by any branch.` | 17 | 2026-07-03〜07-08 | [nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) 1847 行 |
| `Skipping metrics for unscheduled eval tag. tag='eval1'. … train.eval …` | 6 | 2026-08-27 | 旧文言（`train.eval` は `run.eval` へ移った） |
| `Learner::Load: optimizer option differs from checkpoint; config applied. …` | 6 | 2026-10-16〜10-22 | dqn_based_agent.cpp 4270〜4295 行 |
| `Eval session has fewer adopted episodes than episode groups. …` | 5 | 2026-09-22 | trainer.cpp 1150 行 |
| `DefaultDQNAgent.intrinsic.learn_batch_size=32 exceeds num_envs=1; …` | 1 | 2026-11-07 | [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 110 行 |
| `DefaultDQNAgent.intrinsic.enabled=true with learner.enabled=false: …` | 1 | 2026-11-07 | default_dqn_agent.cpp 119 行 |
| `MetricsLogObserverBase::OnUpdate(): value not found. tag=56_agent_in/32_learn_count …` | 1 | 2026-11-07 | [observers.cpp](../../core/anet-core/src/observers.cpp) 1115 行 |
| `Checkpoint has no skip networks; …` | 1 | 2026-10-23 | default_dqn_agent.cpp 440 行 |

各 workspace の直近 3 本では、Atari の Run（Atari-04・Atari5-01・AtariRnd-01・AtariTube-01）が全部 `eval_target` の WARN を 1 件ずつ出し、LunarLander・GridMaze・DropMerge-01 の直近の Run は 0 件だった。Atari の `eval_target` は `run.@eval2ch_r1` が `interval = 0` で意図して止めており（[Atari.txt](../../apps/runner/config/Atari.txt) 517 行）、WARN は PRD052 の D9 と PRD081（92 行）の契約どおりに出ている。評価だけの Run（`mz_nm_rnd_ev_taurnd`）の 4 件の分析は [issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md) の事象にある。

### 2.2 WARN の出し方と表示

- ログは [log.hpp](../../core/anet-core/include/anet/log.hpp) の `anet::log::warn()`（216 行）と、prefix 付きの `Logger::warn()`（196 行）で出す。レベルは info・verbose・warn・error・fatal の 5 つで、WARN と ERROR の間のレベルは無い。Run のログには `[W]` で残る。
- 「対象ごとに 1 回だけ出す」共通の部品は無い。メンバの真偽値（`grad_noise_intervals_warned_`）、メンバの集合（`warned_dormant_metric_tags_`、`policy_churn_warned_intervals_`、`warned_unsupported_env_config_names_`）、状態での固定（`ImageData.cpp` の `cache_prepare_state_`）、構築が 1 回であることへの依存（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) 482 行のコメントは「1度だけWARN」だがフラグは無い）が、箇所ごとに書かれている。
- Logs pane は表示 level 以上の WARN を数え、▾の `Mark all as viewed` で確認済みにする。表示しない level の記録と Clear した記録は数えない（160 §3.5）。

### 2.3 WARN を出す箇所の棚卸し

`core/` と `apps/` の `.cpp`・`.hpp`（`third_party/`、テスト、ログ基盤自身、コメントアウトした行を除く）で WARN を出す箇所は 62 行ある。区分ごとの件数は次のとおりで、全件と扱いの既定は §5.5 の表にある。

| 区分 | 件数 | 例 |
|---|---|---|
| A 書いた設定が効かない・使われない定義・ほかの設定で休止 | 19 | `intrinsic_rnd.cpp` 258 行の `multiplier_max` の無視、trainer.cpp 866 行の dormant tag |
| B 設定の組み合わせや値が推奨外 | 14 | dqn_based_agent.cpp 1525 行の spatial と num_envs |
| C 自動の代替・置き換え | 8 | default_dqn_agent.cpp 110 行の `learn_batch_size` の切り詰め |
| D 実行時のデータや状態の異常 | 18 | observers.cpp 1115 行の値の欠け |
| E 旧キーの案内 | 1 | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) 131〜145 行の `app.metrics_logger.runs_dir`（[issue-20261103-2c9a07](../management/issues/issue-20261103-2c9a07.md) で起票済み） |
| F そのほか | 2 | LogPanel のクリップボードの失敗、Env の設定 dump の非対応 |

[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則・設定値の扱いと食い違う箇所:

- 未知の値を WARN だけで流す: [scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp) 420 行（`Unknown squash_type.`。正規化を呼ぶたびに出る。DefaultDQNAgent の観測の正規化は `post_process_type` の値域を構築時に検証していない）と、observers.cpp 1374・1379・1384・1404・1633・1640 行（metrics の定義の未知の event・step・target・属性）。表は未知の列挙値と閉じたスキーマ内の未知フィールドを fail-fast とする（186 行）。
- 互換性の無い組み合わせを読み替える: [rainbow_agent.cpp](../../core/anet-core/src/rainbow_agent.cpp) 54 行（`use_qr=true` かつ `num_quantiles<=1` を scalar DQN として扱う）。
- 明示要求の一部を別の方法で代える: [replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) 1213 行（PER の rejection sampling が埋まらないと残りを一様に引く。固定も回数の抑えも無い）。
- 指定値を切り詰める: default_dqn_agent.cpp 110 行（`learn_batch_size` を num_envs に切り詰める。既定の 32 のままでも num_envs<32 なら出る）。
- auto の代替で選択を固定しない: [metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) 327 行（nvenc が起動しないと動画ごとに libx264 で試し直し、WARN も動画ごとに出る）。
- 明示した入力が無くても続ける: [config.cpp](../../core/anet-core/src/config.cpp) 555 行（ファイルが開けない）・640 行（`$include` 先が無い）は WARN で続け、605・626・632 行（include 行の解析の失敗）は ERROR のログを出して次の行へ進む。[Run 実行ガイド](../design/020_user_guide_run.jp.md) の 377 行はこれを「エラーにならず続行する」と案内している。
- 重複を除かずに繰り返す: observers.cpp 91・1115 行、[probe.cpp](../../core/anet-core/src/probe.cpp) 251 行、scaler_impl.cpp 541 行（`// DEBUG` のブロックで生の観測を出す）、replay_buffer_impl.cpp 1213 行。NN を構築・複製するたびに出る nn_impl.cpp 1838・1847・1923 行と [nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) 1676 行も、target・Actor 用の複製の数だけ重なる。
- 旧キー専用の WARN: RunnerApp.cpp の `WarnDeprecatedRunsDirConfig`。クリーンブレーク方針（78・88 行）に反する。

INFO で休止や選択の結果を知らせている箇所も既にある（trainer.cpp 1095・1107 行の `eval.[<tag>]: definition-only`、dqn_based_agent.cpp 2684 行の `Learner: learning is disabled …`、metrics_logger.cpp 313 行の `VideoLogger(auto): …`）。

### 2.4 設定の読み方と、読まれない葉

- `ConfigData` は文字列の key/value の平らな集まりで、`Read` はキーがあればその値を、無ければ呼び出し側の既定を使う（[config.hpp](../../core/anet-core/include/anet/config.hpp) 95〜105 行）。どの葉が読まれたかは残さない。
- `Config` の `ReadConfig` は default prefix と override prefix の順に読み、読んだ値をモジュールごとの記録へ残す（config.hpp 184〜209 行）。この記録が Run の `config/<module>.txt` になる。ただし記録は「読んだ葉」ではない。ProfiledValue は全欄を記録せずに読み、実効の type が使う欄だけを記録する（[schedule.hpp](../../core/anet-core/include/anet/schedule.hpp) 284〜349 行）。
- 読み口は、`Config` の派生が 75、`ReadConfig`・`ANET_READ_CONFIG` の呼び出しが 610 ある。ほかに `ConfigData` の全キーを走査する読み口が 29 か所（`net.block` などのカタログの読み口）、`ConfigData` を直接 `Get`・`Read` する読み口がコアに 16 か所、アプリに約 10 か所（[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp) 84・85 行など）ある。
- NN の block は、`net.block.[<名前>].*` を全体とAgent の下から集め、`type` を除いた葉を相対のキーで block ごとの設定にまとめる（nn_impl.cpp 787〜845 行）。構造に現れた block だけを、module の factory がその設定から自分のキーを読んで作る（nn_impl.cpp 1776〜1806 行。Conv2d は `conv.out_channels` などを読む。nn_modules.cpp 2969 行・3050 行付近）。factory が読まない葉は、構築時にエラーにも WARN にもならない（0c34dd）。
- 実際の Run（`mz_rbw_imp_s1`）の `config_data.txt` は 1,362 葉、モジュールごとの記録は 671 葉で、差の 1,009 葉の内訳は `net.block` 447、`metrics.scalar` 123、`M1` 110、`RainbowAgent` 57、`A3` 56、`MuZeroAgent` 44、`A1` 35、`DefaultDQNAgent.actor` 17、`app.eval_trace_panel` 15、`DefaultDQNAgent.net` 14、`run.eval` 14、`A2` 12、`DefaultDQNAgent.intrinsic` 9 などだった。この差には、記録に残らない読み方（NN の branch、ProfiledValue の type、GUI の panel）で読まれた葉も含まれるので、記録は読まれたかの判定に使えない。
- 実効側に残る、読まれなくてよい葉の種類は既に整理されている。プロファイル（`@` を持つ定義）は `ConfigData` から除かれる（[100](../design/100_runtime_and_configuration.jp.md) §2.1 の手順 7）。上書き層（A1・E1・M1 など）と mode（`app.online`・`app.batchrun`）は選択の源として残り、「既知 prefix として機械識別でき」る（[PRD059](done/059_config_concept_tree_alignment_10prd.md) の 36 行、D20・D25）。カタログの項目は名前で参照される部品の定義である（[CONTEXT.md](../../CONTEXT.md)「カタログ」）。共通の設定には、その Run で作らない Agent の既定（`RainbowAgent.*` は `?=` が 51 行、`=` が 6 行）も入る。
- 既定葉 `?=` の集合は `Properties`・`ConfigManager` が解決の途中で持つが、`ConfigData` には渡さない（config.hpp 154〜160 行、config.cpp 675〜803 行）。
- 遅れて設定を読む読み口がある。`Conv2dPanel` は View の menu で開いたときに作られ、そこで `Conv2dPanel.conv2d.*` を読む（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) 1519 行、[Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp) 20〜30 行）。
- 綴り違いの `Conv2dPanel.conv2d.mergin_x ?= 2` は既定葉で書かれている（[common.txt](../../apps/runner/config/common.txt) 75・76 行。[metrics_image.txt](../../apps/runner/config/metrics_image.txt) 96・97 行も同じ）。`=` だけを見る検出では拾えない。
- 静的な検査は `?=` と `=` の使い分けを見る [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) と、`[shipped_config]` のテストだけで、書いた葉が読まれたかは見ない。
- 2026-11-07 時点の `nn.txt` は、`Conv1d_S`・`Conv1d_M`・`Conv2d_S` などの葉を `conv.*` の下に書いている（[nn.txt](../../apps/runner/config/nn.txt) 123〜139 行。PRD116 の実装の作業ツリー）。

### 2.5 規約・文書・過去の判断の今の記述

- [AGENTS.md](../../AGENTS.md): 81・89 行（未知キーを一律検出しない場合は NoCare）、156〜157 行（書き忘れの可能性は一度だけ WARN してよい）、184〜196 行（Fail-Fast の表。意図された休止は「必要なら対象ごと 1 回だけ WARN」）。
- [CONTEXT.md](../../CONTEXT.md) の「dormant」: dormant タグを参照する scalar は「タグごと 1 回の WARN で skip」。
- [100](../design/100_runtime_and_configuration.jp.md) §5.1（169 行）と §6.1（208 行）: dormant tag の一度だけの WARN。
- [130](../design/130_neural_networks.jp.md) 287 行: どの bind にも使われない入力は key ごとに WARN。
- [140](../design/140_observability.jp.md) 232 行: 不明な event・step 軸・target は WARN のあと既定値を使う。
- [020](../design/020_user_guide_run.jp.md) 377 行: include 先が無くても続ける。
- [実装 品質確認観点](../quality/impl.md) 項番 24・25・27: 意図された休止をエラーにしない、代替は auto だけで 1 回 WARN、WARN は対象ごとに 1 回。
- ADR: [0027](../adr/0027-eval-definition-schedule-separation.md) の Consequences は dormant タグ参照の WARN の文言を定め、[0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) は「scalar は現行どおり dormant タグでは WARN して捨てる」とする。[0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) は checkpoint の optimizer の options を今の設定で置き換える判断で、その中で「checkpoint と違った項目は WARN で示す」と決めている（3 行）。
- 地図の「選ばなかった形」（[316](../wiki/categories/316_config.md)、[324](../wiki/categories/324_dqn.md)）:
  - 個別の葉と継承値の競合の WARN — 不採用（PRD072）。本 PRD は競合を見ないので開き直さない。
  - 旧キー（`use_qr`・`num_quantiles`）の tripwire — 不採用（PRD001 の D14）。本 PRD は旧キー専用の検出を作らない（D6）。
  - DQN 側で Noisy の有無を見て効かない倍率に WARN を出す形 — 不採用（PRD107、ADR 0048）。本 PRD の一覧は「読まれたか」だけを見るので、読まれて効かない葉には出さず、開き直さない。
  - 眠る bare の実行時・静的検査 — 先送り（PRD101 §7）。再開条件は「眠った値に気づかずに Run を回した事故が実際に起きたとき」。今回の RND の件は「読まれない葉」で、眠る bare（読まれて効かない値）とは別だが、近い事故である。扱いは §9 の 11。
- [317](../wiki/categories/317_metrics.md)・[318](../wiki/categories/318_rl_core.md) の「変えるときの条件」の「WARN・fail-fast の文言と時機を変えない」（PRD094）は、読み口と配線の整理のときの条件である。本 PRD は WARN のレベルと時機を意図して変えるので、変える箇所を §5.5 に列挙する。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: WARN の意味を 1 つにする。WARN は「利用者が確かめて、設定・コード・実行環境を直すべき状態」に限り、意図した休止と、設定どおりに動いた結果の通知は INFO にする。
- G2: 正常な Run で WARN が 0 件になる。Logs の印が付けば、確かめるべきことがあると読めるようにする。
- G3: 書いた設定が読まれないことを検出する。実体化したカタログ項目の読まれない葉は構築時に止め、それ以外の読まれない葉は Run の構築の終わりに WARN で一覧する。NoCare をやめる。
- G4: 出るべきなのに出ていない失敗を正す。未知の値と明示した入力の不在を、WARN で流さずに止める。
- G5: 題材の issue 2 件（0c34dd・a9a0e3）が片付く。

### 3.2 非ゴール

| 退けた案・範囲 | 理由 |
|---|---|
| ログ基盤の作り替え（structured・JSON logging、thread-local の文脈） | 別の関心（[PRD038](done/038_env_log_auto_prefix_10prd.md) が範囲外とした）。WARN の意味と検出は今の基盤で表せる。本当の制約ではなく、別 PRD で選べる |
| Logs pane・toolbar の表示の変更 | 既にある表示（160 §3.5）に意味を持たせるのが本 PRD である。表示を変える必要は見つかっていない。本当の制約ではなく、必要が出たら別 PRD で変えられる |
| 設定の値が指す名前の検証（`stack_keys` の名前、[issue-20261103-b754ff](../management/issues/issue-20261103-b754ff.md)） | 別の関心。値の参照先の検証で、葉が読まれたかとは別の仕組みになる。本当の制約ではなく、その issue で扱える |
| WARN と ERROR の間に新しいレベルを足す | レベルを増やすと G1 の線引きがぼやける。必要が出たら別 PRD で選べる（本当の制約ではない） |
| 過去の Run のログの読み替え | 本当の制約である。クリーンブレーク方針（過去の Run artifact は当時の記録として残し、読み替えのコードを足さない） |
| ERROR を出して続けている箇所のうち、設定の読み込みと metrics の定義以外（trainer.cpp 1020・1042 行、DropMergeEnv.cpp 900 行） | 別の関心。設定の失敗の扱いに範囲を限る。本当の制約ではなく、別の課題として起票した（§9 の 16） |

### 3.3 成功の測り方

- M1: §7 の A2 の Run（出荷設定の各 Env の既定の `run.$` と、番号付き bat の Run の組み合わせ）のログで、`[W]` の行が 0 件。
- M2: 題材の事例（block の直下の `out_channels`、評価だけの RND の Run）と綴り違い（`mergin_x`）を再現する設定で、§5 のとおり ERROR・WARN・INFO になる。
- M3: §5.5 の表の全行が、表の扱いに移っている（表の行ごとのテストか検索で確かめる）。

### 3.4 用語（暫定）

本 PRD は次の 2 つの語を使う。グリルで確定したら [CONTEXT.md](../../CONTEXT.md) へ足すか決める（§9 の 13）。

- **読まれない葉**: Run の実効設定に残った葉のうち、どの読み口も読まなかったもの。「未知キー」（AGENTS.md の語）は現行契約に無いキーを指すが、契約にあるキーでも持ち主が作られなければ読まれないので、別の語にする。
- **在庫**: 読まれなくても誤りではない葉（D4）。[100](../design/100_runtime_and_configuration.jp.md) §2.1 の手順 6 が未選択のプロファイルを「休止した在庫」と呼ぶのを、選択の源・未参照の共有のカタログ項目・作らなかったクラスの設定・休止した prefix へ広げる。「dormant」は評価タグの状態の語なので使わない。

## 4. 確定事項（暫定の既定。会話で合意した決定ではない。§9 とあわせてグリルで見直す）

### D1. ログのレベルの線引き

| # | 状態 | レベル |
|---|---|---|
| 1 | 次の (a)〜(d) のどれか。(a) 休止を宣言する設定（`enabled = false`、`learner.enabled = false`、`interval = 0`、schedule の記述が無い定義だけの評価タグなど）による休止が起きた。(b) checkpoint を読み込むとき、今の設定に合わせて checkpoint の値を置き換えた、または checkpoint に無い部分を初期化した（ADR 0053、[200](../design/200_dqn_agents.jp.md) §8.2 の契約）。(c) 明示した設定が、設定の外の値（環境変数）に勝った。(d) 選んだモードの定義そのものを説明する通知で、書いた値を捨てず、ほかの設定の効き目も弱めないもの | INFO。対象ごとに 1 回 |
| 2 | Run の構築が終わるまで（設定の読み込み・部品の構築・checkpoint の読み込み）に判定する、現行契約に反する設定・Run の入力・部品の呼び出し（未知の列挙値、閉じたスキーマ内の未知の葉、互換性の無い組み合わせ、明示した入力の不在、部品の契約に反する呼び出し）と、明示要求の不履行・非対応。今 `ANET_SYSTEM_ERROR` で止めている実行時の検査（Tensor の形など） | ERROR（`ANET_SYSTEM_ERROR` で止める） |
| 3 | 次の (a)〜(g) のどれか。(a) 読まれない葉（D3(b)）。(b) 書いた値が、ほかの設定（モードなど）のために使われない。(c) 動くが、ほかの設定や Run の形との組み合わせで、ある設定の効き目が無くなるか弱まる構成、書き忘れの可能性がある構成、または支えない組み合わせ（worker thread との併用など）かどうかを部品が判定できず、利用者に確かめさせる構成。(d) auto 系の代替。(e) Run の構築のあとに起きて止めずに続ける異常（メトリクスの値の欠け、値の長さの不一致、episode の乱れによる描画の打ち切り、終了時の保存が 0 byte、GUI の配置の修復、Replay の抽選の代替）。(f) 補助の操作・ファイル・出力先の失敗。(g) 実装の欠けで機能の一部を省いた | WARN。対象ごとに 1 回 |
| 4 | 1〜3 のどれにも当たらない状態 | INFO（知らせる必要が無ければ出さない）。開発用の診断は `ANET_LOG_DEBUG` |

行の当て方: 上から順に当て、最初に当たった行のレベルで出す。1 と 3 は括弧ではなく (a)〜の列挙が条件で、列挙に無い状態は当たらない。

- 「Run の入力」は Run の結果に効く入力（設定、データセット、checkpoint、ROM など）を指す。workspace の履歴のような補助のファイルは含めない。
- 2 の「判定する」は、今構築時に判定しているものと、本 PRD が構築時の判定へ移すもの（D7、§5.5 の「ERROR へ」の行）を指す。判定する作りが無いもの（終了時の保存で Agent が Save を持たないことなど）は、構築のあとに起きる 3 の (e) に当たる。
- 2 の実行時の部分は、今 ERROR で止めている検査に限る。本 PRD は実行時の検査の ERROR を足しも減らしもしない。
- 1 の (d) と 3 の (b)・(c) の境目は、「書いた値を捨てるか、ほかの設定の効き目を弱めるか」である。`feature_gradient = stop` の説明は、選んだモードの定義（skip の損失の勾配を共有部へ流さない）を述べるだけなので 1 の (d)。`max_length` を `max_length_mode = full` で書いた場合は値を捨てるので 3 の (b)。spatial のラダーを lane が 32 未満の Run で使う場合はラダーの効き目が弱まるので 3 の (c)。
- 在庫（D4）の 1〜3 に当たる定義が使われなかったことは、どのレベルでもログに出さない（§5.6 の 2 行目）。D4 の 4 の休止の宣言は、持ち主が 1 の (a) の INFO で出す。

理由: Logs の未確認の警告は WARN 以上を数える（§2.2）ので、WARN を「手を打つべき状態」に限れば印に意味が出る（G1・G2）。1 を先に当てるのは、列挙した (a)〜(d) は利用者の選択どおりに起きたことで、ほかの行の条件（置き換え・初期化）に見えても手を打つことが無いため。2 を構築時に限るのは、[AGENTS.md](../../AGENTS.md) の「起動時・構築時に検証できるものを実行時まで潜伏させない」（193 行）に合わせ、構築のあとの出力の側の異常で Run を落とさないため。4 を INFO にするのは、行に当たらない状態を WARN にすると、WARN の意味がまた広がるため。依存: なし。

今の規約との差: [AGENTS.md](../../AGENTS.md) の Fail-Fast の表（186 行）は「契約違反の内部状態」と「壊れた入力データ」を一律に fail-fast とするが、今の作りは、構築のあとの GUI の配置の修復（PRD110）や描画の打ち切りを WARN で続け、workspace の履歴の壊れも WARN で続けている。D1 はこの作りを 2 と 3 の境目として書き、表の 186 行をこの条件へ改める（§5.8）。表の 187 行（意図された休止）は 1 の INFO へ改める。

既存の判断との食い違い: 1 の「checkpoint の値を今の設定で置き換えた」を INFO にするのは、[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) の「checkpoint と違った項目は WARN で示す」のレベルの部分を改める（options を今の設定で置き換える判断は残る）。該当は §5.5 の dqn_based_agent.cpp 4270〜4295 行と intrinsic_rnd.cpp 498 行。確定したら ADR 0053 の先頭へ注記する（§5.8。§9 の 10）。

### D2. 正常な Run の WARN を 0 件に保つ

§7 の A2 の Run で WARN が 0 件であることを、本 PRD の受入にする。あとの PRD が WARN を足すときも、同じ Run で出ないことを確かめる（§5.8 の移行で、PRD 品質確認観点か実装品質確認観点への 1 項目の追加を候補にする。§9 の 18）。

理由: 正常な Run で 1 件でも出れば、印が常に付き、D1 の線引きが効かない（G2）。依存: D1。

### D3. 読まれない葉を 2 つの仕組みで検出する

- (a) カタログ項目の閉じたスキーマ: カタログの項目を実体化する読み口は、その項目の葉のうち読まなかったものがあれば、構築時に `ANET_SYSTEM_ERROR` で止める。葉が `?=` か `=` かによらない。対象のカタログは §5.3 の表。実体化しなかった項目は検査しない。その葉は、共有のカタログなら D4 の 2 の在庫になり、1 つの持ち主の下だけで参照される定義（branch の config_profile）なら (b) の一覧に出る。
- (b) Run の読まれない葉の一覧: Run の構築を終えた時点で、実効設定の葉のうち、どの読み口も読まず、在庫（D4）にも当たらない葉を、1 件の WARN に一覧して出す（§5.4）。

理由: (a) の項目の葉はその読み口だけが読むので、局所の情報だけで判定でき、[AGENTS.md](../../AGENTS.md) の表の「閉じたスキーマ内の未知フィールド」に当たる（汎用機構と利用側の責任境界）。0c34dd を直す（G3・G5）。(b) の綴り違い・階層違い・クリーンブレーク後の旧キーは、持ち主を名前から決められない（`mergin_x` の上の `Conv2dPanel.conv2d` は読み口があるが、`Conv2dPanel.conv2d.mergin_x` を誰が読むはずかは書いた人にしか分からない）ので、全体の差でしか拾えない。全体は、在庫との区別が名前の形に頼れない（D4）ので、止めずに WARN にする（§9 の 4）。依存: D4。

### D4. 在庫（読まれなくても一覧に出さない葉）

次の表を上から順に当て、最初に当たった行で在庫とする。どの行にも当たらず、どの読み口も読まなかった葉が D3(b) の一覧に出る。

| # | 在庫 | 例 |
|---|---|---|
| 1 | 選択の源として参照された通常 prefix の葉（解決の記録の選択に現れたもの） | 上書き層 `A1.*`・`M1.*`、mode の `app.online.*`・`app.batchrun.*` |
| 2 | 共有のカタログ（どこからでも名前で参照できるもの）を走査した読み口が、参照されなかった項目として見送った葉。1 つの持ち主の下だけで参照される定義（`net.branch.[<b>].config_profile.[<名前>]` はその branch だけが参照する）の未参照の項目は在庫にしない | 構造に現れない `net.block.[<名前>].*`、参照されない `actor.[<名前>].*` |
| 3 | 登録済みのクラスのうち、この Run で作らなかったクラスの設定の葉 | Atari の DefaultDQN の Run の `RainbowAgent.*`・`MuZeroAgent.*`・`ImageClsAgent.*` |
| 4 | 持ち主が休止を宣言した prefix の葉。持ち主は休止を INFO で 1 回出す | `DefaultDQNAgent.intrinsic.enabled=false` のときの `DefaultDQNAgent.intrinsic.*` |

既定葉 `?=` で残った葉も一覧の対象に含める（在庫にしない。§9 の 1）。

理由: 1・2 は PRD059 が「機械識別できる非実効行」とした種類で、解決の記録とカタログの読み口が既に知っている。名前の形（文字＋番号）では判定しない（PRD072 が上書き層の予約名による判定を不採用にしたのに合わせる。[316](../wiki/categories/316_config.md) の選ばなかった形）。3 は Agent・Env の登録が知っている。4 はモードで使わない葉の持ち主だけが知っている。`?=` を含めるのは、綴り違いの `mergin_x` が `?=` で書かれており（§2.4）、外すと共通の設定の綴り違いと旧キーが NoCare のまま残るため。正常な Run での誤検出は D2 の受入で 0 件にする（持ち主が読む、休止を宣言する、設定を直す）。依存: D3、D5。

### D5. 持ち主は自分の葉を構築時に読むか、休止を宣言する

- 遅れて作る読み口（View の menu で開く panel など）は、Run の構築時に設定を読んで検証する。
- モードで使わない葉を持つ持ち主は、その葉も読んで検証するか、D4 の 4 の休止を宣言する。どちらにするかは持ち主ごとに実装計画で決める。

理由: D3(b) の一覧の時点で読まれていない葉は、一覧に出る。また、[AGENTS.md](../../AGENTS.md) の「検証は必要な情報が揃う最も早い自然な境界で行う」に合う（`Conv2dPanel` の設定の誤りは今は panel を開くまで分からない）。依存: D3、D4。

### D6. NoCare をやめる

[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針の「設定基盤が未知キーを一律検出しない場合は NoCare」（81・89 行）を、「旧キーは D3 の汎用の検出で、ほかの読まれない葉と同じに扱われる」に改める。旧キー専用の WARN・検出・tripwire を作らない決まりは変えない。`WarnDeprecatedRunsDirConfig` は消す（旧キーの WARN を起票済みの issue の方針と同じ。その issue を取り込むかは §9 の 14）。

理由: 方針の「現在の汎用機構が未知識別子を検出できる場合は通常の未知識別子として扱う」に、D3 の汎用の検出が当たる。旧キーだけを特別に扱わないので、クリーンブレーク方針と両立する（G3）。依存: D3。

### D7. 明示した入力の不在と、未知の値を止める

- `$include` 先が無い、include 行のファイル名が空・解析できない、存在を確かめたファイルが開けない場合は、`ANET_SYSTEM_ERROR` で止める（config.cpp 555・605・626・632・640 行）。
- metrics の定義の未知の event・step・target・属性（observers.cpp 1374・1379・1384・1404・1633・1640 行）は、`ObserverFactory` の構築時に `ANET_SYSTEM_ERROR` で止める。
- DefaultDQNAgent の観測の正規化の `post_process_type` は、Agent の構築時に値域を検証する（scaler_impl.cpp 420 行の毎回の WARN を消す）。
- RainbowAgent の `use_qr=true` かつ `num_quantiles<=1` は、構築時に `ANET_SYSTEM_ERROR` で止める（rainbow_agent.cpp 54 行の読み替えを消す）。

理由: どれも [AGENTS.md](../../AGENTS.md) の表の「意図しない状態」か「明示要求の不履行」に当たる（G4）。依存: D1。

### D8. 題材の a9a0e3 の扱い

- dormant tag を参照する scalar の skip は INFO にする（trainer.cpp 866 行）。文言と、tag ごとに 1 回であることは変えない。`eval.[<tag>]: definition-only` の INFO は別の対象（定義の側）なので、両方残す。
- 評価だけの Run の RND: `learn_batch_size` と num_envs の検査は、補助学習が動くとき（learner が有効）だけ行う。learner を止めた Run での休止の通知は INFO にする（default_dqn_agent.cpp 110・119 行）。checkpoint の読み込みの検査（同 406〜408 行）は変えない。
- `interval = 0` で止めた tag と、schedule の記述が無い tag を分けない（どちらも dormant で INFO。§9 の 2）。

理由: どれも利用者が意図して組んだ状態で、D1 の INFO に当たる（G2・G5）。`learn_batch_size` の検査は、補助学習が休止していれば意味を持たない。依存: D1。

既存の判断との食い違い: dormant の skip を INFO にするのは、[ADR 0027](../adr/0027-eval-definition-schedule-separation.md) の Consequences の「metrics の dormant タグ参照 WARN」の行と、[ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) の「scalar は現行どおり dormant タグでは WARN して捨てる」の行のレベルの部分を改める（skip する振る舞いと文言は残る）。PRD052 の D9 と PRD081 の 92 行が定めたレベルも置き換わる（両 PRD の本文は当時の記録として書き換えない）。確定したら両 ADR の先頭へ注記する（§5.8）。

### D9. 既存の WARN の扱い

§5.5 の表のとおり、各行を D1 の線引きで「WARN のまま」「INFO へ」「ERROR へ」「消す」「1 回にする」のどれかへ移す。

理由: G1・G2・G4。表の行ごとの理由は表に書く。依存: D1、D7、D8。

## 5. 実装契約

### 5.1 ログのレベル（D1）

- WARN と INFO の文言は、今の規約どおり、対象の識別子（キー・tag・パス）、指定値か必要量、期待値か上限を含める（[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則の 196 行）。
- 「対象ごとに 1 回」は、Run（RunManager）ごとに数える。同じプロセスで Run を作り直したら、また 1 回出る。NN の複製（target・Actor 用・skip・RND）で同じ構築を繰り返しても、同じ対象は 1 回にする。
- ログのレベルの名前、Run のログの `[W]`・`[I]` の書式、Logs pane の数え方は変えない。

### 5.2 読んだ葉の記録

- Run の実効設定から読み口へ渡した設定は、どの読み方（`Config` の読み取り、`ConfigData` の直接の `Get`・`Read`、カタログの項目へ分けた設定、複製した設定）で読んでも、元の完全なキーで「読んだ」と記録される。
- キーの有無を見るだけの問い合わせ（`Has`）は、読んだに数えない。
- 全キーを走査するカタログの読み口は、実体化した項目の葉を読み口（factory など）に読ませる。実体化しなかった項目は、共有のカタログ（D4 の 2）なら在庫として記録し、1 つの持ち主の下だけで参照される定義（`net.branch.[<b>].config_profile.[<名前>]`）なら在庫として記録しない（その葉は読まれない葉として残り、§5.4 の一覧に出る）。走査しただけでは読んだに数えない。
- 解決は、選択の源として参照した通常 prefix を D4 の 1 として記録へ渡す。
- 記録は Run の構築のあいだだけ伸び、学習のホットパスでは読み取りが起きない（§5.7）。

### 5.3 閉じたスキーマの検査の対象（D3(a)）

| カタログ | 項目を実体化する読み口 | 読まれない葉があるときのエラーに含めるもの |
|---|---|---|
| `net.block.[<名前>]` | module の factory | block の名前・`type`・読まれない葉のキーと値・その type が読むキーの一覧 |
| `net.config_profile.[<名前>]`、`net.branch.[<b>].config_profile.[<名前>]` | config_profile の読み口 | 項目の名前・読まれない葉 |
| `<Agent>.actor.[<名前>]` | Actor 設定の読み口 | 同上 |
| `<Agent>.nn_forward.[<名前>]` | NN 実行設定の読み口 | 同上 |
| `run.eval.[<tag>]`、`run.eval_schedule.[<tag>]` | configured eval の定義・schedule の読み口（dormant の tag も宣言の検証で読む） | 同上 |
| `metrics.scalar.[<tag>]`、`metrics.trace.[<tag>]`、`metrics.image.[<tag>]` | `ObserverFactory` | 同上 |

表に無いカタログが見つかったら、実装計画で同じ扱いに足す（[CONTEXT.md](../../CONTEXT.md)「カタログ」の定義に当たるもの）。

### 5.4 Run の読まれない葉の一覧（D3(b)）

- 時機: Run の構築（RunManager、Agent、Env、Observer の構築と、online のときの RunnerFrame の panel の構築）を終え、最初の Train step より前。online と batchrun で同じ。
- 対象: Runner が `config/config_data.txt` に保存する実効設定の葉のうち、§5.2 で読んだと記録されず、D4 の在庫にも当たらない葉。CLI で与えた葉も含む。
- 出し方: 1 件の WARN。1 行目に件数、続く行に `key = value` をキーの順に 1 行ずつ。0 件なら出さない。
- 一覧に出た葉があっても Run は止めない。

### 5.5 既存の WARN の扱い（D9）

「残す」は WARN のまま。「1 回」は対象ごとに 1 回にする（§5.1）。行番号は起票時点。

| 場所 | 文言（要点） | 区分 | 扱い（暫定の既定） | 理由 |
|---|---|---|---|---|
| LogPanel.cpp 658 | Log copy failed: the clipboard could not be opened. | F | 残す | 補助の操作の失敗（D1 の 3） |
| RunnerApp.cpp 407 | Last workspace is unavailable; falling back to _default. | C | 残す | 自動の選択の代替で、AGENTS.md の表どおり |
| RunnerApp.cpp 131〜145・414 | Config key app.metrics_logger.runs_dir／root_dir is deprecated and ignored. | E | 消す | D6。旧キーは D3(b) の一覧に出る |
| RunnerApp.cpp 732 | Agent save produced zero bytes; … | D | 残す | 終了時の保存が 0 byte で、止めずに続ける（D1 の 3）。MuZero の試作の Agent は Save で 0 を返す（[muzero_proto_agent.hpp](../../core/anet-core/include/anet/muzero_proto_agent.hpp) 184 行） |
| RunnerFrame.cpp 1029・1036 | No main pane occupies the centre. … / Both main panes occupy the centre. … | D | 残す | 想定外の経路を GUI の配置の修復で正しい状態へ戻し、Run の結果は変わらない（D1 の 3。PRD110） |
| default_dqn_agent.hpp 234 | use_optimistic_target=true copies … without optimism | B | 残す | 推奨外の組み合わせ |
| default_dqn_agent.hpp 381 | learner.use_tbo … double-compressed | B | 残す | 同上 |
| default_dqn_agent.hpp 484 | learner.enabled=false without auto_load_file; … | B | 残す | 書き忘れの可能性 |
| default_dqn_agent.hpp 542 | … skip.max_length active_range=… is unused with max_length_mode=… | A | 残す | 書いた値が今のモードで効かない |
| default_dqn_agent.hpp 557 | … skip.nn_forward_key=… is unused with skip.nn_forward_mode=shared | A | 残す | 同上 |
| default_dqn_agent.hpp 590 | learner.skip.feature_gradient=stop with target_window=…: skip loss gradients do not reach shared features | B | INFO へ | `stop` は選べるモードで、文言はその定義の説明（D1 の 1 の (d)。PRD104 の STOP の Run で毎回出る） |
| default_dqn_agent.cpp 110 | intrinsic.learn_batch_size=N exceeds num_envs=M; … | C | 補助学習が動くときだけ検査する。動くときの扱いは §9 の 7 | D8。切り詰め自体は AGENTS.md の「丸めない」と食い違う |
| default_dqn_agent.cpp 119 | intrinsic.enabled=true with learner.enabled=false: … suspended | A | INFO へ | D8（D1 の 1） |
| default_dqn_agent.cpp 440 | Checkpoint has no skip networks; … initialized skip networks … | D | INFO へ | skip を有効にした設定で、checkpoint に無い部分を今の設定で初期化した（D1 の 1。[200](../design/200_dqn_agents.jp.md) 606 行が基準の Run から skip の学習へ移る契約としている） |
| dqn_based_agent.cpp 269・273・282・286 | PER の `per_eps`・`actor_approx`・clip の組み合わせ | A・B | 残す | 推奨外の組み合わせ・効かない値 |
| dqn_based_agent.cpp 1525 | ActionPolicy spatial exploration is enabled with num_envs < 32. | B | 残す・1 回 | ActionPolicy を作るたび（Actor・skip ごと）に出ている |
| dqn_based_agent.cpp 2835 | Gradient noise subscription intervals differ: … | B | 残す | 推奨外の購読 |
| dqn_based_agent.cpp 2864 | Policy churn metrics observe one hard-target phase: … | B | 残す | 同上 |
| dqn_based_agent.cpp 4252 | Learner::Load: checkpoint has no primary_skip optimizer role; initialized moments … | D | INFO へ | split を選んだ設定で、checkpoint に無い optimizer の状態を初期化した（D1 の 1。200 の 602 行の契約） |
| dqn_based_agent.cpp 4270〜4295 | Learner::Load: optimizer option differs from checkpoint; config applied. … | D | INFO へ・1 回 | ADR 0053 の判断どおり設定が勝った通知（D1 の 1）。ADR 0053 の「WARN で示す」を改める。§9 の 10 |
| intrinsic_rnd.cpp 258 | Ignored … multiplier_max=… with multiplier_mode=unclipped | A | 残す | 書いた値が効かない |
| intrinsic_rnd.cpp 279〜293 | Ignored … obs_norm.<key>=… with statistics_mode=variance_feedback | A | 残す | 同上 |
| intrinsic_rnd.cpp 498 | RND checkpoint optimizer options replaced: … | D | INFO へ | 4270 行と同じ（ADR 0053 のレベルの部分を改める） |
| init.cpp 76 | CUDA_LAUNCH_BLOCKING overridden by backend.cuda_launch_blocking. … | B | INFO へ | 明示した設定が環境変数に勝った（D1 の 1） |
| app_util.cpp 453・467 | Workspace history could not be read. … / … entry ignored. … | D | 残す | 補助のファイルの壊れ（Run の入力ではない。D1 の 3） |
| app_util.cpp 581・587 | Failed to redirect stdout／stderr to … | D | 残す | 補助の出力先の失敗（D1 の 3） |
| config.cpp 555・640 | Properties: Failed to open file／include file. … | A | ERROR へ | D7（D1 の 2） |
| config.cpp 605・626・632（ERROR を出して続ける） | include 行の解析の失敗 | — | ERROR で止める | D7 |
| image.cpp 671 | ImageProvider tag '<tag>' already exists. | D | ERROR へ | 構築時の同じ tag の二重登録は、部品の契約に反する呼び出し（D1 の 2） |
| metrics_logger.cpp 327 | … Retrying with libx264. | C | 残す・選択を固定して 1 回 | AGENTS.md の表の「以後は選択を固定」 |
| nn_impl.cpp 1838 | net.branch.[b].config_profile.[g] is defined but not used by that branch. | A | 消す | branch の config_profile はその branch だけが参照する定義で、未参照の項目は在庫ではない（D4 の 2）。その葉は読まれない葉として D3(b) の一覧に WARN で出る（D1 の 3 の (a)）ので、この WARN と重なる |
| nn_impl.cpp 1847 | net.config_profile.[g] is defined but not used by any branch. | A | 消す | 全体の config_profile はカタログで、未参照の項目は在庫（D4 の 2）。在庫はログに出さない（D1） |
| nn_impl.cpp 1923 | NetworkBodyBuilder: input key … not bound … | A | 残す・1 回 | 書き忘れの可能性（AGENTS.md 168 行の例） |
| nn_modules.cpp 1676 | ResBlock dropout_rate is enabled with BatchNorm. … | B | 残す・1 回 | block 数 × 構築数だけ出ている |
| observers.cpp 91 | HeatMapVectorObserver: size mismatch. … | D | 残す・1 回 | log_interval ごとに出ている |
| observers.cpp 705・1181 | … force-ended due to unexpected episode_start … | D | 残す | 外から届いた episode の乱れで描画を打ち切り、Run の結果は変わらない（D1 の 3） |
| observers.cpp 1115 | MetricsLogObserverBase::OnUpdate(): value not found. … | D | 残す・1 回 | メトリクスの値の欠けで、その更新を飛ばして続ける（D1 の 3）。更新のたびに出ている。休止が原因の欠けの扱いは §9 の 19 |
| observers.cpp 1374・1379・1384・1404・1633・1640 | Unknown event／step／target／attribute … | A | ERROR へ | D7（D1 の 2） |
| probe.cpp 251 | BatchExperienceVectorProbe::GetVector() failed to get value. … | D | 残す・1 回 | 呼ぶたびに出ている |
| replay_buffer_impl.cpp 1213 | PER Rejection Sampling failed to fill batch. Falling back to uniform. | C | 残す・1 回。代替の是非は §9 の 8 | Run の構築のあとに起きて止めずに続ける抽選の代替（D1 の 3）。明示した PER の一部を代える |
| rainbow_agent.cpp 54 | use_qr is true but num_quantiles <= 1. Treating as Scalar DQN. | C | ERROR へ | D7（D1 の 2） |
| scaler_impl.cpp 420 | Unknown squash_type. | A | 消す（構築時の検証で止める） | D7（D1 の 2） |
| scaler_impl.cpp 541 | Outlier Row Detected! … Full Obs: … | D | `ANET_LOG_DEBUG` へ | `// DEBUG` のブロックで生の観測を出す、開発用の診断（D1 の 4） |
| trainer.cpp 866 | Skipping metrics for unscheduled eval tag. … | A | INFO へ | D8（D1 の 1） |
| trainer.cpp 1150 | Eval session has fewer adopted episodes than episode groups. … | B | 残す | 推奨外の組み合わせ |
| trainer.cpp 1212 | Skipping Env config dump because GetConfigData is unsupported. … | F | 残す | Env の実装の欠け（コードで直すこと） |
| AtariEnv.cpp 483 | AtariEnv.display_screen=true … unsupported with worker threads … | B | 残す・1 回 | `display_screen` は `num_envs = 1` と `env.worker_type = SINGLE_THREAD` の単発のデバッグ用で（[220](../design/220_atari_env.jp.md) §4.2 の 167 行）、Env は worker の種類を知らないので、支えない組み合わせかを判定できず利用者に確かめさせる（D1 の 3 の (c)）。lane ごとに出ている。worker thread と併用したときだけ止めるかは §9 の 9 |
| ImageData.cpp 327・350・360 | ImageDataset auto cache … | C | 残す | auto の代替で、選択を固定している |

### 5.6 題材の事例の挙動

| 入力 | 挙動 |
|---|---|
| 構造に現れる `net.block.[Conv2d_X]` に `type = Conv2d` と `out_channels = 32`（`conv.` なし） | Network の構築時に `ANET_SYSTEM_ERROR`。block の名前、`type`、`out_channels = 32`、Conv2d が読むキー（`conv.out_channels` など）を含める |
| 同じ block が構造に現れない | 何も出さない（D4 の 2） |
| 共通の設定に `Conv2dPanel.conv2d.mergin_x ?= 2` | Run の構築の終わりに、読まれない葉の一覧の WARN に `Conv2dPanel.conv2d.mergin_x = 2` が出る |
| Atari の `run.@eval2ch_r1`（`eval_target` を `interval = 0`） | INFO の `Skipping metrics for unscheduled eval tag. tag='eval_target'. …`。WARN は出ない |
| RND ありの checkpoint を `learner.enabled = false`・`run.train.num_envs = 1` で読む評価だけの Run | `learn_batch_size` の WARN は出ない。補助学習の休止は INFO |

### 5.7 性能

読んだ葉の記録は設定を読むたびに 1 回起き、設定の読み取りは Run・Agent・Env・Observer の構築時に集まる。学習の step のループで `ConfigData` を読む箇所は、実装計画で検索して確かめる（見つかったら構築時へ移すか、記録の単価を測る）。性能の受入は置かない。

### 5.8 移行対象

この一覧が移行対象の正本である。

| 対象 | 扱い |
|---|---|
| [AGENTS.md](../../AGENTS.md) 81・89 行 | D6 の文へ改める |
| AGENTS.md 156〜157 行、184〜196 行 | D1 の線引きへ改める。186 行の「意図しない状態」（契約違反の内部状態、壊れた入力データを含む）を D1 の 2 の条件（Run の構築までに判定できるもの、Run の結果に効く入力、今 ERROR で止めている実行時の検査）へ改め、Run の構築のあとに起きて止めずに続ける異常と補助のファイル・出力先の失敗を WARN の行として足す。187 行（意図された休止）は INFO へ。156〜157 行は D1 の 3 の言い方へ |
| [実装 品質確認観点](../quality/impl.md) 項番 24・25・27 | D1 の言い方へ揃える。正常な Run の WARN 0 件の確かめを 1 項目足すかは §9 の 18 |
| [CONTEXT.md](../../CONTEXT.md)「dormant」 | 「タグごと 1 回の WARN で skip」を INFO へ |
| [100](../design/100_runtime_and_configuration.jp.md) §2.1・§5.1（169 行）・§6.1（208 行）・§7.3・§8 | 読んだ葉の記録、在庫、一覧の時機、dormant の INFO、include の失敗、テストの案内 |
| [130](../design/130_neural_networks.jp.md) §7.1（287 行「key ごとに 1 build 1 回だけ WARN」）と block の章 | block の閉じたスキーマ、WARN を Run ごとに 1 回 |
| [140](../design/140_observability.jp.md) 232 行（不明な event・step 軸・target は WARN 後に既定値）・252 行（scalar の既存の WARN は変えない）・286 行（value not found の WARN） | metrics の定義の未知の値を止める。value not found を対象ごとに 1 回 |
| [200](../design/200_dqn_agents.jp.md) §6.2（398 行「learner.enabled=false では一度WARNし」）・§6.5（506 行「feature_gradient=stop は…一度 WARN する」）・§8.2（600 行「差があればWARNする」、602 行「追加状態を初期化して一度WARN」、606 行「skip を新規初期化して一度 WARN する」、619 行「checkpoint と異なる項目は…WARN へ出し」）・§9.4（699 行「購読ゼロならNoCare」） | RND の休止、stop の説明、optimizer の差、skip・split の初期化を INFO。購読の無い可塑性の設定は D5 に合わせて読むか休止を宣言する |
| [212](../design/212_rnd_lifelong_novelty.jp.md) §3.5（262 行「N が batch size より小さいときは構築時に 1 回警告し」）・§4.5（454 行「差があれば WARN する」）・§5.1（470 行「learner.enabled=false では一度警告して…休止する」） | `learn_batch_size` の検査の条件（D8）、RND の optimizer の差を INFO、休止を INFO |
| [020](../design/020_user_guide_run.jp.md) §3.4.9（258 行「見つからない場合は WARN を出して続行する」）・377 行の表・§3.6・§6.4（517 行「checkpoint と違う値はログに WARN が出る」） | include の失敗は止まる。読まれない葉の一覧の見方。optimizer の差は INFO |
| [220](../design/220_atari_env.jp.md) §4.2（167 行「env は `display_screen=true` のとき log.warn を 1 回出す」、68・143 行の表）・[040](../design/040_development_environment.jp.md) §3.6（185 行） | 暫定の既定（WARN を Run ごとに 1 回）では変えない。§9 の 9 で併用を止める形に決めたら追随させる |
| 上の設計文書の `.en.md`（100・130・140・200・212・020。220・040 は §9 の 9 による） | 同じ章を追随させる |
| ADR 0027・0047・0053 | dormant の WARN を INFO にする判断（0027・0047）と、checkpoint の options の差を INFO にする判断（0053）が確定したら、それぞれの先頭へ注記する（本文は書き換えない） |
| [trainer_test.cpp](../../core/anet-core/src/trainer_test.cpp) 1056 行 | dormant のレベルの断定を INFO へ |
| [intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) 1262 行 | `exceeds num_envs` の断定を D8 の条件へ |
| WARN のレベルを断定するテスト（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) 24 か所、[config_test.cpp](../../core/anet-core/src/config_test.cpp) 3、[nn_test.cpp](../../core/anet-core/src/nn_test.cpp) 2、trainer_test.cpp 4、[GridMazeEnv_test.cpp](../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp) 1 の `wxLOG_Warning`） | §5.5 の扱いに合わせて見直す。新しく止まる設定（include の不在、未知の属性）を入力に使うテストも探す |
| 出荷設定（`apps/runner/config/*.txt`、番号付き bat） | A2 の一覧に出た葉を直す（綴り違い・旧キー・持ち主のない葉）。綴り違いの `mergin_x`（§2.4。起票済みの issue を取り込むかは §9 の 14）を含む |
| [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) | 変えない（演算子の検査で、本 PRD の検出とは別） |
| `core/anet-core/testdata/prd117/README.md`（新規） | A2 の Run の組み合わせと、WARN を数える手順 |
| wiki の [316](../wiki/categories/316_config.md)・[315](../wiki/categories/315_nn.md)・[317](../wiki/categories/317_metrics.md) | 実装の完了後に点検で拾う |

検索語（実装計画で再検索する）: `log::warn()`・`LOG::warn()`・`.warn()`・`log.warn`・`wxLOG_Warning`・`NoCare`・`WARN`・`警告`・`Failed to open include`・`Unknown event`・`definition-only`。起票時の検索は 3 回行った。

- 規約・用語集・設計文書（jp）・品質確認観点・プロセス・skill を `NoCare|1 ?回だけ ?WARN|一度だけ(の)? ?WARN|WARN 1 回|1 回の WARN|1回WARN|WARNして|WARN後|エラーにならず続行` で検索: 9 ファイル 19 件。
- 設計文書（jp）の `WARN|警告` を含む行を、§5.5 で扱いを変える行の文言と設定キー（`include`・`feature_gradient`・`learn_batch_size`・`intrinsic`・`optimizer`・`checkpoint`・`config_profile`・`bind`・`Unknown`・`dormant` など）で絞って検索: 7 ファイル 22 件。
- 設計文書（jp）を小文字の `log\.warn|\.warn\(\)` と、扱いを変えた行の設定キー（`display_screen`・`cuda_launch_blocking`・`primary_skip`）で検索: 220 の 167 行（上の表の行）、120 の 69 行と 140 の 308 行（Logger の説明で対象外）、`display_screen` の設定の表（220 の 68・143 行。上の表の行）、040 の 185 行（上の表の行）、200 の 602 行（上の表の行）。`cuda_launch_blocking` を説明する設計文書は無かった。

3 回の結果のうち、上の表に行番号で挙げていない行は次のとおり（行単位。3 回目の結果は上の項目に書いた）。

- [210](../design/210_metrics_viewer.jp.md) の 129・474・972 行: Metrics Viewer 自身のログで、Run のログではないので対象外。
- §5.5 で「残す」とした WARN を説明する行: 020 の 515 行、160 の 31 行、200 の 448・721・751 行、212 の 196・215 行。

### 5.9 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ConfigUsageTracker` | 新規 | Run の実効設定の葉ごとに、読まれたか・在庫かを記録し、読まれない葉の一覧を返す | 設定管理・config |
| `ConfigData` | 変更 | 読み取りを記録へ届ける。複製と、カタログの項目へ分けた設定からの読み取りも、元のキーで届ける | 設定管理・config |
| `Config` | 変更 | 持ち主の休止の宣言を記録へ渡す | 設定管理・config |
| `ConfigResolver` | 変更 | 選択の源として参照した通常 prefix を記録へ渡す | 設定管理・config_impl |
| `Properties` | 変更 | include の不在・解析の失敗・開けないファイルで止める | 設定管理・config |
| `NetworkStructBuilder` | 変更 | 実体化した block・config_profile の読まれない葉で止め、未参照の block と全体の config_profile を在庫として記録する（branch の config_profile の未参照の項目は記録しない） | NN・nn_impl |
| `ObserverFactory` | 変更 | metrics の定義の未知の値・属性で止め、定義の項目の読まれない葉で止める | 可視化・メトリクス・observers |
| `RunManager` | 変更 | 構築の終わりに読まれない葉の一覧を出す。dormant の skip を INFO にする | RL 共通基盤・trainer |
| `DefaultDQNAgent` | 変更 | RND の休止を INFO にし、`learn_batch_size` の検査を補助学習が動くときに限る。`post_process_type` を構築時に検証する | DQN・default_dqn_agent |
| `RainbowAgent` | 変更 | 互換性の無い `use_qr` と `num_quantiles` で止める | DQN・rainbow_agent |
| `VideoLogger` | 変更 | auto の代替の選択を固定する | 可視化・メトリクス・metrics_logger |
| `RunnerApp` | 変更 | 旧キーの WARN を消す | RunnerApp |
| `RunnerFrame`・`Conv2dPanel` | 変更 | panel の設定を Run の構築時に読む | RunnerApp |

§5.5 の表で「1 回」「INFO へ」とした行の持ち主（`DQNBasedAgent`、`NetworkBodyBuilder`、各 Observer、`ReplayBuffer` の実装など）も変更に当たる。共通の「対象ごとに 1 回」の部品の形と置き場は実装計画で決める（§6 の 3）。

### 5.10 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md) | net.blockの未知の葉（conv.なしのout_channelsなど）が構築時に検出されず黙って無視される | `316_config`, `315_nn` | [nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) 787〜845・1776〜1806 行、[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) 2969・3022〜3075 行 | D3・D4・D5・D6 | A3・A4 |
| [issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md) | Atariの正常なRunで毎回出るWARN（eval_targetの未スケジュール、評価だけのRunのRND）で本物のWARNが埋もれる | `318_rl_core`, `317_metrics`, `324_dqn` | [trainer.cpp](../../core/anet-core/src/trainer.cpp) 860〜870 行、[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 109〜119 行 | D1・D2・D8 | A2・A5・A6 |
| [issue-20261103-2c9a07](../management/issues/issue-20261103-2c9a07.md) | 旧キーapp.metrics_logger.runs_dir/root_dir専用のWARNが残り、クリーンブレーク方針に反する | `110_runner_app` | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) 131〜145・414 行 | §9 の 14（取り込むと決めたら D6） | A1・A4 |
| [issue-20261103-66100f](../management/issues/issue-20261103-66100f.md) | Conv2dPanel.conv2d.mergin_x/yとimage.@conv2d.conv2d.mergin_x/yが綴りの誤りで読まれない | なし | [common.txt](../../apps/runner/config/common.txt) 75・76 行、[metrics_image.txt](../../apps/runner/config/metrics_image.txt) 96・97 行、[Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp) 22・23 行 | §9 の 14（取り込むと決めたら D2・D4・D5） | A2・A4 |

表の 3・4 行目は、取り込むかを §9 の 14 で決める候補である。取り込むと決めたら、その issue の親を本 PRD の索引の issue に付け替える。

## 6. 細部（暫定で置いた既定）

1. 一覧の WARN の 1 行目: `Unread config leaves: N. These leaves were not read by any component.`（理由: ログは英語の規約。件数で重さが分かる）
2. 一覧の行数の上限: 置かない（理由: 正常な Run は 0 件が前提で、多いときほど全部を見たい）
3. 「対象ごとに 1 回」の部品: ログ基盤（log.hpp の近く）に、対象の鍵ごとに 1 回だけ出す共通の部品を置き、§5.5 の「1 回」の行と新しい箇所はそれを使う（理由: §2.2 の箇所ごとの書き方をそろえ、Run ごとに数える §5.1 を 1 か所で守る）
4. 一覧を Run の `json/` にも保存するか: 保存しない（理由: Run のログに残り、`config_data.txt` から再現できる）
5. A2 の Run の長さ: 各 Run とも Train の 2,000 exp step で止める。それまでに発火しない eval schedule（`interval = 0` で止めたものと、初回の発火が 2,000 exp step より後のものの両方）は待たない（理由: 構築の終わりの一覧と、構築・配線で出る WARN はこの長さで出揃う。評価の中で出る WARN は、A5・A6 とテストの設定で確かめる。Atari の既定の評価は約 50 万 exp step 後で、待つと A2 の費用が大きくなる）
6. ERROR の文言の言語: 英語（ログの規約）

## 7. 検証と受入

以下は実装工程で満たす条件で、起票時点では実行していない。

### A1. 既存の WARN の扱い

§5.5 の全行が表の扱いになっている。レベルを変えた行と ERROR へ移した行は、行ごとにテストでレベル（`wxLOG_Info`・`wxLOG_Warning`・`ANET_SYSTEM_ERROR`）を確かめる。`ANET_LOG_DEBUG` へ移した行は、その状態を起こしても WARN が出ないことを確かめる（DEBUG の出力の有無はビルドの `ANET_ENABLE_DEBUG_LOG` で変わるので確かめない）。「1 回」の行は、同じ対象で 2 回以上起こしても 1 件になることを確かめる。消した行は検索で残りが 0 件。

### A2. 正常な Run の WARN 0 件

次の Run を Debug か Release のビルドで流し、各 Run のログ（`run_*.log`）の `: [W] ` の行を数えて、全 Run で 0 件。数える単位は Run。

- 出荷設定の Env の設定ファイル（`apps/runner/config/` の `Atari.txt`・`CartPole.txt`・`DropMerge.txt`・`GridMaze.txt`・`GridMaze_muzero.txt`・`ImageCls.txt`・`LunarLander.txt`）の既定の `run.$`。
- 番号付きの bat（`apps/11_batch_run.bat`・`12_batch_run.bat`・`13_batch_run.bat`）に今書かれている Run のうち、`run.$` の組み合わせが違うもの（`goto` で飛ばされている段落を含む）。
- Atari の `run.@evalonly`。

step 数は §6 の 5。データセットが手元に無い Env（ImageCls など）で流せなければ、その Run を未達として理由を記録する。A2 の Run の組み合わせと数え方は `core/anet-core/testdata/prd117/README.md` に置く。

### A3. block の閉じたスキーマ

§5.6 の 1・2 行目をテストで確かめる。`?=` と `=` の両方で書いた入力で同じ結果になる。§5.3 の表のほかのカタログも、表に並ぶカタログの名前ごと（`net.config_profile`、`net.branch.[<b>].config_profile`、`<Agent>.actor`、`<Agent>.nn_forward`、`run.eval`、`run.eval_schedule`、`metrics.scalar`、`metrics.trace`、`metrics.image` の 9 つ）に 1 つずつ、実体化した項目の読まれない葉で止まることと、未参照の項目では止まらないことを確かめる。

### A4. 読まれない葉の一覧

テストの設定で、次を確かめる。

- 綴り違いの葉（`?=` と `=` の両方）と、CLI で与えた未知の葉が、一覧の WARN に 1 件ずつ出る。
- D4 の 1〜4 の例（選択の源の上書き層、未参照の block、作らなかったクラスの設定、休止を宣言した prefix）は出ない。
- branch の config_profile のうち、その branch が使わない項目の葉は出る（D4 の 2 の除外）。
- 一覧に出る葉が 0 件なら WARN を出さない。
- 一覧は Run ごとに 1 件。

### A5. dormant の tag

trainer_test.cpp の dormant の tag のテストで、skip の通知が INFO で tag ごとに 1 件、WARN が 0 件。

### A6. 評価だけの RND の Run

RND ありの checkpoint を `learner.enabled = false`・`run.train.num_envs = 1` で読む構成で、`learn_batch_size` の WARN が出ず、休止の INFO が 1 件。learner が有効で num_envs < `learn_batch_size` の構成では、§9 の 7 で決めた扱いになる。

### A7. 止める失敗

D7 の各入力（include 先が無い、include 行の解析の失敗、metrics の定義の未知の event・step・target・属性、`post_process_type` の範囲外、RainbowAgent の `use_qr` と `num_quantiles`）で、構築時に `ANET_SYSTEM_ERROR` になり、文言にキーか パス、指定値、期待値が入る。

### A8. ビルドと既存のテスト

[AGENTS.md](../../AGENTS.md) の手順で Debug ビルドを通し、既定のテストスイート、`"[shipped_config]"`、`AnetRLRunner-gui-test "[log_panel]"` を流す。既知の失敗（PRD093 の attention のテスト 1 件）以外の失敗が無い。結果と未実施の理由を実装計画に書く。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 読まれない葉の一覧を ERROR にする | 先送り | 在庫の区別を実際の Run で確かめてからにする（§9 の 4）。再開条件: A2 の WARN 0 件が 1 か月保たれ、一覧に出た葉がすべて誤りだったとき |
| 眠る bare の検出（読まれて効かない値） | 範囲外（§9 の 11 で再確認） | 「読まれたか」と「効いたか」は別の仕組みになる。本当の制約ではなく、別 PRD でも本 PRD に足しても扱える。PRD101 §7 の再開条件の判断はユーザーに残す |
| 値の参照先の検証（`stack_keys` など） | 範囲外 | 別の関心（§3.2）。[issue-20261103-b754ff](../management/issues/issue-20261103-b754ff.md) で扱う |
| WARN と ERROR の間のレベル | 不採用 | §3.2 |

## 9. 未決

暫定の起票なので、§4 の確定事項もすべてグリルで見直す。ここには、とくに決めることが多い論点を並べる。

| # | 事項 | 暫定の既定 | 決め方 |
|---|---|---|---|
| 1 | 既定葉 `?=` を一覧の対象に含めるか | 含める（D4） | 選択。含めないと共通の設定の綴り違い（`mergin_x`）と旧キーが残る。含めると正常な Run での誤検出を 0 にする手間が増える |
| 2 | 意図した休止を INFO にするか、WARN のまま 1 回にするか。`interval = 0` と schedule の記述が無い tag を分けるか | INFO にし、分けない（D1・D8） | 選択。ADR 0027・0047 の WARN の部分を改める判断になる |
| 3 | 一覧の時機と、遅れて作る読み口 | 構築の終わり。遅れて作る読み口は構築時に読む（D5・§5.4） | ④。記録だけを先に作り、A2 の Run で遅れて読まれる prefix を数えてから決める |
| 4 | 一覧を WARN にするか ERROR にするか | WARN（D3(b)） | 選択。ERROR は §8 の先送りの条件で再開 |
| 5 | 在庫の 4 種類で足りるか | D4 の表 | ④。A2 の Run で一覧に出た葉を種類ごとに数え、誤検出が残る種類があれば足す |
| 6 | 閉じたスキーマの対象のカタログ | §5.3 の表 | ③と④。カタログを走査する読み口 29 か所を実装計画で洗う |
| 7 | 補助学習が動くときの `learn_batch_size` > num_envs | 今の切り詰め＋WARN のまま | 選択。ERROR にする、または値を num_envs 以下に書くことを求めるかは PRD116 の決定に関わる |
| 8 | PER の rejection sampling の一様への代替 | WARN を Run ごとに 1 回にするだけ | 選択。代替自体を止めるか（明示要求の不履行）、別の PRD で扱うか |
| 9 | `AtariEnv.display_screen=true` を worker thread と併用したとき | WARN を Run ごとに 1 回（§5.5） | ④。worker の種類を知る側（BatchEnv の構築）で併用を判定できるかを確かめる。判定できれば、併用は非対応の明示要求として止め（D1 の 2）、単発のデバッグ用の構成では出さない。そのときは [220](../design/220_atari_env.jp.md) §4.2 と [040](../design/040_development_environment.jp.md) §3.6 を追随させる |
| 10 | checkpoint の optimizer の options の差 | INFO（§5.5） | 選択。ADR 0053 の「checkpoint と違った項目は WARN で示す」のレベルの部分を改める判断になる。設定を変えずに別の checkpoint を読んだ取り違えを WARN で知らせたいなら WARN のまま |
| 11 | 眠る bare の検出（PRD101 §7 の先送り） | 本 PRD の範囲外 | 選択。今回の件が PRD101 の再開条件に当たると見るなら、別 PRD で起こすか本 PRD に足す |
| 12 | ADR を起こすか | グリルで確定したら起こす（WARN の線引きと NoCare の撤回。3 条件を満たす見込み） | グリルで確定後 |
| 13 | 用語「読まれない葉」「在庫」を CONTEXT.md に足すか | グリルで確定したら足す | グリルで確定後 |
| 14 | 取り込む issue | [issue-20261103-2c9a07](../management/issues/issue-20261103-2c9a07.md)（旧キーの WARN）と [issue-20261103-66100f](../management/issues/issue-20261103-66100f.md)（`mergin_x`）を取り込む。[issue-20261103-582880](../management/issues/issue-20261103-582880.md)（`runner_type` を黙って serial にする）と b754ff は取り込まない | 選択。2c9a07 と 66100f は D2・D6 の受入が依存する |
| 15 | A2 の Run の組み合わせと step 数 | §7 の A2・§6 の 5 | 選択（費用）。bat の組み合わせの数で数時間かかりうる |
| 16 | ERROR を出して続けている、設定以外の箇所（trainer.cpp 1020・1042 行、DropMergeEnv.cpp 900 行） | 範囲外 | [issue-20261107-16ffd8](../management/issues/issue-20261107-16ffd8.md) に起票した |
| 17 | `default_dqn_agent.hpp` 590 行の `feature_gradient=stop` の説明を INFO にするか | INFO（§5.5） | 選択 |
| 18 | あとの PRD が WARN を足すときの確かめを、品質確認観点の項目にするか | 実装品質確認観点に 1 項目足す | 選択 |
| 19 | 意図した休止が原因のメトリクスの値の欠け（learner を止めた Run の `intrinsic.learn_count` など）を INFO にするか | WARN（D1 の 3。observers.cpp 1115 行は原因を知らない） | 選択。休止した値を出す側が値（0 や NaN）を出すか、メトリクスの定義を休止に合わせるかで、WARN を出さずに済ませられる |

## 10. 依存と順序

- PRD116 の `nn.txt` の block の直し（作業ツリーにある）が先に入っていること。入っていないと、A2 の Atari の RND の Run が D3(a) で止まる。
- D3(b) を入れると、A2 の Run で出た葉を直すまで D2 の受入が通らない。設定の直しには 66100f の `mergin_x` が入る（§9 の 14）。
- 記録（§5.2）を先に作り、A2 の Run で一覧を測ってから、在庫の種類（§9 の 5）と遅れて読む読み口（§9 の 3）を決める順にすると、手戻りが少ない。

## 11. 参照

- 題材: [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md)、[issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md)
- 規約: [AGENTS.md](../../AGENTS.md) のクリーンブレーク方針・汎用機構と利用側の責任境界・Fail-Fast 原則・設定値の扱い、[実装 品質確認観点](../quality/impl.md) の「失敗の扱い」「ログ」「設定ファイル」
- 設計文書: [100](../design/100_runtime_and_configuration.jp.md)、[130](../design/130_neural_networks.jp.md)、[140](../design/140_observability.jp.md)、[160](../design/160_applications_and_tools.jp.md) §3.5、[020](../design/020_user_guide_run.jp.md)
- ADR: [0027](../adr/0027-eval-definition-schedule-separation.md)、[0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)、[0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)
- 地図: [316 設定管理](../wiki/categories/316_config.md)、[315 NN](../wiki/categories/315_nn.md)、[317 可視化・メトリクス](../wiki/categories/317_metrics.md)、[318 RL 共通基盤](../wiki/categories/318_rl_core.md)、[324 DQN](../wiki/categories/324_dqn.md)
