# PRD117: WARN の意味の線引きと、読まれない設定の検出 PRD

> 2026-11-07 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、`default_dqn_agent.cpp`・`default_dqn_agent.hpp`・`intrinsic_rnd.cpp`・`nn.txt` は PRD116 の未コミットの変更を含む作業ツリーの位置である。実装時は近傍の記述で探し直す。
> 2026-11-10 改訂: グリルを行い、暫定の印を外した。§4 を確定事項として書き直し（D1〜D13）、起票時の未決表の 19 行を確定事項・細部・範囲外へ振り分けた（対応は §9.1）。主な決定: 意図した休止と、checkpoint の値を今の設定で置き換えた通知は INFO（D8・D10）。checkpoint に無い部分の初期化は WARN のまま（D10）。構築後の異常を WARN で続ける今の作りを規約に書く（D1）。読まれない設定キーの一覧は `?=` を含めて WARN（D3・D4）。作ったカタログ項目の読まれない設定キーは止め、使わない定義は出荷設定の検査のテストで見る（D3・D13）。`learn_batch_size` が lane の数より大きい構成は止める（D11）。休止した機能の値は NaN で返す（D12）。課題 582880 を取り込んだ（D7）。§2.6 に再グリルで確かめた事実を足し、§2 の AGENTS.md と設計文書の行番号を 2026-11-10 時点に直した（コードの行番号は起票時点のまま）。ADR 0071 を起こし、ADR 0027・0047・0053 の先頭に注記した。起草のあと、別モデルの検討を受けて、在庫の 1 の条件（D4）、一覧を出す側と時機（§5.4）、読んだ設定キーの記録の契約（§5.2）、D1 の 3 の (j)、A2 の流し方、D13 の理由を直した。レビューの指摘で、§5.3 にカタログごとの「Run で実体化しない項目」を足して A3・A9・D13 を合わせ、§5.8 に D11 で止まるテストの補助を足し、D6 に ADR 0027 の NoCare との食い違いを書き、§5.5 の「1 回」の読み方を 1 つにした。レビューのあとで、RND を無効にしたときの値の鍵も NaN にすると決めた（D12）。
> 2026-11-11 改訂: §9 の 20・21 を課題に起票した。用語の「読まれない葉」を「読まれない設定キー」に改め（本文の「葉」も「設定キー」にした。ログの文面は `Unread config keys`）、既存の用語「既定葉」「個別葉」「葉キー」の言い換えを範囲に足し（G6・D14・A11。「既定葉」は「既定キー」）、設定で無効にした機能の値の鍵の扱いを D12 にまとめた（PER は実装が既に NaN を返しており、設計文書 150 の記述が古かった。外れているのは RND だけ）。
> 起点: 2026-11-07 の会話。ユーザーは [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md)（`net.block` の未知の葉が黙って無視される）と [issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md)（Atari の正常な Run で毎回 WARN が出る）を題材に、「出なくてよい WARN は出ず、出るべき WARN が出るように全体を見直したい。使われない設定はこれまで NoCare だったが、UI の工夫で WARN に実際の意味を持たせられるようになった今、この点も見直す時期だ」として、調査と方針案の整理、暫定 PRD の起票を求めた。UI の工夫は、Logs pane の未確認の警告の件数と、Panels toolbar の Logs の印である（[160](../design/160_applications_and_tools.jp.md) §3.5）。
> 関連: [PRD052](done/052_eval_schedule_separation_10prd.md)（dormant tag の WARN、旧キーの NoCare）、[PRD059](done/059_config_concept_tree_alignment_10prd.md)（カタログ・プロファイル・上書き層）、[PRD072](done/072_config_selection_final_value_10prd.md)（既定キー `?=`）、[PRD081](081_evalpanel_episode_trace_10prd.md)（dormant tag の WARN を scalar の参照に限る）、[PRD101](done/101_exploration_schedule_profiled_value_10prd.md)（眠る bare の検査の先送り）、[PRD116](116_rnd_lifelong_novelty_10prd.md)（RND の Network が意図と別の形で組まれた件）。

## 1. 問題

### 1.1 WARN が正常な Run で出続け、意味を失っている

Logs pane は表示した WARN 以上の記録を「未確認の警告」として数え、Panels toolbar の Logs に印と件数を出す（[160](../design/160_applications_and_tools.jp.md) §3.5 の 153 行）。ところが、意図どおりに組んだ Atari の Run は毎回 WARN を 1 件出す（§2.1）。印が常に付けば、利用者は印を見なくなり、本物の WARN も読み飛ばされる。

### 1.2 書いた設定が読まれなくても、黙って通る

設定基盤は、書かれた設定キーが読まれたかを見ない（§2.4）。[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針も、旧キーについて「設定基盤が未知キーを一律検出しない場合は NoCare」としている（85・93 行）。2026-11-06〜07 の Atari の RND の Run は、block の直下に書いた `out_channels` などが読まれず、1 Network 105,006,592 パラメータ（意図は 854,240）で動いた（[PRD116](116_rnd_lifelong_novelty_10prd.md) §2.1）。設定の綴り違い（`mergin_x`、[issue-20261103-66100f](../management/issues/issue-20261103-66100f.md)）も同じく黙って通っている。

### 1.3 WARN を出す基準が 1 つにまとまっていない

[AGENTS.md](../../AGENTS.md) の Fail-Fast の表は、意図された休止に「必要なら対象ごと 1 回だけ WARN」を許し（191 行）、汎用機構の節は書き忘れの可能性に「必要に応じて一度だけ WARN してよい」とする（160 行）。意図した状態と、意図と違う可能性のある状態が、同じ WARN で出る。WARN だけで流している未知の値や、止めるべき失敗も残っている（§2.3）。

### 1.4 設定の用語の「葉」が、日本語として通じない

設定の階層の末端にある `key = value` の 1 つを、用語集と設計文書は「葉」と呼んでいる（「既定葉」「個別葉」「葉キー」）。英語の leaf の直訳で、日本語の「葉」には階層の末端という意味が無い。本 PRD も、起票時は「読まれない葉」と名付けていた（§2.6）。

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

`core/` と `apps/` の `.cpp`・`.hpp`（`third_party/`、テスト、ログ基盤自身、コメントアウトした行を除く）で WARN を出す箇所は 62 行ある。区分ごとの件数は次のとおりで、全件と扱いは §5.5 の表にある。

| 区分 | 件数 | 例 |
|---|---|---|
| A 書いた設定が効かない・使われない定義・ほかの設定で休止 | 19 | `intrinsic_rnd.cpp` 258 行の `multiplier_max` の無視、trainer.cpp 866 行の dormant tag |
| B 設定の組み合わせや値が推奨外 | 14 | dqn_based_agent.cpp 1525 行の spatial と num_envs |
| C 自動の代替・置き換え | 8 | default_dqn_agent.cpp 110 行の `learn_batch_size` の切り詰め |
| D 実行時のデータや状態の異常 | 18 | observers.cpp 1115 行の値の欠け |
| E 旧キーの案内 | 1 | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) 131〜145 行の `app.metrics_logger.runs_dir`（[issue-20261103-2c9a07](../management/issues/issue-20261103-2c9a07.md) で起票済み） |
| F そのほか | 2 | LogPanel のクリップボードの失敗、Env の設定 dump の非対応 |

[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則・設定値の扱いと食い違う箇所:

- 未知の値を WARN だけで流す: [scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp) 420 行（`Unknown squash_type.`。正規化を呼ぶたびに出る。DefaultDQNAgent の観測の正規化は `post_process_type` の値域を構築時に検証していない）と、observers.cpp 1374・1379・1384・1404・1633・1640 行（metrics の定義の未知の event・step・target・属性）。表は未知の列挙値と閉じたスキーマ内の未知フィールドを fail-fast とする（190 行）。
- 互換性の無い組み合わせを読み替える: [rainbow_agent.cpp](../../core/anet-core/src/rainbow_agent.cpp) 54 行（`use_qr=true` かつ `num_quantiles<=1` を scalar DQN として扱う）。
- 明示要求の一部を別の方法で代える: [replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) 1213 行（PER の rejection sampling が埋まらないと残りを一様に引く。固定も回数の抑えも無い）。
- 指定値を切り詰める: default_dqn_agent.cpp 110 行（`learn_batch_size` を num_envs に切り詰める。既定の 32 のままでも num_envs<32 なら出る）。
- auto の代替で選択を固定しない: [metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) 327 行（nvenc が起動しないと動画ごとに libx264 で試し直し、WARN も動画ごとに出る）。
- 明示した入力が無くても続ける: [config.cpp](../../core/anet-core/src/config.cpp) 555 行（ファイルが開けない）・640 行（`$include` 先が無い）は WARN で続け、605・626・632 行（include 行の解析の失敗）は ERROR のログを出して次の行へ進む。[Run 実行ガイド](../design/020_user_guide_run.jp.md) の 379 行はこれを「エラーにならず続行する」と案内している。
- 重複を除かずに繰り返す: observers.cpp 91・1115 行、[probe.cpp](../../core/anet-core/src/probe.cpp) 251 行、scaler_impl.cpp 541 行（`// DEBUG` のブロックで生の観測を出す）、replay_buffer_impl.cpp 1213 行。NN を構築・複製するたびに出る nn_impl.cpp 1838・1847・1923 行と [nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) 1676 行も、target・Actor 用の複製の数だけ重なる。
- 旧キー専用の WARN: RunnerApp.cpp の `WarnDeprecatedRunsDirConfig`。クリーンブレーク方針（82・92 行）に反する。

INFO で休止や選択の結果を知らせている箇所も既にある（trainer.cpp 1095・1107 行の `eval.[<tag>]: definition-only`、dqn_based_agent.cpp 2684 行の `Learner: learning is disabled …`、metrics_logger.cpp 313 行の `VideoLogger(auto): …`）。

### 2.4 設定の読み方と、読まれない設定キー

- `ConfigData` は文字列の key/value の平らな集まりで、`Read` はキーがあればその値を、無ければ呼び出し側の既定を使う（[config.hpp](../../core/anet-core/include/anet/config.hpp) 95〜105 行）。どの設定キーが読まれたかは残さない。
- `Config` の `ReadConfig` は default prefix と override prefix の順に読み、読んだ値をモジュールごとの記録へ残す（config.hpp 184〜209 行）。この記録が Run の `config/<module>.txt` になる。ただし記録は「読んだ設定キー」ではない。ProfiledValue は全欄を記録せずに読み、実効の type が使う欄だけを記録する（[schedule.hpp](../../core/anet-core/include/anet/schedule.hpp) 284〜349 行）。
- 読み口は、`Config` の派生が 75、`ReadConfig`・`ANET_READ_CONFIG` の呼び出しが 610 ある。ほかに `ConfigData` の全キーを走査する読み口が 29 か所（`net.block` などのカタログの読み口）、`ConfigData` を直接 `Get`・`Read` する読み口がコアに 16 か所、アプリに約 10 か所（[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp) 84・85 行など）ある。
- NN の block は、`net.block.[<名前>].*` を全体とAgent の下から集め、`type` を除いた設定キーを相対のキーで block ごとの設定にまとめる（nn_impl.cpp 787〜845 行）。構造に現れた block だけを、module の factory がその設定から自分のキーを読んで作る（nn_impl.cpp 1776〜1806 行。Conv2d は `conv.out_channels` などを読む。nn_modules.cpp 2969 行・3050 行付近）。factory が読まない設定キーは、構築時にエラーにも WARN にもならない（0c34dd）。
- 実際の Run（`mz_rbw_imp_s1`）の `config_data.txt` は 1,362 件、モジュールごとの記録は 671 件で、差の 1,009 件の内訳は `net.block` 447、`metrics.scalar` 123、`M1` 110、`RainbowAgent` 57、`A3` 56、`MuZeroAgent` 44、`A1` 35、`DefaultDQNAgent.actor` 17、`app.eval_trace_panel` 15、`DefaultDQNAgent.net` 14、`run.eval` 14、`A2` 12、`DefaultDQNAgent.intrinsic` 9 などだった。この差には、記録に残らない読み方（NN の branch、ProfiledValue の type、GUI の panel）で読まれた設定キーも含まれるので、記録は読まれたかの判定に使えない。
- 実効側に残る、読まれなくてよい設定キーの種類は既に整理されている。プロファイル（`@` を持つ定義）は `ConfigData` から除かれる（[100](../design/100_runtime_and_configuration.jp.md) §2.1 の手順 7）。上書き層（A1・E1・M1 など）と mode（`app.online`・`app.batchrun`）は選択の源として残り、「既知 prefix として機械識別でき」る（[PRD059](done/059_config_concept_tree_alignment_10prd.md) の 36 行、D20・D25）。カタログの項目は名前で参照される部品の定義である（[CONTEXT.md](../../CONTEXT.md)「カタログ」）。共通の設定には、その Run で作らない Agent の既定（`RainbowAgent.*` は `?=` が 51 行、`=` が 6 行）も入る。
- 既定キー `?=` の集合は `Properties`・`ConfigManager` が解決の途中で持つが、`ConfigData` には渡さない（config.hpp 154〜160 行、config.cpp 675〜803 行）。
- 遅れて設定を読む読み口がある。`Conv2dPanel` は View の menu で開いたときに作られ、そこで `Conv2dPanel.conv2d.*` を読む（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) 1519 行、[Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp) 20〜30 行）。
- 綴り違いの `Conv2dPanel.conv2d.mergin_x ?= 2` は既定キーで書かれている（[common.txt](../../apps/runner/config/common.txt) 75・76 行。[metrics_image.txt](../../apps/runner/config/metrics_image.txt) 96・97 行も同じ）。`=` だけを見る検出では拾えない。
- 静的な検査は `?=` と `=` の使い分けを見る [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) と、`[shipped_config]` のテストだけで、書いた設定キーが読まれたかは見ない。
- 起票時点の作業ツリー（PRD116 の実装）の `nn.txt` は、`Conv1d_S`・`Conv1d_M`・`Conv2d_S` などの設定キーを `conv.*` の下に書いていた。この直しは PRD116 でコミットされた（§2.6）。

### 2.5 規約・文書・過去の判断の今の記述

- [AGENTS.md](../../AGENTS.md): 85・93 行（未知キーを一律検出しない場合は NoCare）、160〜161 行（書き忘れの可能性は一度だけ WARN してよい）、188〜200 行（Fail-Fast の表。意図された休止は「必要なら対象ごと 1 回だけ WARN」）。
- [CONTEXT.md](../../CONTEXT.md) の「dormant」: dormant タグを参照する scalar は「タグごと 1 回の WARN で skip」。
- [100](../design/100_runtime_and_configuration.jp.md) §5.1（169 行）と §6.1（208 行）: dormant tag の一度だけの WARN。
- [130](../design/130_neural_networks.jp.md) 287 行: どの bind にも使われない入力は key ごとに WARN。
- [140](../design/140_observability.jp.md) 234 行: 不明な event・step 軸・target は WARN のあと既定値を使う。
- [020](../design/020_user_guide_run.jp.md) 379 行: include 先が無くても続ける。
- [実装 品質確認観点](../quality/impl.md) 項番 24・25・27: 意図された休止をエラーにしない、代替は auto だけで 1 回 WARN、WARN は対象ごとに 1 回。
- ADR: [0027](../adr/0027-eval-definition-schedule-separation.md) は本文で「旧位置キーの検出は NoCare」と決め、Consequences で dormant タグ参照の WARN の文言を定める。[0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) は「scalar は現行どおり dormant タグでは WARN して捨てる」とする。[0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) は checkpoint の optimizer の options を今の設定で置き換える判断で、その中で「checkpoint と違った項目は WARN で示す」と決めている（3 行）。
- 地図の「選ばなかった形」（[316](../wiki/categories/316_config.md)、[324](../wiki/categories/324_dqn.md)）:
  - 個別の葉と継承値の競合の WARN — 不採用（PRD072）。本 PRD は競合を見ないので開き直さない。
  - 旧キー（`use_qr`・`num_quantiles`）の tripwire — 不採用（PRD001 の D14）。本 PRD は旧キー専用の検出を作らない（D6）。
  - DQN 側で Noisy の有無を見て効かない倍率に WARN を出す形 — 不採用（PRD107、ADR 0048）。本 PRD の一覧は「読まれたか」だけを見るので、読まれて効かない設定キーには出さず、開き直さない。
  - 眠る bare の実行時・静的検査 — 先送り（PRD101 §7）。再開条件は「眠った値に気づかずに Run を回した事故が実際に起きたとき」。今回の RND の件は「読まれない設定キー」で、眠る bare（読まれて効かない値）とは別だが、近い事故である。扱いは §8（範囲外のまま）。
- [317](../wiki/categories/317_metrics.md)・[318](../wiki/categories/318_rl_core.md) の「変えるときの条件」の「WARN・fail-fast の文言と時機を変えない」（PRD094）は、読み口と配線の整理のときの条件である。本 PRD は WARN のレベルと時機を意図して変えるので、変える箇所を §5.5 に列挙する。

### 2.6 再グリルで確かめた事実（2026-11-10）

- PRD116 のコミット（66d3e411）で、`nn.txt` の `Conv2d_S` などの設定キーは `conv.*` の下に直った（[nn.txt](../../apps/runner/config/nn.txt) 151〜155 行）。同じ名前の block の `type` を 2 通りに書いた箇所と、プロファイルで block の `type` を上書きした行は、出荷設定に 0 件。
- 出荷設定（`apps/runner/config/*.txt`）の代入の行は約 3,800 行で、うち `?=` は約 1,200 行。名前の違う block は 224 個（`nn.txt` だけで 43 個）。block の設定キーを読む処理は module を作る処理と一体で（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) 1784〜1799 行）、module の種類は 28。module は block の設定だけで作れ、最初の入力が来るまで重みを確保しない（[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の `Conv2dModule` など）。
- 出荷設定の `$include` は 167 行で、先の無いものは 0 件。無くてよいファイル（workspace の `history.txt`・`prefs.txt`）は、呼ぶ側が先に有無を確かめてから読む（[app_util.cpp](../../core/anet-core/src/app_util.cpp) 445 行、[WorkspaceDialog.cpp](../../apps/runner/src/WorkspaceDialog.cpp) 227 行）。
- checkpoint の読み込みは、内的報酬の有無が設定と合わないときと、skip の Network を持つ checkpoint を skip 無効で読むときは ERROR で止め、skip 有効の設定で skip の Network を持たない checkpoint を読むときは初期化して WARN を出す（[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 406〜409・439〜441 行）。
- PRD116 は、`intrinsic.learn_batch_size` が lane の数 N より大きいときに N 件へ縮めて WARN を 1 回出すと決めている（[212](../design/212_rnd_lifelong_novelty.jp.md) §3.5 の 262 行、379 行の表、[200](../design/200_dqn_agents.jp.md) 402 行）。既定は 32（[agent.txt](../../apps/runner/config/agent.txt) 14 行）。出荷設定で RND を使う Run の lane は 256（LunarLander・GridMaze）と 128・64（Atari）で、縮む構成は無い。縮めたときも、Run の設定の dump（`config/config_data.txt`）には 32 が残る。
- PER の抽選の代替の WARN（replay_buffer_impl.cpp の `PER Rejection Sampling failed to fill batch. Falling back to uniform.`）は、§2.1 の 616 本に 1 件も無い。試行の上限は batch の大きさの 10 倍で、回数の抑えは無い。
- `GetScalar()` の契約は既にある。未知の鍵は値なし、既知の鍵で今は値が成立しないときは `NaN` を返し、Observer は値なしを未知の鍵として扱う（[140](../design/140_observability.jp.md) 238・316 行）。PER の鍵は、PER を無効にしても、既知の鍵として `NaN` を返す（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の `DefaultReplayBuffer::GetScalar`。コメントは「PER を切っただけで metrics 定義を書き換えずに済むよう、既知キーは PER 無効でも NaN を返す」とする。2026-08-27 から）。140 の 238 行も、0 や既定値へ読み替えない状態の例に「PER無効」を挙げる。[150](../design/150_replay_buffer.jp.md) §7.5 の 314 行（en は 315 行）は「uniform sampler では PER の鍵を値なしで表す」と書いていて、実装と食い違う。RND は、無効のとき鍵が無く値なしで（[212](../design/212_rnd_lifelong_novelty.jp.md) 442 行、[200](../design/200_dqn_agents.jp.md) §9.1 の 633 行。intrinsic_test.cpp 390 行が断定している）、140 の決まりから外れている。
- `run.train.runner_type` は、`pipeline`（大文字小文字を無視）と一致するかだけを見て、ほかの文字列をすべて serial として扱う（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `RunnerFactory::CreateMainRunner`。読み取りは 816 行）。WARN は出さない。出荷設定の値は `serial` が 2 か所、`pipeline` が 1 か所だけ。
- テストは自動では流れない。CI（`.github/workflows/windows-ci.yml`）は pull request と手動の起動だけで動き、`anet-core-test` を流す段を持たない。`[shipped_config]` のテストは既定のテストスイートに入っていて、設定を変えたときに流す決まりがある（[実装 品質確認観点](../quality/impl.md) 156 行）。同じ観点は「Run の構築時に止まる検査を前倒しするだけのテストは作らない」とする（417 行、項番 73）。
- 選ばれなかった mode の設定キーは実効設定に残る。bat から流した Run（`app.$=app.batchrun>P1`）の `config/config_data.txt` には `app.online.*` と `app.batchrun.*` の両方があり、解決の記録（`json/config_resolution.json`）の選択に現れるのは `app.batchrun` だけだった（AtariRnd-01 の `run_20261110-120903_mz_tg_nmdd_b01_s1`）。Env の設定は `app.$ = app.online > P1` と書き、bat の引数がそれを置き換える（[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 6 行、`apps/12_batch_run.bat` 66 行）。
- RunnerApp は RunManager を裏のスレッドで作り、そのあと main のスレッドの `CompleteRunInitialization` で、view の factory、画像の metrics の登録（`ImageProviderManager::CreateAndRegister` が `metrics.image.[<tag>]` を読む）、RunnerFrame の初期化（panel が `app.eval_trace_panel.*` などを読む）を行う（[RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) 431〜435・509〜529 行、[image.cpp](../../core/anet-core/src/image.cpp) 693〜754 行）。RunnerFrame は mode によらず作られる（RunnerApp.cpp 400 行）。
- block の設定は、全体の `net.block.[<名前>].*` を Agent の下の同じ block の設定キーで設定キーごとに上書きしてから作る（nn_impl.cpp 806〜822 行）。上書きされた全体の側の設定キーは、factory に渡る前に捨てられる。出荷設定に、Agent の下で block の設定キーを上書きした行は 0 件。
- `app.save_agent_on_close` の既定は true で（RunnerApp.cpp 66 行）、`GridMaze_muzero.txt` は上書きしていない。MuZero の試作の Agent は保存で常に 0 を返すので（muzero_proto_agent.hpp 184 行）、その既定の Run は閉じるたびに 0 byte の WARN を出す。Atari の `run.@evalonly`（[Atari.txt](../../apps/runner/config/Atari.txt) 541〜557 行）は `auto_load_file` を持たない。
- 設定の値は `${完全なキー}` で別の設定キーを 1 段だけ参照できる（[100](../design/100_runtime_and_configuration.jp.md) §2.1 の手順 7）。
- カタログの読み口には、定義した全部の項目を読むものと、参照された項目だけを読むものがある。Actor 設定は、全部の項目を構築時に型付きの設定へ変える（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) 193〜195 行）。scalar・trace・画像の metrics の定義と、評価の定義・schedule も、定義した全部を読む（observers.cpp の `ObserverFactory`、image.cpp 697〜754 行）。参照された項目だけを読むのは、構造に現れた block、branch が使う config_profile、参照された NN 実行設定（default_dqn_agent.hpp 650 行）である。
- 「葉」は、[CONTEXT.md](../../CONTEXT.md) の「既定葉」「個別葉」「葉キー」（PRD072）に始まる。現用の範囲（[実装 品質確認観点](../quality/impl.md) の項番 71 が定める範囲と、規約・品質確認観点・プロセス・skill）の、Git が追跡しているファイルの 52 ファイル 151 行に出る（2026-11-11 の `git grep`。うち 2 行は「言葉」）。「既定葉」が 60 か所、「個別葉」が 20 か所、「葉キー」が 1 か所で、ほかに「葉値」「確定葉」、数え方の「6葉」などがある。用語集では、`=` で書く側の正式な語は「個別指定」で、「個別葉」はその括弧書きの別名である。
- 課題と PRD の結び付けは課題の `prd` 欄で表す（PRD120、[ADR 0069](../adr/0069-issues-point-to-prds-n-to-1-and-prd-completion-is-derived.md)）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: WARN の意味を 1 つにする。WARN は「利用者が確かめて、設定・コード・実行環境を直すべき状態」に限り、意図した休止と、設定どおりに動いた結果の通知は INFO にする。
- G2: 普通の Run（D2）で WARN が 0 件になる。Logs の印が付けば、確かめるべきことがあると読めるようにする。
- G3: 書いた設定が読まれないことを検出する。実体化したカタログ項目の読まれない設定キーは構築時に止め、それ以外の読まれない設定キーは Run の構築の終わりに WARN で一覧する。NoCare をやめる。
- G4: 出るべきなのに出ていない失敗を正す。未知の値と明示した入力の不在を、WARN で流さずに止める。黙って別の値として扱っている `run.train.runner_type` の未知の値も止める。
- G5: 本 PRD を指す課題 6 件（§5.10）が片付く。
- G6: 設定の用語から、日本語として通じない直訳の「葉」をなくす。

### 3.2 非ゴール

| 退けた案・範囲 | 理由 |
|---|---|
| ログ基盤の作り替え（structured・JSON logging、thread-local の文脈） | 別の関心（[PRD038](done/038_env_log_auto_prefix_10prd.md) が範囲外とした）。WARN の意味と検出は今の基盤で表せる。本当の制約ではなく、別 PRD で選べる |
| Logs pane・toolbar の表示の変更 | 既にある表示（160 §3.5）に意味を持たせるのが本 PRD である。表示を変える必要は見つかっていない。本当の制約ではなく、必要が出たら別 PRD で変えられる |
| 設定の値が指す名前の検証（`stack_keys` の名前、[issue-20261103-b754ff](../management/issues/issue-20261103-b754ff.md)） | 別の関心。値の参照先の検証で、設定キーが読まれたかとは別の仕組みになる。本当の制約ではなく、その issue で扱える |
| WARN と ERROR の間に新しいレベルを足す | レベルを増やすと G1 の線引きがぼやける。必要が出たら別 PRD で選べる（本当の制約ではない） |
| 過去の Run のログの読み替え | 本当の制約である。クリーンブレーク方針（過去の Run artifact は当時の記録として残し、読み替えのコードを足さない） |
| ERROR を出して続けている箇所のうち、設定の読み込みと metrics の定義以外（trainer.cpp 1020・1042 行、DropMergeEnv.cpp 900 行） | 別の関心。設定の失敗の扱いに範囲を限る。本当の制約ではなく、[issue-20261107-16ffd8](../management/issues/issue-20261107-16ffd8.md) で扱う |
| 黙って通している未知の値の洗い出し（`run.train.runner_type` のほか） | 別の関心。WARN を出していない箇所は §2.3 の棚卸しに入っておらず、洗い出しは監査の仕事である。本当の制約ではなく、見つかった件ごとに課題で扱える |

### 3.3 成功の測り方

- M1: §7 の A2 の Run のログで、`[W]` の行が 0 件。
- M2: 題材の事例（block の直下の `out_channels`、評価だけの RND の Run）と綴り違い（`mergin_x`）を再現する設定で、§5.6 のとおり ERROR・WARN・INFO になる。
- M3: §5.5 の表の全行が、表の扱いに移っている（表の行ごとのテストか検索で確かめる）。
- M4: 出荷設定の 7 本の Env の設定を既定の `run.$` で解決した結果にある、Run で実体化しない項目を持つカタログ（§5.3）の項目の全部で、読まれない設定キーが 0 件（§7 の A9）。
- M5: 現用の範囲に、「既定葉」「個別葉」「葉キー」が 0 件（§7 の A11）。

### 3.4 用語

本 PRD は次の 3 つの語を使う。「読まれない設定キー」と「在庫」は、[CONTEXT.md](../../CONTEXT.md) に足した。「既定キー」は、実装のときに用語集の「既定葉」を改める（D14）。

- **読まれない設定キー**: Run の実効設定に残った設定キーのうち、どの読み口も読まなかったもの。「未知キー」（AGENTS.md の語）は現行契約に無いキーを指すが、契約にあるキーでも持ち主が作られなければ読まれないので、別の語にする。設定キーは、設定の階層の末端にある `key = value` の 1 つを指す。起票時は「読まれない葉」と呼んでいたが、「葉」は leaf の直訳で日本語として通じにくいので、2026-11-11 に改めた（D14）。
- **既定キー**: `?=` で書く既定値の設定キー。今の用語集の語（「既定葉」）を改める（D14）。
- **在庫**: 読まれなくても誤りではない設定キー（D4）。[100](../design/100_runtime_and_configuration.jp.md) §2.1 の手順 6 が未選択のプロファイルを「休止した在庫」と呼ぶのを、選択の源・未参照の共有のカタログ項目・作らなかったクラスの設定・休止した prefix へ広げる。「dormant」は評価タグの状態の語なので使わない。

### 3.5 機構とゴールの対応

| 機構・概念 | 決定 | 効くゴール |
|---|---|---|
| ログの出し分けの規則 | D1・D8・D9・D10・D12 | G1・G2 |
| 読んだ設定キーの記録 | §5.2 | G3 |
| カタログ項目の閉じたスキーマの検査 | D3(a)・D13 | G3・G5（0c34dd） |
| Run の読まれない設定キーの一覧 | D3(b)・D6 | G3・G5（66100f・2c9a07） |
| 在庫と、持ち主の休止の宣言 | D4・D5 | G2（一覧と両立させる） |
| 1 回だけ出す共通の部品 | §5.1 | G1・G2 |
| 普通の Run で WARN 0 件を確かめる受入 | D2 | G2 |
| 止める失敗の追加 | D7・D11 | G4・G5（582880） |
| 設定の用語の言い換え | D14 | G6 |

## 4. 確定事項

2026-11-10 のグリルで決めた（D12 の、設定で無効にした機能の扱いと、D14 は、2026-11-11 に決めた）。決定と、起票時の未決表の行との対応は §9.1 にある。本 PRD は新しい設定キーを足さない。

### D1. ログのレベルの線引き

| # | 状態（条件） | レベル |
|---|---|---|
| 1 | 次の (a)〜(d) のどれか。(a) 機能を動かさないことだけを意味する設定値（有効フラグの偽、間隔の 0、schedule を持たない評価タグ）で利用者が機能を止め、その休止が起きた。または、止めた機能を参照する設定を飛ばした。(b) checkpoint を読み込むとき、checkpoint が持つ値を今の設定の値で置き換えた。(c) 明示した設定が、設定の外の値（環境変数）に勝った。(d) 選んだモードの定義そのものを説明する通知で、書いた値を捨てず、ほかの設定の効き目も弱めないもの | INFO。対象ごとに 1 回 |
| 2 | 次の (a)・(b) のどれか。(a) Run の構築が終わるまでに判定する、現行契約に反する設定・Run の入力・部品の呼び出しと、明示要求の不履行・非対応。(b) 今 `ANET_SYSTEM_ERROR` で止めている実行時の検査 | ERROR（`ANET_SYSTEM_ERROR` で止める） |
| 3 | 次の (a)〜(j) のどれか。(a) Run の読まれない設定キー（D3(b)）。(b) 書いた値が、ほかの設定（モードなど）のために使われない。(c) 動くが、ほかの設定や Run の形との組み合わせで、ある設定の効き目が無くなるか弱まる構成、書き忘れの可能性がある構成、または支えない組み合わせかどうかを部品が判定できず、利用者に確かめさせる構成。(d) auto 系の代替。(e) checkpoint を読み込むとき、設定が求める部分が checkpoint に無く、初期化した。(f) Run の構築のあとに起きて、学習と評価の値を変えずに、出力（メトリクス・描画・GUI・終了時の保存）の一部だけが欠ける異常。(g) 実行時のデータで起き、止めずに代替で続ける抽選。(h) 補助の操作・ファイル・出力先の失敗。(i) 実装の欠けで機能の一部を省いた。(j) 互換期間を明示して残した非推奨の設定を、利用者が使った | WARN。対象ごとに 1 回 |
| 4 | 1〜3 のどれにも当たらない状態 | INFO（知らせる必要が無ければ出さない）。開発用の診断は `ANET_LOG_DEBUG` |

行の当て方: 通知が知らせる状態を上から順に当て、最初に当たった行のレベルで出す。1〜3 は (a)〜の列挙が条件で、列挙に無い状態は当たらない。

- 1 の (a) は、休止したこと、または休止した機能への参照を飛ばしたことだけを知らせる通知に当たる。休止に加えて別の条件（読み込む checkpoint が無い、など）を知らせる通知は、その条件で 2 か 3 に当てる。
- 「Run の構築」は、設定の読み込み・部品の構築・checkpoint の読み込みを指し、最初の step の前に終わる。
- 「Run の入力」は Run の結果に効く入力（設定、データセット、checkpoint、ROM）を指す。workspace の履歴のような補助のファイルは含めない。
- 2 の (a) の「判定する」は、今構築時に判定しているものと、本 PRD が構築時の判定へ移すもの（D7・D11、§5.5 の「ERROR へ」の行）を指す。判定する作りが無いもの（終了時の保存で Agent が Save を持たないことなど）は、構築のあとに起きる 3 の (f) に当たる。
- 2 の (b) は、今 ERROR で止めている検査に限る。本 PRD は実行時の検査の ERROR を足しも減らしもしない。
- 1 の (d) と 3 の (b)・(c) の境目は、「書いた値を捨てるか、ほかの設定の効き目を弱めるか」である。
- 「対象ごとに 1 回」の対象と数え方は §5.1 による。
- 在庫（D4）の 1〜3 に当たる定義が使われなかったことは、どのレベルでもログに出さない。D4 の 4 の休止の宣言は、持ち主が 1 の (a) の INFO で出す。

例（条件ではない）:

- 1 の (a): `learner.enabled = false` の Run での RND の休止、dormant の tag を参照する scalar の skip。(b): optimizer の学習率が checkpoint と違う。(c): `backend.cuda_launch_blocking` が環境変数 `CUDA_LAUNCH_BLOCKING` に勝つ。(d): `feature_gradient = stop` の説明（skip の損失の勾配を共有部へ流さない、というモードの定義を述べるだけ）。
- 2 の (a): 未知の列挙値、実体化したカタログ項目の読まれない設定キー、互換性の無い組み合わせ、include 先の不在、同じ tag の二重登録。(b): Tensor の形の検査。
- 3 の (b): `max_length` を `max_length_mode = full` で書いた（値を捨てる）。(c): spatial のラダーを lane が 32 未満の Run で使う（ラダーの効き目が弱まる）。(e): skip 有効の設定で、skip の Network を持たない checkpoint を読む。(f): メトリクスの値が取れない、episode の乱れで描画を打ち切る、終了時の保存が 0 byte、GUI の配置の修復。(g): PER の rejection sampling が埋まらず、残りを一様に引く。(h): workspace の履歴が壊れている、クリップボードを開けない。(i): Env が設定の dump に対応していない。(j): 今は該当する設定が無い（[AGENTS.md](../../AGENTS.md) の設定値の扱いの 208 行が定める通知）。

理由: Logs の未確認の警告は WARN 以上を数える（§2.2）ので、WARN を「手を打つべき状態」に限れば印に意味が出る（G1・G2）。1 を先に当てるのは、列挙した (a)〜(d) は利用者の選択どおりに起きたことで、手を打つことが無いため。2 の (a) を構築時に限るのは、[AGENTS.md](../../AGENTS.md) の「起動時・構築時に検証できるものを実行時まで潜伏させない」（197 行）を守りつつ、構築のあとの出力の側の異常で長い Run を落とさないため。3 の (e) を WARN にするのは、意図した移行か checkpoint の取り違えかを部品が見分けられず、取り違えると学習していない Network で評価した結果が黙って出るため（D10）。3 の (g) を名指しするのは、抽選の分布が変わるので (f) の「値を変えない異常」に入らないため（D9）。4 を INFO にするのは、行に当たらない状態を WARN にすると、WARN の意味がまた広がるため。依存: なし。

今の規約との差: [AGENTS.md](../../AGENTS.md) の Fail-Fast の表（190 行）は「契約違反の内部状態」と「壊れた入力データ」を一律に fail-fast とするが、今の作りは、構築のあとの GUI の配置の修復（PRD110）や描画の打ち切りを WARN で続け、workspace の履歴の壊れも WARN で続けている（§2.3 の D 区分の 18 行）。D1 はこの作りを 2 と 3 の境目として書き、表の 190 行をこの条件へ改める（§5.8）。表の 191 行（意図された休止）は 1 の INFO へ改める。

### D2. 普通の Run の WARN を 0 件に保つ

- 「普通の Run」は、意図どおりに組んだ構成の Run を指す。受入では §7 の A2 の Run（出荷設定の各 Env の既定の `run.$`、番号付き bat の Run の組み合わせ、Atari の `run.@evalonly`）で確かめる。
- A2 の Run で WARN が 0 件であることを、本 PRD の受入にする。
- あとの実装が WARN・INFO を足す・変えるときの確かめを、[実装 品質確認観点](../quality/impl.md) の「失敗の扱い」に 1 項目足す。足した・変えた WARN と INFO が D1 のどの行に当たるかを示し、WARN は、意図どおりに組んだ構成の Run（関係する Env の既定の Run 1 本）で出ないことを確かめる。

理由: 普通の Run で 1 件でも出れば、印が常に付き、D1 の線引きが効かない（G2）。項目を実装の観点に置くのは、WARN の多くが PRD ではなく実装の中で足されるため。依存: D1。

### D3. 読まれない設定キーを 2 つの仕組みで検出する

- (a) カタログ項目の閉じたスキーマ: カタログの項目を実体化する読み口は、その項目の設定キーのうち読まなかったものがあれば、構築時に `ANET_SYSTEM_ERROR` で止める。設定キーが `?=` か `=` かによらない。対象のカタログは §5.3 の表。実体化しなかった項目は、Run では検査しない。その設定キーは、共有のカタログなら D4 の 2 の在庫になり、1 つの持ち主の下だけで参照される定義（branch の config_profile）なら (b) の一覧に出る。
- (b) Run の読まれない設定キーの一覧: Run の構築を終えた時点で、実効設定にある設定キーのうち、どの読み口も読まず、在庫（D4）にも当たらない設定キーを、1 件の WARN に一覧して出す（§5.4）。Run は止めない。

場面ごとの挙動（block の例）:

| 場面 | 起票時 | 本 PRD |
|---|---|---|
| block を定義したが、どの structure にも出てこない | 何も出ない | 何も出ない（在庫） |
| block を使い、書いた設定キーをその種類の module が全部読む | 何も出ない | 何も出ない |
| block を使ったが、その種類の module が読まない設定キーが混じっている（綴り違い、階層の誤り） | 何も出ない。その設定キーは黙って無視される | 構築時に止まる |

(a) の帰結:

- 上書き層や Run プロファイルで、既にある block の `type` だけを別の種類に書き換える書き方は、元の種類の設定キーが残るので止まる（設定には設定キーを消す書き方が無い）。別の名前の block を定義し、structure の側で差し替える。出荷設定にこの書き方は 0 件（§2.6）。
- カタログの読み口は、今のモードで使わない設定キーも読んで検証する。使わない設定キーがあることでは止めない（書いた値が今のモードで効かないことを知らせる既存の WARN は D1 の 3 の (b) として残る）。

理由: (a) の項目の設定キーはその読み口だけが読むので、局所の情報だけで判定でき、[AGENTS.md](../../AGENTS.md) の表の「閉じたスキーマ内の未知フィールド」に当たる（汎用機構と利用側の責任境界）。Network の形の誤りは Run を 1 本まるごと無駄にするので、止める価値が一番高い。0c34dd を直す（G3・G5）。(b) の綴り違い・階層違い・クリーンブレーク後の旧キーは、持ち主を名前から決められない（`mergin_x` の上の `Conv2dPanel.conv2d` は読み口があるが、`Conv2dPanel.conv2d.mergin_x` を誰が読むはずかは書いた人にしか分からない）ので、全体の差でしか拾えない。(b) を止めずに WARN にするのは、在庫の種類で足りるかを Run で確かめる前（§9 の 5）に ERROR にすると、正しい設定の Run が起動できなくなる恐れがあるため。弱点は、夜間の bat では印を見る人がおらず、誤った設定のまま走りきることである（ERROR にする条件は §8）。使わない定義を Run で検査しないのは D13 による。依存: D4、D13。

### D4. 在庫（読まれなくても一覧に出さない設定キー）

次の表を上から順に当て、最初に当たった行で在庫とする。どの行にも当たらず、どの読み口も読まなかった設定キーが D3(b) の一覧に出る。

| # | 在庫（条件） |
|---|---|
| 1 | 解決の入力にある選択の宣言のどれかに、選択の源として現れた通常 prefix の設定キー。あとの宣言に置き換えられた宣言と、選ばれなかった Run プロファイルの中の宣言も含む |
| 2 | 共有のカタログ（どこからでも名前で参照できるもの）を走査した読み口が、参照されなかった項目として見送った設定キー。1 つの持ち主の下だけで参照される定義の未参照の項目は在庫にしない |
| 3 | 登録済みのクラスのうち、この Run で作らなかったクラスの設定キー |
| 4 | 持ち主が休止を宣言した prefix の設定キー。持ち主は休止を INFO で 1 回出す |

例（条件ではない）: 1 は上書き層 `A1.*`・`M1.*` と、選択の宣言に現れた mode の prefix（Env の設定が `app.$ = app.online > P1` と書き、bat が `app.$=app.batchrun>P1` で置き換えた Run の `app.online.*` と `app.batchrun.*`）。2 は構造に現れない `net.block.[<名前>].*` と参照されない `nn_forward.[<名前>].*`（在庫にしない側の例は、その branch だけが参照する `net.branch.[<b>].config_profile.[<名前>]`）。3 は Atari の DefaultDQN の Run の `RainbowAgent.*`・`MuZeroAgent.*`・`ImageClsAgent.*`。4 は `DefaultDQNAgent.intrinsic.enabled=false` のときの `DefaultDQNAgent.intrinsic.*`。

既定キー `?=` で残った設定キーも、`=` や CLI の設定キーと同じに一覧の対象に含める（在庫にしない）。

理由: 1・2 は PRD059 が「機械識別できる非実効行」とした種類で、解決とカタログの読み口が既に知っている。1 を選ばれた宣言だけに限らないのは、選ばれなかった側の設定キーも実効設定に残るためである（§2.6）。どの宣言にも現れない prefix に設定キーを書いたまま選び忘れた場合は、一覧に出る。名前の形（文字＋番号）では判定しない（PRD072 が上書き層の予約名による判定を不採用にしたのに合わせる。[316](../wiki/categories/316_config.md) の選ばなかった形）。2 の除外は、今も branch が使わない config_profile に WARN を出している（nn_impl.cpp 1838 行）のに合わせる。3 は Agent・Env の登録が知っている。4 はモードで使わない設定キーの持ち主だけが知っている。`?=` を含めるのは、綴り違いの `mergin_x` が `?=` で書かれており（§2.4）、外すと共通の設定の綴り違いと旧キーが黙って残るため。また、`?=` の情報は解決のあと捨てているので（§2.4）、含めないほうが配線が増える。普通の Run での誤検出は D2 の受入で 0 件にする（持ち主が読む、休止を宣言する、設定を直す）。依存: D3、D5。

### D5. 持ち主は自分の設定キーを構築時に読むか、休止を宣言する

- 遅れて作る読み口（View の menu で開く panel など）は、Run の構築時に設定を読んで検証する。
- モードで使わない設定キーを持つ持ち主は、その設定キーも読んで検証するか、D4 の 4 の休止を宣言する。どちらにするかは持ち主ごとに実装計画で決める。
- 例（条件ではない）: RunnerApp は、どの選択の宣言にも現れない mode の prefix（Env の既定の online の Run での `app.batchrun.*`）を、読んで検証するか、休止を宣言する。

理由: D3(b) の一覧の時点で読まれていない設定キーは、一覧に出る。また、[AGENTS.md](../../AGENTS.md) の「検証は必要な情報が揃う最も早い自然な境界で行う」（154 行）に合う（`Conv2dPanel` の設定の誤りは今は panel を開くまで分からない）。依存: D3、D4。

### D6. NoCare をやめる

[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針の「設定基盤が未知キーを一律検出しない場合は NoCare」（85・93 行）を、「旧キーは D3 の汎用の検出で、ほかの読まれない設定キーと同じに扱われる」に改める。旧キー専用の WARN・検出・tripwire を作らない決まりは変えない。`WarnDeprecatedRunsDirConfig` は消す（2c9a07）。

理由: 方針の「現在の汎用機構が未知識別子を検出できる場合は通常の未知識別子として扱う」（85 行）に、D3 の汎用の検出が当たる。旧キーだけを特別に扱わないので、クリーンブレーク方針と両立する（G3）。依存: D3。

既存の判断との食い違い: [ADR 0027](../adr/0027-eval-definition-schedule-separation.md) の「旧位置キーの検出は NoCare」は、D3 の検出で変わる。評価の定義の中に残った旧位置のキー（`run.eval.[<tag>].interval` など）は、実体化した項目の読まれない設定キーとして構築時に止まり（D3(a)）、ほかの場所に残った旧キーは一覧に出る（D3(b)）。旧位置キー専用の検出を作らない判断は残る。PRD001 の D14（旧キーの tripwire を作らない。§2.5）も、専用の検出を作らない点は残り、旧キーが汎用の検出で見つかる点だけが変わる。ADR 0027 の先頭の注記に書き、判断は [ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) に書く。

### D7. 明示した入力の不在と、未知の値を止める

- `$include` 先が無い、include 行のファイル名が空・解析できない、読むように指定されたファイル（主の設定、重ねる設定、include 先）が開けない場合は、理由によらず `ANET_SYSTEM_ERROR` で止める（config.cpp 555・605・626・632・640 行）。
- metrics の定義の未知の event・step・target・属性（observers.cpp 1374・1379・1384・1404・1633・1640 行）は、`ObserverFactory` の構築時に `ANET_SYSTEM_ERROR` で止める。
- DefaultDQNAgent の観測の正規化の `post_process_type` は、Agent の構築時に値域を検証する（scaler_impl.cpp 420 行の毎回の WARN を消す）。
- RainbowAgent の `use_qr=true` かつ `num_quantiles<=1` は、構築時に `ANET_SYSTEM_ERROR` で止める（rainbow_agent.cpp 54 行の読み替えを消す）。
- 同じ tag の ImageProvider の二重登録は、`ANET_SYSTEM_ERROR` で止める（image.cpp 671 行）。
- `run.train.runner_type` は、`serial` と `pipeline`（今と同じく大文字小文字を無視して比べる）のほかの値を、構築時に `ANET_SYSTEM_ERROR` で止める（582880）。

理由: どれも [AGENTS.md](../../AGENTS.md) の表の「意図しない状態」か「明示要求の不履行」に当たる（G4）。include の不在で止めても、出荷設定と、無くてよいファイルの読み取りは止まらない（§2.6）。`run.train.runner_type` は WARN を出していないが、未知の列挙値を止めるという同じ直しなので、一緒に入れる。黙って通しているほかの未知の値は洗い出さない（§3.2）。`post_process_type` と、metrics の定義の event・step・target・属性の値域は、今のコードが受け付ける値のままで、足しも減らしもしない。依存: D1。

### D8. 題材の a9a0e3 の扱い

- dormant tag を参照する scalar の skip と、結び付け先の無い trace の skip は INFO にする（trainer.cpp 866 行）。文言と、tag ごとに 1 回であることは変えない。`eval.[<tag>]: definition-only` の INFO は別の対象（定義の側）なので、両方残す。
- `interval = 0` で止めた tag と、schedule の記述が無い tag を分けない（どちらも dormant で INFO）。
- 評価だけの Run の RND: learner を止めた Run での休止の通知は INFO にする（default_dqn_agent.cpp 119 行）。`learn_batch_size` と lane の数の検査（D11）は、補助学習が動くとき（learner が有効）だけ行う。checkpoint の読み込みの検査（同 406〜409 行）は変えない。

理由: どれも利用者が意図して組んだ状態で、D1 の 1 の (a) に当たる（G2・G5）。2 つの止め方を分けないのは、[ADR 0027](../adr/0027-eval-definition-schedule-separation.md) が `interval = 0` を schedule のエントリが無いのと同じ挙動と決めており、分けると休止の出どころを運ぶ分岐が増えるため。分けると、定義だけの tag を意図して置く使い方（EvalPanel 用の tag）でも WARN が出てしまう。`learn_batch_size` の検査は、補助学習が休止していれば意味を持たない。依存: D1、D11。

既存の判断との食い違い: dormant の skip を INFO にするのは、ADR 0027 の Consequences の「metrics の dormant タグ参照 WARN」の行と、[ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) の「scalar は現行どおり dormant タグでは WARN して捨てる」の行のレベルの部分を改める（skip する振る舞いと文言は残る）。PRD052 の D9 と PRD081 の 92 行が定めたレベルも置き換わる（両 PRD の本文は当時の記録として書き換えない）。両 ADR の先頭へ注記し、判断は [ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) に書く。

### D9. 既存の WARN の扱い

§5.5 の表のとおり、各行を D1 の線引きで「WARN のまま」「INFO へ」「ERROR へ」「消す」「1 回にする」のどれかへ移す。PER の rejection sampling が埋まらないときに残りを一様に引く代替（replay_buffer_impl.cpp 1213 行）は残し、WARN を Run ごとに 1 回にする。

理由: G1・G2・G4。表の行ごとの理由は表に書く。PER の代替を残すのは、§2.1 の 616 本で 1 度も起きていない保険で、起きる条件が分からないまま止める処理に変えると、正当な極初期の状況で Run を落とす恐れがあるため。実行時の検査の ERROR を足さない（D1 の 2 の (b)）こととも合う。代替そのものの是非は本 PRD の範囲外とする（§9 の 20）。依存: D1、D7、D8、D10、D11。

### D10. checkpoint の読み込みの通知

- checkpoint が持つ optimizer の設定（学習率・weight decay・eps・betas など）が今の設定と違い、今の設定で置き換えたことの通知は INFO にする（dqn_based_agent.cpp 4270〜4295 行、intrinsic_rnd.cpp 498 行）。項目ごとの値は今と同じく出す。
- 設定が求める部分が checkpoint に無く、初期化したことの通知は WARN のままにする（default_dqn_agent.cpp 440 行の skip の Network、dqn_based_agent.cpp 4252 行の `primary_skip` の optimizer の状態）。

理由: 置き換えは [ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) が決めた「今の設定が勝つ」契約どおりの結果で、利用者が直すものが無い（D1 の 1 の (b)）。学習率を変えて続きから学習する Run や、古い checkpoint を今の設定で評価する Run で、毎回印が付くのを避ける。初期化は、基準の Run から skip の学習へ移る使い方（[200](../design/200_dqn_agents.jp.md) §8.2 の 602・606 行）と、skip 有効の設定で skip の Network を持たない checkpoint を取り違えて読んだ場合が、同じ通知になる。取り違えると学習していない Network で行動した結果が黙って出るので、WARN で確かめさせる（D1 の 3 の (e)）。依存: D1。

既存の判断との食い違い: 置き換えの通知を INFO にするのは、ADR 0053 の「checkpoint と違った項目は WARN で示す」のレベルの部分を改める（options を今の設定で置き換える判断は残る）。ADR 0053 の先頭へ注記し、判断は ADR 0071 に書く。

### D11. `intrinsic.learn_batch_size` が lane の数より大きい構成は止める

RND の補助学習が動く Run（learner が有効）で、`DefaultDQNAgent.intrinsic.learn_batch_size` が `run.train.num_envs` より大きいときは、構築時に `ANET_SYSTEM_ERROR` で止める（キー、値、lane の数を示す）。1 step の件数は常に `learn_batch_size` とし、PRD116 の D7 が定めた「lane の数に縮める」（k = min(`learn_batch_size`, N)）をやめる。learner が無効の Run では検査しない（D8）。

理由: Run の設定は dump（`config/config_data.txt`）を真実として読むので、dump の値（32）と実際の件数（N）が食い違う経路を残さない。[AGENTS.md](../../AGENTS.md) の「指定値を勝手に丸めない」（207 行）と「範囲外を clamp しない」（212 行）にも合う。出荷設定で RND を使う Run の lane は 64 以上で、影響を受けるのは lane を 32 未満に落として RND を試すときだけである（引数で `learn_batch_size` を明示する）。依存: D1、D8。

既存の決定との関係: PRD116 の D7・§5 の表（285・369 行）が定めた縮小と WARN を置き換える。PRD116 の本文は当時の記録として書き換えず、設計文書 212・200 を直す（§5.8）。

### D12. 休止した機能の値は、鍵を保って NaN で返す

- 宣言した scalar の値が取れないときの Observer の WARN（`value not found`、observers.cpp 1115 行）は、tag ごとに 1 回にして残す（D1 の 3 の (f)）。
- 休止を持つ部品は、休止中も、宣言した値の鍵を消さずに `NaN` を返す。`GetScalar()` の今の契約（未知の鍵は値なし、既知の鍵で値が成立しないときは `NaN`。§2.6）を、休止にも当てる。
- 機能そのものを設定で無効にして、部品を作らない場合も同じに、その機能の値の鍵を既知の鍵として `NaN` を返す。140 の決まり（238 行）のとおりで、PER は既にそうなっている（§2.6）。今この決まりから外れているのは RND で（`DefaultDQNAgent.intrinsic.enabled = false` のとき鍵が無い）、本 PRD で直す。[212](../design/212_rnd_lifelong_novelty.jp.md) と [200](../design/200_dqn_agents.jp.md) が「RND が無効なら鍵は値なし」としているのを改める。
- ほかに外れている機能が A2 の Run で見つかったら（無効にした機能の鍵の `value not found` の WARN として現れる）、同じに直す。150 の 314 行の古い記述は、実装に合わせて直す（§5.8）。

理由: Observer は値が取れない原因を知らないので、休止が原因のときと、鍵の綴りを誤ったときが同じ WARN になる。休止は出す側が知っているので、出す側が「休止中で値が無い」と分かる形で返せば、Observer の WARN は本物の誤り（鍵の綴り違い、出す側の不具合）だけに残る（G1）。値が取れないことを INFO にする形は、本物の誤りを INFO に埋めるので採らない。RND を無効にした場合も同じにするのは、RND を使う構成のまま引数で RND だけを切った対照の Run（`run.@rnd` を付けて `intrinsic.enabled=false`）で、宣言した scalar の tag ごとに WARN が出るのを避けるためである。dormant の tag を参照する scalar を INFO で飛ばす（D8）のと同じ考え方で、PER の実装のコメントも同じ理由を書いている（PER を切っただけで metrics の定義を書き換えずに済むように）。未知の鍵は値なしのままなので、綴り違いの検出は変わらない。依存: D1。

既存の決定との関係: PRD116 が定めた「RND 無効時のキーは値なし」（212 の 442 行、200 の 633 行）を置き換える。PRD116 の本文は当時の記録として書き換えず、設計文書 212・200 を直す（§5.8）。

### D13. 使わない定義の誤りは、出荷設定の検査のテストで見る

- 定義しただけで使わなかったカタログの項目の読まれない設定キーは、Run では検査しない（D3(a)）。その誤りは、項目を初めて使う Run の構築時に止まって見つかる。
- 出荷設定の検査のテスト（タグ `[shipped_config]`）を足し、Run で実体化しない項目を持つカタログ（§5.3 の表。block、config_profile、NN 実行設定）について、出荷設定の 7 本の Env の設定を既定の `run.$` で解決した結果にある項目を、使われるかどうかによらず全部、読み口に読ませて、読まれない設定キーが無いことを確かめる（§7 の A9）。選ばれなかった Run プロファイルの中だけで足される設定キーは対象に入らず、その Run プロファイルを使う Run で項目が実体化したときに、構築時に見つかる。
- コミットの準備などに、このテストを流す手順は足さない。

理由: 使う Run の構築時には必ず止まるので、誤った設定で Run が回る経路は無い。遅れて失うのは起動のやり直し 1 回である。Run のたびに全項目を検査する形は採らない（§8）。block の module は設定だけで作れ、最初の入力が来るまで重みを確保しないので（§2.6）、使わない block も作って捨てれば検査できる。しかし、Run の構築の経路に 3 つの扱いが増える。検査のために作る module が Run の乱数を消費しないようにすること、補間の印（config_profile）が入った設定キーに仮の値を入れること、使わない block の未知の `type` の扱いである。誤りを含む書きかけの block が印を付け続けるか、関係のない Run を止めることにもなる。害の経路が無い誤りを早く見つけるために、同じ seed の再現性を崩す種を足さない。テストは自動では流れない（§2.6）ので、人が手で直した設定の未使用の定義の誤りは、次にテストが流れるか、その定義を使うまで残る。害の経路が無いので、この窓を狭める手順は足さない。このテストは、どの Run も作らない項目を見るためのもので、[実装 品質確認観点](../quality/impl.md) の「Run の構築時に止まる検査を前倒しするだけのテストは作らない」（417 行）には当たらない。依存: D3。

### D14. 設定の用語から「葉」をなくす

| 今の語 | 改めた語 |
|---|---|
| 葉、葉キー（設定の階層の末端にある `key = value` の 1 つ） | 設定キー |
| 既定葉（`?=` で書く既定値） | 既定キー |
| 個別葉 | 個別指定（用語集の今の正式な語に一本化する） |
| 葉値 | 設定キーの値 |
| 確定葉 | `=` で書いた設定キー |
| 数え方の「N 葉」 | N 件 |

- 直すのは現用の範囲である。[実装 品質確認観点](../quality/impl.md) の項番 71 が定める範囲（コード、テスト、設定、bat、`docs/design/` の jp と en、`CONTEXT.md`、wiki の実装の地図）に、規約・品質確認観点・プロセス・skill（`AGENTS.md`、`docs/quality/`、`docs/process/`、`.claude/skills/`、`.agents/skills/`）を足したもの。
- 過去の記録（`docs/memo/` のほかの PRD と実装計画、`docs/experiments/`、`docs/adr/`、`CHANGELOG.md`、wiki の知見のページ、Run の出力とその写し）は直さない。課題の題と本文も直さない（wiki の地図の「課題」の区画は、課題の題から道具が作る）。
- `core/anet-core/testdata/` は、README・スクリプト・検証の記録を直す。Git が無視するファイル（Run の設定の写しなど）は直さない。
- 英語の名前（`check_default_leaves.py`、コードの識別子、英語のログ）は変えない。
- [AGENTS.md](../../AGENTS.md) の応答言語ルールの「避ける語」の表に、「既定葉」「個別葉」「葉キー」を検索する行として足す。単独の「葉」は「言葉」と紛れるので、検索しない行として足す。
- 設定のキーと値、振る舞いは変えない。

理由: 「葉」は英語の leaf の直訳で、日本語では階層の末端の意味を持たず、読んで意味が取れない（G6）。「設定キー」は規約と用語集が既に使っている語で、新しい語を増やさない。「既定キー」は「設定キー」に寄せた短い語である。本 PRD に入れるのは、本 PRD の実装が同じ文書（AGENTS.md、品質確認観点、CONTEXT.md、020、100、212）を直すので、まとめて済むためである。依存: なし。

## 5. 実装契約

### 5.1 ログのレベル（D1）

- WARN と INFO の文言は、今の規約どおり、対象の識別子（キー・tag・パス）、指定値か必要量、期待値か上限を含める（[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則の 200 行）。
- 「対象ごとに 1 回」は、Run ごとに数える。同じプロセスで Run を作り直したら、また 1 回出る。NN の複製（target・Actor 用・skip・RND）で同じ構築を繰り返しても、同じ対象は 1 回にする。
- 「対象」は、通知が指す設定のキー・tag・ファイル・部品を指す。利用者の操作の失敗は、操作 1 回が対象で、操作のたびに出す。
- 1 回だけ出す共通の部品を 1 つ置く。
  - 同じ通知かどうかは、通知を出すソースの箇所（呼ぶ側が書かなくても決まる）と、呼ぶ側が任意で渡す対象の名前の組で決める。対象の名前を渡さない箇所は、箇所ごとに 1 回になる。対象ごと（tag ごと、キーごと）に出す箇所は、対象の名前を渡す。
  - 出した記録は、Run を作る側が、Run を作り始めるとき（部品を作る前）に空にする。
  - WARN と INFO の両方で使える。複数のスレッドから呼べる。
  - §5.5 で「1 回」とした行と、本 PRD が足す通知は、この部品を使う。§2.2 に挙げた個別のフラグと集合も、この部品に置き換える。
- 休止の宣言の INFO は、持ち主の prefix と、休止の理由になった設定のキーと値を含める。
- 足す・変える通知の文面は英語で書く（[実装 品質確認観点](../quality/impl.md) の「ログ」）。
- ログのレベルの名前、Run のログの `[W]`・`[I]` の書式、Logs pane の数え方は変えない。

### 5.2 読んだ設定キーの記録

- Run の実効設定から読み口へ渡した設定は、どの読み方（`Config` の読み取り、`ConfigData` の直接の `Get`・`Read`、カタログの項目へ分けた設定、複製した設定）で読んでも、元の完全なキーで「読んだ」と記録される。
- キーの有無を見るだけの問い合わせ（`Has`）は、読んだに数えない。
- 全キーを走査するカタログの読み口は、実体化した項目の設定キーを読み口（factory など）に読ませる。実体化しなかった項目は、共有のカタログ（D4 の 2）なら在庫として記録し、1 つの持ち主の下だけで参照される定義（`net.branch.[<b>].config_profile.[<名前>]`）なら在庫として記録しない（その設定キーは読まれない設定キーとして残り、§5.4 の一覧に出る）。走査しただけでは読んだに数えない。
- 同じ項目の同じ設定キーを、より局所の定義が上書きしたとき（全体の `net.block.[<名前>].*` を Agent の下の同じ block の設定キーが上書きする、全体の config_profile を branch の config_profile が上書きする）、上書きされた側の設定キーは読んだに数える。
- 解決は、入力にある選択の宣言のどれかに選択の源として現れた通常 prefix を、D4 の 1 として記録へ渡す。
- 記録は、複数のスレッドから届けられる（RunManager は裏のスレッドで、アプリの側の構築は main のスレッドで設定を読む。§2.6）。
- 一覧を出したあとに起きた読み取りは、一覧に影響しない。一覧の時点までに読むのは持ち主の義務である（D5）。
- 記録は Run の構築のあいだだけ伸び、学習のホットパスでは読み取りが起きない（§5.7）。

### 5.3 閉じたスキーマの検査の対象（D3(a)）

| カタログ | 項目を実体化する読み口 | Run で実体化しない項目 | 読まれない設定キーがあるときのエラーに含めるもの |
|---|---|---|---|
| `net.block.[<名前>]` | module の factory | structure に現れない block（在庫） | block の名前・`type`・読まれない設定キーとその値・その type が読むキーの一覧 |
| `net.config_profile.[<名前>]` | config_profile の読み口 | どの branch も使わない項目（在庫） | 項目の名前・読まれない設定キー |
| `net.branch.[<b>].config_profile.[<名前>]` | config_profile の読み口 | その branch が使わない項目（在庫にせず、設定キーは §5.4 の一覧に出る） | 同上 |
| `<Agent>.actor.[<名前>]` | Actor 設定の読み口 | 無い（定義した全部を構築時に読む） | 同上 |
| `<Agent>.nn_forward.[<名前>]` | NN 実行設定の読み口 | どこからも参照されない項目（在庫） | 同上 |
| `run.eval.[<tag>]`、`run.eval_schedule.[<tag>]` | configured eval の定義・schedule の読み口 | 無い（dormant の tag も宣言の検証で読む） | 同上 |
| `metrics.scalar.[<tag>]`、`metrics.trace.[<tag>]` | `ObserverFactory` | 無い（定義した全部を読む） | 同上 |
| `metrics.image.[<tag>]` | `ImageProviderManager` | 無い（定義した全部を読む） | 同上 |

- 表に無いカタログが見つかったら、実装計画で同じ扱いに足す（[CONTEXT.md](../../CONTEXT.md)「カタログ」の定義に当たるもの）。
- Run で検査するのは実体化した項目だけである。表の「Run で実体化しない項目」は、Run では検査せず、出荷設定の検査のテスト（D13、§7 の A9）で見る。「無い」と書いたカタログは、定義した全部の項目が Run で検査の対象になる。どの読み口がどちらに当たるかの根拠は §2.6 にある。
- 読み口は、今のモードで使わない設定キーも読んで検証する（D3 の (a) の帰結）。

### 5.4 Run の読まれない設定キーの一覧（D3(b)）

- 時機: Run の構築を終えた直後、最初の step より前に 1 回。ここでの Run の構築は、RunManager（Agent、Env、Observer を含む）の構築と、そのあとにアプリが行う構築（RunnerApp では、画像の metrics の登録と RunnerFrame の panel の初期化）の両方を指す。online と batchrun で同じ。
- 出す側: 一覧を出す口は core に 1 つ置き、Run を作る側が構築の終わりに 1 回呼ぶ。RunnerApp は、アプリの側の構築を終えて Train を始める直前に呼ぶ。RunManager の構築の終わりでは、アプリの側が読む設定キー（`metrics.image.*`、`app.eval_trace_panel.*` など）がまだ読まれていない（§2.6）。
- 対象: Runner が `config/config_data.txt` に保存する実効設定にある設定キーのうち、§5.2 で読んだと記録されず、D4 の在庫にも当たらない設定キー。`?=` の設定キーと、CLI で与えた設定キーも含む。
- 出し方: 1 件の WARN。1 行目は `Unread config keys: N. These keys were not read by any component.`（N は件数）。続く行に `key = value` をキーの順に 1 行ずつ。行数の上限は置かない。0 件なら出さない。
- 一覧に出た設定キーがあっても Run は止めない。一覧を Run の `json/` には保存しない。

### 5.5 既存の WARN の扱い（D9）

「残す」は WARN のまま。どの行の通知も、D1 のとおり対象ごとに 1 回で出す（§5.1）。今 2 回以上出ることを確かめた行に「1 回」と書いた。「残す」だけの行も、実装で同じ対象に 2 回以上出ると分かったら 1 回にする。利用者の操作ごとの失敗（LogPanel.cpp 658 行）だけは、操作のたびに出す。「D1 の行」は、その通知に当てた D1 の行である。行番号は起票時点。

| 場所 | 文言（要点） | 区分 | 扱い | D1 の行 | 理由 |
|---|---|---|---|---|---|
| LogPanel.cpp 658 | Log copy failed: the clipboard could not be opened. | F | 残す（操作のたびに出す） | 3 の (h) | 補助の操作の失敗。操作 1 回が対象（§5.1） |
| RunnerApp.cpp 407 | Last workspace is unavailable; falling back to _default. | C | 残す | 3 の (d) | 自動の選択の代替 |
| RunnerApp.cpp 131〜145・414 | Config key app.metrics_logger.runs_dir／root_dir is deprecated and ignored. | E | 消す | 通知を消す（旧キーは 3 の (a) の一覧に出る） | D6 |
| RunnerApp.cpp 732 | Agent save produced zero bytes; … | D | 残す | 3 の (f) | 終了時の保存が 0 byte で、止めずに続ける。MuZero の試作の Agent は Save で 0 を返す（[muzero_proto_agent.hpp](../../core/anet-core/include/anet/muzero_proto_agent.hpp) 184 行）ので、`GridMaze_muzero.txt` は終了時の保存を切る（§5.8） |
| RunnerFrame.cpp 1029・1036 | No main pane occupies the centre. … / Both main panes occupy the centre. … | D | 残す・1 回 | 3 の (f) | 想定外の経路を GUI の配置の修復で正しい状態へ戻し、Run の結果は変わらない（PRD110）。修復のたびに出ている |
| default_dqn_agent.hpp 234 | use_optimistic_target=true copies … without optimism | B | 残す | 3 の (c) | 効き目が無くなる組み合わせ |
| default_dqn_agent.hpp 381 | learner.use_tbo … double-compressed | B | 残す | 3 の (c) | 同上 |
| default_dqn_agent.hpp 484 | learner.enabled=false without auto_load_file; … | B | 残す | 3 の (c) | 書き忘れの可能性（休止だけを知らせる通知ではない） |
| default_dqn_agent.hpp 542 | … skip.max_length active_range=… is unused with max_length_mode=… | A | 残す | 3 の (b) | 書いた値が今のモードで効かない |
| default_dqn_agent.hpp 557 | … skip.nn_forward_key=… is unused with skip.nn_forward_mode=shared | A | 残す | 3 の (b) | 同上 |
| default_dqn_agent.hpp 590 | learner.skip.feature_gradient=stop with target_window=…: skip loss gradients do not reach shared features | B | INFO へ | 1 の (d) | `stop` は選べるモードで、文言はその定義の説明（PRD104 の STOP の Run で毎回出る） |
| default_dqn_agent.cpp 110 | intrinsic.learn_batch_size=N exceeds num_envs=M; … | C | ERROR へ（補助学習が動くときだけ検査する） | 2 の (a) | D11。縮める処理を消す |
| default_dqn_agent.cpp 119 | intrinsic.enabled=true with learner.enabled=false: … suspended | A | INFO へ | 1 の (a) | D8 |
| default_dqn_agent.cpp 440 | Checkpoint has no skip networks; … initialized skip networks … | D | 残す | 3 の (e) | D10 |
| dqn_based_agent.cpp 269・273・282・286 | PER の `per_eps`・`actor_approx`・clip の組み合わせ | A・B | 残す | 3 の (c) | 優先度の効き目が無くなるか弱まる組み合わせ |
| dqn_based_agent.cpp 1525 | ActionPolicy spatial exploration is enabled with num_envs < 32. | B | 残す・1 回 | 3 の (c) | ActionPolicy を作るたび（Actor・skip ごと）に出ている |
| dqn_based_agent.cpp 2835 | Gradient noise subscription intervals differ: … | B | 残す | 3 の (c) | 効き目が弱まる購読 |
| dqn_based_agent.cpp 2864 | Policy churn metrics observe one hard-target phase: … | B | 残す | 3 の (c) | 同上 |
| dqn_based_agent.cpp 4252 | Learner::Load: checkpoint has no primary_skip optimizer role; initialized moments … | D | 残す | 3 の (e) | D10 |
| dqn_based_agent.cpp 4270〜4295 | Learner::Load: optimizer option differs from checkpoint; config applied. … | D | INFO へ・1 回 | 1 の (b) | D10。ADR 0053 の「WARN で示す」を改める |
| intrinsic_rnd.cpp 258 | Ignored … multiplier_max=… with multiplier_mode=unclipped | A | 残す | 3 の (b) | 書いた値が効かない |
| intrinsic_rnd.cpp 279〜293 | Ignored … obs_norm.<key>=… with statistics_mode=variance_feedback | A | 残す | 3 の (b) | 同上 |
| intrinsic_rnd.cpp 498 | RND checkpoint optimizer options replaced: … | D | INFO へ | 1 の (b) | D10 |
| init.cpp 76 | CUDA_LAUNCH_BLOCKING overridden by backend.cuda_launch_blocking. … | B | INFO へ | 1 の (c) | 明示した設定が環境変数に勝った |
| app_util.cpp 453・467 | Workspace history could not be read. … / … entry ignored. … | D | 残す | 3 の (h) | 補助のファイルの壊れ（Run の入力ではない） |
| app_util.cpp 581・587 | Failed to redirect stdout／stderr to … | D | 残す | 3 の (h) | 補助の出力先の失敗 |
| config.cpp 555・640 | Properties: Failed to open file／include file. … | A | ERROR へ | 2 の (a) | D7 |
| config.cpp 605・626・632（ERROR を出して続ける） | include 行の解析の失敗 | — | ERROR で止める | 2 の (a) | D7 |
| image.cpp 671 | ImageProvider tag '<tag>' already exists. | D | ERROR へ | 2 の (a) | 構築時の同じ tag の二重登録は、部品の契約に反する呼び出し（D7） |
| metrics_logger.cpp 327 | … Retrying with libx264. | C | 残す・選択を固定して 1 回 | 3 の (d) | AGENTS.md の表の「以後は選択を固定」 |
| nn_impl.cpp 1838 | net.branch.[b].config_profile.[g] is defined but not used by that branch. | A | 消す | 通知を消す（設定キーは 3 の (a) の一覧に出る） | branch の config_profile はその branch だけが参照する定義で、未参照の項目は在庫ではない（D4 の 2）。一覧の WARN と重なる |
| nn_impl.cpp 1847 | net.config_profile.[g] is defined but not used by any branch. | A | 消す | 通知を消す（在庫はログに出さない） | 全体の config_profile はカタログで、未参照の項目は在庫（D4 の 2） |
| nn_impl.cpp 1923 | NetworkBodyBuilder: input key … not bound … | A | 残す・1 回 | 3 の (c) | 書き忘れの可能性（AGENTS.md 172 行の例） |
| nn_modules.cpp 1676 | ResBlock dropout_rate is enabled with BatchNorm. … | B | 残す・1 回 | 3 の (c) | block 数 × 構築数だけ出ている |
| observers.cpp 91 | HeatMapVectorObserver: size mismatch. … | D | 残す・1 回 | 3 の (f) | log_interval ごとに出ている |
| observers.cpp 705・1181 | … force-ended due to unexpected episode_start … | D | 残す・1 回（Observer ごと） | 3 の (f) | 外から届いた episode の乱れで描画を打ち切り、Run の結果は変わらない。録画中に episode の開始が来るたびに出ている |
| observers.cpp 1115 | MetricsLogObserverBase::OnUpdate(): value not found. … | D | 残す・1 回（tag ごと） | 3 の (f) | D12。更新のたびに出ている。休止が原因の欠けと、機能の無効が原因の欠けは、出す側が NaN を返すので出なくなる |
| observers.cpp 1374・1379・1384・1404・1633・1640 | Unknown event／step／target／attribute … | A | ERROR へ | 2 の (a) | D7 |
| probe.cpp 251 | BatchExperienceVectorProbe::GetVector() failed to get value. … | D | 残す・1 回 | 3 の (f) | 呼ぶたびに出ている |
| replay_buffer_impl.cpp 1213 | PER Rejection Sampling failed to fill batch. Falling back to uniform. | C | 残す・1 回 | 3 の (g) | D9。文面に、この Run では以後出さない旨を足す |
| rainbow_agent.cpp 54 | use_qr is true but num_quantiles <= 1. Treating as Scalar DQN. | C | ERROR へ | 2 の (a) | D7 |
| scaler_impl.cpp 420 | Unknown squash_type. | A | 消す（構築時の検証で止める） | 2 の (a) | D7 |
| scaler_impl.cpp 541 | Outlier Row Detected! … Full Obs: … | D | `ANET_LOG_DEBUG` へ | 4 | `// DEBUG` のブロックで生の観測を出す、開発用の診断 |
| trainer.cpp 866 | Skipping metrics for unscheduled eval tag. … | A | INFO へ | 1 の (a) | D8 |
| trainer.cpp 1150 | Eval session has fewer adopted episodes than episode groups. … | B | 残す | 3 の (c) | 効き目が弱まる組み合わせ |
| trainer.cpp 1212 | Skipping Env config dump because GetConfigData is unsupported. … | F | 残す | 3 の (i) | Env の実装の欠け（コードで直すこと） |
| AtariEnv.cpp 483 | AtariEnv.display_screen=true … unsupported with worker threads … | B | 残す・1 回 | 3 の (c) | `display_screen` は `num_envs = 1` と `env.worker_type = SINGLE_THREAD` の単発のデバッグ用で（[220](../design/220_atari_env.jp.md) §4.2 の 167 行）、Env は worker の種類を知らないので、支えない組み合わせかを判定できず利用者に確かめさせる。lane ごとに出ている。worker thread と併用したときだけ止めるかは §9 の 9 |
| ImageData.cpp 327・350・360 | ImageDataset auto cache … | C | 残す | 3 の (d) | auto の代替で、選択を固定している |

WARN を出していなかった `run.train.runner_type` の未知の値（trainer.cpp の `RunnerFactory::CreateMainRunner`）は、この表の行ではなく D7 で止める（D1 の 2 の (a)）。

### 5.6 題材の事例の挙動

| 入力 | 挙動 |
|---|---|
| 構造に現れる `net.block.[Conv2d_X]` に `type = Conv2d` と `out_channels = 32`（`conv.` なし） | Network の構築時に `ANET_SYSTEM_ERROR`。block の名前、`type`、`out_channels = 32`、Conv2d が読むキー（`conv.out_channels` など）を含める |
| 同じ block が構造に現れない | Run では何も出さない（D4 の 2）。出荷設定の 7 本の Env の設定を既定の `run.$` で解決した結果にあれば、出荷設定の検査のテストが落ちる（D13） |
| 共通の設定の block の `type` だけを、Run プロファイルで別の種類に変える | 元の種類の設定キーが読まれず、Network の構築時に `ANET_SYSTEM_ERROR`（D3 の (a) の帰結） |
| 共通の設定に `Conv2dPanel.conv2d.mergin_x ?= 2` | Run の構築の終わりに、読まれない設定キーの一覧の WARN に `Conv2dPanel.conv2d.mergin_x = 2` が出る |
| Atari の `run.@eval2ch_r1`（`eval_target` を `interval = 0`） | INFO の `Skipping metrics for unscheduled eval tag. tag='eval_target'. …`。WARN は出ない |
| RND ありの checkpoint を `learner.enabled = false`・`run.train.num_envs = 1` で読む評価だけの Run | `learn_batch_size` の検査をしない。補助学習の休止は INFO。RND の scalar は NaN で、`value not found` の WARN は出ない |
| `run.@rnd` を付けたまま、引数で `DefaultDQNAgent.intrinsic.enabled=false` にした Run | RND の scalar は NaN になり、`value not found` の WARN は出ない |
| learner が有効で、`run.train.num_envs = 10`、`learn_batch_size` は既定の 32、RND 有効 | Agent の構築時に `ANET_SYSTEM_ERROR`（キー、値 32、lane の数 10 を含める） |
| 学習率を変えて、checkpoint の続きから学習する | INFO の `Learner::Load: optimizer option differs from checkpoint; config applied. …`。WARN は出ない |
| skip 有効の設定で、skip の Network を持たない checkpoint を読む | WARN の `Checkpoint has no skip networks; …` が 1 回 |
| `run.train.runner_type = pipline` | 構築時に `ANET_SYSTEM_ERROR`（キー、値、`serial`・`pipeline` を含める） |
| `$include <missing.txt>`（先が無い） | 設定の読み込みで `ANET_SYSTEM_ERROR`（パスと、探した場所を含める） |

### 5.7 性能

読んだ設定キーの記録は設定を読むたびに 1 回起き、設定の読み取りは Run・Agent・Env・Observer の構築時に集まる。学習の step のループで `ConfigData` を読む箇所は、実装計画で検索して確かめる（見つかったら構築時へ移すか、記録の単価を測る）。1 回だけ出す共通の部品が記録を引くのは、通知を出そうとした箇所だけで、普通の step の経路には入らない。性能の受入は置かない。

### 5.8 移行対象

この一覧が移行対象の正本である。設計文書の行番号は 2026-11-10 時点。

| 対象 | 扱い |
|---|---|
| [AGENTS.md](../../AGENTS.md) 85・93 行 | D6 の文へ改める |
| AGENTS.md 160〜161 行、188〜200 行 | D1 の線引きへ改める。190 行の「意図しない状態」（契約違反の内部状態、壊れた入力データを含む）を D1 の 2 の条件へ改め、D1 の 3 の (e)〜(i) を WARN の行として足す。191 行（意図された休止）は INFO へ。192 行（自動選択の不履行）は D1 の 3 の (d) として残す。160〜161 行は D1 の 3 の (c) の言い方へ。200 行の「WARN は対象ごと 1 回」に、Run ごとに数えることを足す。208 行（互換期間を設けた非推奨設定の WARN）は D1 の 3 の (j) に当たり、変えない |
| [実装 品質確認観点](../quality/impl.md) 項番 24・25・27 | D1 の言い方へ揃える（24 は INFO、27 は Run ごとに数える）。「失敗の扱い」に D2 の項目を 1 つ足す |
| [CONTEXT.md](../../CONTEXT.md) | 「dormant」の「タグごと 1 回の WARN で skip」を INFO へ。「読まれない設定キー」「在庫」を足す |
| [100](../design/100_runtime_and_configuration.jp.md) §2.1・§5.1（169 行）・§6.1（208 行）・§7.3・§8、325 行の表 | 読んだ設定キーの記録、在庫、一覧の時機、dormant の INFO、include の失敗、テストの案内。`run.train.runner_type` の未知の値は止まる |
| [130](../design/130_neural_networks.jp.md) §7.1（287 行）と block の章 | block の閉じたスキーマ（使わない block は検査しないこと、`type` だけの上書きは止まること）、WARN を Run ごとに 1 回 |
| [140](../design/140_observability.jp.md) 234 行（不明な event・step 軸・target は WARN 後に既定値）・254 行（scalar の既存の WARN は変えない）・288 行（value not found の WARN）・238・316 行（NaN と値なしの契約） | metrics の定義の未知の値を止める。value not found を tag ごとに 1 回。休止中の機能と、設定で無効にした機能は、鍵を保って NaN を返す（D12。238・316 行の決まりは変えず、休止にも当たることを足す） |
| [150](../design/150_replay_buffer.jp.md) §7.5 の 314 行（en は 315 行） | 「uniform sampler では PER の鍵を値なしで表す」を、実装のとおり「既知の鍵は PER が無効でも NaN を返す」に直す（D12。コードは変えない） |
| [200](../design/200_dqn_agents.jp.md) §6.2（398 行「learner.enabled=false では一度WARNし」・402 行の k=min）・§6.5（506 行の `feature_gradient=stop` の WARN）・§8.2（619 行と 020 が言う options の差の WARN）・§9.1（633 行「無効時はキーが無くnullopt」）・§9.4（699 行「購読ゼロならNoCare」） | RND の休止と stop の説明と optimizer の差を INFO。RND が無効のときの鍵は NaN（D12）。k = `learn_batch_size`（D11）。購読の無い可塑性の設定は D5 に合わせて読むか休止を宣言する。602・606 行（初期化して一度 WARN）は変えない |
| [212](../design/212_rnd_lifelong_novelty.jp.md) §3.5（262 行）・379 行の表・442 行（「RND無効時のキーは `nullopt`」）・§4.5（454 行「差があればWARNする」）・§5.1（470 行「一度警告して…休止する」） | `learn_batch_size` > lane の数は構築時に止まる（D11）。RND が無効のときの鍵は NaN（D12）。RND の optimizer の差を INFO、休止を INFO |
| [020](../design/020_user_guide_run.jp.md) §3.4.9（260 行「見つからない場合はWARNを出して続行する」）・379 行の表・§3.6・§6.4（519 行「checkpointと違う値はログにWARNが出る」） | include の失敗は止まる。読まれない設定キーの一覧の見方。optimizer の差は INFO |
| [220](../design/220_atari_env.jp.md) §4.2（167 行、68・143 行の表）・[040](../design/040_development_environment.jp.md) §3.6（185 行） | WARN を Run ごとに 1 回にするだけなら変えない。§9 の 9 で併用を止める形になったら追随させる |
| 上の設計文書の `.en.md`（100・130・140・200・212・020。220・040 は §9 の 9 による） | 同じ章を追随させる。D12 の RND の鍵は、200 の en の 635 行（Disabled keys return nullopt）と、212 の en の 406 行 |
| ADR 0027・0047・0053 | 先頭へ注記する（本文は書き換えない）。判断は ADR 0071 に書く。本 PRD の改訂で済ませる |
| [trainer_test.cpp](../../core/anet-core/src/trainer_test.cpp) 1056 行 | dormant のレベルの断定を INFO へ |
| [intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) 1259・1315 行 | `exceeds num_envs` の断定を、learner が有効なら ERROR、無効なら検査なし、へ |
| intrinsic_test.cpp の補助 `IntrinsicAgentData`（112〜131 行）と `MakeIntrinsicAgent`（132 行。既定の lane は 2） | 補助は `learn_batch_size` を設定しないので、コードの既定の 32（default_dqn_agent.hpp 88 行）が lane の数を超え、learner が有効なテストが D11 で構築時に止まる。当たるのは、`MakeIntrinsicAgent` の呼び出し 15 か所のうち、`learn_batch_size` を明示する 326・1134・1170・1259 行のテストと、RND を無効にした Agent を作る 389 行のテストを除くもの。補助が `learn_batch_size` を lane の数以下に設定するように直す |
| 実体化したカタログ項目に読まれない設定キーを持つ、既存のテストの設定 | 検索では挙げられない（読まれない設定キーは、§5.2 の記録を作るまで分からない）。記録と D3(a) の検査を入れたあと、既定のテストスイートを流して、止まった設定を直す（A8） |
| intrinsic_test.cpp 389〜390 行 | RND を無効にした Agent の鍵が値なしであるという断定を、`NaN` を返す断定へ（D12） |
| config_test.cpp 1534 行 | `Properties: Failed to open include file` の WARN の断定を、ERROR で止まる断定へ |
| [observers_test.cpp](../../core/anet-core/src/observers_test.cpp) 1953 行 | `value not found` の断定を、tag ごとに 1 回へ |
| WARN のレベルを断定するテスト（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) 24 か所、[config_test.cpp](../../core/anet-core/src/config_test.cpp) 3、[nn_test.cpp](../../core/anet-core/src/nn_test.cpp) 2、trainer_test.cpp 4、[GridMazeEnv_test.cpp](../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp) 1 の `wxLOG_Warning`） | §5.5 の扱いに合わせて見直す（`optimizer option differs` は INFO へ、`no primary_skip optimizer role` は WARN のまま）。新しく止まる設定を入力に使うテストは、この表の intrinsic_test.cpp と config_test.cpp の行に挙げた。未知の `runner_type` と未知の属性を入力に使うテストは、2026-11-10 の検索では無かった（実装計画で引き直す）。`wxLOG_Warning` を持つ `LogPanel_test.cpp`（7 か所）と `log_test.cpp`（1 か所）は、Logs pane とログ基盤自身のテストで対象外 |
| 出荷設定（`apps/runner/config/*.txt`、番号付き bat） | A2 の一覧に出た設定キーを直す（綴り違い・旧キー・持ち主のない設定キー）。綴り違いの `mergin_x`（66100f。common.txt 75・76 行、metrics_image.txt 96・97 行）を含む。`GridMaze_muzero.txt` に `app.save_agent_on_close = false` を足す（保存に対応していない Agent に、終了時の保存を求めているため）。新しく止まる書き方（metrics の定義の未知の属性など）が見つかれば直す |
| [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp) | 出荷設定の検査のテスト（D13、A9）を足す |
| [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) | 変えない（演算子の検査で、本 PRD の検出とは別） |
| `core/anet-core/testdata/prd117/README.md`（新規） | A2 の Run の組み合わせ、止める条件、WARN を数える手順 |
| 設定の用語の「葉」を使う、現用の文書・コード・設定（D14。2026-11-11 の検索で 52 ファイル 151 行） | D14 の表のとおり言い換える。多いのは、[020](../design/020_user_guide_run.jp.md)（jp 35 行、en 5 行）、[100](../design/100_runtime_and_configuration.jp.md)（8 行）、wiki の [316](../wiki/categories/316_config.md)（6 行。ほかに課題の区画の 1 行があり、これは除く）、[CONTEXT.md](../../CONTEXT.md)（6 行）、[実装 品質確認観点](../quality/impl.md)（5 行）、[212](../design/212_rnd_lifelong_novelty.jp.md)（5 行）、config_impl.cpp（4 行）、`core/anet-core/testdata/` の README とスクリプト、`apps/runner/config/` のコメント。設定のキーと値は変えない |
| [AGENTS.md](../../AGENTS.md) の応答言語ルールの「避ける語」の表 | 「既定葉」「個別葉」「葉キー」の行と、単独の「葉」の行を足す（D14） |
| wiki の [316](../wiki/categories/316_config.md)・[315](../wiki/categories/315_nn.md)・[317](../wiki/categories/317_metrics.md) | 実装の完了後に点検で拾う |

検索語（実装計画で再検索する）: `log::warn()`・`LOG::warn()`・`.warn()`・`log.warn`・`wxLOG_Warning`・`NoCare`・`WARN`・`警告`・`Failed to open include`・`Unknown event`・`definition-only`・`learn_batch_size`・`runner_type`・`value not found`。

2026-11-10 の改訂で、追跡しているファイルを次の 3 つの検索で引き直した。

- 規約・用語集・設計文書（jp）・品質確認観点・プロセス・skill（`AGENTS.md`、`CONTEXT.md`、`docs/design/*.jp.md`、`docs/quality`、`docs/process`、`.agents/skills`、`.claude/skills`）を、大文字小文字を区別せずに `NoCare|1 ?回だけ ?WARN|一度だけ(の)? ?WARN|WARN 1 回|1 回の WARN|1回WARN|WARNして|WARN後|エラーにならず続行|一度WARN|一度 WARN|一度警告|1 回警告` で検索: 11 ファイル 28 件。
- `learn_batch_size` を `docs/design/*.md`、`docs/quality`、`AGENTS.md`、`CONTEXT.md`、`.agents/skills` で検索: 4 ファイル 15 件（200 の jp・en、212 の jp・en）。
- `runner_type` を `docs/design/*.md`、`docs/quality`、`AGENTS.md`、`CONTEXT.md` で検索: 2 ファイル 2 件（100 の jp・en の 325 行）。
- `葉` を、D14 の現用の範囲（`AGENTS.md`、`CONTEXT.md`、`docs/quality`、`docs/process`、`.claude/skills`、`.agents/skills`、`docs/design`、`docs/wiki/apps`、`docs/wiki/categories`、`core`、`apps`、`viewers`、`reports/tools`）で、`git grep`（Git が追跡しているファイル）で検索: 52 ファイル 151 行（うち「言葉」2 行）。Git が無視するファイル（`core/anet-core/testdata/prd103/baseline-config/` などの、Run の設定の写し）は入らない。`既定葉` 60 か所、`個別葉` 20 か所、`葉キー` 1 か所。

1 つ目の結果のうち、上の表に行番号で挙げていない行は次のとおり（行単位）。

- skill の本文で検索語そのものを例に引く行（`.agents/skills/anet-create-prd/SKILL.md` と `.claude/skills/anet-create-prd/SKILL.md` の 203 行）: 対象外。
- [210](../design/210_metrics_viewer.jp.md) の 129・474・975 行: Metrics Viewer 自身のログで、Run のログではないので対象外。
- §5.5 で「残す」とした WARN を説明する行: 200 の 508・574・602・606・721・751 行、212 の 196・215 行。
- AGENTS.md の 192 行: 表の行として上の表に含む。

### 5.9 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ConfigUsageTracker` | 新規 | Run の実効設定にある設定キーごとに、読まれたか・在庫かを記録し、読まれない設定キーの一覧を返す。一覧を WARN で出す口を持つ | 設定管理・config |
| `OnceLogFilter` | 新規 | 通知を出す箇所と対象の名前の組ごとに、その Run でまだ出していないかを答える。記録を空にする口を持つ | ユーティリティ・log |
| `ConfigData` | 変更 | 読み取りを記録へ届ける。複製と、カタログの項目へ分けた設定からの読み取りも、元のキーで届ける | 設定管理・config |
| `Config` | 変更 | 持ち主の休止の宣言を記録へ渡す | 設定管理・config |
| `ConfigResolver` | 変更 | 入力の選択の宣言に、選択の源として現れた通常 prefix を記録へ渡す | 設定管理・config_impl |
| `Properties` | 変更 | include の不在・解析の失敗・開けないファイルで止める | 設定管理・config |
| `NetworkStructBuilder` | 変更 | 実体化した block・config_profile の読まれない設定キーで止め、未参照の block と全体の config_profile を在庫として記録する（branch の config_profile の未参照の項目は記録しない） | NN・nn_impl |
| `ObserverFactory` | 変更 | metrics の定義の未知の値・属性で止め、scalar・trace の定義の項目の読まれない設定キーで止める | 可視化・メトリクス・observers |
| `ImageProviderManager` | 変更 | 画像の metrics の定義の項目の読まれない設定キーと、同じ tag の二重登録で止める | 可視化・メトリクス・image |
| `RunManager` | 変更 | dormant の skip を INFO にする | RL 共通基盤・trainer |
| `RunnerFactory` | 変更 | `run.train.runner_type` の未知の値で止める | RL 共通基盤・trainer |
| `DefaultDQNAgent` | 変更 | RND の休止を INFO にし、補助学習が動くときの `learn_batch_size` > lane の数で止める。休止中と無効のときも、RND の値の鍵を保って NaN を返す。`post_process_type` を構築時に検証する | DQN・default_dqn_agent |
| `RainbowAgent` | 変更 | 互換性の無い `use_qr` と `num_quantiles` で止める | DQN・rainbow_agent |
| `VideoLogger` | 変更 | auto の代替の選択を固定する | 可視化・メトリクス・metrics_logger |
| `RunnerApp` | 変更 | Run を作り始めるときに 1 回だけ出す記録を空にし、アプリの側の構築の終わりに読まれない設定キーの一覧を出す。どの選択の宣言にも現れない mode の prefix を読んで検証するか、休止を宣言する。旧キーの WARN を消す | RunnerApp |
| `RunnerFrame`・`Conv2dPanel` | 変更 | panel の設定を Run の構築時に読む | RunnerApp |

§5.5 の表で「1 回」「INFO へ」とした行の持ち主（`DQNBasedAgent`、`NetworkBodyBuilder`、各 Observer、`ReplayBuffer` の実装など）も変更に当たる。

### 5.10 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261107-c17068](../management/issues/issue-20261107-c17068.md) | WARN の意味の線引きと、読まれない設定の検出（NoCare の見直し） | `316_config`, `312_util`, `315_nn`, `317_metrics`, `318_rl_core`, `324_dqn`, `110_runner_app` | [AGENTS.md](../../AGENTS.md) 85・93・188〜200 行、[config.hpp](../../core/anet-core/include/anet/config.hpp) 95〜105 行 | D1・D2・D3・D4・D5・D6・D7・D9・D10・D11・D12・D13・D14 | A1・A2・A4・A7・A8・A9・A10・A11 |
| [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md) | net.blockの未知の葉（conv.なしのout_channelsなど）が構築時に検出されず黙って無視される | `316_config`, `315_nn` | [nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) 787〜845・1776〜1806 行、[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) 2969・3022〜3075 行 | D3・D4・D5・D6・D13 | A3・A4・A9 |
| [issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md) | Atariの正常なRunで毎回出るWARN（eval_targetの未スケジュール、評価だけのRunのRND）で本物のWARNが埋もれる | `318_rl_core`, `317_metrics`, `324_dqn` | [trainer.cpp](../../core/anet-core/src/trainer.cpp) 860〜870 行、[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 109〜119 行 | D1・D2・D8・D11・D12 | A2・A5・A6 |
| [issue-20261103-2c9a07](../management/issues/issue-20261103-2c9a07.md) | 旧キーapp.metrics_logger.runs_dir/root_dir専用のWARNが残り、クリーンブレーク方針に反する | `110_runner_app` | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) 131〜145・414 行 | D6 | A1・A4 |
| [issue-20261103-66100f](../management/issues/issue-20261103-66100f.md) | Conv2dPanel.conv2d.mergin_x/yとimage.@conv2d.conv2d.mergin_x/yが綴りの誤りで読まれない | `316_config`, `110_runner_app`, `317_metrics` | [common.txt](../../apps/runner/config/common.txt) 75・76 行、[metrics_image.txt](../../apps/runner/config/metrics_image.txt) 96・97 行、[Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp) 22・23 行 | D2・D4・D5 | A2・A4 |
| [issue-20261103-582880](../management/issues/issue-20261103-582880.md) | train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う | `318_rl_core` | [trainer.cpp](../../core/anet-core/src/trainer.cpp) 805・816・1066 行 | D7 | A7 |

## 6. 細部

決めた既定と理由を残す。契約の正本は、1〜4 が §5.4、5 が §5.2、6〜8・10 が §5.1、9 が §5.3、11〜16 が §5.5、17 が §7 の A2 である。

1. 読まれない設定キーの一覧の文面: 1 行目は `Unread config keys: N. These keys were not read by any component.`、続く行に `key = value` をキーの順に 1 行ずつ。0 件なら出さない（理由: ログは英語の規約。件数で重さが分かる）
2. 一覧の行数の上限: 置かない（理由: 普通の Run は 0 件が前提で、多いときほど全部を見たい）
3. 一覧を出す時機: Run の構築（アプリの側の構築を含む）を終えた直後、最初の step の前に 1 回。online と batchrun で同じ（理由: 構築時に読む決まり（D5）なので、この時点で全部の持ち主が読み終えている）
4. 一覧を Run の `json/` にも保存するか: 保存しない（理由: Run のログに残り、`config_data.txt` から再現できる）
5. キーの有無を見るだけの問い合わせ（`Has`）: 「読んだ」に数えない（理由: 有無を見ただけでは、値が使われたと言えない）
6. 「対象ごとに 1 回」の数え方: Run ごと。同じプロセスで Run を作り直したら、また 1 回出す。NN の複製で同じ対象を繰り返し作っても 1 回（理由: 印は Run ごとに確かめるもの）
7. 1 回だけ出す共通の部品: ログ基盤の近くに 1 つ置く。同じ通知かどうかは、通知を出すソースの箇所と、任意で渡す対象の名前の組で決める（理由: 箇所ごとの書き方をそろえ、数え方を 1 か所で守る。箇所だけでは「tag ごとに 1 回」が出せない。関数の中に持つ 1 回きりのフラグは、Run を作り直したときに二度と出ない）
8. ERROR・WARN・INFO の文面の言語: 英語（理由: ログの規約）
9. カタログの項目で止めるときの文面に入れるもの: block は、項目の名前、`type`、読まれない設定キーとその値、その種類が読むキーの一覧。ほかのカタログは、項目の名前と読まれない設定キー（理由: 文面だけで直し方が分かる）
10. 休止の宣言の INFO に入れるもの: 持ち主の prefix と、休止の理由になった設定のキーと値（理由: どの設定で止まったかが、ログだけで分かる）
11. PER の代替の WARN: 今の文に、この Run では以後出さない旨を足す（理由: 2 回目以降が黙って続くことを知らせる）
12. `feature_gradient = stop` の説明: INFO（理由: 選んだモードの定義を述べる通知で、設定どおりの結果）
13. `CUDA_LAUNCH_BLOCKING overridden by backend.cuda_launch_blocking`: INFO（理由: 設定が環境変数に勝つのは契約どおり）
14. `Outlier Row Detected! … Full Obs: …`: `ANET_LOG_DEBUG` へ（理由: `// DEBUG` のブロックで生の観測を出す診断）
15. 動画の codec の auto の代替: 最初に代えたら選択を固定し、WARN は 1 回（理由: AGENTS.md の表の「以後は選択を固定する」）
16. `AtariEnv.display_screen = true` の WARN: §9 の 9 の結果が出るまでは Run ごとに 1 回（理由: 今は lane ごとに出ている）
17. A2 の評価だけの Run の長さ: 最初の評価の session が終わるまで（理由: 学習が無いので learn 側の出力が来ない。値の欠けは評価側の出力で確かめる）

## 7. 検証と受入

以下は実装工程で満たす条件で、改訂の時点では実行していない。

### A1. 既存の WARN の扱い

§5.5 の全行が表の扱いになっている。レベルを変えた行と ERROR へ移した行は、行ごとにテストでレベル（`wxLOG_Message`・`wxLOG_Warning`・`ANET_SYSTEM_ERROR`）を確かめる。`ANET_LOG_DEBUG` へ移した行は、その状態を起こしても WARN が出ないことを確かめる（DEBUG の出力の有無はビルドの `ANET_ENABLE_DEBUG_LOG` で変わるので確かめない）。「1 回」の行は、同じ Run の中で同じ対象を 2 回以上起こしても 1 件になることを確かめる。「残す」だけの行のうち、実装で 1 回にした行も同じに確かめる。消した行は、次の文字列を `core/` と `apps/` の `.cpp`・`.hpp`（`third_party/` とテストを除く）で検索して、残りが 0 件。

- `is deprecated and ignored`
- `WarnDeprecatedRunsDirConfig`
- `is defined but not used by`
- `Unknown squash_type`
- `Treating as Scalar DQN`

1 回だけ出す共通の部品は、テストで次を確かめる。同じ箇所で対象の名前が違えば、名前ごとに 1 件出る。同じ箇所と同じ対象なら、何回呼んでも 1 件。記録を空にして Run を作り直すと、また 1 件出る。

### A2. 普通の Run の WARN 0 件

次の Run を Debug か Release のビルドで流し、各 Run のログ（`run_*.log`）の `: [W] ` の行を数えて、全 Run で 0 件。数える単位は Run。

- 出荷設定の Env の設定ファイル（`apps/runner/config/` の `Atari.txt`・`CartPole.txt`・`DropMerge.txt`・`GridMaze.txt`・`GridMaze_muzero.txt`・`ImageCls.txt`・`LunarLander.txt`）の既定の `run.$`。
- 番号付きの bat（`apps/11_batch_run.bat`・`12_batch_run.bat`・`13_batch_run.bat`）に、実装の時点で書かれている Run のうち、`run.$` の組み合わせが違うもの（`goto` で飛ばされている段落を含む。2026-11-10 時点で約 17 通り）。
- Atari の `run.@evalonly`。

各 Run の長さ:

- 学習する Run は、Run のフォルダの `metrics.jsonl` に、learn 側（`@learn`）で宣言した scalar のレコードが 1 件以上書き出されるまで流す。learn 側の scalar を宣言していない Run は、learn step が 1 以上になるまで流す。評価の完了は待たない。
- 評価だけの Run は、最初の評価の session が終わるまで流す（[140](../design/140_observability.jp.md) §7.2 の評価セッションの実行ログで、最初の session の終わりの行が出るまで）。
- どの Run も、止める条件に達したあと、正規の終了（`app.batchrun.exp_exit_step` を引数で与える、など）で閉じ、終了時の通知までをログに含めて数える。Run ごとの step 数は README に書く。
- Atari の `run.@evalonly` には、同じ A2 の Atari の既定の Run が終了時に保存した checkpoint を `DefaultDQNAgent.auto_load_file` で与える。与えないと、読み込む checkpoint が無いことの WARN（§5.5 の default_dqn_agent.hpp 484 行）が出る。
- アプリの側で読む設定キー（`metrics.image.*`、`app.eval_trace_panel.*` など）が一覧に出ないことも、この Run で確かめる。
- 評価の中で出る WARN は、A5・A6 のテストで確かめる。

データセットが手元に無い Env（ImageCls など）で流せなければ、その Run を未達として理由を記録する。A2 の Run の組み合わせ、止める条件、数え方は `core/anet-core/testdata/prd117/README.md` に置く。全部で約 25 本、1〜2 時間の見込みである（測っていない）。

### A3. block の閉じたスキーマ

§5.6 の 1〜3 行目をテストで確かめる。`?=` と `=` の両方で書いた入力で同じ結果になる。§5.3 の表のほかのカタログも、表に並ぶカタログの名前ごと（`net.config_profile`、`net.branch.[<b>].config_profile`、`<Agent>.actor`、`<Agent>.nn_forward`、`run.eval`、`run.eval_schedule`、`metrics.scalar`、`metrics.trace`、`metrics.image` の 9 つ）に 1 つずつ、実体化した項目の読まれない設定キーで止まることを確かめる。`run.eval` は、dormant の tag の読まれない設定キーでも止まることを確かめる。表の「Run で実体化しない項目」を持つ 3 つ（`net.config_profile`、`net.branch.[<b>].config_profile`、`<Agent>.nn_forward`）では、その項目（どの branch も使わない項目、その branch が使わない項目、どこからも参照されない項目）の読まれない設定キーで止まらないことを確かめる。今のモードで使わない設定キーを持つ項目が、その設定キーでは止まらないことも、モードを持つカタログ 1 つで確かめる。

### A4. 読まれない設定キーの一覧

テストの設定で、次を確かめる。

- 綴り違いの設定キー（`?=` と `=` の両方）と、CLI で与えた未知の設定キーが、一覧の WARN に 1 件ずつ出る。
- D4 の 1〜4 の例（選択の源の上書き層、未参照の block、作らなかったクラスの設定、休止を宣言した prefix）は出ない。あとの宣言に置き換えられた選択の源の prefix の設定キーも出ない。どの選択の宣言にも現れない prefix に書いた設定キーは出る。
- 全体の block の設定キーを Agent の下の同じ block の設定キーで上書きした設定で、上書きされた側の設定キーが一覧にも出ず、構築も止まらない。
- branch の config_profile のうち、その branch が使わない項目の設定キーは出る（D4 の 2 の除外）。
- 一覧に出る設定キーが 0 件なら WARN を出さない。
- 一覧は Run ごとに 1 件。
- 一覧の WARN が出ても Run の構築は成功する。

### A5. dormant の tag

trainer_test.cpp の dormant の tag のテストで、skip の通知が INFO で tag ごとに 1 件、WARN が 0 件。`interval = 0` で止めた tag と、schedule の記述が無い tag の両方で同じ結果になる。

### A6. RND の休止と件数

- RND ありの checkpoint を `learner.enabled = false`・`run.train.num_envs = 1` で読む構成で、Agent の構築が成功し、`learn_batch_size` の通知が出ず、休止の INFO が 1 件。RND の scalar の鍵は `NaN` を返し、`value not found` の WARN が 0 件。
- `DefaultDQNAgent.intrinsic.enabled = false` の Agent で、RND の scalar の鍵が `NaN` を返す（値なしではない）。どの機能も持たない鍵は、値なしのまま。
- learner が有効で `run.train.num_envs` < `learn_batch_size` の構成では、Agent の構築時に `ANET_SYSTEM_ERROR` になり、文言にキー、値、lane の数が入る。
- learner が有効で `run.train.num_envs` = `learn_batch_size` の構成では止まらず、1 step の件数が `learn_batch_size` になる。

### A7. 止める失敗

D7 の各入力（重ねる設定のファイルが開けない、include 先が無い、include 行の解析の失敗、metrics の定義の未知の event・step・target・属性、`post_process_type` の範囲外、RainbowAgent の `use_qr` と `num_quantiles`、同じ tag の ImageProvider の二重登録、`run.train.runner_type` の未知の値）で、構築時に `ANET_SYSTEM_ERROR` になり、文言にキーかパス、指定値、期待値が入る。`run.train.runner_type` は、`serial`・`pipeline` と、その大文字小文字違いでは止まらない。

### A8. ビルドと既存のテスト

[AGENTS.md](../../AGENTS.md) の手順で Debug ビルドを通し、既定のテストスイート、`"[shipped_config]"`、`AnetRLRunner-gui-test "[log_panel]"` を流す。既知の失敗（PRD093 の attention のテスト 1 件）以外の失敗が無い。結果と未実施の理由を実装計画に書く。

### A9. 出荷設定の検査のテスト

タグ `[shipped_config]` のテストが、A2 と同じ 7 本の Env の設定ファイルのそれぞれを既定の `run.$` で解決し、Run で実体化しない項目を持つカタログ（§5.3 の表。`net.block`、`net.config_profile`、`net.branch.[<b>].config_profile`、`<Agent>.nn_forward`）に定義された項目を、その Run で使われるかどうかによらず全部、読み口に読ませて、読まれない設定キーが 0 件であることを確かめる。項目の本数・名前・値は等値比較しない（[実装 品質確認観点](../quality/impl.md) の項番 73）。読み口を単体で呼べないカタログは、その項目を参照する設定を組んで構築を通す形で確かめる（どちらの形かを実装計画に書く）。block は、定義を 1 つずつ factory で作る形で確かめられる（module は設定だけで作れる。§2.6）。補間の印（config_profile）が入った設定キーには、仮の数値を入れて作る。検査の仕組みそのものは、block の直下に `out_channels` を書いた fixture で落ちることを確かめる。

### A10. checkpoint の読み込みの通知

- 学習率が checkpoint と違う設定で読み込むと、`optimizer option differs` の通知が INFO で出て、WARN が 0 件。RND の optimizer も同じ。
- skip 有効の設定で、skip の Network を持たない checkpoint を読み込むと、`Checkpoint has no skip networks` の WARN が 1 件。`primary_skip` の optimizer の状態を持たない checkpoint を split で読み込むと、`no primary_skip optimizer role` の WARN が 1 件。

### A11. 用語の言い換え

D14 の現用の範囲を `git grep`（Git が追跡しているファイル）で引いて、次を確かめる。範囲は `AGENTS.md`、`CONTEXT.md`、`docs/quality`、`docs/process`、`.claude/skills`、`.agents/skills`、`docs/design`、`docs/wiki/apps`、`docs/wiki/categories`、`core`、`apps`、`viewers`、`reports/tools`。

- `既定葉`、`個別葉`、`葉キー`、`葉値`、`確定葉` のそれぞれを固定文字列で検索して、残りが 0 件。
- 除くもの: AGENTS.md の「避ける語」の表の行と、CONTEXT.md の `_Avoid_` の行（避ける語として挙げるため）。wiki の地図の「課題」の区画（課題の題をそのまま写すため）。
- 単独の「葉」は、同じ範囲を `葉` で検索し、残りが上の除くものと「言葉」だけ。
- 設定のキーと値が変わっていないことは、[実装 品質確認観点](../quality/impl.md) の設定変更時の確かめ（`check_default_leaves.py` と、全キー・文字列値の比較）で確かめる。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 読まれない設定キーの一覧を ERROR にする | 先送り（実験の結果が要る） | 在庫の区別を実際の Run で確かめてからにする。再開条件: 本 PRD の実装をコミットした日から 1 か月のあいだ、A2 の Run と番号付き bat の Run のログ（`run_*.log`）で、一覧の WARN に出た設定キーがすべて設定の誤りだった（在庫の見分けの漏れが 1 件も無かった）とき |
| 眠る bare の検出（読まれて効かない値） | 範囲外（別の関心） | 「読まれたか」と「効いたか」は別の仕組みになる。PRD101 §7 の再開条件（眠った値に気づかずに Run を回した事故）は揃っていない。今回の事故は読まれない設定キーだった（PRD116 §2.1）。揃ったら別 PRD で起こす |
| 値の参照先の検証（`stack_keys` など） | 範囲外（別の関心） | §3.2。[issue-20261103-b754ff](../management/issues/issue-20261103-b754ff.md) で扱う |
| WARN と ERROR の間のレベル | 不採用 | §3.2 |
| `interval = 0` で止めた tag と、schedule の無い tag を分ける | 不採用 | D8。ADR 0027 が 1 つにまとめた dormant をまた 2 つに分け、休止の出どころを運ぶ分岐が増える |
| 使わない定義を Run のたびに検査する（一覧に出す、または止める） | 不採用 | D13。作って捨てれば検査できるが、Run の構築の経路に、乱数を消費しない扱い・補間の印の仮の値・未知の `type` の扱いが増える。誤りを含む書きかけの定義が印を付け続けるか、関係のない Run を止める |
| コミットの準備で出荷設定の検査のテストを流す | 不採用 | D13。害の経路が無く、テストの実行体が古いときの扱いという別の不確かさを持ち込む |
| `type` が上書きされた block だけ検査を緩める | 不採用 | D3。上書きされたかを運ぶ配線と例外の規則が増え、読まれない設定キーが黙って通る経路が残る |
| 値が取れないことを INFO にする | 不採用 | D12。鍵の綴り違いのような本物の誤りを INFO に埋める |
| 整形した文面をキーにして 1 回だけ出す | 不採用 | §5.1。step や件数のように変わる値が文面に入ると毎回出る。抑えるときも毎回文面を組み立てる |
| PER の抽選が埋まらないとき Run を止める | 不採用 | D9。起きる条件が分からず、実行時の検査の ERROR を足すことになる。代替の是非は §9 の 20 |
| `learn_batch_size` の既定を lane の数に結び付ける（`?= ${run.train.num_envs}`） | 不採用 | D11。既定が「全部の lane」に変わり、lane が 64〜256 の出荷設定で RND の 1 step の件数が 2〜8 倍になる。32 件という既定の意味も変わる |
| 「既定葉」を「既定指定」（「個別指定」と対にする）か「既定値の指定」に言い換える | 不採用 | D14。「設定キー」に寄せた短い語の「既定キー」にした |
| 既存の用語の言い換えを、本 PRD と別の課題にする | 不採用 | D14。本 PRD の実装が同じ文書を直すので、まとめて済ませる |
| Run を閉じるときにも、読まれない設定キーの一覧をもう 1 回出す | 不採用 | D3。「対象ごとに 1 回」の例外になる。無人の Run で気づけない弱点はどの WARN にも当たるので、運用の側の手当てとして §9 の 21 に送る |

## 9. 未決

Run を流して確かめる事項と、範囲外として課題に起票した事項を残す。

| # | 事項 | 確かめ方と、結果ごとの扱い |
|---|---|---|
| 3 | 一覧の時機に間に合わない、遅れて作る読み口がいくつあるか | §5.2 の記録と §5.4 の一覧を先に作り、A2 の Run で一覧に出た設定キーを持ち主ごとに数える。読む時機が遅いだけの持ち主は、構築時に読むようにする（D5） |
| 5 | 在庫の 4 種類（D4）で足りるか | 3 と同じ Run で、一覧に出た設定キーを種類ごとに数える。学習の部品を作らない構成（Atari の `run.@evalonly`）から先に数える（§10）。設定の誤りなら設定を直す。誤りでも読む時機の遅れでもない種類が残ったら、在庫の種類を足して本 PRD を改訂する |
| 9 | `AtariEnv.display_screen=true` を worker thread と併用したとき | 実装の最初に、worker の種類を知る側（BatchEnv の構築）で併用を判定できるかを確かめる。判定できれば、併用は非対応の明示要求として止め（D1 の 2 の (a)）、単発のデバッグ用の構成では何も出さない。そのときは [220](../design/220_atari_env.jp.md) §4.2 と [040](../design/040_development_environment.jp.md) §3.6 を追随させる。判定できなければ、WARN を Run ごとに 1 回のままにする（§5.5） |
| 20 | PER の rejection sampling が埋まらないときの一様への代替を、やめるか | 範囲外。本 PRD は WARN を 1 回にするだけ（D9）。明示した PER の一部を別の方法に代える点が、明示要求を黙って代替しない規約と食い違う。[issue-20261111-e90762](../management/issues/issue-20261111-e90762.md) に起票した |
| 21 | 無人で走る Run（番号付き bat）で、WARN に気づけない | 範囲外。本 PRD は一覧を構築の終わりに 1 回出すだけ（D3）。bat の終わりに各 Run の WARN の件数をまとめて出す、などの運用の側の手当ては、[issue-20261111-9e6e27](../management/issues/issue-20261111-9e6e27.md) に起票した |

### 9.1 再グリルで振り分けた行（2026-11-10）

起票時の未決表の行と、行き先の対応。番号は起票時のもの。

| # | 起票時の事項 | 行き先 |
|---|---|---|
| 1 | 既定キー `?=` を一覧の対象に含めるか | D4（含める） |
| 2 | 意図した休止を INFO にするか。2 つの止め方を分けるか | D1 の 1 の (a)・D8（INFO、分けない） |
| 3 | 一覧の時機と、遅れて作る読み口 | 時機は D5・§5.4。数は §9 の 3 に残す |
| 4 | 一覧を WARN にするか ERROR にするか | D3(b)（WARN）。ERROR は §8 |
| 5 | 在庫の 4 種類で足りるか | §9 の 5 に残す |
| 6 | 閉じたスキーマの対象のカタログ | §5.3（CONTEXT.md「カタログ」に当たるもの全部） |
| 7 | 補助学習が動くときの `learn_batch_size` > num_envs | D11（止める） |
| 8 | PER の rejection sampling の一様への代替 | D9（残して 1 回）。是非は §9 の 20 |
| 9 | `display_screen` と worker thread の併用 | §9 の 9 に残す |
| 10 | checkpoint の optimizer の options の差 | D10（置き換えは INFO、初期化は WARN） |
| 11 | 眠る bare の検出 | §8（範囲外のまま） |
| 12 | ADR を起こすか | ADR 0071 を起こした |
| 13 | 用語を CONTEXT.md に足すか | §3.4（足した） |
| 14 | 取り込む issue | §5.10（2c9a07・66100f・582880 を取り込む。b754ff は §3.2） |
| 15 | A2 の Run の組み合わせと step 数 | §7 の A2 |
| 16 | ERROR を出して続けている、設定以外の箇所 | §3.2（範囲外。[issue-20261107-16ffd8](../management/issues/issue-20261107-16ffd8.md)） |
| 17 | `feature_gradient=stop` の説明を INFO にするか | §6 の 12（INFO） |
| 18 | あとの実装が WARN を足すときの確かめ | D2（実装品質確認観点に 1 項目） |
| 19 | 意図した休止が原因のメトリクスの値の欠け | D12（鍵を保って NaN） |

## 10. 依存と順序

- PRD116 の `nn.txt` の block の直しは、コミット済みである（§2.6）。
- D3(b) を入れると、A2 の Run で出た設定キーを直すまで D2 の受入が通らない。設定の直しには 66100f の `mergin_x` が入る。
- 記録（§5.2）と一覧（§5.4）を先に作り、A2 の Run で一覧を測ってから、在庫の種類（§9 の 5）と遅れて読む読み口（§9 の 3）を片付ける順にすると、手戻りが少ない。
- 在庫の確かめ（§9 の 5）は、学習の部品を作らない構成（Atari の `run.@evalonly`）から先に行う。読まれない設定キーが一番多く出る見込みで、休止の宣言の粒度（D4 の 4）が早く決まる。
- 1 回だけ出す共通の部品（§5.1）を先に作ると、§5.5 の「1 回」の行をまとめて移せる。
- 用語の言い換え（D14）は、ほかの決定に依らない。ほかの直しと同じ文書に当たるので、文書の直しの最後にまとめて行うと、差分が読みやすい。

## 11. 参照

- 課題: §5.10 の 6 件
- 規約: [AGENTS.md](../../AGENTS.md) のクリーンブレーク方針・汎用機構と利用側の責任境界・Fail-Fast 原則・設定値の扱い、[実装 品質確認観点](../quality/impl.md) の「失敗の扱い」「ログ」「設定ファイル」「テスト」
- 設計文書: [100](../design/100_runtime_and_configuration.jp.md)、[130](../design/130_neural_networks.jp.md)、[140](../design/140_observability.jp.md)、[160](../design/160_applications_and_tools.jp.md) §3.5、[200](../design/200_dqn_agents.jp.md)、[212](../design/212_rnd_lifelong_novelty.jp.md)、[020](../design/020_user_guide_run.jp.md)
- ADR: [0027](../adr/0027-eval-definition-schedule-separation.md)、[0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)、[0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)、[0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)
- 地図: [316 設定管理](../wiki/categories/316_config.md)、[315 NN](../wiki/categories/315_nn.md)、[317 可視化・メトリクス](../wiki/categories/317_metrics.md)、[318 RL 共通基盤](../wiki/categories/318_rl_core.md)、[324 DQN](../wiki/categories/324_dqn.md)
