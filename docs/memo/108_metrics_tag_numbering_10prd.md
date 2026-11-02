# メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号）PRD

> 2026-10-31 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。番号はユーザーの指示で連番の 108 を振った。行番号は起票時点のもので、実装時は近傍の名前で探し直す。
> 起点: 2026-10-31、beam_rider と freeway の RAM メトリクスに番号 4 を足すとき、Atari の env・評価の群の中の 50〜58 が RAM 1〜3 で埋まっていて、4 の tag を暫定の 59 に置いた（[PRD082](082_atari_ram_progress_metrics_10prd.md) D20）。振り方の案を比べるうちに、ユーザーが「全体を見た時、現状は理想とかけ離れてるね。Agent関連メトリクスがいろんな番号に散らばってる」「きれいに使えそうなのは6x以降」「ramだけに番号をあまりさきたくない。かと言って可読性を大きく落としたくもない」と述べ、番号の再編を全体整理として本 PRD で行うと決めた。既存 Run との tag の互換が切れることは許容し、メイン機の Workspace（Atari-xx と Atari5-xx の両方）を切り替える意向も同時に示した。
> 関連: [PRD082](082_atari_ram_progress_metrics_10prd.md)（RAM メトリクス。D19 の tag の作り方、D20 の暫定の 59）、[実装 品質確認観点](../quality/impl.md)（「metrics の tag の文字数」「並び順を番号で決める設定」の規約。項番 67）、[140 可観測性 §6](../design/140_observability.jp.md)（Metric 設定 contract）、[220 §4.9](../design/220_atari_env.jp.md)（RAM メトリクス）。

## 1. 目的と問題

メトリクスの tag は `<群>/<指標>` の形をとる。本書では次のように呼ぶ。

- **群**: tag の `/` の前の文字列（例 `53_evalg`）。
- **群の番号**: 群の先頭の 2 桁（例 `53`）。
- **群の名前**: 群の番号と `_` を除いた残り（例 `evalg`）。
- **群の中の番号**: `/` の後ろの先頭の 2 桁（例 `10_game_score_mean` の `10`）。

Metrics Viewer は tag を文字列の順で並べる（[MetricsRepository.java:357](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java) の `ORDER BY t.key`）ので、番号が並びと読み方を決める。

今の番号には次の問題がある（§2）。

1. **agent の群が散らばっている。** agent の群は 32〜50 に置かれ、その間に env の群（42）が挟まり、画像の群は 43〜45・53・55・60 に飛んでいる。
2. **1 つの群の番号に複数の群がある。** 10・36・37・38・39・53 に、群の名前の違う群が 2 つ以上ある。atari5.py が書き出す `60_atari5` も、`60_agent_img` と同じ番号である。
3. **群の中の番号に決まりが無い。** 番号の決まりがあるのは設定の並び順（impl.md 項番 67）だけで、tag の番号の決まりは文字数（40 文字以内）しか無い。Atari の env・評価の群では RAM 1〜3 が 50〜58 を使い切り、RAM 4 を足すときに置き場が無かった。

## 2. 現状（2026-10-31 に確認したこと）

### 2.1 群の番号

`apps/runner/config/*.txt` の `[<群>/<NN>_…]` から数えた（括弧は定義の数）。

| 群の番号 | 群 |
|---|---|
| 00・10・20〜22 | `00_train`(1)、`10_eps`(1)、`10_train`(4)、`20_eps`(10)、`21_eval`(38)、`22_eval`(12) |
| 32〜41 | `32_agent_base`(12)、`33_agent_action`(9)、`34_agent_plasticity`(52)、`35_agent_churn`(13)、`36_agent_iqn`(16)、`36_agent_munchausen`(7)、`36_agent_quantile`(1)、`37_agent_learn`(4)、`37_agent_qtd`(44)、`38_agent`(12)、`38_agent_loss`(13)、`39_agent`(1)、`39_agent_per`(89)、`40_agent_rs`(6)、`41_agent_on`(8) |
| 42 | `42_env`(86) |
| 43〜50 | `43_agent_img`(6)、`44_agent_img`(2)、`45_agent_img`(9)、`46_agent_replay_fit`(13)、`47_agent_noisy`(11)、`48_agent_attention`(4)、`49_agent_gradnoise`(15)、`50_agent_skip`(72) |
| 51〜55・60 | `51_eval1`(61)、`52_eval2`(63)、`53_agent_img`(2)、`53_evalg`(16)、`54_evalgs`(16)、`55_agent_img`(2)、`60_agent_img`(57) |
| 90 | `90_perf`(26) |

- 共通の群は `metrics_scalar.txt` が持ち、env 固有の群（`42_env`・評価の群・画像の群の一部）は env の設定ファイル（Atari・DropMerge・GridMaze・GridMaze_muzero・LunarLander・ImageCls・CartPole）が持つ。
- `38_agent`・`39_agent` は ImageCls だけが使う（損失と学習率）。`37_agent_learn` は GridMaze_muzero だけが使う。
- 評価の群は、Atari と DropMerge の `51_eval1`〜`54_evalgs` と、GridMaze・LunarLander・DropMerge の `21_eval`・`22_eval` の 2 通りがある。
- 設定の外に、[atari5.py](../../viewers/metrics-tools/atari5.py) が Atari-5 の集約を疑似 Run として実 Run と同じ Workspace に書き出す群 `60_atari5` がある（1078 行の `EXPORT_GROUP`。[AGENTS.md](../../AGENTS.md) の Run 結果分析の節）。

### 2.2 Atari の env・評価の群の中

`42_env`・`51_eval1`・`52_eval2`・`53_evalg`・`54_evalgs` の中は、1x が得点（10 平均、11 平均の EMA、12〜15 HNS、16 max）、2x が長さ、3x が残機、4x が Breakout の閾値、5x が RAM である（[Atari.txt](../../apps/runner/config/Atari.txt) の `M1.[…]` の行）。RAM は 1〜3 が mean 50/52/54・EMA 51/53/55・max 56/57/58 で、4 は暫定で mean・EMA・max とも 59 にした（PRD082 D20）。evalg・evalgs は EMA を持たない。

### 2.3 tag 名を持つもの

- 設定: `apps/runner/config/` の `metrics_scalar.txt`・`metrics_image.txt` と env の設定ファイル。
- テスト: [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp)（RAM の tag の規則、`50_agent_skip` などの群の確認）、[observers_test.cpp](../../core/anet-core/src/observers_test.cpp)、[dqn_munchausen_test.cpp](../../core/anet-core/src/dqn_munchausen_test.cpp)（`36_agent_munchausen`）、[inspect_run_test.py](../../viewers/metrics-tools/inspect_run_test.py)、[atari5_test.py](../../viewers/metrics-tools/atari5_test.py)。
- 解析ツール: [atari5.py](../../viewers/metrics-tools/atari5.py)（`60_atari5`）、[btr_to_metrics.py](../../viewers/metrics-tools/btr_to_metrics.py)。後者は RAM の max の tag を番号によらず `56_ram_metric_{n}_max` にしている（171〜183 行。1 以外では今の設定の tag と合わない）。
- PRD の検証資材: `core/anet-core/testdata/prd0xx/` と `prd1xx/` の解析スクリプト（prd084・092・093・097・098・100・101・103・104・106 など）。
- 文書: [AGENTS.md](../../AGENTS.md)（Run 結果分析の節が群を挙げる）、[020 Run 実行ユーザーガイド](../design/020_user_guide_run.jp.md)、[030 分析ユーザーガイド](../design/030_user_guide_analysis.jp.md)、[140 可観測性](../design/140_observability.jp.md)、[200 DQN 系列](../design/200_dqn_agents.jp.md)（`35_agent_churn`・`46_agent_replay_fit`・`47_agent_noisy`・`48_agent_attention`・`49_agent_gradnoise`）と各 `.en.md`、[220 Atari 環境](../design/220_atari_env.jp.md)、`docs/memo/` の PRD と実装計画、ADR（[0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md) など）、wiki のページ。
- 過去の記録: 実験記録と、その設定の写し（`docs/experiments/**/config/*.txt`）、Run のフォルダ。

### 2.4 Workspace

番号付きの bat は `:use_ws <名前> <設定>` で Workspace を選び、無ければ `config/_main.txt` に `$include <設定>` を書いて作る（[11_batch_run.bat](../../apps/11_batch_run.bat) 47〜51 行）。11・12 は `Atari5-01`、13（サブ機）は `AtariSub-01` を使う。Atari-xx の最新は `Atari-04` で、今はどの bat も使っていない。

## 3. ゴール・非ゴール・成功の測り方

**ゴール**

- G1: 群の番号を、十の位で群の分類が分かる形にし、1 つの群の番号には 1 つの群だけを置く。agent の群は 6x 以降にまとめる。
- G2: 群の中の番号の決まりを規約に書き、新しい指標（RAM の番号を足すなど）が暫定の番号を使わずに置き場を得られるようにする。
- G3: RAM の指標が使う群の中の番号を 1 つの十の位に収め、tag から RAM の番号と統計（平均・EMA・max）が読めるようにする。
- G4: 今使っているもの（設定・テスト・解析ツール・文書）を 1 回で新しい番号へ移し、Atari の新しい Run は新しい Workspace に積む。

**非ゴール**

- 過去の Run・実験記録・設定の写しを書き換えること。旧 tag を新 tag へ読み替える仕組み（Viewer の別名、変換ツール）を作ること。理由は [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針。ユーザーは既存 Run との互換が切れることを許容した。
- 記録する値・event・軸・EMA の係数を変えること。変えるのは tag の名前だけである。
- 設定のキー（`ram_metric.[n]` などの source key、`metrics.scalar.@…` のプロファイル名）を変えること。

**成功の測り方**

1. 同梱の設定と atari5.py の群で、1 つの群の番号を使う群が 1 つだけで、群の番号の十の位が群の名前の分類（D2）と合う（§7 の受入 1・4）。
2. RAM の tag の群の中の番号が 50 + RAM の番号で、暫定の 59 が無い（受入 1）。
3. 新しい Workspace の Run を Viewer で開くと、agent の群が 6x〜8x に並ぶ（受入 3）。

## 4. 確定事項（暫定の既定。§9 で見直す）

| # | 決定 | 理由 |
|---|---|---|
| D1 | 範囲は、同梱の設定（`apps/runner/config/*.txt`）が定義する scalar・trace・画像の tag の全部と、atari5.py が書き出す群 | 群の番号の重なりは env の設定ファイルをまたいで起きており、atari5.py の群も実 Run と同じ Workspace に並ぶ（§2.1）。一部だけ直すと重なりが残る |
| D2 | 群の番号の十の位を、群の名前の分類に当てる。`train`・`eps`・`eval` は 00〜29（Runner の進み）、`env` は 40〜49、`eval1`・`eval2`・`evalg`・`evalgs` と評価の集約 `atari5` は 50〜59、名前が `agent` で始まる群は 60〜89、`perf` は 90〜99。今ある env と評価 session の群の番号は変えない | ユーザーの「きれいに使えそうなのは6x以降」。env と評価の群は Run の成績の比較に使う tag（実験記録は評価の群の値で成績を比べる。例 [2026-10-15 の実験記録](../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md) の `53_evalg`）なので、番号を保てば過去の Run と成績を並べ続けられる。D3 の agent の群・`10_eps`・`60_atari5` を動かせば、問題 1・2 が解ける |
| D3 | agent の群と `10_eps`・`60_atari5` は §5.1 の表のとおり移す。agent の群は、6x に学習の更新、7x に方策・行動・入力の正規化、8x に NN の状態と画像を置く。1 つの群の番号に 1 つの群 | 分類ごとに並べると、Viewer で関係する群が隣り合う。表は起票時の既定で、分類の境目は §9 の 1 で決め直す |
| D4 | 群の中の番号は、十の位を分類に、一の位を分類の中の順に使う。添字で並ぶ指標（RAM の番号）は一の位を添字にし、同じ添字の統計は同じ番号のまま名前の末尾（`_mean`・`_mean_ema`・`_max`）で分ける。RAM は `5n_ram_metric_n_mean`・`5n_ram_metric_n_mean_ema`・`5n_ram_metric_n_max`（n = 1〜9） | 今の tag も一の位を分類の中の順に使っている（得点の 10・11・12…）。[impl.md](../quality/impl.md) の設定の並び順の規約は「一の位は 0 で空ける」を本則にするが、tag にはこの本則を当てず、同じ規約の「添字で並ぶ列だけ一の位を使う」だけを RAM に当てる。RAM は今と同じ 5x の十の位だけを使い（ユーザーの「ramだけに番号をあまりさきたくない」）、番号を 9 個まで足せる。tag の番号の下 1 桁と名前の両方から RAM の番号が読める（「可読性を大きく落としたくもない」） |
| D5 | 旧 tag の別名・変換・案内を置かない。Atari の新しい番号の Run は新しい Workspace（`Atari5-02`・`Atari-05`・`AtariSub-02`）に積む。Atari 以外の env の Workspace は切り替えない | 非ゴール。Atari では RAM の tag（§5.2）が成績の読みに使う評価の群の中で変わるので、同じ Workspace に旧 tag と新 tag の Run が混ざると、Viewer で同じ指標が別の tag に分かれる。Workspace を分ければ、比べる Run の組が tag ごとに揃う。Atari 以外の env では成績の比較に使う群（`21_eval`・`22_eval`・`51_eval1`・`52_eval2`）の tag が変わらず、分かれるのは agent の群の tag だけなので、それを受け入れて切り替えない（§9 の 5） |
| D6 | tag の番号の決まり（D2 の群の番号の分類、D4 の群の中の番号）を、設定の並び順の規約とは別に [impl.md](../quality/impl.md) の「metrics の tag の文字数」の隣へ規約として書き、観点の項目を足して、tag を足した・変えた実装のレビューで見る。[140 §6](../design/140_observability.jp.md) からその規約を指す。同梱の設定のテスト（`[shipped_config]`）で確かめるのは、D2 の分類、群の番号の重なりの無さ、RAM の tag の形（D4 のうち RAM だけ）、40 文字の上限である | 規約が無かったことが問題 3 の原因である。テストで群の番号を確かめれば、新しい群を足したときに分類から外れたものが落ちる。群の中の番号の一般則は、今ある agent の群の中の番号を本書で振り直さないので（§8）、テストにせず、足した・変えた tag をレビューで見る |
| D7 | 切り替えは番号付きの bat（11・12・13）が止まっている間に行い、bat の Workspace の行もそのときに直す | cmd は bat を実行しながら読み進めるので、走っている bat を書き換えると以降の行がずれる。Run の途中で設定の tag が変わらないようにもする |

## 5. 実装契約

### 5.1 群の番号の割り当て（既定）

変えない群: `00_train`・`10_train`・`20_eps`・`21_eval`・`22_eval`・`42_env`・`51_eval1`・`52_eval2`・`53_evalg`・`54_evalgs`・`90_perf`。

| 旧 | 新 | 分類 |
|---|---|---|
| `10_eps` | `11_eps` | Runner の進み（`10_train` との重なりを解く） |
| `60_atari5`（atari5.py） | `59_atari5` | 評価の集約 |
| `32_agent_base` | `60_agent_base` | 学習の更新 |
| `38_agent_loss` | `61_agent_loss` | 学習の更新 |
| `37_agent_qtd` | `62_agent_qtd` | 学習の更新 |
| `37_agent_learn` | `63_agent_learn` | 学習の更新 |
| `36_agent_munchausen` | `64_agent_munchausen` | 学習の更新 |
| `39_agent_per` | `65_agent_per` | 学習の更新 |
| `46_agent_replay_fit` | `66_agent_replay_fit` | 学習の更新 |
| `49_agent_gradnoise` | `67_agent_gradnoise` | 学習の更新 |
| `38_agent` | `68_agent` | 学習の更新（ImageCls） |
| `39_agent` | `69_agent` | 学習の更新（ImageCls） |
| `33_agent_action` | `70_agent_action` | 方策・行動・入力の正規化 |
| `36_agent_iqn` | `71_agent_iqn` | 方策・行動・入力の正規化 |
| `36_agent_quantile` | `72_agent_quantile` | 方策・行動・入力の正規化 |
| `47_agent_noisy` | `73_agent_noisy` | 方策・行動・入力の正規化 |
| `50_agent_skip` | `74_agent_skip` | 方策・行動・入力の正規化 |
| `40_agent_rs` | `75_agent_rs` | 方策・行動・入力の正規化 |
| `41_agent_on` | `76_agent_on` | 方策・行動・入力の正規化 |
| `34_agent_plasticity` | `80_agent_plasticity` | NN の状態と画像 |
| `35_agent_churn` | `81_agent_churn` | NN の状態と画像 |
| `48_agent_attention` | `82_agent_attention` | NN の状態と画像 |
| `43_agent_img`・`44_agent_img`・`45_agent_img`・`53_agent_img`・`55_agent_img`・`60_agent_img` | `83_agent_img`〜`88_agent_img`（旧の番号の順） | NN の状態と画像 |

群の中の番号（`/` の後ろ）は、§5.2 の RAM を除いて変えない。

### 5.2 Atari の env・評価の群の RAM

| RAM | 旧 | 新 |
|---|---|---|
| 1 | `50_ram_metric_1_mean`・`51_ram_metric_1_mean_ema`・`56_ram_metric_1_max` | `51_ram_metric_1_mean`・`51_ram_metric_1_mean_ema`・`51_ram_metric_1_max` |
| 2 | `52_ram_metric_2_mean`・`53_ram_metric_2_mean_ema`・`57_ram_metric_2_max` | `52_ram_metric_2_mean`・`52_ram_metric_2_mean_ema`・`52_ram_metric_2_max` |
| 3 | `54_ram_metric_3_mean`・`55_ram_metric_3_mean_ema`・`58_ram_metric_3_max` | `53_ram_metric_3_mean`・`53_ram_metric_3_mean_ema`・`53_ram_metric_3_max` |
| 4 | `59_ram_metric_4_mean`・`59_ram_metric_4_mean_ema`・`59_ram_metric_4_max`（暫定） | `54_ram_metric_4_mean`・`54_ram_metric_4_mean_ema`・`54_ram_metric_4_max` |

5 つの群とも同じ。evalg・evalgs は EMA を持たない（今と同じ）。`51_ram_metric_1_mean_ema` と `52_ram_metric_2_mean` は旧と同じ文字列のまま残る。最長の tag は `51_eval1/51_ram_metric_1_mean_ema`（33 文字）で、40 文字の上限に収まる。

### 5.3 移行対象

| 種類 | 対象 | 直し方 |
|---|---|---|
| 設定 | `apps/runner/config/` の `metrics_scalar.txt`・`metrics_image.txt`・`Atari.txt`・`DropMerge.txt`・`DropMerge_optuna.txt`・`GridMaze.txt`・`GridMaze_muzero.txt`・`LunarLander.txt`・`LunarLander_cpu.txt`・`ImageCls.txt`・`CartPole.txt` | §5.1・§5.2 の表で tag を書き換える |
| テスト | `config_baseline_test.cpp`（RAM の規則、群の確認）、`observers_test.cpp`、`dqn_munchausen_test.cpp`、`inspect_run_test.py`、`atari5_test.py` | 新しい tag に合わせる。D6 の規則のテストを足す |
| 解析ツール | `atari5.py`（`EXPORT_GROUP` を `59_atari5` へ）、`btr_to_metrics.py`（RAM の tag を §5.2 の形にし、max の不具合も直す） | 新しい tag を出す・読む |
| 文書 | AGENTS.md、020・030・140・200・220（`.jp.md` と `.en.md`）、impl.md（D6 の規約と観点の項目） | 群と tag の例を新しい番号に直す |
| 実装計画の無い PRD | `docs/memo/` の、`NNN_*_2[0-9]impl.md` を持たず `done/` にも無い PRD（起票時点で 096・107 と `999_*`。本書を除く） | 契約・設定例・受入に出る tag を新しい番号に直す（これから実装するときに新しい tag を使うため） |
| PRD082 | D19・D20 の行 | 本書へ置き換えた旨の注記を足す（本文は書き換えない） |
| bat | `11_batch_run.bat`・`12_batch_run.bat` の `:use_ws Atari5-01` を `Atari5-02` へ、`13_batch_run.bat` の `:use_ws AtariSub-01` を `AtariSub-02` へ | D7 のとおり bat が止まっている間に直す |
| Workspace | `Atari-05` | bat の `:use_ws` と同じ形（`config/_main.txt` に `$include <Atari.txt>`）で作る |
| wiki | `docs/wiki/` のページ | 直接は書き換えない。上で直した設計文書と PRD を `anet-ingest-wiki` で取り込み、追従させる。実験記録から写した過去の tag はそのまま |
| 直さない | 実験記録とその設定の写し、Run のフォルダ、ADR、`done/` にある PRD と、実装計画を持つ PRD の本文と実装計画（実装した時点の記録）、`core/anet-core/testdata/` の検証資材（§9 の 4） | 非ゴール |

### 5.4 クラス一覧

足す・変える・消すクラスは無い（設定・テスト・解析ツール・文書だけを変える）。

## 6. 細部（暫定で置いた既定）

1. 画像の群 6 つの新しい群の番号は、旧の番号の順に 83〜88 を振る（理由: 今の並びを保つ）。
2. `10_eps` は `11_eps` にする（理由: 00〜29 の分類の中で、`10_train` の隣に置く）。
3. `60_atari5` は `59_atari5` にする（理由: 評価の集約なので 5x に置き、評価 session の群の後ろに並べる。55〜58 は評価 session を足すときのために空ける）。
4. 新しい Workspace の名前は、今の番号の次（`Atari5-02`・`Atari-05`・`AtariSub-02`）。

## 7. 検証と受入

1. 規則のテスト（`[shipped_config]`）
   - 母集団: 同梱の env の設定ファイル（Atari・DropMerge・DropMerge_optuna・GridMaze・GridMaze_muzero・LunarLander・LunarLander_cpu・ImageCls・CartPole）を、それぞれ `_main.txt` とあわせて読み込んだ設定データのキーの全部。未選択のプロファイルの定義と、`M1.[…]` のような別名の行も含む。キーに含まれる `[<2 桁の数字>_<英小文字・数字・_>/` から群を取り出す。
   - 確かめること:
     - (a) 全 env の設定ファイルをまたいで、1 つの群の番号を使う群が 1 つだけ。
     - (b) 群の番号の十の位が、群の名前の分類（D2）と合う。D2 に無い群の名前は不合格。
     - (c) 名前が `ram_metric_` を含む tag は Atari の 5 つの群にだけあり、群の中の番号が 50 + RAM の番号である。
     - (d) tag が 40 文字以内。
2. 旧 tag の不在: 次の範囲を、§5.1 の旧の群（`60_atari5` を含む）の文字列で、前後が英数字でも `_` でもない位置に限って検索し（正規表現 `(?<![0-9A-Za-z_])<旧の群>(?![0-9A-Za-z_])`。`/` の前に置いた tag も、`` `34_agent_plasticity`群 `` のように群だけを挙げた記述も当たり、`38_agent_loss` の中の `38_agent` は当たらない）、0 件。RAM は、`ram_metric_<n>_` の前の 2 桁が 50 + n でない tag が 0 件。
   - 範囲: `apps/runner/config/`、`core/` の `src/`、`apps/runner/src/`、`apps/metrics-viewer/src/`、`viewers/metrics-tools/`、`docs/design/`、`docs/quality/`、`AGENTS.md`、§5.3 の「実装計画の無い PRD」。
   - 範囲の外: 本書、§5.3 の「直さない」、PRD082（注記だけ）、wiki。
3. smoke: 各 env の設定で短い Run（`--config` 直接モード、CLI で `app.batchrun.exp_exit_step=1`）を流す。Run が起動時に書く `json/metrics.scalar.defs.json`・`json/metrics.trace.defs.json` の tag の群が、すべて §5.1 の新の群か変えない群で、旧の群が無い。定義は起動時に書かれるので、指標が発火するかどうかには依らない。既定の連鎖に入らない群は、受入 1 で確かめる。
4. 解析ツール:
   - `atari5_test.py`・`inspect_run_test.py` が通る。
   - atari5.py の書き出す群が `59_atari5` である。
   - `btr_to_metrics.py` を同じ入力で RAM の番号 1 と 2 として 2 回走らせると、RAM の tag（mean・EMA・max）がそれぞれ `51_`・`52_` で始まる。
5. 既存テスト: `anet-core-test` と `AtariEnv-test` の結果が本書の前と同じ（前から落ちているものを除く）。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| 旧 tag の別名・Viewer での読み替え・変換ツール | 不採用 | 非ゴール。クリーンブレーク方針。旧 Run との比較は旧 Workspace で行う |
| env と評価の群（`42_env`・`51_eval1`〜`54_evalgs`）の番号を動かす | 不採用（既定） | 成績の比較に使う tag で、動かすと過去の Run と並べられなくなる（D2）。本当の制約ではなく選択なので、§9 の 2 で開き直せる |
| agent の群の中の番号（`/` の後ろ）の振り直しと、その一般則のテスト | 先送り | 本書は群の番号の重なりと散らばりを解く（D6 の理由）。再開条件: 群の中で同じ番号の重なりや、D4 から外れたものが見つかったとき |
| RAM の案 A（統計ごとに 5x・6x・7x）・案 B（RAM の番号ごとに十の位）・案 C（50・51・52 を統計に、RAM の番号は名前）・案 E（全部 50） | 不採用（既定） | A と B は十の位を 3 つ以上使う。C と E は、添字の一の位の規約（impl.md）から外れる。§9 の 3 で開き直せる |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | agent の群の分類の境目（6x・7x・8x に何を置くか。munchausen を学習の更新に置くか、方策に置くか。画像の群を NN の状態と同じ十の位に置くか） | §5.1 の表 |
| 2 | env と評価の群の番号も動かすか。評価の群が `21_eval`・`22_eval`（GridMaze・LunarLander・DropMerge）と `51_eval1`〜`54_evalgs`（Atari・DropMerge）の 2 通りあるのを揃えるか | 動かさない、揃えない（D2） |
| 3 | RAM の振り方（§8 の案 A・B・C・E と D4 のどれか） | D4（5n） |
| 4 | 検証資材（`core/anet-core/testdata/` の解析スクリプト）を新しい tag へ移すか | 移さない（それぞれの PRD の段の Run を読むもので、その Run は旧 tag）。再開条件: 切り替えのあとにその PRD の段を流すとき、そのスクリプトだけ移す |
| 5 | Atari 以外の Workspace（DropMerge・GridMaze・GridMaze_muzero・LunarLander・ImageCls・CartPole）と、番号付きの bat が使わない Atari の `AtariTube-01` も切り替えるか | 切り替えない（D5）。`AtariTube-01` は使われているかが分からず、使うときに新しい番号の Workspace を作れば足りる。成績の比較に使う群（`21_eval`・`22_eval`・`51_eval1`・`52_eval2`）は変わらず、agent の群の tag だけが旧 Run と分かれる |
| 6 | 群の番号の決まりを ADR にするか | 起こさない。規約（impl.md）とテストで足りるなら、ADR の 3 条件の「戻しにくい」に当たらない |
| 7 | `38_agent`・`39_agent`（ImageCls）を 1 つの群にまとめるか、名前を付け直すか | まとめず、群の番号だけ 68・69 へ |

## 10. 依存と順序

- PRD082 D20（RAM 4 の暫定の 59）を本書の §5.2 で置き換える。実装のとき、PRD082 の D19・D20 の行に本書への注記を足す。
- 切り替えは、11・12・13 の bat が止まっている間に行う（D7）。

## 11. 参照

- [PRD082 RAM メトリクス](082_atari_ram_progress_metrics_10prd.md)
- [実装 品質確認観点](../quality/impl.md)（「metrics の tag の文字数」「並び順を番号で決める設定」）
- [140 可観測性（設計文書）](../design/140_observability.jp.md)、[200 DQN 系列（設計文書）](../design/200_dqn_agents.jp.md)、[220 Atari 環境（設計文書）](../design/220_atari_env.jp.md)、[210 Metrics Viewer（設計文書）](../design/210_metrics_viewer.jp.md)
