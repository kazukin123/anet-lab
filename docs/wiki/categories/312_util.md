# 312 ユーティリティ

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- 設定からRunを組み立てる実行基盤のうち、初期化、seed、schedule、log、application共通処理の所在を引く。（出典: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2・§3・§7）。
- ProfiledValueは、定数をrootのbare値で、時間で変わる値を`type`（`linear`・`cosine`・`cosine_restart`・`exponential`・`phased`）と対応する欄で表す。読み取りと記録を分け、非constantの配下に継承されたbare値は保持しても評価とdumpに使わず、dumpには実効typeが参照する欄だけを記録する（出典: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2.1「ProfiledValue の定数と時間の値」）。

## 成立の経緯

- 2026-06-15: 学習の同 seed 再現性のため `setDeterministicAlgorithms` をグローバル既定で有効化する（出典: [ADR 0006](../../adr/0006-deterministic-algorithms.md)、本文）。
- 2026-07-16: Train Actorの周期同期に共通scheduleを接続した（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、題・目的）。
- 2026-07-20: batch入力の移行で共通の定義・接続を更新した（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-17: PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md)、題・目的）。
- 2026-08-17: 飛び飛びのカウンタに対する interval を bucket-crossing で定義する（出典: [ADR 0028](../../adr/0028-interval-fires-on-bucket-crossing.md)、本文）。
- 2026-08-21: AMP対応で共通初期化の接続を更新した（出典: [PRD033](../../memo/done/033_imagecls_bf16_head_10prd.md)、Context（背景・目的））。
- 2026-08-21: Envインスタンスname PRD（出典: [PRD037](../../memo/done/037_env_instance_name_10prd.md)、題・目的）。
- 2026-08-21: Envログ name prefix 自動付与 PRD（出典: [PRD038](../../memo/done/038_env_log_auto_prefix_10prd.md)、題・目的）。
- 2026-08-21: PRD 045: EmaFilter バイアス補正ウォームアップ（出典: [PRD045](../../memo/done/045_ema_debias_warmup_10prd.md)、題・目的）。
- 2026-08-21: ProfiledValue による ImageCls learning_rate 制御（出典: [PRD031](../../memo/done/031_imagecls_cosine_lr_10prd.md)、題・目的）。
- 2026-08-21: TBO診断に関わるlog基盤を更新した（出典: [PRD008](../../memo/done/008_Transformed%20Bellman%20Operator_10prd.md)、3. 報酬スケーラとの共存と warn）。
- 2026-08-21: TensorDict可視化の共通定義を更新した（出典: [PRD010](../../memo/done/010_image_metrics_v2_10prd.md)、Phase 3-4 — network sweep 基盤 + SweepStateHeatMap observer / UI 復旧）。
- 2026-08-21: `backend.deterministic_algorithms` 導入（学習の同 seed 再現性確保）仕様書（出典: [PRD015](../../memo/done/015_deterministic_algorithms_10prd.md)、題・目的）。
- 2026-08-21: forward traceで使う共通の定義を更新した（出典: [PRD006](../../memo/done/006_conv2d_visual_trace_10prd.md)、5.1 【新規】`TraceSink` 型（当時の名前。今のコードに無い））。
- 2026-08-27: PRD 068: batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）（出典: [PRD068](../../memo/done/068_batchrun_fatal_error_handling_10prd.md)、題・目的）。
- 2026-09-04: 評価episode集約で使う共通utilityを更新した（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、F1. metrics 経路は既に「1 セッション 1 点」であり、停止条件だけ変えても N 本平均は録れない）。
- 2026-09-24: InitRLの初期化に明示deviceの可用性検査を加えた（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD10・Complexity audit）。

- 2026-10-25: ProfiledValueに指数とphaseの終値規則を加え、定数をbareで読み、実効typeが使う欄だけをdumpする（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§4.1 D2〜D4）。
- 2026-10-26: 既存の自動seed入口を使い、GridMazeの自動採番をRun seedから独立させた（出典: [PRD102](../../memo/done/102_gridmaze_seed_mode_10prd.md)、§2・§5.2）。
- 2026-10-29: 整数のProfiledValueをActorのスキップ長上限へ接続した（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§4 D1・D2・§5.1）。
- 2026-11-10: InitRLのdeviceの検査を、設定に書かれているキーだけを見る形にし、持ち主の設定クラスと二重に持っていた既定値の表を消す仕様を定めた（出典: [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md)、§4 D9）。

## 選ばなかった形

- SDPA の backend を math に固定する形 — 不採用（棄却）: 決定化はできるが math は遅く、mem-efficient の決定的 backward で同じ再現性が得られるので、backend を局所的に縛る複雑さと版差の脆さを払う価値がない（出典: [ADR 0006](../../adr/0006-deterministic-algorithms.md)、Considered Options）。
- SDPA 化を撤回して MultiheadAttention の経路へ戻す形 — 不採用（棄却。本 ADR の範囲外）: SDPA に実時間の利点は出なかったが、非決定は他の atomic 系 op でも起こり得るので本質解でなく、差分・互換のコストも大きい（出典: [ADR 0006](../../adr/0006-deterministic-algorithms.md)、Considered Options）。
- EmaFilter のウォームアップを累積平均から固定 α へ切り替える案（案 B） — 不採用（却下）: 序盤にトレンドする学習メトリクスでは一様平均が古い値を引きずってラグが出、切替点に折れ目ができ、設定項目ゼロの案 A に劣る（出典: [PRD045](../../memo/done/045_ema_debias_warmup_10prd.md)、5. 代替案と却下理由）。
- 値をバッファに溜めてウォームアップ後に遡及出力する案（案 C） — 不採用（却下）: MetricsLogger に過去 step への遡及出力 API が要って責務が漏れ、SQLite キャッシュ（PK=序数）と非互換で、グラフ冒頭が同じ値のフラット線になり、ライブビューも遅れる（出典: [PRD045](../../memo/done/045_ema_debias_warmup_10prd.md)、5. 代替案と却下理由）。
- 分散からウォームアップの終わりを自動判定する案（案 E） — 不採用（却下）: 平滑と収束判定の 2 関心を 1 機構に混ぜ、案 A では判定自体が要らない（出典: [PRD045](../../memo/done/045_ema_debias_warmup_10prd.md)、5. 代替案と却下理由）。
- thread-local logging context・structured logging・JSON logging — 範囲外（対象外）: Env ログの name prefix を扱う PRD の対象外に置かれた（出典: [PRD038](../../memo/done/038_env_log_auto_prefix_10prd.md)、対象外）。
- Logger の debug メンバ関数 — 不採用（追加しない）: debug ログの価値（guard による整形コストゼロ、ファイル名・行番号・関数名の自動付与、ビルドでの消去）はマクロでしか保てない（出典: [PRD038](../../memo/done/038_env_log_auto_prefix_10prd.md)、解決方針・対象外）。
- InitRLのdeviceの検査が、既定値を持つ設定クラスの定数を引く形 — 不採用: Runの設定の型をtrainer.cppの外へ出す変更が要り、書かれていないキーを飛ばせば今の結果のまま二重持ちが消える（出典: [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md)、§4 D9・§7）。
- 整形した文面をキーにして通知を 1 回だけ出す形 — 不採用: step や件数のように変わる値が文面に入ると毎回出て、抑えるときも毎回文面を組み立てる。通知を出す箇所と対象の名前の組をキーにする（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、§5.1・§8）。
- WARN と ERROR の間に新しいログのレベルを足す形 — 不採用: レベルを増やすと「WARN は利用者が直すべき状態」という線引きがぼやける（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、§3.2・§8）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- EMA内部の状態を変えても公開シグネチャを変更しない（出典: [PRD045](../../memo/done/045_ema_debias_warmup_10prd.md)、3. API 契約）。
- 同 seed の再現を確かめるときは `backend.deterministic_algorithms` を true、`backend.deterministic_warn_only` を false にする。warn_only を true にすると決定版の無い op を素通りさせるので再現性は保証されない（出典: [ADR 0006](../../adr/0006-deterministic-algorithms.md)、Consequences；[PRD015](../../memo/done/015_deterministic_algorithms_10prd.md)、1.1 API）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、PRD103の検証資材でmetrics checksumを一致させる（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§5.7・§7の受入1）。
- InitRLのdeviceの検査を変えるときは、明示したcudaの一括の報告を保つ。検査は設定に書かれたキーだけを見るので、持ち主のクラスの既定値に明示のcudaを置くときは検査を見直す。`[prd070]`のテストはCUDAが使える機械ではskipになるので、環境変数`CUDA_VISIBLE_DEVICES`を`-1`にして流し、skipを合格に数えない（出典: [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md)、§2.1・§6 A6）。

## 関係

- [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§3。
- 設計: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2・§3・§7。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [app_util.hpp](../../../core/anet-core/include/anet/app_util.hpp)
- [common.hpp](../../../core/anet-core/include/anet/common.hpp)
- [diag.hpp](../../../core/anet-core/include/anet/diag.hpp)
- [exception.hpp](../../../core/anet-core/include/anet/exception.hpp)
- [init.hpp](../../../core/anet-core/include/anet/init.hpp)
- [json_util.hpp](../../../core/anet-core/include/anet/json_util.hpp)
- [log.hpp](../../../core/anet-core/include/anet/log.hpp)
- [random.hpp](../../../core/anet-core/include/anet/random.hpp)
- [schedule.hpp](../../../core/anet-core/include/anet/schedule.hpp)
- [str_util.hpp](../../../core/anet-core/include/anet/str_util.hpp)
- [thread.hpp](../../../core/anet-core/include/anet/thread.hpp)
- [util.hpp](../../../core/anet-core/include/anet/util.hpp)
- [app_util.cpp](../../../core/anet-core/src/app_util.cpp) / [app_util_test.cpp](../../../core/anet-core/src/app_util_test.cpp)
- [app_util_impl.hpp](../../../core/anet-core/src/app_util_impl.hpp)
- [common.cpp](../../../core/anet-core/src/common.cpp)
- [diag.cpp](../../../core/anet-core/src/diag.cpp)
- [exception.cpp](../../../core/anet-core/src/exception.cpp) / [exception_test.cpp](../../../core/anet-core/src/exception_test.cpp)
- [init.cpp](../../../core/anet-core/src/init.cpp) / [init_test.cpp](../../../core/anet-core/src/init_test.cpp)
- [json_util.cpp](../../../core/anet-core/src/json_util.cpp)
- [log.cpp](../../../core/anet-core/src/log.cpp) / [log_test.cpp](../../../core/anet-core/src/log_test.cpp)
- [random.cpp](../../../core/anet-core/src/random.cpp)
- [schedule.cpp](../../../core/anet-core/src/schedule.cpp) / [schedule_test.cpp](../../../core/anet-core/src/schedule_test.cpp)
- [str_util.cpp](../../../core/anet-core/src/str_util.cpp)
- [thread.cpp](../../../core/anet-core/src/thread.cpp) / [thread_test.cpp](../../../core/anet-core/src/thread_test.cpp)
- [util.cpp](../../../core/anet-core/src/util.cpp) / [util_test.cpp](../../../core/anet-core/src/util_test.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [PRD053](../../memo/done/053_interval_gate_perf_ema_10prd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD070](../../memo/done/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md) — PRD122: 小さなリファクタリングとテストの空白をまとめて片づける

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-c8fb06](../../management/issues/issue-20261103-c8fb06.md) | 未着手 | 3.0 | — | 成果物 | PRD | 312_util, 110_runner_app | 基盤 | 機能改善 | [PRD071](../../memo/071_uncaught_fatal_visibility_10prd.md) | 残件有り | PRD 071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| [issue-20261103-78df9c](../../management/issues/issue-20261103-78df9c.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util | 基盤 | 不具合 | — | 残件有り | device指定の文法誤りで、ParseDeviceとInitRLが同じ誤りのfatalを2回ログに出す |
| [issue-20261103-9b314a](../../management/issues/issue-20261103-9b314a.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util | 基盤 | 不整合 | — | 残件有り | 明示cudaが使えないときのエラーがキー名だけを並べ、指定値を書かない |
| [issue-20261103-d4258c](../../management/issues/issue-20261103-d4258c.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util, 316_config | 基盤 | 不整合 | — | 残件有り | 実効deviceを設計文書は「採用値」と呼び、PRD070とJSONの欄はeffective（実効）と呼ぶ |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD945](../../memo/945_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-df24e5](../../management/issues/issue-20261103-df24e5.md) | 未着手 | 1.4 | — | 成果物 | 監査 | 312_util | 基盤 | 不具合 | — | 判定不能 | AnetExceptionのstack_trace()がwxStackWalkerに頼っていて、Runnerのプロセスでは空になる |
| [issue-20261103-6a32b4](../../management/issues/issue-20261103-6a32b4.md) | 未着手 | 1.0 | — | 成果物 | PRD | 315_nn, 321_agent_core, 318_rl_core, 312_util | アルゴリズム | リファクタリング | [PRD938](../../memo/938_nn_trace_capture_unification_10prd.md) | 残件有り | forward の名前付き出力を NetworkForwardCapture に一本化する |
| [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [PRD954](../../memo/954_scalar_threshold_rate_10prd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | 成果物 | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | アルゴリズム | リファクタリング | — | 未確認 | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-2dd820](../../management/issues/issue-20261103-2dd820.md) | 未着手 | 1.0 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools, 322_env_individual | 開発 | リファクタリング | [PRD957](../../memo/957_source_layout_10prd.md) | 残件有り | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける） |

着手できる 10 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
