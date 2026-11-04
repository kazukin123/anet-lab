# 317 可視化・メトリクス

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- eventからscalar、trace、画像、動画、GraphVizとartifactへ変換し、値の取得と保存の境界をまとめる。（出典: [可観測性](../../design/140_observability.jp.md)、§1.1・§2・§3）。

## 成立の経緯

- 2026-06-09: 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する（出典: [ADR 0002](../../adr/0002-tensordict-function-unify.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [PRD010](../../memo/done/010_image_metrics_v2_10prd.md)、題・目的）。
- 2026-08-21: anet::nn::Network GraphViz 仕様（出典: [PRD004](../../memo/done/004_nndot_10prd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../adr/0037-metrics-trace-channel-and-session-end-event.md)、本文）。
- 2026-09-19: Actorカタログ変更後のメトリクス保存を回帰確認した（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-21: scalarとtraceのtarget解決を共用しメトリクス配線を分離した（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D1〜D3）。
- 2026-10-29: 評価sessionの採用episodeのstep traceを、Evaluation Traceと同じ列の書き方で`metrics.step_trace.[<tag>]`に宣言し、traceの行としてRunの記録に残す案を暫定で起票した（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、題・§4 D1・D2・D6）。

## 選ばなかった形

- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§6）。
- metricsのtrace定義tagをEvaluation Traceの列として参照する形 — 不採用: 見るだけでも記録の宣言が要り、EMA・間隔・集約を持つscalar統計はstepの生値に合わない（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- trace の level（off / record / forensic） — 不採用（cut）: 宣言の有無で足りる。forensic は想像上の需要である（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、4.0 決定一覧 D2・9. 複雑さ監査）。
- trace の行の episode_id・model_version の欄 — 範囲外（cut。912・932 へ）: コードに概念が無い。再開条件: episode_id は 932 で EpisodeId が実装されたとき、model_version は 912 に着手するとき（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、8. フェーズとゲート・9. 複雑さ監査）。
- data を宣言順に出す専用の JSON 経路 — 不採用（cut）: 名前と値の対応で目的を満たし、取得順と定義の keys 配列だけを宣言順で保つ（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、9. 複雑さ監査）。
- eval scalar の @episode_end を WARN 付きで受理する alias — 不採用（cut）: リポジトリの外に workspace は無く、クリーンブレークの方針どおり fail-fast のメッセージで置換先を案内する（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、4.0 決定一覧 D5・D11・9. 複雑さ監査）。
- traceの1行DSLに`@train`と複数のtargetを足す形 — 不採用: 1定義1targetの文法と検証（PRD069）を変えることになり、複数targetの列はEvaluation Traceの列の書き方がすでに表している（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§8）。
- 画面の列（`app.eval_trace_panel.columns.*`）をそのまま記録に使う形 — 不採用: 画面の設定と記録の宣言が結び付く（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§8）。
- 残機を失ったstepだけを書く形 — 不採用: 残機はAtari固有の値で、汎用のtraceのeventにEnv固有の判定を持ち込む（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§8）。
- step traceの行の間引き — 先送り: 評価専用のRunなら行数が許容の範囲。再開条件: 行数が読み手やViewerの扱える量を超えたとき（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§8）。
- 学習側（`$train`）のstep trace — 範囲外: 行数が学習の長さに比例して増える。再開条件: 学習中の行動をstepごとに追う必要が出たとき（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§8）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch入力への移行でtrainのmax_steps episode、episode_start、可視化cadenceを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- 512/1024動画はautoでNVENCを選択し、従来どおり出力できることを確かめる（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、Testing Decisions）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- step traceの宣言が無い設定では、observerもstep部品も作らず、PRD103の検証資材の6構成のmetrics checksumを実装の前後で一致させる。既存の1行DSLのtraceの定義の記録（`json/metrics.trace.defs.json`）も前後で同一に保つ（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§4 D2・D4・§7の受入1）。

## 関係

- 設計: [可観測性](../../design/140_observability.jp.md)、§1.1・§2・§3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [graphviz.hpp](../../../core/anet-core/include/anet/graphviz.hpp)
- [heat_map.hpp](../../../core/anet-core/include/anet/heat_map.hpp)
- [metrics.hpp](../../../core/anet-core/include/anet/metrics.hpp)
- [metrics_logger.hpp](../../../core/anet-core/include/anet/metrics_logger.hpp)
- [probe.hpp](../../../core/anet-core/include/anet/probe.hpp)
- [graphviz.cpp](../../../core/anet-core/src/graphviz.cpp)
- [heat_map.cpp](../../../core/anet-core/src/heat_map.cpp)
- [metrics.cpp](../../../core/anet-core/src/metrics.cpp) / [metrics_test.cpp](../../../core/anet-core/src/metrics_test.cpp)
- [metrics_impl.hpp](../../../core/anet-core/src/metrics_impl.hpp)
- [metrics_logger.cpp](../../../core/anet-core/src/metrics_logger.cpp) / [metrics_logger_test.cpp](../../../core/anet-core/src/metrics_logger_test.cpp)
- [metrics_logger_impl.hpp](../../../core/anet-core/src/metrics_logger_impl.hpp)
- [probe.cpp](../../../core/anet-core/src/probe.cpp) / [probe_test.cpp](../../../core/anet-core/src/probe_test.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 優先度 | 待ち | 種別 | 工程 | 領域 | 性質 | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261103-ba635a](../../management/issues/issue-20261103-ba635a.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 317_metrics | 不具合 | 未確認 | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む |
| 2 | [issue-20261103-6661d2](../../management/issues/issue-20261103-6661d2.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 110_runner_app, 317_metrics, 312_util, 322_env_individual | 索引 | 未確認 | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| 3 | [issue-20261103-71c4f4](../../management/issues/issue-20261103-71c4f4.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 315_nn, 317_metrics, 321_agent_core | 索引 | 未確認 | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| 4 | [issue-20261103-dd8058](../../management/issues/issue-20261103-dd8058.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 317_metrics | 不具合 | 未確認 | wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する） |
| 5 | [issue-20261103-e6dbb4](../../management/issues/issue-20261103-e6dbb4.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 317_metrics | 改善 | 未確認 | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| 6 | [issue-20261103-eeb2c3](../../management/issues/issue-20261103-eeb2c3.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 317_metrics | ずれ | 未確認 | video_codecのエラー文が実キーapp.metrics_logger.video_codecではなくrootの無い名前を出す |
| 7 | [issue-20261103-2349d0](../../management/issues/issue-20261103-2349d0.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 315_nn, 324_dqn, 314_tensor, 317_metrics, 110_runner_app, 322_env_individual | 索引 | 実装済み | TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）PRD |
| 8 | [issue-20261103-50644a](../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 320_env_core, 317_metrics, 321_agent_core, 318_rl_core, 110_runner_app | 索引 | 未確認 | 評価 session の step trace を Run の記録に残す PRD |
| 9 | [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 索引 | 未確認 | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 10 | [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | 索引 | 未確認 | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る PRD |
| 11 | [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 索引 | 未確認 | 直前行動観測の共通部品（`prev_action` フレーム段） |
| 12 | [issue-20261103-b4e5c9](../../management/issues/issue-20261103-b4e5c9.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 110_runner_app, 317_metrics, 318_rl_core, 321_agent_core, 312_util, 320_env_core, 314_tensor, 316_config, 311_dev_tools, 322_env_individual | 索引 | 未確認 | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) 暫定 PRD |
| 13 | [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 317_metrics, 314_tensor, 318_rl_core | 索引 | 未確認 | TimeHistogram fine-bin保持・描画時集約 PRD |

着手できる 13 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
