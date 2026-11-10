# 317 可視化・メトリクス

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- eventからscalar、trace、画像、動画、GraphVizとartifactへ変換し、値の取得と保存の境界をまとめる。（出典: [可観測性](../../design/140_observability.jp.md)、§1.1・§2・§3）。
- TimeHeatMapのScale描画・Histgram・SweepedHeatMapは画像自身が画素領域を所有し、生成元を破棄した後も残る画像のコピーが画素を保持する。確保失敗は描画対象と要求寸法を示して停止する（出典: [可観測性](../../design/140_observability.jp.md)、§7.3）。

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

- 2026-11-04: 画像メモリの所有権の不一致を、既存の寸法・画素値・画像種別を保って修正した（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§1.2、[可観測性](../../design/140_observability.jp.md)、§7.3）。

## 選ばなかった形

- HeatMap派生の描画処理の統合と、画像処理全体の確保・入力検証の一括変更 — 範囲外: 画像の4経路の不具合修正とは別の関心。再開条件: 描画の重複整理、または別の具体的な不具合を修正対象に選んだとき（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§1.3）。

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
- 宣言した scalar の値が取れないことを INFO にする形 — 不採用: 鍵の綴り違いのような本物の誤りを INFO に埋める。休止した機能と、設定で無効にした機能は、鍵を保って NaN を返す（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D12・§8）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch入力への移行でtrainのmax_steps episode、episode_start、可視化cadenceを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- 512/1024動画はautoでNVENCを選択し、従来どおり出力できることを確かめる（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、Testing Decisions）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- step traceの宣言が無い設定では、observerもstep部品も作らず、PRD103の検証資材の6構成のmetrics checksumを実装の前後で一致させる。既存の1行DSLのtraceの定義の記録（`json/metrics.trace.defs.json`）も前後で同一に保つ（出典: [919_eval_session_step_trace](../../memo/919_eval_session_step_trace_10prd.md)、§4 D2・D4・§7の受入1）。

- 画像の所有権を直すときは寸法・全画素・画像種別を固定入力で比較し、生成・破棄の不一致を修正前に検出するメモリ検査を修正後にも通す。通常Debugの完走だけで代替せず、環境不足の項目は未達として記録する（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§5 A1〜A3・A7）。

## 関係

- 設計: [可観測性](../../design/140_observability.jp.md)、§1.1・§2・§3。
- 画像の所有権と検証: [可観測性](../../design/140_observability.jp.md)、§7.3・§9。
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
- [heat_map.cpp](../../../core/anet-core/src/heat_map.cpp) / [heat_map_test.cpp](../../../core/anet-core/src/heat_map_test.cpp)
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
- [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md) — PRD115: 画像メモリの所有権とDropMergeのNOOP判定の修正

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261106-16b7af](../../management/issues/issue-20261106-16b7af.md) | 対応中 | 2.6 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 315_nn, 317_metrics | アルゴリズム | 機能追加 | [PRD116](../../memo/116_rnd_lifelong_novelty_10prd.md) | 残件無し | RND による一生の新しさだけの群（内的報酬の第一の群） |
| 2 | [issue-20261103-955e50](../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [PRD081](../../memo/081_evalpanel_episode_trace_10prd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| 3 | [issue-20261107-a9a0e3](../../management/issues/issue-20261107-a9a0e3.md) | 未着手 | 8.0 | — | 成果物 | その他 | 318_rl_core, 317_metrics, 324_dqn | 学習基盤 | 機能改善 | [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) | 未確認 | Atariの正常なRunで毎回出るWARN（eval_targetの未スケジュール、評価だけのRunのRND）で本物のWARNが埋もれる |
| 4 | [issue-20261103-ba635a](../../management/issues/issue-20261103-ba635a.md) | 未着手 | 6.5 | — | 成果物 | 監査 | 317_metrics | UI | 不具合 | — | 残件有り | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む |
| 5 | [issue-20261103-66100f](../../management/issues/issue-20261103-66100f.md) | 未着手 | 5.0 | — | 成果物 | 監査 | 316_config, 110_runner_app, 317_metrics | 基盤 | 不整合 | [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) | 残件有り | Conv2dPanel.conv2d.mergin_x/yとimage.@conv2d.conv2d.mergin_x/yが綴りの誤りで読まれない |
| 6 | [issue-20261107-c17068](../../management/issues/issue-20261107-c17068.md) | 未着手 | 2.6 | — | 成果物 | PRD | 316_config, 312_util, 315_nn, 317_metrics, 318_rl_core, 324_dqn, 110_runner_app | 基盤 | 機能改善 | [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) | 残件有り | WARN の意味の線引きと、読まれない設定の検出（NoCare の見直し） |
| 7 | [issue-20261103-eeb2c3](../../management/issues/issue-20261103-eeb2c3.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 317_metrics | UI | 不整合 | — | 残件有り | video_codecのエラー文が実キーapp.metrics_logger.video_codecではなくrootの無い名前を出す |
| 8 | [issue-20261103-e6dbb4](../../management/issues/issue-20261103-e6dbb4.md) | 未着手 | 2.3 | — | 成果物 | 監査 | 317_metrics | UI | リファクタリング | — | 残件有り | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| 9 | [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 10 | [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | 成果物 | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| 11 | [issue-20261103-50644a](../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | 成果物 | PRD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [PRD919](../../memo/919_eval_session_step_trace_10prd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| 12 | [issue-20261103-71c4f4](../../management/issues/issue-20261103-71c4f4.md) | 未着手 | 1.3 | — | 成果物 | PRD | 315_nn, 317_metrics, 321_agent_core | アルゴリズム | 機能追加 | [PRD920](../../memo/920_nn_block_metrics_10prd.md) | 残件有り | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| 13 | [issue-20261103-7227bf](../../management/issues/issue-20261103-7227bf.md) | 未着手 | 1.2 | — | 成果物 | PRD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [PRD947](../../memo/947_q_statistics_config_screening_10prd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| 14 | [issue-20261103-51220d](../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [PRD932](../../memo/932_episode_forensics_10prd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| 15 | [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | 成果物 | PRD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| 16 | [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 2.0 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [PRD921](../../memo/921_agent_prev_action_obs_10prd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |

着手できる 15 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化
<!-- issues:end -->
