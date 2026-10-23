# 317 可視化・メトリクス

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
- 2026-10-21: scalarとtraceのtarget解決を共用しメトリクス配線を分離した（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D1〜D3）。

## 選ばなかった形

- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§6）。
- metricsのtrace定義tagをEvaluation Traceの列として参照する形 — 不採用: 見るだけでも記録の宣言が要り、EMA・間隔・集約を持つscalar統計はstepの生値に合わない（出典: [ADR 0057](../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- trace の level（off / record / forensic） — 不採用（cut）: 宣言の有無で足りる。forensic は想像上の需要である（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、4.0 決定一覧 D2・9. 複雑さ監査）。
- trace の行の episode_id・model_version の欄 — 範囲外（cut。912・932 へ）: コードに概念が無い。再開条件: episode_id は 932 で EpisodeId が実装されたとき、model_version は 912 に着手するとき（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、8. フェーズとゲート・9. 複雑さ監査）。
- data を宣言順に出す専用の JSON 経路 — 不採用（cut）: 名前と値の対応で目的を満たし、取得順と定義の keys 配列だけを宣言順で保つ（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、9. 複雑さ監査）。
- eval scalar の @episode_end を WARN 付きで受理する alias — 不採用（cut）: リポジトリの外に workspace は無く、クリーンブレークの方針どおり fail-fast のメッセージで置換先を案内する（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、4.0 決定一覧 D5・D11・9. 複雑さ監査）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch入力への移行でtrainのmax_steps episode、episode_start、可視化cadenceを維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- 512/1024動画はautoでNVENCを選択し、従来どおり出力できることを確かめる（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、Testing Decisions）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。

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
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0008 P1 defect — VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む
- F-0011 P2 defect — wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する）
- F-0016 P2 refactor — HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->
