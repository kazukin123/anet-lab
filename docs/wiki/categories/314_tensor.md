# 314 テンソル処理

## 概要・責務

- device転送とframe stackを担い、sampleのTensorとCPUに残すmetadata、非同期転送の寿命を扱う。（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.2・§2.3）。

## 成立の経緯

- 2026-06-09: 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する（出典: [ADR 0002](../../adr/0002-tensordict-function-unify.md)、本文）。
- 2026-06-15: ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）（出典: [ADR 0005](../../adr/0005-sample-prefetch-stale-per.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、題・目的）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [PRD006](../../memo/done/006_conv2d_visual_trace_10prd.md)、題・目的）。
- 2026-08-21: Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、題・目的）。
- 2026-08-21: ReplayBuffer PER priority 転送境界整理 PRD（出典: [PRD021](../../memo/done/021_replay_per_priority_transfer_10prd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [PRD010](../../memo/done/010_image_metrics_v2_10prd.md)、題・目的）。
- 2026-09-05: trace取得時のscalerの読み口と回帰を確認した（出典: [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md)、題・目的）。
- 2026-09-19: Actorごとのframe stackerの構築・同期を接続した（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-24: device文字列を解決するParseDeviceをtensor_utilにまとめた（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD8）。
- 2026-09-27: ActionContext廃止に合わせstackerをActor私有としstacker側のAgent依存を除いた（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、確定した設計判断5・7）。
- 2026-10-21: scalarの読み口統一に合わせScaler群の値取得を接続した（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D1・§4.3 規約・ADR・コードから導いたこと）。

## 選ばなかった形

- 低解像度動画をNVENC用にpaddingする形 — 先送り: 出力寸法と毎frame処理の複雑化に見合う実需要がない。再開条件: NVENCを強制維持する実需要が出たとき（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、ゴール / 非ゴール）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続の観測パイプラインPRDで導入する。（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、複雑さ監査）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- H2DとLearner streamを安全につなぐeventのrecord/waitを維持し、性能改善のために消さない（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、Context / Problem Statement）。
- 観測のstack→device転送→正規化の並び、乱数の派生・消費順、aux集合を変えない。同seed・config・exp予算のDefaultDQN/LunarLander/stack4でmetrics checksumを比べ、checkpointのraw checksumはゲートにしない（出典: [PRD083](../../memo/083_action_context_retire_10prd.md)、Context・確定した設計判断2〜4・11）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。

## 関係

- 設計: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.2・§2.3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [image.hpp](../../../core/anet-core/include/anet/image.hpp)
- [scaler.hpp](../../../core/anet-core/include/anet/scaler.hpp)
- [stacker.hpp](../../../core/anet-core/include/anet/stacker.hpp)
- [tensor_check.hpp](../../../core/anet-core/include/anet/tensor_check.hpp)
- [tensor_util.hpp](../../../core/anet-core/include/anet/tensor_util.hpp)
- [transfer.hpp](../../../core/anet-core/include/anet/transfer.hpp)
- [image.cpp](../../../core/anet-core/src/image.cpp) / [image_test.cpp](../../../core/anet-core/src/image_test.cpp)
- [scaler_test.cpp](../../../core/anet-core/src/scaler_test.cpp)
- [scaler_impl.cpp](../../../core/anet-core/src/scaler_impl.cpp) / [scaler_impl.hpp](../../../core/anet-core/src/scaler_impl.hpp)
- [stacker.cpp](../../../core/anet-core/src/stacker.cpp) / [stacker_test.cpp](../../../core/anet-core/src/stacker_test.cpp)
- [tensor_check.cpp](../../../core/anet-core/src/tensor_check.cpp)
- [tensor_util.cpp](../../../core/anet-core/src/tensor_util.cpp)
- [transfer.cpp](../../../core/anet-core/src/transfer.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD083](../../memo/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD094](../../memo/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
- F-0026 P2 test-gap — 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効）

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->
