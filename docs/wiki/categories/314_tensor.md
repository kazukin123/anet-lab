# 314 テンソル処理

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

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
- 2026-09-24: device文字列を解決するParseDeviceをtensor_utilにまとめた（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD8）。
- 2026-09-27: ActionContext廃止に合わせstackerをActor私有としstacker側のAgent依存を除いた（出典: [PRD083](../../memo/done/083_action_context_retire_10prd.md)、確定した設計判断5・7）。
- 2026-10-21: scalarの読み口統一に合わせScaler群の値取得を接続した（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D1・§4.3 規約・ADR・コードから導いたこと）。

## 選ばなかった形

- 低解像度動画をNVENC用にpaddingする形 — 先送り: 出力寸法と毎frame処理の複雑化に見合う実需要がない。再開条件: NVENCを強制維持する実需要が出たとき（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、ゴール / 非ゴール）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続の観測パイプラインPRDで導入する。（出典: [PRD083](../../memo/done/083_action_context_retire_10prd.md)、複雑さ監査）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- H2DとLearner streamを安全につなぐeventのrecord/waitを維持し、性能改善のために消さない（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、Context / Problem Statement）。
- 観測のstack→device転送→正規化の並び、乱数の派生・消費順、aux集合を変えない。同seed・config・exp予算のDefaultDQN/LunarLander/stack4でmetrics checksumを比べ、checkpointのraw checksumはゲートにしない（出典: [PRD083](../../memo/done/083_action_context_retire_10prd.md)、Context・確定した設計判断2〜4・11）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。

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
- [PRD070](../../memo/done/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD083](../../memo/done/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD116](../../memo/116_rnd_lifelong_novelty_10prd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD
- [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md) — PRD122: 小さなリファクタリングとテストの空白をまとめて片づける

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261111-5f695d](../../management/issues/issue-20261111-5f695d.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 314_tensor | 基盤 | リファクタリング | — | 未確認 | 報酬の scaler と観測の normalizer で、clip と変換の同じ数行が 4〜6 か所にあり、clip が無効のときの扱いが既に食い違っている |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD945](../../memo/945_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 未着手 | 2.0（引継ぎ） | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [PRD944](../../memo/944_observation_pipeline_10prd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | 成果物 | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | アルゴリズム | リファクタリング | — | 未確認 | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | 成果物 | PRD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261103-9df5c5](../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | [issue-20261103-80c355](../../management/issues/issue-20261103-80c355.md) | 成果物 | PRD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [PRD942](../../memo/942_notifier_scoped_detach_10prd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |

着手できる 5 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-80c355](../../management/issues/issue-20261103-80c355.md) — Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る
<!-- issues:end -->
