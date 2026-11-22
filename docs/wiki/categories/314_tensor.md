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
- 2026-11-12: 報酬スケーラに、lane ごとの割引収益の標準偏差で割る種類（`dynamic_mode = return_std`）を足し、動的な種類の状態を checkpoint に保存する仕様を、グリルで決めた（出典: [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§5 D9・D16・§6.6）。
- 2026-11-16: 観測の正規化の部品に統計だけを更新する口を足し、累積の統計の部品のbatch方向の集計の式を替える仕様を、グリルで決めた（出典: [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md)、§5.1 D4〜D7）。

## 選ばなかった形

- 低解像度動画をNVENC用にpaddingする形 — 先送り: 出力寸法と毎frame処理の複雑化に見合う実需要がない。再開条件: NVENCを強制維持する実需要が出たとき（出典: [PRD025](../../memo/done/025_videolog_codec_auto_ffmpeg_guard_10prd.md)、ゴール / 非ゴール）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続の観測パイプラインPRDで導入する。（出典: [PRD083](../../memo/done/083_action_context_retire_10prd.md)、複雑さ監査）。
- 累積の統計の部品のbatch方向の集計の式を替えない形と、RNDだけが使う集計の口を足す形 — 不採用: 替えないと計算deviceがCPUの構成に遅い行が残り、口を分けても、その行を通るのは出荷の設定ではRNDだけで経路が二重になる（出典: [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md)、§5.1の退けた案）。
- 累積の統計も計算deviceに置く形 — 不採用: 得が合流と転送の0.083 msにとどまり、診断値を読むたびの同期と、推定器の写しの作りに触ることになる（出典: [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md)、§5.1 D8）。
- 統計つきの正規化に、平均と標準偏差の写しを持たせない形 — 不採用: 別モデルの検討が推したが、開発者が、写しを作る時点と置き場の決まりを書いて入れると選んだ（出典: [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md)、§5.1 D5・退けた案）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- H2DとLearner streamを安全につなぐeventのrecord/waitを維持し、性能改善のために消さない（出典: [PRD020](../../memo/done/020_device_transfer_common_part_10prd.md)、Context / Problem Statement）。
- 観測のstack→device転送→正規化の並び、乱数の派生・消費順、aux集合を変えない。同seed・config・exp予算のDefaultDQN/LunarLander/stack4でmetrics checksumを比べ、checkpointのraw checksumはゲートにしない（出典: [PRD083](../../memo/done/083_action_context_retire_10prd.md)、Context・確定した設計判断2〜4・11）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md)、§4.1 D3・§8 1）。
- 累積の統計の部品の集計の式を変えるときは、全要素の集計の行（報酬スケーラが通る）を動かさず、新しい経路とCPU・倍精度の参照の計算の差を単体テストで確かめる（観測の統計の平均と標準偏差が相対1e-4以内）（出典: [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md)、§5.1 D7、§7の5）。

## 関係

- 設計: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.2・§2.3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [image.hpp](../../../src/core/base/include/anet/image.hpp)
- [scaler.hpp](../../../src/core/base/include/anet/scaler.hpp)
- [stacker.hpp](../../../src/core/base/include/anet/stacker.hpp)
- [tensor_check.hpp](../../../src/core/base/include/anet/tensor_check.hpp)
- [tensor_store.hpp](../../../src/core/base/include/anet/tensor_store.hpp)
- [tensor_util.hpp](../../../src/core/base/include/anet/tensor_util.hpp)
- [transfer.hpp](../../../src/core/base/include/anet/transfer.hpp)
- [image.cpp](../../../src/core/base/src/image.cpp) / [image_test.cpp](../../../src/core/base/src/image_test.cpp)
- [scaler_test.cpp](../../../src/core/base/src/scaler_test.cpp)
- [scaler_impl.cpp](../../../src/core/base/src/scaler_impl.cpp) / [scaler_impl.hpp](../../../src/core/base/src/scaler_impl.hpp)
- [stacker.cpp](../../../src/core/base/src/stacker.cpp) / [stacker_test.cpp](../../../src/core/base/src/stacker_test.cpp)
- [tensor_check.cpp](../../../src/core/base/src/tensor_check.cpp)
- [tensor_store.cpp](../../../src/core/base/src/tensor_store.cpp) / [tensor_store_test.cpp](../../../src/core/base/src/tensor_store_test.cpp)
- [tensor_util.cpp](../../../src/core/base/src/tensor_util.cpp)
- [transfer.cpp](../../../src/core/base/src/transfer.cpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD069](../../memo/done/069_metrics_trace_channel_10prd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [PRD070](../../memo/done/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD083](../../memo/done/083_action_context_retire_10prd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [PRD094](../../memo/done/094_evalpanel_step_history_10prd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [PRD116](../../memo/116_rnd_lifelong_novelty_10prd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD
- [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) — PRD117: WARN の意味の線引きと、読まれない設定の検出 PRD
- [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） PRD
- [PRD122](../../memo/122_small_refactor_and_test_gap_fixes_10prd.md) — PRD122: 小さなリファクタリングとテストの空白をまとめて片づける
- [PRD127](../../memo/127_eval_trace_intrinsic_reward_columns_10prd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）PRD
- [PRD130](../../memo/130_rnd_estimate_statistics_perf_10prd.md) — PRD130: RND の採点の統計の区間を軽くし、2 つの統計方式を型で分ける PRD
- [PRD182](../../memo/182_source_layout_10prd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）PRD
- [PRD191](../../memo/191_rnd_raw_error_and_recent_learning_10prd.md) — PRD191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） PRD
- [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md) — PRD194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-9df5c5](../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | PRD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [PRD146](../../memo/146_notifier_scoped_detach_10prd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 未着手 | 2.0（引継ぎ） | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [PRD161](../../memo/161_observation_pipeline_10prd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| [issue-20261111-5f695d](../../management/issues/issue-20261111-5f695d.md) | 未着手 | 2.7 | — | コード | 監査 | 314_tensor | 基盤 | リファクタリング | — | 未確認 | 報酬の scaler と観測の normalizer で、clip と変換の同じ数行が 4〜6 か所にあり、clip が無効のときの扱いが既に食い違っている |
| [issue-20261113-b96b2c](../../management/issues/issue-20261113-b96b2c.md) | 未着手 | 2.3 | — | コード | PRD | 324_dqn, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | 観測の正規化の統計が checkpoint に保存されず、統計を使う構成では checkpoint 引き継ぎのたびに 0 から始まる |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD169](../../memo/169_outside_core_extension_layout_10prd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261116-cc5942](../../management/issues/issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261122-d9499f](../../management/issues/issue-20261122-d9499f.md) | 未着手 | 1.0 | — | コード | PRD | 316_config, 324_dqn, 314_tensor, 315_nn | 基盤 | 機能改善 | — | 未確認 | 見直したあとの設定の形と通知の決まりに沿わない設定キーと通知が、RND の外に残る（全体は調べていない） |
| [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | PRD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261122-4c0af5](../../management/issues/issue-20261122-4c0af5.md) | 未着手 | 0.8 | — | コード | PRD | 318_rl_core, 314_tensor, 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | 観測をためる仕組み（フレーム積層の履歴、Replay の観測の置き場、補助学習の rollout の置き場）が別々に実装されていて、共通の置き場の部品へ載せ替えていない |

- 関連する課題: [issue-20261119-4976bf](../../management/issues/issue-20261119-4976bf.md) — RNDのpredictorの補助学習が今のintakeの観測だけから引き、lane数を超えるバッチと同じ標本の再利用（Burdaらの4エポック）ができない
着手できる 10 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
