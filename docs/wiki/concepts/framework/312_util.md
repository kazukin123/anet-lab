---
title: 312 ユーティリティ
page_kind: フレームワーク
aliases:
  - 312_util
  - warn_once
description: 設定からRunを組み立てる実行基盤のうち、初期化、seed、schedule、log、application共通処理の所在を引く。
updated: 2026-11-28
---
# 312 ユーティリティ

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [フレームワーク](../../index.md#フレームワーク)

## 概要・責務

- 設定からRunを組み立てる実行基盤のうち、初期化、seed、schedule、log、application共通処理の所在を引く。（出典: [実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md)、§2・§3・§7）。
- ProfiledValueは、定数をrootのbare値で、時間で変わる値を`type`（`linear`・`cosine`・`cosine_restart`・`exponential`・`phased`）と対応する欄で表す。読み取りと記録を分け、非constantの配下に継承されたbare値は保持しても評価とdumpに使わず、dumpには実効typeが参照する欄だけを記録する（出典: [実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md)、§2.1「ProfiledValue の定数と時間の値」）。

## 成立の経緯

- 2026-06-15: 学習の同 seed 再現性のため `setDeterministicAlgorithms` をグローバル既定で有効化する（出典: [ADR 0006](../../../adr/0006-deterministic-algorithms.md)、本文）。
- 2026-07-16: Train Actorの周期同期に共通scheduleを接続した（出典: [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md)、題・目的）。
- 2026-07-20: batch入力の移行で共通の定義・接続を更新した（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-17: RFD0053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、題・目的）。
- 2026-08-17: 飛び飛びのカウンタに対する interval を bucket-crossing で定義する（出典: [ADR 0028](../../../adr/0028-interval-fires-on-bucket-crossing.md)、本文）。
- 2026-08-21: AMP対応で共通初期化の接続を更新した（出典: [RFD0033](../../../memo/done/0033_imagecls_bf16_head_10rfd.md)、Context（背景・目的））。
- 2026-08-21: Envインスタンスname RFD（出典: [RFD0037](../../../memo/done/0037_env_instance_name_10rfd.md)、題・目的）。
- 2026-08-21: Envログ name prefix 自動付与 RFD（出典: [RFD0038](../../../memo/done/0038_env_log_auto_prefix_10rfd.md)、題・目的）。
- 2026-08-21: RFD0045: EmaFilter バイアス補正ウォームアップ（出典: [RFD0045](../../../memo/done/0045_ema_debias_warmup_10rfd.md)、題・目的）。
- 2026-08-21: ProfiledValue による ImageCls learning_rate 制御（出典: [RFD0031](../../../memo/done/0031_imagecls_cosine_lr_10rfd.md)、題・目的）。
- 2026-08-21: TBO診断に関わるlog基盤を更新した（出典: [RFD0008](<../../../memo/done/0008_Transformed Bellman Operator_10rfd.md>)、3. 報酬スケーラとの共存と warn）。
- 2026-08-21: TensorDict可視化の共通定義を更新した（出典: [RFD0010](../../../memo/done/0010_image_metrics_v2_10rfd.md)、Phase 3-4 — network sweep 基盤 + SweepStateHeatMap observer / UI 復旧）。
- 2026-08-21: `backend.deterministic_algorithms` 導入（学習の同 seed 再現性確保）仕様書（出典: [RFD0015](../../../memo/done/0015_deterministic_algorithms_10rfd.md)、題・目的）。
- 2026-08-21: forward traceで使う共通の定義を更新した（出典: [RFD0006](../../../memo/done/0006_conv2d_visual_trace_10rfd.md)、5.1 【新規】`TraceSink` 型（当時の名前。今のコードに無い））。
- 2026-08-27: RFD0068: batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）（出典: [RFD0068](../../../memo/done/0068_batchrun_fatal_error_handling_10rfd.md)、題・目的）。
- 2026-09-04: 評価episode集約で使う共通utilityを更新した（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、F1. metrics 経路は既に「1 セッション 1 点」であり、停止条件だけ変えても N 本平均は録れない）。
- 2026-09-24: InitRLの初期化に明示deviceの可用性検査を加えた（出典: [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md)、確定したD10・Complexity audit）。
- 2026-10-25: ProfiledValueに指数とphaseの終値規則を加え、定数をbareで読み、実効typeが使う欄だけをdumpする（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.1 D2〜D4）。
- 2026-10-26: 既存の自動seed入口を使い、GridMazeの自動採番をRun seedから独立させた（出典: [RFD0102](../../../memo/done/0102_gridmaze_seed_mode_10rfd.md)、§2・§5.2）。
- 2026-10-29: 整数のProfiledValueをActorのスキップ長上限へ接続した（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§4 D1・D2・§5.1）。
- 2026-11-11: InitRLのdeviceの検査を、設定に書かれているキーだけを見る形にし、持ち主の設定クラスと二重に持っていた既定値の表を消した（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D9）。
- 2026-11-12: 通知を出すソースの箇所と対象の名前の組ごとにRunで1回だけ通す共通の部品（`LOG::warn_once`・`LOG::info_once`）を足し、Runを作る側が部品の構築より前に通知の記録を空にするようにした。明示した設定が環境変数に勝ったときの通知はINFOにした（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§5.1・§5.5）。
- 2026-11-17: forwardの途中のTensorを押し出すTraceCallback型を消した（出典: [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md)、§4 D9）。
- 2026-11-19: ユーティリティのファイルを基盤のライブラリ（`src/core/base`、ターゲット`anet-core-base`）へ移した（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D3・D4）。
- 2026-11-20: `InitRL`は呼ぶたびにbackendの設定を当て直す関数のまま残し、中で呼ぶNNの初期化関数の登録だけを、何度呼んでも1回にした（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D11・§6.5）。
- 2026-11-21: device指定の文法の誤りのfatalを1回にするため、ParseDeviceがキーの名前を必須で受け取ってInitRLが例外を捕まえない形にし、明示のcudaが使えないときのエラーにキーごとの指定値を並べた（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D8・D9・§6.3）。
- 2026-11-22: TensorSpec に、値の尺度と stack 軸の 2 つの属性を持たせる（出典: [ADR 0085](../../../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md)、本文）。
- 2026-11-23: `TensorSpec`に、値の尺度（`scale`）と、stack軸を持つか（`has_stack_axis`）の属性を足し、specのJSONに`scale`と`stack_axis`の欄を出すようにした（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.3 D17・D19）。
- 2026-11-23: テストのタグで使う文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にそろえた（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D12・§9.6）
- 2026-11-25: `InitRL`が、NNの標準のmoduleと同じ所で、Forwarderの標準の部品を登録するようにした（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.1 D6）。

## 選ばなかった形

- GridMaze の入力 seed を Run seed から派生させる案は不採用。既存の自動採番とは異なる契約になり、盤面の再現は採用 seed の指定で足りる（出典: [RFD0102](../../../memo/done/0102_gridmaze_seed_mode_10rfd.md)、§8）。

- SDPA の backend を math に固定する形 — 不採用（棄却）: 決定化はできるが math は遅く、mem-efficient の決定的 backward で同じ再現性が得られるので、backend を局所的に縛る複雑さと版差の脆さを払う価値がない（出典: [ADR 0006](../../../adr/0006-deterministic-algorithms.md)、Considered Options）。
- SDPA 化を撤回して MultiheadAttention の経路へ戻す形 — 不採用（棄却。本 ADR の範囲外）: SDPA に実時間の利点は出なかったが、非決定は他の atomic 系 op でも起こり得るので本質解でなく、差分・互換のコストも大きい（出典: [ADR 0006](../../../adr/0006-deterministic-algorithms.md)、Considered Options）。
- EmaFilter のウォームアップを累積平均から固定 α へ切り替える案（案 B） — 不採用（却下）: 序盤にトレンドする学習メトリクスでは一様平均が古い値を引きずってラグが出、切替点に折れ目ができ、設定項目ゼロの案 A に劣る（出典: [RFD0045](../../../memo/done/0045_ema_debias_warmup_10rfd.md)、5. 代替案と却下理由）。
- 値をバッファに溜めてウォームアップ後に遡及出力する案（案 C） — 不採用（却下）: MetricsLogger に過去 step への遡及出力 API が要って責務が漏れ、SQLite キャッシュ（PK=序数）と非互換で、グラフ冒頭が同じ値のフラット線になり、ライブビューも遅れる（出典: [RFD0045](../../../memo/done/0045_ema_debias_warmup_10rfd.md)、5. 代替案と却下理由）。
- 分散からウォームアップの終わりを自動判定する案（案 E） — 不採用（却下）: 平滑と収束判定の 2 関心を 1 機構に混ぜ、案 A では判定自体が要らない（出典: [RFD0045](../../../memo/done/0045_ema_debias_warmup_10rfd.md)、5. 代替案と却下理由）。
- thread-local logging context・structured logging・JSON logging — 範囲外（対象外）: Env ログの name prefix を扱う RFD の対象外に置かれた（出典: [RFD0038](../../../memo/done/0038_env_log_auto_prefix_10rfd.md)、対象外）。
- Logger の debug メンバ関数 — 不採用（追加しない）: debug ログの価値（guard による整形コストゼロ、ファイル名・行番号・関数名の自動付与、ビルドでの消去）はマクロでしか保てない（出典: [RFD0038](../../../memo/done/0038_env_log_auto_prefix_10rfd.md)、解決方針・対象外）。
- InitRLのdeviceの検査が、既定値を持つ設定クラスの定数を引く形 — 不採用: Runの設定の型をtrainer.cppの外へ出す変更が要り、書かれていないキーを飛ばせば今の結果のまま二重持ちが消える（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D9・§7）。
- 整形した文面をキーにして通知を 1 回だけ出す形 — 不採用: step や件数のように変わる値が文面に入ると毎回出て、抑えるときも毎回文面を組み立てる。通知を出す箇所と対象の名前の組をキーにする（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§5.1・§8）。
- WARN と ERROR の間に新しいログのレベルを足す形 — 不採用: レベルを増やすと「WARN は利用者が直すべき状態」という線引きがぼやける（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§3.2・§8）。
- 文脈を足して投げ直す箇所のfatalの二重出力を、ログを重ねない投げ直しの口を作って直す形 — 先送り: 別の関心（fatalの出し方）で、設定キーの名指しとは分けられる。再開条件: RFD0071を再グリルするとき（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5.1）。
- 初期化関数を1回しか呼べない形にして、テストを直す形 — 不採用: `InitRL`が中でNNの初期化関数を呼ぶので`InitRL`を2回呼ぶだけで止まり、NNの初期化関数を外へ出す変更と、直接の呼び出し46か所の書き換えが要る（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- `InitRL`も何度呼んでも1回だけにする形 — 不採用: 設定を変えて呼び直すテストが、2回目から設定を当てられなくなる（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§6.5）。
- `TensorSpec`の値の種類を4つ（生の量・尺度つきの量・番号・符号）に分ける形 — 不採用（却下）: その区別を使う箇所が無い（出典: [ADR 0085](../../../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md)、Considered Options）。
- `TensorSpec`に加工ごとのフラグ（正規化しない、重ね済み）を持つ形 — 不採用（却下）: specが加工の名前を知り、加工が増えるたびに欄が増える（出典: [ADR 0085](../../../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md)、Considered Options）。
- 正規化から外すキーを、設定のキーの一覧だけで決める形 — 不採用（却下）: 実装が足すキーを設定に書き忘れると対象になる（出典: [ADR 0085](../../../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md)、Considered Options）。
- 複数の区間を跨いだとき、遅れた分を同じイベントで繰り返し発火する — 不採用: 途中時点の状態は残っておらず、同じ値を複数回記録するだけになる（出典: [ADR 0028](../../../adr/0028-interval-fires-on-bucket-crossing.md)、Considered Options）。
- テストのタグや検証資材だけ旧い呼び名を残す形 — 不採用: 現行のテストと参照手順に旧名が残り、同じ変更の参照が分かれる（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D9・D12）

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- EMA内部の状態を変えても公開シグネチャを変更しない（出典: [RFD0045](../../../memo/done/0045_ema_debias_warmup_10rfd.md)、3. API 契約）。
- 同 seed の再現を確かめるときは `backend.deterministic_algorithms` を true、`backend.deterministic_warn_only` を false にする。warn_only を true にすると決定版の無い op を素通りさせるので再現性は保証されない（出典: [ADR 0006](../../../adr/0006-deterministic-algorithms.md)、Consequences；[RFD0015](../../../memo/done/0015_deterministic_algorithms_10rfd.md)、1.1 API）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、RFD0103の検証資材でmetrics checksumを一致させる（出典: [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md)、§5.7・§7の受入1）。
- 1回だけ通す通知の部品を変えても、ログのレベルの名前、Runのログの`[W]`・`[I]`の書式、Logs paneの数え方は変えない。部品のテストで、同じ箇所で対象の名前が違えば名前ごとに1件、同じ箇所と対象なら何回呼んでも1件、記録を空にしてRunを作り直すとまた1件出ることを確かめる（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§5.1・§7 A1）。
- 小さなリファクタリングとテストの追加の前後で、RFD0098の検証資材の短いRun（4環境、seed 1、決定的モード）を流し、時間と資源のtagを除く全scalarの点を一致させる（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D8・§6 A7）。
- ファイルの置き場やライブラリを移すときは、着手前に保存した実行体と、RFD0128 §6.4の前後比較の組（seed 1本、決定的なbackend）で、全scalarの点、設定の記録、ログのWARNとERRORの行の一致を確かめ、テストのケースの数を前後でそろえる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D24・§6 C3・S3-6）。
- InitRLのdeviceの検査を変えるときは、明示したcudaの一括の報告を保つ。検査は設定に書かれたキーだけを見るので、持ち主のクラスの既定値に明示のcudaを置くときは検査を見直す。`[rfd0070]`のテストはCUDAが使える機械ではskipになるので、環境変数`CUDA_VISIBLE_DEVICES`を`-1`にして流し、skipを合格に数えない（出典: [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§2.1・§6 A6）。
- Env 名のログ接頭辞を変えても、ログレベルと warning 以上で即時 flush する動作を保つ（出典: [RFD0038](../../../memo/done/0038_env_log_auto_prefix_10rfd.md)、受入条件）。
- 間隔をまたぐ判定へ移しても、従来の剰余判定が記録した step 0 の最初の点を保つ。評価間隔の変化が学習に波及していないことは同じ seed の学習曲線で確かめる（出典: [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md)、§0・§4）。
- ProfiledValue の指数スケジュールを変えるときは、最上位では指定した終値へ漸近し続け、phased の最終区間ではその区間の最終 step で計算した値を以後も保つという境界の違いを確かめる（出典: [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md)、§4.1 D2・§6 の単体テスト）。
- 基盤の初期化で Forwarder の部品を登録する箇所を動かすときは、同じ `type` の二重登録、登録されていない `type` の設定、依頼の種類を扱えない部品を含む構成が、それぞれ登録時または構築時のエラーになることを確かめる（出典: [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.1 D6・§5 D20）。
- テストのタグを改名する場合は、タグを指定する手順も追随させ、既定のテストを途中の変更でも実行できることを確かめる（出典: [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md)、§6 D4・§9.6）

## 外の事例

記述なし

## 関係

- [実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md)、§3。
- 設計: [実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md)、§2・§3・§7。
- ユーザーガイド: [Run実行ガイド](../../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [app_util.hpp](../../../../src/core/base/include/anet/app_util.hpp)
- [common.hpp](../../../../src/core/base/include/anet/common.hpp)
- [diag.hpp](../../../../src/core/base/include/anet/diag.hpp)
- [exception.hpp](../../../../src/core/base/include/anet/exception.hpp)
- [init.hpp](../../../../src/core/base/include/anet/init.hpp)
- [json_util.hpp](../../../../src/core/base/include/anet/json_util.hpp)
- [log.hpp](../../../../src/core/base/include/anet/log.hpp)
- [random.hpp](../../../../src/core/base/include/anet/random.hpp)
- [schedule.hpp](../../../../src/core/base/include/anet/schedule.hpp)
- [str_util.hpp](../../../../src/core/base/include/anet/str_util.hpp)
- [util.hpp](../../../../src/core/base/include/anet/test/util.hpp)
- [thread.hpp](../../../../src/core/base/include/anet/thread.hpp)
- [util.hpp](../../../../src/core/base/include/anet/util.hpp)
- [app_util.cpp](../../../../src/core/base/src/app_util.cpp) / [app_util_test.cpp](../../../../src/core/base/src/app_util_test.cpp)
- [app_util_impl.hpp](../../../../src/core/base/src/app_util_impl.hpp)
- [common.cpp](../../../../src/core/base/src/common.cpp)
- [diag.cpp](../../../../src/core/base/src/diag.cpp)
- [exception.cpp](../../../../src/core/base/src/exception.cpp) / [exception_test.cpp](../../../../src/core/base/src/exception_test.cpp)
- [init.cpp](../../../../src/core/base/src/init.cpp) / [init_test.cpp](../../../../src/core/base/src/init_test.cpp)
- [json_util.cpp](../../../../src/core/base/src/json_util.cpp)
- [log.cpp](../../../../src/core/base/src/log.cpp) / [log_test.cpp](../../../../src/core/base/src/log_test.cpp)
- [random.cpp](../../../../src/core/base/src/random.cpp)
- [schedule.cpp](../../../../src/core/base/src/schedule.cpp) / [schedule_test.cpp](../../../../src/core/base/src/schedule_test.cpp)
- [str_util.cpp](../../../../src/core/base/src/str_util.cpp)
- [thread.cpp](../../../../src/core/base/src/thread.cpp) / [thread_test.cpp](../../../../src/core/base/src/thread_test.cpp)
- [util.cpp](../../../../src/core/base/src/util.cpp) / [util_test.cpp](../../../../src/core/base/src/util_test.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0036](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [RFD0053](../../../memo/done/0053_interval_gate_perf_ema_10rfd.md) — PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化
- [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [RFD0070](../../../memo/done/0070_device_spec_auto_10rfd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md) — GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD
- [RFD0101](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [RFD0104](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md) — TempoRL の補助学習のみモード（毎 step 判断し、スキップの出力は補助タスクとしてだけ学ぶ）PRD
- [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0122](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD
- [RFD0170](../../../memo/0170_nn_trace_capture_unification_10rfd.md) — RFD0170: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る RFD
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD
- [RFD0200](../../../memo/0200_forwarder_public_interface_rework_10rfd.md) — forwarder.hpp の公開の口の作り直し（作る側と作られるものを分け、spec を依頼ごとに渡す） RFD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-c8fb06](../../../management/issues/issue-20261103-c8fb06.md) | 未着手 | 3.0 | — | コード | RFD | 312_util, 110_runner_app | 基盤 | 機能改善 | [RFD0071](../../../memo/0071_uncaught_fatal_visibility_10rfd.md) | 残件有り | RFD0071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| [issue-20261103-5addea](../../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-785e68](../../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-df24e5](../../../management/issues/issue-20261103-df24e5.md) | 未着手 | 1.4 | — | コード | 監査 | 312_util | 基盤 | 不具合 | — | 残件有り | AnetExceptionのstack_trace()がwxStackWalkerに頼っていて、Runnerのプロセスでは空になる |
| [issue-20261103-c16aab](../../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | コード | RFD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [RFD0154](../../../memo/0154_scalar_threshold_rate_10rfd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261111-d1d12b](../../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |

着手できる 6 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
