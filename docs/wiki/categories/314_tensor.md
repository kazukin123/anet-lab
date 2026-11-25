# 314 テンソル処理

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- device転送とframe stackを担い、sampleのTensorとCPUに残すmetadata、非同期転送の寿命を扱う。（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.2・§2.3）。
- `LaneTimeTensorStore`は、lane一式のtensorを時間順に保持し、laneと時間を指定して読む。RNDのrolloutの補助学習に使う（出典: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§3）。
- `Forwarder` は TensorDict を受け渡す共通の口で、Network も実装する。`ForwarderBuilder` は依頼と入力 spec を受けて、入出力 spec を持つ `SpecForwarder` を作る。部品の種類は公開の登録口で追加できる（出典: [Forwarder と Perception](../../design/135_forwarder_and_perception.jp.md)、§1・§5）。

## 成立の経緯

- 2026-06-09: 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する（出典: [ADR 0002](../../adr/0002-tensordict-function-unify.md)、本文）。
- 2026-06-15: ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）（出典: [ADR 0005](../../adr/0005-sample-prefetch-stale-per.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 RFD（出典: [RFD0025](../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md)、題・目的）。
- 2026-08-21: Conv2d 可視化「実Forwardトレース化」実装指示書（出典: [RFD0006](../../memo/done/0006_conv2d_visual_trace_10rfd.md)、題・目的）。
- 2026-08-21: Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 RFD（出典: [RFD0020](../../memo/done/0020_device_transfer_common_part_10rfd.md)、題・目的）。
- 2026-08-21: ReplayBuffer PER priority 転送境界整理 RFD（出典: [RFD0021](../../memo/done/0021_replay_per_priority_transfer_10rfd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [RFD0010](../../memo/done/0010_image_metrics_v2_10rfd.md)、題・目的）。
- 2026-09-05: trace取得時のscalerの読み口と回帰を確認した（出典: [RFD0069](../../memo/done/0069_metrics_trace_channel_10rfd.md)、題・目的）。
- 2026-09-19: Actorごとのframe stackerの構築・同期を接続した（出典: [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-09-24: device文字列を解決するParseDeviceをtensor_utilにまとめた（出典: [RFD0070](../../memo/done/0070_device_spec_auto_10rfd.md)、確定したD8）。
- 2026-09-27: ActionContext廃止に合わせstackerをActor私有としstacker側のAgent依存を除いた（出典: [RFD0083](../../memo/done/0083_action_context_retire_10rfd.md)、確定した設計判断5・7）。
- 2026-10-21: scalarの読み口統一に合わせScaler群の値取得を接続した（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.1 D1・§4.3 規約・ADR・コードから導いたこと）。
- 2026-11-11: 報酬スケーラの`use_auto_post_scale`、観測の正規化のrobust updateとSymLogの後処理という、既定のプロファイルで動く経路の計算をテストで固定した（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§1.1・§4 D3）。
- 2026-11-12: RNDのwhitenの統計に使うため、観測の正規化の部品（`ObservationNormalizer`）をcheckpointへ保存できるようにし、正規化の設定の値域を呼び出し側のキーで検証する関数を足した（出典: [RFD0116](../../memo/0116_rnd_lifelong_novelty_10rfd.md)、§4 D5・D12・§5.6）。
- 2026-11-12: 画像の記録の設定（`metrics.image.[<tag>]`）の読み口が、項目の種類で使わない既知のフィールドも構築時に型付きで読む閉じたスキーマになり、同じtagの二重登録と観測の正規化の未知の後処理の種類を、WARNで流さずに止めるようにした（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D7・§5.3・§5.5）。
- 2026-11-15: 報酬スケーラに、lane ごとの割引収益の標準偏差で割る種類（`dynamic_mode = return_std`）を足し、動的な種類の状態を checkpoint に保存するようにした（出典: [RFD0121](../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、§5 D9・D16・§6.6）。
- 2026-11-16: 観測の正規化の部品に統計だけを更新する口を足し、累積の統計の部品のbatch方向の集計の式を替えた（出典: [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D4〜D7）。
- 2026-11-19: テンソル処理のファイルを基盤のライブラリ（`src/core/base`、ターゲット`anet-core-base`）へ移した（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D3・D4）。
- 2026-11-21: `ParseDevice`がキーの名前を必須で受け取り、文法の誤りの文面にそのキーと指定値を書くようにし、報酬スケーラ・観測の正規化・画像の記録のエラーとWARNが示す設定キーを完全キーにした（出典: [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D2・D8・§6.3）。
- 2026-11-22: RNDのpredictorの補助学習のrolloutをためるため、lane×時間のtensorをキーごとにためてlaneと時間の番号で読む汎用の置き場（`LaneTimeTensorStore`、機能グループ`tensor_store`）を足す仕様を決めた。フレーム積層の履歴とReplayの観測の置き場は載せ替えない（出典: [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.1 D2・D56・§8.4）。
- 2026-11-23: Perception の FrameStack・Normalize・DropKeys を Forwarder の並びにまとめ、Frame の尺度と stack 軸を spec で表す。対象キーは 1 枚でも stack 軸を持つ（出典: [Forwarder と Perception](../../design/135_forwarder_and_perception.jp.md)、§2〜§4）。
- 2026-11-24: Forwarder の公開の口を、依頼ごとに Forwarder を作る形へ作り直す仕様を決めた。実行するものを `Forwarder` の 1 つにし、並び（`ForwarderBuilder`）と部品は Forwarder を作る側に限り、spec は依頼ごとに渡し、部品の種類は登録制にする（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.1 D1〜D10）。
- 2026-11-24: 依頼ごとの spec と部品登録を実装した。`SetSpec` は宣言した尺度・範囲・ラベルを変更する。4 環境の変更前後で scalar と trace が一致した（出典: [RFD0200 実装メモ](../../memo/0200_forwarder_public_interface_rework_20impl.md)、主な変更・4 環境の等価性）。
- 2026-11-24: Forwarder は依頼ごとに作り、作る側（並びと部品）は Forwarder を兼ねない（出典: [ADR 0094](../../adr/0094-forwarders-are-built-per-request-and-builders-are-not-forwarders.md)、本文）。
- 2026-11-24: 重ねる枚数とキーの設定は Agent 全体に置き、Sample の束の spec は ReplayBuffer が答える（出典: [ADR 0095](../../adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md)、本文）。
- 2026-11-24: stack 軸の確保と、並びが成り立つことの確かめは、Forwarder の仕組みではなく使う側が受け持つ（出典: [ADR 0096](../../adr/0096-stack-axis-and-chain-validity-are-ensured-by-the-caller-not-the-forwarder-mechanism.md)、本文）。
- 2026-11-25: Forwarder の並びは、全 block の未読キーを確かめ、次に値の型と値域、最後に並びで使える Forward の種類との組み合わせを確かめる。外部の工場と部品にもこの順を適用する（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§6.3・§6.7）。
- 2026-11-25: `LaneTimeTensorStore`に、たまっている範囲の全体を、時刻×laneの形のまま複製せずに読む口（`GetAll`）を足した（出典: [RFD0202](../../memo/0202_ppo_family_and_on_policy_core_parts_10rfd.md)、§7.7）。

## 選ばなかった形

- 低解像度動画をNVENC用にpaddingする形 — 先送り: 出力寸法と毎frame処理の複雑化に見合う実需要がない。再開条件: NVENCを強制維持する実需要が出たとき（出典: [RFD0025](../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md)、ゴール / 非ゴール）。
- ObservationPipelineクラスと専用ファイル・テストの追加 — 範囲外: 固定3段の観測加工はActor本体で満たせ、段の要件が揃う前のクラス化は作り直しになる。再開条件: 段の要件が揃う後続の観測パイプラインRFDで導入する。（出典: [RFD0083](../../memo/done/0083_action_context_retire_10rfd.md)、複雑さ監査）。
- 累積の統計の部品のbatch方向の集計の式を替えない形と、RNDだけが使う集計の口を足す形 — 不採用: 替えないと計算deviceがCPUの構成に遅い行が残り、口を分けても、その行を通るのは出荷の設定ではRNDだけで経路が二重になる（出典: [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1の退けた案）。
- 累積の統計も計算deviceに置く形 — 不採用: 得が合流と転送の0.083 msにとどまり、診断値を読むたびの同期と、推定器の写しの作りに触ることになる（出典: [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D8）。
- 統計つきの正規化に、平均と標準偏差の写しを持たせない形 — 不採用: 別モデルの検討が推したが、開発者が、写しを作る時点と置き場の決まりを書いて入れると選んだ（出典: [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D5・退けた案）。
- 報酬スケーラのWARNと、DefaultDQNAgentの経路での後処理の種類の検証を、テストの空白の片づけで直す形 — 範囲外: RFD0117の範囲（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§7）。→ RFD0117 で採用。未知の後処理の種類を構築時の検証で止める（出典: [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§5.5）。
- 汎用の置き場を作らず、RNDの中だけでintakeごとの観測を並べて持つ形 — 不採用: ためる仕組みを汎用にして共通化する開発者の指示による。部品と契約と受入は要らなくなるが、あとで載せ替える先が無い（出典: [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.4）。
- フレーム積層の履歴やReplayの観測の置き場も、同じRFDで新しい置き場の部品に載せ替える形 — 範囲外: 全RunのActorの毎step、Pushと抽出に触る。課題に残した（出典: [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§6.4・§1.4）。
- 置き場の部品を`tensor_util`か`stacker`の機能グループに入れる形 — 不採用: `tensor_util`に状態を持つ置き場を入れると関心が混ざり、`stacker`に入れると、載せ替えのときにReplayがフレーム積層のヘッダを読む向きになる（出典: [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§8.4）。
- 並びが Forwarder を兼ね、依頼なしで spec を答える形 — 不採用: 責務が 2 つになり、どの依頼の spec かが決まらない（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6）。
- 並びの決まり（`FrameStack` は先頭に 1 つ、`Normalize` は 1 つまで）を残す形 — 不採用: 設定で自由に並べられる仕組みに、並べ方の決まりは要らない（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6）。
- 束の spec や、重ねる枚数とキーを、並びが答える形 — 不採用: 束を作るのは ReplayBuffer である（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6）。
- 重ねる設定を、並びの部品か、ReplayBuffer の設定に置く形 — 不採用: 行動選択と ReplayBuffer の両方が使う設定なので、Agent 全体の 1 か所に置く（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6）。
- 図を、並び（`ForwarderBuilder`）が描く形 — 不採用: 作る側に、作ったものの情報を書かせることになる（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6）。
- 重ねるのを、並びの外へ出す形（呼び出し元が重ねてから、並びに通す） — 不採用: Forward の種類は 2 つで済み、開始の印も並びに入らないが、重ねる位置を、設定の並びで書けなくなる（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6、§5.2 D13）。
- spec の宣言を変える部品（`SetSpec`）から、当てる Forward の種類を絞る口を外す形 — 不採用: 絞ると、行動選択と Sample で正規化する対象がずれる設定が書けるが、宣言を変える手段を先に用意しておく考え方で、開発者が残すと選んだ（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.6、§5.3 D19）。
- Network を `SpecForwarder` にする形 — 先送り: 出力の spec を導くには、Head まで仮の forward を通すことになる。再開条件: 出力の spec を使う側が出たとき（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§2 の NG5）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- H2DとLearner streamを安全につなぐeventのrecord/waitを維持し、性能改善のために消さない（出典: [RFD0020](../../memo/done/0020_device_transfer_common_part_10rfd.md)、Context / Problem Statement）。
- 観測加工の変更では、Actor と Sample の Perception を episode 境界でも一致させる。GridMaze・LunarLander・DropMerge・Atari の全4環境で、変更前後とも同じ seed と `backend.$=backend.@deterministic` を指定する。両側の `backend.deterministic_algorithms` と `backend.cudnn_deterministic` がともに `true` であることを確認し、学習 Run の scalar と trace を照合する（出典: [RFD0161](../../memo/0161_observation_pipeline_10rfd.md)、D34・D35・§9）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.1 D3・§8 1）。
- 小さなリファクタリングとテストの追加の前後で、RFD0098の検証資材の短いRun（4環境、seed 1、決定的モード）を流し、時間と資源のtagを除く全scalarの点を一致させる（出典: [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md)、§4 D8・§6 A7）。
- ファイルの置き場やライブラリを移すときは、着手前に保存した実行体と、RFD0128 §6.4の前後比較の組（seed 1本、決定的なbackend）で、全scalarの点、設定の記録、ログのWARNとERRORの行の一致を確かめ、テストのケースの数を前後でそろえる（出典: [RFD0182](../../memo/0182_source_layout_10rfd.md)、§4.1 D24・§6 C3・S3-6）。
- `LaneTimeTensorStore`に、上限に達したら古い時間から上書きする書き方を足すときは、作るときに選ぶ形にし、上限を超えたらERRORにする今の契約を選んだ使い手の動きを変えない（出典: [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md)、§8.5）。
- 累積の統計の部品の集計の式を変えるときは、全要素の集計の行（報酬スケーラが通る）を動かさず、新しい経路とCPU・倍精度の参照の計算の差を単体テストで確かめる（観測の統計の平均と標準偏差が相対1e-4以内）（出典: [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md)、§5.1 D7、§7の5）。
- Forwarder の公開の口を変えるときも、4 環境・同じ seed・決定的なバックエンドの Run で、scalar と trace の全レコードの一致を確かめる（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§7 の「等価性の比べ方」）。
- 行動選択・Sample・可視化の 3 つの Perception の出力の spec を、同じに保つ（出典: [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md)、§5.4 D21）。

## 関係

- 設計: [ReplayBuffer](../../design/150_replay_buffer.jp.md)、§2.2・§2.3・§3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [forwarder.hpp](../../../src/core/base/include/anet/forwarder.hpp)
- [image.hpp](../../../src/core/base/include/anet/image.hpp)
- [scaler.hpp](../../../src/core/base/include/anet/scaler.hpp)
- [stacker.hpp](../../../src/core/base/include/anet/stacker.hpp)
- [tensor_check.hpp](../../../src/core/base/include/anet/tensor_check.hpp)
- [tensor_store.hpp](../../../src/core/base/include/anet/tensor_store.hpp)
- [tensor_util.hpp](../../../src/core/base/include/anet/tensor_util.hpp)
- [transfer.hpp](../../../src/core/base/include/anet/transfer.hpp)
- [forwarder.cpp](../../../src/core/base/src/forwarder.cpp) / [forwarder_test.cpp](../../../src/core/base/src/forwarder_test.cpp)
- [image.cpp](../../../src/core/base/src/image.cpp) / [image_test.cpp](../../../src/core/base/src/image_test.cpp)
- [scaler_test.cpp](../../../src/core/base/src/scaler_test.cpp)
- [scaler_impl.cpp](../../../src/core/base/src/scaler_impl.cpp) / [scaler_impl.hpp](../../../src/core/base/src/scaler_impl.hpp)
- [stacker.cpp](../../../src/core/base/src/stacker.cpp) / [stacker_test.cpp](../../../src/core/base/src/stacker_test.cpp)
- [tensor_check.cpp](../../../src/core/base/src/tensor_check.cpp)
- [tensor_store.cpp](../../../src/core/base/src/tensor_store.cpp) / [tensor_store_test.cpp](../../../src/core/base/src/tensor_store_test.cpp)
- [tensor_util.cpp](../../../src/core/base/src/tensor_util.cpp)
- [transfer.cpp](../../../src/core/base/src/transfer.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0061](../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0069](../../memo/done/0069_metrics_trace_channel_10rfd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [RFD0070](../../memo/done/0070_device_spec_auto_10rfd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [RFD0083](../../memo/done/0083_action_context_retire_10rfd.md) — ActionContext の廃止（Actor が RNG と frame stacker を直接持つ）
- [RFD0094](../../memo/done/0094_evalpanel_step_history_10rfd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [RFD0116](../../memo/0116_rnd_lifelong_novelty_10rfd.md) — RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 RFD
- [RFD0117](../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0121](../../memo/0121_extrinsic_intrinsic_value_decomposition_10rfd.md) — 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） RFD
- [RFD0122](../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) — RFD0122: 小さなリファクタリングとテストの空白をまとめて片づける
- [RFD0127](../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）RFD
- [RFD0130](../../memo/0130_rnd_estimate_statistics_perf_10rfd.md) — RFD0130: RND の採点の統計の区間を軽くし、2 つの統計方式を型で分ける RFD
- [RFD0161](../../memo/0161_observation_pipeline_10rfd.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD
- [RFD0182](../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) — RFD0191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） RFD
- [RFD0194](../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD
- [RFD0200](../../memo/0200_forwarder_public_interface_rework_10rfd.md) — forwarder.hpp の公開の口の作り直し（作る側と作られるものを分け、spec を依頼ごとに渡す） RFD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-9df5c5](../../management/issues/issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [RFD0146](../../memo/0146_notifier_scoped_detach_10rfd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261111-5f695d](../../management/issues/issue-20261111-5f695d.md) | 未着手 | 2.7 | — | コード | 監査 | 314_tensor | 基盤 | リファクタリング | — | 残件有り | 報酬の scaler と観測の normalizer で、clip と変換の同じ数行が 4〜6 か所にあり、clip が無効のときの扱いが既に食い違っている |
| [issue-20261113-b96b2c](../../management/issues/issue-20261113-b96b2c.md) | 未着手 | 2.3 | — | コード | RFD | 324_dqn, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | 観測の正規化の統計が checkpoint に保存されず、統計を使う構成では checkpoint 引き継ぎのたびに 0 から始まる |
| [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261122-258ef8](../../management/issues/issue-20261122-258ef8.md) | 未着手 | 1.2 | — | コード | RFD | 321_agent_core, 324_dqn, 314_tensor | 学習基盤 | 不具合 | — | 残件有り | 統計つきの観測の正規化で、Actor が Agent の排他の外で統計を読み、学習と並んで走る評価の Actor で読み書きが競合する |
| [issue-20261116-cc5942](../../management/issues/issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261122-d9499f](../../management/issues/issue-20261122-d9499f.md) | 未着手 | 1.0 | — | コード | RFD | 316_config, 324_dqn, 314_tensor, 315_nn | 基盤 | 機能改善 | — | 残件有り | 見直したあとの設定の形と通知の決まりに沿わない設定キーと通知が、RND の外に残る（全体は調べていない） |
| [issue-20261103-f40ae8](../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | RFD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [RFD0056](../../memo/0056_timehistogram_fine_grid_10rfd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261122-4c0af5](../../management/issues/issue-20261122-4c0af5.md) | 未着手 | 0.8 | — | コード | RFD | 318_rl_core, 314_tensor, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 観測をためる仕組み（フレーム積層の履歴、Replay の観測の置き場、補助学習の rollout の置き場）が別々に実装されていて、共通の置き場の部品へ載せ替えていない |
| [issue-20261122-ba42c9](../../management/issues/issue-20261122-ba42c9.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | RND の観測の加工（byte を float に直す、自前の統計で正規化する）が、Q の側の Forwarder の並びと別に書かれている |

着手できる 11 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
