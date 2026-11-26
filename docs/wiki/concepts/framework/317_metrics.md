# 317 可視化・メトリクス

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [フレームワーク](../../index.md#フレームワーク)

## 概要・責務

- eventからscalar、trace、画像、動画、GraphVizとartifactへ変換し、値の取得と保存の境界をまとめる。（出典: [可観測性](../../../design/140_observability.jp.md)、§1.1・§2・§3）。
- TimeHeatMapのScale描画・Histgram・SweepedHeatMapは画像自身が画素領域を所有し、生成元を破棄した後も残る画像のコピーが画素を保持する。確保失敗は描画対象と要求寸法を示して停止する（出典: [可観測性](../../../design/140_observability.jp.md)、§7.3）。
- VideoLoggerは最初のframeの幅と高さで作ってRun終了時に閉じ、以降の幅か高さが違う画像と無効な画像は動画から除いて、tagごとにRunで1回WARNする（出典: [可観測性](../../../design/140_observability.jp.md)、§7）。
- 既知のkeyで現在値が成立しない疎なscalarと、休止中や設定で無効にした機能の既知のkeyは`NaN`を、未知のkeyは`nullopt`を返す。Observerは非有限値をEMAと平均へ加えない（出典: [可観測性](../../../design/140_observability.jp.md)、§7.1）。

## 成立の経緯

- 2026-06-09: 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する（出典: [ADR 0002](../../../adr/0002-tensordict-function-unify.md)、本文）。
- 2026-07-01: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 RFD（出典: [RFD0025](../../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md)、題・目的）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-08-21: V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書（出典: [RFD0010](../../../memo/done/0010_image_metrics_v2_10rfd.md)、題・目的）。
- 2026-08-21: anet::nn::Network GraphViz 仕様（出典: [RFD0004](../../../memo/done/0004_nndot_10rfd.md)、題・目的）。
- 2026-09-05: メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、題・目的）。
- 2026-09-05: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する（出典: [ADR 0037](../../../adr/0037-metrics-trace-channel-and-session-end-event.md)、本文）。
- 2026-09-19: Actorカタログ変更後のメトリクス保存を回帰確認した（出典: [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-10-19: Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む（出典: [ADR 0057](../../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、本文）。
- 2026-10-21: scalarとtraceのtarget解決を共用しメトリクス配線を分離した（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.1 D1〜D3）。
- 2026-10-29: 評価sessionの採用episodeのstep traceを、Evaluation Traceと同じ列の書き方で`metrics.step_trace.[<tag>]`に宣言し、traceの行としてRunの記録に残す案を暫定で起票した（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、題・§4 D1・D2・D6）。
- 2026-11-04: 画像メモリの所有権の不一致を、既存の寸法・画素値・画像種別を保って修正した（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§1.2、[可観測性](../../../design/140_observability.jp.md)、§7.3）。
- 2026-11-12: Probeの値が取れない通知を対象ごとにRunで1回にし、動画のcodecの自動選択がlibx264へ切り替えたあとはRunの中で選択を固定して通知を1回にした（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D1・§5.5）。
- 2026-11-15: Evaluation Traceの行を作る部品が、列の宣言から`$agent_step`の鍵を集め、RunnerのActorとあわせて、step診断の部品を作るときに渡すようにした（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D3・§5.3）。
- 2026-11-19: 可視化・メトリクスのファイルを基盤のライブラリ（`src/core/base`、ターゲット`anet-core-base`）へ移した（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D3・D4）。
- 2026-11-20: VideoLoggerが、構築時と違う大きさの画像を動画に書かず、tagごとにRunで1回だけWARNを出して続けるようにした（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D8）。
- 2026-11-21: video_codecのエラーが`app.metrics_logger.video_codec`の完全キーを示すよう、MetricsLoggerの設定の型が完全キーの記録を運ぶようにし、hiddenの設定検査と設計文書の指標名を出荷定義の名前に合わせた（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D1・D11）。

## 選ばなかった形

- HeatMap派生の描画処理の統合と、画像処理全体の確保・入力検証の一括変更 — 範囲外: 画像の4経路の不具合修正とは別の関心。再開条件: 描画の重複整理、または別の具体的な不具合を修正対象に選んだとき（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§1.3）。
- step traceをtraceチャネルへ記録する配線 — 範囲外: 鏡写しインスタンスを購読先にする配線と同じ関心である（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- Event/Observerの型の統一 — 範囲外: step traceとは別の関心。再開条件: 新しいEvent型を足す変更が出たとき（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§6）。
- metricsのtrace定義tagをEvaluation Traceの列として参照する形 — 不採用: 見るだけでも記録の宣言が要り、EMA・間隔・集約を持つscalar統計はstepの生値に合わない（出典: [ADR 0057](../../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、Considered Options）。
- trace の level（off / record / forensic） — 不採用（cut）: 宣言の有無で足りる。forensic は想像上の需要である（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、4.0 決定一覧 D2・9. 複雑さ監査）。
- trace の行の episode_id・model_version の欄 — 範囲外（cut。141・144 へ）: コードに概念が無い。再開条件: episode_id は 144 で EpisodeId が実装されたとき、model_version は 141 に着手するとき（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、8. フェーズとゲート・9. 複雑さ監査）。
- data を宣言順に出す専用の JSON 経路 — 不採用（cut）: 名前と値の対応で目的を満たし、取得順と定義の keys 配列だけを宣言順で保つ（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、9. 複雑さ監査）。
- eval scalar の @episode_end を WARN 付きで受理する alias — 不採用（cut）: リポジトリの外に workspace は無く、クリーンブレークの方針どおり fail-fast のメッセージで置換先を案内する（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、4.0 決定一覧 D5・D11・9. 複雑さ監査）。
- traceの1行DSLに`@train`と複数のtargetを足す形 — 不採用: 1定義1targetの文法と検証（RFD0069）を変えることになり、複数targetの列はEvaluation Traceの列の書き方がすでに表している（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§8）。
- 画面の列（`app.eval_trace_panel.columns.*`）をそのまま記録に使う形 — 不採用: 画面の設定と記録の宣言が結び付く（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§8）。
- 残機を失ったstepだけを書く形 — 不採用: 残機はAtari固有の値で、汎用のtraceのeventにEnv固有の判定を持ち込む（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§8）。
- step traceの行の間引き — 先送り: 評価専用のRunなら行数が許容の範囲。再開条件: 行数が読み手やViewerの扱える量を超えたとき（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§8）。
- 学習側（`$train`）のstep trace — 範囲外: 行数が学習の長さに比例して増える。再開条件: 学習中の行動をstepごとに追う必要が出たとき（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§8）。
- 宣言した scalar の値が取れないことを INFO にする形 — 不採用: 鍵の綴り違いのような本物の誤りを INFO に埋める。休止した機能と、設定で無効にした機能は、鍵を保って NaN を返す（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、D12・§8）。
- step診断を呼ぶたびに列の鍵を渡す形 — 不採用: 列の宣言はRunの間変わらないので、部品を作るときの1回で足りる（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§2.3・§4.1）。
- 知らない`$agent_step`の鍵を、step診断の部品を作るときに検査して起動時に止める形 — 不採用: 知らない鍵の検査は5つのtargetに共通の1か所にあり、`$agent_step`だけを早めると止まる時機がtargetごとに分かれる（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D10・§4.1）。
- 動画の大きさ違いでRunを止める形 — 不採用: 失敗の扱いの表では、構築後にメトリクスの一部だけが欠ける異常はWARNを出して続ける行に当たり、止める形はその行と食い違う。診断用の動画のために学習を止めることにもなる（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D8・§5.1）。
- 大きさが変わったら新しい動画ファイルに切り替える形 — 不採用: ファイル名の決まり、動画の記録の行、読む側の扱いが増え、使われる見込みが薄い（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- 動画の最初の画像が無効なとき、最初の有効な画像が来るまでVideoLoggerを作らない形 — 範囲外: 題材の事象（作ったあとの読み）の外である（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§6.3・§9のU3）。
- 40文字を超えているskipの指標のtagを、40文字に収まる名前へ変える形 — 不採用: 10月29日以後のRunの記録とtagが分かれ、同じグラフに重ならなくなる（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D11・§5.1）。
- 個体の識別子を scalar の step の小数部分へ埋め込む — 不採用: 整数の step、範囲指定、LOD の集約と同じ軸で扱えなくなる（出典: [ADR 0037](../../../adr/0037-metrics-trace-channel-and-session-end-event.md)、Considered Options）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- batch入力への移行でtrainのmax_steps episode、episode_start、可視化cadenceを維持する（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、1.3 維持する外部contract）。
- 512/1024動画はautoでNVENCを選択し、従来どおり出力できることを確かめる（出典: [RFD0025](../../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md)、Testing Decisions）。
- 読み口と配線の整理でmetrics出力・定義レコード・WARN・fail-fastの文言と時機を変えない。同seed・決定的モードの短Runでchecksum・defs・WARNを比較する（出典: [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md)、§4.1 D3・§8 1）。
- step traceの宣言が無い設定では、observerもstep部品も作らず、RFD0103の検証資材の6構成のmetrics checksumを実装の前後で一致させる。既存の1行DSLのtraceの定義の記録（`json/metrics.trace.defs.json`）も前後で同一に保つ（出典: [0184_eval_session_step_trace](../../../memo/0184_eval_session_step_trace_10rfd.md)、§4 D2・D4・§7の受入1）。
- 画像の所有権を直すときは寸法・全画素・画像種別を固定入力で比較し、生成・破棄の不一致を修正前に検出するメモリ検査を修正後にも通す。通常Debugの完走だけで代替せず、環境不足の項目は未達として記録する（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§5 A1〜A3・A7）。
- step診断へ列の鍵の一覧を渡しても、既存の3つの値（`q_sa`・`state_value`・`approx_priority`）の計算と、知らない鍵が最初の読み出しで止まる時機を変えない（出典: [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md)、§4 D10・§5.3）。
- 動画の大きさの確かめを足しても、PNGの個別保存、最初の画像が無効なときの挙動、ffmpegの異常でRunを止める扱いを変えない（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D8・§6.3）。
- ファイルの置き場やライブラリを移すときは、着手前に保存した実行体と、RFD0128 §6.4の前後比較の組（seed 1本、決定的なbackend）で、全scalarの点、設定の記録、ログのWARNとERRORの行の一致を確かめ、テストのケースの数を前後でそろえる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D24・§6 C3・S3-6）。
- scalar の出力行は step 軸の名前を持たない。同じタグに記録する軸を変更すると既存系列と区別できなくなるため、軸の意味を保つ（出典: [可観測性](../../../design/140_observability.jp.md)、§6）。
- 個体 trace の導入前後では、決定的な同 seed、foreground 評価の条件で、時間に依存しない scalar の件数、step、値を突き合わせる（出典: [ADR 0037](../../../adr/0037-metrics-trace-channel-and-session-end-event.md)、Consequences）。
- 観測の一部を probe で選ぶときも既存のパーサを使い、キーを付けていない既存設定の読み方を保つ（出典: [RFD0010](../../../memo/done/0010_image_metrics_v2_10rfd.md)、§2）。
- scalar と trace の定義処理を共通化しても、scalar の既存の上書き順、既定値、診断を維持する。trace の data オブジェクトのキー順には依存しない（出典: [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md)、§6）。

## 外の事例

記述なし

## 関係

- 設計: [可観測性](../../../design/140_observability.jp.md)、§1.1・§2・§3。
- 出力とlifetime、疎なscalar: [可観測性](../../../design/140_observability.jp.md)、§7・§7.1。画像の所有権と検証は同、§7.3・§9。
- ユーザーガイド: [Run実行ガイド](../../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [graphviz.hpp](../../../../src/core/base/include/anet/graphviz.hpp)
- [heat_map.hpp](../../../../src/core/base/include/anet/heat_map.hpp)
- [metrics.hpp](../../../../src/core/base/include/anet/metrics.hpp)
- [metrics_logger.hpp](../../../../src/core/base/include/anet/metrics_logger.hpp)
- [probe.hpp](../../../../src/core/base/include/anet/probe.hpp)
- [graphviz.cpp](../../../../src/core/base/src/graphviz.cpp)
- [heat_map.cpp](../../../../src/core/base/src/heat_map.cpp) / [heat_map_test.cpp](../../../../src/core/base/src/heat_map_test.cpp)
- [metrics.cpp](../../../../src/core/base/src/metrics.cpp) / [metrics_test.cpp](../../../../src/core/base/src/metrics_test.cpp)
- [metrics_impl.hpp](../../../../src/core/base/src/metrics_impl.hpp)
- [metrics_logger.cpp](../../../../src/core/base/src/metrics_logger.cpp) / [metrics_logger_test.cpp](../../../../src/core/base/src/metrics_logger_test.cpp)
- [metrics_logger_impl.hpp](../../../../src/core/base/src/metrics_logger_impl.hpp)
- [probe.cpp](../../../../src/core/base/src/probe.cpp) / [probe_test.cpp](../../../../src/core/base/src/probe_test.cpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0069](../../../memo/done/0069_metrics_trace_channel_10rfd.md) — メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す
- [RFD0094](../../../memo/done/0094_evalpanel_step_history_10rfd.md) — Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
- [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md) — RFD0115: 画像メモリの所有権とDropMergeのNOOP判定の修正
- [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0127](../../../memo/0127_eval_trace_intrinsic_reward_columns_10rfd.md) — Evaluation Trace で RND の内的報酬を見る（統計を変えない採点と step 診断の鍵）RFD
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261103-955e50](../../../management/issues/issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | RFD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [RFD0081](../../../memo/0081_evalpanel_episode_trace_10rfd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261125-195ad5](../../../management/issues/issue-20261125-195ad5.md) | 未着手 | 7.0 | — | 文書 | 監査 | 317_metrics | 学習基盤 | 不整合 | — | 未確認 | 設計140 は trace の行が timestamp を持つと書くが、実装と記録された行は持たず、ADR 0047（全 trace 行に時刻を付ける）は決定のまま実装されていない |
| [issue-20261123-89526b](../../../management/issues/issue-20261123-89526b.md) | 未着手 | 4.0 | — | 文書 | 監査 | 317_metrics | 学習基盤 | 不整合 | — | 残件有り | CONTEXT.md の trace の _Avoid_ が RFD0144（Episode Journal）を記号なしの「144」とだけ書き、RFD0195 の置き換えから漏れている |
| [issue-20261123-3ebab4](../../../management/issues/issue-20261123-3ebab4.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Env の設定のファイル名の衝突検査（RunManager）が、MetricsLogger の private な置換規則と書き出し先の作り方を写して予測している |
| [issue-20261123-446e6f](../../../management/issues/issue-20261123-446e6f.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 不具合 | — | 残件有り | メトリクスの宣言の数値の属性（interval・ema_alpha・clip ほか）を std::stoi・stof で読むので、後ろの文字が黙って捨てられ、数でない値は設定キーの無い例外になる |
| [issue-20261123-ad7829](../../../management/issues/issue-20261123-ad7829.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Runner の counts と scalar の鍵に、一度も進まない sim_step と、reward・reward_ema と同じ値を返す別名 train_reward・train_reward_ema が残る |
| [issue-20261125-29c3b2](../../../management/issues/issue-20261125-29c3b2.md) | 未着手 | 3.5 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | metrics.graph の宣言の文法が設計140 に無く、実装は未知の @ や $ の語を黙って捨て、interval を両方 0 以下にすると黙って 1000 step ごとになる |
| [issue-20261115-e5a0eb](../../../management/issues/issue-20261115-e5a0eb.md) | 未着手 | 3.4 | — | コード | RFD | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 性能改善 | — | 残件有り | 可塑性メトリクスの計算が、500 learn step ごとにメインスレッドで約 2 秒走り、その間は学習が止まる |
| [issue-20261111-be150a](../../../management/issues/issue-20261111-be150a.md) | 未着手 | 3.0 | — | コード | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | StepTraceColumnSpec の target と display を文字列で持ち、metrics.cpp・gui.cpp・EvalTracePanel.cpp の 3 ファイル 17 か所で文字列を比べている |
| [issue-20261123-6ffe02](../../../management/issues/issue-20261123-6ffe02.md) | 未着手 | 3.0 | — | 文書 | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 調査 | — | 残件有り | scalar メトリクスの Observer の 8 引数のコンストラクタと Parsed の struct を、cfe994 と ea3483 の 2 件が別々の直し方で扱っている |
| [issue-20261123-fb50b6](../../../management/issues/issue-20261123-fb50b6.md) | 未着手 | 3.0 | — | 文書 | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 不整合 | — | 残件有り | CONTEXT.md が定義の中で使う step の軸（exp step・train step など）・agent step・lane を、用語として持っていない |
| [issue-20261125-ae4c45](../../../management/issues/issue-20261125-ae4c45.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | 設計140 の章立てと図が今の形に合わない（節番号が「6.x」のまま 2 つ、env_step の節が関連文書の下、図に SessionEnd と trace が無く画像の Observer の種類と作り手が違い、コードマップに metrics・image が無い） |
| [issue-20261125-bdbcac](../../../management/issues/issue-20261125-bdbcac.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | 設計140 が本文で書く仕組みの ADR 8 本（評価セッションの集約・エピソード長・trace の行の時刻・forward の capture・遷移の結果・eval schedule・WARN の表・nn_trace の渡し方）を、どの設計文書もリンクしない |
| [issue-20261125-eb6135](../../../management/issues/issue-20261125-eb6135.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | scalar の宣言の正本である設計140 §6 に、target を省いたときの探索順（出荷設定の 57 件が使う）・属性の既定・ema_alpha だけで EMA になること・$action_info などの event の制限が無く、§7 に動画の設定キーが無い |
| [issue-20261111-645ecc](../../../management/issues/issue-20261111-645ecc.md) | 未着手 | 2.7 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | メトリクス宣言の語（event・step・target）と enum の対応が 5 か所以上に分かれ、解決済みの定義も同じ値を文字列と enum の 2 形で持っている |
| [issue-20261123-6bb71d](../../../management/issues/issue-20261123-6bb71d.md) | 未着手 | 2.5 | — | コード | 監査 | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 不整合 | — | 残件有り | コードの名前・エラー文・メトリクスの tag に CONTEXT.md の避ける語（feature key・mean Q・action churn）が残り、エラー文は実在しない設定キー名 feature_key を示す |
| [issue-20261118-dc5de3](../../../management/issues/issue-20261118-dc5de3.md) | 未着手 | 2.4 | — | コード | その他 | 317_metrics, 110_runner_app | 基盤 | 機能改善 | — | 残件有り | Atari の Run の列ごとにメトリクスの組と tag がずれ、本線・Taiga 寄せ・易しいゲームの Run どうしを Viewer で重ねて比べられない |
| [issue-20261103-e6dbb4](../../../management/issues/issue-20261103-e6dbb4.md) | 未着手 | 2.3 | — | コード | 監査 | 317_metrics | UI | リファクタリング | — | 残件有り | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| [issue-20261123-79bf13](../../../management/issues/issue-20261123-79bf13.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 観測のキー「base か base.subkey」の解き方が rl.cpp・Replay・probe.cpp・image.cpp の 4 か所にあり、subkey が空のときの扱いと dtype が既に食い違う |
| [issue-20261103-5addea](../../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [RFD0142](../../../memo/0142_agent_prev_action_obs_10rfd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261111-0ef722](../../../management/issues/issue-20261111-0ef722.md) | 未着手 | 2.0 | — | コード | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | probe.hpp・probe.cpp の約 3 分の 1 がどこからも使われず、残る 2 つの Probe クラスと 5 つの IdxExtractor も写しになっている |
| [issue-20261123-cbe34a](../../../management/issues/issue-20261123-cbe34a.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 317_metrics, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | 共通の Runner が 1 フレームごとに MetricsLogger を flush し（学習スレッドでは毎 step）、設計 140 とアプリが書く「境界でだけ flush」と食い違う |
| [issue-20261103-0ef7b8](../../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-50644a](../../../management/issues/issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | RFD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [RFD0184](../../../memo/0184_eval_session_step_trace_10rfd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261123-eac719](../../../management/issues/issue-20261123-eac719.md) | 未着手 | 1.4 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 画像の Observer 4 種は observers（318）にあり、それを作って Notifier へ付けるのは image（317）の ImageProviderFactory で、observers と image が互いを読んでいる |
| [issue-20261103-71c4f4](../../../management/issues/issue-20261103-71c4f4.md) | 未着手 | 1.3 | — | コード | RFD | 315_nn, 317_metrics, 321_agent_core | 学習基盤 | 機能追加 | [RFD0135](../../../memo/0135_nn_block_metrics_10rfd.md) | 残件有り | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| [issue-20261103-7227bf](../../../management/issues/issue-20261103-7227bf.md) | 未着手 | 1.2 | — | コード | RFD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [RFD0153](../../../memo/0153_q_statistics_config_screening_10rfd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| [issue-20261103-51220d](../../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | RFD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [RFD0144](../../../memo/0144_episode_forensics_10rfd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261111-d1d12b](../../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-f40ae8](../../../management/issues/issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | RFD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [RFD0056](../../../memo/0056_timehistogram_fine_grid_10rfd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |

着手できる 31 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
