---
title: アプリケーションとツール
page_kind: サマリ
material: "[[docs/design/160_applications_and_tools.jp.md|160_applications_and_tools.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-27
---
# アプリケーションとツール

## 元の材料

[アプリケーションとツール](../../../design/160_applications_and_tools.jp.md) blob:a1440fe301848b9ffbf1e9d76df819e3c3f6b77b

## 要点

- Runner、Metrics Viewer、Optuna の共通の受け渡し先は Run ディレクトリである。
- Runner が設定から学習・評価を開始し、Evaluation Trace や Logs を画面に表示する。
- Viewer は Run の記録を読み、表示範囲に必要なデータをブラウザへ返す。
- Optuna harness は複数 seed の試行と Runner 子プロセスの起動・終了・タイムアウトを管理する。
- 開発ツールは wiki の材料・リンク・取り込み記録や、課題一覧とトピック別まとめの対応を検査・更新する。

## 内容

### 2. コンポーネント定義

- `TrainPanel` — Train eventからEnv固有Viewを更新し、GUI timerで断面を描画する。
- `LogPanel` — wxLogの画面表示とlevel filterを提供する。WARN以上とVERBOSEの色分け、薄い時刻、未確認のWARN/ERRORの件数、Auto-scroll・Wrap・Indent・全体コピー・クリア・検索の操作帯を持つ。
- `DefaultViewFactory` — Env class IDからTrain/Eval用Viewを生成する。
- `RunScanner` — runs directory直下から`metrics.jsonl`または`metrics.jsonl.gz`を持つRunを列挙する。
- `MetricsCacheDatabase` — RunごとのSQLite cache、source fingerprint、generation、短命connectionを管理する。
- `dropmerge_optuna.py` — DropMerge domain、CLI、探索parameterを定義するentry script。
- `DropMergeDomain` — 探索空間、generated config、cost、score tagをまとめるdomain adapter。

### 4. 静的構造

- RunnerはviewerやDashboardへ直接接続しない。Run directory、Optuna DB、artifact storeがprocess間contractである。この境界により、学習中でも別processから追記済みmetricsを閲覧できる。

### 5. 処理フロー

- HTTP requestはMetricsマスタを直接読まず、background writerがcommitしたSQLite snapshotを読む。
- 各rangeは前回応答へ依存せず、clientはviewport左右1画面を含む3画面windowを置換する。
- 完全検証済みの`ready` / `error` Runはsource/cache属性が不変なら本文・fingerprint・SQLiteへ入らず、即時処理可能な`converting` backlogがないcycleは10秒sleepする。
- 取り込み中Runがある間はRun metadataを4秒pollして進捗表示だけを更新し、Auto Reloadは30秒ごとにmetadataを取り直して最新stepへfollow中の系列だけrangeを更新する。
- `run-study`は各trialで`run-trial`子processを起動せず、自身がrunner子processを管理する。multi-seedは1 trial内で逐次実行され、`--n-jobs > 1`では複数params候補が並列に進む。

### 6. Entry pointと運用境界

- 通常Runnerはworkspaceを選び、実験差分は末尾の`key=value` overrideで渡す。`--config`はworkspaceを使わない完全自己記述起動に限定する。
- Viewerは`data/workspaces`を既定の親directoryとし、画面のselectorでcurrent workspaceを切り替える。親directoryは`--metricsviewer.workspaces-dir`で差し替える。
- Optuna harnessは既存workspaceと`config/_main.txt`をpreflightし、Runを`runs/`、SQLite storage・artifact store・harness logを`optuna/`へ固定する。run系の出力overrideは選択workspaceの`optuna/`配下だけを許可する。
- Optuna harnessのPythonはrepository rootの`.venv`を使う。
- Viewer frontendはbuild工程を持たず、Plotlyはjarに含めたWebJarから配信する。ブラウザからの外部接続なしでグラフを描ける。

### 7. Lifetime・エラー・性能特性

- `RunnerApp`が`RunManager`、`RunnerThread`、Frameを保持し、loggerの初期化・flush・停止順序を統括する。Run directoryと`run_dir_`は`MetricsLogger`が所有する。`app.show_error_dialog`は未指定時`true`で、online構成は`true`、batchrun構成は`false`へ解決する。構成名はコードで推測しない。
- 初期化中（構築スレッド開始から結果受領まで）の拒否可能なCloseは一度だけ記憶して拒否し、構築後に初期化の後半と学習開始を省いて閉じる。RUNNINGなら学習前のAgentを保存できる。RUNNINGでなければ`Failed to initialize RunManager.`を出し、終了要求がなければウィンドウを残す。HeatMapは初期化までUpdate UIとhandlerで抑止する。
- source kind、size、mtime、先頭/commit直前hashが一致しない場合はSQLite cacheを破棄して新しいgenerationで再構築する。
- 中断時はまず`Ctrl+C`を1回送り、harnessへrunner停止とtrial state更新を行わせる。
- wikiは外の正本を束ね、書く部分の意味はエージェントが判断する。材料・取り込み・記録はwikiの規約に従う。 資料ごとのサマリには要約時の blob を記録し、概念のページは元の資料を直接引く。
- `reports/tools/chat_utterances.py` は、Claude Code と Codex の会話の記録から、開発者の発言の行だけを、語か期間（日本時間の日付、両端を含む）で引く。読み取りだけで、日付・会話の識別子・次の発言までに AI が道具を使った回数を添え、最後に飛ばしたファイルと壊れた行の数を出す。道具の結果、環境が差し込んだ行、通知、サブエージェントの会話、質問欄への返答、人の操作でない起動の会話は返さない（§7.4）。検証資材は RFD0202 の `chat_utterances_test.py`（§8）。

### 8. テストと変更時の確認事項

- RFD0122検証資材: windowを作らないQ-Values加工の単体テスト、Panelのキー転送の特性テスト、coreの9箇所の破壊検査と4環境の全scalar比較。GUIの表示・フォーカス・実キーの動作は人が確認する。
- RFD0110検証資材: 主paneの初期表示・Centre昇格とdock位置復帰、三導線の開閉制限、最大化中の開閉、Train描画timer、未知経路の保険とWARN、既定スイートの失敗集合を確認する。captionのドラッグと実画面の幅は人が確認する。
- RFD0109検証資材: 構築中のtimer・ログ・pane操作、初期化中のCloseとOS終了問い合わせ、失敗と不完全な構築、ウィンドウ先行破棄、既定スイートの失敗集合、4環境の全scalar点の完全一致。実機の描画とOSの「応答なし」は人が確認する。
- RFD0105検証資材: Traceモードの転送矢印、forcedの通知時点、並べ替えと再生順、ダンプの書式と実画面での表示時間、4環境の数値等価性。
- RFD0094検証資材: 列、時点、診断、再生、snapshot、5構成の数値等価性。表示・操作の変更では実wxWidgetの`AnetRLRunner-gui-test`で幅・配色・toggle・赤枠を確認し、UI処理の計測と実Runの速度を区別する。

## 主要な概念

- [RunnerApp](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [311 開発ツール](../../concepts/framework/311_dev_tools.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
