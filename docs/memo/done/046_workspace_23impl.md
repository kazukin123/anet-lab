# Workspace 機構 PH2 Metrics Viewer 実装メモ

## 概要

Metrics Viewer を固定 `runs` ディレクトリ方式から、1 JVM につき 1 つの current workspace を持つ方式へクリーンブレークする。workspace 固有の Run 走査・取り込み・in-memory cache を不変 snapshot に閉じ、切替中の query と ingest が別 workspace の同名 Run を混ぜないようにする。

## 主な変更

- `metricsviewer.runs-dir` を削除し、`metricsviewer.workspaces-dir=workspaces` と `metricsviewer.initial-workspace=_default` に置換する。initial workspace 名と UNC root は起動時に検証し、不在 workspace は WARN + 空一覧として扱って自動生成しない。
- `GET /api/workspaces.json` と `POST /api/workspace` を追加し、直下 workspace の名前昇順列挙、閉じた request schema、400 `invalid_request`、404 `unknown_workspace`、同一 workspace の 204 no-op を実装する。
- current workspace を epoch・root・workspace 固有サービス群からなる snapshot として管理する。`RunScanner`、`LodPageCache`、`RunWarningRegistry`、`GzipInputSessions`、`IngestScheduler`、`MetricsRepository` は snapshot ごとに生成し、`MetricsCacheDatabase` と query semaphore は process-global に維持する。
- 全 API と ingest cycle は処理開始時に snapshot lease を一度だけ取得する。切替は共通 ingest gate で cycle と複数 POST を直列化し、新 snapshot への swap 後に旧 snapshot を retire する。旧 snapshot の close 必須 resource は利用者ゼロ時に一度だけ閉じる。
- frontend の global controls に workspace selector を追加し、`anet.metricsviewer.workspace` へ保存する。切替時は stale request を無効化し、DataCache・Run 選択・色・viewport・hidden legend・初回選択状態をリセットして新しい Run 一覧を取得する。
- selector 表示後に消えた workspace の切替が404 `unknown_workspace`になった場合は、対象 option を除去して切替前へ戻し、一覧再取得とToast通知を行う。server current自体が消えた場合はdisabledな`(missing)` optionとして保持し、同じ状態を一度だけ通知する。
- workspace 一覧は専用timerを増やさず、初期表示、selector focus、切替結果、手動Reload、Auto Reloadで再取得する。
- Spring 設定 metadata、通常 launcher、Metrics Viewer の利用・設計文書を新契約へ更新し、`22_metrics_viewer_java_optuna.bat` を削除する。

## テスト

- Public interface / surface: workspace REST API、既存 Run/metrics/priority API、Spring 起動設定、browser DOM/localStorage。
- 優先 behavior: 2 workspace の列挙と切替後の同名 Run 表示を tracer bullet とし、API エラー/no-op、initial/UNC 検証、同名同 generation の遅延 query、ingest gate、lease/close-on-zero、A→B→A DB lock、frontend 復元・fallback・state reset・外部rename後の復帰と一覧更新を順に確認する。
- TDD 順序: behavior ごとに 1 テストを RED にし、最小実装で GREEN にしてから次へ進む。refactor は対象 behavior が GREEN になった後だけ行う。

## 検証

```powershell
cd apps\metrics-viewer
mvn -B test
git diff --check
```

## 前提

- 対象は PH2 のみとし、PH1 の未コミット変更を保持する。PH3 Optuna、外部 workspace attach、既存 Run の移動・自動 migration は行わない。
- `metricsviewer.runs-dir` の alias や互換分岐は残さない。
- workspace 切替や不在 initial のためにフォルダまたは cache DB を自動生成しない。
- Workspace / Run作業セットの用語と ADR 0021 は既に十分なため、`CONTEXT.md` と ADR は更新しない。
- 既存 `20impl`〜`22impl` は変更せず、stage、commit、push は行わない。
