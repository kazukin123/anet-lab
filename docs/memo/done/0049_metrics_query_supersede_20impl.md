# Metrics query supersede 実装メモ

## 概要

同じブラウザタブの新しい metrics query が古い query をサーバ側で停止し、同時実行枠を最新 request へ明け渡す。`POST /api/metrics.json` へ query channel と単調増加 sequence の必須ヘッダを追加し、workspace 切替と shutdown にも同じ cancel 経路を適用する。

## 主な変更

- `MetricsQueryCoordinator` が fair semaphore、channel ごとの最新 sequence、最大64件の LRU、identity 付き ticket、workspace epoch watermark、terminal な shutdown を一体で管理する。同一 channel の新 sequence は実行中・枠待ち ticket を停止し、遅着した古い sequence は実行しない。
- package-local な query execution token を Repository、Projector、LOD cache へ明示的に渡し、workspace epoch の束縛、cancel checkpoint、実行中 SQL `Statement` の登録を行う。SQL cancel と外部処理は coordinator の critical section 外で行う。
- `MetricsService` を「body/header 検証 → coordinator → Workspace lease → repository query」の順へ変更する。supersede は 409、別 channel による枠不足は既存の 503、shutdown 中の lease 取得失敗は 503 とする。
- `WorkspaceManager` は `SWITCHED` の旧 epoch だけを切替 gate 解放後に cancel する。`NO_OP` と `UNKNOWN` は query を止めない。shutdown は `cancelAll()`、`LoadingThread` 停止、`WorkspaceManager.shutdown()` の順にする。
- `POST /api/metrics.json` は `X-Query-Channel` と `X-Query-Sequence` を必須にする。channel は非blankかつ1〜128文字、sequence は0〜JavaScript safe integerとし、不正値は 400 `invalid_request` にする。
- frontend はページロード時に `crypto.randomUUID()` で channel を生成し、最初を0として POST ごとに sequence を増やす。409 `superseded` は `Update failed` と `console.error` の対象外にし、既存の `AbortController` は維持する。
- `docs/design/210_metrics_viewer.jp.md` のコンポーネント、コードマップ、構造図、HTTP、lifetime、並行制御を新契約へ更新する。既存の PRD 049、ADR 0023、`CONTEXT.md` の用語は重複作成せず保持する。

## テスト

- Public interface / surface: `MetricsQueryCoordinator.run()` / `cancelWorkspace()` / `cancelAll()`、`POST /api/metrics.json` のヘッダと 400/409/503、`WorkspaceManager.switchWorkspace()` の結果別挙動、frontend の channel・sequence・更新失敗表示。
- 優先 behavior: 同一 channel の sequence 1 実行中に2が到着して1が停止し2が成功する tracer bullet から始める。続いて枠待ち停止、別 channel、遅着 sequence、identity cleanup、LRU、epoch race、`Statement` race、terminal shutdown、HTTP、workspace、Playwright の順に確認する。
- TDD 順序: 1 behavior ごとに1テストを RED にし、最小実装で GREEN にしてから次へ進む。全 slice が GREEN になった後だけ整理・refactorを行う。

## 検証

```powershell
cd C:\dev\anet-lab\apps\metrics-viewer
mvn -B test
git -c safe.directory=C:/dev/anet-lab diff --check
```

## 前提

- `max-concurrent-queries`、5秒待機、`Retry-After: 2`、frontend abort は変更しない。
- channel 値は trim せず、whitespace-only だけを blank として拒否する。
- `runs.json` の枠編入、query 高速化、foreground/background 優先度、排他構造の再設計、未使用コード削除は対象外とする。
- 無関係な未コミット変更と `apps/metrics-viewer/.checkstyle` は変更しない。
