# Workspace 切替のロック待ち改善 実装メモ

## 概要

Metrics Viewer の workspace 切替が、最大 100 万行を 4 block 処理する ingest cycle 全体の完了を待って長時間応答しない問題を修正する。定常時の block 上限と priority 3 : background 1 の配分は維持し、切替要求中だけ現在 block を完全行境界で早期 commit して block 間で切替ゲートを譲る。

## 主な変更

- `ingestGate` を fair lock にし、待機中の切替要求数を管理する。切替要求はゲート待機前に登録し、ゲート内処理の終了時に解除する。
- `IngestScheduler` は作業セットを 4 slot ごとに 1 回走査しながら、1 block だけ進める操作を提供する。既存の 4 block cycle と 3 : 1 配分は維持する。
- `WorkspaceManager.runIngestCycle()` は 1 つの snapshot lease を保持したまま block ごとにゲートを取得・解放する。epoch が変わったら旧 snapshot の残り block を実行せず終了する。
- `MetricsIngestor` と `SourceReader` は内部の早期 yield 判定を受け取り、切替要求を検出したら次の完全行を同じ transaction へ反映して commit する。offset、L0、LOD、TagStats の commit 境界は変更しない。
- `MAX_BLOCK_LINES = 1_000_000`、HTTP API、設定キー、レスポンス形式は変更しない。設計資料へ定常時上限と切替時の早期 commit を記録する。

## テスト

- Public interface / surface: `WorkspaceManager.switchWorkspace()` / `runIngestCycle()`、`IngestScheduler.runCycle()`、`MetricsIngestor.ingestBlock()` と workspace REST API の既存契約。
- 優先 behavior: 取り込み中の切替を tracer bullet とし、現在 block の安全な終了後に POST が進み、旧 workspace の残り block が実行されないことを確認する。次に早期 yield の commit・再開、1 block 操作の 3 : 1 配分、既存の同時切替・no-op・unknown・close-on-zero を確認する。
- TDD 順序: behavior ごとに 1 テストを RED にし、最小実装で GREEN にしてから次のテストへ進む。refactor は関連テストが GREEN の後だけ行う。

## 検証

```powershell
cd apps\metrics-viewer
mvn -B test
git diff --check
```

## 前提

- 切替待ちは通常サイズの次の完全行境界までとし、厳密な時間 SLA は設けない。極端に巨大な単一 JSONL 行の途中では中断しない。
- 固定 block 上限の縮小、新しい設定項目、互換分岐は追加しない。
- 既存の `20impl`〜`23impl`、PH1・PH3、無関係な未コミット変更は変更しない。
