# PRD 068: batchrun 致命エラー処理 実装メモ

## 概要

batchrun 構成では、解決済み設定以降のエラーをモーダル表示せずログへ出し、fatal 時は速やかに非ゼロ終了する。online 構成では従来どおりエラーダイアログを表示する。3 本の batch launcher は各 Run の失敗を記録して後続 Run を継続し、最後に集約した成否を終了コードへ反映する。

## 主な変更

- `app.show_error_dialog: bool` を追加する。`app.online.show_error_dialog = true`、`app.batchrun.show_error_dialog = false` とし、未指定時は `true` とする。不正値は既定 `true` の状態で fail-fast し、online / batchrun をコードから推測しない。
- `ConfigData` 解決直後に実効値を読み、`false` なら既定 GUI logger を所有権安全な非モーダル `wxLogStderr` 経路へ切り替える。Run 成立前は親 stderr、`StandardStreamLogger` 開始後は `stderr.log`、通常 logger 構築後は既存 Run log へ記録する。部分初期化時と終了時にも logger target を復元する。
- `ShowErrorDialog` を `ReportError(message, detail, show_dialog)` へクリーンブレークする。常に英語で error log を出して flush し、許可時だけ modal dialog を開く。fatal、未知例外、checkpoint 保存、Open Run Folder に適用するが、後二者の非 fatal エラーは Run 継続かつ終了コードへ影響させない。ユーザー操作用 dialog は対象外とする。
- main thread 所有の `fatal_error_seen_` を両例外 callback で設定する。`OnRun()` は `wxApp::OnRun()` の元の非ゼロ値を保ち、元が 0 でも fatal 記録済みなら 1 を返す。`OnExit()` は cleanup 専用とする。
- `apps/11_batch_run.bat`、`apps/12_batch_run.bat`、`apps/18_batch_run_atari5.bat` は各 Run 直後の `%ERRORLEVEL%` を保存する。失敗時は日時付きの `[ERROR] RUN FAILED exit_code=<code> args=<args>` と件数を記録し、subroutine は 0 を返して後続 Run を継続する。全 Run 後に件数を表示し、既存の `pause` 後、全成功なら 0、失敗があれば 1 を返す。Run 一覧と順序、CP932 + CRLF、`apps/12_batch_run.bat` の既存未コミット実験追加を保持する。
- PRD 068、`CONTEXT.md`、Runner 利用ガイドと applications/tools 設計文書を確定契約へ同期する。ADR は追加しない。

## テスト

- Public interface / surface: `ConfigManager` の選択・CLI override、`RunnerThread` の例外 callback、Runner プロセスの終了コードと stderr、3 batch launcher の出力・継続・集約終了コード。
- 優先 behavior:
  1. `app.$` が online / batchrun の `app.show_error_dialog` をそれぞれ `true` / `false` へ解決し、source-prefix CLI override が選択前に適用される。
  2. throwing Runner の worker 例外で callback が一度だけ現在例外を受け取り、thread が停止する。
  3. Debug Runner を batchrun 構成かつ不正な `app.log_flush_interval_ms` で起動すると、15 秒以内に modal で停止せず、stderr に設定エラーを残して非ゼロ終了する。
  4. 3 launcher は一時コピーと stub executable により、失敗後も後続 Run を実行し、失敗行・件数・最終非ゼロを返す。全成功時は 0 を返す。
- TDD 順序: 上記 behavior ごとに 1 テストを RED にし、必要最小限の実装で GREEN にしてから次へ進む。private 実装ではなく外部観測可能な契約を検証する。

## 検証

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "[config][resolver][cli]"
core\anet-core\bin\Debug\anet-core-test.exe "[trainer][thread]"
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\batchrun_fatal_error_handling_test.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
```

上記の前に `VsDevCmd.bat` 経由で `x64-Debug` をビルドする。

## 前提

- 正常終了は 0、fatal は非ゼロのみを公開契約とし、例外種別別の終了コードは導入しない。
- worker 例外は既存経路で main thread へ転送されるため `fatal_error_seen_` は atomic にしない。
- `bad allocation` 原因調査、error dialog の UI 改善、外部 crash 通知、初回失敗時中断、専用 startup log、`Thrad` 誤字修正、実 dialog の UI 自動化、worker fatal の完全 E2E 故障注入は対象外とする。
- wxWidgets の契約上、process 終了コードは `OnRun()` の戻り値で決まり、`OnExit()` の戻り値は cleanup の成否通知に用いない。

