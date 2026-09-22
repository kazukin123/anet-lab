# PRD076 終了時評価セッション排水 実装メモ

## 概要

- background 評価を完走または協調キャンセルしてからログ・Metrics を閉じる。既定は完走待ち、全体 deadline は 3600 秒とする。
- 手動終了では train thread を join 後に 3 択を提示し、Keep running なら同じ thread を元の pause 状態のまま再開する。
- PRD076 と ADR0045 を今回の裁定へ同期する。`CONTEXT.md` の「排水」は既に整合しているため変更しない。

## 主な変更

- core API:
  - `ShutdownMode { WAIT, CANCEL }` を追加する。
  - 4 種の Observer 基底へ既定 no-op の `Shutdown(deadline, mode)` と `WillBlockOnShutdown()` を追加し、全 `RunnerScoped*Observer` が実体へ転送する。
  - `Notifier` に一括 `Shutdown()` / `WillBlockOnShutdown()` を追加する。途中の例外は捕捉せず、その呼び出しを中断する。
  - `EvalRunner::RunSession(counts, std::stop_token = {})` とし、停止要求時は `SessionEndEvent` と scalar を出さず、取消ログと `eval.[tag].session_cancelled` JSON を記録する。
- `EpisodeEvalObserver`:
  - `wait_on_exit` とセッションごとに再生成する `stop_source` を所有する。
  - WAIT、設定による CANCEL、deadline 超過 CANCEL、foreground no-op、複数回呼び出しを実装する。
  - `WillBlockOnShutdown()` は train thread join 後に future readiness を照会する。重複する atomic 状態は持たない。
  - 明示 `Shutdown()` は worker 例外を再送出する。デストラクタ安全網だけは例外を捕捉して FATAL を記録し、`std::terminate` を避ける。
- 設定・Runner:
  - `run.eval_schedule.[tag].wait_on_exit` を追加し、既定 `true`、起動時 `scheduled` ログにも出す。
  - `app.drain_timeout_sec` を追加し、既定 3600、正整数以外はキー・値・期待範囲を含めて fail-fast する。
  - `RunnerApp` に排水照会・実行・close-veto 後の再開口を追加する。正常排水後の `OnExit()` は no-op とし、開始ログを重複させない。例外で未完了なら `OnExit()` が同じ排水を再試行する。
  - `OnClose()` は手動終了なら Stop/join、必要時3択、checkpoint保存、排水、`ShutdownRunLogging()` の順にする。メニュー Exit は `Close()`、予算到達は従来どおり `Close(true)` とする。
- config ファイルには既定行を追加しない。現用 Atari 設定やユーザーの未コミット変更も触らない。
- PRD076、ADR0045、`010_framework_overview.jp.md`、`020_user_guide_run.jp.md`、`100_runtime_and_configuration.jp.md`、`140_observability.jp.md`、凍結PRD026を現行契約へ更新する。`.en.md` は対象外とする。

## テスト

- Public interface / surface: Observer/Notifier の shutdown API、`EvalRunner::RunSession()`、`EpisodeEvalObserver`、`run.eval_schedule.[tag].wait_on_exit`、`app.drain_timeout_sec`、Runner の close behavior。
- 優先 behavior:
  1. `[prd076]` tracer bullet として、`AttachScoped()` した background 評価を `Notifier::Shutdown(WAIT)` で完走させ、`SessionEndEvent`、scalar、wrapper 転送を確認する。
  2. 明示 CANCEL、`wait_on_exit=false`、deadline 超過を1 Step以内で終了する。
  3. CANCEL/TIMEOUT で session-end/scalar を出さず、部分 trace、取消ログ2行、座標・経過時間・完了数を持つ JSON を残す。
  4. 冪等性、foreground no-op、4種 wrapper 転送、`WillBlockOnShutdown()` の4状態を確認する。
  5. worker 例外の再送出とデストラクタの FATAL 捕捉を確認する。
  6. `wait_on_exit` の既定、明示 false、scheduled ログを確認する。
  7. `app.drain_timeout_sec` の0、負数、非整数を Runner 起動テストで拒否する。
- TDD 順序: 上記を1 behaviorずつ RED、最小実装で GREEN、関連テスト再実行の順に進め、GREEN 後だけ refactor する。
- 実 Run: WAIT、timeout-cancel、同時複数スロット、exit code 0、artifact、経過時間を確認する。GUIでは3択と Keep running 後の再開を手動確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[prd076]"
core\anet-core\bin\Debug\anet-core-test.exe
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\batchrun_fatal_error_handling_test.ps1
git diff --check
```

## 完了条件

- WAIT は `session end`、scalar、`eval_episodes` 本の trace、exit code 0を満たす。
- CANCEL/TIMEOUT は scalar 不在、取消ログ、JSON、上限＋1 Step＋数秒以内、exit code 0を満たす。
- 単一変更として完成させ、部分出荷、commit、pushは行わない。

## 前提と範囲外

- 明示 `Shutdown()` は worker 例外を再送出する。デストラクタ安全網のみ例外を捕捉し FATAL を記録する。
- 手動 close は train thread を Stop/join してから shutdown が blocking か照会する。Keep running は同じ `RunnerThread` を、それまでの pause 状態を保って再開する。
- `Env::Step()` が返らない故障、待機中操作UI、PRD999の初回発火制御、PRD026 R1、`PinnedThreadPool` 改造は範囲外とする。
