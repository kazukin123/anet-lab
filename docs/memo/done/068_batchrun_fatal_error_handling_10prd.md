# PRD 068: batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）

> 状態: 確定。D1〜D8、complexity audit、受入基準を実装契約とする。
> 起点: 2026-08-27 の Atari-5 初回スイープで phoenix が `bad allocation` で落ちた際、**モーダルダイアログが
> 後続 4 本を 13 分 48 秒ブロックした**こと。batchrun 構成は定義上無人なので、ダイアログは通知として機能せず
> 待ち時間だけを生む。
> 関連: `../experiments/default-dqn/atari/2026-08-27_atari5.md`（実測。phoenix の異常終了）、
> `apps/18_batch_run_atari5.bat`、`docs/design/100_runtime_and_configuration.jp.md`（app プロファイルの選択）。

## Context（背景・目的）

### 実測（2026-08-27 Atari-5 スイープ）

`18_batch_run_atari5.bat` の前身で 5 ゲームを連続実行した際の Run 間隔。

| 遷移 | 間隔 |
|---|---|
| battle_zone → double_dunk | 3 秒 |
| double_dunk → name_this_game | 4 秒 |
| name_this_game → phoenix | 5 秒 |
| **phoenix（17:07:13 に `bad allocation`）→ qbert（17:21:01）** | **13 分 48 秒** |

正常遷移は 3〜5 秒なので、**約 13 分 45 秒はダイアログの前で人間を待っていた**時間である。
本スイープは 1 本 57 分 × 5 = 4.75h だったが、予算を 50M へ上げると 1 本 2.4h × 5 = 約 12h になる。
**1 本目で落ちた場合、気づくまでの時間がそのまま残り 4 本の遅延になる。**

### 問題は 3 つある

1. **batchrun 構成でダイアログが後続をブロックする。** 止まるべきは失敗した Run であり、それは既に止まっている
   （プロセスが死んでいる）。ダイアログが追加で止めているのは無関係な後続である。
   しかも batchrun 構成は無人なので、通知としての価値もない。
2. **fatal を process 終了コードへ反映する経路がない。** `wxApp::OnRun()` の結果を補正していないため、
   main loop が 0 を返す fatal 終了を完走と見分けられない。`OnExit()` は cleanup 用であり、process 終了コードには使われない。
3. **bat が `errorlevel` を見ていない。** スイープの出力が自己記述的でなく、後から各 Run のログを
   grep して回る必要がある。

## 実装着手時に確認した事実（実装の下地）

| # | 事実 | 位置 |
|---|---|---|
| 1 | 既存 `ShowErrorDialog` は `LOG::error()` の後に `ShowModal()` していた。ログと表示の責務を分離できる | `apps/runner/src/ErrorDialog.cpp` |
| 2 | fatal、未知例外、checkpoint 保存、Open Run Folder の通知が modal 経路を通り得る | `RunnerApp.cpp`、`RunnerFrame.cpp` |
| 3 | 未知例外は素の `wxMessageBox` を直接呼んでいたため、ログを共通化する必要がある | `RunnerApp.cpp` |
| 4 | wxWidgets の process 終了コードは `OnRun()` の戻り値であり、`OnExit()` の戻り値は使われない | [wxApp API](https://github.com/wxWidgets/wxWidgets/blob/master/interface/wx/app.h)、[entry implementation](https://github.com/wxWidgets/wxWidgets/blob/master/src/common/init.cpp) |
| 5 | `showFatalError()` の入口は `OnExceptionInMainLoop()` と `OnUnhandledException()` の 2 つ | `RunnerApp.cpp` |
| 6 | ワーカスレッド例外は `catch(...)` でログを書いて `OnException()` を呼ぶ（ログ文字列は `Thrad [name]: Exception caught.` — **`Thrad` は誤字**） | `core/anet-core/src/thread.cpp` |
| 7 | **`RunnerApp::Config` は prefix `"app"` で読む。** 選択チェーン（`app.$ = app.online > P1`）解決後の実効値しか見えず、**コードからは online / batchrun を判別できない** | `RunnerApp.cpp` |
| 8 | 実装着手時の構成差は `exp_pause_step` / `exp_exit_step` の正負に暗黙依存していた | `RunnerApp.cpp` |
| 9 | **`app.online.*` / `app.batchrun.*` の並行キーは既に前例がある** | `apps/runner/config/common.txt`、`Atari.txt` |
| 10 | bat に `errorlevel` の検査は無い | `apps/11_batch_run.bat`、`apps/12_batch_run.bat`、`apps/18_batch_run_atari5.bat` |

**事実 7 + 9 が設計上重要である。** 選択チェーンが既にモードごとの値を運んでいるので、
`app.online.show_error_dialog` / `app.batchrun.show_error_dialog` の 2 行を置けば、
**コード側にモード判定を足さずに**モード由来の既定が成立する。実効設定は単一の `app.show_error_dialog` になる。

## 比較履歴

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: モード由来の既定 + 終了コード + bat** | `app.online.show_error_dialog = true` / `app.batchrun.show_error_dialog = false` を置き、fatal を通ったら非ゼロ終了、bat で `errorlevel` を拾う | 既定が正しい側に倒れる。既存の並行キーと同形。コードにモード判定不要 | 3 箇所（config / 終了コード / bat）に触る |
| B: 裸のフラグ 1 個 | `app.show_error_dialog = false` を batch 用 config へ書く | 変更最小 | **書き忘れると元の症状に戻る**。batch を新規に作るたび再発しうる |
| C: bat 側だけで回避 | `start /wait` にタイムアウトを付ける等 | アプリ無改修 | ダイアログが出たまま。終了コードも 0 のままで失敗検出は解決しない |

案 A を採用する。モード名を実行時に推測せず、選択チェーンで単一の実効設定へ解決する。

## 確定した D1〜D8

| # | 決定 | 契約 |
|---|---|---|
| D1 | キー名と既定 | `app.show_error_dialog: bool`。`app.online.show_error_dialog=true`、`app.batchrun.show_error_dialog=false`、未指定時は `true`。不正値は既定 `true` の状態で fail-fast する |
| D2 | fatal と終了コード | main thread 所有の `fatal_error_seen_` を持つ。`OnRun()` は元の非ゼロを保ち、元が 0 でも fatal 済みなら 1 を返す。公開契約は正常 0 / fatal 非ゼロだけとする |
| D3 | 非 fatal UI 操作 | checkpoint 保存と Open Run Folder も同じ表示フラグへ従うが、失敗後は Run を継続し、終了コードへ影響させない |
| D4 | 未知例外 | 共通 `ReportError` で英語の error log と flush を必ず行い、許可時だけ dialog を表示する |
| D5 | 抑止時のログ先 | `ConfigData` 解決直後に `wxLogStderr` へ切り替える。Run 成立前は親 stderr、標準 stream logger 開始後は `stderr.log`、通常 logger 構築後は既存 Run log へ残す |
| D6 | launcher の検出 | Run 直後に `%ERRORLEVEL%` を保存する。成功時だけ既存 `END`、失敗時は日時付き `[ERROR] RUN FAILED exit_code=<code> args=<args>` と件数を出す |
| D7 | 失敗後の継続 | subroutine は常に 0 で戻り、残りの Run を必ず継続する。最後の `pause` 後、全成功は 0、1 件以上失敗は 1 を返す |
| D8 | worker 例外 | 既存どおり worker から main thread へ現在例外を転送し、同じ fatal 経路で扱う。latch は atomic にしない |

ユーザー操作用の About、file picker、workspace 選択 dialog は `app.show_error_dialog` の対象外とする。

## Complexity audit

- Keep: 1 個の bool 設定、早期 stderr target、fatal latch + `OnRun()`、3 launcher の局所集計、設定/thread/smoke/launcher の検証、文書同期。
- Shrink: 終了コードは 0 / 非ゼロだけ、latch は非 atomic、batch helper は作らず、PowerShell smoke は 1 本にまとめる。
- Defer: worker fatal の完全 E2E 故障注入、online dialog の UI 自動化。
- Cut: 例外種別別終了コード、初回失敗時中断、専用 startup log、ADR、`Thrad` 誤字修正。

## 受入基準

1. `app.$=app.online` は実効 `app.show_error_dialog=true`、`app.$=app.batchrun` は `false` へ解決され、source-prefix CLI override も選択前に適用される。
2. batchrun 構成で解決済み設定以降に fatal が起きても modal で停止せず、英語の error log を flush して 15 秒以内に非ゼロ終了する。
3. online 構成では従来どおり error dialog を表示する。未指定時も `true` とし、不正 bool は黙って `false` にしない。
4. worker 例外は callback が一度だけ現在例外を受け取り、thread が停止し、main thread の fatal 処理へ転送できる。
5. Save Checkpoint / Open Run Folder の失敗は同じ表示方針へ従うが、Run 継続と終了コード 0 の契約を維持する。
6. 3 launcher は失敗した Run の code と args を出力し、後続 Run を続け、最後に失敗件数を表示して 1 を返す。全成功時は成功件数と 0 failures を表示して 0 を返す。
7. launcher の Run 一覧・順序、最終 `pause`、`apps/12_batch_run.bat` の既存実験追加、CP932 + CRLF を保持する。
8. `OnExit()` は cleanup 専用のまま、process 終了コードは `OnRun()` が決める。

## 非目標

- **`bad allocation` そのものの原因究明。** 本 PRD はエラー時の振る舞いのみ扱う。phoenix の確保元は
  `2026-08-27_atari5.md` の pending に残っている（プロセスメモリのメトリクスが無く事後追跡不能）。
- ダイアログ自体の UI 改善（スタックトレース表示、クリップボードコピー等は現状のまま）。
- クラッシュレポートの外部送信。
- 例外種別別の終了コード、初回失敗時の batch 中断、専用 startup log。
- 実 dialog の UI 自動化、worker fatal の完全 E2E 故障注入。
- `thread.cpp:61` の誤字 `Thrad` 修正。
