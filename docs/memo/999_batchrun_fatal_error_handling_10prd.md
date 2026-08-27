# batchrun の致命エラー処理（ダイアログ抑止・終了コード・bat 検出）暫定 PRD

> 状態: 暫定メモ。案と決定事項 D1〜D8 は未確定。詳細は別途グリルで詰める。本 PRD は実装着手を意味しない。
> 起点: 2026-08-27 の Atari-5 初回スイープで phoenix が `bad allocation` で落ちた際、**モーダルダイアログが
> 後続 4 本を 13 分 48 秒ブロックした**こと。batch モードは定義上無人なので、ダイアログは通知として機能せず
> 待ち時間だけを生む。
> 関連: `../experiments/default-dqn/atari/2026-08-27_atari5.md`（実測。phoenix の異常終了）、
> `apps/12_batch_run_atari5.bat`、`docs/design/100_runtime_and_configuration.jp.md`（app プロファイルの選択）。

## Context（背景・目的）

### 実測（2026-08-27 Atari-5 スイープ）

`12_batch_run_atari5.bat` で 5 ゲームを連続実行した際の Run 間隔。

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

1. **batch モードでダイアログが後続をブロックする。** 止まるべきは失敗した Run であり、それは既に止まっている
   （プロセスが死んでいる）。ダイアログが追加で止めているのは無関係な後続である。
   しかも batch は無人なので、通知としての価値もない。
2. **終了コードが常に 0。** `RunnerApp::OnExit()` は無条件で `return 0` を返す（§事実 4）。
   ダイアログを消すだけだと**失敗が完走と見分けられなくなる**ため、ここが①の前提条件になる。
3. **bat が `errorlevel` を見ていない。** スイープの出力が自己記述的でなく、後から各 Run のログを
   grep して回る必要がある。

## 現行コードで確定している事実（実装の下地）

| # | 事実 | 位置 |
|---|---|---|
| 1 | `ShowErrorDialog` は **`LOG::error()` でログを書いてから** `dlg.ShowModal()` する。つまり**ログはダイアログの有無と無関係に残る** | `apps/runner/src/ErrorDialog.cpp:13`（ログ）、`:81`（modal） |
| 2 | `ShowErrorDialog` の呼び出しは **3 箇所すべて無条件** | `RunnerApp.cpp:631`（AnetException）、`:634`（std::exception）、`RunnerFrame.cpp:172`（`ShowUiOperationError`） |
| 3 | 未知例外は素の `wxMessageBox` を直接呼ぶ。**4 箇所目のブロック点**でありログも残らない | `RunnerApp.cpp:636` |
| 4 | **`OnExit()` は無条件で `return 0`** | `RunnerApp.cpp:613-621` |
| 5 | `showFatalError()` の入口は 2 つ | `RunnerApp.cpp:640`（`OnExceptionInMainLoop`）、`:644`（`OnUnhandledException`） |
| 6 | ワーカスレッド例外は `catch(...)` でログを書いて `OnException()` を呼ぶ（ログ文字列は `Thrad [name]: Exception caught.` — **`Thrad` は誤字**） | `core/anet-core/src/thread.cpp:60-63` |
| 7 | **`RunnerApp::Config` は prefix `"app"` で読む。** 選択チェーン（`app.$ = app.online > P1`）解決後の実効値しか見えず、**コードからは online / batchrun を判別できない** | `RunnerApp.cpp:70` |
| 8 | 現在のモード区別は `exp_pause_step` / `exp_exit_step` の正負による**暗黙**のもの | `RunnerApp.cpp:385`（exit）、`:399`（pause） |
| 9 | **`app.online.*` / `app.batchrun.*` の並行キーは既に前例がある** | `Atari.txt:193`/`:196`（exp_pause/exit_step）、`:739-740`（`eval_panel.auto_start`） |
| 10 | bat に `errorlevel` の検査は無い | `apps/12_batch_run_atari5.bat` |

**事実 7 + 9 が設計上重要である。** 選択チェーンが既にモードごとの値を運んでいるので、
`app.online.show_error_dialog` / `app.batchrun.show_error_dialog` の 2 行を置けば、
**コード側にモード判定を足さずに**モード由来の既定が成立する。実効設定は単一の `app.show_error_dialog` になる。

## 案（グリルで選択）

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: モード由来の既定 + 終了コード + bat** | `app.online.show_error_dialog = true` / `app.batchrun.show_error_dialog = false` を置き、fatal を通ったら非ゼロ終了、bat で `errorlevel` を拾う | 既定が正しい側に倒れる。既存の並行キーと同形。コードにモード判定不要 | 3 箇所（config / 終了コード / bat）に触る |
| B: 裸のフラグ 1 個 | `app.show_error_dialog = false` を batch 用 config へ書く | 変更最小 | **書き忘れると元の症状に戻る**。batch を新規に作るたび再発しうる |
| C: bat 側だけで回避 | `start /wait` にタイムアウトを付ける等 | アプリ無改修 | ダイアログが出たまま。終了コードも 0 のままで失敗検出は解決しない |

案 A を軸に詰める想定。

## 決定事項（未確定）

| # | 論点 | メモ |
|---|---|---|
| D1 | キー名と既定 | `app.<mode>.show_error_dialog`（online=true / batchrun=false）でよいか。名前は `suppress_*` でなく肯定形を推奨（既存 `auto_start` と同じ向き） |
| D2 | **終了コードの持ち方** | fatal を通った事実をどこに持って `OnExit()` へ渡すか（`RunnerApp` のメンバ / atomic）。値は 1 でよいか、種別で分けるか |
| D3 | `ShowUiOperationError` の扱い | `RunnerFrame.cpp:172` は UI 操作（保存等）の失敗で、batch では原理的に起きないはず。同じフラグで抑止するか、対象外にするか |
| D4 | 未知例外の素 `wxMessageBox` | `RunnerApp.cpp:636` は**ログすら残らない**。抑止対象に含めるのは当然として、**ログを追加すべき**ではないか（本 PRD で直すか別件か） |
| D5 | 抑止時の代替通知 | ログのみで足りるか、stderr にも 1 行出すか（bat のコンソールに出ると気づきやすい） |
| D6 | bat の検出形式 | `if errorlevel 1 echo *** FAILED ***` の文言と、END 行との関係 |
| D7 | **1 本目失敗でスイープを中断するか** | ゲーム横断は独立なので継続が既定で正しいが、config エラーなら残り全部が確実に落ちる。「1 本目だけ fail-fast」は bat 2 行で済む。実装不要なので先行適用も可 |
| D8 | ワーカスレッド例外と main loop 例外 | `thread.cpp` 経由と `OnExceptionInMainLoop` 経由で扱いを分ける必要があるか。今回の `bad allocation` は前者 |

## 受入基準（案）

1. **batch 構成でダイアログが出ない。** 意図的に fatal を発生させた smoke で、モーダルが表示されず即座にプロセスが終了する。
2. **同構成で終了コードが非ゼロ。** bat から `errorlevel` で検出できる。
3. **online 構成では従来どおりダイアログが出る**（既定の回帰なし）。
4. **ログの内容が抑止前後で同一。** `ShowErrorDialog` はログを先に書くので（事実 1）、抑止してもログは欠けない。
5. **bat の出力に失敗が現れる。** 5 本走らせて 1 本失敗させたとき、出力だけで «どれが落ちたか» が分かる。
6. 正常終了時の終了コードは 0 のまま。

## 非目標

- **`bad allocation` そのものの原因究明。** 本 PRD はエラー時の振る舞いのみ扱う。phoenix の確保元は
  `2026-08-27_atari5.md` の pending に残っている（プロセスメモリのメトリクスが無く事後追跡不能）。
- ダイアログ自体の UI 改善（スタックトレース表示、クリップボードコピー等は現状のまま）。
- クラッシュレポートの外部送信。
- `thread.cpp:61` の誤字 `Thrad` 修正。grep しづらいだけで実害はないが、触るついでに直すかは D 枠外の判断。
