# PRD 071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through）

> 確定。D1〜D6、complexity audit、受入基準を実装契約とする。
> 起点: 2026-09-06。CUDA 非搭載機で Runner がウィンドウ表示後に無言終了し、run ログは config ダンプの途中で
> 切れ、エラーダイアログも出なかった。PRD 068 が整備したのは「捕まった fatal の後始末」（表示方針・終了コード・
> launcher 検出）であり、**捕まらない fatal** は非目標に残っていた。
> 関連: PRD 068（batchrun の致命エラー処理。本 PRD はその続き）、PRD 070（device の auto 対応。本 PRD は
> その fail-fast を利用者へ見せる側）、`core/anet-core/src/thread.cpp`、`apps/runner/src/RunnerApp.cpp`

## Context（背景・目的）

### 実測（2026-09-06）

| 環境 | 設定 | 結果 |
|---|---|---|
| CUDA 非搭載機（ZIP v0.3.0 / 最新ビルドの両方） | `agent.device_type = 1` | **無言終了**。ダイアログなし。run ログは config ダンプの途中で切れる。`stderr.log` には torch の警告のみ |
| CUDA 非搭載機 | `agent.device_type = 0` | 正常動作 |
| CUDA 搭載機 + `CUDA_VISIBLE_DEVICES=-1` | `agent.device_type = 1` | **正常系**。NN 構築時に main thread で `c10::Error`、ダイアログ表示、`System error: No CUDA GPUs are available` がログに残る |

3 行目が再現できているのに 1 行目が再現できない、という状態が続いている。本 PRD は原因の特定ではなく、
**どの経路で死んでも痕跡が残る**ことを契約にする。

### 実装着手時に確認した事実

| # | 事実 | 位置 |
|---|---|---|
| 1 | 現行 `ReportError` は `LOG::error()` の直後に `wxLog::FlushActive()` を呼ぶ。ハンドラへ到達していればログに残るはず | `ErrorDialog.cpp:11`（PRD 068 D4） |
| 2 | 実機のログに `System error:` が無く途中で切れる。事実 1 と合わせるとハンドラ未到達 | 実測 |
| 3 | `stderr.log` には torch の警告が残る。torch が stderr へ直接書くため。**ロガー経由の分だけが失われている** | 実測 |
| 4 | `std::set_terminate` はコードベースに 1 件も無い | grep |
| 5 | `PinnedThreadPool::WorkerLoop` の `task()` は裸で呼ばれる。`std::thread` から例外が抜けると即 `std::terminate` | `thread.cpp:198` |
| 6 | 同ファイルの `ParallelFor` はタスクを try/catch し `exception_ptr` で呼び出し元へ返す。**WorkerLoop だけが防御を持たない** | `thread.cpp:98` |
| 7 | PRD 068 D8 の worker 例外転送は `RunnerThread`（`thread.cpp:60`）が対象。`PinnedThreadPool` は対象外 | PRD 068 |
| 8 | 例外時は `pending_tasks_.fetch_sub()` も飛ぶ。仮に terminate しなくても待ち側がハングする | `thread.cpp:200-203` |
| 9 | 通常ログは `app.log_flush_interval_ms = 500` の定期 flush 頼み。`ReportError` が flush するのは wxLog だけで、`standard_stream_logger_` と `MetricsLogger` は通らない | `common.txt:120`、`RunnerApp.cpp:507` |
| 10 | `/SUBSYSTEM:WINDOWS` なので CRT の abort メッセージはどこにも表示されない | `apps/runner/CMakeLists.txt` |
| 11 | wxWidgets は `wxUSE_EXCEPTIONS 1` でビルドされている。main thread の例外は `OnUnhandledException()` へ届く | vcpkg `setup.h:134` |

**事実 5〜7 が本 PRD の中心である。** 068 は `RunnerThread` の worker を fatal 経路へ載せたが、
`PinnedThreadPool` の worker は載っていない。ここから例外が出ると、ログもダイアログも終了コードも残らない。

## 比較履歴

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: 3 層で守る** | worker catch + set_terminate + エラーログ write-through | 経路を特定せずに痕跡を保証できる。診断のたびに環境依存の再現待ちをしなくて済む | 3 箇所に触る |
| B: worker catch だけ | `thread.cpp:198` を塞ぐ | 最小 | 原因が別経路（noexcept デストラクタ内の throw 等）なら空振りする |
| C: set_terminate だけ | 最後の砦だけ置く | 1 箇所 | worker 例外が terminate 前提のままで、068 の fatal 経路（終了コード・表示方針）に載らない |

案 A を採用する。B と C は単独では、今回のように再現条件が環境依存のとき確証を得られない。

## 確定した D1〜D6

| # | 決定 | 契約 |
|---|---|---|
| D1 | worker catch | `PinnedThreadPool::WorkerLoop` の `task()` を try/catch で囲む。`pending_tasks_.fetch_sub()` は例外時も必ず通す |
| D2 | worker 例外の転送 | 捕まえた例外は PRD 068 D8 と同じ fatal 経路（main thread へ現在例外を転送）へ合流させる。転送先が無い場合でも英語のエラーログだけは必ず残す |
| D3 | set_terminate | `OnInit()` の冒頭で `std::set_terminate` を設定する。ハンドラは wx を経由せず、ファイルへ直接 write → `MessageBoxW` → 非ゼロで `_exit`。wx やロガーが壊れている状態でも文字が残ることを優先する |
| D4 | write-through | `[E]` レベルのログは即 flush する。エラーは低頻度なので定期 flush の性能利点を損なわない |
| D5 | 致命経路の同期 flush | `showFatalError()` と set_terminate ハンドラでは、wxLog に加えて `standard_stream_logger_` と `MetricsLogger` も flush してから次へ進む |
| D6 | 終了コード | PRD 068 D2 の契約（正常 0 / fatal 非ゼロ）を terminate 経路でも守る |

## Complexity audit

- Keep: worker の try/catch、set_terminate ハンドラ 1 個、`[E]` の write-through、致命経路の同期 flush。
- Shrink: set_terminate ハンドラは wx 非依存の最小実装（ファイル write + `MessageBoxW` + `_exit`）に留め、専用のクラッシュレポート形式は作らない。
- Defer: worker fatal の E2E 故障注入（PRD 068 でも Defer）。
- Cut: minidump 生成、クラッシュレポートの外部送信、例外種別ごとの終了コード。

## 受入基準

1. `PinnedThreadPool` の worker タスクで例外を投げると無言終了せず、ログにメッセージが残る。online 構成ではダイアログが出て、batchrun 構成では非ゼロ終了する（PRD 068 の契約と同じ振る舞い）。
2. worker タスクで例外が出ても `pending_tasks_` のカウントが漏れず、待ち側がハングしない。
3. `std::terminate` を通る経路（noexcept 関数内の throw など）でも、ファイルにメッセージが残り、MessageBox が出て、非ゼロで終了する。
4. 落ちる直前に書かれた `[E]` ログが、定期 flush の周期に関わらずファイルに落ちている。
5. PRD 068 の受入基準 1〜8 を壊さない。
6. CPU 機で PRD 070 の fail-fast が起きたとき、そのエラーがダイアログとログの両方に出る。

## 非目標

- 無言終了の**原因の特定**。本 PRD は痕跡の保証のみを扱う。原因側は PRD 070 の fail-fast で先に潰す。
- minidump 生成、クラッシュレポートの外部送信。
- `thread.cpp:61` の誤字 `Thrad` 修正（PRD 068 で Cut 済み）。
- エラーダイアログの UI 改善。
