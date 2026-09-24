# 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない

学習予算の末尾で発火した背景評価セッション（ε=0 断面、400 エピソード）が、`RunnerFrame::OnClose()` → `MetricsLogger::Reset()` の停止経路に待たれずに失われた（`run_20260917-225635`。判定に使う断面そのものを失い、復旧に評価専用 Run を 1 本要した）。停止経路は train スレッドしか止めず、`EpisodeEvalObserver` の背景プールはどこからも触られない。一方で「待たずに捨てる」は素朴には作れない。`PinnedThreadPool::Stop()` は実行中のタスクが返るまで worker を join し、`EvalRunner::RunSession()` にはキャンセル要求を見る点が無いため、future の `get()` を飛ばしても同じだけ待つ。detach や leak で捨てれば、スレッドが `Reset()` と `RunManager` の破棄を生き延びて解放後の物を触る。

**終了経路では、進行中の背景評価セッションを完走またはキャンセルで終わらせてから、記録先（file logger / metrics）を閉じる（排水）**ことを決定する。キャンセルは `RunSession()` のステップ単位のループに `std::stop_token` を見せる協調停止で作り、キャンセルしたセッションは何も集計しない。`PinnedThreadPool` は触らない。排水は checkpoint 保存の後・`ShutdownRunLogging()` の前に置き、冪等にして `OnClose()` と `OnExit()` の両方から呼ぶ。完走を待つかは評価スロット単位の `run.eval_schedule.[tag].wait_on_exit`（既定 `true`）で選び、排水全体には app レベルの単一の上限 `app.drain_timeout_sec`（既定 3600、無制限モード無し）を置いて、超過したら残りをキャンセルして正常終了する。人が閉じたときは train thread を先に Stop/join してから進行中評価を照会し、待ちが発生するなら「完走を待って閉じる / キャンセルして閉じる / 閉じない」を 1 回聞く。閉じない場合は同じ thread を元の pause 状態で再開し、予算到達では聞かない。明示 `Shutdown()` は worker 例外を再送出するが、デストラクタ安全網だけは例外を FATAL に記録して `std::terminate` を避ける。

## Considered Options

- **detach / leak で捨てる**: スレッドが `MetricsLogger::Reset()` 後に trace を書いて null 参照（PRD 026 の経路）し、`EvalRunner` / Env / Agent の解放後アクセスにもなる。`std::thread` は join せずに破棄できない。却下。
- **`PinnedThreadPool` にタスク単位の stop token を配る**: 利用者が 1 つしか無い口を汎用層に置くことになる。pool の「投入済みタスクを捌いてから止まる」意味論はキャンセルと独立で、pool を直しても `RunSession()` が要求を見なければ止まらない。却下。
- **スロット単位の上限時間**: 配線は単純だが、スロットが複数あると合計が有界にならず、スケジュール定義に終了ポリシーが混ざる。却下。
- **上限時間の無制限モード、またはキー無しで無制限**: `0=無制限` のような番兵値は禁止（AGENTS.md）。opt-in にすると設定し忘れた無人 batchrun で exe が有界に終わらない。長く待ちたければ大きな値を書く。却下。
- **部分セッションの確定**: `eval_episodes` が Run 内で不揃いになり、同一 Run 内の断面同士が比較不能になる点を新設する。却下。
- **待機中にも中断できる進捗ダイアログ**: 排水に入る前の 1 回の選択で、設定ミスの Run を即落とす場面は足りる。待機中に強制終了して記録を失った事例が出たら足す。延期。
- **CANCEL 側の第 2 期限と強制終了**: 協調停止は本人が中断点へ戻る前提で、Env の `Step()` が返らない故障には安全な強制手段が無い（detach は解放後アクセス、プロセス強制終了は flush の意味論が別）。WAIT の timeout 後も同じ `get()` で待つので両経路に共通の前提であり、未観測の故障なので実例が出たら別 PRD で扱う。延期。
- **スロット単位の deadline や走査順の制御**: 各セッションは自分の pool で並行に進むため、共有 deadline を順に待っても全員が deadline まで進める。順序で損をするスロットは無く、緩和は作らない。却下。
- **手動クローズと予算到達を区別しない**: 第 1 回の裁定では区別しなかったが、Atari で `wait_on_exit=true` を常設すると、設定ミスの Run を即落としたい場面で強制終了に追い込む（最後の flush 以降の metrics と exit code 0 を失う）。`CanVeto()` で区別して聞く形へ改めた。

## Consequences

- 予算末尾に断面が乗る Run は終了がセッション 1 本ぶん（Atari の ε=0 / 400 エピソードで約 20 分）伸び、上限は既定 3600 秒で有界になる。末尾発火を避けるための interval 回避策は不要になる。
- キャンセルしたセッションは scalar を出さず、完了済み採用エピソードの trace 行だけが残る。「trace 行あり・scalar 無し」は起点のクラッシュと同型になるため、クラッシュとの区別は実行ログ（`cancelling ... reason=close|config|timeout` と `session cancelled ... completed=n`）と、Metrics マスタへ既存 json チャネルで出す `eval.[<tag>].session_cancelled` レコードで行う。scalar / trace の契約は触らない。
- `stop_source` はセッション投入のたびに作り直し、`Shutdown()` は終了時専用とする。999（初回発火の抑止）が入るまでは、短い配線 Run で t=0 のセッションが終了時に in-flight になるため、配線 arm 側で `greedy_dist` を切るか `wait_on_exit=false` にする移行依存がある。
- 排水中の例外は `Shutdown()` が再送出し、`OnClose()` から `OnExceptionInMainLoop()`（exit code 1）へ抜ける。残りのスロットは冪等な `OnExit()` の 2 回目呼び出しが排水する。
- 詳細契約と Complexity audit は [PRD 076](../memo/done/076_eval_session_drain_on_exit_10prd.md) に置く。PRD 026 の R2 はこの決定で先行実装され、026 は R1 だけを残して凍結のまま。用語「排水」は [CONTEXT.md](../../CONTEXT.md) を参照する。
