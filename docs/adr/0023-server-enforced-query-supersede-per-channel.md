# metrics query の latest-wins をサーバ側で強制し、その単位を query channel（ブラウザタブ）とする

Metrics Viewer で Run や tag を素早く切り替えると `POST /api/metrics.json` が `503 query_busy` を返し、画面に `Update failed` が出ていた。frontend は既に latest-wins を実装しており、新しい request を出す前に直前の request を `AbortController` で打ち切っている。しかし HTTP 切断は Servlet スレッドから観測できないため、abort された query はサーバ側で最後まで走り続け、query 同時実行数を制限する fair semaphore の permit を保持し続ける。既定の 2 permit が「もう誰も結果を待っていない query」で占有され、最新の query だけが 5 秒待って 503 を受け取っていた。

**frontend が既に表明している latest-wins の意図を、サーバ側で能動的に強制する**ことを決定する。新しい query が到着したら、同じ発行系列に属する古い query をサーバが取り消して permit を明け渡す。**その発行系列の単位を query channel = 1 つのブラウザタブ**とし、channel 識別子と channel 内で単調増加する連番を HTTP ヘッダ（`X-Query-Channel` / `X-Query-Sequence`、いずれも必須）で受け取る。異なる channel は相互に取り消さず、プロセス全体の同時実行枠だけを共有する。取り消された query は `409` + `{"code":"superseded"}` を返す。同時実行枠と supersede は `MetricsQueryCoordinator` が一体で所有し、semaphore・lock・`Statement` を外へ露出させない。

単位をタブにしたのは、frontend の打ち切り単位（タブごとに 1 本）と完全に一致するためである。単位を「サーバ全体で metrics query 1 本」にすると、タブを 2 つ開いた時点で互いを取り消し合い、どちらもグラフを描けなくなる。

停止は query ループ要所の cancel checkpoint を主機構とし、`Statement.cancel` を保険として併用する。checkpoint を主にできるのは、長時間ブロックする単一 DB 文が存在しないためである（最大 50 万行の生データ取得も行の取り出しは Java 側ループ、step の二分探索は 1 行取得 × 約 log2(N) 回、LOD page 読みは 1024 行上限）。

## Considered Options

- **HTTP 切断の検出**（Servlet の非同期処理や書き込み失敗で abort を検知）: Servlet スレッドは応答を書き始めるまで切断を観測できず、本件の query は応答生成前の段階で時間を使う。検出の信頼性が構造的に低いため却下。
- **同時実行枠を増やす / 待ち時間を延ばす**: 放棄された query が permit を握る構造は変わらず、切替の速さに応じて同じ症状が再発する。閾値を動かしただけなので却下。
- **query 自体の高速化**（Run ディレクトリ全走査の除去など）: 占有時間は縮むが、原因である「放棄された query が走り続ける」ことに触れない。独立した改善として別扱いにする。
- **単位をサーバ全体で 1 本にする**: ヘッダ追加が不要で実装は最小になるが、複数タブが相互に取り消し合って両方描画できなくなる。却下。
- **単位を HTTP セッションにする**: frontend が識別子を生成せずに済むが、同じブラウザの別タブは同一セッションになるため複数タブ問題を解決しない。却下。
- **識別子を request ボディのフィールドで送る**: 既存の strict schema 検証にそのまま乗り、API 契約が 1 箇所に閉じる。ただしボディを持たない `GET /api/runs.json` へ将来広げる際に方式を変える必要がある。ヘッダを採用。
- **連番をサーバ側で採番する**: frontend の実装は楽になるが、ブラウザが同一オリジンへ複数接続を張るため到着順が逆転しうる。逆転すると最新の描画が古いデータで上書きされる。却下。

## Consequences

- `POST /api/metrics.json` は 2 ヘッダを必須とする。未指定は `400 invalid_request`。任意ヘッダにすると「未指定＝supersede 対象外」という第 2 の実行経路が恒久化するため、任意にはしない。静的ファイルは同じ jar から配信されるため frontend とサーバのバージョンずれは起きず、互換分岐は持たない。
- `503 query_busy` の契約は残るが、以後これが出るのは真に別 channel が同時実行枠を使い切っている場合だけになる。
- workspace 切替は `SWITCHED` のときだけ旧 epoch の query を取り消す。`NO_OP` と `UNKNOWN` では取り消さないため、誤入力や同じ workspace の再選択で表示中のグラフが壊れない。取り消しは要求を出すだけで完了を待たない（`ingestGate` を保持した HTTP スレッドを止めないため）。
- アプリ終了時は `LoadingThread` 停止より前に全 query を取り消す。長い query が Run lifecycle READ lock を握って `LoadingThread` の 30 秒 join を空振りさせる既存の問題が解消し、新しい待ち時間定数は増えない。
- `getMetrics()` の取得順序が「検証 → 枠 → Workspace lease」へ変わり、枠待ちの間に旧 workspace の gzip 資源を pin しなくなる。
- cancel は専用の非チェック例外で伝え、`MetricsRepository` の既存 3 箇所の広い `catch (Exception)` の先頭で再 throw する。「1 つの Run や series の失敗で応答全体を失わせない」という既存の意図は変更しない。
- channel ごとの最新連番は件数上限付き LRU（固定値 64）で保持する。上限超過で落ちた channel から遅着した request は新規 channel 扱いで実行されるが、被害は「もう誰も見ていない request が 1 回だけ走る」ことに限られる。設定項目は増やさない。
- `GET /api/runs.json` は今回 supersede と同時実行枠のどちらにも入れない。将来入れる場合、ヘッダ方式のまま同じ仕組みを適用できる。
- 詳細設計は `docs/memo/049_metrics_query_supersede_10prd.md`。
