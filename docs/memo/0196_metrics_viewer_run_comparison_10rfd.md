# RFD0196: MetricsViewer で Run を比べる作業に要る機能を足す（設定表、Run・tag の絞り込み、tag の情報パネル、Plotly の同梱）

> 2026-11-22 起票（anet-create-prd、Claude Code）。題材は課題 1 件（§10 の対応表）。グリルで、ゴールと非ゴール（§2）、設定表に出すキー（D1）、設定の読み口（D2）と読み方（D3）、表示の切り替え（D4）、統計表の範囲（D5）、見出しの統計（D6）、絞り込みの一致の規則（D7）と一括操作（D8）、Plotly の同梱（D9）をユーザーが選び、細部 16 件（§5.2）は推奨の既定で埋めた。D10〜D26 は、規約と決定から導いて一覧で示し、ユーザーが戻さなかった。グリルの途中からプランモードで進め、承認を受けてから起草した。別モデルの検討は、ユーザーが入れないと選んだ。起草のあとの未決 6 件（§8）は、ユーザーが推奨の既定でまとめて埋めた（D24 の名前と D25 の送り方の直し、D27〜D30。§5.3）。レビュー（3 往復で OK）の指摘で見つかった未決 2 件も、ユーザーが推奨の既定で埋めた（D31・D32。§5.4）。

> 関連: 設計 [210](../design/210_metrics_viewer.jp.md)・[030](../design/030_user_guide_analysis.jp.md)・[160](../design/160_applications_and_tools.jp.md)・[140](../design/140_observability.jp.md)、ADR [0015](../adr/0015-metrics-cache-disposable-derivative.md)・[0023](../adr/0023-server-enforced-query-supersede-per-channel.md)・[0052](../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md)・[0089](../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md)・[0090](../adr/0090-query-channel-is-per-tab-and-query-kind.md)、[111_metrics_viewer_tag_sets_10prd.md](0111_metrics_viewer_tag_sets_10rfd.md)、[049_metrics_query_supersede_10prd.md](done/0049_metrics_query_supersede_10rfd.md)、[096_metrics_step_axis_interval_10prd.md](0096_metrics_step_axis_interval_10rfd.md)、[154_scalar_threshold_rate_10prd.md](0154_scalar_threshold_rate_10rfd.md)、[108_metrics_tag_numbering_10prd.md](0108_metrics_tag_numbering_10rfd.md)

## 1. 問題

### 1.1 経緯

2026-11-21 の会話で、MetricsViewer に足りない機能と問題点を洗い出した。ユーザーはその中から、設定の比較、Run・tag の検索、グラフごとの統計、オフラインでの動作の 4 つを選び、課題と RFD の数を増やさないよう 1 件の課題（[issue-20261121-201669](../management/issues/issue-20261121-201669.md)）にまとめた。

Run のグラフを重ねて比べる作業は、比べる前に各 Run の設定を確かめ（[030](../design/030_user_guide_analysis.jp.md) の §4.1）、両方にデータがある共通の `exp_step` の範囲で水準・安定性・傾向を読む（同 §4.3）手順になっている。今の Viewer では、この 2 つを画面の中で済ませられない。

### 1.2 今のコードで確かめたこと

| 部分 | 事実 | 根拠 |
|---|---|---|
| 設定 | Viewer は Run の設定を出さない。Run の情報パネルに出るのは取り込みの状態・最大 step・tag 数と隔離の数・点数の合計・取り込みと tag の問題だけである | [210](../design/210_metrics_viewer.jp.md) §3.2 の `RunInfoPanel` の行、[metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 2083〜2095 行 |
| 設定 | 設定の差分を出せるのは CLI の `inspect_run.py config <run…> --diff` だけである | [inspect_run.py](../../apps/tools/inspect_run.py) 273 行、[030](../design/030_user_guide_analysis.jp.md) §6.6 |
| 検索 | Run 一覧は Run 名の降順、tag 一覧は選んだ Run の tag を合わせた名前順で、絞り込みは `Selected Only` だけである。文字を入れる欄はタグセットの名前の入力だけである | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 2198・2309 行、[index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html) 39〜68 行 |
| 統計 | グラフの見出しの `Min / Max / Avg / Std` は、選んだ Run 全部の TagStats を合わせた全期間の値である。Run ごとの値も、表示中の範囲の値も出さない | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 1424〜1437・3312 行 |
| オフライン | 画面は Plotly を `https://cdn.plot.ly/plotly-2.27.0.min.js` から読む。repo にも jar にも写しは無く、`integrity` 属性も無い。設計文書は完全にオフラインの環境では取り方を別に用意する必要があると書いている | [index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html) 10 行、[210](../design/210_metrics_viewer.jp.md) §11.5、[160](../design/160_applications_and_tools.jp.md) §6 |

## 2. ゴールと非ゴール

### 2.1 ゴール

- G1: 選んだ Run の設定を表に並べ、Run 間で違う行を見分けて、その行だけに絞れる。キー名の文字でも絞れる。
- G2: Run 一覧と tag 一覧を文字で絞り込める。
- G3: グラフごとに、Run ごとと全体の統計を表で読める。
- G4: インターネットにつながらなくてもグラフを描ける。

G1〜G4 は、題材の課題の 4 つの部分に当たり、どれも片付くと課題が片付く。グリルの初めに合意した（G1 の、キー名の文字で絞る部分は、合意の返事でユーザーが足した）。

### 2.2 非ゴール（退けた案と、本当の制約か）

| 非ゴール | 退けた案 | 本当の制約か |
|---|---|---|
| 設定以外の json（tag の定義など）・ログ・`dot/` の表示 | 設定表と同じタブに、Run フォルダの json とログを並べる | 制約ではない。別の関心で、別の課題か RFD で足せる |
| seed をまとめた帯の描画 | 統計表の全体の行に合わせて、seed の平均と範囲を線にする | 制約ではない。別の関心である |
| あとから系列を足す仕組み（EMA の後付けなど） | 統計表に、範囲の EMA を足す | 制約ではない。[0154_scalar_threshold_rate_10rfd.md](0154_scalar_threshold_rate_10rfd.md) も同じ仕組みを待つ別の関心である |
| キャッシュのディスク使用量 | 設定の読み口をキャッシュにして、キャッシュの扱いを一緒に見直す | 制約ではない。別の関心である |
| 横軸の種類の表示 | 統計表の `View` の列の組に軸の名前を出す | 制約ではない。[0096_metrics_step_axis_interval_10rfd.md](0096_metrics_step_axis_interval_10rfd.md)（[issue-20261103-52bda0](../management/issues/issue-20261103-52bda0.md)）の範囲である |
| 設定表の上書き層や、使っていない名前空間の区別 | モジュールが読んだキーに「実効」の印を付ける | 制約ではない。ユーザーが要らないと選んだ（D1） |

### 2.3 足す仕組みとゴールへの寄与

| 足すもの | ゴールへの寄与 |
|---|---|
| メインエリアのタブ（`Graphs`・`Config`）（D4） | G1。設定表の置き場になり、グラフの並びを保ったまま行き来できる |
| 設定表（D1・D10〜D13・D16〜D19・D29） | G1 の本体 |
| 設定の API と、フレームワークの設定ファイル様式を読む部品（D2・D3・D23・D27） | G1。今の API は設定を返さない（P4） |
| Run 一覧と tag 一覧の絞り込み欄と、絞り込み中の一括操作（D7・D8・D22・D30・D31） | G2 |
| tag の情報パネル（D5・D14・D15・D20・D21・D32） | G3 |
| 範囲統計の API（D5・D24・D25） | G3 の、表示中の範囲の列の組。全期間の列の組は今の TagStats で足りる（D26） |
| Plotly の WebJar（D9） | G4 |

新しい設定キー、メトリクス、TensorDict キーは足さない。

### 2.4 成功の測り方

- 2 本の Run を選んで `Config` を開き `Diff Only` を押すと、`inspect_run.py config <run_a> <run_b> --diff` が返すキーと同じキーの行だけが残る（§7 の A3・A5）。
- seed だけ違う 3 本を、Run 一覧に語を入れて `Select All` を押す 2 操作で選べる（A6）。
- グラフをズームして tag 名に hover すると、Run ごとの、その範囲の平均と最後の値を読める（A8）。
- ブラウザの外への通信を止めてもグラフが描ける（A1）。

## 3. 前提

グリルで示した前提と、起草で足した前提。確かめの列は、共通の前提（P1〜P12）が「済み」、1 つの論点だけの前提と起草で足した前提（P13〜P26）が「対象外」である。P2 は、ユーザーの返事と実物が食い違ったので、実物を数えて直してから合意した。

| 番号 | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実 | Run の設定は、解決済みの全キーを「キー → 文字列の値」で平らに並べた `config_data` として残る。例の Run（AtariRnd-01 の `run_20261119-202718_mz_tg_g99ne_rr4_s2`）は 1,587 キー。Run フォルダの `config/config_data.txt`・`json/config_data.json` と、Metrics マスタの json 行（キャッシュの `json_lines`）に同じ中身がある | 例の Run のファイルを数えた（2026-11-22）、[210](../design/210_metrics_viewer.jp.md) §8.2 の `json_lines` | 済み |
| P2 | 事実 | `config_data` には、実効値のほかに上書き層（`A1`〜`A3`、`E1`、`M1`）と、使っていない名前空間（`RainbowAgent.*` など）のキーも入る。例の Run で 1,587 キーのうち上書き層 217 と `app.online`・`app.batchrun` 15、モジュールが読んだキー 422。LunarLander-01 の `run_20261119-114710_ll` で 1,009 キーのうち上書き層 66、読んだキー 366。`@` プロファイルと `.$` の選択の行は入らない | 2 Run のファイルを数えた（2026-11-22）、[030](../design/030_user_guide_analysis.jp.md) §6.6 | 済み |
| P3 | 既存部品の働き | CLI の `config --diff` は、値か有無が Run 間で違うキーだけを返し、キーが無いことを値と分けて表す | [inspect_run.py](../../apps/tools/inspect_run.py) 273 行、[030](../design/030_user_guide_analysis.jp.md) §6.6 | 済み |
| P4 | 事実・過去の判断 | Viewer の HTTP API は `json_lines` を読まない。キャッシュはマスタからいつでも作り直せる使い捨ての導出物である | [210](../design/210_metrics_viewer.jp.md) §8.2、[ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md) | 済み |
| P5 | 既存部品の働き | RunInfoPanel は、Run 行の hover（少し待って出る）・右クリック・タッチの長押しで出て、パネルの上にいる間は閉じず、外側の操作・Escape・一覧のスクロールで閉じる。Run 名は 1 回のクリックで全体を選べる。`Copy Name` は Clipboard API が使えないか拒まれたとき、選択範囲のコピーに切り替える | [210](../design/210_metrics_viewer.jp.md) §3.2、[metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 1930〜2142 行 | 済み |
| P6 | 既存部品の働き | グラフの見出しは tag 名・`Log`・percentile の下限と上限のボタン・警告・1 行の統計で、統計は選んだ Run を合わせた全期間の値である | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 1356〜1364・1424〜1437・3312 行 | 済み |
| P7 | 事実 | TagStats は Run×tag ごとに、commit 済みの有効な全点から作る範囲によらない統計で、件数・平均・分散・標準偏差・最小・最大・最初と最後の step・最新値を持つ。統計は LOD から作らない | [210](../design/210_metrics_viewer.jp.md) §8.2 の `tag_stats`、[TagStats.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/TagStats.java) | 済み |
| P8 | 既存部品の働き | メインエリア（`#main-area`）はグラフの並びだけを持つ縦に送る要素で、上に 40px の余白がある。右上の操作は画面に固定した要素（`#floating-controls`）で、この余白の右に重なる。右上の操作（Height・LOD・Reset View・Scroll Lock）はグラフ用で、スクリーンショットモードはサイドパネルとこれらを隠す | [index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html) 72〜98 行、[metrics-viewer.css](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) 521〜527 行、[210](../design/210_metrics_viewer.jp.md) §3.4 | 済み |
| P9 | 守る線 | 操作の部品は即時実行・トグル・巡回・開閉・連続値の 5 種類で、checkbox を使わない。絞り込み中に一括操作を押したときは、絞り込みを解除してから実行する | [210](../design/210_metrics_viewer.jp.md) §3.3 | 済み |
| P10 | 過去の判断 | タグセットを足したときも、`Select All`・`Clear All`・`Selected Only` の動きは変えなかった | [0111_metrics_viewer_tag_sets_10rfd.md](0111_metrics_viewer_tag_sets_10rfd.md) の D3 | 済み |
| P11 | 事実 | Run 一覧と tag 一覧は名前順で、絞り込みは `Selected Only` だけである。LunarLander-01 は 165 Run、1 Run の tag は 189〜237 本。Run 名・tag・設定キーは `run_20261119-202718_mz_tg_g99ne_rr4_s2`・`42_env/10_game_score_mean`・`A1.actor.[train].policy.epsilon.end` のように `[`・`]`・`.`・`/` を含む | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 2198・2309 行、workspace を数えた（2026-11-21） | 済み |
| P12 | 事実 | Plotly は CDN から読み、写しも SRI も無い。リリースの zip は Viewer の jar を同梱し、第三者ライセンスの一覧は Maven の依存から作る `THIRD-PARTY.txt` を `licenses/` に入れる | [index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html) 10 行、[release.yml](../../.github/workflows/release.yml) 241〜248 行 | 済み |
| P13 | 事実 | 741 Run のうち 738 Run に `json/config_data.json` と `config/config_data.txt` の両方がある。残る 3 Run はどちらも無い。キャッシュを 60 件抜き出すと、どれも `json_lines` に `config_data` の行があり、状態は `ready` だった | workspace の全 Run を数えた（2026-11-22） | 対象外 |
| P14 | 事実 | TXT と JSON は、Runner が同じ設定の中身から続けて書く。TXT は解決した順に `key = value` を 1 行ずつ、エスケープなしで書き、Windows では行末が CRLF になる。JSON はキーの辞書順で、値は文字列なので数の丸めを受けない。全 738 Run で、キーの集合と値が一致した | [RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 390〜391 行、[config.cpp](../../src/core/base/src/config.cpp) 569〜576 行、全 Run を突き合わせた（2026-11-22） | 対象外 |
| P15 | 事実 | CLI の TXT の読み込みは、最初の `=` で分けて前後の空白を除く。`java.util.Properties` は `\` をエスケープとして読むので、Windows のパスを壊す | [inspect_run.py](../../apps/tools/inspect_run.py) 178〜187 行、[Java SE 17 の `Properties.load`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Properties.html#load(java.io.Reader)) | 対象外 |
| P16 | 事実 | グラフの幅は、グラフのブロックの幅を測って決める（最小 1）。表示していないブロックは幅が 0 と測られるので、見えるようにしたときに寸法の合わせ直しが要る | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 1589〜1591 行（幅の測り方）・1396〜1405 行（幅の合わせ直し） | 対象外 |
| P17 | 事実 | C++ の読み込み側 `Properties::Load` の行の規則は、§6.3 の表から、R1 の UTF-8 として読めないバイト列の扱い（C++ はバイト列のまま扱う、本書は失敗にする。D19）と、R5 の `$include` の扱い（C++ は取り込む、本書は失敗にする。D18）を除いたものである。この規則で全 738 Run の TXT を読むと、JSON と一致した。`$` で始まる設定キー（2026-07 の ImageCls の Run の `$net.…`）は、`$include` でなければ普通のキーとして読む | [config.cpp](../../src/core/base/src/config.cpp) 669〜848 行、全 Run を規則で読んだ（2026-11-22） | 対象外 |
| P18 | 事実 | 全点の表 `scalars` の主キーは `(tag_id, ordinal)` で、step の索引は無い。LOD は区間ごとに件数・最小・最大・平均・最後の値を持ち、標準偏差に要る二次モーメントを持たない。例の Run で最大の tag は 234,373 点、中央値は 6,000 点。最大の tag の後半の範囲の集計は 14.7 ms（読み込み済み）、全点の初回は 409 ms だった | [210](../design/210_metrics_viewer.jp.md) §8.2、例の Run のキャッシュを測った（2026-11-22） | 対象外 |
| P19 | 事実 | CLI の `--config-key` と `--metric` は、`*` と `?` だけを特別な文字とし、`[` `]` を文字どおりに照合する | [inspect_run.py](../../apps/tools/inspect_run.py) 216〜230 行 | 対象外 |
| P20 | 既存部品の働き | 今の一括操作は、Runs の `Select All` が選択を全 Run にし、`Select Latest` が最新の 1 Run だけを選び、Tags の `Select All`・`Clear All` が一覧の tag を足す・外す。どれも先に `Selected Only` を解除する | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 2292〜2300・2362〜2375 行 | 対象外 |
| P21 | 事実 | Maven Central の `org.webjars.npm:plotly.js-dist-min` に 2.27.0 は無い。2.x は 2.35.3 まで、3.x は 3.0.1 まである。`licenses/` は Maven の依存でないライブラリのライセンスを、1 ライブラリ 1 ファイルで手で置く（8 件） | Maven Central を検索した（2026-11-22）、`licenses/` を数えた（2026-11-22） | 対象外 |
| P22 | 過去の判断 | step の範囲は、LOD の L1 の `step_first` の二分探索で序数の範囲へ写す。step の補助の索引は持たない | [ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md) | 対象外 |
| P23 | 用語 | （グリルの時点の定義。本書の D25 で直した。§4）query channel は 1 つのブラウザタブに対応する問い合わせの系列で、同じ channel の新しい問い合わせが古いものを取り消す。違う channel どうしは取り消さず、プロセス全体の同時実行枠だけを共有する | [ADR 0023](../adr/0023-server-enforced-query-supersede-per-channel.md)、[CONTEXT.md](../../CONTEXT.md) の「query supersede」（「query channel」の項目は本書で直した） | 対象外 |
| P24 | 事実 | `localStorage` に保存する状態は 12 件と決めて列挙している | [210](../design/210_metrics_viewer.jp.md) §7 の `localStorage` の表 | 対象外 |
| P25 | 事実 | 見出しの統計を確かめている既存のテストは 2 つある（[MetricsPlotPlaywrightTest.java](../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsPlotPlaywrightTest.java) 418〜420 行、[OutlierRangePlaywrightTest.java](../../src/apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/OutlierRangePlaywrightTest.java) 425 行） | `src/apps/metrics-viewer/src/test` を `graph-stats` と `Min .*/ Max` で引いた（2026-11-22、出力を切らずに 3 行） | 対象外 |
| P26 | 過去の判断 | Run フォルダの出し入れで Run を登録・解除する運用のため、キャッシュの接続は要求ごとに短く開いて閉じる（Windows では開いているファイルを動かせない） | [ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md) | 対象外 |

## 4. 用語

[CONTEXT.md](../../CONTEXT.md) に 1 語を足し、1 語の定義を直した。

- **query channel**（定義を直した）: 「1 つのブラウザタブに対応する」発行系列から、「1 つのブラウザタブの、問い合わせの種類（グラフの描画、範囲統計）ごとの」発行系列に直した（D25）。取り消しは同じ channel の中だけで起き、違う channel はプロセス全体の同時実行枠だけを共有する決まり（P23）は変えない。

- **範囲統計**: 1 つの scalar tag で、指定した step の閉区間に入る commit 済みの有効な全点を対象とする正確な統計。TagStats と同じ項目を持ち、LOD から作らない。TagStats は範囲によらない全期間の統計なので、別の語にする。表示に合わせて数え直す値だが、範囲を決めるのは step の閉区間で、表示の状態ではないので「viewport stats」と呼ばない。

本書の中だけで使う説明の語（用語集には足さない）。既存の語に、Run を列に並べた設定の表と、tag の情報を出すパネルを指すものが無いので名を付けた。画面の部品の呼び名で、ほかの文書で概念として使う語ではないので、用語集には足さない。

- **設定表**: `Config` タブに出す、行が設定キー、列が選んだ Run の表。
- **tag の情報パネル**: グラフの tag 名から出す、tag 名・`Copy Name`・統計表を持つパネル。RunInfoPanel の tag 版である。

## 5. 確定事項

2026-11-22 のグリルで決めた。D1〜D9 はユーザーが案を比べて選んだ（D1 と D3 は、ユーザーの指示）。D10〜D26 は、規約と決定から導いて一覧で示し、ユーザーが戻さなかった。細部 16 件は §5.2 に、ユーザーがまとめて採った推奨の既定として書いた。D27〜D30 と、D24 の名前・D25 の送り方は、起草のあとの未決について、ユーザーがまとめて採った推奨の既定である（§5.3）。D31・D32 は、レビューの指摘で見つかった未決について、ユーザーがまとめて採った推奨の既定である（§5.4）。

| 番号 | 決定 | 理由と前提 |
|---|---|---|
| D1 | **設定表に出すキー。** `config_data` のキーをすべて出す。上書き層のキーや、使っていない名前空間のキーも、ほかのキーと同じに並べ、区別も印もしない | ユーザーの指示（2026-11-22）「`@` プロファイルを使っていないせいで出るのは気にしなくて良い」。P2 のとおり、上書き層と使っていない名前空間のキーが行として並び、gamma を変えた 2 本では `A1.learner.gamma` と `DefaultDQNAgent.learner.gamma` の 2 行が違う行になる。依存: P2 |
| D2 | **設定の読み口。** Run フォルダの `config/config_data.txt` を読む | ユーザーの選択。中身は JSON と同じで（P14）、CLI と同じファイルであり、フレームワークの本体の設定ファイル様式に合う。キャッシュは作り直し中に出せない（P4・P13）。依存: P4・P13・P14 |
| D3 | **TXT の読み方。** フレームワークの設定ファイル様式として、C++ の読み込み側 `Properties::Load` の行の規則（§6.3）で読む。C++ と違うのは、R1 の読めないバイト列（D19）と R5 の `$include`（D18）の 2 つだけである。書き出し側（`ToPropertiesString`）の形には合わせない | ユーザーの指示（2026-11-22）「設定ファイルの様式は読み込み側が正本。書いたファイルを読むのではなく、フレームワーク様式の TXT を読む」。P17 のとおり、この規則で今の全 Run が JSON と一致する。`java.util.Properties` は使えない（P15）。依存: D2・P15・P17 |
| D4 | **表示の切り替え。** タブ `Graphs`・`Config` を、右上の操作と同じく画面に固定した要素として、メインエリアの上の 40px の余白の左に重ねて置く。グラフと一緒に送られない。[210](../design/210_metrics_viewer.jp.md) §3.3 の部品の種類に「タブ」を足す | ユーザーの選択。今の 40px の余白に収まり縦の場所を増やさず、送っても消えないので今どちらを見ているかが一目で分かる（置き方は、この選択の理由から導いた）。§3.3 の 5 種類に無い形なので規約に足す。依存: P8・P9 |
| D5 | **統計表の範囲。** 全期間と表示中の範囲の両方を出し、列の組で分ける（列の組は `All` と `View`。行は各 Run と全体）。`View` はグラフの今の横軸の範囲の範囲統計で、server が全点から数える。`All` は TagStats から出す | ユーザーの選択。全期間と表示中の値を見比べる需要はほぼ無く、列で分けると縦横がそろう。030 §4.3 の同じ範囲で読む作業に効く。範囲の標準偏差は LOD から組めないので、全点から数える（P18）。依存: P7・P18 |
| D6 | **見出しの統計。** グラフの見出しの 1 行の統計（`.graph-stats`）だけを消す。tag 名・`Log`・percentile の下限と上限のボタン・警告は残す | ユーザーの選択。同じ値は tag の情報パネルの全体の行の `All` の列で読める（D5）。合わせた値を Run の値と読み違えるもと（課題の影響）を消す。依存: D5・P6 |
| D7 | **文字の絞り込みの一致の規則。** Run 一覧・tag 一覧・設定表のキーの 3 か所で同じ規則にする。入力を空白で区切った語のすべてを、大文字小文字を区別しない部分一致で含む行を残す。空の入力はすべての行を残す。語を区切る文字は D31 による | ユーザーの選択。キーと tag に `[`・`]`・`.`・`/` が多く（P11）、正規表現では逃がしが要る。glob では部分一致にも `*` が要る（P19）。依存: P11・P19 |
| D8 | **絞り込み中の一括操作。** 文字で絞り込んでいる間の一括操作は、当たる行だけに効かせ、文字の絞り込みは残す。Runs の `Select All` は当たる Run を選択に足す。`Select Latest` は当たる Run のうち最新の 1 つだけを選ぶ。Tags の `Select All`・`Clear All` は当たる tag だけを足す・外す。`Selected Only` は今と同じく一括操作の前に解除する。文字の入力が空なら、今と同じ動きになる。210 §3.3 の規約に、文字の絞り込みの扱いを書き足す | ユーザーの選択。G2 は欲しい群を探して選ぶためで、今の規約のままでは見つけた行を 1 つずつ選ぶ手間が残る。依存: D7・P9・P10・P20 |
| D9 | **Plotly の同梱。** `org.webjars.npm:plotly.js-dist-min` を Maven の依存にして jar に含め、画面は CDN から読まない。2.27.0 の WebJar は無いので版を上げる（§5.2 の 1） | ユーザーの選択。repo が増えず、ライセンスが `THIRD-PARTY.txt` に自動で載る。依存: P12・P21 |
| D10 | **値の比べ方。** 値は文字列のまま比べる（`1,048,576` の桁区切りもそのまま） | P1 のとおり値は文字列で、CLI も文字列で比べる（[inspect_run.py](../../apps/tools/inspect_run.py) 296 行）。依存: P1・P3 |
| D11 | **違う行。** 選んだ Run の間で、値か有無が違うキーの行を違う行とする。キーが無いセルは、値（空文字を含む）と見分けられる形で出す | CLI の `--diff` と同じ判定にそろえる。依存: P3 |
| D12 | **設定表の列。** サイドパネルで選んだ Run を列にする | G1 による |
| D13 | **違う行だけに絞る操作。** 名詞句のトグルボタン `Diff Only` にする。スイッチや checkbox は使わない | 210 §3.3 の規約（P9） |
| D14 | **tag の情報パネルの `Copy Name`。** tag 名をコピーする。RunInfoPanel と同じコピーの経路（Clipboard API が使えないか拒まれたときは選択範囲のコピー）を使う | P5 |
| D15 | **全体の行。** `All` も `View` も、Run ごとの件数・平均・分散を件数で重みを付けて合わせる。最小・最大は Run ごとの最小・最大から取る。`Last` は空にする | 今の見出しの統計の合わせ方（[metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 3312 行）と同じ。最新値は Run ごとの値で、合わせる意味が無い。依存: D5・P6 |
| D16 | **ファイルが無い Run。** `config/config_data.txt` が無い Run は、列に「設定なし」と出す | 列は選んだ Run（D12）で、無いことを黙って隠さない。依存: D12・P13 |
| D17 | **設定表の間の右上の操作。** `Config` を出している間は、グラフ用の操作（Height・LOD・Reset View・Scroll Lock）を隠す | グラフ用の操作で、スクリーンショットモードも同じものを隠している。依存: P8 |
| D18 | **`$include`。** `$include` の行がある Run は、その列をエラーにする。Viewer は取り込みをしない | 対応しない入力を黙って飛ばさない（[AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」）。今の Run の TXT に `$include` の行は無い（P17）。依存: D3・P17 |
| D19 | **読めない Run。** C++ がエラーにする行（§6.3 の R8〜R10）がある Run、UTF-8 として読めないバイト列がある Run（R1）、読み出しに失敗した Run は、その列をエラーにし、ほかの Run は出す | R8〜R10 は C++ と同じ判定にする（D3）。R1 は C++ と違う。C++ はバイト列のまま扱うが、Java は文字に直して表に出すので、読めないときの扱いを決める必要がある。フレームワークの設定ファイルは UTF-8 で（読み込みが UTF-8 の BOM を飛ばす。[config.cpp](../../src/core/base/src/config.cpp) 721〜727 行）、読めないバイト列を置換文字に替えると黙って値を替えることになる（[AGENTS.md](../../AGENTS.md) の「設定値の扱い」）。1 Run の欠けや破損でほかの Run を隠さないのは、CLI と同じ（[030](../design/030_user_guide_analysis.jp.md) §6.7）。依存: D3・P17 |
| D20 | **tag の情報パネルの出し方。** RunInfoPanel と同じにする。グラフの tag 名に hover すると少し待って出る。右クリックでも出て、タッチでは長押しで出る。パネルの上にいる間は閉じない。外側の操作・Escape・メインエリアのスクロールで閉じる | ユーザーの案「Run リストと同じように」と P5 |
| D21 | **帰結。** ズームしていないグラフでは、`All` と `View` の列の組が同じ値になる | D5（ズームしていなければ、表示中の範囲が全期間になる） |
| D22 | **絞り込みの対象。** Run 一覧は Run 名、tag 一覧は tag のキー、設定表は設定キーだけで照合する。値では照合しない | D7 と、ユーザーの発言（2026-11-22）「設定キーで簡易検索」 |
| D23 | **設定の API。** `POST /api/config.json`。契約は §6.2 | 210 §9.1・§9.4 の形（閉じた本文、`runIds` の受け方、処理の間の workspace の固定）にそろえる。依存: D2・D3・D16・D18・D19 |
| D24 | **範囲統計の API。** `POST /api/range-stats.json`。契約は §6.2 | 210 §9.3 の範囲の決まりと `availability` の決まりにそろえる。項目は TagStats と同じにする（P7）。名前は、グリルで示した `/api/stats.json` から、起草のあとの未決（U1）で直した。用語集の「範囲統計」にそろえ、TagStats と紛れないようにする。依存: D5 |
| D25 | **問い合わせの打ち切り。** 範囲統計の問い合わせは、グラフの問い合わせを打ち切らない。範囲統計は、同じタブでもグラフとは別の query channel で送る。用語集の「query channel」を、タブと問い合わせの種類ごとの系列に直す | 同じ query channel の新しい問い合わせは古いものを取り消すので（P23）、同じ単位で送るとパネルを開くたびにグラフの読み込みが止まる。違う channel どうしは取り消さないので、channel を分ければ今の取り消しの仕組みのまま満たせる。[ADR 0023](../adr/0023-server-enforced-query-supersede-per-channel.md) の、channel の単位を 1 つのブラウザタブとした判断と食い違うので、その単位を [ADR 0090](../adr/0090-query-channel-is-per-tab-and-query-kind.md) で改め、ADR 0023 の先頭に注記を足した。サーバ側の supersede の規則（ADR 0023 の残りの判断）は変えない。用語集の定義も直す（U2）。依存: D24・P23 |
| D26 | **`All` の列の組。** `/api/runs.json` の TagStats から出す。新しい API は要らない | P7・D5 |
| D27 | **設定のファイルの開き方。** 設定の API は、`config/config_data.txt` を要求ごとに開いて読み、閉じる。開いたまま持たない | Run フォルダの出し入れで Run を登録・解除する運用を妨げない（P26）。依存: D2・P26 |
| D28 | **英訳。** `*.en.md` は実装で直さず、`anet-translate-docs` で作り直す | [0194_config_key_full_names_in_messages_10rfd.md](0194_config_key_full_names_in_messages_10rfd.md) の D15 と同じ扱い。英訳は日本語版の更新を追って作り直す |
| D29 | **設定表の列の状態。** §6.1 の列の状態の表のとおりに出す。ファイルが無い Run は「設定なし」、エラーと作業セットに無い Run は `⚠` を出し、内容を title に入れる | tag 一覧の警告の `⚠` と title の形（[metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 2322〜2328 行）にそろえる。依存: D16・D18・D19 |
| D30 | **絞り込みの文字と workspace の切り替え。** workspace を切り替えても、Runs と Tags の絞り込みの文字は残す | 絞り込みは選択でなく見え方で、workspace の切り替えが配り直す Run の選択と色（[030](../design/030_user_guide_analysis.jp.md) §3 の `Workspace` の行）とは別である。依存: D7 |
| D31 | **語を区切る文字。** D7 の絞り込みは、JS の正規表現の `\s` が当たる文字すべてを語の区切りにする。半角の空白・タブ・改行の類に加えて、全角の空白（U+3000）、改行しない空白（U+00A0）、U+2000〜U+200A の空白、U+FEFF などを含む。続く区切りは 1 つとして扱い、前後の区切りは無視する | IME で語のあいだに全角の空白が入っても、半角の空白と同じ行が残るようにする。依存: D7 |
| D32 | **統計表のセルの中身。** 選んだ Run はどれも行を出す。セルは §6.1 の「統計表のセル」の表に上から当てる。その tag を持たない Run は `All`・`View` を `-` にする。ただし、取り込みが済んでいない Run（`ingest.state` が `pending` か `converting`）は、tag がまだ無くても空にする。範囲統計の `availability` が `not_found`・`empty` の Run の `View` は `-`、`pending` の Run と応答待ちの Run の `View` は空にする（取り込みが進んでから問い合わせ直したときに埋まる）。全体の行は、どれかの Run の同じ組のセルが空なら空、件数の合計が 0 なら `-` にする | 選んだ Run の行の並びを、tag の有無で変えない。`-` は値が無いこと、空はまだ分からないことを表し、§5.2 の 12（範囲に点が無い Run は `-`）と、問い合わせが済むまでは空にする決まりにそろえる。取り込みが済んでいない Run の tag の無さと、一部の Run が済んでいない全体の行は、まだ分からないものとして空にする（レビューの指摘を受け、この理由から当てる順を導いた）。依存: D5・D15・D26 |

### 5.1 選ばなかった形

| 形 | 扱い | 理由 |
|---|---|---|
| モジュールが読んだキーに「実効」の印を付ける、読んだキーだけに絞るトグルを置く | 不採用（D1） | ユーザーが要らないと選んだ |
| Run フォルダの `json/config_data.json` を読む | 不採用（D2） | 新しい読み込み部品が要らず、書式に強いが、CLI と読む元が分かれる。フレームワークの本体の形式ではない |
| キャッシュの `json_lines` を読む | 不採用（D2） | 取り込み待ちと作り直し中の Run は出せない。キャッシュの状態ごとの応答を決める必要がある（P4・P13） |
| 値を数として比べる（`1.00` と `1.0` を同じ値とする） | 不採用（D10） | CLI と違う行の判定が分かれる。数でない値と数の値で比べ方が変わる。本当の制約ではなく、CLI とそろえる選択である |
| 書き出し側の形（各行を最初の ` = ` で分け、空白を除かない）で読む | 不採用（D3） | 読み込み側が様式の正本である |
| `java.util.Properties` で読む | 不採用（D3） | `\` をエスケープとして読み、Windows のパスを壊す（P15） |
| トグルボタン `Config` | 不採用（D4） | 今どちらを見ているかを押下色だけで示す |
| 別ページ（`/config.html`） | 不採用（D4） | 選んだ Run を URL などで渡す仕組みが要り、選び直しを追わない |
| グラフに重ねるパネル | 不採用（D4） | 既存のパネルとの重ね順を決める必要がある |
| 全期間だけ | 不採用（D5） | ズームした範囲で比べられない |
| 表示中の範囲だけ | 不採用（D5） | 全期間と並べて見られない |
| 全期間と表示中の範囲を行で分ける | 不採用（D5） | 行が 2 倍になる |
| 見出しの統計を残す、表示中の範囲の値にする | 不採用（D6） | 残すと二重の表示と読み違えが残る。範囲の値にすると、画面の全グラフについてズームやパンのたびに問い合わせる |
| glob（CLI と同じ `*`・`?`） | 不採用（D7） | 部分一致にも `*` を打つ |
| 正規表現 | 不採用（D7） | `[train]` を `\[train\]` と書く必要があり、壊れた式のエラーの表示が要る |
| 文字の絞り込みも解除してから全体に実行する（今の規約のまま） | 不採用（D8） | 見つけた行を 1 つずつ選ぶ手間が残る |
| `static/` に `plotly-2.27.0.min.js` を置く | 不採用（D9） | repo が数 MB 増え、ライセンスを手で足す |
| build のときに CDN から取ってくる | 不採用（D9） | build が CDN につながり、取得用のプラグインと検査値を pom に持つ |

### 5.2 細部（推奨の既定で埋めた 16 件）

2026-11-22 に一覧で示し、ユーザーが「推奨で埋める」を選んだ。

| 番号 | 事項 | 既定 | 理由 | 書いた先 |
|---|---|---|---|---|
| 1 | Plotly の版 | `org.webjars.npm:plotly.js-dist-min:2.35.3` | 2.x の最新。3.x は主版が変わり API の変更が入る（記憶による） | D9・§6.6 |
| 2 | タブの名前と状態 | `Graphs`・`Config`。既定は `Graphs`。開いていたタブをブラウザ（`localStorage`）に覚えさせ、開き直すと戻す | タグセットの選択などと同じ扱い | §6.1 |
| 3 | 設定タブの間のグラフ | 描き直しを止め、`Graphs` に戻ったときに寸法を合わせ直して追いつく | 見えないグラフは寸法を測れない（P16） | §6.1 |
| 4 | 設定表の行の並び | キーの辞書順 | 解決した順は Run ごとに違いうる | §6.1 |
| 5 | 設定表の列の並びと見出し | Run 一覧と同じ並び。見出しは Run 名と Run 色のチップ。キーの列は横に送っても左に固定する | ユーザーが示した MLflow の表と、Run 一覧と同じ見え方 | §6.1 |
| 6 | 違う行とキーが無いセルの見せ方 | 違う行は薄い強調の背景。キーが無いセルは薄い文字の `(none)` | 値の空文字と見分ける | D11・§6.1 |
| 7 | 絞り込み欄と `Diff Only` の置き場 | Runs と Tags は見出し行の下に 1 行の入力欄。設定表は表の上に入力欄と `Diff Only`。入力のたびにすぐ絞る。文字と `Diff Only`（既定は off）はブラウザに覚えさせない | 開き直したら全件から始める | §6.1 |
| 8 | 選んだ Run が 0 件のとき | 設定表に `Select runs` と出す。1 件のときは違う行が無い | 比べる相手が無い | §6.1 |
| 9 | 設定表を読み直す時機 | 選ぶ Run が変わったときと `Reload` のとき。`Auto Reload` の周期では読み直さない | Run の設定は始まったあとで変わらない | §6.1 |
| 10 | 統計表の列 | `All`・`View` のそれぞれに `Count`・`Mean`・`Std`・`Min`・`Max`・`Last`。`Std` は TagStats と同じ母標準偏差。中央値は出さない | 中央値は全期間でも全点の並べ替えが要る | D5・§6.1 |
| 11 | 列の組の見出し | `All` と `View <from>–<to>`（範囲の step を今の数の書式で） | 範囲を表で読める | §6.1 |
| 12 | 範囲内に点が無い Run | `View` の列を `-` にする | 0 と見分ける | §6.1 |
| 13 | パネルを開いたままズームやパンをしたとき | 操作が止まってから 200 ms 後に問い合わせ直す | パンの途中で問い合わせを重ねない | §6.1 |
| 14 | 数の書式 | 今の `formatMetricValue`、件数は `formatCount` | グラフの表示と同じにする | §6.1 |
| 15 | パネルの出る場所 | グラフの tag 名の下。画面に収まるように寄せる | RunInfoPanel と同じ寄せ方 | §6.1 |
| 16 | スクリーンショットモード | タブを隠し、今のタブの中身をそのまま出す | 設定表も貼り付け用の絵にできる | §6.1 |

### 5.3 起草のあとの未決（推奨の既定で埋めた 6 件）

2026-11-22 に、自己点検のあとで一覧で示し、ユーザーが「推奨で埋める」を選んだ。

| 番号 | 事項 | 既定 | 書いた先 |
|---|---|---|---|
| U1 | 範囲統計の API の名前 | `/api/range-stats.json` | D24・§6.2 |
| U2 | query channel の語の定義 | 1 つのブラウザタブの、問い合わせの種類ごとの系列に直す。範囲統計はグラフと別の channel で送る | D25・§4・§6.2 |
| U3 | 設定の API のファイルの開き方 | 要求ごとに開いて閉じる | D27・§6.2 |
| U4 | 英訳 | 実装で直さず `anet-translate-docs` で作り直す | D28・§6.6 |
| U5 | 設定表の列の状態の見せ方 | 「設定なし」と、`⚠` と title | D29・§6.1 |
| U6 | 絞り込みの文字と workspace の切り替え | 文字を残す | D30・§6.1 |

### 5.4 レビューのあとの未決（推奨の既定で埋めた 2 件）

2026-11-22 に、レビューが OK になってから一覧で示し、ユーザーが「推奨で埋める」を選んだ。

| 番号 | 事項 | 既定 | 書いた先 |
|---|---|---|---|
| U7 | 絞り込みで語を区切る文字 | JS の `\s` が当たる文字すべて（全角の空白 U+3000 を含む） | D31・D7 |
| U8 | 統計表のセルの中身 | tag を持たない Run と `not_found` は `-`、`pending` は空、全体の件数 0 は `-`（当てる順は、レビューの指摘を受けて §6.1 の表に書いた） | D32・§6.1 |

## 6. 実装契約

### 6.1 画面

**タブ（D4）。** 右上の操作（`#floating-controls`）と同じく画面に固定した要素として、`#main-area` の上の 40px の余白の左に `Graphs` と `Config` を並べる。`#main-area` を送ってもタブは動かない。押したほうを `.active` と `aria-selected="true"` で示す（210 §3.3 に「タブ」の行を足す。下の「規約の追記」）。開いていたタブを `localStorage` の `anet.metricsviewer.mainTab`（`graphs` か `config`）に保存し、読めない値は `graphs` として読む（`localStorage` の表は 13 件になる。P24）。`Config` の間は、グラフ用の右上の操作を隠し（D17）、グラフの描き直しを止める。`Graphs` に戻ったら寸法を合わせ直し、止めていた間の更新に追いつく（§5.2 の 3）。スクリーンショットモードはタブを隠し、今のタブの中身を出す（§5.2 の 16）。

**設定表（G1）。** 列は選んだ Run を Run 一覧と同じ並びで並べ、見出しに Run 名と Run 色のチップを出す。行は、選んだ Run の設定キーの和集合をキーの辞書順に並べる（§5.2 の 4・5）。キーの列は横に送っても左に固定する。違う行（D11）は薄い強調の背景にし、キーが無いセルは薄い `(none)` にする。表の上に絞り込み欄と `Diff Only` を置く。`Diff Only` が on なら違う行だけを残し、絞り込み欄は D7 の規則で設定キーを照合する（D22）。2 つは両方を満たす行を残す。列の状態は次の表に上から当て、最初に当たった行で扱う（D29）。

| 条件 | 列の見出し | セル |
|---|---|---|
| API の `status` が `ok` | Run 名とチップ | 値か `(none)` |
| `missing`（ファイルが無い） | Run 名とチップと「設定なし」 | 空 |
| `error`（D18・D19） | Run 名とチップと `⚠`。`⚠` の title にメッセージ | 空 |
| `not_found`（作業セットに無い） | Run 名と `⚠`。title にメッセージ | 空 |

違う行の判定（D11）は `ok` の列だけで行う。`ok` の列が 1 つ以下なら違う行は無い。選んだ Run が 0 件なら、表の代わりに `Select runs` を出す（§5.2 の 8）。読み直すのは選ぶ Run が変わったときと `Reload` のときである（§5.2 の 9）。

**絞り込み欄（G2）。** Runs と Tags の見出し行の下に 1 行の入力欄を置き、入力のたびに D7 の規則で行を絞る（D22）。`Selected Only` と両方が効いているときは、両方を満たす行を残す。一括操作は D8 による。入力の文字はブラウザに覚えさせない（§5.2 の 7）。workspace を切り替えても文字は残す（D30）。

**tag の情報パネル（G3）。** グラフの tag 名（`.graph-title`）から D20 のとおりに出す。中身は、上から tag 名（1 回のクリックで全体を選べる文字）、`Copy Name`（D14）、統計表である。統計表の行は選んだ Run（Run 色のチップと Run 名）と、最後に全体（`All runs`）。列は `All` と `View <from>–<to>` の 2 つの組で、組ごとに `Count`・`Mean`・`Std`・`Min`・`Max`・`Last`（§5.2 の 10・11）。`All` は `/api/runs.json` の TagStats から出す（D26）。`View` は、そのグラフの今の横軸の範囲（ズームしていなければ全期間。D21）で範囲統計の API に問い合わせる。全体の行は D15 で合わせる。セルの中身は、次の「統計表のセル」の表による（D32）。

**統計表のセル。** 行の種類と列の組ごとに、表を上から当て、最初に当たった行で扱う。

| 行・列の組 | 条件 | セル |
|---|---|---|
| Run の行・`All` | その Run の TagStats に、この tag がある | TagStats の値 |
| Run の行・`All` | Run の `ingest.state` が `pending` か `converting` | 空 |
| Run の行・`All` | 上のどれでもない（tag を持たない） | `-` |
| Run の行・`View` | 範囲統計の応答がまだ無い | 空 |
| Run の行・`View` | `availability` が `pending` | 空 |
| Run の行・`View` | `availability` が `ok` | 範囲統計の値 |
| Run の行・`View` | 上のどれでもない（`empty`・`not_found`） | `-`（§5.2 の 12） |
| 全体の行・各組 | 同じ組の Run の行に、空のセルが 1 つでもある | 空 |
| 全体の行・各組 | 値を持つ Run の行の件数の合計が 0（どの Run の行も `-`） | `-` |
| 全体の行・各組 | 上のどれでもない | 値を持つ Run の行を D15 で合わせた値（`Last` は空） |

パネルを開いたままズームやパンをしたら、止まって 200 ms 後に問い合わせ直す（§5.2 の 13）。

**見出しの統計（D6）。** `.graph-stats` を消す。見出しは tag 名・`Log`・percentile の下限と上限のボタン・警告になる。

**規約の追記（210 §3.3）。**

- 部品の種類の表に「タブ」の行を足す。ラベルは表示の名前（`Graphs`・`Config`）で、押した 1 つを `.active` と `aria-selected="true"` で示す。
- 「絞り込み中に一括操作を押したときは、絞り込みを解除してから実行する」を、`Selected Only` の絞り込みの決まりと書き分ける。文字の絞り込みは解除せず、当たる行だけに効かせる（D8）。

### 6.2 API の契約

**`POST /api/config.json`（D23）。** 今の workspace の Run の `config/config_data.txt` を読む。

| 項目 | 契約 |
|---|---|
| request | `{"runIds": ["run_…", …]}` だけを受け付ける閉じた本文。`runIds` は必須で、空でない文字列の並びである。空の並びも受け付ける（応答も空）。型・必須・未知の欄の違反は 400 `invalid_request` |
| workspace | 処理の開始時に取った作業セットを、処理の終わりまで使う（210 §9.1 と同じ） |
| ファイル | 要求ごとに開いて読み、閉じる。開いたまま持たない（D27） |
| response | `{"runs": [ … ]}`。request の順に、Run ごとに 1 要素 |
| Run の要素 | `runId`、`status`（下の表）、`message`（`ok` 以外のとき）、`entries`（`ok` のとき。キーと値の組の並びで、キーの辞書順） |

`status` は次の表に上から当て、最初に当たった行の値にする。

| 条件 | `status` | `message` の中身 |
|---|---|---|
| Run が作業セットに無い | `not_found` | Run の id |
| `config/config_data.txt` が無い | `missing` | 探したパス |
| 読み出しに失敗した（I/O） | `error` | パスと失敗の内容 |
| UTF-8 として読めないバイト列がある（§6.3 の R1） | `error` | パス、行番号、規則の番号（R1） |
| §6.3 の R5・R8〜R10 で失敗した | `error` | パス、行番号、規則の番号、その行の内容 |
| 上のどれでもない | `ok` | — |

**`POST /api/range-stats.json`（D24）。** 範囲統計を返す。

| 項目 | 契約 |
|---|---|
| request | `{"series": [{"runId", "tagKey", "fromStep", "toStep"}, …]}`。`fromStep`・`toStep` は 210 §9.3 と同じ必須の閉区間（safe integer）。header の `X-Query-Channel`・`X-Query-Sequence` は §9.3 と同じ |
| 打ち切り（D25） | browser は、範囲統計を、グラフの問い合わせと別の query channel で送る。channel の値は、そのタブのグラフの channel の値と別の、§9.3 の形（1〜128 文字の非 blank の文字列）にする。同じタブのグラフの問い合わせを取り消さず、プロセス全体の同時実行枠だけを共有する（P23）。同じ範囲統計の channel の新しい問い合わせは、古いものを取り消す |
| response（系列ごと） | `runId`・`tagKey`・`fromStep`・`toStep`（request の反映値）、`generation`、`availability`（210 §9.3 の表と同じ決まり）、`ok` のときだけ `count`・`mean`・`variance`・`stdDev`・`minValue`・`maxValue`・`lastValue`（範囲内で序数が最大の点の値）・`minStep`・`maxStep` |
| 数え方 | 範囲に入る commit 済みの有効な全点（L0）から数える。LOD から作らない（P7）。step の範囲は既存の範囲検索と同じく序数の範囲へ写す（P22）。`variance` は母分散（TagStats と同じ） |
| error | 210 §9.3 の表と同じ（400・409・503）。点数による 422 は無い |

### 6.3 TXT の読み方（D3）

[config.cpp](../../src/core/base/src/config.cpp) の `Properties::Load`（709〜848 行）と、そこから呼ぶ `Trim`（669 行）・`NormalizePropertyKey`（678 行）の行の規則を、ファイルの先頭から 1 行ずつ当てる。C++ と違うのは 2 つだけである。R1 は、C++ がバイト列のまま扱うのに対し、UTF-8 として読めないバイト列があれば失敗にする（D19）。R5 は、`$include` を取り込まずに失敗にする（D18）。ほかは C++ と同じ規則である。行の中の規則は R1 から番号順に当て、行を飛ばすか失敗する規則に当たったら、その行の残りの規則は当てない。規則ごとに除く「空白」の文字は、C++ のそれぞれの箇所に合わせて、表の中に書く。どの規則も、ここに挙げない文字（全角の空白 U+3000 など）は除かない。

| 番号 | 規則 |
|---|---|
| R1 | ファイルを UTF-8 として読む。UTF-8 として読めないバイト列があれば、その Run を失敗にする（D19）。ファイルの先頭の UTF-8 の BOM は読み飛ばす。行は LF で分け、行末の CR は R4 で除く |
| R2 | 空の行と、最初の文字が `#` の行は飛ばす |
| R3 | 行の中の最初の `#` から後を除き、続けて最初の `//` から後を除く |
| R4 | 前後の空白・タブ・CR・LF を除く（`Trim`、671 行）。空になれば飛ばす |
| R5 | `$` で始まり、`$` の後の空白・タブを除いた（753 行）先頭 7 文字が、英字の大文字小文字を区別せず `include` なら、その Run を失敗にする（D18。C++ は取り込む）。それ以外の `$` で始まる行は R6 へ進む |
| R6 | 最初の `=` で、キーの部分と値の部分に分ける。`=` が無い行は飛ばす |
| R7 | `=` の直前が `?` なら既定キーとして扱い、キーの部分から `?` を除く |
| R8 | キーの部分から、空白・タブ・LF・VT・FF・CR の 6 文字をすべて除く（`NormalizePropertyKey` の、C ロケールでの空白の判定、685 行）。残った `?` があれば失敗にする（D19） |
| R9 | キーの `:` は 1 つだけ `.` に変える。`:` が 2 つ以上あるか、`:` の前後が空なら失敗にする（D19） |
| R10 | 既定キーで、キーが `.$` で終わるか、`$` そのものか、`run.@` で始まるなら失敗にする（D19） |
| R11 | 値の部分の前後の空白・タブ・CR・LF を除き（`Trim`）、末尾の `;` を続く限り除き、もう一度前後の空白・タブ・CR・LF を除く |
| R12 | キーが空なら、その行は使わない |
| R13 | 既定キーでない行は、同じキーの値を上書きする（後勝ち）。既定キーの行は、同じキーがまだ無いか、既定キーで書かれたものだけを上書きする |

C++ の `Properties::Load` が持つ、選択元の接頭辞の記録（`selection_source_prefixes`）は、設定表に要らないので写さない。C++ の規則が変わったら、Java の読み込みも同じ変更で直す（§6.4）。

### 6.4 変えないものと、意図して変える点

変えないもの:

- Metrics キャッシュの様式と取り込み、`/api/runs.json`・`/api/metrics.json`・`/api/runs/prioritize`・タグセットの API。
- TagStats と LOD の作り方。グラフの描画（MinMax・Mean・Band、Log、percentile、Scroll Lock、凡例、概観の画像）。
- 文字の絞り込みが空のときの、Runs と Tags の一括操作と `Selected Only` の動き（P10・P20）。
- RunInfoPanel の中身と出し方。Run 色の配り方。
- 既存の `localStorage` の 12 件の意味。

どれも本当の制約ではなく、範囲の線である。変える案（キャッシュや取り込みに設定を持たせる、描画の見え方を直す、絞り込みが空のときの一括操作も変えるなど）は、クリーンブレークで変えられるが、G1〜G4 に寄与せず、確かめる範囲を広げる。

意図して変える点:

| 変える点 | 決定 |
|---|---|
| グラフの見出しの 1 行の統計が無くなる | D6 |
| 文字で絞り込んでいる間の一括操作が、当たる行だけに効く | D8 |
| Plotly の版が 2.27.0 から 2.35.3 になり、CDN から読まない | D9・§5.2 の 1 |
| `localStorage` の状態が 1 件増える | §6.1 |
| 210 §3.3 の規約に、タブの種類と文字の絞り込みの扱いが足される | D4・D8 |

**互換のための仕組み。** 残さない。CDN から読む経路や、CDN に届かないときの切り替えは持たない。見出しの統計を出す設定は足さない。C++ と Java の読み込みの規則は、R1 の UTF-8 として読めないバイト列の扱いと、R5 の `$include` の扱いのほかは同じ規則の 2 つの実装として持つ（[ADR 0089](../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md)）。

### 6.5 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ConfigTextReader` | 新規 | フレームワークの設定ファイル様式の TXT を §6.3 の規則で読み、キーと値の組を返す。失敗したら行番号と規則の番号を持つ例外を投げる | server の service |
| `RunConfigService` | 新規 | 開始時の作業セットで、Run ごとに `config/config_data.txt` を読み、§6.2 の状態とキーと値の組を返す | server の service |
| `RangeStatsQuery` | 新規 | 1 つの Run・tag の step の閉区間について、全点から範囲統計を数える | server の service |
| `MetricsViewerController` | 変更 | `/api/config.json` と `/api/range-stats.json` の口を足す | server の view |
| `RunConfigResponse`・`RangeStats` | 新規 | 2 つの API の応答の形 | server の view model |
| `MainTabBar` | 新規 | タブの描画・切り替え・保存と、切り替えに合わせた右上の操作の出し入れ | browser |
| `ConfigTableView` | 新規 | 設定表の描画、違う行の判定、`Diff Only`、キーの絞り込み | browser |
| `KeywordMatcher` | 新規 | D7 の一致の規則。3 か所の絞り込みが共有する | browser |
| `TagInfoPanel` | 新規 | tag の情報パネルの出し入れ、統計表の描画、範囲統計の問い合わせ | browser |
| `RunInfoPanel` | 変更 | コピーの経路を `TagInfoPanel` と共有できる形にする | browser |
| `PlotlyController` | 変更 | 見出しの統計を消し、tag 名から tag の情報パネルを出す | browser |
| `UIController` | 変更 | Runs と Tags の絞り込み欄と、絞り込み中の一括操作 | browser |
| `DataFetcher` | 変更 | 設定と範囲統計の要求。範囲統計は別の query channel で送る | browser |
| `MetricsViewerClientApp` | 変更 | タブの状態と、`combinedStats` を全体の行の合わせ方へ移す | browser |

browser の部品をどのファイルに置くか（`metrics-viewer.js` に足すか、ファイルを分けるか）は、実装計画で決める。消すクラスは無い。

### 6.6 移行対象

**コード。** [index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html)（CDN の行を WebJar の読み先へ、タブ、絞り込み欄）、[metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、[metrics-viewer.css](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.css)、[pom.xml](../../src/apps/metrics-viewer/pom.xml)（WebJar の依存）、[MetricsViewerController.java](../../src/apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerController.java) と §6.5 の新しいクラス。

**テスト。** 見出しの統計を確かめている 2 か所（P25）を、tag の情報パネルの全体の行の `All` の列を確かめる形へ直す。§7 の受入のテストを足す。ほかの既存のテストは変えずに通る。

**設計文書と用語集。**

- [210](../design/210_metrics_viewer.jp.md): §2.8 の「Plotly 2.27」の記述、§3.2 の部品の表（§6.5 の browser の部品）、§3.3（D4・D8 の規約の追記）、§3.4（タブ）、§6 の描画の突き合わせ（582 行）の「header（統計と警告）」、§7 の `localStorage` の表（13 件）、§9.3 の `X-Query-Channel` の行（「1つのブラウザタブに対応する」を、タブと問い合わせの種類ごとへ。D25）、§9 に 2 つの API の節、§11.2 の依存の表（WebJar）、§11.5（CDN の記述を消す）、§12 のテストの構成。
- [030](../design/030_user_guide_analysis.jp.md): §3 の操作の表（タブ、設定表、絞り込み欄、絞り込み中の一括操作、tag の情報パネル）、§3 の percentile の段落の「グラフheaderの統計値」、§3 の LOD の段落（115〜116 行）の見出しの `Min / Max / Avg / Std` の記述、§4.1（設定表での確かめ方）、§4.3（tag の情報パネルの `View` の列）。
- [160](../design/160_applications_and_tools.jp.md): §2.2 の `PlotlyController` の行の `TagStats`、§6 の CDN の記述、§8 の Viewer の UI 変更のテストの行。
- 英訳（`210`・`030`・`160` の `*.en.md`）は実装で直さず、`anet-translate-docs` で作り直す（D28）。
- 用語集（[CONTEXT.md](../../CONTEXT.md) の「範囲統計」の追加と「query channel」の定義の直し）、[ADR 0089](../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md)、[ADR 0090](../adr/0090-query-channel-is-per-tab-and-query-kind.md)、[ADR 0023](../adr/0023-server-enforced-query-supersede-per-channel.md) の先頭の注記は、本書の起草で書いた。
- wiki の [Metrics Viewer の表示粒度](../wiki/mechanisms/metrics_resolution.md)（24 行が ADR 0023 の題を引く）は、本書の wiki への書き戻しで、ADR 0090 で単位を改めたことを足す。

**ライセンス。** `licenses/` には足さない。WebJar は Maven の依存なので、`THIRD-PARTY.txt` に載る（P12・P21）。

**直さないもの。** 過去の記録（実験記録、ADR、保管場所の RFD と実装記録、課題記録の事象）に書かれた CDN と見出しの統計の記述は、当時の記録として残す（過去の記録を書き換えない決まり。[AGENTS.md](../../AGENTS.md) の「クリーンブレーク方針」）。RFD0089 の検証資材（[render_overview_benchmark.cjs](../../src/core/base/testdata/rfd0089/render_overview_benchmark.cjs) と [README](../../src/core/base/testdata/rfd0089/README.md)）は、当時の描画の比べ方を再現する資材なので、CDN の 2.27.0 を読むまま残す。

**検索で当たるファイルの線引き。** 追跡しているファイルの全体を、固定文字列 `cdn.plot.ly`・`plotly-2.27`・`Plotly 2.27` で大文字小文字を区別せずに引くと、本書を除いて 7 ファイルに当たる（2026-11-22）。

| 扱い | ファイル |
|---|---|
| 直す（2） | [210](../design/210_metrics_viewer.jp.md)（219・1073 行）、[index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html)（10 行） |
| 残す・英訳（1） | [210 の英訳](../design/210_metrics_viewer.en.md)（`anet-translate-docs` で作り直す） |
| 残す・過去の記録（2） | [issue-20261121-201669](../management/issues/issue-20261121-201669.md) の事象、[089 の実装記録](done/0089_metricsviewer_comparison_resolution_22impl.md) |
| 残す・検証資材（2） | RFD0089 の `render_overview_benchmark.cjs` と README |

### 6.7 ほかの RFD との順序

- [0096_metrics_step_axis_interval_10rfd.md](0096_metrics_step_axis_interval_10rfd.md)（未実装）: 横軸の種類をグラフ上で分かるようにする。本書の D6 で見出しから統計が消えるので、どちらが先でもよい。096 が先なら、本書は見出しの軸の表示を残したまま統計だけを消す。本書が先なら、096 は統計の無い見出しに軸を足す。096 が横軸を経験数へ切り替える形を入れたら、`View` の範囲はその軸の範囲になる。範囲統計の API が受ける step がどの軸かは、096 の再グリルで扱う。
- [0154_scalar_threshold_rate_10rfd.md](0154_scalar_threshold_rate_10rfd.md)（暫定）: Viewer に系列の導出を足す。どちらが先でもよい。導出した系列を tag の情報パネルの統計表に出すかは、154 の再グリルで扱う。
- [0108_metrics_tag_numbering_10rfd.md](0108_metrics_tag_numbering_10rfd.md)（暫定）: tag の番号を振り直す。本書は tag を変えないので、順序の制約は無い。

## 7. 受入

テストのタグ・テスト名は実装計画で決める。Viewer のテスト（`mvn test`）で確かめる。

| 番号 | 対象 | 確かめること |
|---|---|---|
| A1 | G4・D9 | Playwright で、`localhost` 以外への要求をすべて止めた（`page.route` で中止させた）状態でページを開き、グラフが描け（`path.js-line` がある）、止めた要求が 0 件であるテストがある。`src/apps/metrics-viewer/src/main` を固定文字列 `cdn.plot.ly` で引くと 0 行である。`mvn package` で作った jar に `META-INF/resources/webjars/plotly.js-dist-min/2.35.3/` があり、`THIRD-PARTY.txt` に `plotly.js-dist-min` の行がある |
| A2 | D3 | `ConfigTextReader` のテストが、§6.3 の R1〜R13 のそれぞれに 1 つ以上ある。少なくとも、BOM、CRLF、UTF-8 として読めないバイト列の失敗、`#` と `//` のコメント、行の途中の `#`、`=` の無い行、値の末尾の `;`、`?=` と個別指定の前後の両方の順、キーの中の空白・タブ・VT・FF が除かれ全角の空白（U+3000）が残ること、`:`、`$` と `include` のあいだに空白・タブを置いた `$include` と大文字の `$INCLUDE` の失敗、`$` で始まる普通のキー、R8〜R10 の失敗（行番号と規則の番号を含む）を確かめる |
| A3 | D2・D3 | 実装のときに 1 回、`data/workspaces` の全 Run について、`ConfigTextReader` で読んだ `config/config_data.txt` と `json/config_data.json` の `data` を比べ、比べた Run の数と、キーの集合か値が違う Run の数（期待は 0）を実装計画に書く。比べるのは両方のファイルがある Run で、片方しか無い Run の数も書く。`ConfigTextReader` が失敗した Run は、違う Run と分けて数を書く（期待は 0） |
| A4 | D16・D18・D19・D23 | `/api/config.json` の結合テストが、`ok`（キーの辞書順の `entries`）・`missing`・`error`（`$include` と R8 の失敗で、`message` に行番号を含む）・`not_found` を 1 つの要求の中で返すことと、未知の欄と `runIds` の欠けで 400 を返すことを確かめる |
| A5 | G1・D4・D10〜D13・D17 | Playwright で、2 Run を選んで `Config` を押すと、列が 2 つで見出しに Run 名がある。値が違うキー・片方に無いキー・同じキーを置いた検証の Run で、違う 2 行だけが強調され、`Diff Only` を押すと 2 行だけが残る。片方に無いセルは `(none)` である。絞り込み欄に、大文字を混ぜた 2 語を入れると、両方の語を含むキーの行だけが残る。`Config` の間は Height・LOD・Reset View・Scroll Lock が見えず、`Graphs` に戻ると見え、グラフが描かれている。ページを開き直すと、開いていたタブに戻る |
| A6 | G2・D7・D8・D31 | Playwright で、Run 一覧に 2 語を入れると両方を含む Run だけが残り、語のあいだを全角の空白（U+3000）や改行しない空白（U+00A0）にしても同じ Run だけが残る。`Select All` を押すと、それまでの選択に当たる Run が足され、文字は残る。`Select Latest` は当たる Run の最新の 1 つだけを選ぶ。Tags で語を入れて `Clear All` を押すと、当たる tag だけが外れる。`Selected Only` を on にして `Select All` を押すと、`Selected Only` は off になり、文字の絞り込みは残る。文字が空のときの一括操作は、既存のテストのまま通る |
| A7 | D5・D24 | `/api/range-stats.json` の結合テストが、検証の Run の全点から数えた値（件数・平均・母分散・最小・最大・範囲内の最後の値・最初と最後の step）と一致することを、範囲の端の点を含む場合と含まない場合で確かめる。範囲に点が無い系列は `empty`、作業セットに無い Run は `not_found` である |
| A8 | G3・D5・D6・D14・D15・D20・D21・D32 | Playwright で、2 Run を選んだグラフの tag 名に hover すると tag の情報パネルが出て、tag 名、`Copy Name`、行が 2 Run と `All runs` の統計表があり、列の組が `All` と `View` である。片方の Run にだけある tag のパネルでは、その tag を持たない Run の行があり、`All`・`View` のセルが `-` である。ズームしていないときは 2 つの組が同じ値で、`All` の全体の行が、今の見出しの統計の値（P25 のテストの期待）と同じである。パネルを出したまま、グラフの上をマウスでドラッグして（Plotly の `nsewdrag` の層の上で押して動かして離す、入力の経路）横軸の範囲を狭めると、200 ms 後に `View` の見出しの範囲と値が、その範囲の値に変わる。`Copy Name` を押すと、tag 名がクリップボードに入る（使えない環境では選択範囲のコピーの経路で）。グラフの見出しに `.graph-stats` が無い |
| A9 | D25 | 2 つを確かめる。(1) Playwright で、tag の情報パネルを出したときにブラウザが送る `/api/range-stats.json` の `X-Query-Channel` の値が、同じタブの `/api/metrics.json` の値と違い、パネルを出すたびに同じ値（範囲統計の channel）を使う。(2) 結合テストで、channel A のグラフの問い合わせを流している間に、channel B で範囲統計を問い合わせても、グラフの問い合わせが 409 にならず、channel B で続けて 2 回問い合わせると古いほうが 409 になる |
| A10 | 設計文書 | `docs/design/*.jp.md` を固定文字列 `cdn.plot.ly`・`plotly-2.27`・`Plotly 2.27` で、大文字小文字を区別せずに引くと 0 行である。英訳（`*.en.md`）は作り直すまで旧い記述が残るので、範囲から除く（§6.6）。210 §3.3 にタブの行と文字の絞り込みの扱いがあり、§9 に 2 つの API の節がある。[CONTEXT.md](../../CONTEXT.md) に「範囲統計」がある |
| A11 | 変えないもの | Viewer のテストを `mvn test` で全部流して通る（P25 の 2 か所は直したもので） |
| A12 | 人の確認 | Plotly 2.35.3 で、実際の workspace の Run を開き、MinMax・Mean・Band、Log、percentile、hover、ズームとパン、概観の画像、スクリーンショットモード、Scroll Lock が、2.27.0 のときと同じに見えるかを人が確かめる。人が同じに見えると判断したら合格にする。違いがあれば、実装の完了報告に書き、直すかを開発者が決める |

後の作業の中で確かめる受入は無い。A3 は実装のときに、その時点の `data/workspaces` で 1 回行う。Run を流す受入は無い。

## 8. 決めること（未決表）

U1〜U6 は、2026-11-22 に推奨の既定で決め、確定事項に書いた（§5.3）。U7・U8 は、レビューの指摘で見つかった未決で、レビューが OK になってから推奨の既定で決め、確定事項に書いた（§5.4）。決めずに残る行は無い。

| 番号 | 事項 | 推奨の既定 | 扱い |
|---|---|---|---|
| U1 | 範囲統計の API の名前 | `/api/range-stats.json` にする。用語集の「範囲統計」に名前をそろえ、TagStats と紛れないようにする | 決めた（2026-11-22、推奨の既定。D24）。起草で見つけた。グリルでは `/api/stats.json` として示した |
| U2 | query channel の語の定義 | 用語集の「query channel」を「1 つのブラウザタブの、問い合わせの種類ごとの発行系列」に直す。範囲統計はグラフと別の channel で送る | 決めた（2026-11-22、推奨の既定。D25）。起草で見つけた。今の定義は「1 つのブラウザタブに対応する」で、D25 の送り方と合わない |
| U3 | 設定の API でファイルを開いたまま持たないこと | 要求ごとに開いて閉じる | 決めた（2026-11-22、推奨の既定。D27）。ADR 0015 の運用の決まり（P26）から導いた |
| U4 | 英訳の扱い | 実装で直さず、`anet-translate-docs` で作り直す | 決めた（2026-11-22、推奨の既定。D28） |
| U5 | 設定表の列の状態の見せ方 | §6.1 の表（`missing` は「設定なし」、`error`・`not_found` は `⚠` と title） | 決めた（2026-11-22、推奨の既定。D29） |
| U6 | 絞り込みの文字と workspace の切り替え | workspace を切り替えても文字を残す | 決めた（2026-11-22、推奨の既定。D30） |
| U7 | 絞り込みで語を区切る文字（D7） | JS の `\s` が当たる文字すべてで区切る（半角の空白・タブに加えて、全角の空白 U+3000 なども区切りになる）。IME で語のあいだに全角の空白が入っても、同じ行が残る | 決めた（2026-11-22、推奨の既定。D31）。レビューの指摘で見つかった |
| U8 | 統計表のセルの中身（§6.1） | その tag を持たない Run は、行を出して `All`・`View` を `-` にする。範囲統計の `availability` が `not_found` の Run の `View` は `-`、`pending` の Run の `View` は空（取り込みが進んでから問い合わせ直したときに埋まる）。全体の行で件数の合計が 0 なら `-` | 決めた（2026-11-22、推奨の既定。D32）。レビューの指摘で見つかった。そのあとのレビューの指摘で、取り込みが済んでいない Run と、一部の Run が済んでいない全体の行の扱いを、D32 の理由（空はまだ分からないこと）から導いて足し、当てる順を §6.1 の表に書いた |

## 9. 範囲外

なし。グリルで範囲外にしたもの（§2.2）は、どれも分析の会話で起票しない候補として示し、ユーザーが選ばなかった。

## 10. issue の対応表

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261121-201669](../management/issues/issue-20261121-201669.md) | MetricsViewer で Run を比べる作業に要る機能が足りない（設定の比較、Run・tag の検索、グラフごとの統計、オフラインでの動作） | 120_metrics_viewer | [metrics-viewer.js](../../src/apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) 1424〜1437・2083〜2095・2198・2292〜2300・2309・2362〜2375・3312 行、[index.html](../../src/apps/metrics-viewer/src/main/resources/static/index.html) 10 行、[inspect_run.py](../../apps/tools/inspect_run.py) 273 行 | D1〜D32 | A1〜A12 |

## 11. グリルの材料

決定として扱わない。

- ユーザーの案（2026-11-22）: 設定は MLflow の Parameters の表に倣い、行を設定キー、列を Run とし、差のある行を色分けして絞れるようにする。メインエリアをタブなどでグラフと設定に切り替える。統計は Run 一覧と同じく、グラフの tag 名に hover するとパネルが出て、コピーできる tag 名と `Copy Name`、その下に統計表（行は各 Run と全体、列は統計の項目）を出す。ユーザーは、MLflow の Parameters の表の画像（`Show diff only` と、`AtariEnv.game` が breakout と qbert で違う行が強調された表）を添えた。
- ユーザーは、合意の返事で「上書き層は気にしなくて良いはず。設定プロファイルで上書きしている場合は出ないようにすでになっているはず」と書いた。AI は実物を数え、上書き層のキーが入ることを示した（P2）。ユーザーは「`@` プロファイルを使っていないせいで出るのは気にしなくて良い」と返し、D1 になった。
- 読み口の問いで、AI は JSON を推奨し、ユーザーは TXT と JSON の中身の一致と、TXT を読む手間を尋ねた。AI は全 Run の突き合わせ（P14）と手間の見積もり（20〜30 行とテスト）を示して問い直し、ユーザーは TXT を選んだ。続けて、AI が書き出し側の形で読む案を示したのに対し、ユーザーは読み込み側の規則で読むよう指示した（D3）。
- 統計の範囲の問いで、AI は表示中の範囲だけを推奨し、ユーザーは両方を列で分ける形を選んだ（D5）。
- Plotly の同梱の問いで、AI は `static/` に置く形を推奨し、ユーザーは WebJar で版を上げる形を選んだ（D9）。
