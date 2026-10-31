# 起動中の Runner を OS のテーマ（ライト／ダーク）の切替に追従させる PRD

> 2026-10-31 起票（暫定・999。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍の記述で探し直す。wxWidgets のソースの行番号は 3.3.3 のもの。
> 起点: 2026-10-31 の会話。Runner の起動後に Windows のテーマを切り替えると、画面の一部だけが新しいテーマになった（ユーザーのスクリーンショット 1 枚）。原因を調べ、直し方の 3 案（再起動を案内する／塗り直しを 1 か所に集めて切替が落ち着いてから行う／フレームを作り直す）を比べ、ユーザーは実装で直す 2 つ目の案を選んだ。
> 関連: [PRD094](094_evalpanel_step_history_10prd.md)（Evaluation Trace の配色とテーマへの追従）、[PRD055](done/055_runner_toolbar_10prd.md)（ツールバーのアイコンのテーマへの追従）、[160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §2.1・§3.4・§3.5・§8。

## 1. 目的と問題

Runner は起動した時点の OS のアプリのモード（ライト／ダーク）に合わせて画面を作る。起動したあとに OS の設定でモードを切り替えると、新しいテーマになる部分と古いテーマのまま残る部分が混ざる。2026-10-31 のスクリーンショットでは、タイトルバー・メニュー・ツールバーは新しいテーマになり、pane の見出し・ステータスバー・Logs のログ欄・Evaluation Trace の表のセルは古いテーマのまま残った（§2 の 8）。

設計文書は、Evaluation Trace と Logs がテーマの変更に追従すると書いている（§2 の 9）。今の挙動はその記述に合っていない。直すには、再起動するしかない。長い Run の途中では再起動できないので、Run が終わるまで混ざった画面で作業することになる。

## 2. 現状（2026-10-31 に確認したこと）

1. 起動時: [RunnerApp.cpp:267](../../apps/runner/src/RunnerApp.cpp) が `SetAppearance(Appearance::System)` を呼び、起動した時点の OS のアプリのモードに合わせる。wxWidgets は 3.3.3 をソースからビルドして使う（[040 開発環境構築ガイド](../design/040_development_environment.jp.md) §3.4）。
2. wx の切替の仕組み: wx は 3.3.3 でライト／ダークの切替への対応を足した（wx の `docs/changes.txt` の 303 行）。MSW では、OS が送る設定の変更の通知（`ImmersiveColorSet`）を色の変更として扱い、`wxSysColourChangedEvent` を送る（`src/msw/window.cpp` の 3555〜3562 行、`include/wx/msw/private.h` の 1192〜1204 行）。通知は、アプリが登録した処理を先に、wx の既定の処理を後に回し、そのあとで Windows 自身の既定の処理へ渡す（`src/msw/window.cpp` の 5078〜5094 行）。wx の既定の処理は、アプリのモードが起動時と変わっていれば、既存のウィンドウ（タイトルバー・スクロールバー・標準コントロール）を新しいモードへ切り替え、子のウィンドウへ通知を回す（同 5266〜5300 行、`src/msw/darkmode.cpp` の 782〜786 行）。この切替の処理は wx のウィンドウの protected な仮想関数で（`include/wx/msw/window.h` の 604・648 行）、アプリから部分ごとに呼ぶ公開の API は無い。
3. 通知の時点の色: 色を返す関数は、ダークかどうかをそのつど OS に問い合わせて決める（`src/msw/settings.cpp` の 99〜107 行、`src/msw/darkmode.cpp` の 240〜260・454〜457 行）。wx の `src/msw/textctrl.cpp` の切替の処理（2949〜2968 行）は、切替がまだ終わっていない場合に備えて、新旧両方のモードの既定の文字色と比べている。通知を受けたその場で読んだ色が、切替の前のものである可能性がある。確かめてはいない（§5.3 の段 0）。
4. テストの制約: wx は MSW では、ウィンドウを作ったあとに `SetAppearance` で外観を変えられない（wx の `interface/wx/app.h` の 1171〜1173 行、`src/msw/darkmode.cpp` の 299〜306 行）。GUI テストはテーマごとに別の process で起動して確かめている（[EvalTracePanel_test.cpp:814](../../apps/runner/src/EvalTracePanel_test.cpp)・1292 行、環境変数 `PRD094_UI_DARK`）。1 つの process の中でライトとダークを切り替えるテストは書けない。RunnerApp を起動するテストは、テストの main が外観を指定したあとで RunnerApp の起動の処理を呼ぶ（同 1292 行のあとに 908・1082 行）。ダークを指定すると外観が固定され、起動の処理の「OS に合わせる」指定は効かない（wx の `src/msw/darkmode.cpp` の 299〜306 行）。ライトの指定は何もしない（同 315〜317 行）ので、起動の処理の指定で OS のアプリのモードに従う。OS がダークの機械では、RunnerApp を起動するライトの process は作れない。
5. Runner の今の受け方: どの部分も、通知を受けたその場で色を読み、決め直す。

   | 部分 | 受け方 | 決め直すもの |
   |---|---|---|
   | RunnerFrame | 通知を受ける（[RunnerFrame.cpp:696](../../apps/runner/src/RunnerFrame.cpp)、1352〜1356 行） | Run 制御ツールバーの再生・Step のアイコン（670〜685 行）。実行状態を保って作り直す |
   | pane の見出しと枠 | wxAuiManager が frame への通知を自分でも受ける（wx の `src/aui/framemanager.cpp` の 612〜617・813 行） | Runner 用の配色（[RunnerFrame.cpp:111-117](../../apps/runner/src/RunnerFrame.cpp)）を含む見出し・枠の色 |
   | 4 本のツールバー pane | 各ツールバーが自分で受ける（wx の `src/aui/auibar.cpp` の 2514〜2520 行） | 地の色と、テーマの描き方を使うか（wx の `src/aui/barartmsw.cpp` の 432〜438 行） |
   | Logs | [LogPanel.cpp:295](../../apps/runner/src/LogPanel.cpp) で受け、`UpdateAppearance`（810〜838 行）を呼ぶ | ログ欄の地と文字、既にある行の色、検索欄、ゴミ箱のアイコン |
   | Evaluation Trace | [EvalTracePanel.cpp:156](../../apps/runner/src/EvalTracePanel.cpp) で受け、`UpdateAppearance`（666〜704 行）を呼ぶ | 操作帯の地とアイコン、表のセル・見出し・罫線・選択の色、列幅 |
   | QValue の表 | 受けない | なし |
   | タイトルバー・メニュー・ステータスバー・スクロールバー | wx が受ける（2 の項） | wx の切替の処理による |

6. 作ったときの色が残る部分: wxGrid は作るときに、表の行の無い部分（セルを描かない地）の色を、セルの既定の背景と同じシステムの色に決める（wx の `src/generic/grid.cpp` の 2985〜3001 行）。あとでセルの既定の背景を変えても、MSW ではこの地の色は変わらない（同 9744〜9750 行。変えるのは GTK と Qt だけ）。Evaluation Trace の `UpdateAppearance` は、この地の色を決め直さない（[EvalTracePanel.cpp:675-681](../../apps/runner/src/EvalTracePanel.cpp)）。QValue の表（[QValuePanel.cpp:164](../../apps/runner/src/QValuePanel.cpp)）は、セル・見出し・罫線・地のどれも作ったときの色のまま残る。選んだ行の強調の色は、選ぶ行が変わったときにだけ読む（同 500〜518 行）。wxGrid が作るときに使う色は、セルの背景が WINDOW、文字が WINDOWTEXT、見出しの地が BTNFACE、罫線が GRIDLINES、選択が HIGHLIGHT である（wx の `src/generic/grid.cpp` の 2960〜3004・3222〜3269 行）。
7. 描くたびに色を読む部分と、テーマに合わせない固定の色: Evaluation Trace の報酬と終端の行の色は、描くたびにセルの既定の背景へ薄く混ぜて作る（[EvalTracePanel.cpp:61-74](../../apps/runner/src/EvalTracePanel.cpp)）。DropMerge の View は描くたびにシステムの色を読む（[DropMergeView.cpp:185](../../core/envs/dropmerge1/src/DropMergeView.cpp) など）。テーマに合わせない固定の色は、Train View の枠（デスクトップの色。[TrainPanel.cpp:38](../../apps/runner/src/TrainPanel.cpp)）、QValue のヒートマップの地（黒。[QValuePanel.cpp:55](../../apps/runner/src/QValuePanel.cpp)）、HeatMap の step の文字（黒。[HeatMapPanel.cpp:352](../../apps/runner/src/HeatMapPanel.cpp)）、Atari の重ね書き（黒地に白。[AtariView.cpp:46-47](../../core/envs/atari1/src/AtariView.cpp)）、Env の View が描く盤面の色である。
8. 観測（2026-10-31 のユーザーのスクリーンショット 1 枚）: タイトルバーとメニューがライトなので、ダークで起動してライトへ切り替えたあとと読める。ライトになっていたのは、タイトルバー、メニュー、ツールバーの地とアイコン（黒）、Evaluation Trace の操作帯・アイコン・列見出し、Logs の操作帯と検索欄である。ダークのまま残ったのは、pane の見出しと枠、ステータスバー、Evaluation Trace のセル（暗い緑の行。暗い既定の背景に緑を混ぜた色）・行の無い部分・横スクロールバー、Logs のログ欄である。Evaluation Trace のアイコン・列見出しとセルの既定の背景は、同じ `UpdateAppearance` の中で決めている（[EvalTracePanel.cpp:669-681](../../apps/runner/src/EvalTracePanel.cpp)）のに、テーマが分かれた。1 回の呼び出しでは説明がつかないので、通知が違う状態で複数回届いた可能性がある。どの経路でそうなったかは確かめていない（§5.3 の段 0）。
9. 設計と受入の記述: [160](../design/160_applications_and_tools.jp.md) §3.4（114 行）は Evaluation Trace が「テーマ・フォント・DPI変更には追従」すると、§3.5（142 行）は Logs が「テーマ変更の通知で全行を塗り直す」と書く。[PRD094](094_evalpanel_step_history_10prd.md) §8 の受入 10（416 行）は、開いたままテーマを変えても配色とアイコンが追従することを求めた。その実装の記録は、OS のテーマの切替を実機では確かめておらず、通知を pane へ送るテストで代えた（[094 の 21impl:100](094_evalpanel_step_history_21impl.md)、[22impl:53](094_evalpanel_step_history_22impl.md)）。[PRD055](done/055_runner_toolbar_10prd.md) の受入 10（269 行）は、ツールバーのアイコンの追従を求めた。
10. 対象の OS: Runner は Windows で使っている。Linux・macOS は未検証である（[040](../design/040_development_environment.jp.md) の 39 行）。
11. 既存のテスト: [LogPanel_test.cpp:214-217](../../apps/runner/src/LogPanel_test.cpp) と [EvalTracePanel_test.cpp:814-819](../../apps/runner/src/EvalTracePanel_test.cpp) は、pane へ通知を直接送り、塗り直しを確かめている。GUI テストの実行ファイルは `AnetRLRunner-gui-test`（[apps/runner/CMakeLists.txt:148](../../apps/runner/CMakeLists.txt)）。

## 3. ゴール・非ゴール・成功の測り方

**ゴール**

- G1: Runner を開いたまま OS のアプリのモードを切り替えると、ライト→ダークとダーク→ライトのどちらでも、§5.2 の領域のすべてが新しいテーマになる。見た目は、そのテーマで起動し直したときと同じ（領域ごとの色が同じ）。段 0 で wx の不具合と分かり §8 に記録した部分は除く（D5）。
- G2: 何度切り替えても（ライト→ダーク→ライト→…）G1 が成り立つ。
- G3: 切り替えても、表示の状態（§5.5）が変わらない。
- G4: 1 回の切替で、Runner の外観の決め直しは 1 回だけ走る。届いた通知の数によらない。

**非ゴール**

- テーマに合わせない固定の色（§2 の 7）をテーマに合わせること。消える案は、Runner が描くもの全部をテーマに合わせる案である。本当の制約ではない（色の決め方を変えれば合わせられる）が、起動時からテーマに合わせていない部分で、本書の問題（途中の切替が半端に反映される）とは別の関心である。
- Runner の設定でライト／ダークを選べるようにすること。消える案は、設定のキーで外観をライトかダークに固定する案である。本当の制約ではない（起動前なら選べる）が、別の関心である。
- wxWidgets のソースを手元で直すこと。消える案は、wx の切替の処理の不具合を wx のソースで直してビルドする案である。本当の制約ではない（wx は手元でビルドしている）が、wx の版を上げるたびに当て直しが要る。wx の不具合と分かった部分は §8 に記録する。
- 開いている modal ダイアログ（workspace の選択、エラーの表示）を追従させること。消える案は、ダイアログも決め直す案である。本当の制約ではないが、表示が短く、開き直せば新しいテーマで出る。
- ハイコントラストへの切替。消える案は、ハイコントラストへの切替でも決め直しを確かめる案である。wx は同じ通知を送る（`include/wx/msw/private.h` の注記）ので本当の制約ではないが、要求が無い。
- Linux・macOS で確かめること。消える案は、Linux・macOS でも受入を通す案である。本当の制約ではない（wx の汎用の仕組みで書く。D8）が、Runner がそこで未検証である（§2 の 10）。
- 表示の DPI とフォントの変更の扱いを変えること。消える案は、DPI とフォントの変更もテーマの切替と同じ入口で受ける案である。本当の制約ではない（同じ入口で受けることもできる）が、Evaluation Trace は別の経路で受けていて（[EvalTracePanel.cpp:157](../../apps/runner/src/EvalTracePanel.cpp)・588〜594 行）、別の関心である。

**成功の測り方**

- G1〜G4: §7 の受入 1〜5 が通る。
- 長い Run の途中で OS のテーマを切り替えても、Runner を再起動しなくて済む（使う場面として記録し、合否にはしない）。

## 4. 確定事項（暫定の既定。会話で合意した決定ではない。§9 とあわせてグリルで見直す）

- **D1. 外観を決め直す入口を RunnerFrame の 1 つにする。OS の色の変更の通知から呼ぶのはこの入口だけにし、Logs・Evaluation Trace・QValue の pane は通知を自分では受けず、RunnerFrame から決め直しを受ける。** 理由: 通知のまとめ方と遅らせ方を 1 か所に持ち、pane ごとに持たない。frame が持つ部分（pane の見出し・ツールバー）と pane の部分を、決まった順で決め直せる。依存: D2・§9 の 3。
- **D2. 通知を受けたその場では塗らず、通知の処理がすべて終わってから 1 回だけ決め直す。決め直しを待っている間に届いた通知は、数えるだけにする。** 理由: 通知のその場で読んだ色は切替の前のものである可能性がある（§2 の 3・8）。Windows は 1 回の切替で通知を複数回送りうる（§2 の 8）。依存: §5.3 の段 0（「処理がすべて終わってから」で足りるかを観測で決める）。
- **D3. 決め直しは、起動時に外観を決めるのと同じ決め方で、その時点のシステムの色から決める。** 理由: 起動し直したときと同じ見た目になり、それを受入の物差しにできる（G1）。起動時と切替時で 2 つの決め方を持たない。依存: なし。
- **D4. 決め直す領域は §5.2 の表とする。新たに対象にするのは、QValue の表と、Evaluation Trace の表の行の無い部分である。** 理由: この 2 つは、決め直しが正しい時点に走っても作ったときの色が残る（§2 の 6）。依存: D3。
- **D5. wx が持つ部分（タイトルバー・メニュー・ステータスバー・スクロールバー・標準コントロール）は、wx の切替の処理に任せ、アプリからは決め直さない。段 0 で、アプリの決め直しのあとも wx の部分に古いテーマが残れば、§9 の 2 の既定を入れて確かめる。それでも残る部分は wx の不具合として §8 に記録する。** 理由: wx の部分を切り替える処理は wx の公開の API に無く、アプリから部分ごとに呼べない（§2 の 2）。依存: §5.3 の段 0。
- **D6. 決め直しで表示の状態（§5.5）を変えない。** 理由: テーマの切替は見た目だけの変更である（G3）。今の Logs と Evaluation Trace の決め直しは、表示の位置と手で変えた列幅を保っている（[LogPanel.cpp:823-831](../../apps/runner/src/LogPanel.cpp)、[EvalTracePanel.cpp:690](../../apps/runner/src/EvalTracePanel.cpp)）。依存: なし。
- **D7. 通知と決め直しを、それぞれ VERBOSE のログに 1 行ずつ出す（§5.4）。** 理由: 段 0 の観測（通知の時点と決め直しの時点の色）と、受入 2・4 の「1 回の切替で決め直しは 1 回」を、外から確かめられるようにする。依存: D2。
- **D8. 遅らせ方とまとめ方は wx の汎用の仕組みで書く。MSW に限る処理（§9 の 2 を入れた場合の再送）は MSW の条件付きにする。** 理由: wx のほかの port は、既存のウィンドウにも外観の変更を反映する（wx の `interface/wx/app.h` の `SetAppearance` の説明）。そこでは決め直しが二度塗りになるだけで害が無い。依存: D5。
- ADR は起こさない。決め直しの入口と時点は戻しやすく、ADR の 3 条件（戻しにくい・文脈が無いと意外・本当の比較の結果）を満たさない。

## 5. 実装契約

### 5.1 流れ

1. OS の色の変更の通知が RunnerFrame に届く。RunnerFrame は届いた数を数え、VERBOSE のログを 1 行出す（§5.4）。通知は wx の既定の処理へそのまま流す（止めない。wx が自分の部分を切り替え、子のウィンドウへ通知を回すため）。RunnerFrame が自分で送った通知（§9 の 2 を入れた場合）は、数えず、ログも出さず、決め直しを予約しない。wx の既定の処理へはそのまま流す。
2. 決め直しをまだ待っていなければ、通知の処理がすべて終わったあとに 1 回だけ決め直しを予約する。待っている間に届いた通知は、数を足すだけにする。
3. 予約した時点が来たら、RunnerFrame の `UpdateAppearance` が §5.2 の順に決め直し、VERBOSE のログを 1 行出し、数を 0 に戻す。
4. frame が閉じる途中（破棄が始まっている）なら、予約した決め直しを行わない。閉じる順序（[160](../design/160_applications_and_tools.jp.md) §7.1）を変えない。

### 5.2 決め直す領域

上から順に決め直す。閉じている pane（Evaluation Trace を閉じている間など）と、浮いている pane も決め直す。

| # | 領域 | 持ち主 | 決め直すもの |
|---|---|---|---|
| 1 | pane の見出しと枠 | RunnerFrame | 見出し・枠・地の色を、Runner 用の配色（非アクティブの見出しの文字の色）を含めて決め直し、frame を描き直す |
| 2 | 4 本のツールバー pane | RunnerFrame | 各ツールバーの地の色とテーマの描き方を使うか、Run 制御ツールバーの再生・Step のアイコン（実行状態を保つ） |
| 3 | タイトルバー・メニュー・ステータスバー・スクロールバー・標準コントロール | wx | 決め直さない（D5） |
| 4 | Logs | LogPanel | 今の決め直しの中身（§2 の 5 の表） |
| 5 | Evaluation Trace | EvalTracePanel | 今の決め直しの中身に、表の行の無い部分の地の色（セルの既定の背景と同じ色）を足す |
| 6 | QValue の表 | QValuePanel | セルの既定の背景と文字、見出しの地と文字、罫線、選択、行の無い部分の地の色を、wxGrid が作るときと同じ決め方（§2 の 6）で決め直す。選んでいる行があれば、その強調を今の色で塗り直す |

補助 pane（HeatMap・Conv2d）と Train／Eval の View は、決め直す色を持たない（§2 の 7）。描き直しで描くたびに読む色は新しくなる。

### 5.3 段 0 の観測と分かれ方

実装の最初の段で、D7 のログを入れた Runner を使って観測する。

- 手順: Logs の表示 level を verbose にして Runner を起動する（起動時に `app.log_panel.level=verbose` を渡す。[160](../design/160_applications_and_tools.jp.md) §3.5）。人が Windows の設定でアプリのモードを ライト→ダーク→ライト の順に切り替える（AI はシステムの設定を変えない）。切り替えるたびに Runner 全体のスクリーンショットを撮り、Logs のログ欄の通知と決め直しの行を残す。
- 分かれ方:

| 観測 | 決め方 |
|---|---|
| 決め直しの時点（通知の処理がすべて終わったあと）で、色が新しいテーマになっている | D2 の既定のまま |
| 決め直しの時点でも、色が古いテーマのことがある | 最後の通知から一定の時間（§6 の 3）のあとに決め直す形に変える。それでも古いことがあれば、時間を延ばさずに止め、§9 の 1 へ戻す |
| アプリの決め直しのあとも、§5.2 の 3（wx の部分）に古いテーマが残る | §9 の 2 の既定を入れて確かめる。それでも残る部分は、wx の不具合として §8 に記録し、受入 4 の比較から外す |
| アプリの決め直しのあとも、§5.2 の 1・2・4〜6 に古いテーマが残る | 本書の実装の漏れとして直す |

### 5.4 ログ

VERBOSE で、英語で出す。色は `#RRGGBB`、外観はその時点の wx の判定（アプリがダークで描いているか）を `true`／`false` で書く。

- 通知: `Appearance change notified: dark=<true|false> window=<#RRGGBB> pending=<その時点で数えた通知の数>`
- 決め直し: `Appearance updated: dark=<true|false> window=<#RRGGBB> notifications=<まとめた通知の数> elapsed_ms=<決め直しにかかった時間>`

`window` は、システムのウィンドウの地の色（WINDOW）である。RunnerFrame が自分で送った通知（§9 の 2 を入れた場合）では、通知の行を出さない。

### 5.5 変えないもの

- Evaluation Trace: Live／Trace のモード、選択行、並べ替え、再生の状態、手で変えた列幅、保持している行と百分位。
- Logs: 表示の先頭の位置と横の位置、Auto-scroll・Wrap・Indent の状態、未読の行数、未確認の警告の数、検索の語。
- pane の配置（ドック／浮き、大きさ）と frame の大きさ。Train／Eval の実行状態と、ツールバーのトグルの表示。
- 学習と評価: 決め直しは UI のスレッドだけで行い、Runner・Env・Agent に触れない。
- 表示の DPI とフォントの変更の扱い（非ゴール）。

### 5.6 移行対象

- [RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)・[RunnerFrame.hpp](../../apps/runner/src/RunnerFrame.hpp): 通知の受け方（§5.1）、決め直しの入口、§5.2 の 1・2、ログ（§5.4）。
- [LogPanel.cpp](../../apps/runner/src/LogPanel.cpp)・[LogPanel.hpp](../../apps/runner/src/LogPanel.hpp): 自分で通知を受ける登録（295 行）を消し、決め直しを RunnerFrame から呼べるようにする。
- [EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)・[EvalTracePanel.hpp](../../apps/runner/src/EvalTracePanel.hpp): 同じく 156 行の登録を消す（157 行の DPI の登録は残す）。表の行の無い部分の地の色を足す。
- [QValuePanel.cpp](../../apps/runner/src/QValuePanel.cpp)・[QValuePanel.hpp](../../apps/runner/src/QValuePanel.hpp): 表の決め直しを足す。
- テスト: [LogPanel_test.cpp:214-217](../../apps/runner/src/LogPanel_test.cpp) と [EvalTracePanel_test.cpp:814-819](../../apps/runner/src/EvalTracePanel_test.cpp) の、pane へ通知を送るところを、決め直しの入口を呼ぶ形に直す。受入 1〜3 のテストを足す。
- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md): §2.1 の RunnerFrame の行に、OS のテーマの切替を 1 か所で受けて落ち着いてから決め直すことを足す。§3.4（114 行）の「テーマ・フォント・DPI変更には追従し」を、テーマ（OS のアプリのモードの切替）には RunnerFrame からの決め直しで追従し、フォントと DPI の変更には今のとおり pane が自分で受けて追従する、と分けて書く。§3.5（142 行）の「テーマ変更の通知で全行を塗り直す」を、「RunnerFrame からの外観の決め直しで全行を塗り直す」に直す。§8 の「Runner UI変更」の行の「theme変更時のSVG差し替え」を、受入 1〜3 のテストと受入 4 の手順の所在に直す。英語版（[160.en](../design/160_applications_and_tools.en.md)）は翻訳の手順で追う。
- 正式に採番するときに、`core/anet-core/testdata/prd<番号>/README.md` を作り、段 0 と受入 4 の手順・領域の一覧・結果の表を置く。160 §8 から参照する。
- 変えないもの: [020 Run 実行ガイド](../design/020_user_guide_run.jp.md)（515 行の「通常・選択の配色はシステムのテーマに従い」は切替のあとも成り立つ）。PRD094・PRD055 の本文も変えない。消える案は、両 PRD の本文（PRD094 §8 の受入 10 など）を新しい受け方に合わせて直す案である。本当の制約ではない（直すことはできる）が、今の仕様の正本は設計文書（160）で（[AGENTS.md](../../AGENTS.md) §設計判断の記録(ADR)を読む条件）、PRD は当時の決定として残す。
- 互換層は置かない。pane が自分で通知を受ける経路を残さない。

### 5.7 性能

- 決め直しは 1 回の切替に 1 回だけ走る。今は通知が届くたびに走る。1 回の切替で通知が複数回届くなら（§2 の 8。確かめていない）、その回数だけ走っているので、回数は減る。
- 1 回の重さは、Logs の行数（既にある行の色を全部塗り直す。行数に上限は無い）と、Evaluation Trace の行数（最大 `max_rows`、既定 1000 行の列幅を測り直す）に比例する。テーマの切替は稀なので、性能の受入は置かない。決め直しにかかった時間はログに出し（§5.4）、段 0 と受入 4 で記録する（合否にしない）。
- step のループと学習（ホットパス）には触れない。

### 5.8 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RunnerFrame` | 変更 | OS の色の変更の通知を数え、通知が落ち着いてから外観の決め直しを 1 回だけ行う。frame の部分（pane の見出し・ツールバー）を決め直し、常設 pane の決め直しを呼ぶ。入口の名前は `UpdateAppearance` | RunnerFrame |
| `LogPanel` | 変更 | 通知を自分で受けず、RunnerFrame から `UpdateAppearance` を受ける | LogPanel |
| `EvalTracePanel` | 変更 | 通知を自分で受けず、RunnerFrame から `UpdateAppearance` を受ける。表の行の無い部分の地の色も決め直す | EvalTracePanel |
| `QValuePanel` | 変更 | 表の外観の決め直し（`UpdateAppearance`）を新たに持つ | QValuePanel |

`UpdateAppearance` は、LogPanel・EvalTracePanel の既存の名前と同じ動詞・同じ意味（UI の性質を今の条件へ更新する）で使う（[094 の 22impl:57](094_evalpanel_step_history_22impl.md)）。

## 6. 細部（暫定で置いた既定）

1. 「通知の処理がすべて終わったあと」は、wx の `CallAfter` で決め直しを UI のイベントの順番の後ろに 1 回積むことで作る（理由: 通知の処理は、アプリの登録した処理・wx の既定の処理・Windows の既定の処理の順に 1 つの呼び出しの中で済む（§2 の 2）ので、その後ろに積めば全部が終わってから読める見込み。段 0 で確かめる）。
2. 予約した決め直しは、frame が破棄されたら行わない（理由: §5.1 の 4）。
3. 段 0 で「処理のあと」が足りなかったときの一定の時間は、最後の通知から 250 ms とする（理由: 人が見て遅れと感じにくく、続けて届く通知をまとめられる長さの見積もり。段 0 で測って直す）。
4. ログは VERBOSE（理由: 切替は稀で行数が少なく、ふだんの Logs の表示 level（info）では出ない）。書式は §5.4。
5. QValue の表の色は、wxGrid が作るときと同じシステムの色（§2 の 6）にする（理由: D3。Evaluation Trace の罫線は 3DSHADOW で決めているが、QValue の表はもともと wxGrid の既定のままなので、それに合わせる）。
6. 受入 1〜3 のテストのタグは `[theme_follow]` とし、RunnerApp を起動する別の process で、明示したときだけ走らせる（理由: 既存の `[eval_steps_menu]` と同じく RunnerApp を起動するため。[160](../design/160_applications_and_tools.jp.md) §8 の 401 行）。

## 7. 検証と受入

**使う場面**: 長い Run の途中で OS のアプリのモードを切り替える。

**受入（外から観測できるもの）**

受入 1〜3 と 5 の process の作り方: ダークの process は、環境変数 `PRD094_UI_DARK` を置いて `AnetRLRunner-gui-test` を起動する（テストの main が RunnerApp の起動の処理より前にダークへ固定する。§2 の 4）。ライトの process は `PRD094_UI_DARK` を置かずに起動する。RunnerApp を起動しないテスト（`[log_panel]`・`[prd094]`・`[trace_ui_visual]`・`[prd105]`）は、OS のモードによらずライトになる。RunnerApp を起動するテスト（`[theme_follow]`・`[prd105_ui_keys]`）は、起動の処理が OS のモードに従うので、OS のアプリのモードがライトの機械でだけライトになる（§2 の 4）。どちらの process も、最初に外観が意図したもの（ダークかライト）かを確かめ、違えば失敗にする（既存の [LogPanel_test.cpp:188-189](../../apps/runner/src/LogPanel_test.cpp) と同じ）。RunnerApp を起動するテストのライトの回は OS がライトの機械で回し、OS がダークで回せなかったときは、失敗ではなく未実施と報告する（AI はシステムの設定を変えない）。受入 1〜3 は、RunnerApp を起動する別の process で `AnetRLRunner-gui-test "[theme_follow]"` を回し、Logs の表示 level を verbose にしてから（LogPanel の表示 level を変える公開の操作で）確かめる。

1. 決め直しの領域（自動。ライトとダークの process で 1 回ずつ）: §5.2 の 1・2・4〜6 の領域のうち、公開の API で色を変えられるものを、テストから別の色（赤）に変えてから、frame に OS の色の変更の通知を 3 回送り、イベントを処理させる。決め直しのあと、各領域の色が、その process のシステムの色から起動時と同じ決め方で決まる色と一致する。確かめる値は、pane の見出しの地と文字、各ツールバーの地（描いた画素で確かめる）、アイコン（Run 制御ツールバーの再生・Step、Logs のゴミ箱、Evaluation Trace の操作帯の 6 つ。描いた画素に、その process の BTNTEXT と同じ色の画素がある）、Logs のログ欄の地と文字、Evaluation Trace のセルの既定の背景と文字・見出しの地と文字・罫線・行の無い部分の地、QValue の表のセルの既定の背景と文字・見出しの地と文字・罫線・行の無い部分の地・選んだ行の強調である。Evaluation Trace を閉じた状態と、浮かせた状態でも同じ。
2. 1 回にまとめる（自動。受入 1 と同じ process）: 受入 1 で通知を 3 回送ったとき、Logs のログ欄に `Appearance updated` の行が 1 行だけ出て、`notifications=3` である。`Appearance change notified` の行は 3 行で、`pending=` が 1・2・3 である（§9 の 2 を入れても、RunnerFrame が自分で送った通知の行は出ないので変わらない）。通知を送らなければ、どちらの行も出ない。frame の破棄が始まってから通知を送っても、決め直さず、落ちない。
3. 表示の状態を変えない（自動。受入 1 と同じ process）: Evaluation Trace を Trace モードにして行を選び、列を並べ替え、列の幅を 1 つ手で変えて再生を始め、Logs の Auto-scroll を OFF にして上と右へスクロールし、Wrap と Indent の片方を切り替え、未確認の警告が 1 件以上ある状態で検索の語を入れ、pane を 1 つ浮かせ、Train を止めた状態で、受入 1 の通知を送る。前後で、次が同じである。Evaluation Trace のモード・選択行・並べ替え・再生中であること・手で変えた列の幅・保持している行の数と最新の行の百分位の値。Logs の表示の先頭の行と横の位置・Auto-scroll・Wrap・Indent・未読の行数・未確認の警告の数・検索の語。各 pane のドック／浮きと大きさ、frame の大きさ。Train と Eval の実行状態と、Run 制御ツールバーの Train・Eval のトグルの表示（再生か一時停止か）。
4. OS の切替（手動。人が行う。AI はシステムの設定を変えない）: 起動時に `app.log_panel.level=verbose` を渡して Logs の表示 level を verbose にし、Logs に各 level の行があり、Evaluation Trace に報酬と終端の行があって行の無い部分が見え、QValue の表で行を選び、pane を 1 つ浮かせた Runner で、Windows の設定のアプリのモードを ライト→ダーク→ライト の順に切り替え、切り替えるたびにスクリーンショットを撮る。同じ構成でそれぞれのテーマで起動し直したスクリーンショットと並べ、§5.2 の 1〜6 の領域ごとに、同じ位置の代表の 1 点の画素の色を領域の一覧に記録する（アイコンは記号の上の 1 点）。全部の領域で、切り替えたものと起動し直したものの色が同じである（§8 に wx の不具合として記録した部分は外し、一覧にその名前を挙げる）。各切替で、Logs のログ欄に `Appearance updated` の行が 1 行だけ出る。
5. 既存のテスト: 通知を pane へ送っていたところを直したうえで、ライトとダークの process の両方で、`AnetRLRunner-gui-test "[log_panel]"`、PRD094 の UI のテスト（`"[prd094]"`・`"[trace_ui_visual]"`）、PRD105 の UI のテスト（`"[prd105]"` と、RunnerApp を起動する別の process の `"[prd105_ui_keys]"`）が通る。
6. 文書: 160 §2.1・§3.4・§3.5・§8 に、OS のテーマの切替を RunnerFrame が 1 か所で受けて落ち着いてから決め直すこと、決め直す領域、段 0 と受入 4 の手順の所在がある。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| 各 pane が自分で通知を受けて、自分で遅らせて塗り直す形 | 不採用 | まとめ方と遅らせ方を pane ごとに持つことになる。frame の部分（pane の見出し・ツールバー）には pane の持ち主がいない。決め直す順を決められない |
| 通知のその場で塗り直し、同じ決め直しを遅らせてもう一度行う形 | 不採用 | 2 回塗ることになり、1 回目は切替の前の色で塗りうる（§2 の 3） |
| frame を作り直す形（AUI の配置を保存して RunnerFrame を作り直す） | 不採用 | 確実だが、RunManager・各 pane・Trace の状態の付け替えが大きい（2026-10-31 の会話の比較） |
| 再起動を案内するだけにする形 | 不採用 | ユーザーが実装で直すことを選んだ |
| テーマに合わせない固定の色（§2 の 7）をテーマに合わせる | 範囲外 | 起動時からテーマに合わせていない部分で、別の関心。再開条件: 固定の色が見づらいという要求が出たとき |
| Runner の設定でライト／ダークを選ぶ | 範囲外 | 別の関心。再開条件: OS と違うテーマで Runner を使いたい要求が出たとき |
| wxWidgets のソースを手元で直す | 範囲外 | wx の版を上げるたびに当て直しが要る。再開条件: wx の不具合と分かった部分が、上流の更新でも直らないとき |
| 開いている modal ダイアログの追従 | 範囲外 | 表示が短く、開き直せば新しいテーマで出る。再開条件: ダイアログを開いたまま切り替える場面が出たとき |
| ハイコントラストへの切替 | 範囲外 | 要求が無い。再開条件: 要求が出たとき |
| Linux・macOS での確認 | 範囲外 | Runner がそこで未検証（§2 の 10）。再開条件: Linux で Runner のビルドが通ったとき |
| 表示の DPI・フォントの変更も、テーマの切替と同じ入口で受ける | 範囲外 | 今は別の経路で追従していて（[EvalTracePanel.cpp:157](../../apps/runner/src/EvalTracePanel.cpp)・588〜594 行）、別の関心である。再開条件: DPI かフォントの変更で、半端な反映が見つかったとき |

段 0 で wx の不具合と分かった部分は、この表に「wx の不具合」として足す（再開条件: wx の更新）。

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | 決め直す時点（通知の処理がすべて終わったあと／最後の通知から一定の時間のあと） | 通知の処理がすべて終わったあと。段 0 で足りなければ最後の通知から 250 ms のあと（§5.3・§6 の 3） |
| 2 | 段 0 でアプリの決め直しのあとも wx の部分に古いテーマが残ったときの扱い（MSW で wx の通知の経路をもう一度通す／wx の不具合として記録するだけ） | MSW では、決め直しの最初に frame と浮いている pane の枠へ色の変更の通知をもう一度送り、wx の切替の処理を落ち着いた状態で通す。RunnerFrame は自分が送った通知を数えず、ログの行も出さず、決め直しを予約しない（§5.1 の 1・§5.4）。それでも残る部分は §8 に記録する |
| 3 | pane が通知を自分で受けるのをやめるか（D1）／受けたまま RunnerFrame からも遅らせて呼ぶか | やめる（1 つの関心を 1 つの仕組みで扱う）。Logs を単独で開くデモ（`[log_panel_demo]`）は、テーマの切替に追従しなくなる |
| 4 | ログのレベル（VERBOSE／INFO）と、通知ごとの行を残すか | VERBOSE。通知ごとの行も残す（切替は稀で行数が少なく、次に半端な反映が出たときの手がかりになる） |
| 5 | 段 0 を本書の実装の最初の段で行うか、実装の前に別に行うか | 実装の最初の段で行う（D7 のログが観測の道具を兼ねる） |
| 6 | Evaluation Trace の決め直しで、保持している行の列幅を測り直すこと（今の決め直しがしている。フォントと DPI のためのもの）を、テーマの切替でも行うか | 行う（今と同じ。列幅は広がるだけで縮まない） |
| 7 | 受入 4 のスクリーンショットの置き場 | `.scratch/` に置き、testdata の README には手順・領域の一覧・結果の表だけを書く |

## 10. 依存と順序

- 先に要る PRD は無い。
- 受入 4 は、[PRD094](094_evalpanel_step_history_10prd.md) §8 の受入 10 と [PRD055](done/055_runner_toolbar_10prd.md) の受入 10 の、実機で確かめていない部分（OS のテーマの切替）を兼ねられる。

## 11. 参照

- [PRD094 Evaluation Trace](094_evalpanel_step_history_10prd.md)、[PRD055 Runner ツールバー](done/055_runner_toolbar_10prd.md)
- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §2.1・§3.4・§3.5・§7.1・§8
- [040 開発環境構築ガイド](../design/040_development_environment.jp.md) §3.4
- [RunnerApp（wiki）](../wiki/apps/110_runner_app.md)
