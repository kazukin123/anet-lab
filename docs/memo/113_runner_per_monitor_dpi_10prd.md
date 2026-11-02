# Runner を Per-Monitor V2 の DPI 対応で動かす PRD（manifest の gdiScaling を外す）

> 2026-11-02 起票（暫定・999。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍の記述で探し直す。wxWidgets のソースの行番号は 3.3.3 のもの。
> 2026-11-02 999 から 113 へ採番した（本文は変えていない）。
> 2026-11-02 追記: PRD094 が、Evaluation Trace の DPI の変更への追従の確かめを本 PRD へ移した。受入 7 に Evaluation Trace を足すかを、§9 の 10 に置いた。
> 起点: 2026-11-02 の会話。`AnetRLRunner-gui-test` で以前から落ちていた 2 件を調べ、そのうち `[log_panel]` の字下げのテストの失敗は、manifest の `gdiScaling=true` で Runner が DPI 非対応のまま動いていることが原因と分かった（§2）。ユーザーは、字下げの幅を RichEdit の配置から測る案（その場で示した推奨案）と、gdiScaling を外す件を、この PRD にまとめて起こすことを選んだ。もう 1 件（構築に失敗した Runner を閉じたあとの落ち）は、同じ日に PRD なしで直した（RunnerFrame の後回しの並べ直しのガード）。
> 関連: [999_runner_theme_switch](999_runner_theme_switch_10prd.md)（OS のテーマの切替。DPI の変更はその入口に入れないとした）、[PRD110](done/110_runner_main_pane_dynamic_centre_10prd.md)（主 pane の配置と幅の規則）、[160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §2.1・§3.5・§8。

## 1. 目的と問題

Windows は、画面の拡大率（表示スケール。150% なら 144 DPI）に対して、アプリがどこまで自分で合わせるかを宣言させる。宣言には段階がある。

- DPI 非対応（unaware）: アプリは常に 100%（96 DPI）のつもりで描き、Windows が出来上がった画面を拡大する。何もしなくても大きさは合うが、拡大の分だけぼやける。
- GDI スケーリング: DPI 非対応のアプリのうち、GDI で描いた文字と線だけを Windows が高い解像度で描き直す互換の仕組み。ビットマップ・アイコン・GDI 以外で描いたものは引き伸ばしのまま残る。
- Per-Monitor V2（以下 PMv2）: アプリが各モニターの DPI で実際の画素を描く。くっきり描けるが、アプリが寸法を DPI に合わせる責任を持つ。モニターをまたいで窓を動かすと、Windows が DPI の変更を通知する。

wxWidgets では、PMv2 のときの寸法の書き方として、100% のときの大きさ（DIP: 機器に依らない画素）で書き、`FromDIP()` でその窓の実際の画素に直すことが勧められている（§2 の 6）。

Runner の manifest は PMv2 を宣言しているが、同じ manifest の `gdiScaling=true` がそれを上書きし、Runner は実行時に DPI 非対応＋GDI スケーリングで動いている（§2 の 1・2）。150%・175% の画面では次のことが起きる。

1. GDI で測った字幅と、ログ欄（RichEdit）が描く字幅が食い違う。Logs の字下げ（折り返した 2 行目以降を本文の頭にそろえる）が約 1 文字足りず、そのテストが落ちる（§2 の 4）。
2. アイコン・GL で描く View（Atari・HeatMap・Conv2d）・バッファに描いてから写す View は、100% で描いて引き伸ばすので、ぼやける見込みである（§2 の 3。実画面では見比べていない）。
3. manifest の記載（PMv2）と実際の動き（DPI 非対応）が違うので、表示の不具合を調べるときに取り違える。

## 2. 現状（2026-11-02 に確認したこと）

1. manifest: [runner.manifest:12-17](../../apps/runner/src/runner.manifest) は `dpiAwareness` に `PerMonitorV2`、`gdiScaling` に `true` を書く。最初の版（2025-12-30）から変わっていない。この manifest は `AnetRLRunner` と `AnetRLRunner-gui-test` の両方に入る（[apps/runner/CMakeLists.txt:20-23・38-42・148](../../apps/runner/CMakeLists.txt)）。
2. 実行時の観測: Windows 11 Pro 26H2（build 26300.9711）、モニターは 144 DPI（150%）と 168 DPI（175%）の 2 枚、システムの DPI は 144。外から `GetDpiAwarenessContextForProcess` で読むと、`AnetRLRunner.exe`（未登録の Env の設定で起動）と `AnetRLRunner-gui-test.exe`（`[log_panel_demo]`）は awareness が 0（DPI 非対応）で、窓の DPI は 96 だった。Explorer から起動しても同じだった。同じシェルから起動した charmap・magnify は Per-Monitor（2）だった。gui-test の実行ファイルの写しから、manifest の gdiScaling の行だけを `mt.exe` で外すと、awareness は 2、窓の DPI は 144 になった。
3. Microsoft の説明: GDI スケーリングは、DPI 非対応の GDI のアプリを手を入れずに見やすくするための互換の仕組みとして足されたもので、アプリの DPI 対応の宣言を上書きする。GDI の文字と線は整数倍の解像度で描いて DWM が拡縮し、ビットマップ・アイコンは引き伸ばし、DirectX などで描いたものは正しく扱わない。Per-Monitor 対応のアプリは使わないほうがきれいに見えるとしている（[Improving the high-DPI experience in GDI based Desktop Apps](https://blogs.windows.com/windowsdeveloper/2017/05/19/improving-high-dpi-experience-gdi-based-desktop-apps/)）。デスクトップアプリには PMv2 を勧め、PMv2 だけが子のウィンドウにも DPI の変更を通知するとしている（[High DPI Desktop Application Development on Windows](https://learn.microsoft.com/ja-jp/windows/win32/hidpi/high-dpi-desktop-application-development-on-windows)）。
4. Logs の字下げの食い違い:
   - [LogPanel.cpp:642-651](../../apps/runner/src/LogPanel.cpp) の `SetIndent` は、本文より前（時刻・区切り・種類。`12:34:56.789 [I] ` の 17 文字）の幅を、`GetTextExtent` で `0` を 17 文字並べて測り、1/10 mm に直して段落の字下げにする。ログ欄のフォント（[LogPanel.cpp:254](../../apps/runner/src/LogPanel.cpp) の `wxFONTFAMILY_TELETYPE` 10pt）に対して、この機械の GDI は ＭＳ ゴシック 13px を選ぶ。
   - 今の Runner の process の中では、GDI は 17 文字を 111px と返した。ログ欄は本文の頭を行頭から 119px 先に描き（行頭 x=1、本文の頭 x=120）、折り返した 2 行目は x=112 から始まった。8px（約 1 文字）足りない。同じ LOGFONT を別の process の GDI で測ると 119px だった。
   - `[log_panel]` の「Logs indent wrapped lines to the message on demand」（[LogPanel_test.cpp:357-391](../../apps/runner/src/LogPanel_test.cpp)）は、許容 ±2px で 375・385・386 行が落ちる。gdiScaling を外した写しでは、`[log_panel]` の 10 件が全部通った。
   - LogPanel とこのテストは 2026-10-31 の版から変わっていない。その日の実装では `[log_panel]` が通ったと記録されている（memo ではなく会話の記録。確かめられない）。OS は 2026-10-25 から再起動しておらず、manifest も変わっていないので、差は表示の環境と推定する（確かめていない）。
5. 寸法の今の書き方: 画面の大きさを px で直書きしている箇所を、次の 6 つの型で洗った。対象は、`apps/runner/src` のテスト以外のファイル、`core/envs` の View のファイル（`*View.cpp`・`*View.hpp`）、`core/anet-core/src/gui.cpp` の 38 ファイルである。型 A・B・D・F と E の線は、§7 の受入 4 の正規表現で検索でき、2026-11-02 の当たりの数は A 14、B 12、D 37、E の線 31、F 1 である。

   | 型 | 書き方 | 箇所と値 |
   |---|---|---|
   | A | `wxSize(数値, 数値)` | [RunnerFrame.cpp:207](../../apps/runner/src/RunnerFrame.cpp)（初期の窓 1024×1024）、[Conv2dPanel.cpp:37](../../apps/runner/src/Conv2dPanel.cpp)・[HeatMapPanel.cpp:339](../../apps/runner/src/HeatMapPanel.cpp)（Panel の初期の大きさ 600×600）、[HeatMapPanel.cpp:60](../../apps/runner/src/HeatMapPanel.cpp)（HeatMap の設定のダイアログ 620×430）・353（step の欄の幅 60）、[AtariView.cpp:30](../../core/envs/atari1/src/AtariView.cpp)（760×520）、[ErrorDialog.cpp:22](../../apps/runner/src/ErrorDialog.cpp)（エラーのダイアログ 650×450）・40（文字欄の高さ 120）、[WorkspaceDialog.cpp:27・82](../../apps/runner/src/WorkspaceDialog.cpp)（workspace の選択のダイアログ 620×500。最小の大きさも同じ）・55（リストの高さ 150）、[EvalTracePanel.cpp:105](../../apps/runner/src/EvalTracePanel.cpp)（アイコン 16×16） |
   | B | 大きさを決める関数に数値を直接渡す | [RunnerFrame.cpp:472-474・485・498・513・1258-1280](../../apps/runner/src/RunnerFrame.cpp)（pane の推奨・最小・浮かせたときの大きさ。400・800・200×200・300×150・100×100・800×400 など）、[QValuePanel.cpp:240](../../apps/runner/src/QValuePanel.cpp)（表の行見出しの幅 30） |
   | C | 名前つきの定数を大きさとして渡す | [RunnerFrame.cpp:39-42・46](../../apps/runner/src/RunnerFrame.cpp)（Logs の高さ 200、Train・Eval の最小幅 200、アイコン 16×16）、[QValuePanel.cpp:14-17・782](../../apps/runner/src/QValuePanel.cpp)（列幅 100・80、最小 50・70、ヒートマップの推奨幅 180） |
   | D | sizer の枠の数値（`wxALL` などに添える数値） | 37 か所（Conv2dPanel 2、ErrorDialog 5、EvalPanel 2、HeatMapPanel 10、LogPanel 1、QValuePanel 6、TrainPanel 1、WorkspaceDialog 10） |
   | E | 描画の余白と線の太さ | 余白: [LunarLanderView.cpp:169・195](../../core/envs/lunarlander1/src/LunarLanderView.cpp)（40）、[DropMergeView.cpp:135-136・163](../../core/envs/dropmerge1/src/DropMergeView.cpp)（20）、[AtariView.cpp:182](../../core/envs/atari1/src/AtariView.cpp)（10。GL）。線: `wxPen` を作るときに渡す太さ 1〜3 が 31 か所（[PlotPanel.cpp](../../apps/runner/src/PlotPanel.cpp) 2、[QValuePanel.cpp:119](../../apps/runner/src/QValuePanel.cpp) 1、[CartPoleView.cpp](../../core/envs/cartpole2/src/CartPoleView.cpp) 3、[DropMergeView.cpp](../../core/envs/dropmerge1/src/DropMergeView.cpp) 10、[GridMazeView.cpp](../../core/envs/gridmaze1/src/GridMazeView.cpp) 1、[LunarLanderView.cpp](../../core/envs/lunarlander1/src/LunarLanderView.cpp) 14）。変数を宣言する形（`wxPen pen(...)`）と、色がシステムの色を返す関数の呼び出しの形を含む。太さをすでに `FromDIP` で渡している [EvalTracePanel.cpp:97](../../apps/runner/src/EvalTracePanel.cpp) と、太さを渡さない（既定の 1 で、数値の直書きが無い）[EvalTracePanel.cpp:39](../../apps/runner/src/EvalTracePanel.cpp)・[RunnerFrame.cpp:116](../../apps/runner/src/RunnerFrame.cpp)・[DropMergeView.cpp:198](../../core/envs/dropmerge1/src/DropMergeView.cpp) は含まない |
   | F | sizer の行と列の間隔 | [HeatMapPanel.cpp:115](../../apps/runner/src/HeatMapPanel.cpp)（HeatMap の設定のダイアログの `wxFlexGridSizer` の間隔 10・10） |

   型に当たるが大きさを表さないか、wx が DIP として扱うので除くもの: `wxSize(-1, -1)`（[TrainPanel.cpp:106](../../apps/runner/src/TrainPanel.cpp)。既定の大きさの意味）、`SetRowLabelSize(0)`（[EvalTracePanel.cpp:186](../../apps/runner/src/EvalTracePanel.cpp)。見出しを隠す）、bitmap bundle を作る・取り出す関数に渡す大きさ（`wxBitmapBundle::FromSVG`・`wxArtProvider::GetBitmapBundle`。[gui.cpp:466](../../core/anet-core/src/gui.cpp)、[RunnerFrame.cpp:424・426・654](../../apps/runner/src/RunnerFrame.cpp)、[LogPanel.cpp:87・329・368・414・416・780](../../apps/runner/src/LogPanel.cpp)。LogPanel の `kToolBitmapSize` はこの用途だけに使う）。ツールバーのアイコンの大きさ（`SetToolBitmapSize` に渡す [RunnerFrame.cpp:381・421](../../apps/runner/src/RunnerFrame.cpp) の `kToolBitmapSize` と [EvalTracePanel.cpp:105](../../apps/runner/src/EvalTracePanel.cpp)）は、段 0（§5.6）で wx が DIP として扱うと分かれば、同じく除く。位置（`wxPoint`）は型に入れない（[LogPanel.cpp:530](../../apps/runner/src/LogPanel.cpp) は原点の問い合わせで大きさではなく、[AtariView.cpp:44](../../core/envs/atari1/src/AtariView.cpp) は View の絵の中の文字の位置で §8 の範囲外に当たる）。workspace の選択のダイアログは、作ったあとで中身に合わせて大きさを決め直す（[WorkspaceDialog.cpp:81](../../apps/runner/src/WorkspaceDialog.cpp) の `SetSizerAndFit`）ので、窓の大きさは中身（型 A のリストの高さと型 D の枠を含む）と最小の大きさで決まる。

   `FromDIP` を使っているのは [EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)（17 か所）と LogPanel（2 か所）だけで、DPI の変更（`wxEVT_DPI_CHANGED`）を受けるのは [EvalTracePanel.cpp:157](../../apps/runner/src/EvalTracePanel.cpp) だけである。GL で描く 3 つは、描く大きさを `GetContentScaleFactor` で実際の画素に直している（[Conv2dPanel.cpp:139](../../apps/runner/src/Conv2dPanel.cpp)・[HeatMapPanel.cpp:571](../../apps/runner/src/HeatMapPanel.cpp)・[AtariView.cpp:147・171](../../core/envs/atari1/src/AtariView.cpp)）。Env の View・PlotPanel・QValuePanel・Evaluation Trace の操作帯は、バッファに描いてから写す（`wxAutoBufferedPaintDC` など。[LunarLanderView.cpp:560](../../core/envs/lunarlander1/src/LunarLanderView.cpp)・[DropMergeView.cpp:635](../../core/envs/dropmerge1/src/DropMergeView.cpp)・[PlotPanel.cpp:29](../../apps/runner/src/PlotPanel.cpp)・[QValuePanel.cpp:91](../../apps/runner/src/QValuePanel.cpp) など）。DropMerge の View は描くたびに pt 指定のフォントを作る（[DropMergeView.cpp:264](../../core/envs/dropmerge1/src/DropMergeView.cpp) など）。Env の View の `OnPaint` には計測の範囲が無い（[LunarLanderView.cpp:558](../../core/envs/lunarlander1/src/LunarLanderView.cpp) など）。
6. wxWidgets の振る舞い: MSW では窓の座標は実際の画素で、100% の大きさから直すには `FromDIP()` を使うことが勧められている（[wxWidgets の High DPI overview](https://docs.wxpython.org/high_dpi_overview.html)）。wx 3.3.3 の AUI は DPI の変更を受けると、各 pane の最小・推奨の大きさと dock の幅を新しい DPI へ拡縮してから並べ直す（wx の `src/aui/framemanager.cpp` の 623〜653 行）。フォントは pt で持つので、窓の DPI に合わせて wx が大きさを直す。
7. gdiScaling を外した写しでの GUI テスト（2026-11-02、144 DPI のモニター）: 既定のテスト 28 件は全部通った（今の実行ファイルでは字下げの 1 件が落ちる）。RunnerApp を起動するテスト 37 タグを 1 タグ 1 process で回すと、今と結果が違ったのは `[prd110_toolbar]` だけで、新たに落ちた（[RunnerFrame_test.cpp:311](../../apps/runner/src/RunnerFrame_test.cpp)。Panels ツールバーのボタンの 1 つの幅が 0。並びきらずに隠れたと推定）。`[prd110_main_pane]` は 34 件をまとめて 1 process で回すタグで、今も写しも Env の二重登録で落ちる（回し方の問題で、DPI とは関係ない）。
8. 関係する PRD: [999_runner_theme_switch](999_runner_theme_switch_10prd.md) は、OS のテーマの切替を RunnerFrame の 1 か所で受けるとし、DPI とフォントの変更は同じ入口に入れず、Evaluation Trace が自分で受ける今の経路を残すとした（同 §3 の非ゴール・§5.6・§8）。

## 3. ゴール・非ゴール・成功の測り方

**ゴール**

- G1: `AnetRLRunner` と `AnetRLRunner-gui-test` が、実行時に PMv2 で動く。process の DPI 対応が Per-Monitor で、窓の DPI がそのモニターの DPI と同じになる。
- G2: 150%・175% のモニターで、§2 の 5 の型 A〜F の大きさ（窓・pane・ダイアログの大きさ、部品の大きさ、表の列幅、sizer の枠と間隔、View の余白と線の太さ）が、今（100% で描いて 1.5 倍・1.75 倍に引き伸ばした見た目）と同じ大きさになる。100% のモニターでは、今と同じ画素の大きさのままである。
- G3: Logs の字下げが、表示の拡大率と、選ばれたフォントによらず、本文の頭にそろう。ログ欄の幅が本文より前の部分（時刻と種類）を 1 行に収められないほど狭い間は対象外で、収められる幅になった時点から成り立つ（§5.4）。
- G4: 150% と 175% のモニターの間で窓を動かしても、G2・G3 が動かした先の DPI で成り立つ。

**非ゴール**

- 見た目の意匠を変えること。消える案は、PMv2 にするのに合わせて、アイコンや View の絵を高い解像度向けに描き直す案である。本当の制約ではない（描き直すことはできる）が、ツールバーのアイコンはすでに SVG から作っていて、別の関心である。
- OS のテーマの切替の扱い。消える案は、DPI の変更もテーマの切替と同じ入口（RunnerFrame の 1 か所）で受ける案である。本当の制約ではない（同じ入口で受けることもできる）が、[999_runner_theme_switch](999_runner_theme_switch_10prd.md) が DPI を分けると決めていて（§2 の 8）、別の関心である。
- Runner 以外の実行ファイル。消える案は、Metrics Viewer なども同じ見直しの対象にする案である。本当の制約ではないが、Metrics Viewer は Java の別のアプリで、ほかの C++ の実行ファイルは窓を持たない。
- Linux・macOS で確かめること。消える案は、ほかの OS でも受入を通す案である。本当の制約ではない（wx の汎用の `FromDIP` で書く）が、Runner がそこで未検証である（[040 開発環境構築ガイド](../design/040_development_environment.jp.md)）。

**成功の測り方**

- G1: §7 の受入 1・2 が通る。
- G2: 書き方は、受入 4（検索）で、型 A〜F の全箇所が DIP で書かれていることを確かめる。画素の大きさは、次のとおり分けて確かめる。
  - 150% で合否にするもの: 受入 3（frame の初期の大きさ、pane の推奨と最小の大きさ、表の列幅と最小の列幅と行見出しの幅、ヒートマップの推奨幅。自動）。
  - 175% で合否にするもの: 受入 7（表の列幅）。175% の frame と pane の大きさは、wx の既定の追従（AUI が DPI の変更で拡縮する。§2 の 6）に任せるので、合否で確かめない。
  - 150%・175% とも合否にせず、書き方（受入 4）を確かめてスクリーンショットを記録するだけのもの: ダイアログの大きさ（workspace の選択のダイアログは中身に合わせて大きさが決まり、期待値を値から決められない。§2 の 5）、pane を浮かせたときの大きさ、部品の大きさ、sizer の枠と間隔、View の余白と線の太さ（数える目印が View ごとに違う。§9 の 9）。
  - 100% は合否で確かめない（`FromDIP` は 100% では値を変えないので、差が出るのは実装の漏れに限られる）。受入 8 は任意の確かめで、合否に入れない。
- G3: 受入 6 が通る。G4: 受入 7 が通る。
- あわせて、受入 5（既存の GUI テスト）・9（学習の速さ）・10（文書）が通る。
- 150% の機械で、受入 5 の範囲の GUI テスト（`[log_panel]` と `[prd110_toolbar]` を含み、複数の RunnerApp を 1 process で起こすまとめのタグを除く）が全部通る（今は `[log_panel]` が落ち、gdiScaling を外すだけでは `[prd110_toolbar]` が落ちる。§2 の 4・7）。

## 4. 確定事項（暫定の既定。会話で合意した決定ではない。§9 とあわせてグリルで見直す）

- **D1. manifest から gdiScaling の行を消し、PMv2 の宣言だけにする。** 理由: G1 の前提で、gdiScaling が PMv2 の宣言を上書きしていることを確かめた（§2 の 2）。Microsoft は GDI スケーリングを DPI 非対応のアプリの互換の手段とし、Per-Monitor 対応のアプリには勧めていない（§2 の 3）。依存: なし。
- **D2. 画面の大きさを表す px の直書き（§2 の 5 の表）は、100% のときの大きさ（DIP）で書き、使う時点でその窓の `FromDIP` で実際の画素に直す。** 理由: wx の勧めるやり方で（§2 の 6）、100% では今と同じ画素のまま変わらず、150%・175% では今の見た目と同じ大きさになる（G2）。使う時点で直せば、DPI が変わったときに値を持ち直さずに済む（D3）。依存: D1（gdiScaling が残ると窓の DPI が 96 のままで、`FromDIP` は何もしない）。
- **D3. DPI の変更は、窓・フォント・AUI の pane の大きさを wx の既定の追従に任せ、アプリが画素で覚えている値だけを、それを持つ部品が `wxEVT_DPI_CHANGED` を受けて決め直す。** 理由: G4 のために要る。wx は窓とフォントと AUI の pane の大きさを自分で直す（§2 の 6）ので、アプリが受けるのは wx が知らない値だけで足りる。Evaluation Trace はすでにこの形で受けている（§2 の 5）。テーマの切替の入口とは分ける（§2 の 8）。依存: D2（使う時点で直す値は覚えないので、決め直す対象から外れる）。
- **D4. Logs の字下げの幅は、ログ欄（RichEdit）自身が描いた位置から測る。最後に足した記録の、行頭から本文の頭までの横の距離を使う。測れないとき（記録がまだ無い、行頭と本文の頭が同じ表示行に無い、本文の頭の座標が取れない。§5.4 の 2）は、今のとおり GDI で測った幅で仮に置き、測れるようになった時点で測った幅に置き換える（§5.4 の 3）。** 理由: 測る側と描く側が別の仕組みである限り、GDI スケーリングやフォントの代替で再び食い違いうる（§2 の 4）。D1 だけでも今の機械ではテストが通るが、D1 と独立に字下げをそろえる（G3）。ユーザーが推奨案として選んだ（冒頭の起点）。依存: なし。
- **D5. 描画の重さは合否にせず前後で測って記録し、学習の速さだけを受入にする（§5.7・§7 の受入 9）。** 理由: 描く画素数は 150% で 2.25 倍、175% で約 3.06 倍になるが、描画は UI のスレッドで行い、頻度は表示の FPS の設定で抑えられ、学習のスレッドとは別である。学習に響くかだけを外から確かめる（[AGENTS.md](../../AGENTS.md) §基本方針の、性能への影響に妥協しない方針）。依存: D1。
- **D6. 自動のテストは、既定のモニター（この機械では 150%）で回す。175% への移動と、100% の表示は、人が確かめる。** 理由: GUI テストの窓は既定のモニターに出る。表示の設定を変えるのは人で、AI はシステムの設定を変えない。依存: なし。
- ADR は起こさない。manifest の 1 行で戻せ、PMv2 と `FromDIP` は Microsoft と wx の勧めるやり方で、文脈が無くても意外ではない（ADR の 3 条件のうち、戻しにくい・意外の 2 つを満たさない）。

## 5. 実装契約

### 5.1 manifest

- [runner.manifest](../../apps/runner/src/runner.manifest) から `gdiScaling` の行を消す。`dpiAwareness` の `PerMonitorV2` は残す。`AnetRLRunner` と `AnetRLRunner-gui-test` は同じ manifest を使うので、両方が変わる（§2 の 1）。

### 5.2 寸法を DIP で書く

§2 の 5 の型 A〜F の全箇所（同じ節の除外を除く）を、100% の値のまま DIP として書き、使う時点でその窓の `FromDIP` で実際の画素に直す。名前つきの定数はそのまま持ち、使う箇所で直す。

| 種類 | 対象 | 直し方 |
|---|---|---|
| 窓・ダイアログ・pane の大きさ | RunnerFrame の初期の窓と各 pane の推奨・最小・浮かせたときの大きさ（Reset Layout でかけ直す値を含む）、Conv2d・HeatMap・Atari の初期の大きさ、エラー・workspace の選択・HeatMap の設定のダイアログの大きさと最小の大きさ | `FromDIP` で直す |
| 部品の大きさ | エラーのダイアログの文字欄の高さ、workspace の選択のリストの高さ、HeatMap の step の欄の幅 | `FromDIP` で直す |
| sizer の枠と間隔 | 型 D の 37 か所と型 F の 1 か所（列の数は直さない） | `FromDIP` で直す |
| 主 pane の幅の規則が使う最小幅 | RunnerFrame の Train・Eval の最小幅（200）と Logs の高さ（200） | 使う時点で `FromDIP` で直す（[PRD110](done/110_runner_main_pane_dynamic_centre_10prd.md) の等分の規則の 200+200 を含む） |
| 表の列幅 | QValuePanel の列幅と最小幅、行見出しの幅、ヒートマップの推奨幅 | `FromDIP` で直し、DPI の変更で決め直す（§5.3） |
| GDI で描く View の余白と線の太さ | LunarLander・DropMerge の余白、型 E の `wxPen` の太さ 31 か所（QValuePanel のヒートマップの 1 か所を含む） | 描くたびに `FromDIP` で直す。線は 1px を下回らない |
| GL で描く View の余白 | Atari の余白（10） | 実際の画素で計算している（§2 の 5）ので、`GetContentScaleFactor` を掛ける（§6 の 2） |
| ツールバーのアイコンの大きさ | RunnerFrame と Evaluation Trace の 16×16 | 段 0（§5.6）で wx が DIP として扱うかを確かめ、扱わなければ `FromDIP` で直す |

Conv2d・HeatMap の Panel に渡す初期の大きさは DIP に直すが、開いたときの pane の大きさはこの値ではなく pane を足す処理が決める（[RunnerFrame.cpp:1389・1420](../../apps/runner/src/RunnerFrame.cpp) の注記）ので、受入では確かめない。Atari の View の初期の大きさも、Train・Eval の pane の大きさに従うので同じである。

変えないもの: Env の盤面の座標を画面へ写す倍率（窓の大きさから計算している）、Atari の画像を整数倍で拡大すること（実際の画素で整数倍にするのでくっきり描ける）、pt で指定したフォントの大きさ（wx が直す。§2 の 6）、型 A〜F に当たらない View の絵の中の長さ（矢印の長さなど。§8）。

### 5.3 DPI の変更への追従

| 部品 | 画素で覚えている値 | DPI の変更で行うこと |
|---|---|---|
| RunnerFrame | なし（§5.2 で使う時点で直す） | wx の既定（AUI が pane の大きさを拡縮する。§2 の 6）に任せる |
| QValuePanel | 表の列幅 | 段 0 で wxGrid が自分で拡縮するかを確かめる。しなければ、`wxEVT_DPI_CHANGED` で列幅を `FromDIP` の値にかけ直す |
| LogPanel | 字下げの幅 | `wxEVT_DPI_CHANGED` で測り直す（§5.4。字形の幅は拡大率に比例しないため） |
| Evaluation Trace | 列幅など | 今のまま（§2 の 5） |
| Env の View・PlotPanel | なし（描くたびに直す） | なし。次の描画で新しい DPI になる |

### 5.4 Logs の字下げ

1. 幅: ログ欄に最後に足した記録の、行頭から本文の頭までの横の距離を、ログ欄が描いた文字の座標で測ったもの。行頭と本文の頭が同じ表示行にあるときだけ使う。本文より前の部分の文字数は、どの記録でも同じである（時刻と種類の書式は記録によらない。[LogPanel.cpp:138-144](../../apps/runner/src/LogPanel.cpp)）。
2. 仮の幅: 測れないとき、今のとおり GDI で測った幅を使う。測れないのは、記録がまだ無いとき、行頭と本文の頭が同じ表示行に無いとき（ログ欄が狭く、本文より前の部分の途中で折り返したとき）、本文の頭の座標が取れないとき（最後の記録が本文より前の部分だけで、後ろに文字が無いとき）の 3 つである。
3. 置き換えと測り直し: 仮の幅の間は、記録を足すたびと、ログ欄の大きさが変わるたびに測り、測れたら測った幅に置き換えて、それ以後は追記で測らない。測った幅は、Clear のあと（記録が無くなるので仮の幅に戻る）と、DPI の変更（`wxEVT_DPI_CHANGED`）を受けたときに測り直す。
4. 単位の変換と当て方は今のまま（1/10 mm に直し、既にある行には全体への 1 回の指定、これから足す行は追記ごとの既定の書式。[160](../design/160_applications_and_tools.jp.md) §3.5）。

### 5.5 変えないもの

- 100% の表示では、すべての寸法が今と同じ画素のままである（`FromDIP` は 100% では値を変えない）。配置・大きさ・振る舞いは今と同じになる。
- [PRD110](done/110_runner_main_pane_dynamic_centre_10prd.md) の主 pane の規則（Centre の占有、等分の条件、200ms の宣言、保険の WARN）と、その GUI テストが確かめる中身（判定）。テストの中の固定の寸法を `FromDIP` で直すことは、判定を変えないので行ってよい（§6 の 6）。
- Close の順序（[160](../design/160_applications_and_tools.jp.md) §7.1）、Train View を隠しても学習を止めないこと、表示の FPS の扱い。
- 設定のキー。足しも変えもしない。
- 学習と評価。描画は UI のスレッドだけで行い、Runner・Env・Agent に触れない。

### 5.6 段 0 の確かめと分かれ方

実装の最初の段で、gdiScaling を外したビルドを使って次の 4 つを確かめる。4 つ目の描画の時間を測るために、段 0 の最初に §5.7 の計測の範囲（§6 の 5）だけを先に足す。段 0 はこの 4 つで終わり、§5.2 以降の直しを待たない。

| 確かめること | 分かれ方 |
|---|---|
| wx のツールバーがアイコンの大きさ（16×16）を DIP として扱うか | 扱えば今のまま。扱わなければ `FromDIP` で直す |
| wxGrid が DPI の変更で列幅を自分で拡縮するか（QValuePanel） | すれば §5.3 の決め直しを足さない。しなければ足す |
| DropMerge の View が描くたびに作る pt のフォントが、描く窓の DPI の大きさになるか | なれば今のまま。ならなければ、その窓の DPI でフォントを作る |
| 150% と 175% での描画 1 回の時間（§5.7 の見積もりの t。測り方は §5.7） | §5.7 の分かれ方による |

段 0 とは別に、§5.2 の直しのあとで `[prd110_toolbar]`（受入 5）が通らなければ、Panels ツールバーの pane の大きさの決め方を調べ、本書の実装の漏れとして直す。

### 5.7 性能

- 描く画素数は 150% で 2.25 倍、175% で約 3.06 倍になる。対象は GL の 3 つと、バッファに描いてから写す View である（§2 の 5）。描画は UI のスレッドで行い、頻度は Train・Eval の表示の FPS（既定の値と実行時の選択）で決まる。学習のスレッドの step のループと学習の更新には触れない。
- 計測の範囲の無い Env の View の `OnPaint` と PlotPanel の描画に、計測の範囲を足す（§6 の 5）。前後の描画の時間を Tracy で記録する（合否にしない）。
- 見積もり（exp step あたり）: 主な描画は Train View と Evaluation View で、頻度の上限は表示の FPS の既定の 30 回／秒ずつ（[common.txt:135・138](../../apps/runner/config/common.txt) の `app.online.*`。バッチの Run の Train View は 0。同 144 行）、合わせて 60 回／秒である。描画 1 回の UI のスレッドの時間を t、学習の速さを s（exp step／秒）とすると、exp step あたりに UI のスレッドが描画に使う時間は 60 × t ÷ s になる。t は今は測っておらず（Env の View の描画に計測の範囲が無い）、PMv2 では描く画素数の分だけ（150% で 2.25 倍、175% で約 3.06 倍）増えうる。この時間は UI のスレッドのもので、学習のスレッドを直接止めない。学習に響くのは、CPU と GPU（GL の 3 つ）を取り合う分だけで、その割合は見積もれないので、受入 9 で学習の速さを直接比べる。
- t の測り方（段 0）: 受入 9 と同じ起動の形・同じ設定（LunarLander、対話の形、`app.main_view=both`）で、150% と 175% のモニターに置いた Runner を Tracy につなぎ、`LunarLanderPanel::OnPaint` の 1 回あたりの平均の時間を、それぞれ 1,000 回以上の描画から取る。Train View と Evaluation View はどちらも LunarLander の View なので、この 1 つの計測の範囲に両方の描画が入る。Atari の View（GL）も同じように測って記録するが、分かれ方には使わない（受入 9 が LunarLander で比べるため）。
- 分かれ方（段 0 の t による）: 175% で 60 × t が 1 秒の 10%（t が約 1.7 ms）を超えたら（分かれ目は §9 の 8）、残りの実装の前に受入 9 の比べ方で学習の速さを先に測り、閾値（§9 の 1）を下回れば、描画の頻度か画素数を抑える案を §9 に足して止める。超えなければそのまま進める。
- Logs の追記（記録 1 件ごとに通る）に足すのは、測った幅が決まっていれば「仮の幅か」の比較 1 つだけである。仮の幅の間（起動直後・Clear のあとの最初の記録まで、ログ欄が狭い間）だけ、追記ごとに文字の座標を 2 つ問い合わせる。記録の数は学習のログ（100 learn step ごと。[RunnerApp.cpp:922](../../apps/runner/src/RunnerApp.cpp)）などで、exp step あたり 1 件を大きく下回る。測れたときの置き換えの重さは、ログ欄の全体への字下げの 1 回の指定で、今の Indent の切替と同じである。
- 計測の範囲は描画 1 回に 1 つで、頻度は上の 60 回／秒以下である。
- 学習の速さは §7 の受入 9 で比べる。

### 5.8 移行対象

- [runner.manifest](../../apps/runner/src/runner.manifest): §5.1。
- [RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp): 初期の窓、pane の大きさ、主 pane の最小幅、ツールバーのアイコンの大きさ（段 0 による）。
- [QValuePanel.cpp](../../apps/runner/src/QValuePanel.cpp)・[QValuePanel.hpp](../../apps/runner/src/QValuePanel.hpp): 列幅・行見出しの幅・ヒートマップの推奨幅と線の太さ・sizer の枠、DPI の変更（段 0 による）。
- [LogPanel.cpp](../../apps/runner/src/LogPanel.cpp)・[LogPanel.hpp](../../apps/runner/src/LogPanel.hpp): §5.4 と、DPI の変更での測り直し、sizer の枠。
- [Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp)・[HeatMapPanel.cpp](../../apps/runner/src/HeatMapPanel.cpp)・[ErrorDialog.cpp](../../apps/runner/src/ErrorDialog.cpp)・[WorkspaceDialog.cpp](../../apps/runner/src/WorkspaceDialog.cpp)・[EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp)・[TrainPanel.cpp](../../apps/runner/src/TrainPanel.cpp)・[PlotPanel.cpp](../../apps/runner/src/PlotPanel.cpp)・[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)（アイコンの大きさ。段 0 による）: §2 の 5 の型 A〜F のうち、そのファイルにあるもの。
- Env の View: [AtariView.cpp](../../core/envs/atari1/src/AtariView.cpp)・[LunarLanderView.cpp](../../core/envs/lunarlander1/src/LunarLanderView.cpp)・[DropMergeView.cpp](../../core/envs/dropmerge1/src/DropMergeView.cpp)（フォントは段 0 による）・[CartPoleView.cpp](../../core/envs/cartpole2/src/CartPoleView.cpp)・[GridMazeView.cpp](../../core/envs/gridmaze1/src/GridMazeView.cpp)。
- テスト: 受入 1・3・6 のテストを足す。PMv2 で落ちる既存のテストの固定の寸法（[LogPanel_test.cpp:24](../../apps/runner/src/LogPanel_test.cpp) の 900×400、PRD110 のテストの [RunnerFrame_test.cpp:400・404](../../apps/runner/src/RunnerFrame_test.cpp) の幅など）は、落ちたものだけ `FromDIP` で直し、判定（確かめる中身）は変えない（§5.5・§6 の 6）。
- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md): §2.1 に、Runner は PMv2 で動き、画面の大きさは DIP で書いて窓の DPI で直し、DPI の変更は wx の追従と、画素で値を覚える部品の決め直しで受けることを足す。§3.5 の「幅は時刻と種類の文字数を固定幅フォントで測り、1/10 mmに直す」を §5.4 の測り方に直す。§8 の Runner UI の確かめに、受入 3・6 のテストのタグと、人が行う受入 2・4・7・8・9 の手順の所在を足す。英語版（[160.en](../design/160_applications_and_tools.en.md)）は翻訳の手順で追う。
- 正式に採番するときに、`core/anet-core/testdata/prd<番号>/README.md` を作り、段 0（§5.6 の 4 つ）と受入 2・4・7・8・9 の手順と結果の表を置く。160 §8 から参照する。
- 互換層は置かない。gdiScaling を残す経路や、DPI 非対応で動かす切替を設けない。

### 5.9 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RunnerFrame` | 変更 | 初期の窓・pane の大きさ・主 pane の最小幅を DIP で持ち、使う時点で直す。ツールバーのアイコンの大きさも、wx が DIP として扱わなければ同じく直す（段 0 による） | RunnerFrame |
| `EvalTracePanel` | 変更 | 操作帯のツールバーのアイコンの大きさを、wx が DIP として扱わなければ `FromDIP` で直す（段 0 による。扱えば変えない） | EvalTracePanel |
| `LogPanel` | 変更 | 字下げの幅をログ欄の描いた位置から測り、測れるまでは仮の幅に置き、Clear のあと・DPI の変更で測り直す。sizer の枠を DIP で持つ | LogPanel |
| `QValuePanel` | 変更 | 表の列幅・行見出しの幅・ヒートマップの推奨幅と線の太さ・sizer の枠を DIP で持ち、DPI の変更で列幅をかけ直す（段 0 による） | QValuePanel |
| `Conv2dPanel`・`SweepHeatMapPanel`・`SweepHeatMapDialog` | 変更 | 初期の大きさ、ダイアログの大きさ、step の欄の幅、sizer の枠と間隔を DIP で持つ | Conv2dPanel・HeatMapPanel |
| `WorkspaceDialog` | 変更 | 大きさ・最小の大きさ・リストの高さ・sizer の枠を DIP で持つ | WorkspaceDialog |
| `EvalPanel`・`TrainPanel` | 変更 | sizer の枠を DIP で持つ | EvalPanel・TrainPanel |
| `PlotPanel` | 変更 | 線の太さを DIP で持つ。描画に計測の範囲を足す | PlotPanel |
| `AtariPanel`・`LunarLanderPanel`・`DropMergePanel`・`CartPolePanel`・`GridMazePanel` | 変更 | 余白と線の太さを DIP（GL は実際の画素への倍率）で持つ。`OnPaint` に計測の範囲を足す | 各 Env の View（AtariView・LunarLanderView・DropMergeView・CartPoleView・GridMazeView） |

エラーのダイアログはクラスを持たない（[ErrorDialog.cpp](../../apps/runner/src/ErrorDialog.cpp) の関数の中で作る）ので、一覧に入れず、§5.8 で扱う（大きさ・文字欄の高さ・sizer の枠を DIP で書く）。

## 6. 細部（暫定で置いた既定）

1. 名前つきの定数は 100% の値（DIP）のまま持ち、使う箇所で `FromDIP` を呼ぶ（理由: DPI の変更のたびに値を持ち直さずに済む。D2・D3）。
2. GL の余白は、`GetContentScaleFactor` を掛けた値を整数に丸めて使う（理由: GL の描画はもともと実際の画素で計算している。§2 の 5）。
3. 線の太さは `FromDIP` の値と 1 の大きい方にする（理由: 細い線が消えないようにする）。
4. Logs の字下げを測る時機は §5.4 の 3 のとおりとする（理由: 測った幅が変わりうるのは DPI の変更だけである（ログ欄のフォントは固定）。仮の幅は、測れる状態（記録が入った、欄が広がった）になった時点で置き換える。測れたあとは追記ごとに測らない。追記は頻繁な経路である）。
5. 計測の範囲の名前は `ClassName::OnPaint`（`ANET_PROFILE_FUNC`）にする（理由: [実装 品質確認観点](../quality/impl.md)「計測」の名前の形）。
6. 既存のテストの固定の寸法は、PMv2 で落ちたものだけを `FromDIP` で直す（理由: 100% では値が変わらないので、落ちないテストを触る理由が無い）。
7. 受入 1 のテストのタグは `[dpi_awareness]` とし、既定のスイートに入れる（理由: RunnerApp を起動せず、窓 1 つで確かめられる）。受入 3 のテストのタグは `[dpi_layout]` とし、RunnerApp を起動する別の process で、明示したときだけ走らせる（理由: 既存の `[prd110_*]` と同じく RunnerApp を起動するため）。

## 7. 検証と受入

**使う場面**: 150% と 175% のモニターを並べた機械で Runner を使い、窓をモニターの間で動かす。

**受入（外から観測できるもの）**

1. DPI 対応（自動）: `AnetRLRunner-gui-test "[dpi_awareness]"` が、自分の process の DPI 対応が Per-Monitor（awareness 2）であり、作った窓の `GetDPI` が、その窓のあるモニターの DPI と同じであることを確かめて通る。
2. DPI 対応（手動。人が行う）: 実の `AnetRLRunner.exe` を起動し、外から `GetDpiAwarenessContextForProcess` で読んだ awareness が 2 で、窓の DPI がモニターの DPI（144 か 168）である。読み方は §2 の 2 と同じ。
受入 3・7 の「期待値」は、§2 の 5 の 100% の値に拡大率（150% なら 1.5、175% なら 1.75）を掛けて丸めた画素である。比べるときの許容は ±1px とする（丸め方の違いを吸収するため）。

3. 寸法（自動。150% の機械）: `AnetRLRunner-gui-test "[dpi_layout]"` を、RunnerApp を起動する別の process で回す。frame の初期の大きさ、起動時にある各 pane（Logs・Train View・Evaluation View・Q-Values・Evaluation Trace）の推奨と最小の大きさ、Q-Values の表の列幅と最小の列幅と行見出しの幅、ヒートマップの推奨幅が、期待値と一致する。
4. 書き方（自動。検索）と記録（手動。合否にしない）: 実装後のツリーで、§2 の 5 と同じ 38 ファイルを、下の正規表現（拡張正規表現。`grep -E` で使える）で検索する。合否の条件は次の 2 つである。
   - 当たった箇所が、`FromDIP` の引数の中にあるもの（型 A の `FromDIP(wxSize(...))` など）か、§2 の 5 の除外に当たるもの（段 0 で DIP と分かった場合のツールバーのアイコンの大きさを含む）だけになる。
   - 型 C の名前を、窓の大きさとして渡す箇所（bitmap bundle に渡す大きさを除く）と、型 E の余白の変数を使う箇所が、すべて `FromDIP` を通る（Atari の余白は `GetContentScaleFactor` の倍率。ツールバーのアイコンの大きさは段 0 で DIP と分かればそのまま）。型 C の名前は `kDefaultLogDockHeight`・`kMinTrainWidth`・`kMinEvalWidth`・`kToolBitmapSize`（RunnerFrame のもの）・`kActionNameColWidth`・`kActionNameColMinWidth`・`kColWidth`・`kColMinWidth`・`kPreferredHeatMapWidth` である。

   検索の結果（型ごとの当たりと、その扱い）を手順の README の表に記録する。正規表現は次のとおり（行頭の記号は型の名前で、正規表現に含めない）。

   ```text
   A:  wxSize\( *-?[0-9]+ *, *-?[0-9]+ *\)
   B1: (SetMinSize|SetMaxSize|SetSize|SetClientSize|SetInitialSize|BestSize|MinSize|MaxSize|FloatingSize|SetDefaultColSize|SetDefaultRowSize|SetColLabelSize|SetRowLabelSize|AddSpacer|SetToolBitmapSize)\( *[0-9]
   B2: (SetColSize|SetColMinimalWidth|SetRowSize|SetRowMinimalHeight)\([^,]*, *[0-9]
   D:  (wxALL|wxLEFT|wxRIGHT|wxTOP|wxBOTTOM)[^,;]*, *[0-9]+ *\)
   E:  wxPen( +[A-Za-z_][A-Za-z_0-9]*)? *\( *[^(),]*(\([^()]*\))? *, *[0-9]+ *[,)]
   F:  (wxFlexGridSizer|wxGridSizer)\( *[0-9]+ *, *[0-9]+ *, *[0-9]+
   ```

   B1 は大きさを最初の引数に取る関数、B2 は 2 つ目に取る関数（最初の引数は列か行の番号）である。E は、`wxPen(...)` と変数を宣言する形の `wxPen 名前(...)` の両方で、色の引数（括弧を 1 組まで含む式。`wxColour(r, g, b)` やシステムの色を返す関数の呼び出し）のすぐ後ろの太さの数値だけに当たり、色の数値と、`FromDIP` を通した太さには当たらない。F は列の数の後ろの間隔の数値に当たる。2026-11-02 の当たりは §2 の 5 のとおり（B2 は 0）である。

   記録は、150% と 175% の各モニターで、workspace の選択・エラー・HeatMap の設定の各ダイアログ、浮かせた Logs の pane、LunarLander・DropMerge・Atari・CartPole・GridMaze の View と PlotPanel のスクリーンショットを、今の実行ファイルと新しい実行ファイルで撮り、手順の README に置く。エラーのダイアログは、起動時に止まる設定の誤り（たとえば `app.main_view` に列挙に無い値）で出す。これらの画素の大きさは合否にしない（§3 の成功の測り方の G2、§9 の 9）。
5. 既存の GUI テスト（自動。150% の機械）: 既定のスイート、`"[log_panel]"`、RunnerApp を起動する隠しテストのタグ（`[prd109_*]`・`[prd110_*]`・`[eval_steps_menu]`・`[prd105_ui_keys]`・`[dpi_layout]`）を 1 タグ 1 process で回し、すべて通る（`[prd110_main_pane]` のように複数の RunnerApp を 1 process で起こすまとめのタグは除く）。`[prd110_toolbar]` を含む。
6. 字下げは測り方によらない（自動）: 次の 2 つを確かめるテストを `[log_panel]` に足す。1 つ目は、ログ欄を本文より前の部分より狭くして最初の記録を足し、そのあと欄を広げると、折り返した 2 行目が本文の頭にそろう（±2px）こと。2 つ目は、`wxDPIChangedEvent` を LogPanel へ送ったあとも、折り返した 2 行目が本文の頭にそろうこと。足した 2 つと既存の字下げのテスト（[LogPanel_test.cpp:357-391](../../apps/runner/src/LogPanel_test.cpp)）が、新しい実行ファイルと、gdiScaling を戻した写し（新しい実行ファイルの manifest に、`mt.exe` で今の manifest を入れ直したもの）の両方で通る。写しで回すのは、新しい実行ファイルでは仮の幅（GDI で測った幅）と描いた幅が一致し、測った幅への置き換えの漏れが見えないためである。1 つ目のテストで欄の幅を変えるのは、テストの frame の大きさを公開の API で決める準備であり、利用者が窓の端をドラッグして欄の幅を変える操作そのものは受入 7 で人が確かめる。2 つ目のテストでイベントを送るのも、測り直しの経路を通すための準備であり、窓をモニターの間で動かす操作そのものは受入 7 で人が確かめる。
7. モニターの間の移動（手動。人が行う）: Logs に折り返した行があり、Q-Values の表と Train・Eval の View を出した Runner を、150% のモニターから 175% のモニターへ動かし、また戻す。動かした先ごとに、スクリーンショットの画素で次を確かめる。Logs の折り返した 2 行目と本文の頭の横の位置の差が ±2px 以内である。Q-Values の表の列幅が、その拡大率の期待値と一致する。Train View と Evaluation View の幅の差が 2px 以内である（[PRD110](done/110_runner_main_pane_dynamic_centre_10prd.md) の等分の規則が当たる配置で）。あわせて、Logs の Clear log を押して記録を無くし（字下げの幅が仮の幅に戻る。§5.4 の 3）、窓の端をドラッグして Logs のログ欄を本文より前の部分より狭くしてから記録を出し、そのあとで窓を広げると、折り返した 2 行目と本文の頭の横の位置の差が ±2px 以内になる。結果を手順の README の表に記録する。
8. 100% の表示（手動。任意で、合否に入れない。人が行う）: 表示の拡大率を 100% にした機械で、今の Runner と新しい Runner を同じ設定で起動し、窓・pane・ダイアログの大きさが同じ画素であることをスクリーンショットで比べる。
9. 学習の速さ（自動に近い手動）: 150% のモニターで、LunarLander の既定の設定の Runner を対話の形で起動する（[apps/10_run.bat](../../apps/10_run.bat) と同じ既定の設定の連鎖で、`app.online.*` が効く。Train View と Evaluation View の表示の FPS はどちらも既定の 30 で、§5.7 の見積もりと同じ前提である。バッチの形の `app.$=app.batchrun` には替えない）。`run.seed=1`、`app.main_view=both`（Train View と Evaluation View を出す）、`app.online.exp_exit_step=200000`（profile と同じ接頭辞で渡す）を渡す。学習は既定のとおり自動で始まる（`app.train_auto_start` の既定は true。[common.txt:122](../../apps/runner/config/common.txt)）。この Run を、今と新しい実行ファイルで交互に 3 巡ずつ（各 3 Run）、同時には 1 本だけ走らせる。Run ごとに `90_perf/12_exp_step_per_sec` の exp step 100,000〜200,000 の点の平均を取り、群ごとの 3 Run の中央値の比（新÷今）が §9 の 1 の閾値以上である。描画の時間（§5.7）は記録だけする。
10. 文書: 160 §2.1・§3.5・§8 が §5.8 のとおり直っている。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| gdiScaling を残す（今のまま） | 不採用 | DPI の宣言が上書きされ、字幅の食い違いとぼやけが残り、manifest と実際の動きが違う（§2 の 2〜4） |
| システムの DPI だけに対応する宣言（System aware） | 不採用 | 起動したときの 1 つの DPI でしかくっきり描けず、150% と 175% の間で窓を動かすとぼやける（§2 の 3） |
| Per-Monitor（V1）の宣言 | 不採用 | 子のウィンドウに DPI の変更が通知されない（§2 の 3） |
| gdiScaling を外すだけにし、字下げは GDI で測ったままにする | 不採用 | 測る側と描く側が別の仕組みのままで、フォントの代替などで再び食い違いうる（D4） |
| DPI の変更もテーマの切替と同じ入口（RunnerFrame の 1 か所）で受ける | 範囲外 | [999_runner_theme_switch](999_runner_theme_switch_10prd.md) が分けると決めた（§2 の 8）。再開条件: 両方を実装したあとで、DPI の変更の反映が半端になる例が出たとき |
| アイコン・View の絵を高い解像度向けに描き直す | 範囲外 | 別の関心。再開条件: PMv2 にしたあとで、ぼやけて見える絵が見つかったとき |
| §2 の 5 の型 A〜F に当たらない、View の絵の中の長さと位置（矢印の長さ、Atari の待ちの表示の文字の位置など）を DIP にする | 範囲外 | G2 は型 A〜F の大きさまでとした。再開条件: PMv2 にしたあとで、小さく見える絵の部品が見つかったとき |
| Runner 以外の実行ファイルの見直し | 範囲外 | Metrics Viewer は Java の別のアプリ。再開条件: そちらで同じ問題が見つかったとき |
| Linux・macOS での確認 | 範囲外 | Runner がそこで未検証。再開条件: Linux で Runner のビルドが通ったとき |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | 学習の速さの受入の閾値と比べ方（受入 9） | 受入 9 の比べ方で、中央値の比が 0.97 以上（描画は UI のスレッドで、学習への影響は小さい見込み。交互の 3 巡の中央値は、時間とともに動く機械の速さの揺れを抑える） |
| 2 | 線の太さも DIP にするか（§5.2） | する（175% では今の 1px の線が細く見える） |
| 3 | 100% の表示での確認を受入の合否に入れるか（受入 8） | 入れない（任意の手動。`FromDIP` は 100% で値を変えないので、差が出るのは実装の漏れに限られる） |
| 4 | 段 0 の 4 項目（ツールバーのアイコン、wxGrid の列幅、DropMerge のフォント、描画 1 回の時間）の結果ごとの直し方 | §5.6 の表と §5.7 の分かれ方のとおり |
| 5 | Logs の字下げを測る時機（§5.4 の 3・§6 の 4） | 仮の幅の間は追記ごとと欄の大きさの変化ごと。測れたあとは Clear のあとと DPI の変更のとき |
| 6 | 受入 1 のテストを既定のスイートに入れるか | 入れる（RunnerApp を起動しない。§6 の 7） |
| 7 | 段 0 を実装の最初の段で行うか、実装の前に別に行うか | 実装の最初の段で行う（gdiScaling を外したビルドがそのまま道具になる） |
| 8 | 段 0 で、学習の速さを先に測るかの分かれ目（§5.7） | 175% で 60 × t が 1 秒の 10%（t が約 1.7 ms）を超えたら先に測る（UI のスレッドの時間は、そのまま学習を遅らせるとは限らないので、学習の側の閾値の 3%（§9 の 1）より緩い 10% を UI のスレッドの側の目安にし、超えたときだけ、残りの実装の前に確かめる） |
| 9 | 受入 3・7 の外の大きさ（ダイアログ、pane を浮かせたときの大きさ、部品の大きさ、sizer の枠と間隔、View の余白と線の太さ）を、画素で合否に入れるか（G2） | 入れず、受入 4 で書き方を検索で確かめ、スクリーンショットを記録するだけにする（workspace の選択のダイアログは中身に合わせて大きさが決まり、View の余白は盤面を収める計算の中にあって数える目印が View ごとに違うので、値から期待値を決めきれない。グリルで、項目ごとの目印を決めて合否に入れるかを決める） |
| 10 | 受入 7（モニターの間の移動）に Evaluation Trace を足すか。[PRD094](done/094_evalpanel_step_history_10prd.md) は、DPI の変更への追従（同 PRD の D10）の確かめを、本 PRD へ移した（2026-11-02） | 足す。Evaluation Trace を開いて行がある状態で動かし、表の列幅と pane の最小幅がその拡大率の期待値と合い、見出し・アイコン・操作バーの枠が欠けないことを確かめる |

## 10. 依存と順序

- 先に要る PRD は無い。
- [999_runner_theme_switch](999_runner_theme_switch_10prd.md) とは独立で、どちらを先に実装してもよい。両方を実装したあとは、テーマの切替の決め直しと DPI の変更の決め直しが同じ pane（LogPanel・QValuePanel・Evaluation Trace）に別々の入口で届く。
- 同じ日に PRD なしで直した、構築に失敗した Runner を閉じたあとの落ちの直し（RunnerFrame の後回しの並べ直しのガード）とは独立である。

## 11. 参照

- [999_runner_theme_switch](999_runner_theme_switch_10prd.md)、[PRD110](done/110_runner_main_pane_dynamic_centre_10prd.md)
- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md) §2.1・§3.5・§7.1・§8
- [RunnerApp（wiki）](../wiki/apps/110_runner_app.md)
- [Improving the high-DPI experience in GDI based Desktop Apps](https://blogs.windows.com/windowsdeveloper/2017/05/19/improving-high-dpi-experience-gdi-based-desktop-apps/)、[High DPI Desktop Application Development on Windows](https://learn.microsoft.com/ja-jp/windows/win32/hidpi/high-dpi-desktop-application-development-on-windows)、[wxWidgets の High DPI overview](https://docs.wxpython.org/high_dpi_overview.html)
