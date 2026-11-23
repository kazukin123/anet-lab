# nn_trace capture 統合 実装メモ

## 概要
PRD170 の D1〜D18 と A1〜A13 に沿い、forward の名前付き出力を NetworkForwardCapture に統合し、MakeActionOptions.nn_trace を指定した step だけ nn_trace を作る。PRD は暫定ではなく、追加判断を要するブロッカーは 0。

## 主な変更
- NN: common.hpp の TraceCallback を削除し、nn.hpp に接頭辞を集約する。Network / NetworkForwardExecutor / ForwardUpTo の入口で capture を Context に入れ、RunForward・Body・Branch・Struct の callback と内部 capture 引数を除く。Body が clear / validate を行い、Head には capture を渡さない。Struct が要求に一致する nn_trace のみ書く。
- Actor: rl.hpp に MakeActionOptions（nn_trace=false）を置き、基底と全 override に空の既定値を付ける。DQN / Rainbow と ImageCls は行動 forward 直後に capture を読み、env 0 を detach・FP32 化・clone して同名の AuxData に積み、capture を解放する。MuZero は初回の要求のみ WARN を出す。
- Runner: DoStep / DoUpdateFrame が options を受け、その回の全 step に伝える。EvalRunner は DoStep(options={}) と DoStep(action, counts, options={}) を変更し、既存の DoStep(action) と DoStep(counts) は空の options を転送する。GUI の手動行動は action, GetCounts(), options の版を呼ぶ。DoStep(action, {}) の既存呼び出しを曖昧にしない。RunSession は空の options。
- 呼ぶ側: EvalPanel が RunnerFrame の IsPaneOpen と同じ判定で Conv2d pane の有無を確認し、タイマー・step・手動行動に渡す。RunnerApp は登録する画像記録の Conv2d interval が 1 以上かを ImageProviderManager から取得し、RunnerThread の構築時に固定して渡す。Observer / Notifier に要求の登録を作らない。
- テスト: 既存 trace テスト、Actor / Runner の差し替え部品（GUI 側を含む）を移行する。core/anet-core/testdata/prd170/ に README.md、prepare_inputs.py、validate.py、performance.py を置き、trace・6構成の scalar・throughput の再実行手順を残す。RunnerApp_test.cpp に補足の CUDA allocator 測定を hidden test として置く。測定だけで必要な同期は製品コードへ入れない。
- 文書: 設計100 §2.2と学習の図にRunnerのoptions伝達、設計110 §2.1・§5・§6.1にActorの第3入力と要求時だけのnn_trace生成、設計130の公開APIクラス図とcaptureの説明、設計140 §2のImage/Graph Observer行と§5.4の要求・通知のシーケンス図を更新する。100・110・130・140の対応する英語版も移行し、翻訳元情報を確認範囲に合わせる。CONTEXT と ADR0031 / 0055 / 0072 は着手時点で仕様を反映済みなので既存変更を保持する。CHANGELOG に契約変更を追記する。

## テスト
- Public interface / surface: Network::Forward / NetworkForwardExecutor::Forward / ForwardUpTo、Actor::MakeAction、Runner::DoStep / DoUpdateFrame、RunSession、RunnerFrame のメニュー・Q-Values クリック・AUI close・タイマー、RunnerApp の画像登録から学習通知まで。
- 優先 behavior / TDD 順序:
  1. 変更前に deterministic CPU の native_stack 5 構成・IQN tau branch・ImageCls の全 trace を辞書順、shape、dtype、全値で保存する（A1）。既定テスト一式と変更前実行体を保存する（A12・A11）。
  2. tracer bullet: DQN MakeAction の無指定呼び出しが trace を作らないテストを RED にし、options と capture の経路を接続して GREEN にする。既存 trace の要求指定・全 override のコンパイル移行は同じ契約変更の必要部分として行う。
  3. ImageCls の無指定 / false / true を RED→GREEN。既存 trace と CPU 全値の一致を確認（A1〜A3）。
  4. 各 Runner の通知と 3 step フレーム、Eval 手動行動、RunSession を 1 behavior ずつ RED→GREEN（A4・A5）。
  5. MuZero の false→true→true の WARN 0→1→1 を RED→GREEN（A8）。
  6. RunnerFrame の利用者イベントと pane 開閉・複数 pane・タイマー、画像登録の interval 正 / 0 / 無登録を RED→GREEN（A6・A7）。
- GREEN 後だけ局所整理し、関連テストを再実行する。

## 検証
- Windows の VsDevCmd.bat 経由で x64-Debug の本体・anet-core-test・AnetRLRunner-gui-test をビルドする。必要な性能測定用 Release も同経路でビルドする。
- 長時間の出力は .scratch/prd170/ のログへ標準出力・標準エラーとも保存し、画面には終了値と要約を返す。待機は 60 秒以下の区切りで行う。
- baseline / after の core 通常テストと GUI 通常テストを比較し、新規失敗 0 を確認する。trace 全値を完全一致比較する。
- PRD103 の 6 構成と短縮条件・比較規則を再利用し、着手時の設定コピーを両実行体へ同じ入力として渡す。履歴資材は改変せず、現行契約への必要な移行を prd170 側のコピーで行う。性能・資源の既存除外群以外は除外しない（A11）。
- 学習 throughput は変更前後を交互に 3 巡以上測り、群別中央値を記録する。ImageCls N128 と Atari で要求前後 100 step 以上の GPU メモリ増分を記録する。閾値は置かない（D18）。
- 現用 core/apps/CONTEXT/docs/design の旧識別子 0 件（A9）、Body / RunForward の引数（A10）、既存 attention / plasticity テスト、図の登場者と分岐（A13）、改行・BOM・差分範囲を点検する。
- 実画面の再生・pane 開閉・2 枚のうち 1 枚を閉じる・一時停止中の各入力・Trace 履歴とダンプを確認する。実施できない項目は理由と再開方法を記録する。
- docs/quality/impl.md の全項目を適用範囲に従い自己点検し、計画と実装を独立した anet-review-agent へまとめて依頼する。直せる指摘を直し、同じレビューへ出し直す。

## 前提
- 基準コミット: e2803e44425488bd7851ed176e8b6b8c42fa65ca。多くの既存変更がある。着手前ファイルを .scratch/prd170/before-source/、Git 状態を initial-status.txt に保存して今回の差分を区別する。既存の index と他作業の変更は保持する。
- 新しい設定キー・metric は作らない。nn_trace は branch の全入力を形によらず含み、描画できる block の番号・鍵・値を維持する。要求のない呼び出しでは文字列生成・clone を行わない。
- PRD161 の強制行動、Learner の購読の整理、calls 表の整理は範囲外。
- capture は同期 forward の借用で、呼び出す Actor が所有する。MuZero の WARN 済みフラグは Actor の State。新しい同期転送や他スレッドのロック待ちを毎 step の経路へ加えない。
- PRD に定められた public interface・受入の範囲を実装する承認は今回の skill 呼び出しで与えられている。追加判断で計画が実質的に変わる場合のみ確認して待つ。

## 決めたこと
- D1〜D18 を採用し、要求の事前登録と互換 callback は追加しない。構造と語は CONTEXT、ADR0072、PRD の通り。
- 手動行動 overload の options は counts の後に置く。空の波括弧を counts に渡す既存呼び出しの意味を保つため。

## 実装上の補足
- nn_trace を完全一致の鍵や branch の接頭辞で要求した場合も既存 capture 契約を維持するため、NetworkForwardCapture に汎用の WantsAny(prefix) を追加する。要求のない capture では鍵文字列を作らず、Context から capture と branch 位置を参照する。要求登録の機構ではなく、今回の同期 forward 内の照合だけである。
- 新しい GetCapture / GetCaptureBranch は既存の Context::HasCapture / WithCapture の持つ同期借用を読む。AppendNnTraceAux は既存の AppendTraceAux の処理を捕捉後の読み出しへ移す。毎 step の処理に CPU 読み戻し・追加のスレッド間ロックを増やさず、同一 device 上で lane 0 の detach・FP32・clone を行う。要求なしでは capture の入れ物も鍵も組み立てない。
- MuZero の WARN は PRD170 D12 に従う実装の欠けの通知（Fail-Fast 表の「実装の欠けによる機能の一部の省略」）で、Actor ごとに最初の true だけ出す。無指定の既定テストでは出ず、新規 Actor に false→true→true を渡すと 0→1→1 件になる。

## 経過
- 2026-11-17: PRD170、実装計画の観点、現行の NN / Actor / Runner 宣言、GUI テスト、対応課題、CONTEXT と ADR0072 を照合。ユーザー判断が必要なブロッカー 0。実装計画を保存。

## 決まらなかったこと
設計判断の未決は無し。実画面の確認と D18 の pane 開閉前後の GPU 測定は、Computer Use の操作許可が timeout したため未実施。既存 GUI テストの失敗は[issue-20261117-eedd0c](../management/issues/issue-20261117-eedd0c.md)に記録した。

## 検証と実装の記録
- 変更前 Debug ビルド成功（build-before.log）。通常 core 一式は 1023 cases（1022 pass / 1 skip）、196026 assertions 成功（core-before.log）。GUI 一式は 39 cases 中 38 pass / 1 fail、894 assertions 中 893 pass（gui-before.log）。失敗は既存の EvalTracePanel_test.cpp:576 の IsTraceMode 判定。
- A1: native_stack 5 構成、IQN、ImageCls の全鍵・shape・FP32 の全値を変更前実行体から保存。初回は IQN の重み初期化が torch の seed 未固定だったため値が揺れた。保存済み変更前 exe と変更後 exe の両方で、manual_seed(128) を行う既存 [prd128][dqn][probe-resubscribe] を辞書順で先に実行してから同じ7構成を採取すると、7 / 7 ファイル完全一致した。証拠: trace-before-seeded/、trace-after-seeded/。Agent seed=123、Catch --rng-seed 170 --order lex。値の除外は無し。
- TDD 1: DQN の無要求テストが失敗（red1.log、1 case / 1 assertion fail）→ capture と Actor 契約を移行→成功（green1.log、11 cases / 91 assertions）。型移行で不足していた NN capture の前方宣言を補った。
- TDD 2: ImageCls の無指定・false が失敗（red2.log、2 / 3 assertions fail）→ 要求時だけ capture を作る→成功（green2.log、3 cases / 18 assertions）。
- TDD 3: Runner の要求入力が未実装でコンパイル失敗（red3-build.log、DoStep 1 引数・DoUpdateFrame 4 引数・手動行動の3引数）→入力と転送を実装し検証中。
- A11: PRD103 の6構成の凍結入力を .scratch/prd170/validation/ へ複製。最初の GridMaze OFF は exit 0、26946 点、SHA256=2278b59c28b93e85d1adbfe203a1548a60589715fbdf576b029b1c9b3d800f98。LunarLander は過去の未読画像設定で現行の閉じたスキーマに拒否されたため、履歴原本を保持し、コピーだけから PRD117 で廃止された auto_scale_mode / histgram.interval を除いた。同じ移行済み入力を両実行体へ与える。

- 2026-11-17: TDD 3 は MetricsLogger の準備漏れを既存ガードで補い、各 Runner・3 step frame・手動行動を含む 21 assertions が成功（green3b.log）。MuZero の WARN は RED で 0 件のままを確認し、GREEN で 0→1→1 を確認（green4b.log）。
- TDD 5: GUI の起動テストは専用プロセスの hidden tag が必要であり、manual Runner の登録名は EvalPanel。準備を揃えた red5c-results.log で pane を開いた4経路の要求が欠けることを確認。実装後 green5-results.log は 1 case / 24 assertions 成功。Step、Q-Values 左クリック、実タイマーから得た同一 timer のイベント、AUI close、2枚のうち1枚を閉じる場合を通した。
- TDD 6: 起動を伴う3条件を同一プロセスの GENERATE にすると Env の重複登録になるため、ANET_PRD170_IMAGE_REGISTRATION=0/1/2 の別プロセスへ分けた。red6b では無登録・interval=0 が成功、interval=1 だけ失敗。登録一覧から options を固定して渡したあとは3条件とも各 7 assertions 成功（green6-*-results.log）。
- A12: core の変更後一式は baseline と同じ Catch seed=187172896 で 1028 cases（1027 pass / 1 skip）、196090 assertions 成功（core-after.log）。その後の定数への置換は同値の整理で、関連テストを再実行する。
- 実画面: computer-use の一覧取得は初回2回 timeout 後にカーネルを復旧して成功。既存のユーザー Run と別の検証用 ImageCls N128 Runner を起動したが、対象の画面取得が Computer Use app approval timed out になった。UI 操作は行わず、自分の一時停止中の検証プロセスだけを終了した。人が確かめる項は未実施として残す。
- GPU メモリ: WDDM の nvidia-smi のプロセス別 used_memory は N/A。UI 操作が許可待ちのため、補足の測定として、GUI を操作しない hidden test で出荷設定の RunManager / EvalRunner を構築し、100 step の要求なしと100 step の要求ありの CUDA allocator current / reserved / peak を記録する。実 pane を開く測定そのものとは分け、D18 の合否外の記録とする。毎 step の製品コードへ計測や同期は追加しない。

- 補足測定の ImageCls は200 stepと統計保存後、測定プロセスの wxEntryCleanup で異常終了した。ImageClsEnv_test.cpp の既存の注記も wx shutdown の不安定さを理由に初期化を process lifetime に寄せているため、測定専用 tag だけを同じ寿命に揃え、通常 GUI テストの後始末は保持する。再実行で終了値も確認する。
- throughput 初回 rr1 はメトリクスの step_axis を省略して train_step になり、exp_step の窓が空になったため採用しない。測定用設定へ $exp_step を明示し、新しい label で3巡を採り直す。

### 最終確認（2026-11-17）

- A1: 最後の定数整理後も trace-after-check/ の7ファイルが trace-before-seeded/ と byte 単位で完全一致（trace-check-comparison.json）。採取テストは seed170 / lex で 9 cases・120 assertions 成功。
- A2〜A5・A8・A10・A11 の関連テストは core-final-check.log で 8 cases・82 assertions 成功。attention・plasticity を含む通常 core 一式は前記の core-after.log で成功しており、その後の変更は同値の定数整理と測定用テストだけ。
- A6・A7: final-gui-results.log は 1 case・24 assertions、final-startup-{0,1,2}-results.log は各 1 case・7 assertions 成功。実イベントから要求と通知までを通している。通常 GUI 一式は gui-after-results.log でも変更前と同じ 39 cases 中 38 pass / 1 fail、894 assertions 中 893 pass で、新規失敗は無い。
- A9: core / apps / CONTEXT.md / docs/design に TraceCallback・MakeActionTraceCallback・AppendTraceAux は0件。A10: Body の Forward / ForwardUpTo と RunForward に capture 引数が無い。A13: 日英の140 §5.4に EvalPanel→Runner→Actor→NN→Notifier→Conv2d と要求なしの分岐を記載。
- A11: after_final の6 Runはすべて exit0。全 scalar の tag・step・値が before2 と完全一致した。比較結果は validation/after_final-compare-summary.json。性能・資源以外の除外、丸め、許容誤差は無し。

| 構成 | scalar 点数 | 変更前後で一致した SHA256 |
|---|---:|---|
| GridMaze OFF | 26946 | 2278b59c28b93e85d1adbfe203a1548a60589715fbdf576b029b1c9b3d800f98 |
| LunarLander OFF | 24399 | 941fa28b97b6a5833504c7a7b314bbc9966e77d6f66ff5edb8e4487e2a5a5a6c |
| Atari OFF | 23185 | c7653fe5692ce1aaea960f9c3b4ccfe3833793e698e0b7c2d1ab2011deb8bd65 |
| DropMerge OFF | 31152 | 6cd959f2df2406a3c32d8371eaed17fddef5e50bcd8e19dec976fa8f2da672c9 |
| GridMaze ON | 37977 | 30eac6139fc315b200f6c341c8afbddc04255aa274c7b5c3479080dc0a41bd36 |
| LunarLander ON | 34628 | 7749bcfc67280271a3fbd2871995921bcba9ef9857a08ab85d35828cc5a4f70f |

### GPU メモリの補足結果

測定専用 tag の起動を wxInitialize に揃え、ImageData が使う初期化カウンタへ接続した後は、memory4-img / memory4-atari がそれぞれ 6 assertions 成功、プロセスも exit0。初期の異常終了した測定は受入の成功に数えない。debugger での追跡は大量の既存 Debug 出力で進行が遅いため、自分の測定プロセスだけを識別して停止し、採用しなかった。通常 GUI テストの起動・終了処理は変えない。

出荷設定のコピー、train N128、manual EvalRunner N1、要求なし100 step→あり100 step。学習と描画は動かしていない。CUDA allocator の統計であり、実 pane を開く前後の測定は未実施のまま。

| 環境 | allocated peak の増分 | allocated current の増分 | reserved current の増分 |
|---|---:|---:|---:|
| ImageCls | 24320512 bytes（約23.2 MiB） | 0 | 6291456 bytes |
| Atari | 177664 bytes（約0.17 MiB） | 0 | 0 |

証拠: gpu-img-verified.json、gpu-atari-verified.json、memory4-*-results.log。閾値を置かず記録にとどめる。既存ユーザー Run は停止していない。

### 自己点検

docs/quality/impl.md の全項目を適用範囲ごとに走査。契約の移行、公開ヘッダ、同期借用、State の所有、既定値と通知経路、要求の完全一致と接頭辞、ログ、計測範囲、テストの公開経路、文書と翻訳元情報、検証資材の参照を照合した。新しい設定・GetScalar・ADR・skill は無いため、それぞれの専用項目は対象外。既存の namespace・設定・他作業の変更は今回の修正対象に広げていない。

selfcheck.log: 最初の39ファイルに、文書移行の差し戻しで100・110の日英4ファイルを加えた43ファイルについて BOM 保持・LF、ヘッダの using namespace、production dynamic_cast、追加 TODO、文字列リテラルの重複候補、7構成 trace、日英の翻訳元 blob を確認する。git diff --check も今回の対象で行う。新規ファイルは計画と検証資材4ファイルであり、既存の CONTEXT / ADR の変更は含めない。

差し戻し対応後のselfcheck-final.logで43ファイルの上記検査が成功。git diff --checkも成功。設計文書に残ったForwardのcallback引数は検索0件になった。日英4組の見出し・表・コードブロックと変更節の対応、追加した相互参照先を確認した。

### 学習 throughput

performance.py --label rr2 の6 Runがすべて exit0。両群とも Debug 実行体、Atari qbert BTR、2048 experiences、同じ入力。各 Run の1024〜2048 exp_step にある32標本の平均（exp/s）は次の通り。群の値は3 Runの中央値。既存ユーザー Run は動作したままなので、共有 GPU の状態の揺れを含む記録であり、速度向上の因果を断定しない。

| 巡 | 変更前 | 変更後 |
|---|---:|---:|
| 1 | 106.345 | 150.232 |
| 2 | 141.437 | 139.328 |
| 3 | 164.352 | 169.652 |
| 中央値 | 141.437 | 150.232 |

中央値の比は1.06219（約+6.2%）。合否の閾値は無し。起動引数、実行体SHA、全標本と値は performance/rr2/summary.json と各 metrics.json に保存した。

### 実行コマンドとログ

実行は repo の .venv を使用。Debug ビルドはすべて VsDevCmd.bat 経由。標準出力と標準エラーを .scratch/prd170/ へ保存して終了値・件数を確認した。6構成の起動引数・実行体 SHA・所要時間は validate.py の各 execution.json にある。再実行は testdata/prd170/README.md を参照。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner AnetRLRunner-gui-test > .scratch\prd170\build-final.log 2>&1'
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd170/validate.py compare --label after_final --reference before2
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe '[prd170],[nn][trace],[nn][branch_capture]' --reporter compact
```

### 工程末に起票した課題

- [issue-20261117-eedd0c](../management/issues/issue-20261117-eedd0c.md): 通常 GUI テストの「Trace keeps one selected row and fixed row heights」が、合成ドラッグ後の IsTraceMode で変更前後とも失敗する。事象が今回の変更前からあるため修正範囲外。開発者が新規起票を選択した。実画面とpane開閉前後のGPU測定の未実施は、検証欄の記録を保持する。

### レビューと差し戻し対応

- 1往復目: 計画・実装とも文書移行の不足で差し戻し（review-20261117-0619.md）。設計100・110の引数伝達、130のクラス図のcallback引数、140のImage/Graph Observer行を指摘された。PRD170 §6.4と照合し、8つの日英文書と計画をまとめて更新した。公開APIが直接関わるため、100のSerial/Pipeline図と110の行動選択図の呼び出しも同じoptionsへ揃えた。実装コードへの指摘と範囲外の新規検出は無し。
- 最終Debug再ビルドは3 targetともexit0（build-final.log）。最後の同値の名前空間修飾整理後のcore関連は8 cases・82 assertions成功（core-rebuild-check.log）。GUI通常一式は、測定専用tagの起動分岐を直した後も同じseedで39 cases中38 pass / 1 fail、894 assertions中893 pass（gui-final-results.log）。既存1件の失敗が変わらない。
- 100・110の追加前にスナップショットの追加コピーが自動承認レビューで拒否された。存在を読み取りで確かめると4ファイルとも未保存だったため、既存ファイルを上書きしないCreateNewで保存してから修正した。最初の比較証拠を変更していない。
- 2往復目: 実装計画・実装ともOK（review-20261117-0629.md）。前回の観点12・18・50の指摘はすべて解消。新しい不足点・範囲外の検出・開発者への確認事項は無し。実画面と実paneのGPU測定は未実施という制限を含むOKであり、画面確認済みとは扱わない。
- issue-20261103-6a32b4 のAI判定を残件無し、判定日を2026-11-17へ更新した。実装後も残る既知の事象は無いという判定であり、人の確認を実施したという意味ではない。statusは対応中を保持し、解決へ変更していない。
- wiki_support.py mapsはexit0、unresolved_prdsは空（maps-final.log）。派生するdocs/management/issues/INDEX.mdと、wikiのカテゴリ312_util・315_nn・318_rl_core・321_agent_coreの課題区画の5ファイルを再生成した。これらは直接変更の43ファイルと分けて扱う。

### 工程末の報告

実施内容: captureへの統合と、要求したstepだけnn_traceを生成する公開経路の実装・検証・自己点検・再レビューを済ませた。変更はNN/Actor/Runner/GUI、関連テスト、検証資材4ファイル、設計100・110・130・140の日英、CHANGELOG、対応課題と本記録の43ファイルに、工程末の新規課題1ファイルを追加した。課題一覧とwikiの派生区画も規定の道具で再生成済み。起票後のmapsはexit0でINDEX.mdとwiki/apps/110_runner_app.mdを更新し、unresolved_prdsは空（maps-eedd0c.log）。新規課題のLF・空の対応内容と差分の検査も成功した。

レビュー結果: 2往復でOK。根拠が無く見送った指摘は無し。再確認ポイントは実画面とpane開閉GPU測定の未実施、通常GUIの既存1件の失敗。コードを変えた通常core一式の結果、trace・scalar完全一致、最終再ビルドと関連テストを保存した。

決まらなかったこと: GUIテスト失敗の原因と対処をissue-20261117-eedd0cへ記録した。

判断してほしいこと: なし。実画面とpane開閉GPU測定は、未実施という制限を保持する。

次にできること: PRD170の工程を完了して停止する。stage・commit・pushは行っていない。
