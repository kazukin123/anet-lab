# PRD097 遷移の結果をStepの結果で運ぶ 再実装メモ

## 概要

2026-10-20の全面改定PRD097とADR0060に従う再実装。旧20implは履歴として保持し、本書を今回の正本とする。基準HEADはef7862d4f62b1b01d0b17b9e3589f68a3ff340f2。開始時のdirty一覧・対象ファイルのコピーとSHA256は`.scratch/prd097/reimplementation-20261020/initial*`へ保存した。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidenceで解決済み: PRD§4.3/ADR0060がResetとStep、自動リセット、結果の問い合わせ、通知の所有、集約・NaN・設定移行を決定。現行M2との差は§4.3.9に対応する。
- 2ximplへ固定: DropMerge/GridMazeは現在の統計計算式と演算順序を結果へ移し、終端以外も定義可能な累積統計を速報値として返す。終端理由・候補判定など未成立の既知鍵はNaN。auxの既存値は維持する。Step結果は所有する値/参照から読み、scalarごとのaux複製はしない。結果のunknown/non-scalarはnullopt、メトリクスenv_stepで最初の読み出し時にfail-fast。
- PRD範囲外: PRD094/081本体、開始通知、QValuePanel、W/S1採用、英訳・過去Runの書換、コミット・push。

## 主な変更

1. rl/env/trainer/ImageClsのM2段と順序検査を撤去し、continue_stateとStep内の自動リセットを戻す。結果は毎回新規に生成し、過去結果の値を守る。ThreadPoolはStepとResetを同じworker内で行う。
2. SingleEnvResult/BatchStepResultにGetScalarを追加。wrapperはlaneのsingle_resultを所有して委譲する。Single既定はauxのスカラー、SHAREDは-1。reward/done/truncatedを結果から読む。終端maskは1結果につきCPUへ1回だけ移す。
3. EpisodeEndEventへstep_resultを追加し、逐次・pipeline・評価・RunSessionで正しい結果を渡す。評価セッションの採用値を結果から蓄積し、次セッションはcontinue_stateを再利用する。
4. EventField/ObserverFactoryにenv_stepを追加。trainは全lane、episode_endは終端lane昇順を既存ScalarSampleAccumulatorで集約し、同じStepのscalar読出し/EMA更新を1回にする。traceは通知laneだけ。learn/session_endの宣言は構築時に停止する。
5. Atari結果はNOOP前の速報値・RGB参照と、ゲーム終端の確定値を所有する。残機喪失のみのゲーム鍵はNaN。派生鍵・RAM・HNSは結果が答え、envはlives/RGBだけ残す。既存のD6処理を維持し、e2の実ALE経路で再検証する。
6. DropMerge/GridMazeの統計を結果へ移し、envの該当GetScalar鍵を削除する。View用last_*auxは維持する。ImageCls結果には採点windowのaccuracy/epoch_countを保持する。
7. AtariViewのRGB、Atari/DropMerge/GridMazeの設定、inspect_run.pyとViewerの定義target検証を移行する。tag・EMA・既存step軸を保つ。default_leavesと全キー値差分で意図した宣言変更だけを確認する。
8. 日本語設計010/100/120/140/150/220と学習シーケンス図を同期する。改訂PRDとCONTEXT/ADRのユーザー編集を保持する。

## テスト

Public surface: Reset/Step/GetScalar/GetAuxDataList、Runner通知、ObserverFactory、metrics/trace出力、設定読込、各env結果、Viewer/解析のtarget読込。

TDDは1 behaviorずつRED→GREEN→必要時だけ整理。最初は実BatchEnv→SerialRunner→終端通知→次行動の縦スライス。以降はlane混在/全終端/なし・truncated・SHARED・過去結果不変、pipeline/eval/session通知、env_step集約/同step EMA1回/trace/不正event、評価集約、ImageCls window、Atari派生鍵・NaN表・NOOPゲーム終端(e2)、DropMerge/GridMaze、env未知鍵の順。

Atari e2は既存のAsterix実ALE試験を再利用する。seed1/frameskip1/sticky0/noop_max0/fire_reset=false/episodic_life=true、32bit LCGで300frame進め、テストfriendでRAM0x53=1,0x47=3を設定。公開Step結果のNOOP前livesとNOOP後確定値を検証する。再現不能は合格にしない。

## 検証

資材は`core/anet-core/testdata/prd097`、今回のログ・実行体・設定・manifestは`.scratch/prd097/reimplementation-20261020`。旧before実行体とsource-config・2回完全一致済み7構成はハッシュと実効設定を再監査して再利用し、不足は再実行する。afterは新ディレクトリへ保存し、DSLを今回の契約へ移す。

- 共通: seed=1、512000経験、deterministic backend、deterministic_warn_only=false、評価同期。Atari1はBreakout/run.@v5_iqn_impala_x2/@classic>E1、Atari2は@100k>E1。評価3タグ各2lane/2episode/interval1000。残りは現行設定pipeline、Lunar CPU、DropMerge serial。
- 等価性: scalar全点・trace全行を比較。除外は90_perf/91_resource/92_resource、trace timestamp。lives7タグは前後で点数/step、before同士は値も一致。移したepisode_end scalarのみpipeline最初/最後stepを両側から除く。Noisy追加除外なし。
- 性能: Atari1/GridMaze/Lunar CPU/DropMerge pipeline/serialをbefore/after交互各3回。後半exp/s点の中央値をRun値とし中央値低下2%以内。before(max/min-1)>2%なら5組、それでも不安定なら判定不能として調査記録・再測定候補を提示する。以前の世代を新測定と混ぜない。
- Debug全体ビルドとCTest既定スイート（隠し以外全件）。Releaseで受入測定。C++は必ずVsDevCmd経由、Pythonは.venv。
- EvalPanelのAtari最終残機喪失画面を実際に確認する。実行不能は理由を記録し必須受入の完了扱いにしない。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```
```powershell
ctest --test-dir out/build/x64-Debug --output-on-failure -j 1
```
```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd097/compare_metrics_test.py
```

## 前提と完了条件

- ユーザーの再実装依頼によりコード・設定変更と検証を実施する。開始時点でAnetRLRunner/anet-core-test稼働なし。
- 無関係なdirtyは保持。対象ファイルでもM2と無関係なhunkは保持する。PRDのクラス一覧は既存機能グループ内で実装し、必要な内部結果型はそのファイルに置く。
- 自己点検はdocs/quality/impl.md全項目。独立an-reviewへ計画・今回差分・検証記録を渡し、修正・再レビューする。原文を保存する。
- 必須受入・全テスト・自己点検・レビューを終え、未決/未実施と扱いを記録して工程を終える。コミットしない。

## 経過

- 2026-10-20: 新PRDの監査を通過し本書をコード変更前に保存。旧before世代と全既存dirtyを保持する。

- tracer RED: `tracer-red-build.log`、GetScalar未実装によるC2039を確認。M2段専用テストは2操作契約の既存版へ戻し、新契約のbehaviorを順に積み直す。worker例外検証は開始時コピーから後続で回復する。

- tracer GREEN: Debug1テスト9assertions成功（tracer-green-test.log）。env本体0、Step結果/次行動の観測は1/0。次のenv_step集約テストはRED（3出力欠落、metrics-red-test.log）。

- env_step集約/EMAはGREEN（14assertions/2tests、metrics-green-test.log）。既存EMAはbias補正あり、2値6/10の期待値は26/3。traceは未対応target拒否でRED後、laneごとの結果読出しを実装しGREEN（21assertions/3tests、trace-green-test.log）。learn/session_endは構築時に拒否しないREDを確認（invalid-red-test.log）。

- 評価セッション: env本体-1000、Step結果10/20のfixtureでPER_LANE/SHAREDともRED。CaptureScalarsの元を結果へ変更してGREEN（115assertions/14tests、session-green-test.log）。
- 結果のaux所有をSingleEnvResultのコンストラクタ（AuxData、既定空）へ集約する。GetAuxDataの値返しは維持するが、既定GetScalarは所有メンバを直接読み、scalarごとのaux複製を避ける。既存の非空auxを返す派生結果は同じ所有メンバへ移す。これはPRD§4.3.1の所有・性能契約を満たすための補助constructorで、旧APIの互換層ではない。LunarLanderのauxも現在のenvを後から読まない結果にし、不変性を守る。

- Single結果の所有auxはGREEN（6tests/39assertions、single-green-test.log）。Atari必須e2はC++ ALEでREDを確認後、301frameの速報aux/RGBと302frameの確定game値を結果へ分けてGREEN（16assertions、atari-noop-green-test.log）。envのgame鍵は未知へ変更。

- Atari派生鍵はhns57欠落でREDを確認（atari-derived-red-test.log）。ゲーム名・既知RAM番号だけの不変なAtariScalarContextをEnv/結果で共有し、設定全体や番号表をStepごとに複製しない。内部ヘッダの前方宣言とshared_ptrを用い、値のスナップショットは結果自身が所有する。

- Atari派生鍵とe2はGREEN35assertions。旧テストのenv統計読出しを各Step結果へ移行後、Atari全39tests/1026assertions成功（atari-regression-test.log）。

- GridMazeは結果キー未実装のRED後、reward_sum/wall_collisionsを累積速報値、is_success/is_holeを終端以外NaNとして結果へ移行しGREEN（done/hole/truncatedの42assertions、grid-green-test.log）。DropMergeは結果キー欠落のREDを確認し、統計の整数/enum/boolを結果へ保持して既存float式をそのまま移す。

- DropMerge結果/Reset/View履歴はGREEN102assertions（drop-green-test.log）。既存テストの統計読出しを、その値を生んだ明示的なStep結果へ移行して全体回帰を行う。

- DropMerge全28tests/378assertionsが成功（drop-regression-test.log）。ImageClsはaccuracyの結果キー欠落でRED（image-red-test.log）。内部ImageClsStepResultへ採点snapshot2値と既存aux/continue_stateを所有させ、後のwindowから独立させる。

- ImageClsは全15tests/150assertions成功（image-green-test.log）。window間のaccuracy変化後も古い結果が当時の値を保持する。LunarLanderも結果auxがenvを後から読んでいるため、その後のStep/Resetで画面用データが変わらないことを次の1テストで検証する。

- LunarLanderは後のStep/Resetでauxが変わるRED後、結果の所有へ移し全7tests/94assertions成功（lunar-green-test.log）。M2から除くのは段の処理であり、終了時に最後の学習futureの例外を再送出する処理は維持すべきため、その公開Shutdown経路を次の1テストで確認する。

- Shutdownは最後の学習例外が返らないRED後、future.getを維持してGREEN（2assertions、shutdown-green-test.log）。段だけを除去し、元M2の例外再送出の修正は保持した。Runner通知・評価fixtureの旧env読出しを結果へ移行し、現在envの値0と遷移値1を別々に確認する。

- Core関連回帰72tests/765assertionsが成功。追加SHARED組込み値はreward欠落のRED後に実装し、ImageCls全16tests/163assertions成功（shared-{red,green}-test.log）。worker例外の旧aux生成経路は現契約ではworker内で呼ばないため、Reset/Step/自動Reset失敗をserial/threadの6組で回帰する。公開結果不変性・不正キー・先読み試験も追加した。
- AtariViewは既存の1回のGetAuxDataListからrgb_frameを取り、現在envの画面を読まない。設定3ファイルをenv_stepへ移行し、tag/EMA/step軸を保持した（*-declaration-migration.json）。代入規約は3457件監査・エラー0。
- inspect_runの公開CLIは新targetのconfig導出でRED後、全73tests成功。現行Java Viewer（apps/metrics-viewer）は非scalar定義JSONをtargetに制限を置かずjson_linesへ保存しているため、PRDが想定したtarget validatorの修正箇所は存在しない。不要なvalidatorは追加しない。
- 比較器は限定した移行tagのpipeline端点除外と最終報告座標基準の性能中央値をRED→GREEN（3tests、comparator-{red,green}.log）。before設定は保持し、afterの3宣言ファイルだけ現行へ差し替える。5構成の交互3組/不安定時5組の測定ハーネスを更新した。
- 日本語設計6文書と学習シーケンス図を更新。Archifyは9artifact checks、composition errors/warningsとも0でdeliver成功。4viewportのoverflow/capture検査成功、1440 dark/2048 lightを目視確認。Viewer固定UIは英語。保存内容はdocs-readback.jsonへhash/mtimeを記録した。古い20impl・過去記録は保持する。

- Debug全体ビルド成功、CTest全6実行体成功（295.02秒、debug-full-build.log/debug-ctest.log）。自己点検: 品質観点1〜55を適用範囲に従って走査し、公開API・設定・所有・段階コメント・計測・エラー言語・NaN/nullopt・LF/差分checkを確認。Agent state変更/skill変更は対象外。必須の系列/性能/手動確認は実行前のため未達であり、完了としない。独立レビューで追加の不足を確認する。

- before再監査:14 Run/241ファイル、元audit-finite-loss.jsonと全hash一致、exe SHA b275bcedb9329eea74a45e0c68280e4647e4ddc8156b1d7d3842228d13cee819を再確認（before-reuse-audit.log）。
- 手動受入の準備でComputer Use skillを読み正規node_repl/@oai/skyを初期化したが、`failed to write kernel assets: 指定されたパスが見つかりません。(os error 3)`で起動できない。kernel reset後の再試行も同じ。現時点で実EvalPanelの目視は未実施。結果RGBの公開APIテストを手動受入の代わりにはしない。

- 独立レビュー指摘: D7端点は0起点。before Atari1のtrace末尾511872、Train scalar末尾511744とコード/テストで確定。比較器の誤った[128,512000]を[0,511872]へ修正し、内側座標を除外しないRED→GREENを追加する。Run自体の条件は変わらないため保存成果物で再比較する。

- レビュー修正:120のImageCls段・220のHNS/RAM/View各節の旧説明を撤去。Atari設定の今回変更したprofile行8件をコロン境界へ変更（値は同じ）。旧M2専用check_shutdown_profile.pyは履歴用コピーをscratchへ保存し、現用testdataから除去。古い結果記録・20implは書き換えない。

- 初回レビュー原文: .scratch/prd097/review-20261020-1634-d7-impl.md（plan OK / impl差し戻し5件）。5件を修正した。HTMLは元skillを改変せず、scratchの生成器コピーのテキスト改行だけをLFへ統一し、同じArchify validate/deliver/visual-checkを実行。公開JSON/HTMLともw/lf、9checks成功、4viewport検査成功。範囲外の新規検出は既存observers.cpp:917のdynamic_cast1件（今回hunk外）で、工程末の記録候補に積む。
- Debug最終件数: Core711pass/1skip（CUDAあり環境でCPU-only分岐テストがskip）147750assertions、Lunar94、Drop378、Grid42、ImageCls176/17tests、Atari1026/39tests。必須Asterixはskipしていない。

- 再レビューはplan/implともOK（原文 .scratch/prd097/review-20261020-1638-d7-impl.md）。コード不具合の新規指摘なし。受入の未実施は引き続き未完了として最終結果を再提出する。
- 受入途中: Atari1 93027点、Atari2 90028点、DropMerge 161610点が一致。DropMergeの起動時にimport済みだった旧比較器は末尾点を残して不一致を示したが、修正済み比較器で保存Runを再比較し完全一致。全7構成終了後に現比較器でREUSE再比較する。

- 追加監査でDropMerge serialの旧before1/2は設定ファイル指定だけで、改訂PRD§7のCLI指定条件が実行引数に無いことを検出。値自体はserialであるが、正式受入にはCLIを追加してbefore3/4・after3を実行し、専用dropmerge_serial_cli.comparison.jsonへ保存する。旧1/2と進行中after1は診断記録として保持する。harness/auditでもCLI引数を必須とし、番号を分離した。その他6構成のbefore再利用監査は引き続き有効。

- GridMaze 90851点、ImageCls 67160点も一致。実行中import済みの旧比較器によるGrid末尾の不一致は、修正済み比較器で再比較して解消。C++差分は再レビュー後から不変。

- 性能検証は短いGridMazeから実行する。各構成のbefore/after交互順と母集団は不変。不安定なら5組まで測る。安定した不合格を検出した世代はそこで止め、原因是正後に新世代で全5構成を再検証する（不合格世代で残りの長いRunを消費しない）。

- 正式D7系列受入は全7構成合格、合計728534点（equivalence-final.log、after-d7-v1/summary-d7.json）。CLI serial before3/4完全一致、after3も162210点一致。Atari1 93027、Atari2 90028、Drop161610、Lunar63648、Grid90851、ImageCls67160。
- before正式再監査14Run/241ファイル、after7Run/158ファイルが合格（audit-{before,after}-final.log）。全実効キー・文字列値の比較も合格（resolved-audit.log/resolved-config-audit.json）。差は意図したメトリクス宣言とapp.runs_dirのみ。実行体世代にはsource-code-manifest.jsonも追加した。
- 交互性能測定を開始。測定中はビルド・テスト・他Runを並行させない。

- GridMaze性能5組:before中央値17312.45703125、after17251.0439453125 exp/s、比0.9964526649321558（0.3547%低下）。before spread=8.2280%>2%のため判定不能。比だけで合格にしない。全反復終了後、ばらつきの原因調査を行い記録する。case別結果はafter-d7-v1/gridmaze.performance.json。

- Lunar性能5組:before4298.774658203125/after4058.33203125 exp/s、比0.9440671712125451（5.5933%低下）、before spread3.8069%で判定不能。中央値差が大きいため残りのv1性能を中断し、描画aux生成の追加コストを診断する。開始済みbefore Atari1_201は中断記録を保持し、受入証跡には使用しない。owned performance controllerとその子Runnerだけを停止した。診断と是正後に必要な全受入を別世代でやり直す。

- NVTX/CUDA診断4本は正常終了（diagnosis-20261020-173310）。正式性能とは別保存。最後500学習StepのLunar CreateAuxDataはbefore500回/41.30ms合計、after142130回/13650.76ms合計（並列・入れ子scopeの合計なのでwallに加算しない）。train ThreadPool Step平均1.189→4.012ms、Root中央値53.971→56.412ms。結果生成で描画Tensor化を全Single Step/Resetへ移した追加コストを確認した。Root内に評価もあり、scope平均だけを正式性能値とはしない。
- 是正する実装前提: Lunar結果は描画のfloat/bool配列・terrain座標とTensorOptionsを値所有し、GetAuxDataで要求時だけTensor化する。envを保持せず、結果不変性を守る。現auxは全キーがvector/matrixなのでGetScalarのスカラー対象は増えない。軽いnative captureは既存Env Step計測内、移した重いTensor化はGetAuxDataで計測する。private CreateAuxDataは撤去する。統計式や学習観測は変更しない。

- Lunarはnative snapshotへ変更後、既存7tests/94assertionsが成功（lunar-lazy-test.log）。全描画キー・dtype・非スカラー扱い・後続Step/Reset後の不変性へ同じ公開テストを補強し、全体Debug/Release検証へ進む。

- Lunar是正後のDebug全体ビルドとCTest6/6が成功（319.15秒、v2-debug-build.log/v2-debug-ctest.log）。Release全体ビルドも成功（v2-release-build.log）。独立再レビューはplan/implともOK（原文 .scratch/prd097/review-20261020-1758-d7-impl.md）。after-d7-v2へ新実行体を固定し、全7構成の再受入を開始した。v1の性能記録と混ぜない。

- 是正後の全件内訳: Core711pass/1skip・147750assertions、Lunar7tests/136assertions、Drop28/378、Grid1/42、ImageCls17/176、Atari39/1026。CUDA条件skip以外の隠しでない全試験を実行した。Lunar全9キーの型と結果不変性も含む。
- v1 Grid診断:最後500stepのRoot中央値15.281→15.671ms、EnvStep平均2.742→2.859ms、Learner12.987→13.278ms。Env/learner重なり比.9589→.9608で並行性の喪失はなく、廃止した段のscopeも0。ログ量244165→244310bytesでWARN急増もない。学習ログの時間窓に絞ったGPU sampleはbefore43/after42、SM中央値2730/2707MHz、温度最大49/48度。clock/負荷の変動はあるが、正式beforeの反復間ばらつきの原因をこの別Runから確定できない。正式v2で再測定し、なお不安定なら調査結果付きの再測定候補へ残す。

- 手動確認用Release Runner（PID53740、run_20261020-180535_prd097_manual）が正常起動。実効設定でCPU、train1lane、自動学習/評価停止、Asterix、episodic_life=false、frame_skip10000を確認した。Step操作で最終画面の目視と終了をユーザーへ依頼中。回答前に合格としない。

- v1 Gridのbefore性能5反復201〜205を、時間・資源以外の全scalar/traceで再比較し、全4比較90851点が完全一致した（grid-v1-before-performance-series.json）。反復間で学習・評価の数値系列が変わったための速度差ではない。ばらつきの原因を確定する材料には足りないため、合格への読み替えは行わない。

- Release実行体でもAtari必須Asterix NOOP終了と派生scalarの2tests/35assertionsが成功（v2-release-atari-acceptance.log）。game_frames302を含む完了値と、301frameの遷移aux/RGBの分離を公開結果APIで検証した。

- after-d7-v2は全7構成728534点の比較が完全一致（v2-equivalence.log/summary-d7.json）。実効設定7構成とafter7Run/158ファイルの監査も合格（v2-resolved-audit.log/v2-audit-after.log）。新exe SHA b994bad918277edf12742b11452438336aafda004cc269535521e127f500691c。
- 性能開始前、専用手動RunnerのStop-Process後の終了待ちが足りず、他Runner検査で開始を停止した（formal測定0本）。Runnerがすべて終了したことを確認し、専用Process objectのWaitForExitを補助スクリプトへ追加して再開する。手動目視の回答は未受領で、合格扱いにしない。

- v2 Grid性能5組206〜210:before中央値17201.1787109375/after17436.8447265625 exp/s、比1.0137005736400582（1.3701%向上）。before spread5.1521%のため判定不能（gridmaze.performance.json）。v1と同じコード・設定で、並行性/数値系列/WARN量/GPU状態の診断では反復のばらつきの原因を確定できていない。正式5構成の測定終了後にv2反復の系列も照合し、原因未確定なら再測定を起票候補へ示す。

- v2 Lunar性能3組206〜208:before4251.376953125/after4304.365478515625 exp/s、比1.0124638501772174（1.2464%向上）、before spread1.6692%で合格。v1で見つけたTensor生成増加をnative snapshotで是正し、基準2%以内を満たした（lunarlander.performance.json）。
- DropMergeにも同じeager Tensor生成が残るため、測定順を短い順Grid→Lunar→Drop pipeline→Drop serial→Atariにする。各構成内の交互順・回数・条件は不変。完了済みv2 Grid/Lunarは保持し、開始直後のbefore Atari202だけ中断・不採用記録として保存、予約planも別名で保持する。次回Atariは未使用番号から予約する。今の段階ではDropコードを変更せず、性能測定で追加コストを確かめる。

- 測定順変更用の補助PythonでUTF-8指定漏れによるCP932 decode errorがあり、更新が未反映のまま起動したbefore Atari203も中断・不採用とした。202/203の記録と予約planは別名で保持。UTF-8を明示して順を修正し、Dropから再開する。受入値の除外を広げたり、完了済みGrid/Lunarの測定を入れ替えたりしない。

- Drop before204の総実行284.3秒は、学習終了19:12:46.685後のAgent保存（852112095bytes）が19:14:36.408まで約110秒かかったことを保存ログで確認した。総実行時間は速度判定に使用しない。保存は学習区間の後であり、この事実だけでexp/sのばらつきの原因と断定しない。

- v2 Drop pipeline性能5組201〜205:before3351.9364013671875/after3314.310546875 exp/s、比0.9887748900973059（1.1225%低下、比は2%以内）、before spread2.6296%で判定不能（dropmerge.performance.json）。追加のnative/lazy refactorは適用しない。ばらつきの調査と再測定候補はGridと同様に残し、残るserial/Atariの正式測定を続行する。

- 判定不能の調査方針:全5構成の正式性能を終えてから、Grid/Dropのbefore5本の全scalar/traceを照合する。Gridの既存NVTX診断に加え、Dropのbefore/v2を別保存のNVTX/CUDAで採取し、描画Tensor生成・Root/Env/learnerの重なりを比較する。大量traceによる容量増加を避け、導入済みnsysのhelpでdelay/duration/kill=false/wait=allを確認し、512000経験の途中30秒程度に採取を限定する。診断だけは両側共通で終了時Agent保存を無効にし、正式性能や系列受入へ混ぜない。原因未確定なら、その限界と再測定候補を明示する。

- 旧手動確認依頼が残った状態で、ユーザーが画面のStepを1回押し、DropMergeであると申告した。現在の唯一のRunnerはafter serial203（PID24484）と確認。手動対象Atariではなく性能用画面を操作させてしまったため、この前後203組を不採用にする。owned controllerと子Runnerを停止し、201/202の完了組は保持、正式planを201/202/204/205/206へ変更して203組を取り直す。旧予約planと途中記録は保持する。正しいAsterix手動画面を改めて開き、回答まで閉じず、性能も再開しない。

- 測定停止中にv2判定不能構成のbefore全5本を再照合。Grid206〜210は全4比較90851点、Drop201〜205は全4比較161610点が完全一致（v2-before-performance-series.json）。Noisy等の追加除外なしで、学習・評価の数値系列差によるばらつきではないと切り分けた。凍結v2に対応するC++/CMake30ファイルの現行hashも30/30一致（v2-source-recheck.json）。導入済みWindows nsys helpではkillの値はtrue/falseなので、時間限定診断は--kill=falseを使う（nsys-profile-help-v2.txt）。

- 手動受入はユーザー提供のAsterixスクリーンショットで確認済み。Lives=0、Score=650、Game step=1、Frame=5805が20:04:44.285のGame overログと一致し、終端時のRGB画面が保持されている。証拠をmanual-evalpanel-terminal.pngとmanual-evalpanel-acceptance.jsonへ保存。確認用Runnerが閉じていることをプロセス照会で確認し、残るserial/Atariの性能測定を再開した。

- v2 Drop serial性能5組201/202/204/205/206が正常終了。before2889.55419921875/after2689.72705078125 exp/s、比0.930844990382417（6.9155%低下）、before spread19.2386%で判定不能。中央値差も大きいため、正式測定終了後の時間限定NVTX/CUDA診断へserialのbefore/v2も追加する。pipelineと同じ条件・CLI serial指定、診断だけAgent保存なし。速度低下を判定不能の記載だけで省略しない。原因が確定しない場合は限界と再測定候補を残す。Atariの正式交互測定へ移行した。

## 最終受入集計（2026-10-20）

- 現行世代はafter-d7-v2。Debug全体ビルド、CTest全6実行体（319.15秒）、Release全体ビルドとRelease必須Asterix公開API35assertionsが成功。C++/CMake30ファイルは凍結実行体のsource-code-manifestと再照合する。コードは1758レビュー以降変更していない。
- 全7構成・728534点のscalar/traceが正式D7条件で一致。before14Run/241ファイル、after7Run/158ファイル、全実効設定比較も合格。serialはCLI指定のbefore3/4・after3。Noisy指標等の追加除外なし。
- 手動EvalPanel受入はAsterixのユーザー提供画像で確認済み。Lives0/Score650/Frame5805が終端ログと一致し、終端RGBが保持される。manual-evalpanel-terminal.pngとmanual-evalpanel-acceptance.jsonを保存。
- 正式性能46Runはすべて正常終了、同じ凍結実行体・seed1・512000経験、serial CLI指定を確認した（v2-performance-execution-audit.json）。手動操作の入ったserial203組と、中断したAtariの記録は含めない。全5構成のperformance.jsonをREUSEのみで集約した。

| 構成 | 組数 | before exp/s | after exp/s | after変化 | before spread | 判定 |
|---|---:|---:|---:|---:|---:|---|
| atari1 | 5 | 4978.385 | 4973.180 | -0.1045% | 12.0979% | inconclusive |
| gridmaze | 5 | 17201.179 | 17436.845 | +1.3701% | 5.1521% | inconclusive |
| lunarlander | 3 | 4251.377 | 4304.365 | +1.2464% | 1.6692% | pass |
| dropmerge | 5 | 3351.936 | 3314.311 | -1.1225% | 2.6296% | inconclusive |
| dropmerge_serial | 5 | 2889.554 | 2689.727 | -6.9155% | 19.2386% | inconclusive |

判定不能4構成は合格へ読み替えない。改訂PRD§7受入4の規定に従って性能合否の母集団から外し、調査と再測定候補を残す。判定可能なLunarは基準を満たす。安定した不合格構成はない。

### ばらつき・追加費用の調査

- 判定不能のbefore全5反復を、時間・資源以外のscalar/trace全点で再比較。各4比較がGrid90851点、Drop161610点、serial162210点、Atari93027点ですべて完全一致（v2-before-performance-series-final.json）。数値系列の違いによる速度差ではない。
- Gridは既存before/v1のNVTX/CUDA診断でEnv/学習の重なりを確認し、並行性喪失やWARN spamを認めない。v2でも同じコードでbeforeのばらつきが残る。正式反復時のclock・CPU等を追跡していないため原因は確定できない。
- Drop pipeline/serialはdiagnosis-drop-20261020-213445へ別保存。before/v2各1本、90秒待ち・30秒採取、CUDA/NVTX。正式条件の512000経験まで完走、診断だけAgent終了保存なし。最初のbeforeはNsightが先に戻り完走判定が早すぎたため一度停止。後の通常closeログと残存Runnerなしで完走を確認し、最初のrecordを保存して更新した。この1本のアプリ終了コードは未捕捉で、Nsight終了コード0・完走ログのみ。後続3本は対象Runner handleで待ち、アプリ/Nsightとも終了コード0。正式性能の証拠には使用しない。
- pipeline診断: Root中央値80.408→80.195ms、Env batch平均20.978→21.951ms。aux生成は採取内298→77070回、inclusive合計15.72→4213.84ms。結果が各laneの描画値を所有するためTensor生成が増えた。scope合計は入れ子・並列を含み、wallへ加算できない。Envと学習の重なりは99.766→99.788%で維持。
- serial診断: Root中央値83.167→85.176ms（約2.4%増）、平均93.466→95.827ms。Env batch平均20.806→21.964ms、aux生成319→79758回、inclusive合計17.31→4045.32ms。Env/学習の重なりは前後とも0。learn_event等にも区間差があり、追加Tensor費だけで正式中央値6.9%低下の全てを説明できない。
- 採取窓は同じ経過時間であり、前後で同一step範囲ではない。同期評価も含む。GPU SM中央値はpipeline2872/2872MHz、serial2872/2880MHz、memory13801MHz、最大温度pipeline70/71℃・serial71/70℃。別Runの30秒観測だけで正式反復のばらつきをGPU温度・clock等へ帰属できない。
- Atariは同一before実行体・設定・全数値系列でも5反復spread12.10%。正式Runの資源時系列を採っておらず、原因未確定。推測でcold cache等へ断定しない。
- 新しいnative/lazy Drop最適化は適用していない。安定した条件の再測定と、serialの追加費用削減を今後の候補とする。全GPU検証とRunnerは終了している。

### 自己点検と残る項目

- 公開API・結果の値所有・NaN/nullopt・通知とlane対応・日本語段階コメント・ProfileRange・設定移行・LF・既存dirty保持を再点検。ソース変更がないため、成功済みの全体ビルド/テストを繰り返さない。
- 必須受入の未実施はない。判定不能4構成の性能は原因調査結果付きで持ち越す。最終独立レビューはOK。プロセス契約§7の候補①②はユーザーが採用し、下記の課題へ記録済み。
- 持ち越し課題: [999_prd097_performance_remeasurement_10prd.md](999_prd097_performance_remeasurement_10prd.md)（安定条件での性能再測定・serial追加費用）、[999_metrics_action_info_dynamic_cast_10prd.md](999_metrics_action_info_dynamic_cast_10prd.md)（既存Observerの型判別整理）。PRD094・開始通知・QValuePanel・英訳・過去記録・commit/pushは対象外。

- 最終独立レビューはplan/implともOK、修正要求0件（原文 .scratch/prd097/review-20261020-2158-final-impl.md）。最終51ファイルのhash/LF、全受入記録と判定不能4構成の持ち越し条件も確認された。コード30/30再照合済み。残る工程末の選択は、性能再測定候補と既存dynamic_cast整理候補の起票だけであり、選択前には課題記録を作らない。commit/pushなし。

## 完了報告

- ユーザーは工程末の候補①②をともに採用した。レジストリのissue-record形式で、[999_prd097_performance_remeasurement_10prd.md](999_prd097_performance_remeasurement_10prd.md)と[999_metrics_action_info_dynamic_cast_10prd.md](999_metrics_action_info_dynamic_cast_10prd.md)を新規作成した。重複する既存課題は見つからなかった。
- 改訂PRD097に沿う再実装、全体ビルド/テスト、全7構成の系列一致、手動画面確認、性能測定・判定不能の調査、自己点検、独立最終レビューを完了した。性能の未確定4構成は上記課題へ持ち越し、合格とは表記しない。
- 必須検証の未実施と追加の判断待ちはない。GPUを使う検証と全Runnerは終了。今回の工程を完了して停止する。commit/pushなし。
