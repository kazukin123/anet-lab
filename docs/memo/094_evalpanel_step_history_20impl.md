# PRD094 Evaluation Trace 実装メモ

## 概要

PRD094の段階1（値の読み口・配線の整理）と段階2（step trace・Evaluation Trace・追従pane・再生）を実装する。正本は同階層の094_evalpanel_step_history_10prd.md。基準コミットは4d191af32b600442e933cb0b7f8e53f0858d7f04。

## 主な変更

- MetricValueReaderでscalar/trace/step traceのtarget解決を共有する。MetricsWiringをRunManager所有として配線・WARN・定義・購読ヒントを移す。段階1は既存の値・探索順・集約・EMA・出力・診断文言と時機を維持する。BatchActionInfo::GetScalarSource()で明示的にscalar提供元を取得し、既存production dynamic_castを撤去する。
- coreの独立した機能グループstep_trace.hpp/cppにStepTraceColumnSpec、StepTrace、StepTraceBuilder、TracePlayerを置く。GUI非依存で列・時点・遅延確定・再生制御を共有し、PRD081にも利用可能にする。通知1回のlane別行を安定IDで識別し、追加/確定を別通知にする。報酬・done・truncatedは常に保持する。
- Agent::CreateStepDiagnostics()（既定なし）とAgentStepDiagnosticsを追加する。DefaultDQN/RainbowはDqnAgentStepDiagnosticsを生成する。ヒント生成・n-step組立て・初期優先度式は既存実装と共有し、PER設定にかかわらず計算する。診断窓は診断部品所有、AgentのResourceはAgent所有。Learner Stateを参照しない。
- RewardScaler::ScaleWithoutUpdate()で倍率・clipを統計/clip率更新なしで適用する。DefaultDQNはAgent既存mutexで読出しを保護し、Rainbowは既存の無変換とする。追加forward/RNG消費なし。
- EpisodeStatsAccumulatorのstep読出しを追加する。Runnerのstep_in_episodeは既存集計から返し、終端では完了長を返す。EvalRunnerはUpdateMetricsとEpisodeEnd処理をTrain通知前へ移す。counts更新時機は維持する。
- 単一のEvalTracePanelを既定非表示で追加する。仮想表・色・数値書式・既定列・上限はD8〜D13どおり。TraceFollowerをView/Eval/QValue/Conv2dへ実装し、不変snapshotを共有所有する。QValue生成は重複させず、Conv2dはCPU活性だけを保持して選択時に描画する。未取得の行はno dataとする。
- TracePlayerはD12どおり。行選択/再生でEval停止、Liveへ戻っても自動再開しない。停止/Esc/手動行動/Resume/非表示を配線する。購読は固定し、非表示中は生成/診断を行わず、再表示時に全窓を空にする。HeatMapは対象外。
- common/Atariへ既定設定を追加し、設計100/140/160・ユーザーガイド020・関係wikiを同期する。CONTEXT/ADRは既存決定を再利用する。

## テスト

Public surface: ObserverFactory/RunManagerの既存metrics出力、EvalRunner::DoStepと通知、列設定、StepTraceBuilder、AgentStepDiagnostics、RewardScaler、TracePlayer、追従paneのsnapshot/Live切替。

純粋refactorは既存GREEN→整理→GREENで確認する。新機能は1 behaviorずつRED→最小実装→GREEN。最初のtracer bulletは実EvalRunnerの通知から$env_step rewardの行が生成される経路。続いて列の正常/拒否、終端時点・予約キー・強制行動・Runner集計、診断とInitialPriorityCompleter経由の一致、n-step/終端/truncation/TBO/Munchausen/scaler/PER無効/非DQN、遅延値の行対応、百分位・履歴・再生端・速度・空履歴・snapshot所有を順に確認する。全テストを先に追加しない。

## 検証

変更前の実行体/DLL/設定/hashと同seedの決定的Runを固定する。Atari、DropMerge、LunarLander CPU、GridMazeで段階1と最終版を比較し、actor_approxでヒント共有化前後の学習系列を比較する。時間/資源値とtimestampだけを明示的に除外し、値・step・defs・WARNを比較する。PRD097固有の移行例外は利用しない。資材はcore/anet-core/testdata/prd094、結果は.scratch/prd094へ保存する。

VsDevCmd.bat経由でDebug/Releaseビルド、関連テスト、全体CTestを実行する。Pythonはリポジトリ.venvを使う。実機ではD12の操作・画面同期、開く前1分×3の学習速度、開いた状態のEval速度95%以上/学習速度変動幅内、1000行のメモリを確認する。利用できない実機検証は未実施と記録し合格扱いしない。

## 前提

- 未決事項監査のユーザー判断ブロッカー0。PRD097は22implと現行コードで完了確認。env_stepは既存結果APIを共用する。
- 百分位はユーザー選択の中央順位100*(小さい値の件数+同値件数/2)/件数。自身を含めて有限値の直近pct_window個から確定時に算出し、行の表示値として固定する。全件同値50%。max_rows=1000、pct_window=10000。
- eval_panelのtarget network既定を維持。traceチャネル保存/metrics DSLのagent_step追加は081へ残す。Notifier全体は直さない。
- ユーザーは探索Run中のビルド/テスト/追加Runを許可し、基本的に質問せず進めるよう明示した。探索終了待ちはしない。こちらの追加Runは同時1本。探索の実行体/設定を保護し必要なら出力を分離する。通常の修正・再試行は自律的に行う。並列負荷を記録し、性能が判定不能なら未検証として報告し、半日規模の反復/プロファイル調査へ広げない。
- 無関係なdirty変更と過去artifactを保持する。commit/pushしない。実装後は品質確認観点で自己点検し、独立an-reviewで計画/実装をレビューして修正・再レビューする。結果/未実施/残る未決を本書へ積み、工程を終了する。

## 経過

- 2026-10-21: ユーザーの実装開始指示を受け、計画をコード変更前に保存した。百分位の選択と探索並行許可は承認済みであり、再確認は行わない。

- 段階1: 変更前のObserver/Runner公開経路55tests/503assertionsが成功。共通読取と配線の抽出後も隔離Debugビルドと同55tests/503assertionsが成功。ログexisting-green.log、stage1-build.log、stage1-green.log。Releaseの段階1実行体を構築中。DLL配置の初回コピー時間は性能測定ではない。

- tracer bulletはstep_trace.hpp未実装のC1083でRED、最小生成器の追加後に1test/4assertionsでGREEN。列契約テストはdisplay:nameがvalueのままになるREDを確認し、字句だけを共有するparserを実装中。新しいmetric_tokens機能グループはmetricsとstep traceの二つの利用者で字句分類を共有するために設けた。
- コード照合の補足: 現行EvalPanel::InitializeはUpdateViewData(event,true)を呼んでおり、PRDの「frameごとのみ」という現状記述より既にstepごとの生成になっている。今回のsnapshotはその生成結果を共有し、追加生成を避ける。PRDの要求する時点と表示は変えない。

- 段階1のReleaseビルド成功。5構成の変更前2反復が一致し、段階1は全47739点のscalar/trace・両定義・WARN本文/順序/件数が一致（stage1/full-equivalence.json）。比較器は数値・step・lane・data・重複・定義・WARNの差を拒否する契約テスト成功。性能合否には使用しない。

- 時点契約はBad optional accessでRED、共通target読取/ActionInfo action/Runner step_in_episode/通知前更新の実装後に既存回帰込み58tests/1621assertionsでGREEN。診断なしAgentは既知列をNaNとして返す。報酬変換は未実装methodでRED、ScaleWithoutUpdate追加後に関連5tests/1130assertionsでGREEN。学習Scaleの演算は保持した。

- DQN診断は実Agentの部品未提供でRED、共有ヒント/系列抽出factory/推定器の接続後24tests/1272assertionsでGREEN。n=1/3・通常/終端/打切り/両フラグ・TBO on/off・Munchausen on/offの32組合せを実Agent→診断と既存InitialPriorityCompleterで比較し、関連6tests/1553assertionsが成功。新factoryは既存NStepQueueController/DefaultExperienceBuilderをその場で生成する共通口で、系列の式・演算順序は動かしていない。
- GUIに依らず契約を確かめるため、step_trace機能内にTracePlayerの操作APIとAppendTracePercentile（有限値窓への追加と中央順位）、StepTraceColumnSpec::Format（数値/行動表示）を置く。百分位も0〜100の数値を同じ数値書式で表示する。この補助surfaceをテスト対象へ固定する。

- プレイヤーは未実装型でRED、GUI非依存の行ID/速度/端/Live制御を実装して関連7tests/1585assertions成功。百分位/Formatも未実装口でRED、実装後8tests/1600assertions成功。
- snapshotの型消去はTraceSnapshot（immutable dataを所有する表示closure）とTraceFollowerの3操作で行う。closureはViewをweakで参照し、通知や借用eventを保持しない。ViewBaseは既存の更新頻度とUIDataStoreの同期を保ち、同じstepの表示データを再生成しない。

- malformedな列位置を無視するREDを確認し、columns配下の番号形式も構築時に検証するよう修正した。追加通知のcallbackへTrainEventを同期引数として渡し、GUIが通知の借用を保持しない契約にした。View snapshotテストは9tests/1609assertionsで成功。
- Computer Useは初期化時に`trusted Node process exited unexpectedly; kernel reset, rerun your request`で失敗した。実機のpane操作・95%速度・1000行メモリは未実施として残し、成功したものとして扱わない。探索共存中に別のUI操作経路や長時間性能調査へ拡大しない。

- Debug全体ビルドは成功（gui-green-build.log）。列の境界を含むPRD094は10tests/1613assertionsで成功。既定葉監査は26861代入・3473検査・errors 0、比較器の契約テストは成功。後続の遅延行対応テストを追加中。

## 検証コマンドと結果

PowerShell、リポジトリルートで実行。探索中の実行体を保護するため、Debug/ReleaseともCMakeのproject includeで実行体の出力を`.scratch/prd094/bin`へ隔離した。検証後はcacheのinclude指定を外し、通常の出力先へ戻す（通常の実行体を再ビルド・上書きしない）。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 6'
```

Debug全体成功（gui-green-build.log）。PRD094の追加テストは以下の公開surfaceを通る。最終追加時点で10tests/1622assertions成功（delayed-row-green.log）。

```powershell
.\.scratch\prd094\bin\Debug\anet-core-test.exe '[prd094]' --abort
```

```powershell
ctest --preset x64-Debug --output-on-failure -j 1
```

全6ターゲットのCTest実行中（ctest-debug.log）。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner -j 6'
```

Release Runner成功（final-release-build.log）。

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

26861代入、3473検査、errors 0。

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd094/compare_metrics_test.py
```

1test成功（数値・step・lane・data・重複・定義・WARNの差を拒否）。

```powershell
$env:PRD094_BINARY_DIR='C:/dev/anet-lab/.scratch/prd094/bin/Release'; $env:PRD094_AFTER_NAME='after-v1'; .\.venv\Scripts\python.exe core/anet-core/testdata/prd094/run_validation.py after
```

5構成を各1本ずつ実行中（final-validation.log）。段階1は5構成47739点と定義・WARNが一致済み。元Run artifactは変更しない。

## 自己点検

品質確認観点docs/quality/impl.mdの全項目を走査した。列の現在契約、借用eventの同期寿命、snapshotの不変共有、診断窓の所有、既定target維持、forward/RNG追加なしを照合。公開ヘッダはPRDのBuilder/診断/Followerと共通読取りを追加し、依存方向はAgentが部品を生成、生成器が部品を所有する。新規step_trace/metric_tokens/EvalTracePanelはそれぞれ独立した生成/字句/UIの機能グループである。

C++ヘッダのusing namespace、production dynamic_cast、追加TODOの不正書式を検索し、いずれも0件。変更範囲は無関係なdirty変更を除外して点検した。改行はLF、batは変更対象外。旧scalar提供元のdynamic_cast撤去に伴い、テストのTestActionInfoもGetScalarSourceへ移行する。設定は共通既定を?=、Atari選択を=で追加し、解決後の既存全キーと文字列値を出力先以外で比較する。

実機のpane操作・速度95%・1000行メモリはComputer Use初期化失敗により未実施で、合格扱いしない。ソース・文書のreadback/hash/mtimeはsource-audit.jsonへ記録。相対リンク検査で見つけた既存optuna.mdリンク3箇所は変更前からのもので、今回の変更の外にある。

- レビュー指摘対応: TraceのQValue snapshotは実行行動でハイライトするimmutableコピーを保持する。既存Liveのraw_actions表示は維持し、未承認の999のLive状態の先読み問題へ拡張しない。強制行動の列/q_saは自動テスト済みだが、QValue native UIの一致はComputer Use初期化失敗のため未検証。

- QValueのTrace数値欄も、IQN観測用full分布の平均ではなくaction_infoのq_valuesへ揃える。分布画像は判断時点の分布を保持し、Liveの既存生成・選択表示は変更しない。連続行動次元・reward予約キー・未知キーの公開Builderテストを補った。

## 最終比較の結果

- after-v1の全5構成で、既存scalar/traceの全47739点、両定義、WARN本文・順序・件数が一致した（after-v1/summary.json、各comparison.json）。Atari5008、DropMerge14033、LunarLander7242、GridMaze11146、actor_approx10310点。
- 全キー・文字列比較は下記で成功。Atariは既存1215キー＋意図した14キー、DropMerge1336＋12、LunarLander811＋12、GridMaze695＋12、actor_approx811＋12。既存値の変更・削除0。app.runs_dirだけは各世代のRun親への一致を別途検査した（after-v1/config-equivalence.json）。

```powershell
.\.venv\Scripts\python.exe core/anet-core/testdata/prd094/compare_configs.py --after after-v1
```

- 比較後の修正は、QValue Trace表示・操作列順・設定エラー・単一行slider・レイアウト非表示のGUI側と、metric_tokens.cppの字下げだけで、学習・Runner・診断の数値処理は変更していない。Releaseはこれらを反映して再ビルドする。性能値の合否にはRunの所要時間を使わない。
- 全CTest初回は5環境ターゲット成功、coreは724tests中1件失敗（149756assertions中1件）。GetScalarSource導入時のTestActionInfoの移行漏れであり、2つのテスト具象型へ明示提供口を追加した。修正後core全体を再実行中。
- 機械検査はヘッダusing namespace・production dynamic_cast・大文字TODO・共有プロセス契約の禁止語がいずれも0件（self-check.json）。git diff --check成功。
- 独立レビュー初回は計画OK、実装差し戻し4件（review-20261021-0139.md）。QValueと操作順、設定範囲エラー、指示付き初期化、設定比較記録をまとめて修正した。自己修正で解消し、ユーザー判断待ちは作らない。

- 再レビューは計画/実装ともOK、追加の不足・範囲外検出・ユーザー確認事項なし（review-20261021-0147.md）。全coreとtrace_readerの結果だけは終了後に追記する。Release修正ビルドは成功。

## 完了報告

- 実装と独立再レビューを完了。計画・実装ともOK。レビューで出た4件は対応済み、未決の設計判断と起票候補はなし。
- Debug全体ビルド後、最終GUI修正と追加テストもDebugでビルド成功（final-debug-build.log）。Releaseも修正後ビルド成功（review-release-build.log）。
- 全6 CTestターゲットを確認した。環境5ターゲットは初回で成功。coreはテスト具象型の移行漏れを直し、次の再実行で724tests中723成功・1skip、149756assertions全成功（ctest-core-fixed.log、375.09秒）。

```powershell
ctest --preset x64-Debug --output-on-failure -R '^anet-core-test$'
```

- 追加の連続行動次元・予約reward・未知キーを含む最終PRD094は11tests/1627assertions成功（final-prd094.log）。実Agent診断・遅延行・Runner時点・percentile・player・snapshot寿命を含む。
- 数値と設定は「最終比較の結果」のとおり全5構成一致。通常のRun artifactと探索用実行体は上書きしていない。検証実行体は`.scratch/prd094/bin/Debug/AnetRLRunner.exe`と`.scratch/prd094/bin/Release/AnetRLRunner.exe`に残す。通常launcherの実行体は更新していない。
- CMakeの一時project includeをDebug/Releaseのcacheから外して再configureし、今後の通常ビルドを元の出力先へ戻した。復帰時のbuildは行わず、既存の実行体を保護した（restore-debug-cache.log、restore-release-cache.log）。
- commit/stage/pushは実施していない。ユーザーの別作業の変更を保持した。

### 未実施の検証

実機のpane表示・選択・強制行動ハイライト・Conv2d同期・再生操作、開く前1分×3の学習速度、表示時Eval95%以上・学習速度の変動幅、1000行メモリは未確認。Computer UseのNode初期化が失敗したためであり、単体テスト・数値比較を代替の合格根拠にはしない。操作ガイドは020 §6.5、受入手順はPRD094 §8とtestdata/prd094/README.mdに残す。半日規模の性能反復には広げなかった。


## レビュー後の修正（2026-10-21）

### 承認された計画

ユーザーのPLEASE IMPLEMENT THIS PLANに基づき、現作業ツリーを基準として8件を修正する。PRD本文の訂正済み箇所を保持し、通常の実行体と別作業を保護する。基準ソース・hashとRelease実行体・DLL・設定は`.scratch/prd094/review-fixes-20261021/`へコード修正前に保存した。

1. RunningStdRewardScalerはScaleで既存の計算が取得したstdをatomic<float>へ公開し、ScaleWithoutUpdateはそこだけを読む。Resetでも実際の初期stdを公開する。Agentの共有lockを診断から外す。学習の式・演算順と統計更新は保持する。checkpointは現行どおり報酬統計を保存・復元しない。
2. paneの開閉判定を最大化とsavedHiddenState込みで共通化する。最大化で隠れた間も記録・診断・snapshotを継続し、明示closeだけがLive復帰と全窓消去になる。OnPaneShowingを追加し、最大化中の開閉とメニューcheckも復元後に保つ。
3. captionだけの更新は再描画とし、Trace開始・Live復帰・追従pane表示変更だけでAUI Updateを呼ぶ。浮動paneのタイトルも直接更新し、layoutとcaptionに別のProfileRangeを付ける。
4. actionのcarrier直接読取りをlane>=0へ限定。scalarは従来のGetScalarSourceへ委譲し、次元はfrom_charsで厳密解析する。
5. TracePlayer::CanPlay(base_rate)でFPS 0時は再生ボタンだけ無効にする。実行中の再生状態は維持し、0では進行しない。FPS復帰で続行する。
6. MetricsWiringのscope解決・event attach・定義補完/出力・購読ヒントを同期メソッドへ整理する。順序・WARN・出力と依存寿命を保持する。
7. 020 §6.5の本文を手順中心にし、仕組みはNOTE/160へまとめる。指定4ヘッダのBOMを基準どおりに戻す。

振る舞い修正は1件ずつRED→GREEN。公開口の診断競合・std更新/Reset・pane状態・scalar未知action・FPS 0をテストする。ロック競合の失敗でも解放・joinを必ず行う。純粋整理は既存GREENを維持する。新しい独立ファイル群は増やさず、GUI状態補助は既存gui機能グループへ置く。

Debug/Releaseは既存のproject includeで`.scratch/prd094/bin`へ隔離し、PRD094/関連/core CTestを実行する。修正前後の5構成のscalar/trace・定義・WARN・既存設定を別世代で比較し、Runは同時1本。終了時はcacheのincludeを外す。実機pane操作・速度・メモリは今回対象外。品質観点の自己点検と独立an-review、結果記録まで行う。commit/stage/pushはしない。

### 経過・検証結果

- 基準の保存と計画の追記をコード修正前に完了した。

- 指摘1: Agent unique lock保持中のObserveが時間内に戻らないREDを確認（review-lock-red.log）。atomic std公開とReset公開、診断の共有lock撤去後、関連4tests/435assertionsでGREEN（review-lock-green.log）。

- 指摘4: scalar actionがlane=-1でANET_CHECKになるREDを確認（review-scalar-red.log）。lane条件と厳密解析の修正後、scalar/連続次元/予約キーの関連3tests/28assertionsでGREEN（review-scalar-green.log）。未知のscalar鍵の既存WARNも保持した。

- 基準driverのREPOが凍結ソースの場所を向いていたため、初回採取は比較から除外した。REPOを実リポジトリへ明示し、修正基準をreview-fixes-20261021/acceptanceへ新規採取する。既存結果は書換えない。ハーネスへPRD094_ACCEPTANCE_ROOTを追加して保存世代を分離し、execution記録にcwdも含める。

- 指摘5: CanPlay不在のRED後、再生可否・速度0保持と復帰・他操作の2tests/46assertionsでGREEN。
- 指摘2: 共通pane判定不在のRED後、最大化・明示開閉・復元・浮動/toolbar判定の1test/10assertionsでGREEN。開閉フックとメニューにも同じ判定を適用した。実機イベント順序の確認は今回対象外。
- 指摘3はレイアウト再構築を減らす整理として、既存の選択・追従テストのGREENを維持しつつ、caption描画とlayoutの計測を分離する。呼出回数と浮動タイトルの実表示は未確認であり、測定したとは扱わない。
- 整理後のDebug関連テストは`[prd094],[observers],[scaler],[runner]`の49tests/2083assertionsでGREEN（`review-wiring-green.log`）。比較器は数値・定義・WARNに加え、追加/欠落/文字列変更/不正な出力先を検出する2testsでGREEN。
- 指定4ヘッダのBOMを復元した。基準hashとの比較で、PRD訂正済み本文と無関係な4変更の内容保持を確認した。020 §6.5は操作手順を本文へ残し、時点・百分位・優先度・Qの仕組みをNOTEへまとめた。140/160も追随した。
- 自己点検: `docs/quality/impl.md`の55観点を走査。機械検査はヘッダusing namespace / 大文字TODOが各0、共有プロセスのanet/PRDが0、計画名・対応PRDが適合、`git diff --check`が成功。production dynamic_castの追加は0。初回の修正一覧からgui.cppが漏れていたため補完し、同ファイルの基準から存在する3件（87・92・100行）は今回範囲外として分離した。公開口の追加は承認済みのCanPlay・pane判定・表示直前フックだけで、永続借用や互換層は追加していない。新規prodファイル・設定変更・skill変更は今回なし。実機受入と計測回数を合格扱いしない。
- Debug/Release全体ビルド・core CTest・修正後5構成の比較はこれから結果を追記する。ビルドはVsDevCmd経由の既存隔離先、基準は採取済みの5構成各1回を使う。
- 独立an-reviewへ修正差分を提出し、実装計画・実装ともOK（差し戻し0）を受領した。全55観点を走査し、上記のgui.cpp既存3件だけを新規の範囲外検出として記録した。レビュー時点の全体ビルド・core CTest・数値比較は未完了だったため、以下へ別に完了結果を記す。

### 修正後の完了確認

- Debug/Release全体ビルド成功（`review-full-build.log`）。`VsDevCmd.bat -arch=x64 -host_arch=x64`をcallして各presetを`-j 6`でビルドし、実行体は`.scratch/prd094/bin/{Debug,Release}`へ出力した。
- 両構成で`anet-core-test.exe "[prd094],[observers],[scaler],[runner]" --abort`が成功（各49tests/2083assertions）。Debugは`review-wiring-green.log`、Releaseは`review-release-related.log`。
- `ctest --test-dir out/build/x64-Debug -R '^anet-core-test$' --output-on-failure`成功（`review-core-ctest.log`）。731test cases中730成功・既存1skip、149819assertions成功。所要850.01秒は回帰テストの記録であり、速度の受入測定には用いない。
- 比較器の`compare_metrics_test.py`は2tests成功。数値・定義・WARNと、既存設定の追加/欠落/文字列変更/出力先違反の検出を確認した。
- `VsDevCmd`経由で両presetを`-U CMAKE_PROJECT_anet-lab_INCLUDE`付きで再configureし、一時注入が両cacheから消えたことを確認した（`review-restore-cmake.log`）。解除後のビルドは行っていない。固定済み比較実行体と隔離Releaseのhash一致も確認した。

修正前後は`review-fixes-20261021/acceptance/{before,after-v1}`へ分離し、seed=1・deterministic・65536経験で各構成1回ずつ、追加Run同時1本で比較した。scalar全点・trace全行（既存の時間/資源群とtrace timestampを除く）と、両定義・WARNがすべて一致した。時間/資源の値は数値等価性から除外する既存契約であり、速度合格の根拠にはしていない。

| 構成 | 一致点数 | 既存設定キー数 | 設定の追加・欠落・値変更 |
| --- | ---: | ---: | --- |
| Atari | 5008 | 1229 | 0 |
| DropMerge | 14033 | 1348 | 0 |
| LunarLander CPU | 7242 | 823 | 0 |
| GridMaze | 11146 | 707 | 0 |
| actor_approx CPU | 10310 | 823 | 0 |
| 合計 | 47739 | 4930 | 0 |

`run_validation.py after`と`compare_configs.py --after after-v1 --existing-only`が成功し、解決済み全キーと文字列値のhashも両側一致した。出力先だけは各世代のRun親と一致することを別途検査した。結果は同世代の`summary.json`・各`*.comparison.json`・`config-equivalence.json`へ保存した。

今回の8件の対応は完了。未確認は指定どおり実機pane操作・UIイベント順序・浮動タイトルの表示・ProfileRange呼出回数・速度・メモリであり、合格に読み替えない。範囲外として見つかった既存gui.cppのdynamic_cast 3件は今回変更していない。訂正済みPRDと無関係な4ファイルは保存時hashのまま保持し、通常の実行体を上書きせず、stage・commit・pushも行っていない。

## GetScalarベースへの汎用化（2026-10-21）

### 承認された計画

ユーザーのPLEASE IMPLEMENT THIS PLANに基づき、641b8c81を基準に案Aを実装する。共通の固定値型AgentStepValuesを削除し、既存Module/GetScalarへ委譲する。TensorDict・新しいscalar用interface・互換層は追加しない。色調整と追加Run・性能測定・実機UI確認は対象外。無関係な4変更を保持する。

- AgentStepDiagnosticsResult.currentはbatchの値を所有するunique_ptr<const Module>とし、GetScalar(key,lane)で読む。lane別pending_keysを持たせ、NaNと未確定を分ける。
- AgentStepCompletionはlane・steps_back・resolved_keys・unique_ptr<const Module> valuesとし、対象1行をGetScalar(key,0)で読む。DQNの具体値型と3キーの認識はDQN実装内に置く。Agentが診断を生成し、窓・演算順・forward・RNG・Reset・ロックなしの診断を維持する。
- 列parserのDQNキー限定を除去する。提供元ありの未知キーは最初のGetScalarで列番号・キー付きfail-fast。既知未成立はNaN。診断非対応Agentは$agent_step列をキーによらずNaNにする。
- StepTraceに列別value_readyを持たせ、選択列のpendingだけを待つ。更新通知は新たに確定した列index（columns配列の0始まり）を同期spanで渡す。部分更新・NaNでの確定を扱い、UIの保持上限から消えた行も通知された列だけを百分位窓へ一度入れる。
- approx_priority専用分岐とhas_priority_column_を共通処理/GUIから除去し、設定・既存表示値を維持する。所有値は通知中にscalarへ変換し、長寿命の借用を追加しない。

Public surfaceはStepTraceBuilder/列設定、AgentStepDiagnostics、既存GetScalarと列更新通知。TDDは任意キーの記録を最初の縦スライスとし、非DQN提供元の複数lane/複数時点での確定、未知・NaN・非対応・Reset・保持上限後の百分位を順に確認する。DQN参照値とmutex競合の既存テストをGetScalarへ移行する。

Debug/ReleaseはVsDevCmdから既存のisolate.cmakeを注入し、.scratch/prd094/binへ隔離する。関連テストとDebug core CTestを実行し、終了時に両cacheの一時設定を外す。PRD・設計140/160・検証資材を更新し、品質観点の自己点検と独立an-reviewを行う。stage・commit・pushは行わない。

### 経過・検証結果

- 未決事項監査: ユーザー判断が必要なブロッカー0。修正範囲はTrace側も汎用化すると回答済み。Moduleの既知NaN/未知nullopt、診断の所有先は既存規約を維持する。列更新のindexと同期span、診断非対応時の空欄維持を計画へ固定した。
- RED→GREEN 1: `[generic_step_keys]`で任意キーentropyが旧parserに拒否されることを確認し、parser制限と非対応時の固定キー提供元を撤去。`generic-keys-red.log`は期待した失敗、`generic-keys-green.log`は列parserを含む2tests/1029assertions成功。最初のテスト入力のコンストラクタ引数不足はRED判定前に修正した。
- RED→GREEN 2: `[generic_step_completion]`を追加し、Module所有値・pending_keys・列別ready・更新spanが旧APIに無いことをコンパイルで確認（`generic-completion-red-build.log`）。共通API・DQN提供元・GUIを同じ変更で移行し、`generic-completion-green.log`は汎用確定とDQN参照値の3tests/464assertions成功。
- 続く契約検証: 2laneで任意キーの即時値、finiteでもpendingの列、既知NaN、別時点で確定する2列、同じキーの複数列、確定済みキーの再通知抑止、NaNでの確定、Reset後の旧行への通知消去、未知キーの列番号・キー付きエラーを確認。表示1行のconsumerと実際の百分位窓を接続し、表示外の部分更新でも通知列だけが一度投入されることを確認した。実機GUIの表示確認とは区別する。
- Debug関連テスト: `powershell -NoProfile -ExecutionPolicy Bypass -File .scratch/prd094/build.ps1 generic-contracts-green '[prd094],[observers],[scaler],[runner]'`成功、53tests/2089assertions（`generic-contracts-green.log`）。`[diagnostic_no_wait]`も含み、Agent mutex保持中の完了、報酬std更新とReset、replayとの優先度一致を維持した。競合テストはロックを解放してからfutureを回収する。
- 旧固定型とhas_priority_column_は現用コードに残存せず、step_traceとEvalTracePanelにDQNのキー判定は残っていない。DQNのヒント・報酬変換・優先度の式と演算順・forward・RNG経路は差分で維持を確認。PRD D5/D6、設計140 §6.1・160 §3.4、検証資材READMEを更新した。
- 品質観点の自己点検: 全55項目を適用範囲で走査。機械検査では変更ヘッダのusing namespace、productionのdynamic_cast、TODO形式違反、共通プロセスへのプロジェクト依存が0件。全13変更ファイルはw/lf、git diff --checkも成功。新しい公開契約は本計画へ固定済み、Moduleは値を所有し借用メンバを増やさず、互換層・設定変更・新規ファイルは無い。
- Debug/Releaseの全体ビルドとDebug core CTestは隔離先で進行中。Release関連テストはCTestとの一時ファイル競合を避けて順次実行する。追加の比較Run・性能測定・実機UI確認は計画どおり行わず、未確認を合格へ読み替えない。
- 独立an-review初回: 計画はOK、実装はPRD D9の通知契約の移行漏れ1件で差し戻し。D9を部分確定ごとの列更新通知へ揃え、最終確定はrow.complete、PRD081の出力はその条件で1行だけと明記した。即時完了行の空span通知も記録した。新規の範囲外検出は無い。
- Debug/Release全体ビルド成功（`generic-debug-build.log`・`generic-release-build.log`）。両方ともVsDevCmdから`cmake --build --preset x64-Debug` / `x64-Release`を実行し、出力先は`.scratch/prd094/bin/{Debug,Release}`。Debug core CTestは進行中。
- 独立an-review再確認: D9の修正が実装と整合し、計画・実装ともOK（差し戻し0）。コードを変更せず、検証の完了記録を残す段階へ進んだ。
- Debug core CTest初回は735tests中733成功・1失敗・1skip（149825assertions中1失敗、`generic-core-ctest.log`）。失敗は未変更の`EpisodeEvalObserver destructor catches a background failure`の終了ログ確認1件で、テストにはworker完了を待つ50ms固定sleepがある。単独実行は1test/1assertion成功（`generic-observer-isolated.log`）。原因を確定したものとは扱わず、ソースは変更せず、全体ビルドと重ねない状態でcore CTestを再実行中（`generic-core-ctest-retry.log`）。
- Release関連テスト: `.scratch/prd094/bin/Release/anet-core-test.exe '[prd094],[observers],[scaler],[runner]' --abort`成功、53tests/2089assertions（`generic-release-related.log`）。初回CTestで失敗した既存Observerテストも成功した。
- Debug core CTest再実行成功: VsDevCmd経由の`ctest --test-dir out/build/x64-Debug -R '^anet-core-test$' --output-on-failure`が100%成功。735tests中734成功・既存1skip、149825assertions全成功（`generic-core-ctest-retry.log`・`generic-core-ctest-retry-details.log`）。初回失敗を消さずに併記し、反復性能測定とは扱わない。
- 終了処理: 両presetで`cmake --preset x64-Debug -U CMAKE_PROJECT_anet-lab_INCLUDE` / `x64-Release`をVsDevCmd経由で実行し成功（`generic-restore-debug.log`・`generic-restore-release.log`）。両cacheに一時includeが残らないこと、通常実行体4本のhashが変わらないことを確認した。復帰後の通常ビルドは行っていない。
- 今回のGetScalar汎用化は完了。自己点検・独立レビューとも対応済み。対象外の色調整・追加比較Run・性能測定・実機UI表示/呼出回数は未実施のまま保持する。無関係な4変更には触れず、stage・commit・pushは行っていない。

## 名前・継承・配置の見直し（2026-10-21）

### 実装計画と前提

現在の作業ツリーを基準とする。ユーザーの修正依頼に従い、AgentStepDiagnosticsProvider系へ名前を統一し、BatchActionInfoをModuleBaseにしてGetScalarSourceを廃止する。TraceFollowerの口はCreateTraceSnapshot、StepTrace.stepはstep_tへ揃える。EpisodeStatsAccumulatorの個数型は変更しない。MetricsWiringはRunManagerのprivate入れ子とする。

値の読取り・列定義/行/生成器はmetrics.hpp/cpp、内部字句分類はmetrics_impl.hppへ集約する。再生・百分位・セル書式はgui.hpp/cppへ移し、生成テストはmetrics_test.cpp、GUI補助テストはgui_test.cppへ分ける。旧step_trace.*・metric_tokens.*は削除する。追加ファイルはこの機能グループとテストだけとし、Logger/Observer/EventFieldの責務は維持する。ソース収集はCMakeのCONFIGURE_DEPENDS付きGLOBであるため、再configureで新旧一覧を反映し、明示一覧は増やさない。

唯一の挙動変更は、Moduleを提供しなかった非DQN ActionInfoの未知scalarがnulloptとなり、既存のvalue not found WARNへ進むこと。lane指定のaction/action.[次元]は基底GetScalarへ移し、scalarのlane=-1では認識しない。DQN固有値の集計・演算、既存WARNの間引き規則、forward/RNG、色は変更しない。Providerのpendingキーは既知NaNを返し、値が成立しなかった場合もNaNで確定を返す契約をコメントに残す。

### テストと検証

- Public surfaceはBatchActionInfo/GetScalar、MetricValueReader、MetricsLogTrainObserver、AgentのProvider生成、StepTraceBuilder、GUIの純粋な再生/表示補助。
- RED→GREEN 1: 非DQN ActionInfoの未知scalarが既存Observer経由でWARNになることを確認する。RED→GREEN 2: 基底GetScalarがlane指定の実行行動へ答え、lane=-1と未知キーにはnulloptを返すことを確認する。その後の純粋な改名・移動は既存関連テストのGREENを維持する。
- 基準ソース・作業差分・実行体/設定を`.scratch/prd094/naming-layout-20261021`へ凍結済み。無関係なbat・921・anet.asta・wiki/logと通常実行体のhashを保存した。既存の汎用化差分はそのまま保持する。
- 比較は既存run_validation/compare_metricsを使い、5構成・同seed・決定的設定・65536経験を修正前後で各1回、追加Runは同時1本。基準の反復性能測定は行わない。scalar/trace/定義/WARNと既存設定を比較する。
- Debug/ReleaseはVsDevCmd経由、既存isolate.cmakeを注入して`.scratch/prd094/bin`へ隔離する。関連`[prd094],[observers],[scaler],[runner]`とDebug core CTestを順次実行し、最後に両cacheの一時includeを外す。実機UI・速度/メモリ測定は対象外。
- PRDは冒頭改訂と§4.4の名称/区分を追随し、設計100/140/160・検証README・カテゴリmatchを更新する。履歴資料の旧名は書き換えず、現用範囲の移行漏れを検索する。自己点検・独立an-reviewと結果記録まで実施し、stage/commit/pushはしない。

### 経過・結果

- 未決事項監査: ブロッカー0。基準は未コミット汎用化を含む現作業ツリー。SnapshotのCreateへの変更、Provider名、EpisodeStatsの個数型維持、生成とGUI補助の配置、非DQN未知scalarのWARN化は指定済み。GUIセル書式の関数名はFormatTraceValueとする。規約上の分割・型の判断を今回の指定へ固定した。
- 文書追随ではPRDの§4.4と冒頭改訂に加え、D5のaction読取りの1文も共通基底GetScalarへ訂正した。旧文の「ModuleのGetScalarを要しない」が新しい継承/委譲と矛盾するためであり、Agent固有実装を要せず全Agentで読める意味は維持する。生成器と百分位consumerの結合テストはmetrics_test.cpp、純粋なGUI補助の単体テストはgui_test.cppとする。
- 修正前の5構成を固定Release実行体から各1回採取した（`naming-before-runs.log`）。同seed・65536経験、同時1本。各Runの実行引数・終了コード・実行体hashは世代別のexecution.jsonに保持した。
- RED→GREEN 1: 既存Observer経由で非DQN ActionInfoの未知scalarのWARNが出ないことをREDで確認（`naming-action-warn-red.log`）。ModuleBase継承とGetScalarSource削除後、同テストと従来scalar契約の2tests/10assertions成功（`naming-action-warn-green.log`）。
- RED→GREEN 2: Module経由の基底action読取りがnulloptとなることをREDで確認（`naming-action-lane-red.log`）。lane別読取り・厳密な次元解析を基底へ移し、DQNとテスト具象型の未知キーを基底へ委譲。関連55tests/2178assertions成功（`naming-action-lane-green.log`）。lane=-1のscalar集計やDQN固有の計算は変更していない。
- GREEN後に指定の改名・移動を適用した。new metrics機能群とgui補助、Provider/Createの各利用側を同じ変更で移行し、旧ファイルを削除した。CMakeはGLOB差分を検出して再生成している。現用のコード・設計文書・PRDに旧API名/旧includeパスの残存が無いことを検索で確認した。
- 改名・移動後もDebug関連55tests/2178assertionsが成功（`naming-layout-green.log`）。新しいmetrics.hpp/cpp・metrics_impl.hppはobservers/guiをincludeせず、生成器と表示補助の依存方向を確認した。新カテゴリのJSONとmatch、変更テキストのLF、公開ヘッダのusing namespace不在、保護対象のhashを検査した。
- 品質観点は全55項目を適用範囲で自己点検した。新しい公開API・配置・既知NaN/未知nulloptの契約は本計画と設計文書へ反映済み。処理移動先へコメントと既存ProfileRangeを保持し、所有値/同期借用の境界は変えていない。既存gui.cppのdynamic_cast 3件は基準と同数で今回増やしておらず、範囲外として保持する。WARN頻度とDQN固有scalarのlane集計も今回の修正に含めない。
- Debug/Release全体ビルド、Release関連テスト、修正後5構成比較、Debug core CTestと一時設定解除はこれから完了を追記する。CTestは全体ビルド・比較Runと重ねず、終了ログの既存50ms待機テストの競合を減らす。実機UI・速度/メモリの測定は未実施として残す。
- Debug/Release全体ビルド成功（`naming-debug-build.log`・`naming-release-build.log`）。両方とも通常の実行体を上書きせず、隔離先へ生成した。Release関連テストと修正後5構成比較へ進み、独立an-reviewには今回の凍結基準との差分43ファイル（既存変更/削除38、新規5）を提出した。
- Release関連テストも55tests/2178assertions成功（`naming-release-related.log`）。純粋な改名・移動の前後と両構成で関連テストのGREENを維持した。
- 独立an-reviewは計画・実装ともOK、差し戻し0。品質観点55項目を適用範囲で確認し、今回の差分への修正要求・開発者確認事項は無かった。比較Run・core CTest・一時設定解除の結果は完了後に追記する。
- 新たな範囲外の検出: 既存DQNActionInfoの既知キーtrain_actor_snapshot_interval/ageは提供状態が無い場合にnulloptを返しており、既知未成立をNaNとする規約と一致しない。凍結基準と既存テストも同じ契約のため、数値挙動の変更を限定する今回の依頼では修正せず、別件の検討候補として記録する。
- 修正後の5構成比較は全件成功（`naming-after-runs.log`）。Atari 5008点、DropMerge 14033点、LunarLander CPU 7242点、GridMaze 11146点、actor_approx CPU 10310点、合計47739点（うちtrace 1697行）が一致した。各構成のscalar/trace定義とWARNも一致。既存の時間/資源群とtrace timestampを除く既定の数値比較であり、速度やメモリの合格根拠にはしない。
- 解決済み設定の比較も全件成功（`naming-config-equivalence.log`）。5構成の既存4930キーに追加・欠落・値変更は0、全キー/文字列値のhashも一致し、世代別出力先の整合も確認した。結果は`naming-layout-20261021/acceptance/after-v1`のsummary.json、各comparison.json、config-equivalence.jsonへ保存した。修正前後各1回のみで、反復性能測定は行っていない。
- Debug core CTest成功: VsDevCmd経由の`ctest --test-dir out/build/x64-Debug -R '^anet-core-test$' --output-on-failure`が100%成功（`naming-core-ctest.log`・`naming-core-ctest-details.log`）。737tests中736成功・既存1skip、149914assertions全成功。全体ビルド・追加Runと重ねず1回で通過した。
- 終了処理: 両presetをVsDevCmd経由の`cmake --preset x64-Debug -U CMAKE_PROJECT_anet-lab_INCLUDE` / `x64-Release`で再configureし成功（`naming-restore-debug.log`・`naming-restore-release.log`）。両cacheに一時includeが残らず、保護対象9ファイル（無関係な変更5ファイルと通常実行体4本）のhashが保存時と一致することを確認した。復帰後の通常ビルドは行っていない。
- 今回の名前・継承・配置の見直しは完了。新規は指定の5ファイルに限定し、旧5ファイルを削除、現用利用側と設計文書を同じ変更で移行した。変更テキスト38ファイルはLFでgit diff --checkも成功、stageは空のまま。実機UI表示・呼出回数・速度/メモリ測定は対象外として未確認。色調整、範囲外の既存挙動、無関係な変更を保持し、stage・commit・pushは行っていない。
