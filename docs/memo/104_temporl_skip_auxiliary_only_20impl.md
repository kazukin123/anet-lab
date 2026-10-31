# TempoRL 補助学習のみモード 実装メモ

## 概要

正本は [PRD104](104_temporl_skip_auxiliary_only_10prd.md)。Actor ごとのスキップ長の制御と、補助学習用の窓の選択を追加する。既定は `full`・`decision` で現行挙動を維持する。

2026-10-29、ユーザーが本計画を承認し、「追加INPUTなしで段1まで実行」と指示した。その後「段0が終わったら一旦区切る（中断してレビューや整理）」で区切り、結果整理とレビューを完了した。さらに「PRD104 ステージ1の再開」計画を承認したため、現在の実行範囲はステージ1の7群×3 seed×2Mと分析・記録・wiki・最終レビューまでとする。ステージ2は設定と判定手順の準備までとし開始しない。

区切りの整理中に、ユーザーが`run_20261028-215839_a5_qb_aln_t103spl_s1`も同じ表へ追加するよう指定した。元3本と同じ1Run内のfull/single各200ゲームを追加採取する。学習時のビルド・構成が異なるため、元3本の多数決は維持し、splitは追加比較の行として示す。段1の停止は維持する。

## 主な変更

- `agent.hpp` の `DQNActorConfig` に `skip.max_length_mode`（single/full/variable）と任意の整数 ProfiledValue、`LearnerConfig` に `skip.target_window`（decision/action_change/episode_end）を追加する。Actor カタログの解釈は Agent が担当し、必須条件・値域・組み合わせ・WARN は PRD §5.2 の表に従う。上限Jの確定後に共通の ProfiledValue 検証を使う。
- 全Actorが常に長さ1となる条件は共通検証を上限1で評価し、違反する全Actorを一つのエラーに含める。`schedule.hpp` の共通検証から範囲の検査結果をログなしで取得できるようにし、型ごとの検査をAgentへ複製しない。既存の不正値に対するfail-fastは維持する。
- DQN Actor の single は毎step提案行動・判断=true・長さ1とし、skip forward・τ問い合わせ・スケジュール更新・J本のauxを維持する。variable は exp_step で上限cを評価し、選択に渡す値だけを1〜cに絞る。継続中の区間は上限低下でも切らない。状態はActorが所有する。`skip_max_length` は variable のc、ほかと無効時はNaN。
- replay の公開ヘッダへ `CreateActionChangeWindowRule(int)` と `CreateEpisodeEndWindowRule(int)` を追加し、実装は既存の機能グループへ置く。前者は行動変化の直前、後者は行動・判断によらず終了で閉じる。両者とも終了遷移を含み、J・dummy検査を守る。抽選・PER・容量条件・保存形式は変えない。Learner が設定から規則を選ぶ。
- 共通既定は `agent.txt` に `?=` で置く。実験の腕は `core/anet-core/testdata/prd104/arms.txt` の `run.@skip_aux` とCLIの葉で指定する。`50_agent_skip/07_max_length` は variable の腕だけで選ぶ。関連設計文書110/140/150/200と英訳を追随させ、既存のCONTEXTとADR0063の追加済み記述は重複させない。

## テスト

- Public interface / surface: ConfigManager/DefaultDQNAgentConfig、CreateActor/MakeAction、CreateLearner/UpdateFromBatch、GetScalar、ReplayBufferのPush/Sample/SampleUniqueUniform、実効設定とmetrics。
- 最初の縦スライスは single+decision の設定からActorを作り、返した行動を経験としてLearnerへ渡し、判断・長さ・有限のskip lossを確かめる。1テストをRED→GREENにしてから次へ進む。
- action_change、episode_endを一つずつ追加し、区間途中の起点、判断境界、行動変化、done/truncated、J、ring wrap、dummyを確認する。
- 複数lane、分布型の選択範囲、上限の時間変化、継続区間の維持、J本のaux、scalarのNaN/未知キーを確認する。
- 設定の不正値、ProfiledValue各型、全Actorの診断、WARN回数、skip/Learner無効時、継承とCLIを確認する。GREEN後にだけ整理し、関連テストを再実行する。

## 検証

変更前の実行体・DLL・設定・解決値をPRD104専用に保存し、PRD103の古いbaselineと取り違えない。既存Runの終了後にGPUを使う検証・実験を開始する。既存Runは別名の実行体で学習中であり、CPUのみのTDDはビルド2並列・Torch1 threadに制限して先行する（GPUの競合を避けつつ実装を進めるため）。

1. `VsDevCmd.bat` 経由で x64-Debug をビルドし、PRD104とPRD098/103、Replay、Schedule、設定baselineの関連回帰を実行する。長時間実験用にx64-Releaseもビルドする。
2. PRD103と同じ短縮条件（8 lane、batch16、warmup256、capacity4096、RR1、seed1、8704 experience、FP32、決定論backend、評価停止）の既定6構成を保存入力で前後比較する。除外は90_perf/・91_resource/・92_resource/だけで全scalar点を完全一致させる。
3. LunarLander短縮Runでaction_change/episode_end/variableのPRD受入条件を確認する。variableはlinear4→1、steps4352とし、指標が3→2→1へ単調に変わることを確認する。
4. 設定全キー・文字列値を比較し、許容差は新しい既定葉だけとする。既定葉検査、改行、BOM、差分を確認する。

### 段0

無効・合算・遮断・重み0.3の4 checkpointを評価する。TempoRLの各Runにはgreedyと派生greedy_singleを置き各200ゲーム。無効はgreedyだけ。同期評価で完走させ、不要なsessionを休止する。元RunのCLI葉を現在のAtari設定へ適用し、実効設定差を確認する。追加tagは53_evalg/60_single_score_mean・53_evalg/62_single_game_len_mean。gと副指標をPRDどおり集計し、分母0以下と読込不能の扱いも同書に従う。摂動つき評価は足さない。

CLIの`run.$`が選んだ構造も葉へ展開する。現在の既定とは異なるため、元dumpのmain_feature/tau_embeddingのstructure、HeadFC512とhead.projectionのNoisy有効、AtariEnv.resize_modeの5葉を補い、解決値の一致をassertする。これは旧重みの形状と評価入力を再現するためで、学習条件の変更ではない。CLI26葉だけでは形状が異なりcheckpointを読めない。旧SAMPLE_CHAINを使った準備出力は未実行のまま退避し、現在の連鎖で再生成する。

### 段1

2026-10-29、ユーザーが再開計画を承認した。開始時の採用結果は0/21本。別チャットで行うAtari評価とのGPU同時実行を許容する。

無効、合算、連続区間、終わりだけ、対照、k=1だけ、終わりだけ・分割の7腕をseed1〜3、各2Mで実行する。同一Releaseビルド・設定snapshotを用い、seedごとにこの腕順で21本を順次実行する。独立Runの失敗は記録して残りを続け、欠けた結果を成功扱いしない。

21_eval/02_policy_rewardのRun全体の有限点の平均をAUCとしてPRDの「以上・上回る・同じ帯」で判定する。必要ならEMAが200へ初到達するstepも記録する。既存の診断指標を保持し、対照が無効と同じ帯かも確認する。段2はレシピの準備まで。

## 前提

- 基準HEAD: `bbeab91b4ee85e179a6ec17194cf320cb8b807ac`。既存のdirty変更は保持し、設定snapshotにその時点の実効入力を含める。
- 未決事項監査: ユーザー判断が必要なブロッカー0。既定・API・組み合わせはPRDとrepo evidenceで解決済み。最新の実行範囲はステージ1の再開・分析・記録・wiki・最終レビューまで。Atariの条件変更・評価は別チャットで担当し、GPU同時実行を許容する。
- 既存Runは停止・再起動しない。通常の不具合は修正・再検証し、実験結果で生じる次の選択は記録する。段2・既定変更・commit/pushは範囲外。
- 新しい機能グループやtest-onlyのproduction APIは追加しない。既存のCreateDecisionWindowRuleと同じCreateの意味・配置へ揃える。hot pathへ同期転送・ロックは追加しない。
- 実験のスクリプトとREADMEはtestdata/prd104、生成した設定・証跡は.scratch/prd104、Runは専用出力先へ置く。原本checkpointと過去のRunは編集しない。

## 経過

- 2026-10-29: 計画承認。既存のAnetRLRunner_ab（PRD103 qbert split seed1）が実行中。コード編集以外の負荷のある検証は競合を避ける。
- 変更前snapshot: .scratch/prd104/baseline-manifest.json、85 files。元のDebug実行体・テスト実行体・DLL・入力・解決値を保存済み。CPUのみのTDDを先行し、GPU検証は既存Run終了後にする。
- singleのTDD: fixtureの設定スコープを調整後、red-single3.logで要求挙動の54失敗（判断率0.55、k平均1.5）を確認。最小実装後green-single.logで104 assertions成功。次にaction_changeの公開Learner経路を追加。
- action_changeのTDD: episode_startのfixtureを連続したepisodeに揃え、red-action-change2.logでk_max=1（要求3）の失敗を確認。実装後107 assertions成功。episode_endも交互の行動でk_max=1のRED後、実装して110 assertions成功。variableはskip_max_length未認識のREDを確認して実装中。
- CPU基準4構成はすべてexit0。GridMaze OFF26946点、LunarLander OFF24399点、GridMaze ON38479点、LunarLander ON34628点。正規化結果とhashは.scratch/prd104/validation/before_*/に保存。
- 上の試採取の点検で、再利用したPRD098の入力生成はGridMazeのrun.@skipが追加したgreedy sessionを止めていなかった。比較driverを解決後の全session列挙へ修正し、before_finalを正式基準として取り直す。旧beforeは証跡として保持し、受入の成功には数えない。

## 検証・レビュー結果

- Debug・Releaseとも `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'` の形式で成功（Releaseはpresetをx64-Releaseへ変更）。ログ: `.scratch/prd104/build-debug.log`、`build-release.log`。
- `core/anet-core/bin/Debug/anet-core-test.exe '[prd104],[profiled_value],[skip_policy]'`（OMP/MKL各1 thread）は35 cases・2418 assertions成功。公開経路、全ProfiledValue型の有効欄、組み合わせの集約診断、保持中の上限低下、複数lane、IQNのEpsilon/UQE/Thompson、既存skip policyのforward/τ回数を含む。ログ: `.scratch/prd104/debug-contracts.log`。
- `run_validation.py before --label before_final --env GridMaze LunarLander` と `after --label after_final --env GridMaze LunarLander`、`compare --label after_final --reference before_final --env GridMaze LunarLander` は4条件とも完全一致。OFFは26946/24399点、ONは38349/34628点。正式な基準・比較ログは `.scratch/prd104/after-final-cpu.log`、`compare-cpu.log` と `validation/before_final_*`・`after_final_*`。試採取のbeforeとは区別する。
- `run_current.py --label smoke1` は3条件とも成功（`.scratch/prd104/smoke.log`、`smoke/smoke1-summary.json`、各audit.json）。action_changeは判断率1086点がすべて1、k_max527点は1以上で1超を含みlossは全点有限。episode_endは同じ判断率と有限loss、k_maxは全点3.5以上。variableの上限527点は先頭3・末尾1で非増加、3/2/1をすべて含む。
- `verify_config.py` は4env×default/skipの8入力すべてunexpected=0。新しいfull/decision既定葉だけの追加で、既存キーの値・削除差はなし（`.scratch/prd104/verify-config.log`、`config/comparison.json`）。
- `core/anet-core/testdata/prd072/check_default_leaves.py` は既存Atariの`app.online.eval_panel.fps`・`app.online.train_panel.fps`の2件を報告。着手前snapshotにも同じ葉があり、今回の追加3葉に指摘はない。`.scratch/prd104/default-leaf-audit.log`。無関係な既存設定は直さない。
- 自己点検で全観点を走査。`git diff --check`成功、既存20ファイルのBOM保持、変更・新規テキストはLF、Python6資材のAST構文確認成功。独立an-reviewへ差分と未実施範囲を提出済み。
- 初回an-reviewは計画OK、実装は共通visitor内のconstant重複（項66/72）1件で差し戻し。kConstantへ揃え、再ビルド・契約テスト・最終実行体での前後比較を行う。原文: `.scratch/prd104/review-20261029-0929.md`。範囲外で検出された既存英訳150/200の古い記述は末尾へ保持する。
- 資材のRun名は、受入証跡を保存するため`tmp`を付けない形へ修正する。既採取Runは移動・改名せず残し、最終確認をbefore_final2/after_final2とsmoke2の新しい名前で採取する。
- 既存Runの親11_batch_run.batには後続Runが複数予約されている。後続キューのみの一時待機を検討したが、自動承認レビューが「段0・段1の承認に、既存ワークロードを遅らせる明示許可は含まれない」として拒否した。操作とhelper作成は実行されていない。既存の学習・キューを操作せず、GPU空き待ちを継続するため、実験開始は当初見込みより遅れる。
- レビュー修正後のDebug全ビルドと同じ契約テストは再度成功（35 cases・2418 assertions、`.scratch/prd104/build-review-debug.log`、`review-contracts.log`）。Release再ビルドと最終名のCPU受入採取は進行中。
- 第2回an-reviewは計画・実装ともOK、追加の確認事項なし。原文: `.scratch/prd104/review-20261029-0940.md`。未完了の実験結果は完了後に同じレビューで確認する。
- レビュー修正後のRelease全ビルドも成功。実験開始前に最終exeを再固定し、SHA256は`2f74a21f48cf72349f79323c06b1e2bee1af3542943d20870905c101f82156e1`。初期凍結exeも別名で保存し、recipes.jsonに両hashとレビュー参照を残した。最終名のCPU4条件も再度完全一致（`.scratch/prd104/compare-final2-cpu.log`）。
- GPU待機中に`preflight_checkpoints.py`を追加し、段0と同じ入力をCPU・全評価休止へ上書きした40 stepで旧checkpointの読込を先に確かめる。本評価の200ゲームや受入4の代わりにはしない。
- 最終名のsmoke2も3条件とも成功。旧4 checkpointのCPU読込はすべてexit0で、アプリログのLoadNetwork/learner_sizeも確認した。`inspect_run.py config --diff --effective-only`で4組の差分を`.scratch/prd104/checkpoint-preflight/*/config-diff.json`へ保存した。mergeの有効差44件では形状・前処理の差がなく、device、評価休止/追加、train lane数、学習をしないLearner設定、新しい既定葉などの予定差だけだった。
- `freeze_current.py`で最終Debug/Releaseのexe・DLL・テスト実行体111ファイルを固定した（`.scratch/prd104/freeze-current.log`、`current-runtime-manifest.json`）。残りのGPU前後比較・関連回帰・実験はこの固定実行体を使う。実験入力とRelease exeのhashは引き続きrecipes.jsonでも検査する。
- 追加INPUTなしの継続用に、このチャットの完了追跡`prd104-1`を30分間隔で設定した。変化のない待機は通知せず、完了・失敗・要対応だけを知らせる。実行済みdriverは二重起動しない。結果の記録・wiki・最終レビューまで終えたら追跡をPAUSEDにする。継続手順は`.scratch/prd104/continuation.md`。
- 第3回an-reviewは、追加したCPU読込・実行体固定資材を含む30ファイルと計画の両方がOK。原文: `.scratch/prd104/review-20261029-1006.md`。GPU前後比較・関連GPU回帰・段0本評価・段1の21本と集計は、実施後に証跡を再提出する。
- 2026-10-29 11:05 JST確認: ユーザーがAtariバッチを止め、既存Runが完了したためGPU検証へ進んだ。既存キューをこちらから操作してはいない。before_final2のAtari/DropMergeは226.0秒、after_final2は207.2秒でexit0。`compare-all.log`で既定6条件すべての全scalar点が完全一致した。
- 同じ固定Debug実行体の関連回帰`[prd104],[prd098],[prd103],[replay_buffer],[profiled_value]`は44.6秒、174 cases・21903 assertionsすべて成功（`.scratch/prd104/related-regressions.log`）。
- 段0本評価を開始。無効checkpointの200ゲームは358.5秒で完走し、平均34142.375点、突破率50.5%、平均長5599.865 step。合算checkpointのfull/single評価を実行中。まだTempoRLとのgや因果の結論は出さない。段1の21本は後続に待機。実行・結果の正本は各execution.json/result.jsonとexperiments.log。未完了を受入成功へ数えない。
- 11:35 JST確認: 合算checkpointのfull/single各200ゲームが1267.3秒で完走。平均点は29756.625/46495.0、突破率71.5%/97.0%、平均長4231.64/10093.145。無効との差を分母にしたgは3.8165365（分母4385.75）だが、残り2 checkpointの評価前なので全体の読みは未確定。`inspect_run.py metrics`と`trace-csv`で独立に抽出し、平均点一致と計600行（無効200＋合算full200＋single200）を確認した。出力は`.scratch/prd104/analysis/stage0/off-merge-*`。
- 合算評価後は別の評価Runが動いているため、残りの段0と段1はドライバーの通常の空き待ち。プロセスや他のバッチは操作せず、自動再開を待つ。失敗は発生していない。

## 段0での区切り（2026-10-29）

- 最新のユーザー指示は「段0が終わったら一旦区切る（中断してレビューや整理）」。4 checkpointすべての本評価が完了したため、段1の自動実行を中断し、完了追跡`prd104-1`も削除した。段1・段2は再開指示前に開始しない。
- 境界の切替中に`run_20261029-122543_t104_stage1_off_s1`が起動した。今回の連続実行Pythonだけを止め、当該Runも12:27:04 JSTにCloseMainWindowで終了した。最後の定期ログは142848 experience（予算2M）で、採用結果は0/21本。原artifactとinterruption.jsonを保持し、成績には数えない。次回は新しいRun名・別実行記録でやり直す。
- 段0のgは合算3.8165、遮断0.7662、重み0.3は0.9635。3本とも0.5以上で、PRDの点推定の判定は「固定が主因」。μ-only評価時の固定の損を示し、学習中のデータの偏り・補助学習だけの効果・seed差は未解決とする。
- [段0の実験記録](../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)へ全結果、評価条件、限界、出典を保存した。標準inspect_runの4 Runまとめ出力とtrace-csvから、7 session×各200件=1400件の平均点・突破率・長さを独立集計して一致を確認した。
- 元学習Runの設定3本と中断Runの設定1本を記録側へ複製。保存用コピーのみLFへ揃え、改行以外の全行が原本と一致することを確認した。無効checkpointの既存複製も原本と一致。元artifactは変更していない。
- `run_experiments.py run --through-stage 0`を追加し、段0の保存済み結果だけを再利用してexit0になること、段1の採用結果が0のままで新Runを起動しないことを確認した（`.scratch/prd104/stage0-finalize.log`）。本体C++と既定設定は第3回レビュー後から変更していない。
- 実験記録を正本としてtemporal_abstraction/qbertのwiki、目録、ingestのblob記録を更新した。既存の別作業のwiki追記は保持した。段1未完了はユーザー指定の今回範囲外であり、成功扱いにはしない。
- 上記の区切りと結果整理を同じan-reviewへ継続レビューする。
- 第4回レビューは実装40ファイルOK、計画「前提」に残った段1までの実行範囲1行だけ差し戻し（`.scratch/prd104/review-20261029-1312.md`）。前提を最新の段0区切り・指定split追加・段1再開指示待ちへ修正した。
- ユーザー指定のsplit checkpointを、代替候補としてではなく追加比較で評価した。`run_20261029-131020_t104_stage0_split`は451.4秒・exit0、同じRun内でfull/single各200。平均点26913.625/31358.875、突破率63.5%/64.0%、平均長2779.0/4050.11、g=0.6149403（分母7228.75）。学習時のビルド・構成が違うため元3本の多数決には混ぜない。
- 追加を含む5 Run・9 session・1800件のtraceを標準inspect_runで抽出し、平均点・突破率・長さを独立集計して一致した（`.scratch/prd104/stage0-split-final-audit.log`）。元4本のresults.jsonと元3本の判定は保持。追加sourceの学習configも改行以外を保持して記録側へ保存した。
- 既存の別枠split評価（μ-only・固定τ、full/singleが別Run）は27206.125/32040.0点。標準metrics出力で出典を確認し、今回の同一Run内の結果とは区別して実験記録に残した。別枠のartifactは変更していない。段1は引き続き停止。
- 第5回の独立an-reviewは、計画と追加splitを含む実装42ファイルの両方がOK。不足点・開発者確認事項なし。原文: `.scratch/prd104/review-20261029-1336.md`。今回の区切りで必要な実装・受入1〜4・段0・指定splitの追加評価・結果整理・wiki・自己点検とレビューを完了した。段1は採用0/21で保留し、段2も実行しない。再開指示前の自動実行・追跡は残していない。

## コミット後の指摘対応（2026-10-29）

- `MakePrd104ConfigData`を同じテストファイルの他の補助関数と同じ無名名前空間へ入れ、内部リンケージに揃えた。
- ステージ0の実験記録からGit管理外のローカル証跡パスを除き、評価Runは表のRunフォルダ名で特定する形にした。得点・g・突破率と結論は変更していない。
- `DecisionWindowRule`と`ActionChangeWindowRule`の行動比較をファイル内の`MakeReplayActionMatcher`へまとめた。CPU連続int64配列の直接比較と、それ以外のTensor比較を維持し、比較方法の選択と起点Tensorの取得は窓ごとに一度だけ行う。
- `docs/quality/impl.md`全項目で自己点検し、今回の変更に対する差分・参照・LF/BOMを確認した。Debugビルドと`[prd104],[replay_buffer]`の再実行は未実施。別枠の評価Run（`evaldiag_noise_tau`）が実行中のため、負荷を掛ける検証を保留した。以前の合格結果を今回の修正後の検証としては数えない。

## ステージ1の再開（2026-10-29）

- ユーザーが再開計画を明示承認した。既存の全Runner待機を適用せず、同じドライバーだけOSの排他ロックで二重起動を防ぐ。実行中の他Runは操作しない。
- 固定済みRelease実行体（SHA256 `2f74a21f48cf72349f79323c06b1e2bee1af3542943d20870905c101f82156e1`）、111ファイルのruntime manifest、固定設定、21条件のhashとseed/群順を照合した。直近の行動比較helper整理と評価進捗ログの変更は、この実行体には含まれない。
- 再開ドライバーは`.scratch/prd104/run_stage1_resume.py`、専用出力は`.scratch/prd104/stage1-resume-20261029/`、Run名は`run_{t}_t104r1_<群略称>_s<seed>`。旧中断Runと旧execution.jsonは保持し、無効群seed1を最初から採り直す。各Runは元の固定configを読み、2M予算の正常終了・終盤ログ・実効設定・有限の評価点を照合する。終了時点のexp_stepそのものがログに無い場合、最後の定期ログ値をそのまま記録し、正確な終端stepと混同しない。
- 成功済み・失敗済み結果は再利用し、失敗後も後続群を続ける。resultが無いのにexecution記録がある場合は上書きせず、再開前の確認を要求して止める。元ステージ0の個別結果・追加splitは変更せず、ステージ1完了時の集約へ結果を加える。
- 起動前検証: `run_stage1_resume.py --validate`は成功。Runを起動しない`test_stage1_resume.py`で排他ロック、21条件の順序、資源不足を模した先頭失敗後の継続、保存結果の再利用、ステージ0保持、ステージ2除外を確認して成功した。
- 15:11:47 JSTに先頭の無効群seed1を`run_20261029-151147_t104r1_off_s1`で開始し、15:12:39の定期ログで110848 experienceへの進行を確認した。21本は実行中であり、完了結果には数えない。
- ユーザーが30分間隔の追跡作成を明示承認したため、このチャットのheartbeat `prd104-1`をACTIVEで作成した。変化がない間は通知せず、分析・記録・最終レビューまで完了したらPAUSEDへ更新する。
- 完了後は標準`inspect_run.py`を入口にAUC・3 seed帯・EMA200・対照・k=1・split・教材kmaxを分析する。学習Runの実効configと日本語記録、wiki、計画結果を保存し、同じan-reviewへ最終レビューを依頼する。実時間・throughputはAtariとの競合があり得る条件として扱う。

## ステージ1の実験完了と結果整理（2026-10-29）

- 23:32 JST、固定実行体で7群×3seed×2Mの全21本が終了コード0、採用21/21となった。`stage1-resume-20261029/complete.json`はpassed=true、successful=21、stage2_executed=false。旧中断Run、ステージ0の個別結果、追加split結果を保持した。
- `inspect_run.py metrics`で全21本をまとめて読み、policy/target報酬・判断率・skip loss・抽選k・kmax・勾配normを抽出した。policy報酬は各312点、最後の定期記録は1,998,848 exp step。固定予算と終了コード0を到達根拠にし、正確な終端stepを記録したとは書かない。
- `.scratch/prd104/finalize_stage1_analysis.py`で標準抽出とMetricsマスタの独立計算を照合した。全期間のpolicy/skip平均、policyの0.5Mごとの区間平均・点数は一致。標準出力は`.scratch/prd104/analysis/stage1-final/inspect-metrics.json`、照合・集約は同`audit-summary.json`。
- AUCの3seed平均は無効20.87、通常TempoRL138.43、Single行動変化窓51.25、終了窓26.69、STOP57.25、k=1 16.55、終了窓split33.09。行動変化窓とSTOPは同seedの無効を3本とも改善したが、範囲は無効と重なる。通常TempoRLだけが無効をPRDの定義で「上回る」。
- PRD §7の主判定は、行動変化窓・終了窓の平均がともに無効の最大64.93以下なので「固定から来る」の行に入る。Singleの改善が存在しない、あるいは固定だけが因果的な原因と証明されたという意味にはしない。k=1の「上回る」は不成立。終了窓splitとmergeは同じ帯であり、clip寄与を検出しない行だが同等性の証明ではない。
- ユーザーの主関心であるSingle対無効を正面に置いた日本語記録を`docs/experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md`へ保存し、全表にRun識別子を併記した。学習Run21本の実効configを同envの`config/`へ複製し、LF以外の全行一致を確認した。
- 同記録にステージ0のgと既存記録を参照し、LLとQbertの証拠を混ぜず集約した。LunarLanderのSingle結果をQbertへ一般化せず、ステージ2は実行していない。AtariとのGPU同時実行があるため実時間・throughputは効果判定に使っていない。
- STOPの改善方向を説明する原因は未特定。乱数消費・追加target計算・追加評価sessionという「軌跡が変わる経路」を改善原因と同一視しない。ユーザー指定の別枠調査向けにOS一時フォルダへ引継ぎ済みで、この実験完了を原因調査完了として扱わない。
- wikiのLunarLander・temporal_abstraction・目録へ出典付きで取り込み、未解明のSTOP改善と補助勾配の寄与を問いとして残した。
- 最終自己点検: `docs/quality/impl.md`全72項目を今回の適用範囲で確認した。今回は実験の記録と逐語config複製であり、productionコード・現用設定・APIを変えないため、C++実装・新規契約移行・TDD・性能計測の追加は対象外。既存の別チャットのC++変更を今回の検証済み範囲へ含めない。
- 検査コマンド: `./.venv/Scripts/python.exe -X utf8 .scratch/prd104/finalize_stage1_analysis.py`、`./.venv/Scripts/python.exe -X utf8 .scratch/prd104/run_stage1_resume.py --validate`、`./.venv/Scripts/python.exe -X utf8 .scratch/prd104/check_stage1_writeback.py`。標準抽出と独立平均、111runtime・21入力hash、21configの内容一致、Single判断率と教材kmaxの全点条件、28ファイルのLF・既存BOM保持・リンク・wiki出典/目録/blob・秘密値走査、対象の`git diff --check`が成功した。公開値走査の一致は複製configの当時の`app.runs_dir`だけで、逐語コピーとして保持した。
- 検査証跡は`.scratch/prd104/analysis/stage1-final/writeback-check.json`。今回はコードを変えないため新たなDebugビルド・GPU回帰は行わず、過去のビルド結果を今回の別作業コードの検証に流用しない。未解明のSTOPの原因はユーザー指定の別枠調査へ残し、ステージ1の実行合否とは分ける。
- 同じ`an-review`（`/root/prd104_review`）による最終継続レビューは、計画と今回の実験記録・config複製・wikiの両方がOK。不足点・修正要求・範囲外検出・開発者への確認事項なし。原文を`.scratch/prd104/review-20261029-2347.md`へ保存し、読み戻しを確認した。レビューはSTOP原因の解明やステージ2を完了とするものではない。
- 今回承認されたステージ1の実行・集計・日本語記録・21config複製・wiki・自己点検・最終レビューを完了した。完了した追跡を残さない最新のheartbeat運用に従い、`prd104-1`を削除して終了した（再開時の予定ではPAUSEDだったが、削除の成功を確認）。Gitのステージング・commit・push、既定採用変更は行っていない。

## 決まらなかったこと（区切り時点）

PRD §9の実験結果待ちの事項は保持する。新たなユーザー判断は現時点でなし。

- レビューの範囲外検出: 150.en §2.2/6.1のγ引数、§2.4のschema K=2（現行3）、§7.7のRNG非消費にwindow長抽選の例外がない。200.en §2.3/2.4に旧uqe_use_tail_mean/uqe_tau_decayが残る。今回の追加行外であり、この実装の判定には使われていない。今回の実装範囲では修正しない。
