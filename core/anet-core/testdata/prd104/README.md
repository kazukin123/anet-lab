# PRD104 検証資材

正本は [PRD104](../../../../docs/memo/104_temporl_skip_auxiliary_only_10prd.md)、実施記録は [実装メモ](../../../../docs/memo/104_temporl_skip_auxiliary_only_20impl.md)。実行はrepo rootのPowerShell、Pythonはrepoの `.venv` を使う。

## 入力と既定の等価性

`capture_baseline.py` は変更前のDebug実行体・DLL・テスト実行体・設定・公開resolverの解決結果を `.scratch/prd104/` へ保存する。既存snapshotは上書きしない。`baseline-manifest.json` に基準HEADと全ファイルのSHA256を記録する。

`run_validation.py before/after/compare --label ...` はPRD103と同じ短縮6構成を実行する。GridMaze/LunarLanderのOFF/ON、Atari qbert BTR OFF、DropMerge nzrep OFF。8 lane、batch16、warmup256、capacity4096、RR1、seed1、8704 experience、FP32、決定論backend、評価停止。両実行体には着手前の同じ入力を渡す。比較から外すのは90_perf/・91_resource/・92_resource/だけで、全scalarのtag・step・値を完全一致比較する。

`verify_config.py` は4 env×default/skipの8入力を公開ConfigManagerで解決し、全キー・文字列値を比較する。新しい既定葉（full・decision）の追加だけを許す。既定葉の検査は `core/anet-core/testdata/prd072/check_default_leaves.py` を使い、既存の指摘と今回の増分を分ける。

## 公開経路とsmoke

`[prd104]` はConfig、Actor/MakeAction、Learner/UpdateFromBatch、ReplayBuffer/Push/Sample、GetScalarを通して検証する。既存の `[prd098]`・`[prd103]`・`[replay_buffer]`・`[profiled_value]` を関連回帰とする。

`run_current.py --label ...` はLunarLanderの3短縮Run（action_change、episode_end、variable）を実行し、metricsの全点を検査する。前者は判断率1・k_maxが1以上で少なくとも1点は1超・有限のloss、後者はk_maxがすべて3.5以上。variableはlinear4→1・steps4352で、07_max_lengthの先頭3・末尾1・非増加・3/2/1の存在を確認する。証跡は `.scratch/prd104/smoke/`。

## 実験

`arms.txt` の `run.@skip_aux` をenvの `run.@skip` の後へ重ねる。終わりだけは target_window=episode_end、対照はfeature_gradient=stop、k=1だけはsingle+decision、終わりだけ・分割はepisode_end+splitの葉で指定する。metricsの連鎖は書き換えない。

`run_experiments.py prepare` は現行設定とRelease実行体を固定し、段0の4入力と読込不能時の条件付きfallback、段1の21入力、段2の4条件付き入力を生成・解決する。`recipes.json` にhashと条件を残す。`run` は既存runnerが終了するのを待ち、段0・段1だけを順次実行する。各Runの引数、実行体hash、入力hash、終了コード、時間、実際のRun名を execution.json に記録する。独立Runの失敗は記録し後続を続ける。元checkpointと過去Runは変更しない。

段0は各checkpointにつき同一processでfull/single各200ゲーム（無効はfullのみ）。段1は7腕×seed1〜3×2M、同じReleaseビルド・設定・256 lane・J=4。AUCは21_eval/02_policy_rewardのRun全体の有限点の平均とし、PRDの「以上・上回る・同じ帯」で判定する。summary.jsonへg、AUC、比較を保存する。段2は実行しない。

段0は現在のAtari連鎖へ元CLI26葉を重ね、元の`run.$`が選んだ形状・前処理も5葉（main_feature/tau_embeddingのstructure、HeadFC512とhead.projectionのNoisy有効、resize_mode）に展開する。元dumpと解決値の一致を検査し、source.jsonに残す。

`preflight_checkpoints.py` は段0入力のdeviceをCPU、全評価intervalを0へ上書きした40 stepのRunで、4 checkpointの読込を先に確かめる。学習と保存は無効のまま。200ゲームの本評価の代わりにはしない。

`freeze_current.py` は最終ビルド後のDebug/Releaseのexe・DLLとテスト実行体をcurrent-debug-bin/current-release-binへ保存し、全hashをcurrent-runtime-manifest.jsonへ記録する。GPUの空き待ち中にrepo側が再ビルドされても実行体を変えない。既存snapshotは上書きしない。実験driverはこのmanifestも検査する。

2026-10-29の追加指示により、現在は段0完了で区切っている。`run --through-stage 0` は段0だけを対象にし、段1を開始しない。既定の`run`は段1まで進むため、再開指示前には使わない。境界の切替中に起動した段1無効seed1のRunは中断・不採用として保存した。再開時はそのRun名とexecution.jsonを再利用せず、固有名と別の実行記録を用意する。

`run_stage0_split.py` は追加指定されたPRD103 split checkpointを、同じ1Run内のfull/single各200ゲームで評価する。元の代替候補入力を再利用するが、結果はstage0_splitへ別保存し、元3本の多数決にも段1にも進めない。既存の出力先は上書きしない。

長時間処理の開始・終了と結果はログへ保存する。停止疑いはログ・metricsの更新とCPU時間を合わせて確認し、経過時間だけでRunを停止しない。環境障害は結果へ記録し、欠けた結果を成功扱いしない。
