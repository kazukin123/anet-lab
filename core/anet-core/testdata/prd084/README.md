# PRD 084 の検証手順

## 実行設定の責任境界

対象は CreateForwardExecutor と NetworkForwardExecutor::Forward。既存 Noisy テストと [nn][noisynet][prepared] で、準備時だけの設定解釈、具象 Module／Head の責任境界、設定の寿命・変更・同名設定、Network の寿命、独立Stateとtarget共有、probeの専用sample実行、同時 forward、出力・勾配・保持・精度・診断非干渉を確認する。通常の部品は本体を直接呼び、設定検索・再解釈・設定用ロックを使わない。既定 Network::Forward(input) は μ-only、TensorDictFunction の型は維持する。保存用設定の全キー・文字列値は変更しない。

リポジトリルートの PowerShell から Debug 全体をビルドする。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

責任境界のテストを実行する。

```bash
core/anet-core/bin/Debug/anet-core-test.exe "[noisynet][context]"
```

関連する NN・DQN・Head・Munchausen のテストと全 core テストも確認する。

```bash
core/anet-core/bin/Debug/anet-core-test.exe "[nn],[dqn],[head],[munchausen]"
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe
```

レビューでは、具体的な設定型が具象クラス内部に閉じていること、共通層に Module 固有のフィールド・読み取り処理・型や有無の判定が無いことを確認する。`GetDefault()` は各計算部品の既定動作を明示的に要求する。具体的な設定値は持たず、Noisy の計算部品はこれを μ-only と解釈する。名前付き設定の項目不足とは区別し、暗黙の fallback にはしない。

Noisy 演算・状態・optimizer・保存読込・診断は `[noisynet]` で検証する。現行の公開面整理の受け入れは `docs/memo/084_noisynet_24impl.md` を正本とする。23implは当時の記録として保持する。21impl・22impl の検証は当時の契約の記録として保持する。

## OFF 回帰の数値比較

初回実装の基準は `.scratch/prd084/old/`、当時の変更後は `new/` に保持する。今回のレビュー修正はコード編集前に現実装の実行体・依存 DLL・設定・seed・checkpoint を `.scratch/prd084/review-fix/before/` に保存し、新しい出力を `after-verified/` に分離する。旧 artifact は上書きしない。Atari 既定構成、指定の SN 付きチェーン、CPU LunarLander の3本を、同じ deterministic backend・seed・終了予算500kで実行する。学習スケジュールを変更せず、Atari の `greedy_dist` を双方で無効化する。

`compare_metrics.py normalize` は全数値 scalar を `(tag, step, float.hex())` に正規化し、時刻以外の数値差を保持する。性能・資源・追加 Noisy の各 metric group を除外する。`compare` は値とタグごとの点数を比較し、欠損・点数差・値の差を exit=1 にする。

採取済み Atari 既定構成の比較は、リポジトリルートから次で再確認できる。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd084/compare_metrics.py compare .scratch/prd084/old/atari_default.metrics.jsonl .scratch/prd084/review-fix/after-verified/atari_default.metrics.jsonl
```

この比較資材は3構成すべてに適用する。Run の正常終了と旧 OFF checkpoint の読み込みは数値比較とは別に確認する。

## 演算・状態・保存・診断

`[noisynet]` は参照出力と入力/μ/σ勾配、bias 有無、σ0=0、符号付きσ、rank2/3、batch/sample、SN の μ 正規化、force_fp32、count/episode の保持、同一時計の idempotence、結合行対応、6 Head の clone/hard/soft/save/load、通常/Fused AdamWのσ group・独立decay・μ-only凍結・moment保存、σ込みclipを確認する。optimizerの3テストは実際のTDLearnerのUpdateFromBatchを通し、公開の更新結果・parameter・gradientを観測する。復元はNetworkModelとLearnerのSave/Loadを使い、内部のSetupOptimizer/Optimizeをテストから直接呼ばない。

診断は既知 Q の三指標、feature/readout の8統計、対応 μ のみの分母、SN 実効 μ、分母ゼロ、購読gating、不足時NaN、TD/QR/IQN×Dueling×target方式の24構成でON/OFFのparameter/gradient/SN buffer一致を確認する。独自の実行設定を読むHeadで、probeのμ-only側にも同じ設定が届くことをσ0=0で確認する。

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
```

## Atari 結合の構成と判定

効率優先チェーンは `v5_iqn_impala_x2 > a5 > a5_apex > va_base > hard500 > rr4 > munch > rfit > a5_metrics > btrnet > btrsn12 > cap2m > eval2ch`。その末尾へ `run.@nz_learning` または `run.@nz_reference` を追加し、Breakout / seed=1 / 128 env / deterministic backend / 終了予算500kを固定する。学習スケジュールは維持する。

`nz_learning` は既存初期化とAMPを継承し、Actor/current/targetはcall/batch。`nz_reference` は4射影のdefault初期化・σ0=0.5、Body2FCのforce_fp32=true、AMP継承、標準call/batch。結合RunのActorだけを専用項目でepisode/sampleへ上書きする。`nz_reference_off` は同じ初期化・精度・保持設定でNoisy enabledだけを外す。

`greedy_dist` はCLIで `run.eval_schedule.[greedy_dist].interval=0` を固定する。ファイルの同名葉は、後から適用される `eval2ch` profile に上書きされるため、解決済み `run.txt` と `definition-only` ログを確認する。終了予算もCLIで指定し、per_beta/epsilon/warmupのスケジュールを縮めない。

両ON Runは全11行を購読する。Atariのfeature依存閉包にはNoisy層がなく、feature4統計は正常なNaN。`ScalarMetricObserver`は既知NaNを出力しないため、11購読宣言とfeature成立値の不在を確認し、NaNそのものは契約テストで確認する。readout4統計とprobe3指標は複数回の有限値を要求する。xavierの初期μ biasが0の場合、最初のbias ratioのみNaNが正しい。

`verify_integration.py CASE RUN_DIR` は11購読宣言・readout/probe成立値・複数evalを検証する。case名は `nz_learning` または `nz_reference`。終了code=0、checkpoint保存完了、実行体・設定・seedの保存は別に確認する。不足時は条件を維持して終了予算を延長し、閾値を下げない。

実行と集計の予算契約は、OFF3構成が各500k、旧OFF checkpointの読込が256、結合2本が各500k以上である。run_review_validation.ps1 の ExperienceBudget は結合2本にだけ適用し、OFF3構成の比較予算を変えない。延長時は新しい OutputRoot を指定して旧Runを保持する。summarize_review_validation.py も同じ条件で受け付ける。

予算・終了code・seedの入力契約は次で検証する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd084/summarize_review_validation_test.py
```

## 設定の監査

既定葉はPRD072の `check_default_leaves.py` で監査する。`compare_resolved_settings.py OLD_CONFIG NEW_CONFIG` は既存の全キーと文字列値を厳密比較する。追加キーは一覧化し、artifactの `app.run_name` / `app.runs_dir` の2キーだけを場所の違いとして除く。旧Runの設定を上書きしない。

設定記法の前後比較には、隠しテスト `[prd084-config-equivalence]` を使う。`ANET_PRD084_CONFIG_BEFORE` に編集前の設定フォルダ、`ANET_PRD084_CONFIG_OUTPUT` に新しい採取先を指定する。Atari既定、効率優先、nz_learning、nz_reference、nz_reference_off、CPU LunarLanderの6構成について、公開ConfigManagerで全キー・文字列値・型付き設定を比較する。設定採取のみで学習資源は生成しない。今回の採取先は `.scratch/prd084/review-followup-20261005/config-final/`。

## 性能測定

`run_performance.ps1` は参照profileからOFF、batch/sample、call/count32/episode、Noisy Body Linearのnone/spectral/spectral_cap、同じAMP下でBody force_fp32 ON/OFF、Noisy診断ON/OFFを1軸ずつ変える。学習比較profileでも3 SN modeごとにNoisy ON/OFFを測る。12組を各2反復のA/B/A/Bで実行する。

| 比較 | A | B |
|---|---|---|
| off | 参照 Noisy ON | 同じ初期化・精度の Noisy OFF |
| actor_share | Actor call/batch | Actor call/sample |
| learner_share | Learner current/target call/batch | Learner current/target call/sample |
| hold_episode | Actor call/sample | Actor episode/sample |
| hold_count32 | Actor call/sample | Actor count=32/sample |
| sn_spectral | Noisy Body SN none | Noisy Body SN spectral |
| sn_cap | Noisy Body SN none | Noisy Body SN spectral_cap |
| precision | AMP継承・Body force_fp32 ON | 同じAMP・Body force_fp32 OFF |
| diagnostics | Noisy診断 ON | Noisy診断 OFF |
| learning_sn_none | 学習profile・SN none・Noisy OFF | 同じ構成の Noisy ON |
| learning_sn_spectral | 学習profile・SN spectral・Noisy OFF | 同じ構成の Noisy ON |
| learning_sn_spectral_cap | 学習profile・SN spectral_cap・Noisy OFF | 同じ構成の Noisy ON |

Actor の比較は専用 perf_actor、Learner の比較は専用 perf_learner を使い、他の用途へ上書きを波及させない。Learner share 軸は current/target の共有範囲を揃え、Munchausen target の結合行経路を通す。precision 軸は Body の force_fp32 だけを変更する。

共通の終了予算は224k経験。200k warmupを含む既存スケジュールを維持し、評価を全構成で無効化する。全Run時間・経験/s、warmupをまたぐ区間を除いた学習中throughputのメトリクス平均、プロセスCPU private bytesの500ms間隔サンプル最大/OS peak working-set、GPUデバイス全体の5秒間隔peakと起動前との差を記録する。GPUはallocator固有の値ではなく、他GPU workloadを同時実行しない。時間/throughput/memoryの差を記録し、速度やスコア改善を合否条件にしない。

比較軸だけを一覧表示する。

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File core/anet-core/testdata/prd084/run_performance.ps1 -ListOnly
```

実測は同じscriptの `-ListOnly` を付けない実行。出力先は新しい時刻付きdirectoryを自動生成し、既存出力先を上書きしない。`summarize_performance.py` は全48Runの正常終了・実効条件、全設定キーの比較軸以外の一致、同一実行体・反復ごとの同一軸を検証し、比較を `summary.json` / `summary.md` へまとめる。

## 実測結果

修正前の結果は [verification.md](verification.md) に保持する。参照profileは全体FP32・episode/sampleであり、CUDA AMP内の選択的FP32とLearner sampleの性能は未検証だった。現行計画の追加検証は [review_verification.md](review_verification.md) に記録する。

## レビュー修正の確認事項

- 具象 LinearProjectionModule が Body と6 Head の設定・検証・保持・抽選を一元化する。parameter 登録とμ初期化順、SN buffer、εのslot識別は既存のまま。
- 不変な行対応はCPUの宣言で範囲を確認し、一度だけGPUへ渡す。検査用Tensorは複製を返し、射影はSelectRowsを使う。
- 準備済み処理は同名・異内容、設定寿命、同時Actor、独自Module/Headの拡張を検証する。
- CUDA autocast内のBody force_fp32、Head/SN保護と、行対応の転送・scalar抽出が各射影に無いことをTorch profilerで確認する。
- NNのRL語は今回の修正では変更せず、後続の整理事項として保持する。
- run_review_validation.ps1 はOFF3本、Atari結合2本、旧checkpoint読込を専用の新しい出力先へ保存する。終了後、compare_metrics.py、compare_resolved_settings.py、verify_integration.pyで照合する。

レビュー修正の6本を一括で再判定する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd084/summarize_review_validation.py .scratch/prd084/review-fix/after-verified
```

summarize_review_validation.py は正常終了・構成別予算（OFF3本は500k、結合2本は500k以上、旧checkpointは256）・seed・同一実行体・依存DLL保存・close checkpoint、OFF3本の全数値点と全既存設定値、結合2本の精度・購読・eval/probe、旧CPU checkpointのSHAとモデル／Learner読込ログを確認し、acceptance-summary.jsonに保存する。

run_review_validation.ps1 の -Resume は、明示した出力先の固定実行体・設定を保ち、正常終了済みのcaseを再利用する。失敗recordとstdout/stderrはfailed-01へ分離し、元Runは保持する。未完了recordまたは現在のビルドとの実行体不一致では再開しない。greedy_distの無効化はAtari構成だけに適用する。

## レビュー修正の最終結果

review_verification.mdに受け入れ結果を記録。review-fix/after-verified/はOFF3・結合2・旧checkpointの6caseすべてpass、review-fix/performance/は12比較48 Runすべてexit0。全解決済み設定の比較軸以外の一致、同一実行体、warmupと学習スケジュール、A/B/A/Bの順序も照合済み。過去のverification.mdの全体FP32測定は当時の条件の記録として保持する。

## 時計の種類と残件の追加修正（22impl）

承認済み計画は docs/memo/084_noisynet_22impl.md。時計はkNoneに加えてkTransientRows/kContinuingRowsの2種類へ統合した。episode_startとepisode保持は維持する。任意のB3は今回は入れない。

上記48本の性能結果は改名前の実行体（F7EB69FD…）での測定。offとhold_episodeのみ新実行体で3反復A/B交互に取り直した。各Run値と中央値は以下を参照。他10組、旧Runとsummary.jsonは当時の測定として保持する。同じ参照構成・Noisyありの14本でも学習中exp/sは1,005〜2,081だったため、数%の差はこの測定では読めない。

### 22implの検証結果

Noisy47件/10497 assertions、全core664passed/1skipped/34680 assertions、Debug/Releaseはexit0。最終実行体SHA-256は 3089EA77E5B79EF027365C8CDB6497BF0C55A2AA10825D818336B083F1612F52。OFF3構成のchecksum/点数/既存設定値は完全一致（review_verification.mdに全文）。NN指定6ファイルのRL語0件、全編集trackedファイルのHEAD BOM/LF一致を確認。性能2組12 Runは全exit0、比較軸以外の設定一致、各Run前後の別学習プロセス0。保留無し、B3は挙動不変の改名検証に限定するため未実施。

## Run ごとの値

| 比較 | 反復 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |
|---|---:|---|---:|---:|---:|---:|
| off | 1 | a | 73.279 | 3675.9 | 23189.77 | 5881.00 |
| off | 1 | b | 71.000 | 2544.2 | 23157.06 | 5668.00 |
| off | 2 | a | 75.239 | 1494.8 | 23218.39 | 5707.00 |
| off | 2 | b | 63.672 | 2406.8 | 23167.88 | 5578.00 |
| off | 3 | a | 59.240 | 3617.7 | 23224.84 | 5726.00 |
| off | 3 | b | 57.086 | 3818.9 | 23135.75 | 5451.00 |
| hold_episode | 1 | a | 61.263 | 3629.1 | 23234.97 | 5637.00 |
| hold_episode | 1 | b | 58.438 | 3734.4 | 23168.38 | 5635.00 |
| hold_episode | 2 | a | 60.328 | 3492.8 | 23172.52 | 5499.00 |
| hold_episode | 2 | b | 59.492 | 3519.9 | 23163.08 | 5635.00 |
| hold_episode | 3 | a | 59.049 | 3632.6 | 23170.12 | 5637.00 |
| hold_episode | 3 | b | 59.266 | 3524.1 | 23172.19 | 5673.00 |

## 3反復の中央値

| 比較 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |
|---|---|---:|---:|---:|---:|
| hold_episode | a | 60.328 | 3629.1 | 23172.52 | 5637.00 |
| hold_episode | b | 59.266 | 3524.1 | 23168.38 | 5635.00 |
| off | a | 73.279 | 3617.7 | 23218.39 | 5726.00 |
| off | b | 63.672 | 2544.2 | 23157.06 | 5578.00 |

## Forward 事前準備の受け入れ

計画: docs/memo/084_noisynet_23impl.md。基準は .scratch/prd084/forward-refactor/before/、変更後の出力は分離する。最終 Release の OFF 3 構成を各500k経験で旧 checksum と照合する。代表 ON は Actor episode/sample、Learner current/target call/sample、Munchausen target、AMP 継承・Body FP32、Breakout seed=1・128 env・全診断・greedy_dist 無効、各500k経験を変更前／後3反復 A/Bで比較する。Noisy 指標も数値比較に含め、warmup200k以降の性能と中央値を保存する。同じ GPU に別の学習 Run があれば測定だけ保留する。過去48 Runは全面再測定しない。

## 事前準備リファクタリングの検証資材

`run_forward_refactor.ps1` の `-Stage off` は最終 Release の OFF3構成と旧 checkpoint 読込、`-Stage on` は保存した変更前／後の実行体による代表 ON の3反復 A/B交互を実行する。既定出力先は `.scratch/prd084/forward-refactor/after/` で、`-OutputRoot` で今回専用の出力先を指定できる。既存記録を上書きせず、`-Resume` は正常終了・同じ実行体・同じ予算の記録だけ再利用する。測定の前後で他の Runner を確認し、検出時は既存プロセスを停止せず測定を保留する。

ON の数値正規化は `compare_metrics.py normalize` の `--include-noisy` を指定する。OFF の既定除外条件は維持する。最終集計は正常終了・OFF checksum・旧 checkpoint の読込、ON の設定・11診断の購読と成立値／NaN・複数 eval／probe、3反復の順序・同一 GPU・全6 Runの数値一致を検証する。warmup200k到達後にメモリを5秒間隔で採取し、区間時間・throughput と各中央値を保存する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd084/summarize_forward_refactor.py .scratch/prd084/forward-refactor/after
```

## Forward 事前準備の最終検証（2026-10-10）

[事前準備リファクタリングの検証記録](forward_refactor_verification.md) に、最終ビルド・テスト・OFF3構成の checksum、代表 ON 6 Run の値・中央値・成立条件と変更ファイルを保存した。過去の受け入れ・測定は当時の契約の記録として保持する。

## 事前準備の再レビュー検証（2026-10-10）

Learner の forward_resources は必須で、有効時は current／target の参照キーとの対応を準備前に検証する。空キーでは既定 Resource を明示する。準備済み Head の子管理は NN 共通部品で行い、独自設定の解釈は子の具象 Module 内部に残す。全子が既定動作なら Head 本体を直接呼ぶ。

今回の選択は OFF3本＋ON1本（各500k経験）と旧 OFF checkpoint 読込。`-Stage on-after` は変更後 ON 単発を保存済み代表構成の109941点と照合する。出力先は `.scratch/prd084/forward-refactor/review-followup-20261010/after/`。集計は `summarize_forward_refactor.py` に `--followup` を付け、数値・診断・設定・同一GPUを検証する。以前の性能6 Runは当時の実行体による証拠として区別し、3反復・48 Runは再測定しない。今回の結果は[検証記録](forward_refactor_verification.md)の再レビュー節へ追記する。

2026-10-10追記: ユーザーによる既存Run一時停止後、最終Release別出力実行体でOFF3本＋ON1本・旧checkpoint確認を実行した。全正常終了・点数／checksum一致。全解決値、診断の成立とNaN、複数eval／probeも確認済み。専用出力のafter/acceptance-summary.jsonと検証記録の再レビュー節を参照する。停止中の既存Runがメモリに常駐した状態のため、単発の性能値は前後性能比較の証拠にはしない。

## 公開面と既定経路の整理（2026-10-10）

State/Contextは内部型、CallInfoは不変クラスとWithRowMappingを維持する。CreateDefaultForwardExecutorはModule/Headを準備せず、名前付きCreateForwardExecutorは既定設定を拒否する。BindNetworkは同じ設定・State・時計種別を継承する。Executorの部分Forwardは削除し、Networkの既定部分Forwardは維持する。カタログの参照存在は読込時、使用する内容は具象準備時に検証する。Noisy probeの平均側は既定Forwardを使う。

[simplify_default]・[simplify_catalog]・[forward_context]・[prepared]と既存[noisynet]を確認する。全coreはCPU/CUDA、設定6構成は公開ConfigManagerの全解決値比較、既定葉はprd072資材で監査する。削除したAPIの残存とNN6ファイルのRL語を検索し、NetworkForwardClock以外のkNoneは対象にしない。

今回の出力は.scratch/prd084/forward-refactor/simplify-20261010に分離する。最終ReleaseはCPU LunarLander OFF500k（88184点）と代表ON500k（Noisy込み109941点）を旧基準と比較する。-Cases lunarlander_cpuでOFFを絞り、指定しない場合は従来3構成を維持する。CPU OFFはGPU検査・計測に依存させない。同GPUの別学習RunがあればONのみ保留する。旧OFF checkpoint読込も確認する。性能再測定はしない。結果は[検証記録](forward_refactor_verification.md)の今回の節を参照する。

## 公開面と命名 2026-10-11

公開のnn.hppは利用側のConfig・CallInfo・Network・NetworkForwardExecutor・診断・Builderを定義する。Module／Head作者のinterface、HeadFactory、HeadGraphVizInfo、WeightNormConfigは内部のnn_impl.hppへ置く。Module／HeadのCreateExecutorがnullptrを返す場合は本体を直接呼ぶ。

現用APIはCreateForwardExecutor／CreateDefaultForwardExecutor、NetworkModuleExecutor／NetworkHeadExecutor、PreparedHeadExecutor。既存タグ・数値と準備済み処理の契約は維持する。API残存、公開型／メンバ／列挙子コメント、BOM／LFの監査ログは.scratch/prd084/forward-refactor/api-20261011/logs/へ保存する。最終ReleaseのCPU OFF500kと旧checkpoint読込を同じ実行体で確認し、結果は[検証記録](forward_refactor_verification.md)の今回節へ記録する。ON Runと性能の再測定は行わない。
