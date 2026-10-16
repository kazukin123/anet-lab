# 勾配ノイズスケールの検証資材

## 対象と受入

DefaultDQN の TD / QR / IQN。FP32 と CUDA BF16 の学習経路を変えず、更新自身の重み付き要素損失から15指標を測る。未購読は計測ゼロ。最初の版の11指標やallだけの先行完成は採用しない。

受入は、計算・成立条件・購読・決定論環境の非干渉、本線smoke、OFF/ON各3回の費用報告と文書同期である。B・RR・学習率を変える因果実験は含めない。既定はk=8、interval=503、EMA係数0.05。費用2%以下は目標で、超過を隠すために既定値を変更しない。

## 単体・公開経路

`anet-core-test` の `[gradnoise]` が次を検証する。

- 既知の小バッチ勾配からの推定式、全batch平均との整合、群の加法性。
- Adamの偏り補正とepsilon、単位・非一様前処理、使用しないparameterのstate不足と残存grad。
- 群ごとのstate不足、空群、部分的使用、負の信号、EMAの未成立・再成立。
- 非正のEMA信号で素/Adamのscale・ratioがNaNとなり、rawとAdam信号・分散は有限となる公開経路。全行正係数の対照も確認する。
- 合成線形回帰での解析値への収束。固定の専用Generatorを使い、学習用RNGを消費しない。
- 15 key、異なる購読間隔の和集合、各keyだけの公開、同時発火でも1回のEMA更新。
- 購読時のk/B/alpha/FP16/feature key検証、未購読NoCare、間隔混在WARNの起動時1回。
- TD/QR/IQN × 通常/Fused AdamW × CPU FP32/CUDA BF16 × Munchausen無効/online_reuse。NoisyNet・PER・SNを含め、parameter・gradient・buffer・optimizer step/moment・LearnerのRNG状態・既存数値指標を比較する。optimizerは公開Saveから復元して観測する。

CUDAが利用できない環境では当該条件をskipし、BF16の受入を未確認として残す。同じseedだけでは異なる環境・backendのbit一致を保証しない。本線の非決定論Run間にはbit一致を要求しない。

MSVCはAGENTS.mdのVsDevCmd経由で初期化し、Debug全体と対象テスト、既定スイートを実行する。Python資材はリポジトリの `.venv/Scripts/python.exe` を使う。`compare_metrics_test.py` と `validate_run_test.py` は小さな合成artifactで読取・除外・未成立率の規則を確認する。

## 変更前基準とCPU比較

コード変更前に `.scratch/prd092/before/` へ実行体・DLL・configを複製し、manifestを残す。2026-10-15の基準はDebug/Releaseの109ファイル、合計5,919,117,968 bytes。保存物をhard linkで更新中の実行体へ結び付けない。

保存したDebug Runnerとconfigから、LunarLander CPU・deterministic backend・seed 920・8 env・B64・容量4096・warmup128・FP32・QR8・終了8192経験の基準を採取した。NoisyとMunchausenを含む。評価は実在するeval/eval_targetのintervalを0にする。

`run_validation.py` の `cpu` ステージは保存済み `cpu-before.txt` を基に、変更後OFF/ONを同じ条件で実行する。既存指標はPRD084のcompare_metricsでtag・step・float.hex値と点数を厳密比較する。性能・資源と新しい49群だけを除き、既存47群も比較する。基準対OFF、OFF対ONを確認し、OFFは全既存キー・文字列値の設定比較も行う。追加キーとartifactの場所だけを許容する。

## 本線smokeと性能

`run_validation.py` は既存の `AnetRLRunner.exe` が起動していれば開始しない。既存Runを停止・終了しない。Releaseの実行体を書き換えるビルドも、その実行体が使用されていない時に行う。

本線の共通入力は `run_20261014-080249_dm_fb_munch_tau3e-2_ndtpen20_nzrep_c100_b512rr2_20m` の保存済みconfigとagent_close.anet。両側で同じcheckpoint・seed 920を使い、B256/RR1へそろえる。BF16・Noisy・architecture・replay容量・既存metricsを維持し、background evalだけ両側で無効化する。出力先は `.scratch/prd092/runs/`、正式な受入証拠なのでRun名にtmpを付けない。

- `smoke` ステージ: Debug、warmup25600経験の後に505更新以上。15定義・source key・intervalを検証し、Adam raw6指標と素のraw2指標の有限値を確認する。scale・ratioの常時有限/正は要求しない。
- `performance` ステージ: 同じReleaseでOFF→ONを各3回、交互に実行する。測定開始learn_step503、終了1509の1006更新を固定する。両側に同じprogress/clock行を足し、既存Runnerのelapse_hour（wall-clock、ms精度）差を秒に戻して比較する。GPUの非同期実行・同期を含むRunの実時間であり、profiler区間の合計は代用しない。process全体の実時間も別に保存する。
- CPUメモリはGetProcessMemoryInfoのPeakWorkingSetとPrivateUsage、GPUはdevice 0のmemory.usedを0.5秒間隔で採取する。同じ方法を全6回に使う。GPU値はdevice全体のサンプルピークで、短い瞬間の真のピークやallocator単体値とは区別する。
- 各回の設定・実行体hash・checkpoint hash・終了code・Run名・実時間・メモリを保存する。中央値、最小・最大、中央値増加率、メモリの中央値差分をperformance-summary.jsonへまとめる。

## 値の未成立率

非有限値はmetricsの記録経路で省かれる。`validate_run.py` は正常終了codeと、全更新に記録した専用progress行から実更新点を求め、各keyの初回発火・bucket-crossingに従う予定点と有限値を照合する。通常の未測定stepや別keyだけの発火点は分母に含めない。

15行の厳密なtag/source key、learn event、learn_step軸、update_result source、interval503をsmokeで照合する。空群は `--empty-group feature` または `readout` で予定点0・率null（未定義）とする。異常終了・ログ欠損を未成立と数えない。理由別の指標は増やさず、raw・EMA・測定条件と併せて読む。

## 初回受入の実行結果（2026-10-15）

15指標と全受入項目を実施した。CUDA BF16も実機で検証済み。検証環境はWindows x64、AMD Ryzen 7 9700X、NVIDIA GeForce RTX 5090（32607 MiB、driver 616.92）、Visual Studio 2022 Developer Command Prompt v17.14.42。英訳は通常の翻訳運用へ委ねる。コミット・pushは行っていない。

- MSVC初期化経由の全DebugビルドとRelease Runnerビルドが成功。対象 `[gradnoise]` は8 cases / 17,578 assertions成功。CPU FP32・CUDA BF16、通常/Fused AdamWを含む。
- 既定C++スイートは685 cases中684 passed・1 skipped、55,539 assertions成功。skipはCUDA利用不可時の検査で、このPCではCUDA利用可能のため。Python比較・未成立率テスト各1件も成功。
- 既定葉監査は25,595 assignments / 3,445 audited / errors=0。変更前と変更後OFFの既存設定値は差分なし（829→833 keys、追加はgrad_noiseの2キーと検証用progress/clockの2行）。
- LunarLander CPUの変更前・変更後OFF・15指標購読ONで、既存74 tags・35,951点が完全一致。既存Noisy指標を含む。正規化結果のSHA256は `c53a5f4cc820955b59eb912a2f8214f5f545d121162ac39b3a765b7e59364db0`。

### 本線smoke

`run_20261015-174306_prd092_smoke` はexit=0、実更新510回（learn_step 0〜509）。15行の実効定義・購読・source keyとraw 8指標の有限値を確認した。BF16・NoisyNet・B256/RR1、seed920、warmup25600経験、共通checkpointを使用。

### Releaseの時間・メモリ

各回で同じcheckpoint・実行体・設定を読み、503更新のwarmup後、learn_step503→1509の1,006更新を比較した。OFF→ONの順を3対繰り返した。実効configでも出力先識別子と49群の追加以外は全6回で一致した。他のビルド・テストを同時実行していない。

| 順序 | Run | 区間実時間 (s) | CPU PeakWorkingSet (MiB) | CPU PrivateUsage最大観測値 (MiB) | GPU deviceピーク観測値 (MiB) |
|---|---|---:|---:|---:|---:|
| 1 | `run_20261015-172808_prd092_perf_1_off` | 68.764 | 3403.23 | 25921.68 | 24781 |
| 2 | `run_20261015-173051_prd092_perf_1_on` | 69.106 | 3419.30 | 25750.52 | 24777 |
| 3 | `run_20261015-173317_prd092_perf_2_off` | 71.776 | 3420.91 | 25921.57 | 24791 |
| 4 | `run_20261015-173545_prd092_perf_2_on` | 69.424 | 3418.20 | 25760.06 | 24133 |
| 5 | `run_20261015-173813_prd092_perf_3_off` | 66.403 | 3397.46 | 25926.82 | 24816 |
| 6 | `run_20261015-174034_prd092_perf_3_on` | 68.345 | 3414.45 | 25783.79 | 24774 |

| 集計 | OFF | ON |
|---|---:|---:|
| 中央値 (s) | 68.764 | 69.106 |
| 最小 (s) | 66.403 | 68.345 |
| 最大 (s) | 71.776 | 69.424 |
| CPU PeakWorkingSet中央値 (MiB) | 3403.23 | 3418.20 |
| CPU PrivateUsage最大観測値の中央値 (MiB) | 25921.68 | 25760.06 |
| GPU deviceピーク観測値の中央値 (MiB) | 24791.00 | 24774.00 |

実時間の中央値増加率は **+0.50%**。2%は目標であり、既定k=8・interval503・alpha0.05は維持した。
ピークメモリの中央値差（ON−OFF）はCPU WorkingSet **+14.97 MiB**、CPU PrivateUsage **-161.62 MiB**、GPU **-17 MiB**。
メモリは起動から終了までを対象とする。CPU WorkingSetはOSのPeakWorkingSet値、PrivateUsageは0.1秒程度のポーリングでの最大観測値。GPUは0.5秒間隔のdevice全体memory.usedであり、allocator単体の真の瞬間ピークではない。負の差をメモリ節約効果とは解釈しない。初期化時の確保を含み、勾配計測単体の瞬間的追加量、サンプル間のピーク、他プロセスの寄与は分離できない。

Release実行体SHA256: `88c74eb94a467f8407b913a2c092f3da8ce88d733e92e841f49408325abd4e3b`。checkpoint SHA256: `105fa2bb50c429e6174a1245e8c2eeead70fc4b9e4f031f02f75b433c89d6bf5`。既存Release実行体は保存時と同じhashのまま保持し、検証用実行体は `.scratch/prd092/release/AnetRLRunner.exe` に分離した。CMakeの出力先一時設定は通常設定に復元済み。

### 値の未成立率

各欄は有限値数/予定点数（未成立率）。CPU ON・最終smoke・正常終了した性能ON 3回の全実更新区間をそれぞれ集計した。feature/readoutとも非空。性能ON列は各Runで予定点を求めてから合計した。通常の未測定stepは分母に入らない。

| tag | CPU ON | 本線smoke | 性能ON 3回合計 |
|---|---:|---:|---:|
| 01_scale_adam | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 02_scale_adam_feature | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 03_scale_adam_readout | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 05_ratio_adam | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 11_scale | 2/2 (0.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 21_signal_adam | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 22_variance_adam | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 31_signal_adam_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 32_variance_adam_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 33_signal_raw | 2/2 (0.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 34_variance_raw | 2/2 (0.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 35_signal_adam_feature_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 36_variance_adam_feature_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 37_signal_adam_readout_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |
| 38_variance_adam_readout_raw | 1/2 (50.0%) | 2/2 (0.0%) | 12/12 (0.0%) |

CPU ONの初回Adam未成立はoptimizer state形成前の測定であり、前回値を埋めない契約に従う。本線は成熟checkpointのAdam stateから開始した。raw・scaleの有限値の有無から原因を一意に決めることはできず、この短い検証は長期学習中の成立率を保証しない。

計測成立と決定論環境での非干渉は支持された。B/RR/学習率変更による成績への因果は、この検証では判定不能である。長期Run・分岐実験は今回の受入に含めていない。

### 証拠と再現

受入ログ・集計は `.scratch/prd092/` に保存した。`09-final-debug-build.log`、`09-final-contract.log`、`12-default-suite.log`、`default-leaves-final.log`、`cpu-comparison.json`、`resolved-settings.json`、`cpu-on-summary.json`、`smoke-summary.json`、`performance-input.json`、`performance-runs.json`、`performance-summary.json`、`performance-effective-settings.json` が最終証拠となる。各Runの全process実時間・実行引数・config hash・終了code・メモリは個別execution.jsonに記録した。
受入Run 10本（CPU 3本、性能6本、smoke 1本）と入力checkpoint元FB Run 1本の実効configは、検証fixtureとして本ディレクトリの [`config/`](config/) に逐語保存し、`.scratch/prd092/config-copies.json` にhashを記録した。実験記録本文に基づく複製ではないため、`docs/experiments/` には置かない。認証情報・個人パス・メール・URL・IPが含まれないことも確認した。
最初のCPU ON試行はprofileの上書きにより15指標が購読されていなかったため受入から除外し、最終profileで15定義を必須照合して再実行した。初回smokeは実更新503回だったため、512回相当の予算と実更新505回以上のチェックを加えて再実行した。過去のRunは保存した。

実行した主要コマンド（リポジトリルートから）:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[gradnoise]"
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd092/compare_metrics_test.py; .\.venv\Scripts\python.exe core/anet-core/testdata/prd092/validate_run_test.py
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd092/run_validation.py smoke
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd092/run_validation.py performance --release-executable C:/dev/anet-lab/.scratch/prd092/release/AnetRLRunner.exe
```

CPUの再比較は保存した変更前実行体・configが必要で、`run_validation.py cpu` が変更後OFF/ONと厳密比較を実施する。Releaseは本検証で分離出力した実行体を指定した。構成から再構築する場合も同じRelease実行体を全6回に使う。

## レビュー修正後の再検証（2026-10-15）

重みノルムの既存の加算順をHEADへ戻し、上記の負信号の公開経路を追加検証した。Debug全体を再ビルドして成功した。

- [gradnoise]: All tests passed (17672 assertions in 9 test cases)。
- [plasticity]: All tests passed (292 assertions in 29 test cases)。
- 既定スイート: test cases:   686 |   685 passed | 1 skipped / assertions: 55633 | 55633 passed。

ログと実行体hashは `.scratch/prd092/21-review-tests.json`、修正5件と変更ファイルの一覧は [実装メモ](../../../../docs/memo/092_gradient_noise_scale_20impl.md) の「レビュー修正（2026-10-15）」に記録した。初回受入のsmoke・性能結果は当時の実行体の記録として保持する。
