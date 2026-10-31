# PRD098 TempoRL 検証

現在の設定修正の正本は [22impl](../../../../docs/memo/098_temporl_skip_policy_22impl.md)。全体の実装記録は [20impl](../../../../docs/memo/098_temporl_skip_policy_20impl.md)、round2出力は `.scratch/prd098/config-20261024-r2/` に置く。

## 単体テストとビルド

MSVC を初期化して Debug 全体をビルドする。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[prd098]"
```

公開 ReplayBuffer の窓・境界・一様抽選、Actor の継続とリセット、TD/QR/IQN の Head と学習、Munchausen の方式、BF16、Noisy、包含型、stop/merge、q_shared、数値 target、checkpoint、GetScalar を検証する。専用 IQN 方策の query、Network ごとの clip と target 同期、PER 完成と sampleable の分離、CPU/CUDA prefetch・probe も含む。`[replay_buffer],[transfer]` と `[dqn],[munchausen],[nn],[observers]` が関連回帰である。

## 段0・smoke

変更前の Debug Runner と DLL を `baseline-bin/`、設定一式を `baseline-config/` に保存する。現在の変更前 snapshot は今回の実装に着手する前の全 Debug ビルドから取得したもので、既存の未コミット変更も含む。`resolved-before/*.json` は既存の `[prd093-config-snapshot]` テストで全キー・文字列値を採取したもの。

`run_validation.py before` は保存したバイナリを、`after` は現在の Debug バイナリを実行する。`--env` で環境を絞れる。再実行時は `--label` に新しい名前を与える。生成run.txtとinclude元configルート配下の全txtの内容・相対名も照合し、同じバイナリ・同じ入力の成功Runだけを再利用する。未参照のtxt変更も安全側に再利用を拒否し、異なるバイナリや失敗した記録へ上書きしない。

4 環境とも seed=1、8 lane、batch=16、warmup=256、replay ratio=1、8,704 experience の短縮条件。primary の NN 構造は保持し、Atari は qbert、DropMerge は nzrep の Noisy 設定を現行キーで重ねる。LunarLander と GridMaze は CPU、Atari と DropMerge は CUDA。評価は停止し、FP32・決定論的 backend で比較する。

比較は学習指標の全 scalar 点を tag・step・浮動小数点値でソートして完全一致を調べる。Noisy の指標も含め、除外は `90_perf/`・`91_resource/`・`92_resource/` の時間・資源だけ。点数・tag 数・除外一覧・checksum は各 Run の `normalized.summary.json` に残す。

smokeはenvの `run.@skip` だけを選び、変種はRunnerのCLI引数で渡す。9種類の必要tagはdefsと実際の有限値の両方で確認する。decision/lengthのTrace列は各envのrun profileが宣言し、Q列はCLIかenvで任意に追加する。

## 短い性能比較

2026-10-31 時点の注記: 下の `performance` は、評価を止める session の一覧を 2026-10-24 の解決済み設定から取るため、`[greedy_dist]`（PRD103 で `[greedy]` へ改名）を止めに行き、今の設定では使えない。受入 4 は [PRD103 の検証資材](../prd103/README.md) の `performance-onoff` で同じ条件（BTR の N128/B256、BF16、prefetch、Release、逆順の 4 Run）で測る。

`run_validation.py performance --label perf1` は Atari の既定の B/N、BF16、prefetch を保ち、OFF→ON→ON→OFF の順に実行する。warmup 後の learn_step 64〜256 の実時間を測り、ON/OFF の比が 1.3 以下か判定する。起動時間を含めない。比較中はほかの学習やビルドを重ねない。

追加レビューでは `--order on-off-off-on` で逆順の4 Runを測る。順序を変えるときは必ず新しい `--label` を使い、結果の `order` と `seconds_per_update` を残す。既定順も逆順も、有効2 Runの合計を無効2 Runの合計で除して1.3以下か判定する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/run_validation.py performance --order on-off-off-on --label perf_reverse1
```

変更前OFFとの比較は `performance-off`。保存した `baseline-bin/`・`baseline-config/` を使い、変更前OFF→変更後OFF→変更後OFF→変更前OFFを同じB/N・BF16・prefetch条件で測る。`comparison=after_off/before_off` の比を記録する。この比較自体の許容率はPRDに定めていないため、合否の閾値は追加しない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/run_validation.py performance-off --label off_compare1
```

Debugのskip更新では、primary・skipの勾配normをまとめてNaN/Inf検査し、非有限ならoptimizer適用前に止める。`[prd098-review-nan]` は両Network・FP32/BF16・NaN/Infの8条件、`[prd098-review-checkpoint]` はOFF→ONのprimary optimizer group数不一致の両方向を確認する。通常の `[prd098]` に含まれる。

通常実行の性能は `--build-config Release` でも同じBTR条件・逆順4 Runを測る。保存結果にはbuild構成を記録し、Debugの診断同期を含む結果と混ぜない。Debugの基準未達をReleaseの成功で置き換えず、両方を実装メモへ残す。変更前snapshotはDebugなので、段0の `before` / `after` / `compare` と `performance-off` はDebug構成に限定する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/run_validation.py performance --build-config Release --order on-off-off-on --label perf_release1
```

## round2の設定と検証

Headの読み口は `head.init.*` / `head.projection.*`。primaryのDuelingは `head.use_dueling`（bool、既定true）、skipは `skip.head.use_dueling`（bool、既定false）。skipの初期化・射影はprimaryの未指定項目を継承する。

agent.txtの@baselineへ無効のskip既定と共有配線を?=で置く。部品はnn.txtのSkipFC1/SkipFC2/SkipReLU、指標はmetrics_scalar.txt、J・IQN融合済みskip_context・run.@skipのenabled/metrics/Traceは各envが持つ。IQNは融合済みのrank3特徴を使い、taus bind自体は必須にしない。有効時だけskip網を構築し、actionのbindとnetwork_modeを検証する。

`run_validation.py smoke` の `--set` は繰返し指定でき、Runnerへkey=valueの一引数として渡す。包含型は `DefaultDQNAgent.skip.network_mode=action_in` と `DefaultDQNAgent.skip.net.branch.[skip_feature].bind=skip_context, action` の2葉。mergeはlearner.skip.feature_gradient=merge、Duelingはskip.head.use_dueling=true、幅はnet.block.[SkipFC1].linear.out_featuresで指定する。成功Runの再利用には生成設定・include全体・実行体に加えてCLI引数の内容と順序も一致する必要がある。

7envの全キー比較は、GridMazeの着手時ON解決値をresolved-before-rawへ保持し、切替だけをOFFへ揃えたsnapshot-configのresolved-beforeと行う。4env OFFの学習指標は時間・資源以外の全scalarを完全比較する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/verify_config_migration.py config
```

profile matrixは4env×default/merge/dueling/action_inの16条件と、GridMazeの幅256/J=1。J=1のRunはenvのj>1割合指標を選ばず、9基本指標とQ列を使う。actionのbindとDOT、IQN rank、Trace Q列設定、skip shape dump、QR/IQN分布軸を確認する。Traceのlane別実値はpublicなStepTraceBuilderを使うC++テストで検証する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/verify_config_migration.py profiles --label r2_profiles1
```

任意のQ列は `app.eval_trace_panel.columns.[420]=$action_info skip_q.[1] label:q1` のように宣言する。jは1..J、範囲外はエラー、無効はNaN。既定run.@skipのTrace列にはQ列を追加しない。

変更前snapshotは実行体・設定・検証スクリプトを一組として保存する。round2の変更前Runはsource-beforeの凍結ドライバで採取する。過去Runや固定入力を現行キーへ書き換えない。

## 長時間学習の準備と判定

長時間学習は準備を基本とし、round2受入6で指定されたGridMaze seed1/500kの分離stop・包含stopだけを再実行する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd098/verify_config_migration.py learning --build-config Release --label r2_learning1
```

評価報酬のstep_axisがexp_stepであることをdefsで確かめ、最終値・平均・最大値・最初に正の評価報酬へ到達したexp stepをlearning-summary.jsonへ残す。1seedの動作確認から一般的な学習効果は結論しない。

prepare-longはround2 rootのlong-configへ設定とCLI引数のcommands.jsonを書き、Runnerは起動しない。変種の葉は設定ファイルへ埋め込まない。

| 段 | 環境・予算 | 腕と seed | 次段へ進む条件 |
| --- | --- | --- | --- |
| 1 | GridMaze 6×10、J=7、500k | 無効、分離 stop、分離 merge、包含 stop。seed 1〜3 | eval の decision_rate が減り、return AUC が基準の seed 間のブレを超えて悪化しない |
| 2 | LunarLander、J=4、2M | 無効、分離 stop、分離 merge。seed 1〜3 | AUC が同等以上、eval decision_rate < 1。落ちるなら stop を残して調査 |
| 3 | qbert、J=10、10M | 無効、分離 stop、包含 stop。seed 1 | 5M までに基準へ追いつく。未達なら J=4 で一度だけ再試行。改善時は seed を追加して 50M |
| 4 | DropMerge nzrep の checkpoint から 50M 以上 | 無効継続、分離 stop。seed 1 | eval_target の maxrank・return が nzrep seed 1/2 のブレを超えて悪化しない。eval_target_greedy も観測 |

段4は、使用する nzrep の 50M checkpoint を `prepare-long --checkpoint` へ渡して生成する。未指定時は段4を開始できる設定に偽装せず、`long-config/status.json` に未生成の理由を記録する。保存済み Run artifact は書き換えない。

判定値は評価 slot のものを使う。train の decision_rate は探索を含むので監視用。j=J の集中が続く場合は専用 skip 方策を平均値の ε-greedy にした腕を追加する。q_shared・専用方策・PER λ の効果比較は今回の長時間実験に含めない。

## 結果

round2の結果と未実施の一覧は22implの「検証・レビュー結果」に記録する。各スクリプトは失敗した環境も記録し、独立した後続環境を続ける。学習効果や長時間の安定性は短い smoke の成功から推定しない。
