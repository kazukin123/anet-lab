# PRD106 検証資材

正本は [PRD106](../../../../docs/memo/106_temporl_skip_forward_noise_key_10prd.md)、作業と結果は [実装メモ](../../../../docs/memo/106_temporl_skip_forward_noise_key_20impl.md)。実行はリポジトリルートの PowerShell。

## 既定の等価性

`capture_baseline.py` は `.scratch/prd106/` に着手前の Debug / Release の実行体・DLL・設定・resolver の出力と SHA256 を保存する。既存の保存先は上書きしない。`baseline-manifest.json` に基準コミットと全ファイルの hash がある。

`run_validation.py` は前後に同じ `baseline-config` を渡す。PRD103 の6構成に skip + NoisyNet の LunarLander を加える。8 lane、seed1、8704 experience、FP32、決定論的 backend、評価停止。Noisy を含め、`90_perf/`・`91_resource/`・`92_resource/` 以外の全 scalar 点を完全一致比較する。各入力・解決設定・実行体 hash・Run 名は `validation/` に残す。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd106/run_validation.py compare --label after
```

## 設定

`verify_config.py` は4 env（GridMaze / LunarLander / Atari / DropMerge）×既定 / skip の8入力を、同じ保存済み公開 ConfigManager resolver に通す。許す差は `skip.nn_forward_mode=shared` の追加だけ。宣言された全 Actor の値を確認し、`skip.nn_forward_key` が既定に出ていないことも検査する。出力は `.scratch/prd106/config/comparison.json`。同じ出力先への再実行は拒否する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd106/verify_config.py
```

`check_default_leaves.py` は PRD072 の既存手順で実行する。全体にある既存の指摘と、今回追加した2行への指摘を区別する。

`verify_dump.py` は現行の凍結設定を使った512 experienceのGridMazeと、完了済みのAtariのGの実効dumpを検証する。全Actorがshared、既定のskip.nn_forward_keyなしである。結果は `.scratch/prd106/dump/summary.json`。7構成の完全一致と8入力の設定比較はすべて成功し、詳細な点数とhashは実装メモと各comparison.jsonを参照する。

## Actor と τ

`[prd106]` が設定の読み取り、専用実行設定、Noisy の選び方と τ の分離を公開経路で確認する。テストのための本体 API は足さない。`[prd098]`・`[prd103]`・`[prd104]` を関連回帰にする。

τ の2つの組が既存経路のまま通る場合だけ、実装前の τ 評価を読みへ使う。経路を直した場合は、G・TQ・TS・TB・TB_indep と必要な G' を実装後の同じビルドで取り直す。fixed/fixed と shared fixed の値が違う場合は G' を基準にする。

## ステージ0

評価専用で、Learner と保存は無効。checkpoint は分割と合算の2本。条件は G / NQ / NS / NB / TQ / TS / TB / TB_indep。分割は seed1/2 各200ゲーム、合算は seed1 から始める。信頼区間が PRD の追加条件に当たる場合だけ、合算の該当する条件へ seed2 を加える。

実行と集計の実際のコマンド、実行体・設定・checkpoint の hash、trace 件数、結果は実装メモに記録する。h と k には同じビルドの G を使い、両者へ違うビルドを混ぜない。G4 の学習は、結果を提示してユーザーが選び、段取りを承認するまで開始しない。

`run_evaluation.py freeze` でReleaseと設定を固定し、`run --build new` がノイズ4条件、`run --build old --conditions TB_indep` が旧ビルドの独立τ条件を実行する。`--checkpoints mrg --seed 2 --conditions G TQ TS TB TB_indep --build old` は合算の追加条件。準備済み入力はhashを照合して再利用し、成功済みの200件は再実行しない。失敗・実行途中の記録があるときは自動でretryしない。評価は最大2条件を並行実行し、実時間は性能の比較に使わない。

`analyze_evaluation.py --label 名前` は実効dumpと200件のtraceを検証し、同じビルドのGだけから集中面を決める。seed1/2が揃った条件をまとめて10,000回のbootstrapへ渡し、比の分母が非正の回の割合も残す。合算でseed2を追加する条件をJSONへ書く。`analyze_evaluation_test.py` は面8、集中面、比と全体の差、分母0の扱いを固定データで確認する。計算のseedは106で、評価seedとは別。
