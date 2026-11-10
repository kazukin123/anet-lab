# PRD918 検証資材

リポジトリルートから実行する。MSVC の初期化と Debug ビルドは `AGENTS.md` の手順による。
検証ログと設定の採取先は `.scratch/prd918/`。既存の Run と過去の検証資材の保存結果は書き換えない。

## 発火列と設定読み込み

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[prd918],[eval_schedule],[interval_gate],[eval_session],[eval_runner]"
```

発火列は `EpisodeEvalObserver::OnLearn` から実際の EvalRunner を通し、テスト用 Env とセッション通知で検証する。
interval 1000、offset 0・500・2500、warmup_count 0・1・3、刻み 1・7・1500、遅い初回イベント、抑止回数過多を含む。
設定読み込みは Env 生成前の負数拒否、休止、interval 必須、scheduled の行を確認する。

## 出荷設定と bat

```bash
$env:ANET_PRD918_OUTPUT = Join-Path (Get-Location) '.scratch/prd918/config-check'; core\anet-core\bin\Debug\anet-core-test.exe "[prd918-config]"
```

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py static
```

env 別の固定値は hidden の採取・検証テストで確認する。通常の出荷設定の規則は `[shipped_config]` で確認する。
全キーの前後比較には、実装前と実装後に既存の `[prd093-config-snapshot]` を使い、`ANET_PRD093_OUTPUT` をそれぞれ `.scratch/prd918/config-before` と `config-after` の絶対パスにする。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py config-diff
```

## 短い Run

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py short-run
```

LunarLander の既定を 300000 exp step まで実行する。既存の経験数指標の記録間隔だけを 1 にして最終遷移を観測する。scheduled の値、初回評価の下限、Greedy とほかのタグの発火座標の不一致、予算完走を確認し、`short-run.json` と Run ログを残す。
既定スイートと GUI の hidden `[runner_close_drain]` も確認する。既知の失敗は実装前の結果と照合する。

最適化ビルドで同じ受入を行う場合は、対応する preset で Runner をビルドしてから次を実行する。証跡は `.scratch/prd918/RelWithDebInfo/` へ分けて保存する。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd918\validate.py short-run --configuration RelWithDebInfo
```
