# PRD102 GridMaze seed_mode 検証

## 契約

seedを指定すればその値で固定し、未指定時だけseed_modeのprocess（既定）・episodeを選ぶ。固定用のmodeやseedのsentinelはない。共通optional読み取りは未指定のDTO初期値を保持し、値がないとき設定記録へ葉を足さない。型変換・prefix順序・不正値の検証は通常の共通経路を使う。

## Debugと公開経路テスト

リポジトリルートからMSVCを初期化してDebugをビルドする。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py tests
```

通常GridMaze・指定seedの通知・process初期化（2つの独立プロセス）・並行8個体・全個体破棄後の再構築・遅延初期化・設定不正・RunManagerの学習/評価/GUI用Envを順番に実行する。各条件が失敗しても後続を実行し、終了値とログを`.scratch/prd102/tests.json`へ残す。既存の同名出力は`.previous-N`付きへ退避し、再検証前の記録を保持する。共通optional設定のタグはanet-core-testで実行する。

processの共有タグは独立プロセスで実行する。episodeは50回Resetして到達可能性と採用seedからの復元を確認する。乱数の重複禁止やプロセス間のseed相違は保証しない。毎Resetの採番は本体の経路もレビューし、test-only APIは足さない。GUIはRunnerFrameと同じCreateEvalRunnerの公開生成経路で確認する。

## 全キー・値の比較

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py settings
```

[manifest.json](manifest.json)の現用17入力を、今回の編集前に保存した`.scratch/prd102/baseline-config`と現用設定でそれぞれ解決する。Run選択は実際のCLI parserから渡す。PRD072の17入力を基にし、既に定義のないAtariのnature_dqn/classic_iqn_impala_x2の2条件は、現用の既定/base条件へ置き換えた。過去のmanifestやbaselineは変更しない。全キー・文字列値を保存し、GridMazeの旧bool削除・process追加・MuZero用のseed既定削除以外に差分がないことを確認する。GridMaze.txtのユーザーの実験用seed指定は保持する。`config-before.json`・`config-after.json`・`config-comparison.json`へ記録する。PRD072の過去baselineを更新しない。

既定葉の監査もリポジトリ規約どおり実行する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

## 固定条件の短Run

変更前の実行体・DLL・設定・driverと実出力を`.scratch/prd102/`へ隔離してから本体を変更する。今回は`before/`にrun seed1・迷路seed6545537233705677626・8 lane・CPU/FP32・8704 experienceの変更前結果を保存した。基準ビルドは実施済みで、実行体と設定のhashは`before/execution.json`に残る。以前のPRD100記録は上書きしない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py after
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd102/run_validation.py compare
```

現行PRD100 driverの実行体隔離・Noisy被覆・正規化を再利用し、出力だけPRD102へ分離する。変更後はseedだけを指定して固定する。時間・資源以外を全tag/step/値で完全比較し、点数・SHA256を`comparison.json`へ保存する。保存済み条件がない場合は比較を成功扱いせず失敗する。再試行時は既存stageを別名へ退避してから実行する。
