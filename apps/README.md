# anet-lab apps

anet-lab の実行アプリケーションと起動 launcher(bat)を置くディレクトリです。

## 構成

| パス | 内容 |
|---|---|
| `runner/` | AnetRLRunner(強化学習 Runner 本体)。実行バイナリは `runner/bin/Release/` |
| `runner/config/` | Runner の設定ファイル群 |
| `runner/workspaces/<workspace>/` | workspace固有の`config/`と`runs/`。実行時生成・Git管理外 |
| `runner/tools/` | Runner関連の内部helper、テスト、Optuna study、ログ監視など |
| `metrics-viewer/` | Java/Spring 製 Metrics Viewer。jar は `metrics-viewer/target/metrics-viewer.jar` |
| `*.bat` | ユーザーが直接操作する起動 launcher 群(下表) |

## 実行要件

- Windows x64
- NVIDIA GPU + CUDA 13 世代対応ドライバ(R580 以降)。CUDA Toolkit のインストールは不要(必要な DLL は同梱)
- Metrics Viewer を使う場合: Java 17 以降
- 一部の補助 bat は追加ツールが必要(下表の「必要環境」参照)

## 起動 launcher 一覧

| bat | 用途 | 必要環境 |
|---|---|---|
| `10_run.bat` | Runner を GUI 起動。引数はRunnerへ渡し、runは選択workspaceの`runs/`に作成される | 同梱物のみ |
| `11_batch_run.bat` | Runner をバッチ実行(設定 override を列挙して連続 run) | 同梱物のみ |
| `21_metrics_viewer.bat` | Python 版 Metrics Viewer(開発用) | `viewers/metrics-tools` + venv |
| `22_metrics_viewer_java.bat` | Metrics Viewer(Java)起動 → http://localhost:8082 | Java 17+ |
| `23_optuna_dashboard.bat` | Optuna Dashboard → http://127.0.0.1:8088 | optuna-dashboard |
| `31_tb_bridge.bat` / `32_start_tb.bat` | TensorBoard ブリッジ/起動(開発用) | Python + TensorBoard |
| `41_mlflow_bridge.bat` / `42_start_mlflow.bat` | MLflow ブリッジ/起動(開発用) | Python + MLflow 3.13.0 |
| `80_dot_to_png_all.bat` / `81_dot_to_png_latest.bat` | run の dot グラフを PNG 変換 | Graphviz(dot) |
| `90_to_mp4_all.bat` / `91_to_mp4_latest.bat` | run の録画を MP4 変換 | ffmpeg |

workspace対応の補助batは第1引数をworkspace pathとして受け取り、内部では`runner/tools/resolve_workspace.bat`を共通利用します。省略時はRunnerが`GetAppDataDir()/last_workspace.txt`へ保存した絶対path、取得できない場合は`_default`を使います。相対pathは`apps/runner/workspaces/`基準です。既存の`runs/`があれば`config/_main.txt`がない手作業のworkspaceも利用できます。補助batはworkspaceや`runs/`を生成しません。

各 bat は作業ディレクトリを `apps/runner` に固定してから実行するため、どこから起動しても動作は同じです。MLflow bridge/serverは選択workspaceの`runs/mlflow.db`を共有します。

## リリース zip について

リリース zip(`anet-lab-<version>-win64.zip`)の内容:

- `LICENSE` — anet-lab本体に適用するApache License 2.0
- `NOTICE` — anet-lab本体の著作権表示
- `apps/` — 本ディレクトリ(実行バイナリ・設定・launcher。ソースは含まない)
- `docs/design/` — 設計ドキュメント
- `licenses/` — 同梱する第三者ソフトウェアのライセンス

Doxygen 生成の API ドキュメントは別 zip(`anet-lab-docs-<version>.zip`)です。
このzipにも直下に`LICENSE`と`NOTICE`を同梱します。
