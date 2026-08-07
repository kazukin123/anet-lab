# anet-lab apps

anet-lab の実行アプリケーションと起動 launcher(bat)を置くディレクトリです。
リリース zip はこのディレクトリ構成をそのまま含みます(開発時とリリース時で同じ bat を使います)。

## 構成

| パス | 内容 |
|---|---|
| `runner/` | AnetRLRunner(強化学習 Runner 本体)。実行バイナリは `runner/bin/Release/` |
| `runner/config/` | Runner の設定ファイル群 |
| `runner/tools/` | 補助スクリプト(Optuna study、ログ監視など) |
| `metrics-viewer/` | Java/Spring 製 Metrics Viewer。jar は `metrics-viewer/target/metrics-viewer.jar` |
| `*.bat` | 起動 launcher 群(下表) |

## 実行要件

- Windows x64
- NVIDIA GPU + CUDA 13 世代対応ドライバ(R580 以降)。CUDA Toolkit のインストールは不要(必要な DLL は同梱)
- Metrics Viewer を使う場合: Java 17 以降
- 一部の補助 bat は追加ツールが必要(下表の「必要環境」参照)

## 起動 launcher 一覧

| bat | 用途 | 必要環境 |
|---|---|---|
| `10_run.bat` | Runner を GUI 起動。run は `runner/runs/` に作成される | 同梱物のみ |
| `11_batch_run.bat` | Runner をバッチ実行(設定 override を列挙して連続 run) | 同梱物のみ |
| `21_metrics_viewer.bat` | Python 版 Metrics Viewer(開発用) | `viewers/metrics-tools` + venv |
| `22_metrics_viewer_java.bat` | Metrics Viewer(Java)起動 → http://localhost:8082 | Java 17+ |
| `22_metrics_viewer_java_optuna.bat` | Optuna seed run 用 Metrics Viewer → http://localhost:8083 | Java 17+ |
| `23_optuna_dashboard.bat` | Optuna Dashboard → http://127.0.0.1:8088 | optuna-dashboard |
| `31_tb_bridge.bat` / `32_start_tb.bat` | TensorBoard ブリッジ/起動(開発用) | Python + TensorBoard |
| `41_mlflow_bridge.bat` / `42_start_mlflow.bat` | MLflow ブリッジ/起動(開発用) | Python + MLflow 3.13.0 |
| `80_dot_to_png_all.bat` / `81_dot_to_png_latest.bat` | run の dot グラフを PNG 変換 | Graphviz(dot) |
| `90_to_mp4_all.bat` / `91_to_mp4_latest.bat` | run の録画を MP4 変換 | ffmpeg |

各 bat は作業ディレクトリを `apps/runner` に固定してから実行するため、どこから起動しても動作は同じです。

## リリース zip について

リリース zip(`anet-lab-<version>-win64.zip`)の内容:

- `apps/` — 本ディレクトリ(実行バイナリ・設定・launcher。ソースは含まない)
- `docs/design/` — 設計ドキュメント
- `licenses/` — 同梱ソフトウェアのライセンス

Doxygen 生成の API ドキュメントは別 zip(`anet-lab-docs-<version>.zip`)です。
