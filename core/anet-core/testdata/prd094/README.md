# PRD094 検証資材

正本は[実装計画](../../../../docs/memo/094_evalpanel_step_history_20impl.md)。Pythonはリポジトリの`.venv/Scripts/python.exe`を使う。

- `run_validation.py before` は既存Release実行体・DLL・設定を固定し、Atari/DropMerge/LunarLander CPU/GridMaze/actor_approxの同seed・65536経験の短いRunを各2回、必ず1本ずつ順次実行する。
- `run_validation.py after` は変更後の実行体を固定して各1回比較する。`PRD094_BINARY_DIR`で隔離ビルドの配置を選び、`PRD094_AFTER_NAME`で`stage1`/`after-v1`などの世代を分離する。保存済み実行体を上書きしない。
- `compare_metrics.py` はscalar全点とtrace全行を、時間/資源3群とtrace timestampだけを除いて比較する。各metricのstep・値・重複、traceのlane・data、scalar/trace定義を比較する。PRD097のlives値・移行端の除外は持たない。
- 結果は既定で`.scratch/prd094/acceptance-20261021/`。`PRD094_ACCEPTANCE_ROOT`で基準を含む保存世代を分離する。実行引数・seed・終了コード・実行体SHA-256・cwdを記録する。実行中の探索とは実行体と設定を分ける。

C++の公開surfaceはObserverFactory/RunManager/EvalRunner、列設定とStepTraceBuilder、AgentStepDiagnostics、RewardScaler、TracePlayer、追従paneのsnapshot/Live切替で検証する。各RED/GREENとビルド・CTestの結果は実装計画と`.scratch/prd094/`のログへ記録する。

実機ではPRD094の受入8に従い、pane追従・再生操作・開く前1分×3の学習速度・表示中のEval速度95%以上と学習速度・1000行のメモリを確認する。探索Runとの共存状況を記録し、競合で判定できない値は合格としない。半日規模の反復性能測定へ広げない。

解決済み全キー・文字列値は`compare_configs.py --after after-v1`で固定beforeと比較する。出力先app.runs_dirは世代ごとのRun親に一致することを別途検査する。共通12キーとAtari固有2キーだけを意図した追加として検査し、既存キーの除去・値変更は許可しない。

レビュー後の比較では修正前ソース・実行体を`review-fixes-20261021`へ固定し、`PRD094_ACCEPTANCE_ROOT`をその`acceptance`へ向ける。基準は5構成を各1回採取し、`after`で各1回比較する。両側ともTrace設定を持つため、設定比較には`--existing-only`を付け、キーの追加も許可しない。これは性能の反復測定ではない。

レビュー修正の回帰テストは`[diagnostic_no_wait]`（失敗時もmutexを解放してjoin）、`[diagnostic_scaling]`（std更新・Reset）、`[scalar_action_contract]` / `[trace_action_dimension]`（委譲・厳密次元）、`[pane_open]`（最大化・明示開閉・復元・浮動pane）、`[trace_play_available]`（FPS 0・復帰）で確認する。実機pane操作・ProfileRangeの呼出回数・速度・メモリは今回の修正検証に含めず、上記の受入手順で別途確かめる。
