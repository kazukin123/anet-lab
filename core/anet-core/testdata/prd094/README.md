# PRD094 検証資材

正本は[実装計画](../../../../docs/memo/094_evalpanel_step_history_20impl.md)。Pythonはリポジトリの`.venv/Scripts/python.exe`を使う。

- `run_validation.py before` は既存Release実行体・DLL・設定を固定し、Atari/DropMerge/LunarLander CPU/GridMaze/actor_approxの同seed・65536経験の短いRunを各2回、必ず1本ずつ順次実行する。
- `run_validation.py after` は変更後の実行体を固定して各1回比較する。`PRD094_BINARY_DIR`で隔離ビルドの配置を選び、`PRD094_AFTER_NAME`で`stage1`/`after-v1`などの世代を分離する。保存済み実行体を上書きしない。
- `compare_metrics.py` はscalar全点とtrace全行を、時間/資源3群とtrace timestampだけを除いて比較する。各metricのstep・値・重複、traceのlane・data、scalar/trace定義を比較する。PRD097のlives値・移行端の除外は持たない。
- 結果は既定で`.scratch/prd094/acceptance-20261021/`。`PRD094_ACCEPTANCE_ROOT`で基準を含む保存世代を分離する。実行引数・seed・終了コード・実行体SHA-256・cwdを記録する。実行中の探索とは実行体と設定を分ける。

C++の公開surfaceはObserverFactory/RunManager/EvalRunner、列設定とStepTraceBuilder、AgentStepDiagnosticsProvider、RewardScaler、gui::TracePlayer、追従paneのsnapshot/Live切替で検証する。各RED/GREENとビルド・CTestの結果は実装計画と`.scratch/prd094/`のログへ記録する。値の読取り・列解析・生成器のテストは`src/metrics_test.cpp`、再生・百分位・書式の単体テストは`src/gui_test.cpp`に置く。表示上限後の生成器と百分位consumerの結合テストはmetrics_test.cppで確認する。

GetScalarへの汎用化では`[generic_step_keys]`・`[generic_step_completion]`を含む`[prd094]`を使う。非DQNのModule提供元から任意キーを読み、lane別pending、列ごとの部分確定、既知NaN、NaNでの確定、未知キー、Reset、表示保持上限を超えた行の百分位更新を検証する。DQNの診断も同じGetScalar契約から参照値一致とmutex保持中の完了を確認する。この変更では追加の比較Run・性能測定・実機UI確認を行わない。

名前・継承・配置の見直しでは`[action_info_module]`・`[action_info_lane]`で、非DQNの未知scalarが既存WARNへ進むことと、基底ActionInfoのlane別action読取り・scalarのlane=-1の未知キー契約をRED→GREENで確認する。共通metrics処理はmetrics.hpp/cppとmetrics_impl.hpp、表示補助はgui.hpp/cppへ集約する。修正前後の比較は`.scratch/prd094/naming-layout-20261021/acceptance`へ世代を分け、5構成を各1回ずつ順次採取する。before側は`run_validation.run('before', case, 1)`で固定し、after側は同じハーネスのafterを使う。両側の既存設定比較には`compare_configs.py --existing-only`を使う。これは上記のGetScalar汎用化だけの検証とは別の修正であり、性能測定の反復・実機UI確認は行わない。

実機ではPRD094の受入8に従い、pane追従・再生操作・開く前1分×3の学習速度・表示中のEval速度95%以上と学習速度・1000行のメモリを確認する。探索Runとの共存状況を記録し、競合で判定できない値は合格としない。半日規模の反復性能測定へ広げない。

解決済み全キー・文字列値は`compare_configs.py --after after-v1`で固定beforeと比較する。出力先app.runs_dirは世代ごとのRun親に一致することを別途検査する。共通12キーとAtari固有2キーだけを意図した追加として検査し、既存キーの除去・値変更は許可しない。

レビュー後の比較では修正前ソース・実行体を`review-fixes-20261021`へ固定し、`PRD094_ACCEPTANCE_ROOT`をその`acceptance`へ向ける。基準は5構成を各1回採取し、`after`で各1回比較する。両側ともTrace設定を持つため、設定比較には`--existing-only`を付け、キーの追加も許可しない。これは性能の反復測定ではない。

レビュー修正の回帰テストは`[diagnostic_no_wait]`（失敗時もmutexを解放してjoin）、`[diagnostic_scaling]`（std更新・Reset）、`[scalar_action_contract]` / `[trace_action_dimension]`（委譲・厳密次元）、`[pane_open]`（最大化・明示開閉・復元・浮動pane）、`[trace_play_available]`（FPS 0・復帰）で確認する。実機pane操作・ProfileRangeの呼出回数・速度・メモリは今回の修正検証に含めず、上記の受入手順で別途確かめる。
