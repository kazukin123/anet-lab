# PRD122 検証

正本は [20impl](../../../../docs/memo/122_small_refactor_and_test_gap_fixes_20impl.md)。

- DebugビルドはAGENTS.mdのMSVC初期化経由で行う。変更前はPRD109のsnapshotでRunner・DLL・configを専用の新規フォルダへ保存する。
- `run_validation.py` はPRD098の短縮条件、PRD109の凍結設定採取と隔離実行、PRD084の全scalar正規化を使う。既存の基準とRunを上書きしない。
- GridMaze/LunarLanderはCPU、qbert/DropMergeはCUDA。seed1、8lane、batch16、warmup256、RR1、8704experience、評価停止、FP32、決定的backend。時間・資源3prefix以外の全点をtag/step/valueで完全比較する。
- 実行例: リポジトリルートで `.\.venv\Scripts\python.exe core/anet-core/testdata/prd122/run_validation.py --baseline .scratch/prd122/baseline --output .scratch/prd122/compare-stage1`。出力先は毎回新しい名前にする。
- coreの追加テストは `[prd122]`。deviceの既存 `[prd070]` は別processで `CUDA_VISIBLE_DEVICES=-1` にして流し、skipが無いことを確認する。
- GUIの加工とPanelの特性テストはGUI専用実行体を明示ビルドして流す。既定のGUI suiteと `[prd105_ui_keys]`・`[eval_steps_menu]`・`[prd109_ui]` は別processに分ける（[PRD109](../prd109/README.md)）。
- 破壊検査はPRD122 A1/A2/A3/A5/A8/A9の指定箇所を1箇所ずつ変え、落ちたテストと復元後の成功を20implへ残す。
- A8/A9の実機操作は人が確認する。自動テストの成功から目視確認済みとはしない。
