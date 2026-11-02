# PRD101 検証資材

実行はリポジトリrootのPowerShellから行う。Pythonは `.venv/Scripts/python.exe` を使う。
基準実行体・DLL・設定・採取値と結果は `.scratch/prd101/` に保存する。goldenは元宣言から移行し、新実装の実測値で更新しない。

- `verify.ps1`: MSVC初期化経由のDebug全体・テストターゲットのビルド、関連/coreテスト、段階2の4環境比較、2種類の短い実Runを順次実行する。再実行時は未使用の `-Label` を指定する。
- `run_validation.py stage1|stage2`: GridMaze・LunarLander・Atari・DropMergeを順次実行し、全学習scalarをtag・step・floatの厳密値で比較する。stage1はbefore、stage2はstage1-summary内のRunが基準。比較除外はPRD §6.1-1の指標一覧と時間・資源指標に限る。失敗後も次の環境へ進む。
- `short_run.py epsilon|learning_rate`: 512 exp_stepのphase切替を、εはAtari、学習率はGridMazeの短いRunで確認する。最終実行体を使い、結果名の既定は `short_epsilon_final` / `short_learning_rate_final`。既存の結果は上書きせず、再実行は `--label` で別名を指定する。
- `check_resolved.py resolved-stage1-final`（段階2は `resolved-stage2-r1 --stage2`）: 変更前7環境の全キーと文字列表記を、PRD D15/D16の置換・bare保持・新指標の宣言と比較する。期待値へ実測値を採用する処理はない。
- `check_contract.py`: 現用コード・設定・bat・testdata・jp設計文書・wikiカテゴリの廃止キーを検査する。C++のDTOメンバ.valueと、設定文字列のroot記法を区別する。
- `../prd072/profiled_migration.py`: CHANGELOGの対応表を参照してPRD072固定入力の旧宣言を新キーへ移す。`../prd072/prepare.py compare` が17入力を比較する。`check_default_leaves.py` は既定葉と実験宣言の演算子を検査する。

C++の `[prd101]` が設定→Policy→schedule→指標、bare dump、指数、ラダー、Learner・optimizerの境界を通す。Replay・Actor・MuZeroの境界も同タグで検証する。Debug全体・testターゲットは `AGENTS.md` のMSVC初期化経由でビルドしてから実行する。

## 保存した段階1の基準

beforeとstage1の4環境で学習系列が一致した。LunarLanderの採用結果は `stage1_r2_LunarLander`。LunarLanderとDropMergeの比較用lastは `1e-4`、現用設定のlastは0を維持する。0への移行では中間laneも変わる。
7環境の比較は `resolved-stage1-final-comparison.json`、PRD072は `prd072-r6-compare.log`、短い探索停止は `short_epsilon/assertions.json` に保存する。

段階2も4環境の全学習系列が一致した。最終実行体でAtariのε停止とGridMazeの学習率切替を確認した。最終core全体と再レビューの結果は[実装メモ](../../../../docs/memo/101_exploration_schedule_profiled_value_20impl.md)に記録する。
