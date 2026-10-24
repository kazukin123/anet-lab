# PRD097 検証資材（結果所有契約）

現行正本は[22impl](../../../../docs/memo/097_env_auto_reset_after_notification_22impl.md)。Pythonはリポジトリの`.venv/Scripts/python.exe`で実行する。

- `run_validation.py after`: 改訂D7の7構成、512000経験、seed1、決定的backend、同期評価。凍結済みbeforeを再監査して利用し、serialだけはCLI指定でbefore3/4・after3を追加実行する。afterは環境変数`PRD097_AFTER_NAME`で世代を分ける（今回の検証先は`after-d7-v3`、既定値は`after-d7-v1`）。afterの設定は凍結beforeを基にし、Atari/DropMerge/GridMazeの宣言ファイルだけを現行DSLへ差し替える。
- `compare_metrics.py`: scalar全点・trace全行。時間/資源3群とtrace timestamp、指定lives7タグの値だけを前後比較で除外する。`--allow-migrated-edges`は定義レコードからenv/train→env_step/episode_endと確認したtagだけ、pipeline最初/最後stepの両側を除外する。他の座標やNoisy指標は比較する。
- `compare_metrics_test.py`: 上記例外の境界と重複行、Noisy、trace、livesのstep差を検出する。
- `run_performance.py`: Atari1/GridMaze/Lunar CPU/DropMerge pipeline/serialを各3組、before/after交互。before spread>2%なら5組へ延長し、なお不安定なら判定不能。各Runは最後に報告された速度点のstepの半分超の中央値。中央値の比が0.98以上で合格。
- `audit_validation.py`: 実効値・評価完走・finite loss・入力設定/DLL/実行体hashの監査。

公開APIテストは各envの`*_test.cpp`とCoreのenv/trainer/observers/episode_end。Atari e2はC++ ALEでAsterixを300frame進め、テストfriendでRAM0x53=1/0x47=3とし、301frameの遷移と302frameの確定ゲーム値を同じStep結果で検証する。段APIは使用しない。

今回の指摘対応ログは`.scratch/prd097/claude-fixes-20261020/`、Run資材は`.scratch/prd097/implementation-20261020/protocol-fixed/after-d7-v3/`。Debug/Release全体ビルド、Debug CTest全6実行体（378.07秒）が成功。DropMergeは29tests/1081assertions、maskのCPU/CUDA逐次・並行読出しは2tests/384assertionsが成功。全7構成728534点のscalar/trace一致、before14Run/241ファイル・after7Run/158ファイルの監査、全7構成の実効設定全キー比較に成功した。詳細は22implを参照する。

前回v2の最終受入: 全7構成728534点一致、正式性能46Run監査成功、Asterixの手動終端画面確認済み。性能はLunar合格、Atari1/GridMaze/Drop pipeline/serialはbefore5反復のspreadが2%を超え判定不能。合格への読み替えは行わず、原因調査と再測定候補を21implへ記録した。v2でのGPU検証は終了。集計はafter-d7-v2/performance.json、調査はreimplementation-20261020/diagnosis-drop-20261020-213445/analysis.json。

現在の指摘対応は22impl。①〜④・⑥を修正し、⑤と性能再測定・追加プロファイルはユーザー指定で実施しない。v2の速度値は過去の測定値であり、修正後の効果を示さない。

## 履歴（旧M2・不採用、現行手順ではない）

この節の説明・進行状況は当時の記録。`check_shutdown_profile.py`は廃止済みで、現用testdataには存在しない。現行の手順は上記と22implを参照する。
### PRD097 検証資材

正本は `docs/memo/097_env_auto_reset_after_notification_20impl.md`。本ディレクトリのPythonはリポジトリルートの `.venv/Scripts/python.exe` で実行する。

#### 構成

- `compare_metrics.py`: scalar全点とtrace全行を比較する。時間・資源タグ3群以外は除外しない。変更前後の比較でだけ `--allow-lives-values` によりPRD指定7タグの値を除外し、点数とstepは維持する。変更前2反復ではこのオプションを使わない。
- `compare_metrics_test.py`: Noisy指標・trace値・残機tagのstep・重複行の差を検出することを確認する。
- `run_validation.py before`: 変更前ReleaseとDLL・設定を保存し、7構成を各2回、seed=1・512000経験で順次実行して比較する。独立caseの失敗も集計する。
- `run_validation.py after`: 変更後を各1回実行し、保存した変更前と比較する。
- `audit_validation.py before` / `after`: seed・経験数・決定的backend・Runner種別とAtariの残機/評価設定・学習/trace成立を確認し、DLL・入力/実効設定のSHA-256を保存する。
- `run_performance.py`: 固定したAtari1の変更前/後を交互に各3回実行する。各Runは512000経験の後半の速度中央値、最終値は3反復の中央値。後/前が0.98以上なら合格とする。

結果は `.scratch/prd097/implementation-20261020/protocol-fixed/`。初回の誤ったAtari設定の試行は一つ上の `before/` に不採用記録として残す。既存成果物を削除・書換しない。構成別の `.execution.json` に実行体ハッシュ、引数、PID、終了コード、Runパスを保存し、`.comparison.json` に比較結果を残す。正常終了済みの同じ実行体の記録だけ再利用する。

検証用overrideは末尾の `run.@prd097` に置く。AtariのRunプロファイルがEnv選択を上書きするため、rootのAtariEnv.$だけを上書きしても@classic/@100kは選択されない。

#### 実行記録

- 2026-10-20: 比較器の契約テストOK。変更前Release全体ビルドと7構成各2回（計14 Run）が完了し、728534点のscalar・traceが一致。`before/audit.json`は14 Run・241ファイルの実効値/成果物/hash監査に合格。
- 公開APIのRED/GREEN記録は実装メモと`.scratch/prd097/*-{red,green}-test.log`。Atari受入2(e2)はAsterixのC++ Env経路で24 assertionが合格し、301 frameの遷移aux・302 frameのゲーム完了・段後のNaNを確認。
- Core境界は関連51テスト・731 assertion、ImageClsは15テスト・148 assertion、Atariは38テスト・1016 assertion、DropMergeは28テスト・342 assertionが合格。GridMazeは専用targetを追加し、done/truncated後のNaNを確認した。
- レビュー修正後のDebug全体ビルドと全6実行体のCTestが成功（310.48秒）。ThreadPoolのworker失敗6経路も12 assertion成功。Release版Asterix必須受入は24 assertion成功。独立レビューは計画/実装ともOK（review-20261020-0406-impl.md）。
- 変更後7 Runと交互3反復の性能受入を実施中。EvalPanel手動受入はUI基盤の初期化エラーで未実施。これらが残る間は完了扱いにしない。

変更後の実行体を再ビルドした場合は、古いsnapshotで続行しない。`PRD097_AFTER_NAME`に新しいディレクトリ名を設定すると、afterの実行体・Run・監査・性能結果を別世代へ分離できる。性能反復は世代ごとに未使用の変更前番号を予約し、前後の交互実行を保つ。保存済みの入力設定は内容が同じなら再保存せず、差がある場合は停止する。


初回変更後世代 `after` は7構成すべて728534点の比較と158ファイルの監査に合格したが、交互3反復の性能比が0.9170470998123731（8.30%低下）で不合格。結果は保存し、この世代を採用しない。段の観測複製をlane単位からバッチ単位へ変え、`after-v2`で数値比較と交互性能を再検証する。


`check_shutdown_profile.py`は、評価セッションを含まない実headless Pipeline RunnerのNVTX SQLiteから、Step数と通知後の段数、最後のStep後の1段を検査する。実終了経路は修正前1000/999/0でRED、修正後1000/1000/1でGREEN。ログは`app-shutdown-{red,green}.log`。

`after-v2`の交互3反復も性能比0.9188551573487035（8.11%低下）で不合格。NVTXではResetが学習workerと重なる区間から外れたことを確認。現在の`after-v3`は実アプリ終了段の修正を含み、7構成比較を実行中。別ALEでの次ゲーム準備はproduction外の予備検証だけを実施し、PRD/計画へ追加する採否を質問中。性能合格・EvalPanel手動受入は未達のため未完了。

監査はAtariのlossタグの存在に加えてfinite値を確認する。変更前の再監査は`--output-name audit-finite-loss.json`へ分離し、14 Run/241ファイル合格。元の`audit.json`は保持する。


再計測107–109は6 Run正常終了。after-v3の変更前中央値5077.56396484375、変更後4864.73291015625経験/秒、比0.9580840229367649（4.1916%低下）で基準2%以内は不合格。`v3-performance-validation.log`、`after-v3/performance.json`。別ALEなしの結果であり、別ALE案のRun全体の高速化率は未計測。GPU検証は終了した。
