# PRD077：sampleable index 列の持続バッファ化 実装メモ

## 概要

承認済み計画に従い、列挙結果・順序を維持して GetValidIndices1D() の capacity サイズの確保とコピーを除去する。

## 主な変更

- ValidIndexManager に mutable torch::Tensor valid_buf_ を追加し、構築時に capacity 分の CPU Int64 ストレージを一度だけ確保する。
- 既存の ForEachSampleableIndex() でバッファへ順番に書き込み、narrow(0, 0, count) を返す。空集合も同じ経路で処理する。
- const 宣言を維持し、宣言コメントに owner の metadata 排他下で呼ぶこと、返り値の内容は次回呼び出しまで有効であることを明記する。
- GetTensorVector() と DumpToLog() は metadata_mutex_ 保持中に clone() する。通常の Sample 系はビューを使う。
- 既存 profiler 範囲、昇順、dummy 除外、history margin を維持する。
- 公開ヘッダ・設定・メトリクス・保存形式は変更しない。コピー禁止や独自同期機構は追加しない。

## テストと実装順序

1. 改修前のソース、設定、Release 実行体と依存物を識別可能な形で保存する。既存テストの失敗を記録する。
2. T1 RED→GREEN: 非空の最初の Tensor を保持して再呼び出しし data_ptr() 一致を検証する。旧実装で失敗を確認後、持続バッファ化と呼び出し側の clone を実装する。
3. T4: clone の内容が状態更新・再呼び出し後も不変、同一状態の列挙結果が一致することを確認する。古いビューの次回呼び出し後の内容は検査しない。
4. T5・T6: 空集合の shape・numel、既存 wrap/stack/n-step/PER テストを確認する。値固定 oracle を再利用する。
5. VsDevCmd 経由で Debug ビルドし、ReplayBuffer テストと anet-core-test 全体を実行する。既存失敗と新規失敗を区別する。

Public surface は ValidIndexManager の公開メソッドと ReplayBuffer の既存 API とする。T4・T5 が追加時点から成功した場合は回帰確認として記録し、人工的な失敗は作らない。

## Run による受入検証

- Breakout、現行 SN12、128 env、seed 1、2M exp steps、RR4、capacity 2,097,152、決定論モード、eval 停止、Release で固定する。
- 旧→新→旧→新の計4本。最初の旧版 Run はコード変更前に採取する。実効設定を保存・比較し、PRD078 や他の変更を比較途中へ混ぜない。
- 学習系列の対象タグ一覧を基準採取時に固定する。時刻・性能・資源計測を除外し、タグ・step・値・欠損位置を正規化して checksum を厳密比較する。checkpoint の raw SHA は合否に使わない。
- 全 Run に共通する 1M～2M 内の計測点を両端に取り、経過時間差から exp/s を算出する。新版2本の算術平均が旧版2本の平均以上なら合格。低下の許容幅は設けない。
- 終了コード、到達 step、実効設定、実行体 hash、checksum、各 throughput と平均値を保存し、本メモへ結果を追記する。基準不足・非決定性・測定負荷混入は未達または判定不能とする。

## 前提・範囲外

- 無確保の対象は capacity サイズのデータストレージであり、Tensor ビューのメタデータまで無確保とは主張しない。
- 性能の結論は今回の決定論モードに限定する。
- O(capacity) 走査、GetSampleableCount()、dummy 機構、一般的な bad_alloc 対策は変更しない。
- ADR・CONTEXT は更新しない。無関係な未コミット変更を保持し、staging・commit・push は行わない。

## 実行記録

- 開始時 HEAD: 71fc8c432db375a190ae6a75a7b7561fe9e4a933。
- 本メモ保存時点ではコード変更・ビルド・Run 実行は未実施。

- 改修前 Release ビルド成功。old/ へ実行体と依存 DLL を保存し、old-runtime-manifest.json に全 hash を記録。
- ユーザーより既存 Run は Pause 中、検証実施 OK と確認済み。
- 証跡: .scratch/prd077/。固定設定 baseline-config/prd077.txt、起動 run.ps1、学習系列比較 compare_metrics.py。

- 初回基準Run run_20260920-082320_prd077_old1 は eval_target が稼働していたため不採用。run.@evaloff は online eval のみ停止する現行定義だった。PID46568 のみ停止し、artifact と rejected-eval-target-* 記録を保持。
- 比較用固定設定に run.eval_schedule.[eval_target].interval = 0 と run.eval_schedule.[eval].interval = 0 を明示し、新規Runで基準を再採取する。製品設定は変更しない。
- 改修前Debug全テスト: 600件、594 passed、4 failed、2 failed as expected。終了コード42。baseline-tests.log に記録。

## 2026-09-20 予算変更の合意

- ユーザー指示により各Runを200万exp stepへ短縮。取得済みの旧版Runは取り直さず、200万stepまでの共通データを比較する。
- 旧版1本目は200万を超えてから正常終了要求を送信。終了時の超過分は比較に含めない。
- 後続は app.exp_exit_step のみ200万へ変更する。学習スケジュールを維持するため、元のmax/half予算変数は変更しない。実効設定比較から終了予算のみ除外する。
- throughput窓は共通の100万～200万step。cap2Mが十分埋まった定常領域の性能検証とは主張しない。

- T1 RED確認: 非空Tensorを保持した2回の列挙で data_ptr 不一致により1件失敗。t1_red.test.log。
- 実装後 T1 GREEN。T4追加後2件成功、T5追加後3件・10 assertions成功。t1_green/t4/t5.test.log。
- 実装差分を確認し、git diff --check 成功。公開ヘッダ・設定・列挙アルゴリズムは変更なし。

## 2026-09-20 アッセイ完了・ユーザー指定で一時中断

- 実装完了。Debug全体ビルド成功。Release runnerの変更後ビルドも成功し、再開用実行体と旧版と同一の依存DLLを `.scratch/prd077/new/` へ保存した。
- 変更後全テスト: **603件、597 passed、4 failed、2 failed as expected**。assertions: 162218、162140 passed、76 failed、2 failed as expected。終了コード42。
- 改修前は600件、594 passed、4 failed、2 failed as expected。追加3件・10 assertionsの成功分だけ増加。行番号を除く失敗式・実値・seed・条件の一覧が完全一致し、**新規失敗0**。192条件のintegrity assayも完走した。
- 証跡: `.scratch/prd077/baseline-tests.log`、`after-tests.log`、`regression-comparison.json`。T1 RED/GREENとT4/T5はそれぞれのtest.logに保存。
- 最初の全回帰試行はPowerShell 5が通常stderrをNativeCommandErrorとして扱ったため中断。`after-tests-native-stderr-interrupted.log` に保持し、PowerShell 7で全体を再実行して上記結果を得た。中断試行を成功には数えていない。
- ユーザーがモデル切り替えのため「アッセイ結果を保存して一旦終了し、残りの200万step比較は次のメッセージで再開する」と指定。後続Runは未起動。
- `pause-before-runs.txt` を `run.ps1` が確認し、起動前に77を返す。制御ログの `Run failed: new1` はこのガードによるもので、new1のRun失敗ではない。`new1.execution.json` は存在しない。状態記録は `paused_by_user` に補正した。

### 再開位置（基準Run・実装・テストはやり直さない）

1. 有効な旧版基準は `run_20260920-082901_prd077_old1`（終了コード0）。実体は `.scratch/prd077/runs/` 内。200万超過分は比較に含めず、取得済みデータを使う。eval_targetが動いた最初の不採用Runと混同しない。
2. **比較スクリプトの共通窓処理を仕上げてから再開する。** 現在の `compare_metrics.py` は全tagに `step <= 2000000` を適用するため、learn stepのtagでは旧版の超過学習分が残る。指標定義のrunner/step_axisを確認し、exp stepは200万まで、learn stepなどは比較Run間の共通prefixに揃えて厳密比較する。欠損位置・中間点の不一致は見逃さない。全4本の最終判定も共通prefixで統一する。既存のold1の正規化出力はこの調整前なので、そのまま厳密一致の証拠にしない。raw artifactから再集計すればよく、Run再採取は不要。
3. 111tagの候補一覧は `checksum-tags.json`、旧版の正規化出力は `old1.metrics.json` にある。再集計時は旧出力を保持する。throughputは100万～200万の共通実測点、許容低下0%。
4. 比較用設定の実効値を起動直後に確認する。両evalのinterval=0、RR4、cap2097152、128env、seed1、deterministic。終了予算だけ `app.exp_exit_step=2000000` とし、元の学習スケジュール変数は維持する。
5. 明示再開後に `pause-before-runs.txt` の起動ガードを解除し、PowerShell 7の `pipeline.ps1` の `ResumeRuns` 入口から **new1 → old2 → new2** を実行する。この入口は基準採取・TDD・全回帰・ビルドを繰り返さない。通常入口やResumeRegressionは使わない。
6. `.scratch/prd077/old/` と `new/` の保存済み実行体・DLLを使う。他の変更やPRD078を比較へ混ぜない。各Run実効設定とmanifest、終了状態、checksum、throughputを確認して本メモへ結果を追記する。

現在の受入状態: 以下の再開後検証により、実装・回帰・学習等価性・throughput非退行をすべて確認済み。

## 2026-09-20 200万step比較結果

- 明示再開後、旧版1本目の取り直しは行わず、保存済みraw artifactを再集計した。調整前の `old1.metrics.json` は `old1.metrics.pre-common-prefix.json` として保持した。
- 指標定義の `(runner, step_axis)` ごとに4本の共通prefix上限を求め、`train_step=15600`、`episode_step=5002`、`exp_step=1999744`、`learn_step=28000` で固定した。旧1本目の余剰な `exp_step=2184192`、`learn_step=31000` までの記録は比較から除外した。
- 111tagの step・値・`null`/非有限値・重複step・出現順を共通prefixで正規化してSHA-256比較した。`new1`、`old2`、`new2` はいずれも `old1` と変更tag 0。定義不一致0、欠落座標系0。学習系列の厳密等価性は合格。
- 実効設定は4本で一致。旧1本目のみ起動済みだったため `app.exp_exit_step=10000000`だが、後続3本は `2000000`。この終了予算とRun名・出力先を除き、両eval interval=0、RR4、capacity 2,097,152、128 env、seed 1、決定論設定を含むmodule実効設定が一致した。
- 4本とも終了コード0。後続3本は `app.exp_exit_step=2000000` による通常終了で、最後に観測された学習系exp座標は1999744。Runは `old1=run_20260920-082901_prd077_old1`、`new1=run_20260920-092647_prd077_new1`、`old2=run_20260920-094359_prd077_old2`、`new2=run_20260920-100056_prd077_new2`。
- 実行体SHA-256は旧版 `F4FA0393071143317110EEF7C95DC6EFFFE817147453D8521B24B649AFCD57F6`、新版 `2507EA4F1D4828B731B88EA54BA319D365E8DAEB9B1C547A02E7CF185BD5D4E7`。同じ世代の2本で一致した。
- throughputは4本共通の `exp_step=1000064..1995008` を使用。`old1=1843.404 exp/s`、`new1=1852.754 exp/s`、`old2=1869.240 exp/s`、`new2=1872.191 exp/s`。旧版平均 `1856.322 exp/s`、新版平均 `1862.473 exp/s`、相対差 `+0.3313%`。許容低下0%の条件で非退行に合格した。
- 合否は `.scratch/prd077/acceptance.json`、実行状態は `pipeline-state.json`、各Runの実効設定は `*.effective-config.json`、終了コード・実行体hash・Run先は `*.execution.json` に保存した。最終受入判定は **passed=true**。
- 200万stepはReplayBuffer capacity 2,097,152未満のため、性能結論は今回の決定論モードと非充填期間に限定する。
