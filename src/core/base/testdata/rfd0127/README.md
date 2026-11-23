# PRD127 の受入検証

> この検証資材の設定キーは PRD191 より前の名前です。

> PRD182 で検証資材の置き場を変更した。本文のコマンドにある旧パスと実行体名は、現行の AGENTS.md「検証」に合わせて読み替える。

## 比較の入力

変更前の Debug 全体ビルドから Runner・DLL と設定一式を `.scratch/prd127/baseline-20261112/` の `baseline-bin/`・`baseline-config/` に保存した。全キーと文字列値は `[rfd0093-config-snapshot]` で `resolved-before/` に採取した。比較はその設定の写しに固定し、変更後の列の追加を基準Runの学習条件へ混ぜない。

`run_validation.py before`、`after --label after-r1`、`compare --label after-r1` の順で実行する。Runner のビルドは AGENTS.md の VsDevCmd 経由で行い、実装後には Debug 全体をビルドする。ログは `.scratch/rfd0127/` に保存する。

条件は RND 無効の GridMaze・LunarLander・Atari・DropMerge 各8,704件、LunarLanderの通常RND 100,000件、同じ構成に Atari の run.@rnd_taiga の8値を明示した100,000件である。どれもseed1・決定的モードで、Traceは使わない。`collect` は通常RND条件の識別名で、warmup_modeは出荷設定のlearn_predictorである。

時刻・資源の群だけを除いた全scalar点のchecksumと、metrics.scalar.defsの全体を比較する。各条件に実行体・設定入力のSHA256、引数、Run、終了コード、実時間を保存する。失敗した記録は保持し、明示的な `--retry-incomplete` で別名へ退避して再実行する。既存Runは削除しない。

## 単体・結合・出荷設定

`anet-core-test.exe "[rfd0127]"` が読むだけの採点・統計方式・Clone・step診断・同期・ロック・checkpoint・学習不変性を確かめる。既存の `[rfd0116]`・`[rfd0123]`・`[step_diagnostics]`・`[diagnostic_replay]`・`[diagnostic_no_wait]` とmetricsの関連テストも実施する。出荷列は `[shipped_config]`、pane開閉は Runner の `[rfd0127]` テストで確認する。

設定変更時はPRD072のcheck_default_leaves.pyと、変更前後の全キー・文字列値の比較を行う。追加を許すのは run.@rnd 選択時の235・600・610の3列だけで、選ばない解決は変更前と同じとする。`verify_configuration.py`は3環境それぞれで、空の選択・RND単独・TempoRL単独・両方の順序の併用の5通り、合計15通りを解決する。全キー・文字列値の比較で既存列の上書きが無いことを、列番号の昇順で235 `rnd_bonus`が230 `reward`の直後にあることを確かめる。実装時に採取済みの既定7環境の比較記録も再検査するが、現在のenvファイルの既定の選択は固定しない。

## 画面と費用

PRD §7-13に従い、利用者のViewメニューかPanelツールバーの操作からTraceを開閉して確認する。3列の位置、最初の行、10秒以上閉じたあとの最初の行、過去の行の不変性、RND非選択時に列が無いことを見る。

PRD §7-14の費用はAtariの実時間優先＋run.@rnd、さらにrun.@rnd_taigaを重ねた2構成で測る。閉じた状態と開いてEvalPanel再生中の状態を交互に3巡以上採る。各状態のtrain exp step/sの中央値、Observeの時間の中央値・最大、同期のうち推定器とβの写しの時間の中央値・最大を設計文書212へ記録する。採点と写しには既存profiler形式の範囲を設ける。

GUIの利用者操作・測定ができない環境では実施済みとせず、課題と実装計画に残件として記録する。費用に合否の閾値は置かず、PRD U1の既定列の採否は測定後の判断とする。

## 受入項目と公開経路の対応

| PRD §7 | 検証 |
|---|---|
| 1 | `RND read-only scoring leaves subsequent learning unchanged`（累積・EMA・variance_feedback） |
| 2 | `RND read-only formula follows current statistics and warmup`（中心化・clip・次の1件） |
| 3 | 上記のwarm-upに加え、`Read-only normalization leaves zero variances unavailable` と非有限値テスト |
| 4 | `RND scoring snapshots remain independent of training`（CPU/CUDA、既定RNGの不変も確認） |
| 5 | `Trace bonus equals the reward consumed by the real learner`（3方式×state/next_state×done×mask） |
| 6 | `Intrinsic diagnostics honor snapshot synchronization and suspension`、`Trace forwards requested intrinsic keys and Actor lifetime controls` |
| 7 | `Intrinsic diagnostics wait only for an un-cloned requested estimator` |
| 8 | `Checkpoint trace scores bonus before the first learning intake`（PRD129で、構築時のβによるボーナスも確認） |
| 9 | 無効・Actor欠けは同期テスト、Rainbowは `Rainbow rejects intrinsic Trace columns at the first scalar read`、非有限値は `Read-only nonfinite errors remain unavailable and warn once` |
| 10 | `Interleaved Trace leaves learner rewards and saved intrinsic state unchanged`（3方式×Actor複製有無×state/next_state） |
| 11 | `run_validation.py` の6条件。途中の失敗・中断は受入に数えない |
| 12 | `[shipped_config]` の列選択規則と、LunarLanderの解決済み全列からの実行時行生成。`verify_configuration.py` で全キーと3列goldenの値も比較 |
| 13 | 利用者と同じ画面操作から確認。実施結果は実装計画に記録する |
| 14 | 上の画面・費用の手順。実測値が得られるまで既定列の採否は未決 |
| 15 | Runnerの `[rfd0127]`。公開 `SetActive` から休止・再開・同期回数と、初期表示済みpaneを確認 |

`verify_configuration.py` は既存 `[rfd0117-resolve]` で、着手前の設定と現在の設定を同じ選択で解決する。環境別の値をC++テストへ写さず、3列の正確な宣言だけ `trace_columns.json` に保存する。既定7環境は `[rfd0093-config-snapshot]` で `resolved-after/` へ採取してから比較する。
