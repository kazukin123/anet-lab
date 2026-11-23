# PRD170 検証資材

> PRD182 で検証資材の置き場を変更した。本文のコマンドにある旧パスと実行体名は、現行の AGENTS.md「検証」に合わせて読み替える。

実装計画は [170_nn_trace_capture_unification_20impl.md](../../../../../docs/memo/0170_nn_trace_capture_unification_20impl.md)。要求しない Actor 呼び出しを既定とし、要求した nn_trace の鍵・形・型・全値と、行動選択・学習の数値を保持する。

## 受入との対応

| 受入 | 公開経路と証拠 |
|---|---|
| A1 | native_stack 5構成、IQN の2次元 τ 入力、ImageCls の全値 JSON。下記の同一 seed 手順で7ファイルを完全一致比較 |
| A2・A3 | `[rfd0170]`、`[native_stack]`、`[image_cls][trace]`、`[nn][trace]`、`[nn][branch_capture]`。無指定・false・true を確認 |
| A4 | `[rfd0170][runner]`。Eval / serial / pipeline の通知、3 step frame、Eval 手動行動 |
| A5 | `[rfd0170][session]`。RunSession の3回の通知に nn_trace が無い |
| A6 | GUI `[rfd0170_gui]`。Step のコマンド、Conv2d メニュー、Q-Values 左クリック、AUI close、実タイマーから取得した timer のイベント。2枚のうち1枚を閉じる場合も確認 |
| A7 | GUI `[rfd0170_startup]` を登録条件ごとの別プロセスで実行。無登録・interval=0・interval=1 |
| A8 | `[rfd0170][muzero]`。同じ新規 Actor の false→true→true で WARN 0→1→1、出力は空 |
| A9・A10 | 現用範囲の旧識別子の検索、Body / RunForward の宣言の確認 |
| A11 | `prepare_inputs.py` と `validate.py`。PRD103 の6構成と同じ短縮条件、性能・資源以外の全 scalar の tag・step・値を完全一致比較 |
| A12 | 変更前後の既定 core / GUI 一式を同じ Catch seed で比較 |
| A13 | 設計文書 140 §5.4 の日英シーケンス図 |

## trace 全値

`ANET_RFD0170_TRACE_DIR` を指定すると、既存テストの7構成が辞書順の鍵、shape、FP32 dtype、全要素を JSON に保存する。nn_trace の Actor seed は123。IQN の重み初期化は torch のグローバル乱数も使うため、既存の seed 固定テスト `[rfd0128][dqn][probe-resubscribe]` を辞書順で先に実行し、その後の torch の状態も同じにする。Catch の seed は170、順序は lex にする。値の除外・丸め・許容誤差は使わない。

変更前の採取用実行体は製品の変更前に保存した `core/anet-core/bin/Debug/anet-core-test-prd170-trace.exe`。変更後は通常の `anet-core-test.exe`。両方に次の同じ filter を渡す。

```text
[rfd0128][dqn][probe-resubscribe],[native_stack],[image_cls][trace],[dqn][iqn][tracer]
```

証拠は `.scratch/prd170/trace-before-seeded/`、`trace-after-final/`、`trace-final-comparison.json`。実行体・ログはローカル資材で、Gitには含めない。

## 6構成の学習数値

初回だけ次を実行し、履歴設定を受入用コピーへ保存する。既存の凍結入力を上書きしない。コピーから削除するのは、PRD117 で廃止された未読の auto_scale_mode と histgram.interval だけ。原本は変更しない。削除した行と SHA256 は input-manifest.json に保存する。

```bash
if (-not (Test-Path .scratch/rfd0170/validation/baseline-config)) { .\.venv\Scripts\python.exe core/anet-core/testdata/rfd0170/prepare_inputs.py }
```

変更前後の実行体は `apps/runner/bin/Debug/AnetRLRunner-prd170-before.exe` と `AnetRLRunner-prd170-final.exe` に保存する。DLL は元の bin ディレクトリのものを使う。`validate.py run` に新しい label を指定して採取し、`compare` に対応する2つの label を渡す。今回の保存済み結果は次で比較できる。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/rfd0170/validate.py compare --label after_final --reference before2
```

6構成は GridMaze / LunarLander / Atari qbert BTR / DropMerge nzrep の skip OFF と、GridMaze / LunarLander の skip ON。N8、batch16、warmup256、capacity4096、RR1、決定的な backend、8704 experiences、評価停止。除外は既存の性能・資源の群だけで、checkpoint の SHA は比較しない。全起動引数、実行体の SHA、終了値、所要時間、正規化データは `.scratch/rfd0170/validation/` に保存する。

## GUI と GPU メモリ

RunnerApp を起動する hidden test は1回の起動につき別プロセスで実行する。`[rfd0170_startup]` は `ANET_RFD0170_IMAGE_REGISTRATION=0/1/2` で無登録・無効・有効を選ぶ。`[rfd0170_gui]` と混ぜて同じプロセスで回さない。

`prepare_inputs.py --ui-output` は現用出荷設定をコピーし、学習停止、Eval の100 step/frame、fps1、終了時保存なしの確認用入力を作る。出荷設定のモデル・batch・dataset は保持する。`[rfd0170_memory]` は `ANET_RFD0170_MEMORY_CONFIG` でこの入力を受け、`ANET_RFD0170_MEMORY_OUTPUT` へ allocator 統計を保存する。GUI を操作せず、実際の RunManager / EvalRunner で要求なし100 step、要求あり100 stepを進める。current / reserved / peak とピークの差を記録し、閾値を置かない。

この補足測定は、pane を開く前後のGPUメモリを実画面で比べる手順そのものではない。実画面の再生、開き直し、停止中の各操作、Trace の履歴・ダンプと合わせて、未実施のものは実装計画に残す。

## throughput

`performance.py` は同じ Atari qbert BTR の凍結入力を2048 experiencesへ短縮し、変更前→変更後の順を3巡回す。学習側の `exp_step_per_sec` を32 experiences間隔で記録し、1024〜2048の平均をRunの値、3 Runの中央値を群の値とする。比較の合否は設けない。起動引数、pid、実行体 SHA、各 Run の値と群別中央値を `.scratch/rfd0170/performance/` へ保存する。実行中のほかの Run は停止しない。
