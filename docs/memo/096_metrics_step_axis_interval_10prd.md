# メトリクスの横軸と間隔を数える単位を分ける PRD

> 2026-10-18 起票（グリル前の暫定版）。ユーザーの希望で、999 ではなく正式番号で起票した。
> 起点: DropMerge の勾配ノイズ系列（[092](done/092_gradient_noise_scale_10prd.md) の本線分析、2026-10-16〜18）。B・RR の違う腕の学習時の診断（`49_agent_gradnoise` など）を、Viewer で経験数の横軸に重ねて見られず、毎回スクリプトで学習ステップの範囲を経験数へ換算した。
> 関連: [092](done/092_gradient_noise_scale_10prd.md) §7.4・A17（腕どうしの比べ方）、[093](done/093_attention_entropy_diagnostics_10prd.md)、[940](940_target_update_step_axis_10prd.md)（target 更新を何の軸で数えるか）、[done/052](done/052_eval_schedule_separation_10prd.md)・[done/053](done/053_interval_gate_perf_ema_10prd.md)（間隔ゲート）、[done/054](done/054_inspect_run_10prd.md)（inspect_run）、[ADR 0029](../adr/0029-analysis-metadata-emitted-by-runner.md)（指標の定義は Runner が出す）

## 1. 目的と問題

メトリクス定義の `$learn_step` / `$exp_step` / `$train_step` は、「記録の横軸」と「`interval:` を数える単位」を兼ねている。この 2 つを別々に指定できない。

学習時の診断（勾配ノイズ・attention エントロピー・可塑性・churn・Noisy・replay_fit）は、費用を抑えるため一部の更新だけで測る。どの更新で測るかは Learner が学習ステップで数えて決めるので、記録も学習ステップの軸に置くしかない（記録の発火と測った更新を一致させるため）。その結果、次のことが起きている。

1. B や RR の違う Run は、同じ経験数でも学習ステップが違う。Viewer で重ねると横にずれ、B512/RR1 の曲線は B256/RR1 の半分の長さに縮んで見える。
2. 同じ経験数の区間で比べるには、Run ごとに「経験数 1M あたりの学習ステップ＝1e6 × RR / B」で範囲を換算する必要がある。inspect_run も各指標の自分の軸でしか範囲を受け付けない。
3. 成績（経験数の軸）と診断（学習ステップの軸）を同じ横位置で読めない。

### ゴール

1. 学習ステップの軸で記録する指標を、Viewer で経験数の横軸でも描ける。
2. inspect_run で、学習ステップの軸の指標を経験数の範囲で集計できる。
3. 学習時の診断の測定間隔と費用は変えない（いつ測るかは今のまま学習ステップで決める）。
4. 横軸の種類が、Viewer のグラフ上で分かる。

### 非ゴール

- 測る中身・測定間隔の値（503 など）の変更。
- 測定そのものを経験数で数えて決めること（経験数あたりの測定密度をそろえる）。§6 で先送りにする。
- EMA の時定数を経験数基準にする設定（092 A17 で不採用）。
- target 更新の軸（940 の範囲）。
- 過去の Run の読み替え。対応が記録されていない（§6）。

## 2. 背景

092 の本線分析では、41a（B256/RR1）・41b（B512/RR1）・41d（B512/RR2）・42A/42D/42H（B256〜B1024、RR1〜RR2）の勾配ノイズを比べた。比較は全部スクリプトで、経験数の区間を各 Run の学習ステップへ換算してから区間平均を取った。Viewer では学習ステップの軸のまま重なるので、41b の勾配ノイズの曲線が半分の長さに描かれ、同じ横位置の成績と対応が取れなかった。

092 のグリル（A17）は、記録の軸を学習ステップのままにし、腕どうしは経験数でそろえた区間で比べると決めた。これは文書上の取り決めで、道具の側は変えていない。本 PRD はその道具の側を作る。

## 3. 現状（2026-10-18 に確認したこと）

| 事実 | 位置 |
|---|---|
| 記録のイベントは全部の軸の値（train / exp / update / learn / episode / sim）を持っている | [rl.hpp](../../core/anet-core/include/anet/rl.hpp) `:100-131`（`StepCounts`） |
| 記録する step は、定義の軸の 1 つだけ（`counts.GetByAxis(step_axis_)`） | [observers.cpp](../../core/anet-core/src/observers.cpp) `:925` |
| `$update_result` を学習ステップの軸で読むと、更新 1 回ごとに 1 記録（学習ステップを 1 ずつ進める）。他の軸では、そのイベントに入った更新の有限値を平均して 1 記録 | [observers.cpp](../../core/anet-core/src/observers.cpp) `:938-990` |
| 学習時の診断の「この更新で測る」は、購読行の `interval` を `vars_.learn_step` で数えて決める | [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) `:2280`、`:2381`、`:2428`、`:2500`、`:3279`、`:3284-3286` |
| 書き方の例: `@learn $learn_step $update_result interval:503`（勾配ノイズ）、`@learn $exp_step $ema interval:100`（IQN の診断） | [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) `:344-358`、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) `:1253-1260` |
| `metrics.defs` は指標ごとに `step_axis` を 1 つ持つ | [observers.cpp](../../core/anet-core/src/observers.cpp) `:1302`、`:1326` |
| Viewer は記録の `step` を 1 つだけ取り込み、横軸の種類を区別・表示しない | [MetricsIngestor.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java) `:380-386` |
| inspect_run の範囲は、各指標の自分の軸（runner と step_axis の組）で解釈する | [inspect_run.py](../../viewers/metrics-tools/inspect_run.py) `:318-340` |
| trace チャネルも記録の軸は 1 つ | [observers.cpp](../../core/anet-core/src/observers.cpp) `:869` |

記録の内訳（DropMerge 本線の 42A、60M、metrics.jsonl 2.22GB・2,306 万行）: 経験数の軸 80.4%、train_step 18.4%、episode 1.2%、**学習ステップの軸 0.1%（約 2.3 万行）**。1 行は平均 約 96 バイト。

`$exp_step $update_result` と書けば、記録だけなら経験数の軸に置ける（上の平均の経路）。ただし Learner は購読行の `interval` を学習ステップで数えるので、経験数の単位で書いた間隔は学習ステップとして解釈される。これが兼用の実害である。

## 4. 提案の骨子

### 4.1 案

| | A：学習ステップの軸の記録に経験数を添える | B：定義で「横軸」と「間隔の単位」を別に書く | C：対応表を 1 本記録して Viewer で描き直す |
|---|---|---|---|
| 中身 | 定義の軸が学習ステップの記録に、そのイベントの `exp_step` を別の欄で添える。Viewer と inspect_run はその欄で経験数の軸に置ける | `interval:` を数える軸と記録の軸を別の属性で書く（例: 記録は経験数、間隔は学習ステップ） | Runner が「学習ステップ ↔ 経験数」の対応を経験数の軸で 1 本記録し、Viewer と inspect_run が補間で描き直す |
| 性能 | 学習ステップの軸の記録は全体の 0.1% なので、増えるのは 1 Run あたり 1MB 程度 | 増えない | ほぼ増えない |
| 保守性 | 記録に座標が付くので、読む側は補間しない。jsonl の欄・キャッシュ・Viewer・inspect_run が変わる | 定義の文法と、学習側のイベントで経験数を使う変更が要る。軸は記録時に 1 つに決まり、表示の切り替えはできない | 変更は小さいが、補間という特別な経路が読む側に残る。対応の記録の間隔で精度が決まる |
| 表示の切り替え | できる（学習ステップでも経験数でも） | できない（記録時に決めた軸だけ） | できる |
| どちらで有利か | **保守性・表示の自由で A** | 文法の素直さでは B | **変更量で C** |

### 4.2 暫定の既定

案 A を推す。学習ステップの軸の記録だけに経験数を添えるので、ファイルはほとんど太らない。全記録に全座標を書く形は、1 行あたり 40〜50 バイト増え、jsonl が 5 割近く太るので取らない。

- **段階 1（Runner と inspect_run）:** 定義の軸が学習ステップの記録に `exp_step` の欄を足す。`metrics.defs` に「この指標は経験数の座標も持つ」ことを書く。inspect_run に範囲の軸を選ぶ指定を足し、学習ステップの軸の指標を経験数の範囲で集計できるようにする。
- **段階 2（Viewer）:** 横軸の切り替え（既定は定義の軸、経験数を選ぶと学習ステップの軸の指標も経験数で描く）と、横軸の名前の表示。キャッシュ（PRD041）に経験数の列を足して作り直す。

train_step の軸の記録（18.4%）は、経験数 = train_step × num_envs の一定の比なので、欄は足さず、Viewer が `metrics.defs` の num_envs で換算する案を未決に置く（§7.2）。

## 5. User Stories

1. 研究者として、B・RR の違う Run の勾配ノイズ（`49_`）を経験数の横軸で重ね、同じ経験数の位置で比べたい。
2. 研究者として、学習時の診断の測定間隔と費用はそのままで、表示だけを経験数にしたい。
3. 研究者として、inspect_run で学習ステップの軸の指標を「経験数 50〜70M」のような範囲で集計したい。
4. 研究者として、成績と診断を同じ横位置で読みたい。
5. 研究者として、グラフの横軸が何かを画面で確かめたい。

## 6. 作らないもの・先送り

- **測定を経験数で数えて決めること（案 B の後半）:** 経験数あたりの測定密度と EMA の時定数を腕どうしでそろえられるが、学習側の判断の仕組みに触れる。段階 1・2 で読めるようになってから、要るかを決める。
- **過去の Run の読み替え:** 対応が記録されていない。評価セッションのログ行（`learn_step=… exp_step=…`）から近似の対応を作る案はあるが、Viewer が設定やログから推し量らない方針（ADR 0029）と合わないので作らない。
- **全記録に全座標を書くこと:** §4.2 のとおり太りすぎる。
- **trace チャネルの座標:** 段階 1 では含めない（未決 §7.2）。
- **EMA の時定数を経験数基準にすること:** 092 A17 のとおり。

## 7. 確定事項と未決

### 7.1 暫定で置いた判断

| ID | 判断 |
|---|---|
| D1 | 推す案は A（学習ステップの軸の記録に経験数を添える） |
| D2 | 学習時の診断の「いつ測るか」は変えない。学習ステップで数える |
| D3 | 記録の `step` は今どおり定義の軸の値で、経験数は別の欄に書く |
| D4 | Viewer の横軸の既定は定義の軸。経験数への切り替えは利用者が選ぶ |
| D5 | 過去の Run は読み替えない |

### 7.2 未決

| 項目 | 選択肢 | 暫定の既定 |
|---|---|---|
| 添える座標 | 経験数だけ / 経験数と train_step | 経験数だけ |
| train_step の軸の記録 | 欄を足す / Viewer が num_envs で換算 | 換算（num_envs が Run の中で変わらないことの確認が要る） |
| Viewer の切り替えの単位 | 全体で 1 つ / グラフごと | 全体で 1 つ |
| キャッシュ（PRD041） | 列を足して作り直す / 別表 | 列を足して作り直す |
| inspect_run の指定 | 範囲の軸を選ぶ引数 / 範囲の書式に軸を含める | 引数 |
| trace チャネル | 段階 1 に含める / 含めない | 含めない |
| 案 B（間隔の軸を別に書く） | 段階 3 で作る / 作らない | 段階 1・2 の後に決める |
| ADR | 「記録は定義の軸に加えて経験数の座標を持てる」を ADR にするか | グリルで決める |

## 8. 成功の測り方と受入

1. 41a（B256/RR1）と 41b（B512/RR1）を同じ条件で回し直した短い Run で、Viewer の経験数の横軸に `49_agent_gradnoise` を重ねると、同じ経験数の位置がそろう（41b の曲線が半分に縮まない）。
2. inspect_run で、学習ステップの軸の指標を経験数の範囲で集計した値が、今のスクリプトの換算（1e6 × RR / B）と一致する。
3. 足した欄以外は、記録の tag・step・値が変更前と完全に一致する（同 seed・決定論設定の等価性）。
4. metrics.jsonl の増加は 1 Run あたり数 MB 以内（学習ステップの軸の記録数 × 欄の長さ）。
5. 学習時の診断の測定回数・費用は変わらない。

## 9. 依存と順序

- 段階 1 は Runner と inspect_run だけで完結する。段階 2 は Viewer（089・090 の後）とキャッシュ（041）に依存する。
- 940（target 更新の軸）とは独立。ただ「何の軸で数えるか」を設定で選ぶ考え方は共通なので、案 B を作るときは揃える。

## 10. 参照

- [092_gradient_noise_scale_10prd.md](done/092_gradient_noise_scale_10prd.md) §7.4、§9.1 A17・A19
- [093_attention_entropy_diagnostics_10prd.md](done/093_attention_entropy_diagnostics_10prd.md)
- [940_target_update_step_axis_10prd.md](940_target_update_step_axis_10prd.md)
- [done/053_interval_gate_perf_ema_10prd.md](done/053_interval_gate_perf_ema_10prd.md)
- [done/054_inspect_run_10prd.md](done/054_inspect_run_10prd.md)
- [ADR 0029](../adr/0029-analysis-metadata-emitted-by-runner.md)
