# 評価 session の step trace を Run の記録に残す PRD

> 2026-10-29 起票（暫定・999。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 起点: qbert の評価（合算の checkpoint、Greedy）を Evaluation Trace で見たところ、最後でない残機を失っても価値（`V_s`）がほとんど下がらなかった（3 ゲーム）。これを評価 session の 200 ゲームで数えたいが、評価 session の step ごとの値を Run の記録に残す口が無い。材料は 2026-10-29 の会話と、PRD106 の再グリルへの引継ぎ文書 `.scratch/prd106/handoff-grill-20261029.md` の §9。
> 関連: [PRD069](done/069_metrics_trace_channel_10prd.md)（trace チャネル。§8 に「trace の `@train`（lane ごとの毎 step 記録）」の待機中のゲート）、[PRD094](094_evalpanel_step_history_10prd.md)（Evaluation Trace。step trace の生成器と列の書き方）、[PRD081](081_evalpanel_episode_trace_10prd.md)（EvalPanel のゲームを trace へ出す。PRD094 から「step trace を trace チャネルへ出す段」を引き継いだが、改訂はまだ）、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)・[ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)（trace の行の契約）、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)（Evaluation Trace の列はメトリクスの定義と分けて宣言し、遷移の時点に合う target だけを読む）、[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)（step の診断値は Agent の step 部品が計算する）、[PRD106](106_temporl_skip_forward_noise_key_10prd.md)。

## 1. 目的と問題

評価 session（eval schedule の 1 回の発火で行う評価の単位。採用 episode N 本を完走させる。[CONTEXT.md](../../CONTEXT.md) の「評価セッション」）の中の 1 遷移ずつの値を、Run の記録として残す手段が無い。

2026-10-29、qbert のサンプル効率優先の 50M の合算の checkpoint を、`[greedy]` の評価 session と同じ条件の EvalPanel で見ると、次のことが分かった（引継ぎ文書 §9）。

- 最後でない残機を失った前後で `V_s` がほとんど変わらない（3 ゲーム・4 回で −0.4〜+0.0）。最後の 1 機のときだけ `V_s` が 0 へ崩れる。
- 飛び降り（最上段から外へ出る行動）が、価値の上で最善の行動として選ばれた例があった。
- 評価 session の記録（ゲームの終わりの 1 行）では、合算の Greedy の 200 ゲームのうち 45 本が 5 面で終わり、そのうち 43 本が 14,550 点・ゲームの長さ 1,063〜1,066 step にそろっていた（[PRD106](106_temporl_skip_forward_noise_key_10prd.md) の再グリルの材料）。

これらが評価 session の多くのゲームで成り立つか（残機を失った step の前後の `V_s`、失う前の判断の行動とスキップ長、14,550 点で終わるゲームの道筋がそろっているか）を数えたい。画面の Evaluation Trace では 1 ゲームずつしか見られず、記録にも残らない。

## 2. 現状（2026-10-29 に確認したこと）

1. trace チャネルの event は `@episode_end` だけで、1 つの定義の target は 1 つ（`$env_step`・`$env`・`$runner`・`$agent`・`$actor` のどれか）である（[140 可観測性（設計文書）](../design/140_observability.jp.md) 247 行）。評価 session の scalar は `@session_end` か、`@train $action_info` に限られる（同 229 行）。`@train $action_info` で読めるのはスキップの判断率・長さ・スキップ長ごとの価値と、スケジュールの値（γ など）だけで（[dqn_based_agent.cpp:1236-1245](../../core/anet-core/src/dqn_based_agent.cpp)）、scalar なので lane をまとめた値になる。
2. EvalRunner は、評価 session の中でも step ごとに `@train` 系の event を出し、そこには EvalRunner 自身の counts が載る（[140 可観測性（設計文書）](../design/140_observability.jp.md) 206 行）。
3. Evaluation Trace（EvalPanel の pane）は、`StepTraceBuilder` が `TrainEvent` から lane・step ごとの行（step trace）を組み、`app.eval_trace_panel.columns.[n]` で宣言した列を読む（[metrics.hpp:43-79](../../core/anet-core/include/anet/metrics.hpp)）。列の target は `$runner`・`$actor`・`$env_step`・`$action_info`・`$agent_step` で、1 列に 1 target・1 key（[metrics.cpp:120-143](../../core/anet-core/src/metrics.cpp)）。列の宣言の置き場は `app.eval_trace_panel.columns.` に固定されている（同 100-108 行）。`$agent_step` の値（`q_sa`・`state_value`・`approx_priority`）は Agent の step 部品（`AgentStepDiagnosticsProvider`）が作り、あとで決まる値は待ってから行を確定する（[rl.hpp:777-790](../../core/anet-core/include/anet/rl.hpp)、[dqn_based_agent.cpp:496-497](../../core/anet-core/src/dqn_based_agent.cpp)）。
4. Evaluation Trace の表は画面に出すだけで、書き出しは無く、最大行数は `app.eval_trace_panel.max_rows`（既定 1,000）である（[EvalTracePanel.cpp:52-54](../../apps/runner/src/EvalTracePanel.cpp)）。
5. PRD069 §8 は、「trace の `@train` / `@learn`（per-step lane fan-out）」を待機中のゲートとし、開けるきっかけを「932 forensic、または lane ごとの毎 step 記録が要るとき」、コストを「検証行列の分岐 + observer 1 種」としている（[PRD069](done/069_metrics_trace_channel_10prd.md) §8）。本書の需要は後者に当たる。
6. PRD094 は「step trace を trace チャネルへ出す段」を PRD081 へ移し、そこで決めることを「trace 行の step 座標」「metrics の DSL への `$agent_step` の追加」「あとで決まる列を待って 1 行を書くこと」とした（[PRD094](094_evalpanel_step_history_10prd.md) §7 の 2）。PRD081 のその改訂はまだ無い。PRD081 の対象は EvalPanel（鏡写しインスタンス）で、評価 session ではない。
7. 用語: 「step trace」は「Runner の遷移 1 つを 1 件とし…Evaluation Trace の 1 行で、trace へ出すときもこの単位で 1 レコードにする」と定義済み（[CONTEXT.md](../../CONTEXT.md) の「step trace」）。「trace」は「1 件の個体（lane のエピソード完了など）を 1 レコードとして…記帳するメトリクスチャネル」（同「trace」）。

## 3. ゴール・非ゴール・成功の測り方

**ゴール**

- G1: 評価 session ごとに、宣言したときだけ、採用 episode の lane・step ごとの step trace を Run の記録（metrics.jsonl）に 1 行ずつ残せる。宣言しなければ、今と同一（observer も step 部品も作らない）。
- G2: 列は Evaluation Trace と同じ書き方・同じ target（`$runner`・`$actor`・`$env_step`・`$action_info`・`$agent_step`）で宣言でき、同じ値になる。
- G3: 残した行を `inspect_run.py` で CSV に落とし、§1 の問い（残機を失った前後の `V_s`、失う前の判断、14,550 点で終わるゲームの道筋）を、評価 session の全ゲームで数えられる。

**非ゴール**

- 学習側（`$train`）の step ごとの記録。消える案は学習中の全 step を残す案で、本当の制約ではない（同じ observer で付けられる）が、行数が学習の長さに比例して桁違いに増え、今回の問いに要らない。
- EvalPanel（鏡写しインスタンス）の step trace を記録に残すこと。PRD081 の改訂の関心である（§9 の 1）。消える案は本書で EvalPanel も扱う案で、本当の制約ではないが、EvalPanel の step 座標の問題（[PRD094](094_evalpanel_step_history_10prd.md) §7 の 2）を含むので分ける。
- 行を間引くこと（何 step に 1 行、何ゲームに 1 本）。消える案は間引きの設定を足す案で、本当の制約ではないが、評価専用の Run で使う前提では行数が許容の範囲（§5.5）なので、要ると分かるまで作らない（§8）。
- Metrics Viewer で step trace を表示すること。消える案は Viewer に step trace の表を足す案で、本当の制約ではないが、G3 の数え方は CSV で足りる。
- trace チャネルの今の 1 行 DSL（`metrics.trace.[<tag>] = …`）の文法を変えること（§4 の D1）。消える案は 1 行 DSL に `@train` と複数の target を足す案で、本当の制約ではない（クリーンブレークで変えられる）が、列の書き方がすでに同じことを表している（§8）。

**成功の測り方**

- G1・G2: §7 の受入 1〜5 が通る。
- G3: 合算の checkpoint の Greedy の評価専用 Run（200 ゲーム）で行を残し、残機を失った step（`lives` が減った行）を全ゲームで取り出して、前後の `state_value` の差の分布と、失う前の最後の判断の行動・スキップ長を数えられる。

## 4. 確定事項（暫定の既定。§9 で見直す）

- **D1. 宣言は `metrics.step_trace.[<tag>]` の下に、評価 session（`scope`）と、Evaluation Trace と同じ書き方の列（`columns.[<n>]`）を置く。** 理由: 1 行に複数の target の値（行動を選んだ時点の `$action_info`、遷移のあとの `$env_step`、あとで決まる `$agent_step`）を並べる必要がある。trace の 1 行 DSL は 1 定義 1 target なので表せない（§2 の 1）。Evaluation Trace の列の書き方は、まさにこの形を表していて、行の組み立ても `StepTraceBuilder` がすでに持つ（§2 の 3）。記録の宣言を `app.eval_trace_panel.*`（画面の設定）と分けるのは、ADR 0057 の「画面の列とメトリクスの定義を分ける」に合わせるためである。依存: §2 の 1・3。
- **D2. 行は trace チャネルの行（`type` が `trace`）として書く。** 形は今の trace の行と同じ（`tag`・`step`・`lane`・`timestamp`・`data`）。`data` には各列の値を列のラベルで入れ、あわせて遷移の `reward`・`done`・`truncated` を入れる。理由: step trace は「trace へ出すときもこの単位で 1 レコード」と用語で決まっている（§2 の 7）。trace の行として書けば、`inspect_run.py trace-csv` などの読み手と、ADR 0037・0047 の行の契約を使い回せる。定義の記録（`metrics.trace.defs`）には、step trace の tag について、event を `train`、列ごとの target・key・ラベルを載せる。既存の 1 行 DSL の trace の定義の形と値は変えない（列の並びは step trace の定義にだけ載る）。依存: D1。既存の ADR との関係: [ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md) は「評価の trace は `@episode_end` に束縛する」と決めている。本書はこれを、step trace の宣言（D1 の別の名前空間）に限って `@train` 系の event へ広げる。1 行 DSL の trace は `@episode_end` のままにする。ADR 0037 の「既存の読み手は trace 行を捨てるだけで壊れない」はそのまま使う。この広げ方を ADR にするかは §9 の 8。
- **D3. 宣言できる scope は評価 session（`$eval.[<name>]`）だけ。** `$train` を書けば構築時にエラーにする。理由: 非ゴール（学習側の記録）。評価 session なら、行数は採用 episode 数とゲームの長さで上が決まる。
- **D4. 宣言が無ければ、observer も step 部品も作らない。** `$agent_step` の列が無ければ、step 部品も作らない。理由: G1（今と同一）。step 部品の計算は、Evaluation Trace と同じく宣言した場合だけの費用にする（ADR 0058）。
- **D5. step 座標は、その event に載る counts（評価 session の `@train` 系 event では EvalRunner 自身の counts。§2 の 2）の `exp_step` とする。** ゲームの中の位置は、列に `$runner step_in_episode` を宣言して残す。理由: 評価専用の Run（学習を止めた Run）では学習側の counts が動かないので、EvalRunner の counts だけが行を並べる座標になる。学習中の Run の評価 session に宣言したときに、学習側の counts（[CONTEXT.md](../../CONTEXT.md) の「学習側 counts」）とどう並べるかは §9 の 3。
- **D6. 採用 episode の行だけを書く。** 採用権の無い episode（session の途中で完了したものも、session の終わりに途中だったものも）の行は書かない。採用かどうかは、行の遷移が属する episode について判定する。採用は episode の開始時に決まる（[CONTEXT.md](../../CONTEXT.md) の「採用エピソード」）が、今は評価 session の Env が group ごとの採用の旗を外へ出していない（[env.hpp:188](../../core/anet-core/include/anet/env.hpp) の非公開の欄。公開されているのは直前 Step で完了した採用 group の一覧だけ。同 167-168 行）。また Env は、採用 episode が終わった Step の中でその group の旗を下ろし、採用権が残っていれば、次の episode に旗を立て直す（[env.cpp:425-438](../../core/anet-core/src/env.cpp)）。そのため終わりの遷移のあとで旗を読むと、次の episode の旗を読んでしまう。そこで `EvalSessionEnv` に「その lane が属する episode group の、今の episode が採用か」を返す読み口を足し、observer は episode の最初の遷移でその episode が採用かを読み、その episode の終わりの遷移（`done` か `truncated` が真の遷移）まで同じ判定を使う。理由: 評価 session の記録（`@session_end` の scalar、`@episode_end` の trace）はどれも採用 episode だけを数えるので、step trace も同じ範囲にそろう。読み手が突き合わせて除く手間が無く、採用 episode はどれも終わりまで進むので、どの episode の行も `done` か `truncated` が真の行で終わる（§7 の受入 3）。依存: CONTEXT.md の「採用エピソード」の定義と、Env が旗を下ろす時機（env.cpp:425-438）。
- **D7. あとで決まる列（`$agent_step` の一部）は、値が決まってから行を書く。session の終わりにまだ決まらない列は NaN で確定させて書く（JSON では `null`）。** 理由: step 部品の契約が「pending の鍵は値が成立しなかった場合も NaN で確定を返し、必ず待機を終える」である（§2 の 3）。
- **D8. 予定の無い（dormant の）評価 session への step trace の宣言は、保持も結び付けもせず、何も書かない。定義の記録にも載せない。構築時に、その tag について WARN を 1 回出す。** これは [ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) の「`RunManager` は dormant タグを参照する trace observer を捨てずに保持し、同じタグで `CreateEvalRunner` された Runner（鏡写しインスタンス）へ結び付け、保持した trace の定義を起動時の定義レコードに載せる」とは違う扱いである。理由: dormant の評価 session に結び付くのは EvalPanel（鏡写しインスタンス）で、その step trace は非ゴールであり、PRD081 の改訂で決める（§9 の 1）。ここで ADR 0047 に従うと、EvalPanel の step 座標の問題（[PRD094](094_evalpanel_step_history_10prd.md) §7 の 2）を決めないまま行が書かれる。WARN を出すのは、宣言が効かないことを黙って見過ごさないため。1 行 DSL の trace の dormant の扱い（ADR 0047）は変えない。依存: 非ゴール（EvalPanel）、§9 の 1。

## 5. 実装契約

確定事項と本節には実装の語（内部の型、入れ物の型、添字の決め方）を書かない。公開 API 名・設定キー・メトリクス名・クラス一覧の名前は書く。

### 5.1 設定キー

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `metrics.step_trace.[<tag>].scope` | `$eval.[<name>]`（宣言済みの評価 session） | なし（宣言するときは必須） | 行を残す評価 session |
| `metrics.step_trace.[<tag>].columns.[<n>]` | Evaluation Trace の列と同じ書き方（`<target> <key> [label:<名前>] [display:<表示>]`。`<n>` は 1 以上の整数） | なし（1 つ以上必須） | 1 列 |

- `<tag>` の長さは検証しない。書き方の決まりとして、[実装 品質確認観点](../quality/impl.md) の「設定ファイル」の metrics の tag の文字数（群名と `/` を含めて 40 文字以内。`metrics.scalar.[tag]` を対象にした決まりで、Viewer の一覧で見切れないため）を、step trace の tag にも当てる。
- `<tag>` は、同じ名前の scalar とは別系列（`(type, tag)` で区別。PRD069 D13）。同じ名前の trace の定義（1 行 DSL）とは重ねられない。どちらも `type` が `trace` になるためで、評価 session が発火するかどうかによらず、構築時にエラーにする。
- `scope` が `$train`、宣言されていない評価 session、または無い場合は、構築時にキーと値を示して止める。
- 列の検証は Evaluation Trace の列と同じ（番号の重複、target が 1 つ、key が 1 つ、ラベルの書式）。違反は構築時に止める。加えて、step trace では列のラベルどうしの重複（ラベルを省いて key がラベルになる場合を含む）と、ラベルが `reward`・`done`・`truncated` と重なることを、構築時にエラーにする（`data` の中で値が上書きされるのを防ぐ）。`display:` は記録には効かない（画面の表示用）ので、書かれていれば検証して使わず、その列について WARN を 1 回出す（§9 の 6）。
- 宣言した評価 session に予定が無い（dormant の）ときは、D8 のとおり何も書かず、定義の記録にも載せず、WARN を 1 回出す（ADR 0047 とは違う扱い）。

設定例（コロン記法）:

```
# 合算の checkpoint の Greedy の評価専用 Run（200 ゲーム）で、毎 step の行を残す
run.@stepdiag : metrics.step_trace.[53_evalg/step].scope = $eval.[greedy]
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[110] = $runner step_in_episode label:ep_step
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[120] = $action_info action
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[130] = $action_info skip_decision label:decision
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[140] = $action_info skip_length label:skip
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[210] = $env_step game_score
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[220] = $env_step lives
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[230] = $env_step ram_metric.[1] label:round
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[310] = $agent_step q_sa
run.@stepdiag : metrics.step_trace.[53_evalg/step].columns.[320] = $agent_step state_value label:v_s
```

行の例:

```json
{"type":"trace","tag":"53_evalg/step","step":1234567,"lane":3,"timestamp":"…","data":{"ep_step":1063,"action":2,"decision":1,"skip":4,"game_score":14525,"lives":1,"round":4,"q_sa":1.509,"v_s":1.509,"reward":0,"done":false,"truncated":false}}
```

### 5.2 判定の網羅

| 宣言 | `scope` | 列 | 答え |
|---|---|---|---|
| 無い | — | — | 何も作らず、何も書かない（今と同一） |
| ある | 無い、`$train`、または宣言されていない評価 session | — | エラー |
| ある | 宣言済みの評価 session | 0 本、または列の規則に違反 | エラー |
| ある | どれでも | 同じ tag の trace の定義（1 行 DSL）がある | エラー（評価 session の発火によらない） |
| ある | どれでも | 列のラベルが重なる、または `reward`・`done`・`truncated` と重なる | エラー |
| ある | 宣言済みの評価 session（予定が無い。dormant） | 正しい | 何も書かず、定義の記録にも載せない。WARN を 1 回（D8） |
| ある | 宣言済みの評価 session（発火する） | 正しい（`$agent_step` 無し） | step 部品を作らずに、採用 episode の lane・step ごとに 1 行を書く（D6） |
| ある | 同上 | 正しい（`$agent_step` あり） | step 部品を作り、採用 episode の行を値が決まった順に書く。session の終わりに決まらない値は NaN（`null`） |

### 5.3 行の中身

- 列の値の読み方は Evaluation Trace と同じ（ADR 0057 の「遷移の時点に合う target だけを読む」）。
- `data` のキーは列のラベル。列のほかに `reward`・`done`・`truncated` を入れる（Evaluation Trace の行の固定の値。§2 の 3）。ラベルの重なりは §5.1 のとおり構築時にエラー。
- 未知のキーで値が取れないときは、今の trace と同じく tag・key・lane・target を示して止める。NaN・±Inf はキーを残して `null` にする（140 の §6.x の trace の行の規則）。
- 行の順は、確定した順（`$agent_step` の値を待つ行は、あとから書かれる）。読み手は `step`・`lane` で並べ直す。

### 5.4 移行対象と等価性

- 設定: 既定の設定は変えない（宣言は使う Run の設定に書く）。
- 文書: [140 可観測性（設計文書）](../design/140_observability.jp.md) の §6.x（trace チャネル）に、step trace の宣言と行の形を足す。[030 Run 分析のユーザーガイド](../design/030_user_guide_analysis.jp.md) §6.9（`trace-csv`。469-486 行）の「1 行が 1 エピソード」「列順は `metrics.trace.defs` の `keys` が正本」を、step trace の tag では「1 行が採用 episode の遷移 1 つ」「列順は列の番号の順」と書き分ける。英語版（`140_observability.en.md`、`030_user_guide_analysis.en.md` の同じ節）は翻訳の工程が追う。「event は `@episode_end` のみ」の文は、1 行 DSL の trace についての記述として残し、step trace の宣言は別の段落にする。CONTEXT.md の「step trace」「trace」は今の定義のまま合う（§2 の 7）。
- 読み手: `inspect_run.py` の `trace-csv` が、step trace の行（列ごとの target を持つ定義）を読めるようにする。Metrics Viewer のキャッシュの取り込みは、trace の行を今と同じ扱いで飛ばすか取り込み、step trace の行で止まらない。
- 互換の仕組みは作らない（新しい宣言なので、旧の形が無い）。
- 用語: [CONTEXT.md](../../CONTEXT.md) の「dormant」と「鏡写しインスタンス」は、trace は dormant のとき鏡写しインスタンスへ結び付くと書いている。D8 では step trace は結び付かないので、実装と同じ変更で、両項目に step trace の扱いを書き足す。起票の時点で直さないのは、D8 が §9 の 1（PRD081 との関係）の結論で変わりうるため。
- 変えないもの: 過去の Run の記録（metrics.jsonl・定義の記録）、実験記録、`app.eval_trace_panel.*`（画面の列）の書き方と挙動、既存の 1 行 DSL の trace の定義の記録の形と値。

### 5.5 性能

- 宣言が無ければ、費用は 0（D4）。
- 宣言したときの行数の目安: qbert の合算の Greedy（200 ゲーム、ゲームの長さの平均 4,528 step）で約 91 万行。1 行を 250 バイトとすると約 230 MB。評価 session の 16 lane で、1 step に 16 行を書く。
- `$agent_step` の列があると、step 部品の計算が step ごとに入る（Evaluation Trace と同じ費用）。
- 学習中の Run の評価 session に宣言すると、評価のたびに同じ量が増える。評価専用の Run で使うことを前提にし、性能の受入は置かない。

### 5.6 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `StepTraceColumnSpec` | 変更 | 列の宣言の置き場（キーの前置き）を受け取り、Evaluation Trace と step trace の記録で同じ列の文法を検証する | 可視化・メトリクス（metrics） |
| `StepTraceBuilder` | 変更 | 評価 session の終わりに、値を待っている行を NaN で確定させて渡す（今の動きで足りれば変更なし） | 可視化・メトリクス（metrics） |
| `MetricsLogStepTraceObserver` | 新規 | 評価 session の `@train` 系 event から step trace を組み、episode の最初の遷移で読んだ採用の判定をその episode の終わりまで使って、採用 episode の確定した行を trace の行として書く（D6） | 可視化・メトリクス（observers） |
| `ObserverFactory` | 変更 | `metrics.step_trace.*` を読み、検証して observer を組む | 可視化・メトリクス（observers） |
| `RunManager` のメトリクス配線（`MetricsWiring`） | 変更 | 予定のある評価 session へ step trace の observer を付け、定義を `metrics.trace.defs` に記録する。dormant の評価 session への宣言は結び付けず、WARN を 1 回出す（D8） | 強化学習 共通基盤（trainer） |
| `EvalSessionEnv` | 変更 | lane が属する episode group の、今の episode が採用かを返す読み口を足す（D6） | Env 共通基盤（env） |
| `inspect_run.py` | 変更 | `trace-csv` で step trace の行（列ごとの target の定義）を CSV へ落とす | メトリクスの読み手（viewers/metrics-tools） |

## 6. 細部（暫定で置いた既定）

1. 宣言のキーの名前は `metrics.step_trace.[<tag>]`（理由: `metrics.trace.[<tag>]` の 1 行 DSL と並べて読め、行は trace になることを名前で示す）。
2. tag の例は、評価 session の既存の群名に `/step` を付ける（例 `53_evalg/step`）（理由: 同じ session の `@episode_end` の trace（`53_evalg/episode`）と並べて探せる）。
3. 列の番号の規則は Evaluation Trace と同じ（百の位が分類…の 3 桁は推奨。番号は並べる順だけを決める）。
4. 行の `data` に入れる固定の値は `reward`・`done`・`truncated` の 3 つ（理由: Evaluation Trace の行と同じ）。

## 7. 検証と受入

**使う場面（G3 の確かめ）**: 合算の checkpoint（`run_20261027-004945_a5_qb_aln_t98mrg_s1`）を、`[greedy]` の評価 session で 200 ゲーム評価する評価専用の Run に、§5.1 の設定例の宣言を足して流す。`trace-csv` で行を落とし、次を数える。

- 残機を失った行（同じ lane・同じ episode の中で `lives` が前の行より減った行）の数と、その前後（例: 5 step 前と 5 step 後）の `v_s` の差の分布。最後の残機（`lives` が 0 になる行）とそれ以外に分ける。
- 失う前の最後の判断（`decision` が 1 の行）の行動とスキップ長、その行の `q_sa` と、その後の継続中の行の `v_s − q_sa` の最大。
- 14,550 点で終わるゲームの、残機を失った面・ep_step がそろっているか。

**受入（外から観測できるもの）**

1. 宣言しない設定で今と同一: 実装の前と後のビルドで、[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md) の 6 構成（同じ短縮条件と比べ方）の metrics checksum が一致する。あわせて、既存の trace（Atari の `53_evalg/episode` など、1 行 DSL）を宣言した短い Run の定義の記録（metrics.jsonl の `metrics.trace.defs` の行と、その写しの `json/metrics.trace.defs.json`）が、前後のビルドで同一である。
2. 単体テスト:
   - §5.2 の各行（エラーになる組み合わせ、dormant、`$agent_step` の有無）がそのとおりになる。
   - 同じ列の宣言で、Evaluation Trace の行（`StepTraceBuilder` が組む行）と、記録に書かれた行の値が、同じ遷移について一致する。
   - 採用 episode の終わりの遷移（`done` か `truncated` が真）の行が書かれる。採用権が尽きた lane の最後の採用 episode の終わりの行も書かれる。採用権の無い episode の行（session の途中で完了したものと、session の終わりに途中だったもの）は書かれない。
   - `$agent_step` の値を待つ行は値が決まってから書かれ、session の終わりにまだ決まらない値は `null` で書かれる。
3. 評価 session が 2 回発火する短い評価専用の Run（Atari、1 session 2 ゲーム程度）で、`$runner step_in_episode` の列を宣言する。lane ごとに Run 全体の行を `step` の順に並べたとき、その列が 1 から 1 ずつ増え、`done` か `truncated` が真の行の次で 1 に戻る（抜けも重複も無い。採用 episode はどれも終わりまで進むので、session の境目でも成り立つ）。`done` か `truncated` が真の行の数は、2 つの session の採用 episode の数の合計（同じ Run の `@episode_end` の trace の行数）と一致する。各行の `step`・`lane`・`data` のキーが定義のとおりである。定義の記録（`metrics.trace.defs`）に、その tag が event `train` と列ごとの target・key・ラベルで載る。
4. `inspect_run.py trace-csv` が、3 の Run の step trace の tag を CSV に落とせる。Metrics Viewer のキャッシュの取り込みが、3 の Run で止まらない。
5. G3 の Run（合算、200 ゲーム）が完走し、`trace-csv` で落とした行から、残機を失った行が全ゲームについて取り出せる（数は Run の結果として記録し、合否にはしない）。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| 学習側（`$train`）の step trace | 範囲外 | 非ゴール。再開条件: 学習中の行動を step ごとに追う必要が出たとき（行数の上限の仕組みと一緒に考える） |
| 行の間引き（何 step に 1 行、何ゲームに 1 本） | 先送り | 評価専用の Run なら行数が許容の範囲（§5.5）。再開条件: 行数が読み手や Viewer の扱える量を超えたとき |
| trace の 1 行 DSL に `@train` と複数の target を足す形 | 不採用 | 1 定義 1 target の今の文法と検証（PRD069）を変えることになる。複数 target の列は Evaluation Trace の列の書き方がすでに表している（D1） |
| `app.eval_trace_panel.columns.*`（画面の列）をそのまま記録に使う形 | 不採用 | 画面の設定と記録の宣言が結び付き、画面の列を変えると記録が変わる（ADR 0057 の分け方に反する） |
| 残機を失った step だけを書く形 | 不採用 | 残機は Atari 固有の値で、汎用の trace の event にすると Env 固有の判定を持ち込む。毎 step の行から読み手が取り出せる |
| Metrics Viewer での表示 | 範囲外 | 非ゴール。再開条件: step trace を Viewer で見たい需要が出たとき |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | PRD081（EvalPanel の step trace を trace チャネルへ出す段）との関係。本書で作る口を EvalPanel にも使うか、本書を PRD081 の改訂に統合するか | 本書は評価 session の口を作り、PRD081 の改訂は EvalPanel をこの口の購読先にする形で取り込む（step 座標の問題は PRD081 で決める） |
| 2 | 宣言の形（`metrics.step_trace.[<tag>]` の別の名前空間か、`metrics.trace.[<tag>]` の下に列の形を足すか） | 別の名前空間（D1・§6 の 1） |
| 3 | 学習中の Run の評価 session に宣言したときの step 座標（EvalRunner 自身の counts か、学習側 counts か） | EvalRunner 自身の counts（D5）。学習中の Run で使う需要が出たら見直す |
| 4 | 採用 episode の行だけを書くために `EvalSessionEnv` に採用の旗の読み口を足すこと（D6）。足さずに全部を書いて読み手が除く形との比べ | 読み口を足して採用 episode の行だけを書く（D6）。足さない形は、読み手が `@episode_end` の trace と突き合わせる手間が残り、session の境目で途中の episode の行が `done` の行なしに切れる |
| 5 | 行の書き出しの重さ（1 step に 16 行の JSON）が評価専用の Run の実時間をどれだけ延ばすか | 測らずに置く。G3 の Run の実時間を、宣言なしの同じ評価と比べて記録する |
| 6 | `display:` を記録の宣言で受け付けるか | 検証して使わず、列ごとに WARN を 1 回出す（Evaluation Trace の列の書き方をそのまま写せるようにしつつ、効かない指定を黙って無視しない） |
| 7 | step trace の定義を `metrics.trace.defs` に載せるか、step trace 用の別の定義の記録にするか | `metrics.trace.defs` に、step trace の tag についてだけ列の並びを載せる。既存の 1 行 DSL の trace の定義の形と値は変えない（D2、受入 1）。読み手（`inspect_run`・Viewer）への影響は実装計画で確かめる |
| 8 | ADR 0037 の「評価の trace は `@episode_end` に束縛する」を step trace の宣言に限って広げることを、ADR にするか | グリルで D1・D2 が決まったら、ADR 0037 の先頭に注記を足し、新しい ADR を起こす（D1・D2 は戻しにくく、文脈が無いと意外で、別案との比較の結果なので、ADR の 3 条件に当たる見込み） |
| 9 | 1 step で終わる採用 episode の行（採用権が尽きた lane で、episode が最初の遷移で終わると、D6 の「最初の遷移で採用を読む」時点では Env がすでに旗を下ろしていて、その episode の行が書かれない。[env.cpp:425-438](../../core/anet-core/src/env.cpp)） | 実装計画で採用の判定を読む時点を決める（例: 直前 Step で完了した採用 group の一覧（[env.hpp:167-168](../../core/anet-core/include/anet/env.hpp)）とあわせて判定する）。受入 2 の単体テストにこの場合を含める（2026-10-29 のレビューの再確認ポイントから） |
| 10 | 評価 session が排水でキャンセルされたときの、途中だった採用 episode の行（`done` か `truncated` の行なしに切れる。D6 の「どの episode の行も終わりの行で終わる」の例外） | D6 の例外として扱い、行はそのまま残す（切れた episode は、同じ session の `@episode_end` の trace に行が無いことで見分ける）。受入 3 の期待値はこの場合も崩れない（2026-10-29 のレビューの再確認ポイントから） |

## 10. 依存と順序

- [PRD094](094_evalpanel_step_history_10prd.md)（`StepTraceBuilder`・列の書き方・`$agent_step`）と [PRD069](done/069_metrics_trace_channel_10prd.md)（trace チャネル）の実装に乗る。
- [PRD081](081_evalpanel_episode_trace_10prd.md) の改訂（§9 の 1）と順序が絡む。どちらを先にしても、step trace を書く口は 1 つにする。
- [PRD106](106_temporl_skip_forward_noise_key_10prd.md) のステージ 0 の読み（Greedy の決まった失敗の扱い）は、本書の G3 の結果を材料にできるが、本書を待たない。

## 11. 参照

- [PRD069 trace チャネル](done/069_metrics_trace_channel_10prd.md)、[PRD094 Evaluation Trace](094_evalpanel_step_history_10prd.md)、[PRD081 EvalPanel のゲームを trace へ出す](081_evalpanel_episode_trace_10prd.md)、[PRD106](106_temporl_skip_forward_noise_key_10prd.md)
- [ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)、[ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)
- [140 可観測性（設計文書）](../design/140_observability.jp.md)
- [317 可視化・メトリクス（wiki）](../wiki/categories/317_metrics.md)、[110 RunnerApp（wiki）](../wiki/apps/110_runner_app.md)
