# メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す

> 起点: 2026-08-25、`42_env/10_game_score_mean` が「128 env の平均」ではなく「その step で完了した env の平均」であると判明したこと。2026-09-05、PRD 060 の評価セッションで eval 側の個体分布が復元不能になり顕在化。
> グリル: 2026-09-05（grill-with-docs。決定 D1〜D16、再点検と6項目の最終簡素化パス済み）。旧題「record チャネル」から改名。
> 関連: [060 評価セッション](done/060_eval_batch_episodes_10prd.md)、[054 inspect_run / `metrics.defs`](done/054_inspect_run_10prd.md)、[912 background eval snapshot 順序](912_background_eval_snapshot_ordering_10prd.md)、[932 episode forensics](932_episode_forensics_10prd.md)、[ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)、[ADR 0029](../adr/0029-analysis-metadata-emitted-by-runner.md)、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)。

## 1. 背景

スカラーメトリクスの行は `{step, tag, value}` の 3 要素しかなく、**主体（誰の値か）を書く欄が無い**。バッチ集約はこの欄が無いことを `mean.` / `max.` / `min.` / `std.` の prefix で埋めているが、集約した時点で個体は失われる。

train 側では `mean.game_score` の分母が「その step でゲームを終えた env の数」になり、Breakout / 128 env では 97% が単独完了だったため偶然エピソード素点の列として使えていた。PRD 060 で eval が N 本の評価セッションになった結果、eval1 / eval2 の `mean.game_score` は N 本を 1 点に畳んだ値になり、**p10 / 閾値越え率 / score×len の同時分布が eval 側で復元できなくなった**。spatial exploration 系の Run では eval1 が唯一の可換軸（train 生スコアは ε ラダー混在）なので、これは Atari-5 の横断比較を直撃する。

本 PRD は、統計（scalar）とは別に **「1 件 = 1 レコード」を識別子付きで `metrics.jsonl` へ出すチャネル（trace）** を設け、その出力側（行契約 / 書き口 / 設定 DSL / 定義レコード / eval の producer）を確定する。

## 2. 現行コードで確定している事実（2026-09-05 時点）

### F1. エピソード確定値は NaN ゲート、集約は NaN を除外する

`AtariEnv::GetScalar` は `game_score` / `game_score.ge.[N]` / `game_len` / `game_frames` / `hns57` / `hns49` を `completion_available_` が立っている間だけ返し、それ以外は NaN（[AtariEnv.cpp:617-645](../../core/envs/atari1/src/AtariEnv.cpp:617)）。`completion_available_` は実 game over / truncation で立ち（[:576](../../core/envs/atari1/src/AtariEnv.cpp:576)、life-loss 継続 reset は [:501](../../core/envs/atari1/src/AtariEnv.cpp:501)）、**次の `Step()` 先頭で落ちる**（[:531](../../core/envs/atari1/src/AtariEnv.cpp:531)）。`lives` だけはゲート無しの現在値。

`DiscreteBatchEnvBase::GetScalar(key, index)` は `index >= 0` で個別 lane へ素通し（key は prefix 無しが契約）、`index < 0` は `mean.|max.|min.|std.` prefix 必須で lane 全体を Welford 集約する（[env.cpp:592-616](../../core/anet-core/src/env.cpp:592)。prefix 無指定は fail-fast）。集約器は `nullopt` を poison、NaN を除外（[util.cpp:10-82](../../core/anet-core/src/util.cpp:10)）。

### F2. 同時完了率は構成で変わる

λ = num_envs / 平均エピソード長。実測:

| | 単独完了 | 2 | 3 | 4 以上 |
|---|---|---|---|---|
| Breakout / 128 env（`run_20260825-002524`、n=36,419） | **97.1%** | 2.6% | 0.3% | 0.05% |
| DropMerge / 256 env（`run_20260819-173220` の `42_env/12_ep_frct_mean`、n=277,596） | 71.3% | 16.7% | 8.8% | 3.2% |

Atari-5 はエピソード長がゲームごとに違うため、**同じタグがゲームごとに違う集約になる**。

### F3. 失われるのは平均値ではなく「同時分布」

2026-08-24 に FIRE デッドロックを否定した決め手は「低スコア帯は `game_len` も短い」という同時分布だった（`score<50` かつ `len>3000` が 0 本）。n=1 だから step で join できたにすぎず、n>1 では score 3 行・len 3 行のどれとどれが対か分からない。

### F4. eval 側で個体値が存在する瞬間は 1 箇所だけ

- `EvalSessionEnv::CaptureScalars(group)` が `inner_->GetScalar(base_key, lane)` を読み、その場で `ScalarSampleAccumulator` へ畳む（[env.cpp:316-325](../../core/anet-core/src/env.cpp:316)）。N 本の `game_score` 列はプロセス内のどこにも残らない（残るのは `EvalSessionResult::episode_returns` だけ、[env.hpp:132-134](../../core/anet-core/include/anet/env.hpp:132)）。
- `EvalSessionEnv::GetScalar(key, index)` は `index < 0` かつ購読 key ならセッション集約、それ以外は inner へ透過（[env.cpp:367-377](../../core/anet-core/src/env.cpp:367)）。lane 指定は inner lane の**現在値**（auto-reset 後）を返す。
- `EvalRunner::RunSession` はセッション中の lane 単位 `EpisodeEndEvent` を**意図的に抑止**し（`DoStepInternal(-1, event_counts, false)`）、完了時に `env_index=-1` の `EpisodeEndEvent` を 1 回だけ出す（[trainer.cpp:342-365](../../core/anet-core/src/trainer.cpp:342)）。counts は train runner の `event_counts`（ADR 0029 の step 座標系）。
- 採用完了 group は `EvalSessionEnv::Step` の内部で分かる（[env.cpp:343-358](../../core/anet-core/src/env.cpp:343)）が、外へは出していない。

### F5. train 側は lane 単位のイベントが既に届いている

`RunnerBase::AccumulateAndNotifyEpisodeEnd` は完了 group ごとに `EpisodeEndEvent{ runner, counts, agent, env, env_index }` を通知する（PER_LANE は lane、SHARED は −1。[trainer.cpp:117-141](../../core/anet-core/src/trainer.cpp:117)、[rl.hpp:923-929](../../core/anet-core/include/anet/rl.hpp:923)）。それを `MetricsLogEpisodeEndObserver::OnEpisodeEnd` が捨てて（[observers.hpp:390-393](../../core/anet-core/include/anet/observers.hpp:390)）、`GetMetricsData` が `target->GetScalar(key_)`（index 既定 −1）を呼ぶ（[observers.cpp:808](../../core/anet-core/src/observers.cpp:808)）。

### F6. 書き口: scalar は backend 直書き、既存 json 書き口は流用不可

- `LogScalar` は header inline で `{"type":"scalar","tag","step","value"}` を `backend_->WriteJsonl` へ直接書く。logger 側の mutex 無し、side file 無し（[metrics_logger.hpp:162-170](../../core/anet-core/include/anet/metrics_logger.hpp:162)）。`JsonlBackend::WriteJsonl` が `mtx_` で直列化する（[metrics_logger.cpp:196-210](../../core/anet-core/src/metrics_logger.cpp:196)）。キューは無く、呼び出しスレッドで同期書き込み。
- `Log(tag, step, json)`（[metrics_logger.cpp:694-715](../../core/anet-core/src/metrics_logger.cpp:694)）は**呼び手ゼロ**で、JSONL 行に `step` を書かず、呼ぶたびに `json/<tag>_<step>.json` を作る。record 用には使えない。
- `Log(tag, json)`（[:688-692](../../core/anet-core/src/metrics_logger.cpp:688)、`LogJsonInternal` [:629-647](../../core/anet-core/src/metrics_logger.cpp:629)）は `{"type":"json","tag","data","timestamp"}` + `json/<tag>.json` ミラー。`metrics.defs` はこれで 1 回だけ書かれる（[trainer.cpp:1015-1027](../../core/anet-core/src/trainer.cpp:1015)）。

### F7. 設定 DSL と定義レコード

- `ObserverFactory` は解決済み `ConfigData::Map()` から `metrics.scalar.[tag]` だけを拾い、値を空白分割してトークン分類する（[observers.cpp:1191-1331](../../core/anet-core/src/observers.cpp:1191)）。トークン: `@train` / `@learn` / `@episode_end`、`$train_step` 等 6 軸、`$agent` / `$env` / `$exp` / `$update_result` / `$runner` / `$action_info`、`$train` / `$eval.[name]`、`$ema`、属性 `key:` / `event:` / `step:` / `target:` / `interval:` / `ema_alpha:` / `clip:`。**それ以外の裸トークンは key（最後のものが勝つ、[:1328](../../core/anet-core/src/observers.cpp:1328)）**。lane / index を指す選択子は無い。
- 検証（[:1348-1361](../../core/anet-core/src/observers.cpp:1348)）: `$eval.[x]` は `@episode_end` か `@train $action_info` のみ、`$action_info` は `@train` のみ、`@episode_end` は `$exp` / `$update_result` 不可。step 軸既定は `@train`→`train_step`、それ以外→`exp_step`（[:1364-1375](../../core/anet-core/src/observers.cpp:1364)）。
- 解決済み定義 `ScalarMetricDef` を tag 単位で持ち（[observers.hpp:453-466](../../core/anet-core/include/anet/observers.hpp:453)）、`ScalarMetricDefsToJson` が `{tag: {step_axis, runner, event, target, source_key, ema_alpha, interval}}` を作る（[observers.cpp:1173-1189](../../core/anet-core/src/observers.cpp:1173)）。`runner` は「EVAL scope かつ `@train` のときだけ eval 名、それ以外は `train`」（[:1165-1170](../../core/anet-core/src/observers.cpp:1165)）。
- RunManager は eval tag ごとに `scope==EVAL && event==EPISODE_END && target==ENV` の定義を `EvalSessionEnv` の購読 key として渡し、N>1 で prefix 必須を検証する（[trainer.cpp:923-941](../../core/anet-core/src/trainer.cpp:923)）。attach 後に `metrics.defs` を 1 行書き、同じ定義を購読ヒントとして Agent へ渡す（[:1015-1033](../../core/anet-core/src/trainer.cpp:1015)）。

### F8. 読み手は未知の `type` を捨てる（1 つだけ致命条件がある）

| 読み手 | 未知 `type` の扱い |
|---|---|
| `inspect_run.py` | `type != "scalar"` は `tag == "metrics.defs"` 以外を黙って skip（[inspect_run.py:887-892](../../viewers/metrics-tools/inspect_run.py:887)）。`type` 欄が無い行だけ `SourceError` |
| Metrics Viewer（Java `MetricsIngestor`） | 非 scalar は生行のまま `json_lines` へ素通し（[MetricsIngestor.java:315-331](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java:315)、[:490-502](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java:490)）。**`step` があれば整数（±2^53 内）でなければ Run 全体が `ERROR`**（[:355-375](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java:355)）。`type` が文字列でない場合も同様。UI に json 行の表示は無い |
| `mlflow_bridge.py` | `type` を見ず、top-level `tag` + 数値 `value` があれば metric として送る（[mlflow_bridge.py:198-217](../../viewers/metrics-tools/mlflow_bridge.py:198)） |
| `tb_bridge.py` | 未知 type は `raw_event` text + stdout 1 行（[tb_bridge.py:115-118](../../viewers/metrics-tools/tb_bridge.py:115)）。害は無いが騒がしい |
| `metrics_viewer.py`（旧 Dash） | `("scalar","json")` の allowlist で落とす（[metrics_viewer.py:60](../../viewers/metrics-tools/metrics_viewer.py:60)） |
| optuna（`optuna_common.py`） | `type == "scalar"` だけ読む（[:937-947](../../apps/runner/tools/optuna_common.py:937)） |
| `compress_workspace_metrics.py` | 内容を解釈しない |

`json_lines(ordinal, type, tag, step, timestamp, json)` は `tag` / `step` / `timestamp` が NULL 可（[MetricsCacheDatabase.java:318-326](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java:318)）。

### F9. 「トレース」は NN activation タップに既に使われている

`using TraceCallback = std::function<void(std::string_view, const torch::Tensor&)>;`（[common.hpp:27](../../core/anet-core/include/anet/common.hpp:27)）。`MakeActionTraceCallback` は env 0 の層別 activation を viewer へ流す（[rl.cpp:20-25](../../core/anet-core/src/rl.cpp:20)）。本 PRD は名前を trace に決めたので、**用語側で区別する**（§4.1）。

### F10. 容量は制約にならない

`run_20260825-002524_atari_breakout_apex_e04`（50M step）の `metrics.jsonl`:

| | 行数 | 全体比 |
|---|---|---|
| 全体 | 15,082,952 | 100% |
| エピソード確定 raw 5 タグ | 182,095 | 1.21% |
| 実エピソード数 | 36,419 | 0.24% |

eval の trace は「セッションあたり N 行」なので、これよりさらに小さい。

### F11. `EpisodeId` / `model_version` はコードに存在しない

`EpisodeId` / `TransitionId` / `episode_id` は `core/` `apps/` に 0 件（932 は未着手）。912 の「採点した network version」も記録欄が無い。どちらも本 PRD では欄を作らない（§8 のゲート）。

## 3. 問題

### A. 集約が構成依存で、ゲーム横断比較が静かに壊れる（train）

Atari-5 では 5 ゲームで λ が異なり、`mean.game_score` の意味がゲームごとに変わる。エラーも警告も出ない。

### B. 同時分布が復元できない（train / eval 共通、eval は 060 で悪化）

eval は N 本が 1 点に畳まれ、`std.` を足しても時間方向の分布（p10、閾値越え率）にはならない。

### C. provenance を書く場所が無い

912（network version）、932（`EpisodeId`）が同じ欠落を別々に抱えている。レコード型があれば「保証する」の前に「記録して事後に判別する」が選べる。本 PRD は器だけを用意し、欄は作らない。

### 3.1 ゴールと非ゴール（ゴールアンカー）

**ゴール**: 統計（scalar）と別に、「1 件 = 1 レコード」を識別子付きで `metrics.jsonl` へ出す **trace チャネルの出力側**を実装する。(a) 行契約、(b) JSONL 専用の書き口、(c) 設定 DSL での宣言、(d) チャネル別の定義レコード、(e) producer 第 1 号 = eval の採用エピソード（train は同じ仕組みで宣言だけ足せば動く状態にする）。

**非ゴール**: inspect_run / Viewer の trace 読み取り、912 の `model_version`、932 の forensic レベル、走行中に裾を見ること（移動窓統計は別件）、既存 scalar 集約タグの廃止、`min.` / `std.` 以外の集約 prefix 追加。

## 4. 確定契約

### 4.0 決定一覧

| # | 決定 |
|---|---|
| D1 | 名前は **trace**（`metrics.trace.$` / `"type":"trace"` / `LogTrace`）。`TraceCallback`（NN activation タップ）とは CONTEXT.md の _Avoid_ で区別する |
| D2 | 契約付きチャネル（旧案④）。level（off / record / forensic）は作らない（宣言すれば ON、書かなければ OFF）。`episode_id` / `model_version` 欄は作らない |
| D3 | trace は scalar と同じ DSL トークン表記を使う汎用チャネル。イベントの明示は必須で、本 PRD のトリガは `@episode_end` のみ。`@train` / `@learn` / `@session_end` は設定読み込み時に fail-fast |
| D4 | eval の N 本は **採用エピソード完了ごとに `EpisodeEndEvent`** を発火して observer へ届ける。`EvalSessionEnv` は trace 用の個体値列を溜めない（既存 scalar 累積器・return 列は維持） |
| D5 | セッション完了は新設 **`SessionEndEvent`**（型分離、継承なし）。eval scalar は `@session_end` に束縛する。`$eval.[x] @episode_end` の scalar は fail-fast、alias / WARN 無し |
| D6 | 行の形: `{"type":"trace","tag","step","lane","data":{key:value...}}`。固定属性は直書き、個別値は `data` 下。`lane` = `env_index`（SHARED は −1）。`timestamp` 無し |
| D7 | 裸トークンは全部キー（1 個以上、重複不可）。取得順と定義の `keys` 配列は宣言順、`data` のキー順序は保証しない。target 明示必須。集約 prefix / `$ema` / `ema_alpha:` / `clip:` / `interval:` / `key:` は fail-fast |
| D8 | 宣言キーが `GetScalar` で `nullopt` なら fail-fast。NaN / ±Inf は `null` で残す |
| D9 | scalar 定義の出力は **`metrics.scalar.defs` へ改名**し、trace は `metrics.trace.defs` を新設。読み取りは **`metrics.scalar.defs` 優先、存在しない場合だけ `metrics.defs`**。現用の過去 Run artifact のための互換例外とする（§4.7） |
| D10 | train 側の宣言は**コメントアウト**で Atari.txt に同梱（コード変更ゼロで動く） |
| D11 | 旧書式（eval scalar の `@episode_end`）は fail-fast のメッセージで置換先を案内する。alias は置かない |
| D12 | コード・現用設定・文書が整合する一つの完成単位。イベント分離だけを独立導入せず、残りは全てゲート（§8）。コミット操作は人間が行う |
| D13 | 系列は `(type, tag)` で識別し、scalar / trace の同名 tag を許可する。チャネル横断の重複検査は作らない |
| D14 | trace のイベント・target・scope・step 軸の重複指定、未知・禁止トークンは設定読み込み時に拒否。別表記でも同じ指定先なら重複。共通パーサーは出現情報を保持し、scalar の既存挙動は維持する |
| D15 | `$runner` は index を無視してカウンタ等を返せる。取得した値を何の意味で記録するかは設定者の責任とし、lane 指定なら必ず失敗するとは保証しない |
| D16 | 等価性は編集前 baseline と決定的 backend・foreground eval の固定構成で比較。時間依存値を除く scalar の件数・step・値を確認し、各 eval で最低3セッション完了を要求する |

### 4.1 用語（CONTEXT.md）

**trace**: 統計（scalar）とは別に、1 件の個体（lane のエピソード完了など）を 1 レコードとして識別子付きで記帳するメトリクスチャネル。固定属性と個別値の集合 `data` を持ち、集約しない。系列はチャネルと tag の組で区別し、同じ tag の scalar とは別系列である。NN の層別 activation を可視化へ流す `TraceCallback` とは別概念。_Avoid_: activation トレース、record、journal（932 の Episode Journal は forensic 側の別概念）。

### 4.2 行の形

```json
{"type":"trace","tag":"51_eval1/episode","step":41547136,"lane":3,"data":{"game_score":422,"game_len":1242,"game_frames":4968,"hns57":31.2}}
```

| 欄 | 型 | 内容 |
|---|---|---|
| `type` | 文字列 | 常に `"trace"` |
| `tag` | 文字列 | 宣言した tag。系列は `(type, tag)` で識別し、scalar / trace の同名 tag を許可する（`51_eval1/...` の慣習は tag 側で担う） |
| `step` | 整数 | `counts.GetByAxis(step_axis)`。scalar と同じ step 座標系契約（ADR 0029） |
| `lane` | 整数 | イベントの `env_index`。PER_LANE は lane index、SHARED（ImageCls eval）は −1 |
| `data` | object | 宣言したキーと値の対応 `{key: value}`。キー順序は保証しない。値は `GetScalar` の float。NaN / ±Inf は JSON 化で `null` になる（キーは残す） |

`timestamp` は付けない（scalar と同じ最小形）。trace 内の同一 `(tag, step)` に複数行が並ぶのは正常（eval セッションの N 行は全て同じ step）。行の identity は CONTEXT.md「序数」のとおり出現順が持ち、`(step, lane)` の一意性は要求しない（N > G では同じ lane が 2 本以上を担う）。同一行に score と len を同居させること自体が問題 B の解であり、join キーは不要。

### 4.3 宣言（設定 DSL）

```
# metrics.trace.[tag] = $target @episode_end key1 [key2 ...] [$step] [$train | $eval.[name]]
metrics.trace.$ = metrics.trace.@atari
metrics.trace.@atari.[51_eval1/episode] = $eval.[eval1] @episode_end $env game_score game_len game_frames hns57
metrics.trace.@atari.[52_eval2/episode] = $eval.[eval2] @episode_end $env game_score game_len game_frames hns57
#metrics.trace.@atari.[42_env/episode]  = $train @episode_end $env game_score game_len game_frames hns57
```

- トークン集合と表記は scalar と同一（`@event` / `$target` / `$step` / `$train` / `$eval.[name]` / `attr:value`）。選択チェーン `metrics.trace.$ = ...` は他スロットと同じ規則で `metrics.trace.[tag]` へ展開される。
- **裸トークンは全部キー**。1 個以上、重複不可。値の取得順と定義レコードの `keys` 配列は宣言順とし、`data` オブジェクトのキー順序は保証しない。既存の `anet::json = nlohmann::json` を使い、順序保持専用の JSON 型・書き込み経路は追加しない。scalar の「最後のキー指定を採る」挙動は維持する。
- キーは素の名前のみ。`mean.` / `max.` / `min.` / `std.` が付いていたら fail-fast（個体に集約は無い。lane 指定 `GetScalar` は prefix 無しが契約、F1）。
- `$target` は**必須**。scalar の総当たり探索（update_result → agent → exp → runner → env）はキーが複数だと別対象に解決しうるので使わない。`$env` / `$runner` / `$agent` を受理し、`$exp` / `$update_result` / `$action_info` は fail-fast。
- `$ema` / `ema_alpha:` / `clip:` / `interval:` / `key:` は設定読み込み時に fail-fast（キーは裸トークンで書く）。`interval:1` のように既定値と同じ指定も拒否する。
- event は **`@episode_end` または `event:episode_end` の明示必須**で、省略は設定読み込み時に fail-fast。`@train` / `@learn` / `@session_end`（属性形も同様）は `ANET_SYSTEM_ERROR`（`trace supports @episode_end only in this version` を含める）。将来のイベント追加は §8 のゲートで再検討する。
- target も `$env` 等または `target:env` 等で明示する。イベント・target・scope・step 軸を複数回指定したら、値が同じでも異なっても拒否する。`@episode_end event:episode_end`、`$env target:env`、`$exp_step step:exp` 等の別表記も同一指定先への重複となる。
- 未知の制御トークン（`@...` / `$...`）、未知属性、既知属性の未知列挙値・不正な構文は設定読み込み時に拒否し、裸キーへ読み替えない。裸キー自体の認識可否は §4.6 の `GetScalar` で検出する。エラーには config key / 指定トークン / 期待する指定を含める。
- 共通パーサーはトークンの種類・値・指定順・出現情報を保持し、trace の検証前に既定値や後勝ちで潰さない。禁止トークンを後続指定で上書きして通すことも認めない。scalar 側の既存の後勝ち・既定値・診断挙動は変更しない（§4.5 のイベント移行は除く）。
- `$step` の既定は scalar と同じ（`@episode_end` → `exp_step`）。scope 既定は `$train`。
- 宣言が無ければ observer も行も生成されない（level は持たない）。

### 4.4 イベント契約（型分離）

**`EpisodeEndEvent`**（無変更）: 1 つの episode group が完了した。`{ runner, counts, agent, env, env_index }`。

- train: 従来どおり `AccumulateAndNotifyEpisodeEnd` が完了 group ごとに 1 回（F5）。
- **configured eval（新）**: 評価セッション中、**採用エピソードが完了するたびに 1 回**。`env` = `EvalSessionEnv`、`env_index` = PER_LANE なら group（= lane）、SHARED なら −1、`counts` = 従来のセッション event と同じ train runner の `event_counts`。非採用の完了は出さない。発火点は `EvalRunner::RunSession` のループで `DoStepInternal(-1, event_counts, false)` から戻った直後（`state_` 更新後、次の `Step()` 前）。この瞬間は inner env の `completion_available_` がまだ立っているので、`EvalSessionEnv::GetScalar(key, lane)` の透過読みで確定値が取れる（F1 / F4。`CaptureScalars` と同じタイミング）。`LastAdoptedGroups()` が返すのは内部 group index（SHARED は 0）であり、EvalRunner がイベント生成時に SHARED の `env_index=-1` へ変換する。Runner 側の return 集約（`completed_episode_returns_`）はセッション末尾の `SetCompletedEpisodeReturns` のままで、`AccumulateAndNotifyEpisodeEnd` は通さない。

**`SessionEndEvent`**（新設）: 評価セッション（採用 N 本）が完了した。`{ runner, counts, agent, env }`。`env_index` は持たない。継承は使わない（欄 4 つの struct。ディスパッチはオーバーロード）。従来 `env_index=-1` で出していた `EpisodeEndEvent` の**置き換え**で、セッションあたり 1 回。`SessionEndObserver::OnSessionEnd`、`Notifier` の Attach / Detach / Notify / AttachScoped、`RunnerScopedSessionEndObserver` を既存 3 種と同型で足す。

`EpisodeEvalObserver`（schedule / background 制御）は無変更。background eval スレッドからの通知は従来のセッション event と同じ経路で、`JsonlBackend::WriteJsonl` の mutex が直列化する（F6）。

### 4.5 scalar チャネルの `@session_end`

- トークン `@session_end`（属性形 `event:session_end`）を追加。`EventType::SESSION_END`。
- 束縛: `MetricsLogSessionEndObserver`（`OnSessionEnd → OnGenericUpdate(counts, agent, runner, env, nullptr, nullptr)`）。`MetricsLogEpisodeEndObserver` は無変更。
- 検証行列（scalar / trace）:

| scope | event | scalar | trace |
|---|---|---|---|
| `$train` | `@train` / `@learn` | OK（従来） | fail-fast（未対応） |
| `$train` | `@episode_end` | OK（`$exp` / `$update_result` 不可、従来） | OK |
| `$train` | `@session_end` | fail-fast（train にセッションは無い） | fail-fast |
| `$eval.[x]` | `@train` | `$action_info` のみ OK（従来） | fail-fast |
| `$eval.[x]` | `@learn` | fail-fast（従来） | fail-fast |
| `$eval.[x]` | `@episode_end` | **fail-fast**（旧書式。メッセージに `eval scalar metrics fire once per evaluation session; replace @episode_end with @session_end` を含める） | OK |
| `$eval.[x]` | `@session_end` | OK（`$exp` / `$update_result` 不可） | fail-fast（未対応） |

- eval scalar を `@episode_end` に残すと、採用エピソードごとに `EvalSessionEnv::GetScalar("mean.game_score", -1)` の**途中経過の平均**が点として出る。誤読を生むので黙って通さない。
- step 軸既定: `SESSION_END` は `EXP`（`@episode_end` と同じ）。`OwningRunner` は変更不要（EVAL + `@train` 以外は `"train"`）。`metrics_def_names::EventToken` に `"session_end"`。
- RunManager の購読抽出（F7）は `event == SESSION_END` へ。
- **設定移行**: `$eval.[` と `@episode_end` を両方含む scalar 行の `@episode_end` を `@session_end` へ機械置換。対象と件数: `apps/runner/config/` の DropMerge.txt (50) / Atari.txt (36) / metrics_scalar.txt (8) / GridMaze.txt (6) / LunarLander.txt (4) / ImageCls.txt (3) / GridMaze_muzero.txt (2) / DropMerge_optuna.txt (2)、`apps/runner/workspaces/atari-live/config/atari_base.txt` (24)。train scope の `@episode_end`（`$runner mean.episode_return @episode_end $train` 等）は触らない。テスト内の同形文字列（`observers_test.cpp:625,656`、`trainer_test.cpp:692,712,715,765,783`）も同時に移行し、旧書式は fail-fast のテストへ転用する。過去 Run の `config_data.txt` は当時の記録として変更しない。

### 4.6 trace observer の値取得

`MetricsLogTraceObserver : EpisodeEndObserver`。メンバ: `tag_`、`keys_`（順序付き）、`step_axis_`、`field_`（必須）。

1. `OnEpisodeEnd(event)` で target を `field_` から解決する: `$env` → `event.env`（無ければ `runner->GetBatchEnv()`）、`$runner` → `event.runner`、`$agent` → `event.agent`。
2. `index = event.env_index`。各 key について `target->GetScalar(key, index)` を呼ぶ。
   - `nullopt`（未知キー）→ `ANET_SYSTEM_ERROR`（tag / key / lane / target を含める）。宣言したキーが取れないのは設定ミスで、初回イベントで検出する。scalar が毎イベント WARN を出し続ける現状の裏返し。
   - 値は `json` へ格納。NaN / ±Inf は JSON 化で `null` になる（`hns57` の未知ゲームが該当。キーは落とさない）。
3. `step = counts.GetByAxis(step_axis_)`。
4. `MetricsLogger::Instance()->LogTrace(tag_, step, index, data)`。

`EvalSessionEnv` は `index >= 0` を inner へ透過するので、PER_LANE の eval でも train でも同じ observer で lane の確定値が読める。SHARED は `index=-1` で読む。N>1 の scalar 購読は集約 prefix 必須、trace は prefix 禁止なので個体キーは inner へ透過し、N=1 で同名の無 prefix 購読がある場合も capture 済みの単一 episode 値と一致する。

`$runner` / `$agent` も受理する。現行の `RunnerBase::GetScalar`（[trainer.cpp:199](../../core/anet-core/src/trainer.cpp:199)）は index を無視してカウンタや reward 等を返すため、lane 指定なら必ず `nullopt` になるわけではない。取得値が対象 episode 固有の量か、イベント時点の共有状態かを選ぶのは設定者の責任で、汎用層は意味的な妥当性を保証しない。未知キーだけが `nullopt` → fail-fast の対象である。per-episode return を event に載せる案と Agent の遅延計算キーへの購読ヒントは §8 のゲートとする。

### 4.7 書き口と定義レコード

**`LogTrace`**（`metrics_logger.hpp`、`LogScalar` の隣。backend 直書き、`log_mutex_` 無し、side file 無し）:

```cpp
inline void LogTrace(const std::string& tag, int64_t step, int64_t lane, const json& data) {
    json obj = {{"type","trace"},{"tag",tag},{"step",step},{"lane",lane},{"data",data}};
    backend_->WriteJsonl(obj);
}
```

既存 `Log(tag, step, json)` は使わない（F6）。削除は本 PRD の対象外（dead code として報告のみ）。

**定義レコード**（ADR 0029 の契約をチャネル別に持つ）:

- scalar は **`metrics.scalar.defs` へ改名**する。writer と reader の現行契約・fixture・現行文書を同じ変更で移行し、ミラーも `json/metrics.scalar.defs.json` になる。定義の event token に `session_end` を追加する。
- **`metrics.trace.defs`** を新設。attach 済み trace 定義（dormant eval tag は除外）を tag 単位で `{tag: {step_axis, runner, event, target, keys:[...]}}`。書き口は `Log(tag, json)`（`type:"json"`、`json/metrics.trace.defs.json` ミラーも既存機構で出る）。空なら書かない。
- `metrics.scalar.defs` に trace 定義を混ぜない理由: `inspect_run.py` の `metric_defs_from_record` は data 内の全 tag を scalar 定義として読むため、trace tag が「定義済み・未観測の scalar」に化ける。
- **読み取り**: `inspect_run.py` の master / cache の両経路で、基本は `metrics.scalar.defs` を読み、なければ旧 `metrics.defs` を読む。writer は空でない定義を新名で1回だけ出力し、ミラーも新名を使う。
- **互換例外**: 対象は旧 `metrics.defs` を持つ現用の過去 Run artifact。移行方法は新しい Run の実行、互換期間は現用 Run 作業セットが全て `metrics.scalar.defs` を持つまで、削除条件も同じとする。過去 artifact は書き換えない。新旧どちらを読んでも `def_source=metrics_defs` とし、旧名を読んだことだけを理由とする WARN は不要。定義不在時の既存設定導出も維持するが、これは新旧レコードの優先選択とは別経路である。
- **`EVENT_NAMES` への追加理由**: `metric_defs_from_record` は `session_end` をそのまま保持できる一方、cache 未構築の新 Run でも [`metrics` の selector 展開](../../viewers/metrics-tools/inspect_run.py:1493)と [`tags --no-observed`](../../viewers/metrics-tools/inspect_run.py:1771)はマスタ走査を省いて設定導出を通る。したがって `metric_def_from_definition` の `EVENT_NAMES` に `session_end` を追加し、既定 `exp_step`・座標系所有者 `train` を導出できるようにする。この経路は過去 Run の互換需要がなくなっても到達可能なので、互換削除条件だけを理由に除去しない。
- 購読ヒント（`ConfigureScalarMetricSubscriptions`）には trace を渡さない（§8 のゲート）。

### 4.8 設定（Atari）

[Atari.txt:1386](../../apps/runner/config/Atari.txt:1386) の `metrics.scalar.$` 行の直後に §4.3 の 5 行（`metrics.trace.$` + eval1 / eval2 の宣言 + コメントアウトした train 宣言）を置く。

- キーは `game_score` / `game_len` / `game_frames` / `hns57`。`hns49` は `game_score` から表引きで再導出できるので入れない。`game_score.ge.[N]` も同様。
- **`lives` は入れない**: 確定ゲートが無く、episode end 時点では auto-reset 後の新エピソードの残機が読める（既存 `mean.lives` の `@episode_end` も同じ挙動）。
- [metrics_scalar.txt:8-11](../../apps/runner/config/metrics_scalar.txt:8) の文法コメントに `@session_end`（eval 専用）と `metrics.trace.[tag] = $target @episode_end key1 key2 ...` を追記する。
- DropMerge の eval trace は今回入れない（要るときに設定 1 行、§8）。

### 4.9 読み手側の制約（本 PRD が守る 3 点）

F8 から導かれる、行を書く側の制約:

1. `type` は必ず文字列で入れる（Java ingestor が Run を `ERROR` にする唯一の経路の 1 つ）。
2. `step` は整数（float / 文字列 / 2^53 超は Run 全体が `ERROR`）。
3. top-level に数値 `value` を置かない（`mlflow_bridge.py` が metric として送ってしまう）。個別値は必ず `data` 下。

`tb_bridge.py` は trace 行ごとに text event と stdout 1 行を出す。害は無いが、trace を使う Run で TensorBoard bridge を回すなら別途 skip を足す（本 PRD の対象外）。

## 5. 実装ノート

| 対象 | 変更 |
|---|---|
| `core/anet-core/include/anet/rl.hpp` / `src/rl.cpp` | `EventType::SESSION_END`、`SessionEndEvent`、`SessionEndObserver`、`RunnerScopedSessionEndObserver`、`Notifier` の Attach / Detach(×2) / Notify / AttachScoped 分岐 / `session_end_observers_` / Clear / LogObservers（`EpisodeEnd` 系のコピー） |
| `core/anet-core/include/anet/env.hpp` / `src/env.cpp` | `EvalSessionEnv::LastAdoptedGroups()`（直前 `Step()` の採用完了 group、index 昇順）。`Step()` 先頭と既存 `Reset()` のセッション開始処理で clear、採用完了を処理した直後に push。SHARED の内部 group は 0、イベントの `env_index` は −1 |
| `core/anet-core/src/trainer.cpp` | `EvalRunner::RunSession`: ループ内で `LastAdoptedGroups()` を走査して `EpisodeEndEvent` を通知、ループ後は `SessionEndEvent` を 1 回。RunManager: 購読抽出を `SESSION_END` へ、`GetSessionEndObservers()` を `RunnerScopedSessionEndObserver` で attach、trace observer を `RunnerScopedEpisodeEndObserver` で attach、`Log("metrics.defs", ...)` を `Log("metrics.scalar.defs", ...)` へ改名し、`Log("metrics.trace.defs", ...)` を追加 |
| `core/anet-core/include/anet/observers.hpp` / `src/observers.cpp` | `@session_end` トークン / 属性、検証行列（§4.5）、`MetricsLogSessionEndObserver`、`ParsedSessionEndObserver` / `session_end_observers_`、`EventToken("session_end")`。`metrics.trace.[` の走査、トークン分類ループ（[observers.cpp:1221-1331](../../core/anet-core/src/observers.cpp:1221)）を名前付き namespace の static helper（`ParseMetricTokens`）へ切り出して scalar / trace で共用。トークンの種類・値・指定順・出現情報を渡し、チャネル別に意味解決・検証する。scalar は `key:` を含む既存の後勝ち・既定値・診断を維持し、trace は §4.3 の明示・重複・禁止指定を検証、`MetricsLogTraceObserver`、`TraceMetricDef` / `TraceMetricDefsToJson` / `GetTraceMetricDefs()` |
| `core/anet-core/include/anet/metrics_logger.hpp` | `LogTrace`（§4.7） |
| `apps/runner/config/*.txt`、`apps/runner/workspaces/atari-live/config/atari_base.txt` | §4.5 の機械置換、§4.8 の追加 |
| `viewers/metrics-tools/inspect_run.py` / `inspect_run_test.py` | 定義 tag を新旧2名で扱い、master / cache とも新名優先・新名不在時だけ旧名を採用。新出力 fixture と旧 Run fixture で読み取りを検証。`EVENT_NAMES` に `session_end` を追加し、定義レコードと cache 未構築時の設定導出を検証 |
| docs | §11 |

コード実装時の順序: **編集前 baseline 採取（§6-9）** → イベント分離（rl → env → trainer → observers の session_end → 設定置換）→ trace チャネル（LogTrace → factory → observer → defs → Atari 宣言）→ inspect_run → テスト・前後 Run 比較 → 現行設計文書の同期。これは内部の作業順であり、イベント分離だけを独立導入しない。コード・現用設定・文書の整合を一つの完成単位とし、1 コミットにまとめる場合の staging / commit は人間が行う。

`static` helper は状態非依存の純粋パースに限る。テストのために本体へ test-only API を足さない（`LastAdoptedGroups()` は EvalRunner が使う production API）。

## 6. 受入条件

以下を実装の受入条件とする。

1. **イベント分離**: `EvalRunner::RunSession` が採用 episode ごとに `EpisodeEndEvent`（`env=EvalSessionEnv`、`counts` は呼び出し元の `event_counts`）を N 回、その後 `SessionEndEvent` を1回通知する。非採用完了では通知しない。PER_LANE は `env_index=group`、SHARED は内部 group 0 を `env_index=-1` へ変換する。`LastAdoptedGroups()` は直前 Step の採用完了だけを index 昇順で返し、次の Step 先頭と Reset で以前の一覧を消す。N>G、N<G、N=1、SHARED、同時完了・非採用完了を制御した既存テストを拡張する。
2. **scalar の挙動不変**: `$eval.[x] @session_end $env mean.score` と `$runner mean.episode_return @session_end $eval.[x]` が、変更前の `@episode_end` と同じセッション集約を1点だけ出す。途中経過を scalar として記録しない。
3. **DSL 検証**: §4.5 の全セルと §4.3 の明示・重複・未知・禁止指定を検証する。trace の event / target 欠落、重複キー、同値・異値・別表記による指定先の重複、`interval:1`、`key:`、未対応トークンの後続指定による上書きを拒否する。取得順と定義の `keys` 配列は宣言順を確認し、`data` オブジェクトの並び順は検査しない。共通化により scalar の既存後勝ち・既定値・診断を変えない。
4. **trace 行と取得時点**: 通知 callback 内で値を読み、実際の JSONL 行が `{"type":"trace","tag","step","lane","data"}` の形で1イベントにつき1行になることを確認する。次の Step で確定値が消えるテスト Env を使い、PER_LANE の個体値と SHARED の episode 値が採用完了時の値に一致する。未知キーの `nullopt` は tag / key / lane / target 付きで fail-fast、NaN / ±Inf はキーを残した `null`。scalar / trace の同名 tag が別系列として記録でき、train も共通 observer で動作する。
5. **定義レコード**: scalar は `metrics.scalar.defs` へ改名し、event token に `session_end` を追加する。trace は `{tag: {step_axis, runner, event, target, keys}}` を `metrics.trace.defs` に記録する。attach 済み定義だけを記録し、dormant eval は除外、trace 定義が空ならそのレコードを出さない。ミラーは `json/metrics.scalar.defs.json` / `json/metrics.trace.defs.json` に出力する。
6. **inspect_run**: master / cache の両経路で、新名を出力する新 Run と旧名を持つ過去 Run を読み、どちらも `def_source=metrics_defs` となり、改名だけを理由とする WARN を出さないことを検証する。定義レコードがない場合の既存設定導出も検証する。cache 未構築の新 Run の `metrics` selector 展開と `tags --no-observed` で `session_end` の定義を導出でき、明示軸を尊重し、省略時は `exp_step`・runner=`train` となることを確認して `inspect_run_test.py` を通す。 補助回帰として、新旧定義名が併存する場合も、レコードの順序を入れ替えて master / cache の両経路で新名が優先されることを確認する。
7. **移行漏れゼロ**: リポジトリ管理下の現用 scalar 設定・サンプル・テスト・現行文書を検索し、eval scalar の `@episode_end` / `event:episode_end` が拒否テストや変更履歴以外に残らないことを確認する。trace の `@episode_end` と過去 Run artifact・履歴資料は移行対象に含めない。
8. **ビルドとテスト**: AGENTS.md の `VsDevCmd.bat` 経由で Debug ビルドと `anet-core-test.exe` を実行し、関連テストと Python の `inspect_run_test.py` を通す。既知の失敗は件数と根拠を記録し、未実施の検証を合格扱いしない。background eval は別の制御したテストで、採用 N 件→session 1件の配送、trace 行数、observer 例外の呼び出し元への伝播を確認する。
9. **等価性 Run**: コード編集前のバイナリで baseline を採取する。Breakout + `run.@evalN10` + `app.$=app.batchrun`、`app.run_name` に `tmp` を含む別出力先を用い、前後とも同 seed・決定的 backend（`backend.@deterministic`）・foreground eval（eval1 / eval2 の `use_background=false`）に固定する。解決済み設定を保存・照合し、比較に必要なイベント移行と trace 追加以外の条件を揃える。両 Run で eval1 / eval2 がそれぞれ最低3セッション完了するまで実行する。時間依存の速度メトリクスは値一致の対象から明示的に除外し、それ以外の scalar の件数・step・値を比較する。baseline 不在・必要セッション未完了・差異未解決は検証不足として明記し、合格扱いしない。
10. **行数・分布の整合**: N=G=10 の変更後 Run で、各セッション・各 eval trace tag にちょうど10行、全行同じ step、lane 0〜9が一度ずつ現れることを確認する。`mean(data.game_score)` は同 step・同 eval の既存 game_score scalar と float 丸め範囲内で一致する。trace 行から score×len の同時分布、p10、閾値越え率を算出し、検証結果を残す。
11. **読み手の無害性**: 変更後 Run を Metrics Viewer が `ERROR` にせず取り込み、既存 scalar を利用できること、`inspect_run.py runs` / `tags` が警告なしで通ることを確認する。trace / scalar の同名 tag ケースも reader 検証に含める。

## 7. 成功指標

| 指標 | 確認方法 |
|---|---|
| 名指しの事故の再現不能化 | PRD 起票時の「eval の p10 / `≥432` 率が復元できない」が、`metrics.jsonl` の trace 行に対する数行のスクリプト（`type=trace` に限定して `(tag, step)` で group → 同時分布・分位点・閾値越え率）で復元できる |
| scalar の挙動不変 | §6-9 の等価性 |
| 新旧チャネルの整合 | §6-10 の同一性 |
| 読み手を壊さない | §6-11 |
| 誤設定が黙って通らない | §6-3 |

## 8. フェーズとゲート

本 PRD の実装はコード・現用設定・文書が整合する一つの完成単位とする。イベント分離だけの独立導入フェーズや「フェーズ 2」は置かない。以下は必要になったときに再設計するゲートであり、将来の拡張口だけの実装も今回追加しない:

| 待機中 | 開けるきっかけ | 開けるコスト |
|---|---|---|
| train 側の宣言（コメント解除） | Atari-5 で train 分布が要るとき | 設定 1 行 |
| trace の `@train` / `@learn`（per-step lane fan-out、per-sample update_result） | 932 forensic、または lane ごとの毎 step 記録が要るとき | 検証行列の分岐 + observer 1 種 |
| trace の `@session_end`（セッション 1 行に `model_version` 等） | 912 着手 | 同上 + 欄の追加 |
| `episode_id` 欄 | 932 で `EpisodeId` が実装されたとき | 固定属性 1 つ |
| inspect_run の `trace` サブコマンド（分位点・閾値越え率） | eval 分布の解析を 2 回目に手書きしたとき | reader 側 |
| trace の購読ヒント（`$agent` の遅延計算キー） | 初めて `$agent` キーを trace に書くとき | 購読ヒントの型拡張 |
| DropMerge eval の trace 宣言 | DropMerge で分布が要るとき | 設定 1 行 |
| `tb_bridge.py` の trace skip | trace を使う Run で TensorBoard bridge を回すとき | 分岐 1 つ |

## 9. 複雑さ監査（グリル 2026-09-05）

| 機構 | 裁定 | 理由 |
|---|---|---|
| level（off / record / forensic） | cut | 宣言の有無で足りる。forensic は 932 の想像上の需要 |
| `episode_id` / `model_version` 欄 | cut | コードに概念が無い。912 / 932 着手時に固定属性として追加 |
| 既存 scalar 集約タグの削除 | cut | Viewer は scalar しか描けない。二重出力は許容 |
| 既存 `type:"json"` 行（config dump / defs）の再契約 | cut | 実害なし |
| trace の `@train` / `@learn` / `@session_end` | defer（fail-fast で閉じる） | 文法と基底は共有、開け口は分岐 1 つ |
| `EvalSessionEnv` が N 本を溜めて最後に N 行出す案 | cut | 「index = lane」契約と「index = 何本目」が同じ引数で衝突し、eval 専用経路になる |
| `EpisodeEndEvent` に種別フィールドを足す案 | cut | 型分離なら scalar 側に実行時フィルタが不要 |
| scalar 定義名の改名 | keep | 新出力は `metrics.scalar.defs` / `metrics.trace.defs` に揃え、reader は新名優先で現用の過去 Run も読む |
| 旧 scalar 定義の互換読み取り | keep | 旧 `metrics.defs` を持つ現用 Run への影響を避けるため、新名不在時の読み取りだけを残す。対象・期間・削除条件は §4.7。writer が生成する新旧それぞれの Run で検証する |
| `data` の宣言順出力専用 JSON 経路 | cut | 名前と値の対応で目的を満たす。取得順と定義の `keys` 配列だけを宣言順で保持 |
| scalar / trace 間の同名 tag 検査 | cut | 系列は `(type, tag)` で区別できる |
| trace の宣言検証 | keep | 重複や禁止指定が後勝ち・JSON 化で黙って消えることを防ぐ |
| train 宣言のコメントアウト同梱 | keep | 共通 observer を使い、train 専用機構と既定の出力を増やさない |
| scalar の不正なイベント束縛の拒否 | keep | セッション集約の途中経過を個体完了ごとに記録する誤構成を防ぐ |
| eval scalar の `@episode_end` alias（WARN 付きで受理） | cut | リポジトリ外に workspace は無い。AGENTS.md のクリーンブレーク方針どおり |

### 9.1 再点検の最終簡素化パス（2026-09-05）

| 項目 | 裁定 | 合意と根拠 |
|---|---|---|
| 1. 機構全体の過剰さ | keep | trace 行・書き口、採用完了 group の公開と episode 通知、session event と scalar 移行、DSL・observer、trace 定義、宣言検証の6機構を維持。旧 scalar 定義の読み取りも、現用の過去 Run への影響を避けるため新名不在時に限定して維持。削ると個体値の保存・取得時点・既存集約との分離・設定による選択・解決済み定義・指定の保存のいずれかを失う |
| 2. 要求の実在性 | defer-behind-gate | 追加イベント、episode_id、model_version、専用 reader、Agent 購読ヒント、DropMerge 宣言、TensorBoard skip は §8 の需要発生まで保留。将来用の拡張口だけも実装しない |
| 3. 前提変更の残滓 | shrink | 改名と新名優先・新名不在時の旧名読み取りを残す。互換対象・期間・削除条件を明示し、writer が生成する新旧それぞれの Run に受入検証を絞る。D9と受入条件・ADRを同期し、変更経緯は §10.1 に集約 |
| 4. 最小解との差分 | shrink | train は共通 observer の動作確認とコメントアウトした宣言例だけ。JSON 本文の順序保証やチャネル横断の tag 検査は追加しない |
| 5. フェーズの独立性 | keep | イベント分離単独では個体分布を保存できないため独立導入しない。作業順は分けても、コード・現用設定・文書の整合を一つの受入単位にする |
| 6. 成功の測定可能性 | keep | 編集前 baseline、決定的 backend・foreground eval の前後比較、各 eval 最低3セッション、各10行と scalar 平均の一致、分布の復元、境界ケースと既存 reader の検証を §6 に固定。検証不足は合格扱いしない |

## 10. 却下した案（経緯）

- **① step の小数部を env 軸に使う**（`step = 41547136 + env/128`）: cache の `step INTEGER`、`--range`、LOD bucket、`(runner, step_axis)` 座標系（ADR 0029）が整数 step 前提。どこかで int キャストされると 128 env が無言で 1 点へ潰れる。時間座標に個体識別を載せる設計で、tag と name を混同しない規律に反する。
- **② scalar 行に識別子カラムを追加する**（`{"step","tag","type":"scalar","value","env":37}`）: 問題 A / B は解けるが、1 エピソードが 5 行に分かれたままで原子性が無く、問題 C（provenance）には識別子の種類ごとにカラムを増やす羽目になる。既存 reader は無改修で通るが、`json_lines` 側の契約を持てない。
- **③ `type:"json"` で env 別レコードを出す**: 既存書き口が step を落とし side file を毎回作る（F6）。契約なしの ad-hoc 行が 2 種類目になる。
- **(a) `EvalSessionEnv` が N 本を溜めて最後の 1 イベントで N 行出す**: §9。
- **`EpisodeEndEvent` に kind（EPISODE / SESSION）フィールド**: §9。
- **eval scalar のトークン `@episode_end` を残して内部で SessionEnd に束縛**: 設定は無傷だが「同じトークンがチャネルと scope で別イベントを指す」ことになり、型を分けた意味が薄れる。

### 10.1 再点検で変更した旧決定（採用契約ではない）

| 旧決定・説明 | 再点検後 | 変更理由 |
|---|---|---|
| D9: 改名のみ（互換読み取りなし） | 改名＋新名優先・新名不在時の旧名読み取り | 現用の過去 Run への影響を避けるため、読み取りに限定した互換例外を設ける |
| D7 / §4.2: `data` も宣言順 | 取得順と `keys` 配列だけ宣言順 | `anet::json` の既存書き口を維持し、名前と値の対応で分布を復元できる |
| §4.2: scalar と同じ命名空間 | `(type, tag)` で別系列 | チャネルが異なれば同名でも区別でき、横断の重複検査は不要 |
| 省略・重複指定の扱いが未確定 | trace の event / target は明示必須、重複・未知・禁止指定は読み込み時に拒否 | 後勝ちや既定値で指定の欠落を隠さない。scalar の既存挙動は維持 |
| §4.6: `$runner` の lane 指定は `nullopt` で失敗する | index を無視して返せる値がある | 現行 `RunnerBase::GetScalar` に合わせ、意味的な選択を設定者の責任とする |
| §6: 同 seed の短い Run で件数・終盤値を比較 | 決定的 backend・foreground eval、時間依存値を除く件数・step・値を比較 | 非決定性・background の時機・実時間値を変更の影響と混同しない |

## 11. 文書同期

### 11.1 今回の文書改訂

コード・設定・Run artifact は変更しない。既存の未コミット変更を保持し、以下の文書だけへ再点検結果を反映する。

| 文書 | 変更 |
|---|---|
| 本 PRD | D1〜D16、行・DSL・値取得・定義の契約、実装ノート、受入条件、6項目の監査、旧決定の変更理由を同期 |
| `CONTEXT.md` | trace と同名 scalar の系列の区別、チャネル内の序数、チャネル別の解決済み定義と順序付き key 列を用語として整理。実装手順・検証規則は持ち込まない |
| `docs/adr/0037-metrics-trace-channel-and-session-end-event.md` | scalar 定義名の改名と新名優先の互換読み取り、trace 宣言検証、順序保証範囲、系列識別、SHARED の group / env_index の区別を同期。実装状態は記載しない |
| `docs/adr/0034-eval-session-aggregation-in-batchenv-decorator.md` | ADR 0037 で予定するイベント分離と購読抽出の移行を追記し、元の決定時点の記述と区別 |
| `docs/adr/0029-analysis-metadata-emitted-by-runner.md` | scalar 定義 tag の改名、trace 定義の別レコード化、fallback 削除条件の新名への読み替えを ADR 0037 参照付きで追記。実装状態は記載しない |

旧決定が採用契約として残っていないこと、文書間の相互参照、差分・改行を確認する。過去の決定や Run artifact を新仕様の実行証拠へ書き換えない。

### 11.2 コード実装時に同期する対象

| 対象 | 変更 |
|---|---|
| `docs/design/140_observability.jp.md` §6.x | `@session_end`、trace DSL・行契約、`metrics.scalar.defs` / `metrics.trace.defs`、eval scope、系列識別を code / config と同じ変更で同期 |
| `docs/design/030_user_guide_analysis.jp.md`、`AGENTS.md` | eval scalar の `@episode_end` を `@session_end` へ。現行実装を説明するため今回は更新しない |
| §4.5・§4.8 の現用設定・サンプルと対応テスト | eval scalar のイベント移行、Atari eval trace 宣言と既定 OFF の train コメント例、DSL 説明をコードと同時に更新 |
| `viewers/metrics-tools/inspect_run.py` / `inspect_run_test.py` | master / cache で新名優先・新名不在時だけ旧名を読み、fixture を両形式で検証。cache 未構築の新 Run の設定導出に必要な `session_end` も追加・検証 |

## 12. スコープ外（再掲）

- 走行中の viewer で裾を見ること（移動窓統計は別件。trace があれば後から consumer として乗る）。
- 希少事象の全軌跡捕捉（932）。
- inspect_run / Viewer の trace 読み取り（§8 のゲート）。
- `min.` / `std.` 以外の集約 prefix。集約 prefix の std は「同時完了した env 間のばらつき」であり、欲しいのは時間方向の分布で、それは trace から算出する。
