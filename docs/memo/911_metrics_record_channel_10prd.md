# メトリクス record チャネル（統計とは別に「1 件 = 1 レコード」を出す）暫定 PRD

> 状態: 暫定メモ。優先度・着手時期・名前・レベル定義は未確定。本 PRD は実装着手を意味しない。
> 起点: 2026-08-25、`42_env/10_game_score_mean` が「128 env の平均」ではなく「その step で完了した env の平均」であると判明したこと。Breakout では 97% が単独完了のため実質エピソード素点の列として使えていたが、これは構成に依存した偶然である。
> 関連: `912_background_eval_snapshot_ordering_10prd.md`（snapshot 順序＝どの network version を採点したかが記録されていない）、`932_episode_forensics_10prd.md`（`EpisodeId` / `TransitionId` の伝播、「Episode Journal」の語）、`060_eval_batch_episodes_10prd.md`（eval の N エピソード化）、`054_inspect_run_10prd.md`（分析ツール）。

## Context（背景・目的）

スカラーメトリクスの行は `{step, tag, value}` の 3 要素しかなく、**主体（誰の値か）を書く欄が無い**。バッチ集約はこの欄が無いことを `mean.` / `max.` / `min.` の prefix で埋めているが、集約した時点で個体は失われる。

本 PRD は、統計（scalar）とは別に **「1 件 = 1 レコード」を識別子付きで出すチャネル**を設けることを提案する。

## 現行コードで確定している事実

### 1. エピソード確定キーは NaN ゲート、集約は NaN を除外する

`AtariEnv::GetScalar` は `game_score` / `game_len` / `game_frames` / `hns57` / `hns49` を実 game over / truncation 以外で NaN にする（[`AtariEnv.cpp:592-609`](../../core/envs/atari1/src/AtariEnv.cpp:592)）。バッチ集約は NaN を分母から除外し、全 env が NaN なら結果も NaN を返す（[`env.cpp:279-333`](../../core/anet-core/src/env.cpp:279)。prefix は `max.` / `mean.` / `min.` の 3 種のみで、無指定は fail-fast）。

したがって `mean.game_score` の分母は num_envs ではなく **その step で実際にゲームを終えた env の数**である。

### 2. 同時完了率は構成で変わる

λ = num_envs / 平均エピソード長。実測:

| | 単独完了 | 2 | 3 | 4 以上 |
|---|---|---|---|---|
| Breakout / 128 env（`run_20260825-002524`、n=36,419） | **97.1%** | 2.6% | 0.3% | 0.05% |
| DropMerge / 256 env（`run_20260819-173220` の `42_env/12_ep_frct_mean`、n=277,596） | 71.3% | 16.7% | 8.8% | 3.2% |

Breakout が単独に寄っているのは λ ≈ 0.07 だからで、**ゲームを変えても、num_envs を上げても、探索を強めても崩れる**（spatial exploration 導入でエピソード数は 2 倍になり λ も 2 倍になった）。Atari-5 はエピソード長がゲームごとに違うため、**同じタグがゲームごとに違う集約になる**。

### 3. 失われるのは平均値ではなく「同時分布」

2026-08-24 に FIRE デッドロックを否定した決め手は「低スコア帯は `game_len` も短い」という同時分布だった（`score<50` かつ `len>3000` が 0 本）。これは n=1 だから step で join できたにすぎない。n>1 では score が 3 行・len が 3 行出ても**どれとどれが対か分からない**。

### 4. 分析は全てオフラインの ad-hoc スクリプトで行われている

本記録に載っている分布統計（p10 / p25 / median / CV / `<100` 率 / `≥432` 率）は、いずれもロギングされたメトリクスではなく `metrics.jsonl` を直接舐めて算出したものである。`inspect_run.py` の range 集計は逐次累積（Welford）で count / mean / population_std / min / max / first / last までしか出せず、**分位点と閾値越え率を持たない**（[`inspect_run.py:1090-1135`](../../viewers/metrics-tools/inspect_run.py:1090), [`:2292`](../../viewers/metrics-tools/inspect_run.py:2292)）。

### 5. 容量は制約にならない

`run_20260825-002524_atari_breakout_apex_e04`（50M step）の `metrics.jsonl`:

| | 行数 | 全体比 |
|---|---|---|
| 全体 | 15,082,952 | 100% |
| エピソード確定 raw 5 タグ | 182,095 | **1.21%** |
| 実エピソード数 | 36,419 | 0.24% |

支配しているのは learn 側（311,908 点 × 約 25 タグ）と `_ema`（毎 round 発火）。1 エピソード 1 レコードにすると約 36k 行・5.5MB で、**現行の 5 タグ 182k 行より小さくなる**。

### 6. 出力の transport は既に存在する

- [`metrics_logger.hpp:177`](../../core/anet-core/include/anet/metrics_logger.hpp:177) に `void Log(const std::string& tag, anet::rl::step_t step, const json& data);` がある。
- 生成される行は `type:"json"` で、cache 側も `json_lines(ordinal, type, tag, step, timestamp, json)` で素通しする。現状は config dump と resolution の 2 種（実測 15 行）が**スキーマ契約なしで**流れている。
- **per-env アクセサも既にある**: `DiscreteBatchEnvBase::GetScalar(key, index)` は `index >= 0` で個別 env を返す（[`env.cpp:274`](../../core/anet-core/src/env.cpp:274)）。env 側の改修は不要。
- 足りないのはメトリクス DSL の index セレクタで、現在の attr は `key` / `event` / `step` / `target` のみ（`observers.cpp` の ObserverFactory）。

### 7. `TraceSink` は既に別概念で使われている

```cpp
using TraceSink = std::function<void(std::string_view, const torch::Tensor&)>;
```
[`common.hpp:27`](../../core/anet-core/include/anet/common.hpp:27)

NN の層別 activation を viewer へ流すタップで、`MakeActionTraceSink` は `.slice(0, 0, 1)` で **env 0 だけ**を抜く（[`rl.cpp:20-25`](../../core/anet-core/src/rl.cpp:20)）。2026-07 のコアレビューで「常時 ON」を指摘済み。**「トレース」という語を新機構に当てると層 activation と衝突する。**

## 問題

### A. 集約が構成依存で、ゲーム横断比較が静かに壊れる

Atari-5 sweep では 5 ゲームで λ が異なるため、`mean.game_score` の意味がゲームごとに変わる。エラーも警告も出ない。

### B. 同時分布が復元できない（§3）

### C. provenance を書く場所が無い

スカラー行に主体欄が無いことは、本件以外でも同じ形で効いている。

| PRD | 欠けている識別子 | 現状の帰結 |
|---|---|---|
| 本 PRD | `env_index` | 集約で潰れる |
| 912 | **採点した network version** | 記録が無いため「順序を保証する」以外の解決が選べない |
| 932 | `EpisodeId` / `TransitionId` | Replay の Sample 元へ遡れない |

912 が重い順序保証へ向かっているのは記録欄が無いためであり、**レコード型があれば「保証する」の前に「記録して事後に判別する」が選択肢に入る**。

## 検討した 4 案

### ① step の小数部を env 軸に使う（不採用）

`step = 41547136 + env/128` のように個体を時間座標へ埋める。

- **利点**: 同じ小数部＝同じ env なので**同時分布が保存される**。かつ **viewer が無改修で 128 env 全部を描ける**（③④ に無い利点。ただし ② も同じ性質を持つ）。
- **不採用の理由**:
  - `scalars(tag_id, ordinal, step INTEGER, value REAL)` の step は INTEGER。スキーマ変更が要る。
  - `inspect_run.py` の `--range 10M:20M` 解析、LOD bucket、`(runner, step_axis)` 座標系（ADR 0029）が整数 step 前提。
  - どこかで int キャストされると **128 env が無言で 1 点へ潰れる**。
  - 時間座標に個体識別を載せる設計であり、tag と name を混同しない既存の規律に反する。
  - `lives`（毎 step × 128 env）まで一般化すると行数が 128 倍。
  - 結局スキーマを触るので ②③④ より安くならない。**②（識別子を別カラムで持つ）が同じ利点を座標を壊さずに得る**ので、①を採る理由は無い。

### ② scalar 行に識別子カラムを追加する

`step` は整数のまま据え置き、**対象を特定する値を別フィールドで持つ**。

```json
{"step":41547136,"tag":"42_env/10_game_score","type":"scalar","value":848,"env":37}
```

- **同時分布が復元される**。join キーが `(step, env)` になり、score 行と len 行が一意に対応する。① が狙っていたものを座標系を壊さずに得る。
- **座標系が無傷**。`step` は時間座標のまま。`--range 10M:20M`、LOD bucket、`(runner, step_axis)` 契約（ADR 0029）はどれも触らない。
- **viewer が（原理的には）無改修で 128 env 全部を描ける**。余分なフィールドを無視すれば同一系列の散布として出る。① の唯一の利点をこちらも持つ。
- **既存の reader は無改修で通る**。`inspect_run.py` の行解析は `tag` / `step` / `value` だけを見て他フィールドを捨てる（[`inspect_run.py:860-868`](../../viewers/metrics-tools/inspect_run.py:860)）。したがって既存の窓集計は「全 env をプールした集計」として**そのまま正しく動く**。
- **行数は増えない**。Breakout では 5 タグ × 36,419 エピソード = 182k 行で現行と同じ（現行が既にエピソードごとに 1 行だから）。増えるのは同時完了があるゲームだけで、それは増えるべきぶんである。
- **cache のスキーマ追加が要る**。`scalars(tag_id, ordinal, step, value)` に識別子欄が無い（[`inspect_run.py:544`](../../viewers/metrics-tools/inspect_run.py:544)）。ただし cache は ADR 0015 で**破棄可能な派生物**と規定されており、マスタ `metrics.jsonl` から再生成できる。移行ではなく再構築で済む。

**難点は一般性**。`env` 1 本で本 PRD の問題 A・B は解けるが、問題 C（provenance）は解けない。912 は `model_version`、932 は `EpisodeId` / `TransitionId` を要求しており、識別子の種類ごとにカラムを増やすのか、汎用の `subject` 欄にするのかを決める必要がある。後者へ倒すと実質 ④ に近づく。

また、1 エピソードの情報が **5 行に分かれたまま**である（④ は 1 レコードに畳む）。join で復元できるので情報は失われないが、原子性は無い。

### ③ `type:"json"` で env 別レコードを出す（暫定解）

既存 `Log(tag, step, json)` をそのまま使い、完了 env ごとに 1 レコード。

- **利点**: 実装は observer 1 本 + reader 1 本。最短。
- **難点**: **スキーマ契約なしの ad-hoc 行が 2 種類目になる**。識別子欄・レベル・読み手の規約が無いまま資産化し、④ が後で吸収する羽目になる。

### ④ record チャネルを設け、レベル付きで宣言する（推奨）

統計（scalar）と別チャネルとして「1 件 = 1 レコード」を契約付きで定義し、**出力レベルを config で宣言**する。

- ③ と **emission のコストは同じ**（transport が共通。§6）。違いは契約の有無だけ。
- 912 / 932 が同じ器に乗る。932 の `EpisodeId` はそのまま本チャネルの識別子欄になる。
- レベルがあることで、`TraceSink` が踏んだ「常時 ON で払い続ける」轍を避けられる。

## 目標契約（案。要確定）

```
metrics.scalar.$ = metrics.scalar.@baseline > M1 > M2      # 既存(統計)
metrics.record.$ = metrics.record.@baseline > R1           # 新設(記録)

metrics.record.[env_episode].level  = record               # off | record | forensic
metrics.record.[env_episode].fields = env,game_score,game_len,game_frames,lives
```

出力行の形（案）:

```json
{"type":"record","tag":"env_episode","step":41547136,
 "env":37,"episode_id":"...","runner":"train",
 "game_score":848,"game_len":2412,"game_frames":9651,"lives":0}
```

レベルの意味（案）:

| level | 内容 | 対応 |
|---|---|---|
| `off` | 出さない | — |
| `record` | 識別子 + 宣言されたスカラー欄 | 本 PRD |
| `forensic` | 上記 + payload（Action 列、Observation 要約など） | 932 |

## 決めるべきこと（未確定）

| # | 論点 | 備考 |
|---|---|---|
| D1 | **名前** | 「トレース」は `TraceSink`（NN activation）と衝突する（§7）。932 が既に「Episode Journal」を使っているので `record` / `journal` 系へ寄せるか。CONTEXT.md の用語追加も含む |
| D2 | level の enum と既定値 | 3 段でよいか。既定は `record` か `off` か |
| D3 | 識別子欄の必須集合 | `env` / `episode_id` / `runner` / `model_version` のどこまでを共通契約にするか。912 が要求するのは `model_version` |
| D4 | `fields` の宣言方法 | config で列挙するか、producer が決めるか。列挙するなら未知キーは fail-fast か |
| D5 | 既存の scalar 集約タグを残すか | viewer は当面 scalar しか描けない。`42_env/10_game_score_mean` 等を残すと二重出力になるが、削ると走行中に何も見えなくなる |
| D6 | `inspect_run.py` のリーダー | `records` サブコマンドを足すか、`metrics` を拡張するか。分位点・閾値越え率（§4 の穴）もここで埋めるか別件か |
| D7 | 既存 `type:"json"` 行（config dump / resolution）の扱い | 新契約へ寄せるか、無契約のまま残すか |
| D8 | 932 との境界 | 本 PRD を 932 の tracer bullet 第 0 段と位置づけるか、独立させるか |
| D9 | producer 第 1 号の範囲 | env エピソードのみか、eval の `model_version`（912）も同時か |
| D10 | viewer 対応 | 本 PRD の範囲外とするか。範囲外なら「走行中に裾は見えないまま」という状態が続く |
| D11 | **② で足りるか（案の選択そのもの）** | ② は問題 A・B を最小改修で解くが問題 C（provenance）を解かない。A・B だけ先に潰して C は 912 / 932 の着手時に改めて設計する、という分割も成立する。逆に ④ を選ぶなら ② は不要になる（record が識別子を内包するため）。**この判断が本 PRD のスコープを決める** |

## スコープ外

- **走行中の viewer で裾を見ること**。これは移動窓統計（runner 側で直近 N ゲームの分位点をスカラーとして出す）の話で、本チャネルとは別。record があれば後から consumer として乗る。
- **希少事象の全軌跡捕捉**（932）。本 PRD は常時 ON の軽量層のみ。
- **eval の N エピソード化**（060）。独立。
- `min.` / `std.` の集約 prefix 追加。集約 prefix の std は「同時完了した env 間のばらつき」であり、n=1 では 0 になる。欲しいのは時間方向の分布であって、これは record から算出する。

## 着手を迫る条件

- **Atari-5 sweep**。5 ゲームで λ が異なるため、現行のまま回すとゲーム横断の分布比較が成立しない。Breakout 単独で回している間は顕在化しない。
- num_envs を 128 から上げたとき。λ が比例して上がる。
- `060` が入って eval が N エピソードになったとき。eval 側も同じ集約問題を持つ。しかも spatial exploration 系の Run では **eval1 が唯一の可換軸**であり（train 生スコアは ε ラダー混在で方策の質を測らない）、060 単独で入れると **本数は増えるのに p10 / 閾値越え率が復元できなくなる**。本 PRD の有無で 060 の投資対効果が変わる。
