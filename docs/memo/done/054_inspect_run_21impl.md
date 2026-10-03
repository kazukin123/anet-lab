# PRD 054 v2 `inspect_run.py` 実装メモ

- 起票日: 2026-08-15
- 対象PRD: `docs/memo/054_inspect_run_10prd.md`（v2）
- 前版: `docs/memo/054_inspect_run_20impl.md`（v1実装。本メモが正本を引き継ぐ）
- 状態: 実装中

## 概要

v1で実装した単一CLIを、`runs` / `tags` / `config` / `metrics` の4 subcommandへ分割する。あわせて、metricのstep座標系をRunnerが出力する `metrics.defs` から読む契約へ移し、集約範囲の語を `window` から `range` へ改める。

C++側（`metrics.defs` の出力）とPython側は独立に実装できる。Python側は `metrics.defs` が無くてもconfig fallbackで動作する。本メモは両方を扱う。

## v1からの主な契約変更

| 変更 | 内容 |
|---|---|
| subcommand | `inspect_run.py <subcommand> [RUN ...]`。旧 `inspect_run.py RUN ...` は廃止 |
| profile | `--profile` とschemaを削除。CONTEXT.mdの用語も撤去済み |
| window → range | `--window` を `--range` と `--range-mode` へ分離。開端点・負端点を追加 |
| step座標系 | group化キーを `step_axis` から `(runner, step_axis)` へ変更 |
| schema | `schema_version` を 2 へ。`windows` → `ranges`、`comparison` 追加 |
| series | 既定で出力せず `--series` でopt-in |

## C++側の実装

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/observers.hpp` | `ObserverFactory::ScalarMetricDef` と `GetScalarMetricDefs()`、free関数 `ScalarMetricDefsToJson()` を追加 |
| `core/anet-core/src/observers.cpp` | scalar metricのparse結果を `scalar_metric_defs_` へ控える。token表記への変換と、counts所有Runnerの決定を `metrics_def_names` namespaceのヘルパに置く |
| `core/anet-core/src/trainer.cpp` | observer attach後に、attach済み定義だけを `MetricsLogger::Log("metrics.defs", ...)` で1レコード出力 |
| `core/anet-core/src/observers_test.cpp` | 解決済み定義とJSON payloadの2 test case |

固定点。

- `step_axis` はconfigの `$xxx_step` と同じ表記で出す。`anet::rl::toString(StepAxis)` は `train` / `exp` を返すので、そちらは使わず別のヘルパを持つ。
- counts所有Runnerの決定は `OwningRunner(scope, event, eval_name)` に閉じ込め、`EvalRunner::DoStep()` のどの分岐に対応するかをコメントで明示する。導出規則が2箇所に散らないようにする。
- dormantなeval tagはobserverをattachしないため、レコードにも載せない。判定はattachループと同じ `resolve_runner(...) == nullptr` を使う。
- JSON生成は `ScalarMetricDefsToJson()` へ切り出し、レコード形式そのものを単体テストする。呼び出し側はfilterして渡すだけにする。
- `interval` は常にintegerで出す（未指定時の既定は `1`）。`ema_alpha` はEMA無効時だけnullにし、既定値を混ぜない。
- `MetricsLogger::LogJsonInternal` が `round_numbers(precision=6)` を通すため、`ema_alpha` は6桁で丸まる。実配置の `0.001` / `0.005` / `0.01` / `0.1` はすべて保たれる。

## 実装契約の固定点

### range のparseと解決

- 端点は `[-]値[%]` または空文字。`--range` は `START:END` 形式で、`:` が無い、または両端が空の指定を拒否する。
- 絶対値は非負整数 + 任意の `K`/`M`/`G`（case-insensitive）。小数付きは拒否。
- 百分率は0以上100以下の有限decimal。負号は別に扱うため、絶対値部分が0..100であることを検査する。
- 同一range内での絶対値と百分率の混在を拒否する。符号違いは許可する。
- 解決は Run × step座標系ごとに行う。
  - 絶対の正値: そのまま。
  - 絶対の負値: `max_step + 値`。
  - 百分率: `max_step * 値 / 100` を `Fraction` で厳密に計算し、下端は `ceil`、上端は `floor`。負の百分率は `max_step * (100 + 値) / 100`。
  - 空の下端: 0。空の上端: `max_step`。
- 解決後 `[0, max_step]` へclampし、なお下端 > 上端なら `status: empty` + warning。実行は止めない。
- `--range-mode common` は、同じstep座標系を持つ全Runの `[min_step, max_step]` の交差。Run1件ならそのRunの全範囲。
- 相対解決（百分率・負端点・`common`）で座標系が `unknown` の場合は `RuntimeFailure`（終了値1）。
- 絶対のみで構成されたrangeはstep座標系を必要としないため、`unknown` でも解決できる。

### step座標系の解決

1. Metricsマスタまたはcacheの `json_lines` から `tag == "metrics.defs"` のレコードを探す。あれば `def_source = "metrics_defs"`。
2. 無ければ `config/config_data.txt` の解決済み `metrics.scalar.[<tag>]` から導出し、`def_source = "config_derived"` としてRunごと1回warning。
3. どちらも得られないtagは `step_axis` / `runner` をともに `unknown` とする。

`runner` の導出（fallback経路）は「runner scopeが `eval.[name]` かつ eventが `train` のときだけ eval 名、それ以外は `train`」。cache経路では `json_lines` を1 read transaction内で読む。master経路では1 pass走査中に拾う。

### 実効config判定

- Run directoryの `config/` 直下から `config_data.txt` を除く `*.txt` を読み、キー集合を作る。
- 返す各keyについて、集合に含まれれば `effective: true`、含まれなければ `null`。**`false` を返さない。**
- `--effective-only` は `true` のkeyだけへ絞る。

### 比較

- `--stat` の既定は `mean`。受理値は `mean` / `last` / `first` / `min` / `max` / `count` / `population_std`。
- Run 2件: `delta` = 2番目 − 1番目、`delta_ratio` = `delta / 1番目`。1番目が0またはnullなら `delta_ratio` は `null`。
- Run 3件以上: Run横断の `mean` / `population_std` / `range`（max − min）。有効値が2件未満なら `null`。
- 値が取れないRunは `null` とし `status` を保持する。

### 出力

- `schema_version: 2`、`subcommand` をenvelopeへ含める。
- JSONは `json.dump(..., allow_nan=False, ensure_ascii=False)`。
- `--output` は同じdirectoryへ一時fileを作りatomic replace。**一時file作成自体の失敗も捕捉して `RuntimeFailure` にする**（v1のB2）。
- Markdownはsubcommandごとにrendererを分ける。`metrics` は比較表と詳細表を常に両方出し、詳細表に `range_status`（v1のB1）、`source_key`、tag全体の観測範囲を含める。

### 既存実装の再利用

- `metrics_source.resolve_run_metrics()` / `open_metrics_binary()` はそのまま。
- `read_cache_series` が既に発行している `SELECT id, key, status FROM tags` を `tags` subcommandでも使う。観測範囲は `tag_stats` から同じtransaction内で取る。
- `compile_selector` を `--metric` のglobにも使う（`*` `?` のみmeta、`[` `]` リテラル）。
- `compute_stats` / `build_series` / `window_bounds` はrange解決結果を受ける形で継続利用。

## テスト

- **Public interface / surface**: 4 subcommandのCLI契約。引数、stdoutのJSON schema v2、Markdown、stderr、終了値、`--output` のfile内容。内部関数を直接叩くテストは書かない。
- **優先behavior**: subcommandの疎通、step座標系の分離、range解決、実効config判定、比較表、終了値。

### TDD順序

1. tracer bullet: `runs` を引数なしで実行し、workspace fixtureの全Runが列挙されて終了値0。
2. `runs` のRun解決（相対/絶対path、Run名、legacy、曖昧性）とartifact/cache状態。
3. `tags` が `metrics.defs` から `step_axis` / `runner` / `source_key` を返す。`--no-observed`。
4. `metrics.defs` 不在時のconfig fallbackと `def_source` / warning。
5. **同じ `exp_step` でも `runner` が違うtagを同時指定したとき、百分率rangeが別々に解決される**（本改訂の中核）。
6. `--range` の絶対・百分率・開端点・負端点、単位混在拒否、`--range-mode common`。
7. `config` の selector / diff / `effective` / `--effective-only`。
8. `metrics` md の比較表と詳細表、`range_status`、`--series` opt-in。
9. `--metric` のglob展開。
10. 出力と終了値、`--output` の書込み失敗時の既存内容保持。

v1の49テストのうち、profile系は削除、window系はrangeへ移行、他はsubcommand形へ追従させる。

## 検証

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run_test.py
```

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\metrics_source_test.py
```

```powershell
git diff --check
```

C++側は本メモの範囲外のためビルドしない。

## 前提

- `metrics.defs` が入るまで、全既存Runは `def_source: "config_derived"` で動作する。
- `--metric` のglobは、その Run の既知tag集合（metrics.defs → cache tags → master 1 pass）に対して展開する。完全一致指定は既知tag集合が得られなくても動作する。
- Git の add / commit / push は行わない。
