# PRD 054: AI向けRun解析CLI `inspect_run.py`

- 起票日: 2026-08-15
- 改訂日: 2026-08-15（v2）
- 状態: implementation ready
- 対象: `viewers/metrics-tools`、`core/anet-core`（metrics定義ダンプ）、AIエージェント向けRun分析導線
- Topic Issue: 開発プロセス / 開発ツール / テスト基盤 `#7`、可視化 / メトリクス `#13`
- 関連: ADR 0015（Metrics Viewer cache）、ADR 0021（Runner workspace）、ADR 0029（解析メタデータはRunnerが出力する）
- 設計文書: `docs/design/030_user_guide_analysis.jp.md`、`docs/design/140_observability.jp.md`、`docs/design/210_metrics_viewer.jp.md`

## 改訂履歴

v1は単一CLIとして起票し実装まで到達したが、初回の実使用で次の2点が判明したためv2へ改訂した。

1. **step軸だけではmetricの座標系が決まらない**。`51_eval1/*` は `@event` の違いで2つの座標系へ分裂しており、config上は区別できない。v1の「解決済みstep軸ごとにgroup化」は誤った100%解決を生む。
2. **設定ファイルからの意図の再導出そのものが不健全**。`.$` マージ選択はdumpに残らず、CLI由来の `.$` だけが残るため、config fileとdumpのどちらを読んでも単独では誤読しうる。

v2ではCLIをsubcommandへ分割し、metricの解決済み定義をRunner自身が出力する契約へ移した。v1で定義したRun解析プロファイルは廃止した。

## Problem Statement

Run分析を依頼されたAIエージェントは、分析そのものへ入る前に次のリポジトリ固有知識を個別に調査している。

- 現行Runが `apps/runner/workspaces/<workspace>/runs/<run>` に配置されること。
- 同名Runが複数workspaceに存在し得ること。
- 実効設定の正本が編集後の設定ファイルやRun名ではなく、Run artifactの `config/config_data.txt` であること。
- scalarの正本が `metrics.jsonl` または `metrics.jsonl.gz` であり、rawが存在するときはrawを優先すること。
- `metrics_cache.db` はMetricsマスタから再生成可能な派生cacheであり、完全に追随している場合だけ解析入力として利用できること。
- 複数tagを同じstep範囲で比較し、tagごとのstep軸も確認する必要があること。

この調査を都度行うと、Run directoryを発見できない、巨大なRun treeを再帰検索する、JSONLをtagごとに何度も走査する、古いcacheを正本として扱う、編集後の設定をRun設定と誤認する、といった失敗が起きる。特に数百MB規模のDropMerge Runでは、単純な抽出方法の違いが解析時間とAIコンテキスト量へ大きく影響する。

### v2で追加された問題

**同じ軸名でもRunnerが違えば別座標系である。** `EvalRunner::DoStep()` は `@train` 系eventへ自分の `step_counts_` を、`@episode_end` へ呼び出し元から渡された `event_counts`（train runnerのもの）を載せる。したがって `51_eval1/13_double_suika_created_mean`（`@episode_end $exp_step`）と `51_eval1/41_noop_uqe_win_rate`（`@train $exp_step $action_info`）は、どちらも `$eval.[eval1]` かつ `$exp_step` と書かれていながら、実測で最大stepが 19,993,856 と 151,185 という別の座標系に落ちる。両者の比はRun中に 0.000039 から 0.0075 へ単調にドリフトするため、定数倍換算も成立しない。

configには「どのRunnerのcountsか」を表現するtokenが存在しないため、この区別は `@event` と `$runner_scope` の組から間接的に導くほかない。解析側がこの導出をPythonで再実装すると、C++側の解決規則と二重管理になり、仕様変更のたびに乖離する。

**設定からの再導出は原理的に不健全である。** `AutoMerge()` は `.$` で終わるキーを新しいmapへコピーしないため、どのprofileが選ばれたかという情報はdumpから消える。実Runでは、config fileが `app.$ = app.online > P`、dumpの最終行が `app.$ = app.batchrun`、実効値は `app.batchrun` 由来（起動batのCLI引数が原因）という状態が実在する。config fileを正本にしてもdumpを正本にしても、単独では逆の結論になる。

**発見手段が無い。** v1のCLIはRun名とtag名を入力として要求するが、それらを列挙する手段を持たない。利用者はRun一覧を `ls` で、tag一覧を `config_data.txt` のgrepで得ており、ツール外作業の主要因になっていた。

## 0. 決定一覧

| ID | 決定 |
|---|---|
| D1 | public CLIは `viewers/metrics-tools/inspect_run.py` 1本とし、`runs` / `tags` / `config` / `metrics` の4 subcommandへ分割する |
| D2 | Runは1個以上のRun名または既存directory pathで指定する。Run名の探索範囲は現行workspace配置の直下だけとする |
| D3 | 各subcommandは自分の関心だけを扱い、必要のないsourceを開かない |
| D4 | 複数Run、複数metric、複数rangeを1回の実行で扱い、Metricsマスタ fallbackはRunごとに1 passとする |
| D5 | JSON schema v2を機械可読な正本とし、Markdownは同じresult modelから生成する |
| D6 | metric rangeの統計は全有効点から計算し、曲線形状確認用に最大128点の決定的な間引き系列も返す |
| D7 | **step座標系はRunnerとstep軸の組で識別する。** 軸名だけで同一性を判定しない |
| D8 | **metricの解決済み定義はRunnerが `metrics.defs` として出力し、解析側はそれを正本とする。** 設定からの導出は当該レコードが無いRunへのfallbackに限る |
| D9 | 完全にcurrentなSQLite cacheだけをread-only利用し、それ以外はMetricsマスタへfallbackする。cacheの再構築・削除・更新は行わない |
| D10 | 実効設定の判定は `config/<module>.txt` との突合で行い、判定できないものを「実効でない」と断定しない |
| D11 | 実装時に `AGENTS.md`、`CONTEXT.md`、Run分析ユーザーガイド、可観測性設計文書を同時更新する |

v1の「publicなsubcommandは設けない」（旧D1）と「再利用する抽出条件を外部JSONのRun解析プロファイルとする」（旧D9）は撤回した。

## 1. CLI契約

### 1.1 Entry point

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py <subcommand> [RUN ...] [options]
```

| subcommand | 位置引数 | 役割 |
|---|---|---|
| `runs` | 0個以上のRUN | Run発見とartifact状態。引数なしで全workspaceのRunを列挙する |
| `tags` | 1個以上のRUN | metric tagのinventory。宣言された定義と観測範囲を返す |
| `config` | 1個以上のRUN | 実効設定の抽出とRun間差分 |
| `metrics` | 1個以上のRUN | scalarの抽出、range集約、Run間比較 |

全subcommand共通のoption。

| option | 回数 | 意味 |
|---|---|---|
| `--format json\|md` | 0または1 | 出力形式。既定は `json` |
| `--output PATH` | 0または1 | stdoutの代わりに保存する出力先 |

subcommand固有のoption。

| subcommand | option | 回数 | 意味 |
|---|---|---:|---|
| `runs` | `--workspace WS` | 0または1 | 列挙対象をこのworkspaceへ限定する |
| `tags` | `--no-observed` | 0または1 | 観測範囲を取らず、宣言情報だけを返す |
| `config` | `--config-key KEY_OR_GLOB` | 0以上 | 実効configのflat keyまたはcase-sensitive glob |
| `config` | `--diff` | 0または1 | 複数Run間で異なるconfig keyを返す |
| `config` | `--effective-only` | 0または1 | `effective: true` と判定できたkeyだけを返す |
| `metrics` | `--metric TAG_OR_GLOB` | 0以上 | 抽出するscalar tag。globを受理する |
| `metrics` | `--range SPEC` | 0以上 | 集約range。複数指定時は各rangeを独立集約する |
| `metrics` | `--range-mode MODE` | 0以上 | 導出規則によるrange |
| `metrics` | `--stat NAME` | 0または1 | 比較表のcellに使う統計。既定は `mean` |
| `metrics` | `--series` | 0または1 | 間引き系列を出力へ含める |

同じ `--metric`、`--config-key`、`--range`、`--range-mode` の重複は、最初の出現位置を残して除去する。`--range` と `--range-mode` の指定順はoptionを跨いで保持する。

### 1.2 Run解決

各 `RUN` は次の順序で解決する。

1. current working directory基準または絶対pathとして既存directoryへ解決できる場合は、そのdirectoryをRun directoryとする。
2. directoryへ解決できなければ、repo rootの `apps/runner/workspaces/*/runs/<RUN>` に完全一致するdirect childを探索する。
3. 候補が0件なら、入力値と探索rootを含むエラーにする。
4. 候補が複数なら、全候補の絶対pathを含む曖昧性エラーにする。任意のworkspaceを暗黙選択しない。

`workspace/run_name` のような独自shorthandは追加しない。`apps/runner/runs_*` 等のlegacy/archive配置は再帰探索せず、明示directory pathでだけ利用可能とする。

pathとして解決したRunがworkspace配下にある場合はworkspace名を結果へ含める。workspace外の明示pathでは `workspace` を `null` とする。

`runs` を位置引数なしで実行した場合は、`apps/runner/workspaces/*/runs/` 直下の全directoryを列挙する。`--workspace` 指定時はそのworkspaceへ限定する。列挙ではMetricsマスタの有無でRunを絞らない。

### 1.3 出力先と終了値

- `--output` 未指定時は結果だけをstdoutへ出す。警告と診断はstderrへ出し、JSONを汚染しない。
- `--output` 指定時は親directoryが既に存在することを要求する。対象fileは上書き可能とする。
- file出力は対象と同じdirectoryに一時fileを作成し、flush・close後にatomic replaceする。途中失敗で既存fileを壊さない。一時file作成自体の失敗も実行時失敗として扱い、tracebackを露出させない。
- `argparse` の構文エラー、range/range-modeの構文エラー、Run未発見・曖昧性、`--output` の親directory不在は終了値 `2` とする。
- source read、SQLite query、JSONL parse、出力書込み、unknown座標系へのrange解決失敗等の実行時失敗は終了値 `1` とする。
- 一部Runだけでtag/config keyが欠損する場合はresult内を `missing` として終了値 `0` とする。
- 明示したmetricまたはconfig selectorのいずれかが全Runで1件も成立しない場合は、resultを出力したうえで終了値 `1` とする。`quarantined` は有効prefixを返せているため成立扱いとする。

## 2. `runs`: Run発見とartifact状態

Metricsマスタのline走査と `config_data.txt` の値dumpを行わない。各Runについて次を返す。

- 入力文字列、Run名、workspace名、Run directoryの絶対path。
- `config/config_data.txt` の絶対path、存在有無、size、mtime、存在する場合のSHA-256。
- `metrics.jsonl` / `metrics.jsonl.gz` の存在状態と、raw優先契約で選ばれたMetricsマスタのpath・kind・size・mtime。
- `metrics_cache.db` のpath、存在有無、size、mtime、検証できた場合のcache状態・理由・`source_meta`。
- Run directory直下の `*.log`、`*.anet` のpath・size・mtime。directoryを再帰走査しない。

`agent_close.anet` の有無と mtime はRunの完了・中断判定の材料になるため、file一覧に含める。

Metricsマスタが無いRunもartifact診断対象とする。

## 3. step座標系とmetrics定義

### 3.1 step座標系

metric点のx座標の同一性は、**step軸名だけでなく、そのカウンタを所有するRunnerとの組**で決まる。本PRDではこの組を**step座標系**と呼び、`(runner, step_axis)` で表す。

2つのtagを同じx軸上で比較してよいのは、step座標系が一致するときだけである。相対range（percentage、末尾相対、`common`）の解決も、step座標系ごとに独立に行う。

### 3.2 `metrics.defs`（C++側の新規契約）

Runnerは、metrics observerを構築した後に、tagごとの解決済み定義を1レコードだけMetricsマスタへ出力する。

- 出力経路は既存の `MetricsLogger::Log(tag, ...)` のjson dump経路を使い、tagを `metrics.defs` とする。
- レコード形式は既存の `type: "json"` をそのまま使う。**新しいrecord typeを増やさない。** これによりMetrics Viewerのingestor、SQLite schema、cache契約は無変更で、レコードは `json_lines` テーブルへ入る。副作用として `json/metrics.defs.json` にも同じ内容がミラーされる。
- 出力タイミングは既存のconfig dump群と同じ初期化フェーズとする。

```json
{"type":"json","tag":"metrics.defs","timestamp":"2026-08-15T00:35:31","data":{
  "51_eval1/41_noop_uqe_win_rate": {
    "step_axis":"exp_step","runner":"eval1","event":"train",
    "target":"action_info","source_key":"action_uqe_win_rate.[0]",
    "ema_alpha":null,"interval":null},
  "51_eval1/13_double_suika_created_mean": {
    "step_axis":"exp_step","runner":"train","event":"episode_end",
    "target":"env","source_key":"mean.ep_double_suika_created",
    "ema_alpha":null,"interval":null},
  "42_env/00_ep_step_mean": {
    "step_axis":"train_step","runner":"train","event":"train",
    "target":"env","source_key":"mean.ep_step",
    "ema_alpha":null,"interval":null}}}
```

各fieldの契約。

| field | 型 | 意味 |
|---|---|---|
| `step_axis` | string | `train_step` / `learn_step` / `episode_step` / `exp_step` / `update_step` / `sim_step` のいずれか。既定解決後の値 |
| `runner` | string | そのtagのstep counterを所有するRunnerの識別子 |
| `event` | string | `train` / `learn` / `episode_end` |
| `target` | string または null | `agent` / `env` / `exp` / `update_result` / `runner` / `action_info` |
| `source_key` | string | 定義値のうちmetric keyとして採用されたtoken |
| `ema_alpha` | number または null | EMA無効時はnull。既定値を混ぜない |
| `interval` | integer | 解決済みの値。未指定時の既定は `1` |

`runner` の決定規則は、observer構築時に確定した内容をそのまま反映する。

- runner scopeが `EVAL` かつ eventが `train` の場合は、その eval 名（`eval1` 等）。
- それ以外（`@episode_end`、`@learn`、`$train` scope）は `train`。

この規則は「eventがどのRunnerのStepCountsを載せるか」に対応する。実装では `ObserverFactory` がobserver構築と同じ場所でこの組を確定させ、`ScalarMetricDef` として控える。Runnerは実際にattachしたものだけをレコードへ載せ、未スケジュールのeval tag（dormant）は除く。

### 3.3 `metrics.defs` を持たないRunのfallback

本改訂より前に生成されたRun artifactには `metrics.defs` が存在しない。互換の扱いを次で固定する。

- **互換対象**: 本改訂より前に生成されたRun artifact。
- **fallback内容**: 解決済み `metrics.scalar.[<tag>]` から `step_axis` / `runner` / `event` / `target` / `source_key` を導出する。導出規則は §3.4 とする。
- **表示**: 各metricへ `def_source: "metrics_defs" | "config_derived"` を立て、`config_derived` のRunについてRunごとに1回だけwarningを出す。
- **移行方法**: 新しいRunを実行する。既存Runへ後から `metrics.defs` を書き込むことはしない（toolはartifactを変更しない）。
- **削除条件**: 現用のRun作業セットが全て `metrics.defs` を持つようになった時点でfallback実装を削除する。

### 3.4 config由来のfallback導出規則

対象は解決済みキー `metrics.scalar.[<tag>]` のみとする。`metrics.scalar.baseline.` 等のマージ元キーと、`M.[` 等のoverlay生キーは読まない。

値はスペース区切りのtoken列で、token順序は自由である。

- step軸: `$train_step` / `$learn_step` / `$episode_step` / `$exp_step` / `$update_step` / `$sim_step`、または `step:` / `step_axis:` 属性。
- event: `@train` / `@learn` / `@episode_end`、または `event:` 属性。未指定は `train`。
- runner scope: `$train`、または `$eval.[<name>]`。未指定は `train`。
- target: `$agent` / `$env` / `$exp` / `$batch_experience` / `$update_result` / `$batch_update_result` / `$result` / `$runner` / `$action` / `$action_info`。
- EMA: `$ema` フラグ、`ema_alpha:` 属性。
- `interval:` 属性。
- 上記いずれにも該当せず `:` 区切りが2要素にならないtokenをmetric key（`source_key`）とする。複数ある場合は最後を採る。

step軸が明示されない場合は、`event == train` なら `train_step`、`learn` と `episode_end` なら `exp_step` とする。

`runner` は §3.2 と同じ規則で決める。すなわち runner scopeが `eval.[name]` かつ eventが `train` のときだけ eval 名、それ以外は `train`。

解決できない場合は `step_axis` と `runner` をともに `unknown` とする。

## 4. Metrics sourceとcache選択

### 4.1 Metricsマスタ

Metricsマスタの選択は既存 `viewers/metrics-tools/metrics_source.py` の `resolve_run_metrics()` と `open_metrics_binary()` を再利用する。

- `metrics.jsonl` があればrawを選ぶ。
- rawがなく `metrics.jsonl.gz` があればgzipを選ぶ。
- 両方なければMetricsマスタなしとする。

Runごとに必要な全tagをset化し、Metricsマスタ fallback時は1 passだけstreaming走査する。tagごと、rangeごとにfileを開き直さない。

### 4.2 cache eligibility

`metrics_cache.db` はread-only URI modeで開く。次の全条件を満たす場合だけmetric sourceとして利用する。

1. SQLite `application_id == 0x414E4554`、`user_version == 1`。
2. 現行Metrics Viewer schema v1の必須table・columnが存在する。
   - `tags`
   - `scalars`
   - `scalars_lod`
   - `tag_stats`
   - `json_lines`
   - `source_meta`
3. `source_meta` の必須keyが存在し、型変換可能である。
   - `generation`
   - `state`
   - `source_kind`
   - `source_size`
   - `source_mtime`
   - `source_head_sha256`
   - `source_commit_tail_sha256`
   - `committed_offset`
4. `state == ready`。
5. 選択済みMetricsマスタと `source_kind`、size、mtime、head fingerprint、commit-tail fingerprintが一致する。
6. `committed_offset == source_size == Metricsマスタのsize`。

fingerprintはMetrics Viewerの現行計算規則に合わせる。headは先頭 `min(現在size, source_size, 65536)` byte、commit-tailは `committed_offset` 直前の最大 65536 byte のSHA-256（小文字hex）で、gzipも展開せず圧縮fileの生byteをhashする。`source_mtime` はミリ秒である。

条件を満たさないcacheは `absent`、`invalid`、`partial`、`stale`、`error` のいずれかとして理由付きで記録し、Metricsマスタへfallbackする。

toolはcacheの作成、migration、checkpoint、修復、削除、更新を一切行わない。cache queryは1 Runにつき1 read transactionとする。

- `metrics`: `tags` とL0の `scalars` から選択tagの全点を読む。統計をLODや `tag_stats` から復元しない。
- `tags`: `tags` と `tag_stats` から全tagの `count` / `min_step` / `max_step` を読む。これはinventory用途であり、rangeを持つ統計の算出には使わない。
- `metrics.defs`: `json_lines` から `tag = 'metrics.defs'` の行を読む。

### 4.3 実行中Runのsnapshot

metric query開始時にMetricsマスタのsizeとmtimeを取得する。

- rawは開始時sizeを読み取り上限とし、それ以後の追記を同じ結果へ混ぜない。上限内の未終端末尾行は取り込まず、resultを暫定とする。
- gzipはimmutable sourceとしてEOFまで読む。未終端行またはgzip破損はsource errorとする。
- query終了後にsizeとmtimeを再取得し、開始時から変化した場合は `source_changed_during_read: true` とwarningを返す。自動retryはしない。
- cache判定は開始時snapshotと比較する。cacheが開始時sourceへ完全追随していなければmaster fallbackとする。

## 5. metric抽出契約

### 5.1 scalar入力

対象はMetrics masterの `type == "scalar"` かつ選択tagに一致するrecordである。stepは非負integer、valueはboolを除く有限数値を要求する。

- 非数値、非finite、float32範囲外のvalueはMetrics Viewerと同様に除外し、除外数をtag診断へ記録する。
- 選択対象外のrecord typeとtagはparse後に無視する。
- JSON不正、必須field欠落、不正step等のsource構造違反はRun単位のsource errorとし、黙って読み飛ばさない。ただしrawのsnapshot末尾にある未終端行は§4.3の暫定契約を適用する。
- 同一tag内でstepが逆行した場合、そのtagを `quarantined` とし、逆行前の有効prefixだけを公開する。他tagの走査は継続する。

cache経路でも `tags.status == error` のtagは `quarantined` として、cacheにcommit済みの有効prefixを返す。cacheとmaster fallbackでstatus、統計、系列の意味を揃える。

### 5.2 tag selectorとinventory

`--metric` と `--config-key` は同じmatching規則を使う。

- glob meta characterは `*` と `?` だけとし、`[` と `]` はリテラル文字として照合する。実効configとmetric tagはどちらも `[tag]` 記法を含むため、`fnmatch` のcharacter class解釈を使わない。
- case-sensitiveとする。
- 結果はselector順、その中では既知tag順・config file出現順とする。同じ対象は1回だけ返す。
- Runごとの不成立はselector単位で `missing` とする。

`--metric` のglob展開に使う既知tag集合は、次の優先順で決める。

1. `metrics.defs` があればそのkey集合。
2. なければcacheが利用可能な場合の `tags` テーブル。
3. どちらも無ければMetricsマスタの1 pass走査で観測されたtag集合。

glob meta characterを含まないselectorは展開せず、そのtagを直接要求する。既知tag集合を得られない場合でも、完全一致指定は動作する。

### 5.3 range

`--range` は両端を明示する指定、`--range-mode` は観測データから導出する規則とする。どちらも指定しない場合、各tagの全観測範囲を1 rangeとして集約し、labelを `all` とする。

`--range` の形式は `START:END` とし、両端inclusiveとする。**重なるrangeでは境界点が両方に数えられる。**

| 端点の形 | 意味 |
|---|---|
| 非負整数 + 任意の `K`/`M`/`G` suffix | 絶対step。suffixはcase-insensitiveで `K=1,000`、`M=1,000,000`、`G=1,000,000,000` |
| 上記に `%` を付けた有限decimal | step座標系の最大観測stepに対する百分率 |
| 先頭に `-` | 最大観測stepからの相対。解決値は `max_step + 値` |
| 空 | 下端は `0`（百分率では `0%`）、上端は最大観測step（百分率では `100%`） |

- 同一range内で絶対値と百分率を混在させない。符号の違いは許可する。
- 両端が空の指定（`:`）は拒否する。
- 小数付きの絶対値、overflowは拒否する。
- 末尾相対の解決値が負になる場合だけ 0 へ切り上げる。上端はclampしない。観測範囲より後ろを指す絶対rangeは末尾1点へ丸めず、空として扱う。
- 解決後になお下端が上端を超える場合は、そのRun×tag×rangeを `status: empty` とし、warningを出す。同じ呼び出し内の他Runでは成立しうるため、実行を止めない。

`--range-mode` の値は次とする。

| mode | 意味 |
|---|---|
| `all` | 全観測範囲。既定と同じ |
| `common` | 同じstep座標系を持つ全Runの観測範囲の交差。`[max(min_step), min(max_step)]` |

`common` はRunが1個の場合そのRunの全観測範囲となる。交差が空の場合は `status: empty` とwarningを返す。

相対的な解決（百分率、負の端点、`common`）はstep座標系ごとに独立に行う。step座標系が `unknown` のtagへこれらを適用する場合は、tagとRunを示して実行時失敗とする。

resultには元の指定表現と解決後のabsolute boundsを両方残す。

### 5.4 統計と間引き系列

各Run×tag×rangeについて、range内の全有効点から次を計算する。

| field | 意味 |
|---|---|
| `count` | 有効点数 |
| `mean` | arithmetic mean |
| `population_std` | 分母 `count` の標準偏差 |
| `min` / `max` | 最小値 / 最大値 |
| `first` / `first_step` | 最初の値 / step |
| `last` / `last_step` | 最後の値 / step |
| `min_step` / `max_step` | range内で観測したstep範囲 |

集約はfloat64のonline accumulatorを用い、全点listを統計専用に複製しない。点が0件の場合は `status: empty`、`count: 0` とし、他の統計fieldはJSONで `null` とする。NaNやInfinityをJSONへ出力しない。

tag全体の観測範囲（range適用前の `min_step` / `max_step` / `count`）もmetric単位で返す。range指定時にstep座標系のずれを即座に検知できるようにするためである。

曲線形状確認用の `series` は最大128点とし、`--series` 指定時だけ出力へ含める。

1. 点数が128以下なら全点を元の順序で返す。
2. 128を超える場合は序数を42個の連続bucketへ等分する。
3. 各bucketから最小値点、最大値点、末尾点を元の序数順に採用する。同じ点は1回へ畳む。
4. 全系列の先頭点と末尾点を必ず保持する。

bucket数42は、重複が一切無い場合でも `42 * 3 + 1 = 127` 点で128点上限に収まる最大値である。同値候補では序数が小さい点を採用する。同一入力に対してcache/master、実行順、Python processによらず同じ系列を返す。

### 5.5 Run間比較

`metrics` は、range×tagごとにRunを横断した比較を返す。

- cellの値は `--stat` が指定する統計とし、既定は `mean`。受理する値は `mean` / `last` / `first` / `min` / `max` / `count` / `population_std`。
- Runが2個の場合、2番目と1番目の差 `delta` と、1番目を基準にした `delta_ratio` を返す。基準が0または欠損の場合は `null` とする。
- Runが3個以上の場合、Run横断の `mean` / `population_std` / `range` を返す。同一設定の反復Runのばらつき幅を、比較のその場で得るためである。
- 値が取れないRunは `null` とし、その理由（`missing` / `empty` / `source_missing` / `source_error`）を保持する。

## 6. 実効設定

実効設定の正本は `config/config_data.txt` とする。各非空行を最初の `=` で分割し、左辺をtrimしたflat key、右辺をtrimした文字列値として読む。値を数値や真偽値へ暗黙変換しない。

`config_data.txt` は実効値だけでなく、マージ元の定義namespace（`metrics.scalar.baseline.` 等）、未選択のprofile（`metrics.scalar.full.` 等）、CLI由来の `.$` 残骸を含む。`.$` による選択履歴はdumpに残らないため、**どのprofileが選ばれたかを `config_data.txt` 単独から復元することはできない。**

そのため実効判定は次の突合で行う。

- Run directoryの `config/` 直下にある `config_data.txt` 以外の `*.txt` を読む。これらは各moduleが実際に読んだ設定の dump であり、完全修飾キー形式である。
- 返す各keyについて、いずれかのmodule dumpに同じkeyが存在すれば `effective: true` とする。
- 存在しない場合は `effective: null`（不明）とする。**`false` とは言わない。** module dumpを出していない領域（`net.*` 等）が存在するためである。
- `--effective-only` は `effective: true` のkeyだけを返す。

`--diff` は全Runのkey unionについて、文字列値または存在有無がRun間で異なるkeyだけを返す。欠損は値 `null` とは別の `present: false` で表現する。`--config-key` 併用時は、そのselector群に一致するkeyだけをdiff対象とする。Runが1件の場合は空のdiffを正常結果として返す。

## 7. result model

JSON rootは共通のenvelopeを持ち、`subcommand` によって内容が決まる。field名はschema v2のpublic contractとする。

```json
{
  "schema_version": 2,
  "subcommand": "metrics",
  "generated_at": "2026-08-15T12:34:56+09:00",
  "ranges": [
    {"label": "-4M:", "kind": "relative"}
  ],
  "runs": [
    {
      "input": "run_name",
      "run_name": "run_name",
      "workspace": "dm-iqn",
      "run_dir": "...",
      "def_source": "config_derived",
      "metrics_source": {
        "selected": "cache",
        "master_path": "...",
        "cache_path": "...",
        "cache_status": "current",
        "cache_reason": null,
        "provisional": false,
        "source_changed_during_read": false
      },
      "metrics": [
        {
          "tag": "51_eval1/41_noop_uqe_win_rate",
          "step_axis": "exp_step",
          "runner": "eval1",
          "def_source": "config_derived",
          "source_key": "action_uqe_win_rate.[0]",
          "status": "ok",
          "excluded": 0,
          "observed": {"count": 151185, "min_step": 1, "max_step": 151185},
          "ranges": []
        }
      ],
      "warnings": []
    }
  ],
  "comparison": [],
  "warnings": []
}
```

配列順はRun入力順、metric選択順、range指定順を保持する。pathは絶対pathで出力する。statusは少なくとも `ok`、`missing`、`empty`、`quarantined`、`source_missing`、`source_error` を区別する。

`runs` / `tags` / `config` はそれぞれ `runs[].artifacts`、`runs[].tags`、`runs[].config` を主とし、関係のないsectionを持たない。

Markdownは同じresult modelから生成する。`metrics` のMarkdownは次の順序とする。

1. 実行条件（生成時刻、range一覧、`--stat`）。
2. Runごとのsource・cache状態。
3. **rangeごとの比較表**（行=tag、列=Run、末尾にdelta/delta_ratioまたはmean/std/range）。
4. Run×tag×rangeの詳細table。`range_status`、`source_key`、tag全体の観測範囲を列に含める。
5. `--series` 指定時のみ、tagごとの間引き系列をcompactな `step:value` 列として記載。
6. warning一覧。missingとemptyもここへ1行ずつ出す。

Markdownだけに存在する解析判断や自動コメントを追加しない。

## 8. 実装範囲

### 8.1 コード

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/src/observers.cpp` ほか | metrics observer構築後に、解決済み定義を集約して `metrics.defs` として1レコード出力する |
| `viewers/metrics-tools/inspect_run.py` | subcommand分割、Run列挙、tag inventory、metrics.defs読み込みとconfig fallback、range解決、実効config判定、比較、JSON/Markdown renderer |
| `viewers/metrics-tools/inspect_run_test.py` | `unittest` と一時fixtureによる主要CLI契約テスト |

既存 `metrics_source.py` をsource選択・openに再利用する。単一CLIのために新規package、component file、共通frameworkを作らない。第三者依存を追加せずPython標準libraryだけで実装する。

C++側とPython側は独立に実装できる。Python側は `metrics.defs` が無くてもconfig fallbackで動作する。

### 8.2 実装時の文書更新

| ファイル | 変更内容 |
|---|---|
| `AGENTS.md` | Run名だけの提示は原則Run分析依頼と扱うこと、subcommandごとの実行例、metricは1回の呼び出しへ束ねること、探索範囲、config dumpのsection規約、eval tagのstep座標系、正本関係、read-only保証を追記 |
| `CONTEXT.md` | 用語「step座標系」を追加 |
| `docs/design/030_user_guide_analysis.jp.md` | subcommand構成での利用手順 |
| `docs/design/140_observability.jp.md` | runner scopeが違えばStepCountsの出所も変わること、`metrics.defs` の契約 |
| `docs/design/010_framework_overview.jp.md` | step軸がグローバルに一意ではないこと |

## 9. テストと受け入れ基準

`inspect_run_test.py` は実Runに依存せず、一時repo/workspace/Run fixture、config、raw/gzip、SQLite schema v1 cache、`metrics.defs` レコードを生成する。

### 9.1 Run・tag・config

- 引数なしの `runs` が全workspaceのRunを列挙し、`--workspace` で限定できる。
- 既存相対path、絶対path、workspace内の一意なRun名を解決できる。legacy directoryは明示pathで解決でき、Run名探索では見つからない。同名Runが複数workspaceにある場合は候補pathを含めて終了値2となる。
- `runs` がMetricsマスタをopenせず、artifact metadataとconfigのSHA-256、cache状態を返す。
- `tags` が `metrics.defs` から `step_axis` / `runner` / `source_key` を返す。`--no-observed` で観測範囲を取らない。
- `metrics.defs` が無いRunでconfig fallbackが働き、`def_source: "config_derived"` とwarningが出る。
- config完全一致、glob、`[tag]` リテラル、欠損、全key diff、selector限定diff、欠損対存在、`effective` 判定と `--effective-only` を検証する。

### 9.2 source・cache

- rawとgzipがあるとrawを選び、gzipだけならgzipを選ぶ。
- 完全currentなcacheをread-only利用し、masterを走査しない。
- cache不存在、partial、stale、error、application ID/schema不正ではmasterへfallbackする。
- cache利用前後でRun artifactのsize、mtime、内容が変わらない。
- raw追記中snapshotは開始時sizeまでを読み、未終端末尾を除外し、source変化と暫定状態を返す。
- 同一fixtureについてcache経路とraw/gzip経路のstatus、統計、seriesが一致する。

### 9.3 step座標系・range・出力

- **同じ `step_axis` でも `runner` が異なるtagを同時に指定したとき、百分率rangeがそれぞれの最大観測stepから独立に解決される。**
- 絶対rangeの `K/M/G`、両端inclusive、境界1点を検証する。
- 開端点 `:20M` / `10M:`、負端点 `-4M:` / `:-4M`、負の百分率を検証する。
- `--range-mode common` が同一step座標系のRun交差を返し、交差が空なら `empty` になる。
- 単位混在拒否、両端空の拒否、unknown座標系での相対range拒否を検証する。
- 手計算可能な点列で `count`、mean、population std、min/max、first/last、step範囲を検証する。
- 128点以下では全点を保ち、129点以上では128点以内、先頭・末尾保持、決定性を検証する。`--series` 未指定では系列を出さない。
- 非数値・非finite値の除外、step逆行tagのquarantine、有効prefix公開を検証する。
- 比較表が2 Runでdelta/delta_ratioを、3 Run以上でmean/std/rangeを返す。
- 一部Runのtag欠損は終了値0、全Run欠損は結果を出力して終了値1となる。
- JSONがstrict parse可能でNaN/Infinityを含まず、Markdownが比較表と詳細表の両方を含み、`range_status` を持つ。
- `--output` が既存fileを置換し、書込み失敗時には既存内容を保持し、tracebackを出さない。

標準検証コマンドは次とする。

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run_test.py
git diff --check
```

## 10. スコープ外

- Metrics Viewer UI、HTTP API、Java実装、SQLite schemaの変更。`metrics.defs` は既存の `type: "json"` を使うためこれらの変更を必要としない。
- cacheの生成、更新、修復、migration、削除。
- 既存Run artifactへの `metrics.defs` の後追い書き込み。
- metricの意味解釈、異常閾値、採否判断、自然言語分析、Run ranking。
- raw全点の無制限出力、plot画像生成、interactive viewer。
- 実Run依存test、網羅的fuzz、性能benchmark。ただしstreaming・1 pass・128点上限の契約はfocused testで確認する。
- configの `.$` マージ選択履歴の復元。原理的に不可能であり、実効判定はmodule dumpとの突合に留める。

## 11. Further Notes

- `inspect_run.py` はRun分析の材料を安定して抽出するtoolであり、分析者が確認すべき「Run成立性、主目的score、変更機構、Env挙動、実時間性能」を一つのscoreへ統合しない。
- 相対range（百分率、末尾相対）は異なる長さのRunを相対進捗で眺める補助である。ハイパラ比較の正式判断では `--range-mode common` か、同じstep座標系のmatched absolute rangeを併記する。
- `metrics_cache.db` は高速経路であって正本ではない。cache contractが将来変わった場合、未知schemaを推測して読むのではなくmaster fallbackする。
- `metrics.defs` は「設定にこう書いた」ではなく「実際にこう構築された」の記録である。設定の書き方が将来変わっても、解析側の読み方は変えなくてよい。
