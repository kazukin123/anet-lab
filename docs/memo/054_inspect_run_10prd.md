# PRD 054: AI向けRun解析CLI `inspect_run.py`

- 起票日: 2026-08-15
- 状態: implementation ready
- 対象: `viewers/metrics-tools`、AIエージェント向けRun分析導線
- Topic Issue: 開発プロセス / 開発ツール / テスト基盤 `#7`、可視化 / メトリクス `#13`
- 関連: ADR 0015（Metrics Viewer cache）、ADR 0021（Runner workspace）、PRD 046（workspace）
- 設計文書: `docs/design/030_user_guide_analysis.jp.md`、`docs/design/210_metrics_viewer.jp.md`

## Problem Statement

Run分析を依頼されたAIエージェントは、分析そのものへ入る前に次のリポジトリ固有知識を個別に調査している。

- 現行Runが `apps/runner/workspaces/<workspace>/runs/<run>` に配置されること。
- 同名Runが複数workspaceに存在し得ること。
- 実効設定の正本が編集後の設定ファイルやRun名ではなく、Run artifactの `config/config_data.txt` であること。
- scalarの正本が `metrics.jsonl` または `metrics.jsonl.gz` であり、rawが存在するときはrawを優先すること。
- `metrics_cache.db` はMetricsマスタから再生成可能な派生cacheであり、完全に追随している場合だけ解析入力として利用できること。
- 複数tagを同じstep windowで比較し、tagごとのstep軸も確認する必要があること。

この調査を都度行うと、Run directoryを発見できない、巨大なRun treeを再帰検索する、JSONLをtagごとに何度も走査する、古いcacheを正本として扱う、編集後の設定をRun設定と誤認する、といった失敗が起きる。特に数百MB規模のDropMerge Runでは、単純な抽出方法の違いが解析時間とAIコンテキスト量へ大きく影響する。

一方、既存のMetrics Viewerは人間向けの可視化アプリケーションであり、AIエージェントがshellから構造化結果を取得するための安定したCLIではない。既存の補助scriptも個別用途向けで、Runの解決、実効設定、cache判定、複数Run・複数tag比較を一つの入口では扱えない。

本PRDでは、Run artifactを変更せずに検査・抽出するAI向けCLI `inspect_run.py` を定義する。既定実行は軽量なartifact inspectionに留め、metricが指定されたときだけcacheまたはMetricsマスタから必要なtagを一括抽出する。

## 0. 決定一覧

| ID | 決定 |
|---|---|
| D1 | public CLIは `viewers/metrics-tools/inspect_run.py` 1本とし、実装も同module内の機能グループとしてまとめる |
| D2 | Runは1個以上のRun名または既存directory pathで指定する。Run名の探索範囲は現行workspace直下だけとする |
| D3 | option未指定時はMetricsマスタを走査せず、artifact、実効config path、cache状態だけを返す |
| D4 | 複数Run、複数metric、複数windowを1回の実行で扱い、Metricsマスタ fallbackはRunごとに1 passとする |
| D5 | JSON schema v1を機械可読な正本とし、Markdownは同じresult modelから生成する |
| D6 | metric windowの統計は全有効点から計算し、曲線形状確認用に最大128点の決定的な間引き系列も返す |
| D7 | absolute windowに加え、人間が指定しやすいpercentage windowを扱う。percentageはRun×step軸の到達stepを基準にする |
| D8 | 完全にcurrentなSQLite cacheだけをread-only利用し、それ以外はMetricsマスタへfallbackする。cacheの再構築・削除・更新は行わない |
| D9 | 再利用する抽出条件は外部JSONの「Run解析プロファイル」とし、解釈・閾値・採否判断を含めない |
| D10 | 実装時に `AGENTS.md`、`CONTEXT.md`、Run分析ユーザーガイドを同時更新する。新規ADRは作らない |

## 1. CLI契約

### 1.1 Entry point

標準実行形式を次とする。

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py RUN [RUN ...] [options]
```

publicなsubcommandは設けない。Runの軽量inspection、config抽出、metric抽出、比較をoptionの組み合わせで表す。

| 引数 / option | 回数 | 意味 |
|---|---:|---|
| `RUN` | 1以上 | Run名、または既存の相対・絶対directory path |
| `--metric TAG` | 0以上 | 抽出するscalar tag。完全一致。指定順を保持する |
| `--config-key KEY_OR_GLOB` | 0以上 | 実効configのflat keyまたはcase-sensitive glob |
| `--diff-config` | 0または1 | 複数Run間で異なる実効config keyを返す |
| `--window RANGE` | 0以上 | metric集約window。複数指定時は各windowを独立集約する |
| `--profile PATH` | 0または1 | Run解析プロファイルJSON |
| `--format json\|md` | 0または1 | 出力形式。既定は `json` |
| `--output PATH` | 0または1 | stdoutの代わりに保存する出力先 |

同じ `--metric`、`--config-key`、`--window` の重複は、最初の出現位置を残して除去する。

### 1.2 Run解決

各 `RUN` は次の順序で解決する。

1. current working directory基準または絶対pathとして既存directoryへ解決できる場合は、そのdirectoryをRun directoryとする。
2. directoryへ解決できなければ、repo rootの `apps/runner/workspaces/*/runs/<RUN>` に完全一致するdirect childを探索する。
3. 候補が0件なら、入力値と探索rootを含むエラーにする。
4. 候補が複数なら、全候補の絶対pathを含む曖昧性エラーにする。任意のworkspaceを暗黙選択しない。

`workspace/run_name` のような独自shorthandは追加しない。`apps/runner/runs_*` 等のlegacy/archive配置は再帰探索せず、明示directory pathでだけ利用可能とする。

pathとして解決したRunがworkspace配下にある場合はworkspace名を結果へ含める。workspace外の明示pathでは `workspace` を `null` とする。

### 1.3 出力先と終了値

- `--output` 未指定時は結果だけをstdoutへ出す。警告と診断はstderrへ出し、JSONを汚染しない。
- `--output` 指定時は親directoryが既に存在することを要求する。対象fileは上書き可能とする。
- file出力は対象と同じdirectoryに一時fileを作成し、flush・close後にatomic replaceする。途中失敗で既存fileを壊さない。
- `argparse` の構文エラー、profile契約違反、Run未発見・曖昧性は終了値 `2` とする。
- source read、SQLite query、JSONL parse、出力書込み等の実行時失敗は終了値 `1` とする。
- 一部Runだけでtag/config keyが欠損する場合はresult内を `missing` として終了値 `0` とする。
- 明示したmetricまたはconfig selectorのいずれかが全Runで1件も成立しない場合は、resultを出力したうえで終了値 `1` とする。

## 2. 軽量inspectionと実効config

### 2.1 option未指定時

`--metric`、`--config-key`、`--diff-config` のいずれも指定されていない場合、Metricsマスタのline走査と `config_data.txt` の値dumpを行わない。各Runについて次を返す。

- 入力文字列、Run名、workspace名、Run directoryの絶対path。
- `config/config_data.txt` の絶対path、存在有無、size、mtime、存在する場合のSHA-256。
- `metrics.jsonl` / `metrics.jsonl.gz` の存在状態と、raw優先契約で選ばれたMetricsマスタのpath・kind・size・mtime。
- `metrics_cache.db` のpath、存在有無、size、mtime、検証できた場合のcache状態と `source_meta`。
- Run directory直下の `*.log`、`stdout.log`、`stderr.log`、`*.anet` のpath・size・mtime。directoryを再帰走査しない。

Metricsマスタが無いRunもconfig/log診断対象としてinspection可能とする。metricを要求した場合だけ、当該Runのmetric結果を `source_missing` とする。

### 2.2 実効config

実効設定の正本は `config/config_data.txt` とする。各非空行を最初の `=` で分割し、左辺をtrimしたflat key、右辺をtrimした文字列値として読む。値を数値や真偽値へ暗黙変換しない。

`--config-key` は次の規則とする。

- glob meta characterを含まないselectorも完全一致として同じmatching処理へ通す。
- Python `fnmatch.fnmatchcase` 相当のcase-sensitive matchingを用いる。
- 結果はselector順、その中ではconfig file出現順とする。同じkeyは1回だけ返す。
- Runごとの不成立はselector単位で `missing` とする。

`--diff-config` は全Runのkey unionについて、文字列値または存在有無がRun間で異なるkeyだけを返す。欠損は値 `null` とは別の `present: false` で表現する。`--config-key` 併用時は、そのselector群に一致するkeyだけをdiff対象とする。Runが1件の場合は空のdiffを正常結果として返す。

## 3. Run解析プロファイル

再利用可能な抽出条件を「Run解析プロファイル」と呼ぶ。JSON schema v1を次で固定する。

```json
{
  "version": 1,
  "name": "dropmerge-iqn-k-search",
  "metrics": ["42_env/11_episode_score_mean_ema"],
  "config_keys": ["agent.*"],
  "windows": ["0:5M", "80%:100%"]
}
```

契約は次のとおり。

- `version`、`name`、`metrics`、`config_keys`、`windows` の全fieldを必須とし、未知fieldを拒否する。
- `version` はinteger `1` だけを受理する。boolをintegerとして受理しない。
- `name` は空白だけでないstringとする。
- 残り3fieldはstring arrayとし、空stringを拒否する。
- `metrics` と `config_keys` の少なくとも一方を非空にする。
- 空の `windows` はmetricの全観測範囲を意味する。
- profileは1実行につき1fileとし、include、継承、built-in profile名は設けない。
- CLIの `--metric` と `--config-key` はprofileのarray末尾へ追加し、最初の出現順で重複除去する。
- CLIで `--window` を1件以上指定した場合、profileの `windows` を全置換する。未指定時だけprofile windowsを使う。

profileには解釈文、閾値、採否判断、Run path、出力形式、出力先、series点数を持たせない。AI支援でprofile JSONを生成し、人間とAIの両方が同じ抽出条件を再利用する用途に限定する。

## 4. Metrics sourceとcache選択

### 4.1 Metricsマスタ

Metricsマスタの選択は既存 `viewers/metrics-tools/metrics_source.py` の `resolve_run_metrics()` と `open_metrics_binary()` を再利用する。

- `metrics.jsonl` があればrawを選ぶ。
- rawがなく `metrics.jsonl.gz` があればgzipを選ぶ。
- 両方なければMetricsマスタなしとする。

Runごとに必要な全tagをset化し、Metricsマスタ fallback時は1 passだけstreaming走査する。tagごと、windowごとにfileを開き直さない。

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

検査はMetrics Viewerの現行fingerprint範囲・計算規則に合わせ、cache contractを独自に再定義しない。条件を満たさないcacheは `absent`、`invalid`、`partial`、`stale`、`error` のいずれかとしてinspection結果へ理由付きで記録し、Metricsマスタへfallbackする。

toolはcacheの作成、migration、checkpoint、修復、削除、更新を一切行わない。cache queryは1 Runにつき1 read transactionとし、`tags` とL0の `scalars` から選択tagの全点を読む。統計をLODや `tag_stats` から復元しない。

### 4.3 実行中Runのsnapshot

metric query開始時にMetricsマスタのsizeとmtimeを取得する。

- rawは開始時sizeを読み取り上限とし、それ以後の追記を同じ結果へ混ぜない。上限内の未終端末尾行は取り込まず、resultを暫定とする。
- gzipはimmutable sourceとしてEOFまで読む。未終端行またはgzip破損はsource errorとする。
- query終了後にsizeとmtimeを再取得し、開始時から変化した場合は `source_changed_during_read: true` とwarningを返す。自動retryはしない。
- cache判定は開始時snapshotと比較する。cacheが開始時sourceへ完全追随していなければmaster fallbackとする。

## 5. metric抽出契約

### 5.1 scalar入力

対象はMetrics masterの `type == "scalar"` かつ明示指定tagに一致するrecordである。stepは非負integer、valueはboolを除く有限数値を要求する。

- 非数値、非finite、float32範囲外のvalueはMetrics Viewerと同様に除外し、除外数をtag診断へ記録する。
- 選択対象外のrecord typeとtagはparse後に無視する。
- JSON不正、必須field欠落、不正step等のsource構造違反はRun単位のsource errorとし、黙って読み飛ばさない。ただしrawのsnapshot末尾にある未終端行は§4.3の暫定契約を適用する。
- 同一tag内でstepが逆行した場合、そのtagを `quarantined` とし、逆行前の有効prefixだけを公開する。他tagの走査は継続する。

cache経路でも `tags.status == error` のtagは `quarantined` として、cacheにcommit済みの有効prefixを返す。cacheとmaster fallbackでstatus、統計、系列の意味を揃える。

### 5.2 step軸

各tagのstep軸は `config/config_data.txt` にある実効 `metrics.scalar.[<tag>]` 定義から解決する。

- `$train_step`、`$learn_step`、`$episode_step`、`$exp_step`、`$update_step`、`$sim_step` の明示指定を優先する。
- 明示指定が無い場合、`@train` は `train_step`、`@learn` と `@episode_end` は `exp_step` とする。
- 解決できない場合は `unknown` とし、absolute windowと全範囲集約は許可する。
- 同じtagがRun間で異なるstep軸へ解決された場合、resultとstderrへwarningを出す。

### 5.3 window

`--window` 未指定、かつprofile windowsが空の場合、各tagの全観測範囲を1 windowとして集約し、labelを `all` とする。

absolute windowは `START:END` とし、両端inclusive、`START <= END` とする。各endpointは非負integerと、case-insensitiveなdecimal suffix `K=1,000`、`M=1,000,000`、`G=1,000,000,000` を受理する。小数付きabsolute値、負数、open endpoint、overflowを拒否する。

percentage windowは `START%:END%` とする。

- endpointは0以上100以下の有限decimalとし、`START <= END` とする。
- 同一window内でabsolute endpointとpercentage endpointを混在させない。
- 各Runについて、選択tagを解決済みstep軸ごとにgroup化し、そのgroup内の最大観測stepを当該軸の100%とする。0%はstep 0とする。
- lower boundは `ceil(max_step * START / 100)`、upper boundは `floor(max_step * END / 100)` で解決する。
- step軸が `unknown` のtagへpercentage windowを適用する場合は、tagとRunを示してfail-fastする。
- Runごとに到達stepが異なるため、resultには元のpercentage表現と解決後のabsolute boundsを両方残す。

### 5.4 統計と間引き系列

各Run×tag×windowについて、window内の全有効点から次を計算する。

| field | 意味 |
|---|---|
| `count` | 有効点数 |
| `mean` | arithmetic mean |
| `population_std` | 分母 `count` の標準偏差 |
| `min` / `max` | 最小値 / 最大値 |
| `first` / `first_step` | 最初の値 / step |
| `last` / `last_step` | 最後の値 / step |
| `min_step` / `max_step` | window内で観測したstep範囲 |

集約はfloat64のonline accumulatorを用い、全点listを統計専用に複製しない。点が0件の場合は `status: empty`、`count: 0` とし、他の統計fieldはJSONで `null` とする。NaNやInfinityをJSONへ出力しない。

曲線形状確認用の `series` は最大128点とする。

1. 点数が128以下なら全点を元の順序で返す。
2. 128を超える場合は序数を連続bucketへ等分する。
3. 各bucketから最小値点、最大値点、末尾点を元の序数順に採用する。同じ点は1回へ畳む。
4. 全系列の先頭点と末尾点を必ず保持する。
5. bucket数は、重複が一切無い場合でも合計が128点以内になるよう決定する。

同値候補では序数が小さい点を採用する。同一入力に対してcache/master、実行順、Python processによらず同じ系列を返す。

## 6. result model

JSON rootは次の構造を持つ。field名はschema v1のpublic contractとする。

```json
{
  "schema_version": 1,
  "generated_at": "2026-08-15T12:34:56+09:00",
  "profile": {"path": null, "name": null},
  "windows": [
    {"label": "80%:100%", "kind": "percentage"}
  ],
  "runs": [
    {
      "input": "run_name",
      "run_name": "run_name",
      "workspace": "dm-iqn",
      "run_dir": "C:\\dev\\anet-lab\\apps\\runner\\workspaces\\dm-iqn\\runs\\run_name",
      "artifacts": {},
      "config": {"selectors": [], "values": []},
      "metrics_source": {
        "selected": "cache",
        "master_path": "...\\metrics.jsonl",
        "cache_path": "...\\metrics_cache.db",
        "cache_status": "current",
        "provisional": false,
        "source_changed_during_read": false
      },
      "metrics": [
        {
          "tag": "42_env/11_episode_score_mean_ema",
          "step_axis": "exp_step",
          "status": "ok",
          "windows": []
        }
      ],
      "warnings": []
    }
  ],
  "config_diff": [],
  "warnings": []
}
```

配列順はRun入力順、metric選択順、window指定順を保持する。pathは絶対pathで出力する。statusは少なくとも `ok`、`missing`、`empty`、`quarantined`、`source_missing`、`source_error` を区別する。

Markdownは同じresult modelから次の順序で生成する。

1. 実行条件とprofile。
2. Runごとのartifact・source・cache状態。
3. config selector結果とconfig diff。
4. Run×tag×windowの統計table。
5. tagごとの間引きseriesをcompactな `step:value` 列として記載。
6. warning一覧。

Markdownだけに存在する解析判断や自動コメントを追加しない。

## 7. 実装範囲

### 7.1 コード

| ファイル | 変更内容 |
|---|---|
| `viewers/metrics-tools/inspect_run.py` | CLI、Run resolver、profile/config reader、artifact inspection、cache validator/query、master streaming reader、window/stat/series計算、JSON/Markdown rendererを単一機能グループとして実装 |
| `viewers/metrics-tools/inspect_run_test.py` | `unittest` と一時fixtureによる主要CLI契約テスト |

既存 `metrics_source.py` をsource選択・openに再利用する。単一CLIのために新規package、component file、共通frameworkを作らない。第三者依存を追加せずPython標準libraryだけで実装する。

### 7.2 実装時の文書更新

| ファイル | 変更内容 |
|---|---|
| `AGENTS.md` | Run名だけの提示は原則Run分析依頼と扱うこと、標準CLI実行例、現行workspace探索範囲、巨大Run treeを再帰 `rg` しないこと、config/master/cacheの正本関係を追記 |
| `CONTEXT.md` | 用語「Run解析プロファイル」を追加。抽出対象とwindowを再利用するread-onlyな選択契約であり、解釈・判定を持たないことを記載 |
| `docs/design/030_user_guide_analysis.jp.md` | `inspect_run.py` のRun指定、複数tag/Run、absolute/percentage window、profile、cache fallback、JSON/Markdown利用例を追記 |

既存のworkspaceおよびMetrics cacheの設計決定内に収まるため、新規ADRは作成しない。

## 8. テストと受け入れ基準

`inspect_run_test.py` は実Runに依存せず、一時repo/workspace/Run fixture、config、raw/gzip、SQLite schema v1 cacheを生成する。

### 8.1 Run・config・profile

- 既存相対path、絶対path、workspace内の一意なRun名を解決できる。
- legacy directoryは明示pathで解決できるが、Run名探索では見つからない。
- 同名Runが複数workspaceにある場合、候補pathを含めて終了値2となる。
- option未指定時にMetricsマスタをopenせず、artifact metadataとconfig SHA-256だけを返す。
- config完全一致、glob、欠損、全key diff、selector限定diff、欠損対存在を検証する。
- profileの正常系、未知field、未知version、型不正、空selector、CLI追加、window全置換を検証する。

### 8.2 source・cache

- rawとgzipがあるとrawを選び、gzipだけならgzipを選ぶ。
- 完全currentなcacheをread-only利用し、masterを走査しない。
- cache不存在、partial、stale、error、application ID/schema不正ではmasterへfallbackする。
- cache利用前後でRun artifactのsize、mtime、内容が変わらない。
- raw追記中snapshotは開始時sizeまでを読み、未終端末尾を除外し、source変化と暫定状態を返す。
- 同一fixtureについてcache経路とraw/gzip経路のstatus、統計、seriesが一致する。

### 8.3 metric・window・出力

- 複数Run×複数tagを1実行で抽出し、masterをRunごとに1回だけopenする。
- absolute windowの `K/M/G`、両端inclusive、境界1点を検証する。
- percentage windowをRun×step軸の最大到達stepから解決し、異なるRun長で異なるabsolute boundsになることを検証する。
- absoluteとpercentageの複数window併用、同一window内の単位混在拒否、unknown軸でのpercentage拒否を検証する。
- 手計算可能な点列で `count`、mean、population std、min/max、first/last、step範囲を検証する。
- 128点以下では全点を保ち、129点以上では128点以内、先頭・末尾保持、決定性を検証する。
- 非数値・非finite値の除外、step逆行tagのquarantine、有効prefix公開を検証する。
- 一部Runのtag欠損は終了値0、全Run欠損は結果を出力して終了値1となる。
- JSONがstrict parse可能でNaN/Infinityを含まず、Markdownが同じresult modelの値を含む。
- `--output` が既存fileを置換し、書込み失敗時には既存内容を保持する。

標準検証コマンドは次とする。実装時に `AGENTS.md` のPython補助ツール標準テスト一覧へ追記する。

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run_test.py
git diff --check
```

## 9. スコープ外

- 本PRD作成時点でのtool実装、文書更新、実験Run。
- Metrics Viewer UI、HTTP API、Java実装、SQLite schemaの変更。
- cacheの生成、更新、修復、migration、削除。
- built-in profile catalog、profile生成command、profile継承。
- metricの意味解釈、異常閾値、採否判断、自然言語分析、Run ranking。
- raw全点の無制限出力、plot画像生成、interactive viewer。
- 実Run依存test、網羅的fuzz、性能benchmark。ただしstreaming・1 pass・128点上限の契約はfocused testで確認する。

## 10. Further Notes

- `inspect_run.py` はRun分析の材料を安定して抽出するtoolであり、分析者が確認すべき「Run成立性、主目的score、変更機構、Env挙動、実時間性能」を一つのscoreへ統合しない。
- percentage windowは異なる長さのRunを相対進捗で眺める人間向け補助である。ハイパラ比較の正式判断では、可能な限り同じstep軸のmatched absolute windowも併記する。
- `metrics_cache.db` は高速経路であって正本ではない。cache contractが将来変わった場合、未知schemaを推測して読むのではなくmaster fallbackする。
- artifact inventoryはRun発見と次の調査入口を提供するためのものであり、configやlogを無条件にstdoutへdumpしない。
