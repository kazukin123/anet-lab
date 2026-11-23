# PRD 054 `inspect_run.py` 実装メモ

- 起票日: 2026-08-15
- 対象PRD: `docs/memo/054_inspect_run_10prd.md`
- 状態: 実装完了（`inspect_run_test.py` 49件緑、実Runスモーク確認済み）

## 概要

AI向けRun解析CLI `viewers/metrics-tools/inspect_run.py` を実装する。Run artifactを変更せず、Run解決・artifact inspection・実効config抽出・Metricsマスタ/cacheからのscalar抽出・window集約・JSON/Markdown出力を1本のCLIで提供する。

第三者依存を追加せずPython標準libraryだけで実装し、source選択は既存 `metrics_source.py` の `resolve_run_metrics()` / `open_metrics_binary()` を再利用する。

## PRDからの逸脱

`--config-key` のmatchingで、PRD §2.2の「Python `fnmatch.fnmatchcase` 相当」から意図的に逸脱する。

- glob meta characterは `*` と `?` だけとし、`[` と `]` はリテラル文字として照合する。
- 理由: 実効configのkeyは `train.eval.[eval1].run_mode`、`metrics.scalar.[21_eval/01_target_reward]`、`action_uqe_win_rate.[0]` のように `[tag]` 記法が偏在する。素の `fnmatch` では `[eval1]` がcharacter classと解釈され、リテラル指定が1件もmatchせず黙って `missing` になる。
- 実効configのkeyに `*` や `?` が出現する実例は0件のため、この2文字だけをglob metaとして残す副作用はない。
- case-sensitiveであることはPRDどおり維持する。

## PRD result modelへの追加

PRD §6のschema v1に対して、実装で次を追加した。既存field名は変更していない。

| field | 追加理由 |
|---|---|
| `runs[].metrics_source.cache_reason` | `cache_status` が `current` 以外のとき、どの条件で落ちたかを機械可読に残す。`absent` では `null` |
| `runs[].artifacts.cache.status` / `.reason` / `.source_meta` | 軽量inspectionでcache状態と `source_meta` を返すというPRD §2.1の要求を満たすため |
| `runs[].metrics[].excluded` | PRD §5.1の「除外数をtag診断へ記録する」を格納する場所 |
| `runs[].metrics[].windows[].start` / `.end` | percentage windowの解決後absolute boundsをPRD §5.3のとおり残す。absoluteでは指定値、`all` では `null` |

`runs[].metrics[].status` が `quarantined` の場合は有効prefixを返せているため、PRD §1.3の「1件も成立しない」判定では成立扱いとする。終了値1になるのは全Runで `missing` / `source_missing` / `source_error` だけだった場合である。

## 主な変更

### `viewers/metrics-tools/inspect_run.py`（新規）

public CLIは1本、実装も同module内の機能グループとしてまとめる（PRD D1）。module内のsection順は次とする。

| section | 責務 |
|---|---|
| 定数 | `FINGERPRINT_BYTES = 65536`、`SERIES_MAX_POINTS = 128`、`SERIES_BUCKETS = 42`、`CACHE_APPLICATION_ID = 0x414E4554`、`CACHE_SCHEMA_VERSION = 1` |
| Run resolver | 既存directory優先、次に `apps/runner/workspaces/*/runs/<RUN>` の完全一致。0件/複数件はエラー |
| config reader | `config/config_data.txt` のflat key読み取りとselector matching |
| profile reader | Run解析プロファイルJSON schema v1の検証 |
| artifact inspector | config/master/cache/logのpath・存在・size・mtime、configのSHA-256 |
| cache validator | read-only URIでのeligibility判定 |
| cache query | `tags` とL0 `scalars` からの選択tag読み出し |
| master reader | snapshot上限つき1 pass streaming走査 |
| step軸 resolver | 解決済み `metrics.scalar.[<tag>]` からのstep軸解決 |
| window | absolute / percentage windowのparseと解決 |
| stats / series | online accumulatorによる統計と決定的間引き |
| result model | schema v1のdataclass群 |
| renderer | JSON / Markdown |
| CLI main | argparse、終了値、`--output` のatomic置換 |

### `viewers/metrics-tools/inspect_run_test.py`（新規）

`unittest` と一時fixtureによるCLI契約テスト。実Runに依存しない。

### 文書更新（PRD §7.2）

| ファイル | 変更 |
|---|---|
| `AGENTS.md` | `## 検証` のテストコマンド一覧へ追記。`## AI エージェントのRun結果分析ルール` へCLI導線を追記 |
| `CONTEXT.md` | `### Metrics基盤` へ用語「Run解析プロファイル」を追加 |
| `docs/design/030_user_guide_analysis.jp.md` | `inspect_run.py` の利用手順を追記 |

新規ADRは作成しない（cacheの破棄可能性はADR 0015、workspaceはADR 0021の枠内）。

## 実装契約の固定点

repo evidenceから確定し、実装中に再検討しない点を列挙する。

### Run解決

- repo rootは `Path(__file__).resolve().parents[2]`（`viewers/metrics-tools/` からの相対）で決める。
- workspace探索rootは `<repo root>/apps/runner/workspaces`。`*/runs/<RUN>` のdirect childだけを見る。
- `apps/runner/runs_*` のlegacy配置は再帰探索せず、明示directory pathでだけ解決できる。
- pathで解決したRunがworkspace配下にある場合だけ `workspace` を埋め、workspace外は `null`。

### 実効config

- `config/config_data.txt` は `utf-8-sig` で読む（BOM耐性）。
- 各非空行を最初の `=` で1回だけ分割し、両辺をtrimする。値は型変換せず文字列のまま保持する。
- 生成元 `ConfigData::ToPropertiesString()` は ` = ` 区切りでcomment・空行を出さないが、parserは空行と `=` 無し行を許容してskipする。
- legacy Runの単一 `config.txt` へはfallbackしない。`config/config_data.txt` が無ければ `exists: false`、`--config-key` は `missing`。

### step軸

- 対象は解決済みキー `metrics.scalar.[<tag>]` **のみ**。`metrics.scalar.baseline.`、`.full.`、`.min.`、overlay生キー `M.[` は未適用の定義なので読まない。
- 値はスペース区切りのtoken列で、token順序は自由。
- 明示軸token: `$train_step`、`$learn_step`、`$episode_step`、`$exp_step`、`$update_step`、`$sim_step`。`step:` / `step_axis:` 属性も同じ軸名を受理する。
- 明示が無い場合は event で決める。`@train`（および event 省略時の既定）は `train_step`、`@learn` と `@episode_end` は `exp_step`。
- 定義が見つからない、または軸を決められない場合は `unknown`。
- `step_axis` の出力値は `$` を除いたtoken名（`exp_step` 等）または `unknown`。

### Metricsマスタとcache

- source選択は `resolve_run_metrics()` に委譲する（raw優先、無ければgzip、両方なければMetricsマスタなし）。
- cacheは `file:<path>?mode=ro` のread-only URIで開く。作成・migration・checkpoint・修復・削除・更新は一切行わない。
- `source_meta` の値は全てTEXTなので、数値keyは `int()` 変換し、失敗を `invalid` とする。
- fingerprintはMetrics Viewerの計算規則をそのまま再現する。
  - head: `sha256(bytes[0 : min(current_size, stored_source_size, 65536)])`
  - tail: `end = clamp(committed_offset, 0, current_size)` として `sha256(bytes[max(0, end - 65536) : end])`
  - gzipも展開せず圧縮fileの生byteをhashする。出力は小文字hex 64桁。
- eligibilityはPRD §4.2の6条件を順に判定し、最初に落ちた理由を記録する。不成立は `absent` / `invalid` / `partial` / `stale` / `error` へ分類してmaster fallbackする。
- cache queryは1 read transactionで `tags` とL0 `scalars` を `ORDER BY ordinal` で読む。統計をLODや `tag_stats` から復元しない。
- `tags.status == 'error'` のtagは `quarantined` とし、cacheにcommit済みの有効prefixを返す。

### metric抽出

- Metricsマスタ fallbackはRunごとに1 passだけ走査する。tagごと・windowごとにfileを開き直さない。
- 全行を `json.loads` する。PRD §5.1が「不正JSONを黙って読み飛ばさない」を要求するため、tagによる事前substring filterは入れない。
  - 578MB級のRunではmaster fallbackが数分かかりうる。実Run 121個中119個が `metrics_cache.db` を持つため、通常はcacheの高速経路に乗る。
- 選択tagの点は `array('q')`（step）と `array('d')`（value）で1組だけ保持し、統計はその配列をonline accumulatorで1走査する。統計専用の複製を作らない。
- 値の除外規則はMetrics Viewerに揃える。null、非数値、bool、非finite、float32範囲外を除外し、除外数をtag診断へ記録する。
- 同一tag内でstepが逆行したらそのtagを `quarantined` とし、逆行前の有効prefixだけを公開する。他tagの走査は継続する。

### window

- `--window` 未指定かつprofile windowsが空なら、各tagの全観測範囲を1 window（label `all`）とする。
- absoluteは `START:END`、両端inclusive、`START <= END`。suffixは case-insensitive な `K`=1,000、`M`=1,000,000、`G`=1,000,000,000。小数付きabsolute、負数、open endpoint、overflowを拒否する。
- percentageは `START%:END%`。endpointは0以上100以下の有限decimal。同一window内でabsoluteとpercentageを混在させない。
- percentageは、選択tagを解決済みstep軸ごとにgroup化し、そのgroup内の最大観測stepを100%とする。lower = `ceil(max_step * START / 100)`、upper = `floor(max_step * END / 100)`。
- resultには元のpercentage表現と解決後のabsolute boundsを両方残す。

### 統計と間引き系列

- window内の全有効点から `count` / `mean` / `population_std`（分母 `count`）/ `min` / `max` / `first` / `first_step` / `last` / `last_step` / `min_step` / `max_step` を計算する。
- 0件は `status: empty`、`count: 0`、他統計fieldは `null`。NaN / Infinity をJSONへ出さない。
- `series` は最大128点。
  1. 点数が128以下なら全点を元の順序で返す。
  2. 128を超える場合、序数を **42個** の連続bucketへ等分する。bucket `i` は序数 `[floor(i*n/42), floor((i+1)*n/42))`。
  3. 各bucketから最小値点、最大値点、末尾点を採用する。同値候補では序数が小さい点を採る。
  4. 全系列の先頭点と末尾点を必ず加える。
  5. 採用序数をsetで畳み、昇順で出力する。
- bucket数42の根拠: 重複が一切無い場合でも `42 * 3 = 126` 点、これに先頭点を足して最大127点となり128点上限に収まる。43 bucketでは `43 * 3 + 1 = 130` 点で上限を超える。末尾点は最終bucketの末尾点と一致するため加算されない。

### 出力と終了値

- `--output` 未指定時は結果だけをstdoutへ出し、警告と診断はstderrへ出す。
- `--output` 指定時は親directoryの存在を要求し、同じdirectoryへ一時fileを作ってから `os.replace` でatomicに置換する。
- JSONは `json.dump(..., allow_nan=False, ensure_ascii=False)` で書く。NaN / Inf 混入は例外として検出する。
- `generated_at` は `datetime.now().astimezone().isoformat(timespec="seconds")`。
- 終了値
  - `2`: argparseの構文エラー、`--window` / profile windowの構文エラー・単位混在・`START > END`、profile契約違反、`--output` 親directory不在、Run未発見・曖昧性。
  - `1`: source read、SQLite query、JSONL parse、出力書込み等の実行時失敗。unknown軸へのpercentage window適用。明示したmetricまたはconfig selectorのいずれかが全Runで1件も成立しない場合（resultは出力する）。
  - `0`: 一部Runだけでtag / config keyが欠損する場合（result内を `missing` とする）。

## テスト

- **Public interface / surface**: `inspect_run.py` のCLI契約。引数、stdoutのJSON schema v1、Markdown、stderr、終了値、`--output` のfile内容。内部関数を直接叩くテストは書かない。
- **優先behavior**: Run名からのJSON抽出end-to-end、raw/gzipのsource選択、cache採否とmaster fallbackの等価性、window解決、統計と間引き系列の正確さと決定性、終了値。

### TDD順序

1つのbehaviorごとにRED -> GREENを完了する。RED中はrefactorしない。

1. tracer bullet: workspace fixtureにRun 1個・`config_data.txt`・raw masterを置き、`RUN --metric TAG` でstrict parse可能なJSONがstdoutへ出て終了値0になる。Run解決 -> config読み -> master走査 -> 統計 -> JSONまでを1テストで貫通させる。
2. Run解決: 相対path、絶対path、workspace内一意名。legacy directoryは明示pathでのみ解決。同名Runが複数workspaceにある場合は候補path付きで終了値2。
3. 軽量inspection: option未指定時にMetricsマスタをopenせず、artifact metadataとconfigのSHA-256だけを返す。
4. config: 完全一致、`*` glob、`[tag]` リテラル、欠損、全key diff、selector限定diff、`present: false`。
5. profile: 正常系、未知field、未知version、bool version拒否、空selector、CLI追加の重複除去、`--window` による全置換。
6. source選択: raw/gzip併存でraw、gzipのみでgzip。
7. cache: 完全currentなcacheのread-only利用とmaster未open、不存在・partial・stale・error・application ID不正・schema不正でのfallback、利用前後のartifact不変、cache経路とmaster経路のstatus/統計/series一致。
8. 実行中snapshot: raw追記中に開始時sizeまでを読み、未終端末尾行を除外し、暫定状態とsource変化を返す。
9. window: `K/M/G`、両端inclusive、境界1点、percentageの軸別解決、Run長違いで異なるabsolute bounds、absoluteとpercentageの併用、単位混在拒否、unknown軸でのpercentage拒否。
10. 統計: 手計算可能な点列での全統計field。0件時の `empty` と `null`。
11. 系列: 128点以下で全点保持、129点以上で128点以内・先頭末尾保持・決定性。
12. 値検査: 非数値・非finite・float32範囲外の除外と除外数記録、step逆行tagのquarantineと有効prefix公開。
13. 出力と終了値: 一部Run欠損で0、全Run欠損で結果出力かつ1、JSONにNaN/Infinityを含まない、Markdownが同じresult modelの値を含む、`--output` の置換と書込み失敗時の既存内容保持。
14. 複数Run×複数tagを1実行で抽出し、masterをRunごとに1回だけopenする。

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

C++側の変更は無いためビルドは不要。

## 前提

- 実装範囲は `viewers/metrics-tools/` への2 file追加と、`AGENTS.md` / `CONTEXT.md` / `docs/design/030_user_guide_analysis.jp.md` の追記だけ。既存Python moduleは変更しない。
- `metrics_source.py` の `resolve_run_metrics()` は Metricsマスタが無いとき `None` を返す。例外にしない契約をそのまま利用する。
- Git の `add` / `commit` / `push` は行わない。
