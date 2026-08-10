# Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)

> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 用語(ワークスペース / workspace config / Run作業セット)はリポジトリルート `CONTEXT.md` が正本。
> 対応 ADR: [0021-run-classification-by-workspace-folder.md](../adr/0021-run-classification-by-workspace-folder.md)

## Context(背景・目的)

`apps/runner/runs/` に性質の異なる実験系列(DropMerge 長期 Run、IQN×LunarLander、IQN×DropMerge、
optuna 探索)の Run が混流し、探しづらい。現状は手動の退避フォルダ(`runs_optuna`、`runs_DropMerge4`、
`apx-longrun/` 等)への移動で凌いでおり、Viewer で見たくなるたびにフォルダを戻している。

本リポジトリの Run 管理は「**フォルダ＝Run の完全な実体**」が意図した設計である
(削除=rmdir、退避/復帰=別ドライブへの移動、SQLite キャッシュも Run フォルダ同梱でパス非依存)。
この思想を 1 階層上へ拡張し、**workspace＝Run の入力(config)と成果物(runs、optuna)を束ねる
自己完結フォルダ**を導入する。タグ・外部 DB などフォルダ外に分類の主データを置く案は採らない
(理由は ADR 0021)。

副次目的として、env 切替(現状 `_main.txt` の `$include` コメントアウト手作業)の解消、
optuna 出力の Viewer 可視化、`23_optuna_dashboard.bat` 等のハードコード整理を含む。

## 0. 決定一覧(グリル確定値)

| # | 決定 |
|---|---|
| D1 | 構造は `apps/runner/workspaces/<ws>/{config/_main.txt, runs/, optuna/}` の箱型 |
| D2 | Runner は常時 workspace モード。未指定は `_default`、無ければ自動生成 |
| D3 | env include は `_main.txt` から全面移管。専用バリデーション無し(既存 fail 経路+メッセージヒント) |
| D4 | 選択優先順位: CLI `--workspace <workspace_path>` > 起動ダイアログ(スキップ設定付き) > 履歴先頭(直近使用) > `_default`。指定はパス(相対=`workspaces/` 基準、絶対可。Eclipse 同形式) |
| D5 | workspace 履歴とスキップ設定は AP データフォルダの**責務別ファイル**へ: `history.txt`(履歴)+`prefs.txt`(選好)。**ファイル削除=その責務のリセット**。読み=Properties / 書き=ConfigData 新設メンバ |
| D6 | `--config` 明示時は workspace 解決を完全無効化(完全自己記述モード)。`--workspace` 併用はエラー |
| D7 | Viewer はサーバ側 current 方式(1 JVM=1 workspace、切替 API+セレクタ)。任意パスのアタッチは後続 PRD |
| D8 | optuna は study=workspace、trial(seed run)=Run。ハーネス `--workspace` は `_default` フォールバック |
| D9 | ハーネスは runner へ `--workspace` を素通ししない。生成 config に workspace include を書き込む |
| D10 | optuna.db / harness.log / artifacts は `workspaces/<ws>/optuna/` へ(workspace 単位、self-contained) |
| D11 | `22_metrics_viewer_java_optuna.bat` は削除。`23_optuna_dashboard.bat` は workspace 引数化 |
| D12 | 既存 runs 系フォルダの移行は手動フォルダ移動のみ。コード migration 無し |
| D13 | AP データはポータブルモード対応: `apps/runner/appdata/` フォルダの**存在**で切替(VS Code `data/` 型マーカー)。無ければ OS 標準 user-data dir。`WriteLastRunName` も `GetAppDataDir()` 利用へ書き換え |
| D14 | Viewer の切替排他は **workspace epoch**+**snapshot 所有**: runId キーの in-memory キャッシュ群は snapshot が所有し、切替=新インスタンスへの atomic swap(in-flight query は自分の旧インスタンスを読み書き=読み取りも構造的に混ざらない)。切替は **ingest ゲートで直列化**し、旧 snapshot の close 必須リソース(gzip stream 等)は利用者ゼロ時点で明示終了 |
| D15 | workspace config テンプレートの正本は追跡対象 `apps/runner/config/_workspace_template.txt`(env ブロックの移動先) |
| D16 | workspace モードでは **AutoMerge+CLI override 完了後の実効 `app.runs_dir` が導出値と一致すること**を検証、不一致は起動エラー(直接キーだけでなく `app.$` 経由の間接上書きも網羅)。自己完結不変条件を維持 |
| D17 | ダイアログ Cancel=アプリ終了。`--config`/`--workspace`/`--select-workspace` の同時指定は全組み合わせエラー |
| D18 | workspace path は **`#`・`//` を含まず、UNC(`\\`・`//` の両表記。正規化後の root 判定)も非対応**の契約。`#`/`//` は受理時点と `SaveProperties` の両方で fail-fast(壊れた値を保存しない)。**UNC の拒否はパス入力境界のみ**(SaveProperties は検査しない) |
| D19 | optuna ハーネスは副作用の**前に**サブコマンド別の事前検証で fail-fast(run 系=workspace root+config、cleanup-running=storage のみ、summarize-study=storage/artifact のみ。箱の自動生成はしない)。preflight は **4 段階分離**(パス解決[mkdir 無し]→source 検証→全成功→target 作成)。**run 系の runs dir は `<ws>/runs` 固定で `--runs-dir` 引数は削除**、`--storage`/`--optuna-artifact-dir` は `<ws>/optuna` 配下限定(跨ぎは summarize-study のみ)。placeholder 展開機能は廃止 |

## 1. ディレクトリ構造と不変条件

```text
apps/runner/workspaces/
  _default/                     # D2: 既定 workspace(無ければ Runner が自動生成)
    config/_main.txt            # workspace config(エントリ名は共通 config と同じ規約)
    runs/run_xxx/               # Run 出力
  dm_long/                      # 通常 workspace の例
    config/_main.txt            #   $include <DropMerge.txt> + workspace 固有 override
    runs/run_xxx/
  dm_opt/                       # optuna workspace の例(構造上は通常 workspace と同格)
    config/_main.txt            #   study の基底 config(env 選択含む)が箱に pin される
    runs/                       #   seed run(metrics.jsonl 持ち) + trial 代表フォルダ(持たない)
      <study>_<trial>/trial/    #   代表フォルダ。Viewer は hasMetricsFile フィルタで自然除外
      <study>_<trial>_s<seed>/
    optuna/
      optuna.db                 # workspace 単位の Optuna storage
      harness.log(.1/.2)
      artifacts/                # Dashboard 用 FileSystemArtifactStore
```

不変条件:

- **workspace は自己完結**。`workspaces/<ws>/` を丸ごと移動・削除・別ドライブ退避しても、
  中の Run・キャッシュ・optuna 履歴・基底 config が一体で移動する。フォルダ外に workspace の
  メタデータを持たない(AP データフォルダの workspace 履歴は「最近開いた場所」の記録であり分類情報ではない)。
- **`runs/` の直下は Run(と optuna trial 代表フォルダ)のみ**。config 等の入力は `config/` に分離。
- workspace の指定は**パス**で行う(Eclipse の workspace 指定と同形式)。相対パスは
  `workspaces/`(runner root 基準)を基点に解決し、絶対パスなら任意の場所(別ドライブ等)を
  workspace にできる。既定の置き場は `workspaces/` 直下で、その場合はフォルダ名 1 語が
  そのまま指定になる(`--workspace dm_long`)。
- **workspace path は `#` および `//` を含まず、UNC パス(`\\server\share` 等の `\\` 始まり)も
  非対応**(D18)。`#`/`//` は Properties 形式がコメントとして切り捨てるため(履歴の round-trip が
  壊れる)。UNC は入力自体に `//` を含まないが、optuna の include 生成(`config_include_line` の
  as_posix 変換)が `//server/...` を生成してコメント構文と衝突し、NAS 上の SQLite
  (optuna.db / metrics_cache.db)は lock 挙動の面でも元々非推奨のため、受理時点で拒否する
  (退避ドライブはドライブレターで足りる。将来必要になればバックスラッシュ維持の include 生成で解禁)。
  **UNC 判定は文字列 prefix でなく、正規化後のパスが UNC root を持つかで行い、
  `\\server\share`・`//server/share` の両表記を拒否する**(Windows では forward-slash も UNC。
  `//` 始まりは禁止文字契約でも落ちるが、判定の正本は正規化後 root とする)。
  受理時点(`--workspace` / 参照 / 新規名 / 履歴読み込み)で違反は即エラー。
  **D18 の適用境界は「workspace path の入力」**: Runner の上記入力、optuna ハーネスの
  `--workspace`、dashboard bat の workspace 引数(§4.3 で起動前に検証)が対象。
  `metricsviewer.workspaces-dir` は workspace path でなく親ディレクトリの設定だが、
  UNC(`\\` 始まり)は導出される `metrics_cache.db` が UNC 上に乗るため
  **Viewer 起動時エラー**とする(§3.1)。**非対応の入力はすべて入力時エラー**で統一し、
  「無保証で受理」する境界を残さない。
- `.gitignore` に `/apps/runner/workspaces` を追加(runs* と同様、成果物は非管理)。
  ポータブルモード用の `/apps/runner/appdata`(§2.2)も追加。

## 2. Runner / config 機構(PH1)

### 2.1 workspace 選択フロー

優先順位(D4):

1. CLI `--workspace <workspace_path>`(`RunnerApp.cpp` の `wxCmdLineEntryDesc desc[]` に OPTION 追加)
2. 起動ダイアログ: CLI 未指定かつ「スキップ設定 OFF」のとき表示。構成は次の 5 要素:
   1. **履歴コンボボックス**: workspace 履歴(§2.2、MRU 順)を列挙。初期選択は履歴先頭
      (=最後に使った workspace)。選択して OK で起動。解決できない項目(退避ドライブ未接続等)は
      グレー表示し、選択されたらエラー表示で再選択を促す(履歴からの自動削除はしない)
   2. **workspaces/ 直下スキャン一覧**: コンボとは別の画面項目。`workspaces/` 直下の
      ディレクトリを列挙し、**ワンクリック(キーボードなら選択+Enter)で即起動**する
      ランチャー的挙動(OK 不要)
   3. 任意パスを選ぶ「参照」ボタン(ディレクトリ選択)
   4. 新規 workspace 名の入力欄(入力されたら `config/_main.txt` テンプレートを生成)
   5. 「今後表示しない」チェック(スキップ設定 ON)
3. 履歴先頭(=最後に使った workspace。前回起動確定時に自動記録)
4. `_default`(`workspaces/_default/` が無ければテンプレートで自動生成)

**ダイアログの状態機械**(D17):

- 初回起動(履歴無し・`workspaces/` 無し): ダイアログを表示。コンボ空・スキャン一覧空、
  新規名入力欄に `_default` をプリセット。OK でテンプレート生成して起動
- OK ボタンは「**履歴コンボに有効な選択がある** or **新規名入力欄が非空**」のときだけ有効
  (空選択での OK は構造的に不可能。スキャン一覧・参照は即確定起動で OK の対象外)
- **Cancel はアプリ終了**(Eclipse と同じ。workspace 無しでは何もできない。
  `_default` へ進む案は「選ばなかったのに箱ができる」副作用があるため不採用)
- **入力元の相互排他**(last-touched wins): OK の解決先になり得るのは履歴コンボと
  新規名入力欄の 2 つのみで、新規名入力欄へのタイプ開始でコンボ選択を解除、
  コンボ操作で入力欄をクリアする(OK 時点でアクティブな入力元が常に一意)。
  参照ボタンとスキャン一覧は**選択即確定起動**(OK を経由しない)なので競合しない

**CLI フラグの競合**(D17、全て起動エラー=fail-fast、黙って無視しない):

| 組み合わせ | 挙動 |
|---|---|
| `--config` + `--workspace` | エラー(D6 既定) |
| `--config` + `--select-workspace` | エラー(同上に統一) |
| `--workspace` + `--select-workspace` | エラー(指示が矛盾) |
| `--select-workspace` 単独 | スキップ設定を無視してダイアログ強制表示 |

**パス解決規則**: 指定値が相対パスなら `<runner root>/workspaces/` を基点に解決、
絶対パスならそのまま使う。解決先ディレクトリが workspace root になる
(`config/_main.txt` を後読みし、`runs/` へ出力する)。存在しないパスの扱い:
`workspaces/` 相対 1 語(名前形式)なら新規作成としてテンプレート生成、
それ以外(絶対パス等)は誤指定の可能性が高いため fail-fast。

**履歴の記録と運用**: 起動確定時(workspace 解決成功後)に、指定文字列をそのまま
(相対は相対のまま)履歴の先頭へ記録する。MRU 順・重複は指定文字列の完全一致で 1 件に畳む・
上限 10 件(超過は末尾切り捨て)。相対指定はリポジトリを移動しても追従し、
絶対指定(別ドライブ等)はそのマシン内で有効。**解決できないパスも履歴に残す**
(退避ドライブ未接続で一時的に見えないだけの可能性があるため、起動時の自動掃除はしない)。
履歴からの削除手段は第 1 弾では持たない(`history.txt` の手編集で個別削除、
ファイル削除で全履歴リセットが可能。config 形式+責務別ファイルにした利点)。
起動時に履歴先頭の解決に失敗したら WARN してダイアログ(スキップ中は `_default`)へ落ちる。

補助: `--select-workspace`(値なしスイッチ)でスキップ設定を無視してダイアログを強制表示。
`--config` 指定時(2.4)はこのフロー全体をスキップする。

**テンプレート**(D15): 正本は git 追跡対象の **`apps/runner/config/_workspace_template.txt`**。
env 選択ブロック(コメントアウト一覧+有効 env 1 行)は `_main.txt` からの削除と
**同一 commit でこのファイルへ移動**する(clean checkout でも正本が常に存在する)。
`_default` 自動生成とダイアログの新規作成は、このファイルを `config/_main.txt` として
コピーする。これにより checkout 直後や release 配布物の初回起動でも従来どおり起動できる。
テンプレートの更新=このファイルの編集(C++ への埋め込みは更新性が悪く不採用)。

### 2.2 AP データフォルダ(D5)

`core/anet-core` の `app_util` に `GetAppDataDir()` を新設する。**ポータブルモード対応**(D13)で、
解決順は次のとおり:

1. **ポータブルモード**: `GetExecutableRootDir()/appdata`(= `apps/runner/appdata/`)が
   **ディレクトリとして存在すれば**それを返す。マーカーフォルダの存在だけで判定し、
   config には依存しない(VS Code の `data/` フォルダ、Notepad++ の `doLocalConf.xml` と同系の方式)。
2. **user モード**(マーカー無し):
   - Windows: `%APPDATA%\anet-lab\runner`
   - Linux: `${XDG_CONFIG_HOME:-~/.config}/anet-lab/runner`

- 実装は環境変数ベースで wx 非依存(core に wx を持ち込まない)。user モードのみ
  初回アクセス時に `create_directories`(ポータブル側はフォルダの存在自体が判定材料なので作らない)。
- 切替・解除は `appdata/` フォルダを作る・消すだけ(「フォルダ=真実」の運用と同型)。
  リポジトリ/配布物ごと退避すれば履歴等の AP データも一緒に移動する。dev リポジトリでは
  `appdata/` を置いておくと checkout ごとに履歴が独立する。release zip は同梱の有無で
  既定モードを選べる(同梱=ポータブル)。
- 将来 Linux 展開時の書き込みパーミッションは、ポータブル時は `appdata/` フォルダ、
  user モード時は user home 配下なので追加対応不要。

**置くもの**: RunnerApp データを**責務別のファイル**に分けて置く。責務ごとに分けるのは
**ファイルを消すだけでその責務のリセットになる**ため(履歴クリアと選好リセットを独立に行える)。
形式はいずれも既存 config と同じ `key = value` 行(読み=`Properties`、書き=`ConfigData` の
新設メンバ、§2.2.1)。

```text
<GetAppDataDir()>/
  history.txt      # 直近使用の記録。消す=履歴リセット
  prefs.txt        # ユーザー選好。消す=既定に戻る(ダイアログ復活)
  layout.txt       # (将来) UI パースペクティブ/レイアウト。消す=レイアウトリセット
```

`history.txt`:

```text
workspace.history.0 = dm_long          # MRU 先頭=最後に使った workspace
workspace.history.1 = D:\archive\apx-longrun
workspace.history.2 = _default
```

`prefs.txt`:

```text
workspace.dialog_skip = true           # 起動ダイアログのスキップ設定
```

- 履歴は**番号付きキー**とする。`ConfigData::Read(std::vector<std::string>)` は
  スペース区切り分割(`anet::Split`)のため、1 キーのリスト表現は空白入り Windows パス
  (`C:\Program Files\...`)で破綻する。
- 書き込みタイミング: `history.txt` は起動確定時(履歴更新と同時)、`prefs.txt` は
  選好の変更時(スキップチェック操作時)。多重起動は last-write-wins とし排他は持たない
  (小さいファイルであり、壊れても致命でない)。
- 将来の AP データは「消す単位=リセット単位」になるよう責務ごとにファイルを追加する
  (総称ファイルは作らない)。

既存前例 `WriteLastRunName`(`RunnerApp.cpp`、exe 隣接 `runname.txt` へ書き込み、
現在は呼び出しコメントアウト)は、この機会に **`GetAppDataDir()/runname.txt` へ書く形に
書き換えておく**(呼び出しの復活はしない。exe 隣接書き込みは release 配布時に install dir が
read-only の可能性があるため廃止)。`.gitignore` の `/apps/runner/runname.txt` は不要になるが
掃除は任意。

### 2.2.1 ConfigData 書き出しメンバの新設(部品化)

読み側(`Properties`)に対し Properties 形式の書き出し共通部品が無く、MetricsLogger が
無名 namespace ヘルパ `ConfigDataToConfigString`(`metrics_logger.cpp`)で自前実装している。
これを `ConfigData` のメンバへ部品化する(既存 `ToJson()` / `ToString()` と並ぶ第 3 の表現形):

- `ConfigData::ToPropertiesString() const` — `key = value` 行を OrderedMap 順(初出順)に出力。
  現行 `ConfigDataToConfigString` と同一挙動。
- `ConfigData::SaveProperties(const std::filesystem::path&) const` — `ToPropertiesString()` を
  UTF-8 で**一意な temp ファイル名**(同 directory 内)へ書き、**replace-existing セマンティクス**で
  宛先を置換して保存する(Windows では既存宛先への単純な `std::filesystem::rename` が失敗するため、
  既存ファイルを上書き置換できる方式を明示要件とする)。値に `#` または `//` を含むキーがあれば
  **保存せずエラー**(D18。Properties の読み側がコメント扱いして round-trip が壊れるため、
  「WARN して壊れた値を保存」はしない。呼び出し側が受理時点で事前検証する契約で、
  ここは最後の防波堤)。検査は `#`・`//` のみで **UNC(`\\`)は検査しない** — UNC は
  Properties 形式を壊さず、その拒否は workspace path の入力境界の責務(D18)。
- MetricsLogger の `ConfigDataToConfigString` は削除し `config_data.ToPropertiesString()` へ置換
  (config_data.txt ダンプの出力は不変)。

### 2.3 config 合成順(後勝ち merge の受け口)

現状 `ConfigManager::LoadFromFile`(`core/anet-core/src/config.cpp`)は
`map_ = props.ToConfigData().Map()` の 1 ファイル代入で、2 ファイル目を重ねる公開 API が無い
(`ConfigData::MergeFromChecked` は同一キー異値を拒否する conflict チェックであり流用不可)。

新しい合成順(workspace モード時):

```text
1. _main.txt を読む(従来どおり。common/metrics/agent/nn の include のみ、env 行は無い)
2. 導出値を注入: app.runs_dir = <workspace root>/runs
   (相対指定なら runner root 相対のまま `workspaces/<path>/runs`、絶対指定なら絶対パス。
    metrics_logger の runs_dir は `root_dir / runs_dir` 結合であり、絶対パスは
    std::filesystem::operator/ の仕様で root_dir を置換するため両形式とも無改修で通る)
3. workspaces/<ws>/config/_main.txt を後勝ち merge で読む(env include はここで解決される)
4. ApplyCmdLineOverrides(1回目) → AutoMerge(.$ 展開) → ApplyCmdLineOverrides(2回目)
   (既存順序を維持。AutoMerge は workspace merge の後に走ることが必須 —
    workspace config が env.$ 等の bind を書けるため)
```

実装は「後勝ちで上書き merge する 2 ファイル目の受け口」を ConfigManager に追加する
(`Properties::Load` の公開化でも、`ConfigData` への Overwrite merge 追加でも良い。
OrderedMap::Set は既存キー上書き+初出順保持なので後者が素直)。

**`app.runs_dir` の導出値保証**(D16): workspace モードでは、config 合成の全段
(workspace merge → CLI override → AutoMerge → CLI override)が完了した後の
**実効 `app.runs_dir` が手順 2 の導出値と一致することを検証**し、不一致なら起動エラー
(fail-fast)。「直接キーの有無」ではなく実効値で検証するのは、`app.$ = app.online` +
`app.online.runs_dir = ...` のように **AutoMerge 経由の間接上書き**が存在するため
(直接キー検査では素通りする)。上書きを許すと `runs` や外部絶対パスへ逃げられ、
「workspace は自己完結」の不変条件(§1)と Viewer の `runs/` 固定前提が破綻する。
**一致判定は注入した文字列との完全一致**(正規化・絶対化はしない): 表記違いの同一パスも
「触った」時点で契約違反としてエラーにする。より厳格で実装が自明、filesystem アクセスも不要。
`--config` 完全自己記述モード(§2.4)では従来どおり自由(optuna ハーネスは明示指定を継続)。

include 解決は無改修で成立する: workspace config 内の `$include <DropMerge.txt>` は
include 元基準(`workspaces/<ws>/config/`)で見つからず、config search dir
(`GetExecutableConfigDir()` = `apps/runner/config`)へフォールバックして解決される
(`ResolveIncludePath` の既存仕様)。`ConfigManagerOptions::config_search_dirs` の変更は不要。

### 2.4 `--config` = 完全自己記述モード(D6)

- `--config` 明示時は workspace 解決(選択フロー・後読み・runs_dir 導出)を一切行わない。
  INFO ログ 1 行で明示する。workspace を使いたい config は自分で
  `$include "workspaces/<ws>/config/_main.txt"` を書く(optuna ハーネスはこれを自動生成、§4)。
- `--config` と `--workspace` の同時指定は起動エラー(fail-fast)。
- これにより「runner が履歴先頭の workspace を勝手に重ねて二重適用する」事故が構造的に起きない。

### 2.5 env include の移管(D3)

- `apps/runner/config/_main.txt` から env 選択ブロック(`#$include <CartPole.txt>` 〜
  `$include <DropMerge.txt>`)を削除し、**同一 commit で `_workspace_template.txt`(D15、§2.1)へ
  移動**する。`_main.txt` は共通部(common / metrics_scalar / metrics_image / agent / nn)のみになる。
- `apps/runner/config/common.txt` の `app.runs_dir = runs` 行を削除(2.3 の導出へ移行。
  `--config` モードでは生成 config が明示指定する)。
- env 未選択の検出は追加しない。既存経路: `env.class_id` 空/未登録 →
  `EnvRepository::GetSingleDiscreteEnvFactory` / `GetBatchEnvFactory` が nullptr
  (`core/anet-core/src/env.cpp`)→ trainer の「Failed to create env.」エラーで Run は始まらない。
  小改善として、このエラーメッセージに class_id 値と
  「workspace config で env を `$include` してください」のヒントを追記する。
- env config(`DropMerge.txt` 等)が持つ `app.run_name` 上書きや NN 既定値は、
  workspace config 経由の include でも従来と同じ相対順(共通部の後)で適用されるため挙動不変。

### 2.6 影響を受ける既存動線

- `apps/10_run.bat`: 変更不要(workspace モードで起動、履歴先頭に従う)。
- `apps/11_batch_run.bat`: `key=value` 位置 override のみで `--config` 無し → workspace モード。
  履歴先頭の workspace に出力される。固定したい場合は bat に `--workspace` を足す(任意)。
- 旧 `GetRunsPath()`(`RunnerApp.cpp`、未使用の死に関数)はこの機会に削除。

## 3. MetricsViewer(PH2)

### 3.1 サーバ側 current 方式(D7)

- プロパティ `metricsviewer.runs-dir` を廃止し `metricsviewer.workspaces-dir`(既定 `workspaces`、
  cwd=`apps/runner` 基準は従来同様)に置換。`application.properties` /
  `additional-spring-configuration-metadata.json` / 起動 bat を追随。
  **UNC の workspaces-dir は起動時エラー**(fail-fast) — 配下の
  `metrics_cache.db` が UNC 上に乗り SQLite の lock 挙動が保証できないため。
  判定は正規化後の UNC root で行い、`\\server\share`・`//server/share` の両表記を拒否する。
- サーバは「現在の workspace」を **workspace epoch 付きの不変 snapshot
  `(epoch, root, in-memory キャッシュ群)`** として保持する(D14。epoch=単調増加。
  初期値: プロパティ `metricsviewer.initial-workspace`、既定 `_default`)。
  `RunScanner` は snapshot の root 配下 `runs/` を走査するよう改修。
- **initial-workspace の入力検証**: 値は**単一の直下ディレクトリ名のみ**。パス区切り
  (`/` `\`)・`..`・絶対パス・ドライブ相対(`C:` 等)・`.`・空・空白のみ・禁止文字(`#` `//`)を
  含む場合は**起動時エラー**(直下 workspace 限定を設定値で迂回させない)。契約としては
  「**正規化後の親ディレクトリが workspaces-dir 自身であること**」で固定する。
  構文が妥当だが実在しない場合は起動成功とし、走査は既存の runsDir 不在時と同様 WARN+空リスト。
- **snapshot 規律**: 全 API 処理(`getRuns` / `getMetrics` / `prioritizeRuns`)と ingest サイクルは
  **処理開始時に snapshot を 1 回取得**し、`listRunId` / `resolveRunDir` を含む全パス解決と
  **キャッシュの読み書き**をその snapshot に対して行う(処理途中で current を再参照しない)。
  これで「旧 workspace で存在確認 → 新 workspace の同名 Run を読取」の混線が構造的に起きない。
- **in-memory キャッシュ群は snapshot が所有する**: LodPageCache・RunWarningRegistry 等の
  runId 文字列キーの状態は snapshot ごとの別インスタンスとし、切替=新インスタンスを持つ
  新 snapshot への atomic swap。in-flight の旧 query は自分の旧インスタンスを読み書きし続ける
  ため、**読み取りも書き込みも構造的に混ざらない**(旧インスタンスは参照が切れたら GC)。
  epoch をキャッシュキーへ足す案は不採用 — Run フォルダをコピーすると `metrics_cache.db` ごと
  複製され **generation も同一**になるため、既存 PageKey(generation/runId/...)では区別できず、
  キーを持つ全箇所への波及も大きい。所有で解決する方が仕様が薄い。
- workspace epoch は PRD041 の**キャッシュ世代(generation)とは別概念**:
  世代=Run 単位のマスタ同一性(全再構築で更新)、epoch=Viewer プロセスが今どの workspace を
  見ているかの同一性(切替で更新)。上記のとおり generation は workspace を区別できない
  (フォルダコピーで同一になる)ので、epoch の代用にならない。

**snapshot 所有リソースのライフサイクル**: 所有対象は「参照消滅後の GC だけでよい cache」と
「close・停止処理が必要な resource」の 2 分類に分け、後者は
**旧 snapshot への新規処理の割り当てを停止 → 既存利用者がゼロになった時点で明示終了**する契約とする。
終了は **close-on-zero 登録方式**: 切替 POST は利用者ゼロを**待たず**、
「利用者カウントがゼロになった時点で close 必須リソースの closeAll を実行する」処理を
旧 snapshot に登録して応答を返す(in-flight の旧 query がゼロなら登録時に即実行)。
旧 query は継続でき、新 workspace の ingest は POST 応答後すぐ開始できる。
**lease の原子性**: 利用者カウントの取得(`acquireLease()`)と retire 判定は
**同一の同期境界内**で行う — query が snapshot を読んだ直後・登録前に close が走る隙間を
作らない。retire 済み snapshot を引いた場合は現行 snapshot を取り直して再試行し、
取得した lease は処理全体を通して `finally` で必ず release する。close-on-zero の実行は
ちょうど 1 回(再入・重複実行しない)。

| 状態 | 分類 | 切替時の扱い |
|---|---|---|
| LodPageCache | GC のみ | 新 snapshot は新インスタンス。旧は参照消滅で GC |
| RunWarningRegistry | GC のみ | 同上 |
| GzipInputSessions(block 間で open stream 保持) | **close 必須** | 新規割り当て停止後、利用者ゼロの時点で `closeAll()` を明示実行 |
| IngestScheduler の priorityRunIds | GC のみ | 新 snapshot は空集合で開始(フロントが切替後に prioritize を再 POST、§3.3) |
| IngestScheduler の priority/background cursor | GC のみ | 新 snapshot で初期化 |

- 上記以外にも workspace 固有の runId / Path を保持するメモリ状態を追加する場合は
  snapshot 所有へ寄せる(プロセスグローバルに置かない)ことを一般則とする。
  **例外(process-global のまま維持する安全機構)**: `MetricsCacheDatabase.lifecycleLocks`
  (絶対パスキーの DB lifecycle lock)と query semaphore は snapshot 所有に**しない**。
  これらは「同一パスの DB への並行アクセスをプロセス全体で直列化する」機構であり、
  snapshot 別に分けると A→B→A と戻ったとき旧 A snapshot の query と新 A snapshot の
  ingest が別 lock を持ち同期できなくなる。絶対パスキーなので workspace を跨いでも衝突しない。
- **JVM 終了時**: shutdown 処理で current snapshot に対しても同じ終了系列を実行する —
  ingest の新規 cycle 開始を停止 → 進行中 cycle の完了(または中断)→
  `GzipInputSessions.closeAll()` 等の close 必須リソースを明示終了。
  切替時と終了時でライフサイクルの出口を 1 本化する。
- **受入条件**: 切替後、旧 workspace のファイル(特に gzip stream が掴む `metrics.jsonl.gz`)を
  Windows 上でロックし続けないこと(利用者ゼロ到達後)。gzip 変換中だった Run を後で
  再訪した場合は既存のキャッシュ再開規則(source offset)に従う。
- Run 判定(`metrics.jsonl` / `.jsonl.gz` を持つ直下ディレクトリ)は不変。
  optuna trial 代表フォルダはこのフィルタで自然に除外される(既存テストで固定済みの挙動)。
- Viewer の列挙・切替対象は `workspaces/` 直下のみ。Runner 側で絶対パス指定した
  外部 workspace の閲覧は後続 PRD の「アタッチ」(§7)で扱う(当面は `workspaces/` 配下へ
  置くかフォルダを戻して見る)。

### 3.2 API 追加

**wire contract**(D14/#5 で固定):

`GET /api/workspaces.json` → 200:

```json
{ "current": "dm_long", "workspaces": ["_default", "dm_long", "dm_opt"] }
```

- `workspaces`: `workspaces/` 直下で `runs/` または `config/` を持つディレクトリ名、**名前昇順**
- `current`: サーバ保持値。列挙に含まれない場合(不在 initial-workspace 等)も**そのまま返す**

`POST /api/workspace` request(閉じたスキーマ):

```json
{ "name": "dm_long" }
```

**判定順序**(この順で評価、先に該当したものが応答。エラー body は既存契約
`{"code": ..., "message": ...}` 形式に従う):

1. body 不正 / `name` 欠落・空文字・非文字列 / 未知フィールドあり
   → 400 + `{"code": "invalid_request", "message": ...}`
2. **切替ゲートを取得**する。ゲートは (a) LoadingThread が新しい ingest cycle を開始することを
   止め、(b) 複数の切替 POST を直列化する(同時 POST は順に処理される)
3. **ゲート内で `name == current` を再評価** → 同じなら **204 no-op**(epoch 不変、
   再初期化しない、ゲート解放)。ゲート取得**後**に再評価するのは、同一 target への同時 POST が
   両方ともゲート前の古い current を読んで二重に切替扱いになるのを防ぐため
   (**同一 target への同時 POST では swap と epoch 増加はちょうど 1 回**)。
   冪等性を優先し、不在の current(initial-workspace 不在等)と同名の POST も
   「既にその状態」として成功させる(列挙照合より先)
4. 列挙に存在しない name(パストラバーサル含む) → 404 +
   `{"code": "unknown_workspace", "message": ...}`(ゲート解放)
5. 切替実行 → 204

切替実行(手順 5)のゲート内処理:

1. 進行中の ingest cycle(取り込みブロック)の完了を待つ
2. **新しい in-memory キャッシュ群インスタンスを持つ新 snapshot `(epoch+1, 新root, 新キャッシュ群)`
   へ atomic に swap**(§3.1)
3. 旧 snapshot に **close-on-zero を登録**(§3.1): 新規割り当てを停止し、利用者ゼロ時点で
   close 必須リソース(GzipInputSessions 等)が閉じられる。利用者ゼロを**待たずに**次へ進む。
   GC のみで良い cache は放置(参照が切れたら GC)
4. ゲートを解放し 204 を返す。次の ingest cycle は **swap 後の新 snapshot を取得して開始**する
   (POST 成功後に旧 workspace への新規 ingest 処理は発生しない)

ingest cycle も API query と同じ snapshot 規律に従う: cycle 開始時に snapshot を 1 回取得し、
cycle 中は同じ snapshot を使い続ける(途中で current を再参照しない)。

Run フォルダ内の `metrics_cache.db` はパス非依存(同一性検証は kind/size/mtime/SHA-256)なので、
workspace を移動・切替してもそのまま有効。

### 3.3 フロントエンド

- workspace セレクタ(`<select>`)を global-controls 領域に追加。起動時に
  `/api/workspaces.json` を取得し、localStorage の第 5 キー
  `anet.metricsviewer.workspace`(既存 4 キーのパターン踏襲)に選択を永続化、
  保存値があれば起動時に POST で復元する。**保存値が列挙に無い場合は POST せず
  サーバの `current` に従い、保存値を `current` で上書きする**(サイレントフォールバック)。
- 切替時のクライアント処理: DataCache クリア、selectedRuns / runColorMap / viewports /
  hiddenLegendSeries / initialSelectionApplied をリセット、query revision バンプ、
  Run リスト再取得(初回選択ロジックが「最新 1 件」を選び直す)。
- Playwright テスト(runs-dir をプロパティ指定している 5 ファイルと
  `RunListPlaywrightTest` の DOM 前提)を workspaces-dir 前提に追随。
  セレクタ追加による `index.html` 構造変更もテストの selector に反映。

### 3.4 起動 bat(D11)

- `apps/22_metrics_viewer_java.bat`: workspaces-dir 既定で起動(実質変更は僅少)。
- `apps/22_metrics_viewer_java_optuna.bat`: **削除**。optuna workspace はセレクタで開ける。
  2 つの workspace を同時に見たい場合は `--server.port` を変えて手動で 2 プロセス起動する
  (bat は用意しない)。

## 4. optuna ハーネス(PH3)

### 4.1 `--workspace` と導出既定値(D8, D10)

`--workspace <workspace_path>` を追加(既定 `_default`)。runner と同じパス形式
(相対=`<runner root>/workspaces/` 基準、絶対可)で解決し、指定(または fallback)から
以下を導出する。**明示指定で個別に勝てるのは `--storage` / `--optuna-artifact-dir` 系のみ**
(いずれも `<ws>/optuna` 配下限定、後述):

| 項目 | 導出値(`<ws>` = 解決済み workspace root。相対指定なら runner root 相対のまま) |
|---|---|
| runs dir | `<ws>/runs` に**固定**。**`--runs-dir` 引数は run 系 CLI から削除**(完全一致しか許されない引数を互換温存しない=クリーンブレーク) |
| `--storage` | `sqlite:///<ws>/optuna/optuna.db` |
| `--optuna-artifact-dir` | `<ws>/optuna/artifacts` |
| harness.log | `<ws>/optuna/harness.log`(引数化しない)。**生成は run-trial / run-study のみ**(現実装の attach 箇所と同じ。dry-run / cleanup-running / summarize-study は生成しない) |
| `--source-storage` / `--source-artifact-dir`(summarize-study) | 同上の storage / artifacts(workspace 導出) |
| `--target-storage` / `--target-artifact-dir`(summarize-study) | **省略時は対応する source を継承する**。workspace 既定へは出さず、workspace 非依存を維持する。`--target-storage` は現行挙動を維持し、`--target-artifact-dir` は独立した既定値(現行 `runs_optuna/artifacts`)を廃止して source 継承へ変更する |

対象サブコマンド: dry-run / run-trial / run-study / summarize-study / cleanup-running。
trial 採番のディレクトリ走査(`scan_existing_trial_numbers`)は既存機構のまま新パスで動く。
**`{study}` 等の placeholder 展開は廃止**する(現行の展開対象は `runs_dir` のみであり、
`--runs-dir` 削除で対象が消滅する。`resolve_runs_root` / `scan_existing_trial_numbers` /
`make_trial_context` の format 展開処理を削除して簡素化)。
パス解決の副作用(`storage_url_from_text` の `parent.mkdir`)は
後述の preflight 4 段階に従い解決段階から分離する。
`--runs-dir` 削除の追随箇所: `build_run_study_copy_args`(copy args から除去)、
`build_study_user_attrs`(`last_runs_dir` 廃止)、`make_manifest`、usage 文字列。

**ハーネス側のパス検証**: `--workspace` の値は runner と同じ契約(D18)で検証する —
`#`・`//` を含む、または UNC(`\\` 始まり)なら、**preflight 段階 2 より前**
(=DB・artifact・出力先のいかなる生成よりも前)に即エラー終了する。

**run 系は workspace 外への出力を禁止**(D19): dry-run / run-trial / run-study では、
runs dir は `<ws>/runs` 固定(引数なし)、`--storage` / `--optuna-artifact-dir` は
**解決結果が `<ws>/optuna` 配下でなければエラー**とする(preflight 段階 2 で検証)。
これにより ADR 0021 の「storage / artifact を含む自己完結」が無条件に成立する。
summarize-study の跨ぎ集約用 override(source/target)は集約が目的の機能であり、この禁止の対象外。

**引数ごとの許容 root**(canonical containment だけでは wrong-bucket を防げないため、
D1/D10 の `config/runs/optuna` 役割分離を出力先契約にも適用する):

| 項目 | 許容される解決結果 |
|---|---|
| runs dir | `<ws>/runs` に固定(引数なし。サブディレクトリ化は不可 — 直下 1 階層のみ走査する Viewer から Run が不可視になるため、導出値そのもの以外を許さない) |
| `--storage` / `--optuna-artifact-dir` | `<ws>/optuna` **配下** |

wrong-bucket 指定(`--storage=<ws>/runs/x.db` 等、workspace root 配下だが optuna/ 外)は拒否する。

**「workspace root 配下」の判定方法**(文字列 prefix 判定は禁止):

1. `--storage` 系の値はまず「**SQLite URL(`sqlite:///...`) | filesystem path**」として解析する。
   **scheme delimiter の `//` は検査対象外**(正常な既定値 `sqlite:///...` を誤拒否しない)。
   非 SQLite scheme(`postgresql://` 等)は拒否。URL は filesystem path へ変換し、
   以降の検査は**変換後の path 部分だけ**に適用する
2. workspace root と target の両方を `resolve(strict=False)` 相当で正規化する
   (存在しない末尾要素があっても、存在する親までの symlink / junction を解決する)
3. **path-component 単位の包含判定**を行う(`workspaces/dm_opt2` が `workspaces/dm_opt` の
  prefix 文字列一致で誤通過する事故を構造的に防ぐ)
4. 手順 2 の正規化により、既存親ディレクトリの symlink / junction を経由した箱外 escape も拒否される
5. path 部分に `#`・重複 separator(`//`)・UNC root が含まれる場合も拒否する
  (禁止文字契約と同じ理由)

**workspace の事前検証**(D19): runner は `--config` 起動のため Runner 側の `_default`
自動生成を通らず、現行 `Properties` は include 不在を WARN で継続するため、放置すると
trial 採番・DB 更新・artifact 作成の**後**に env 未選択で失敗し中途生成物が残る。
これを防ぐため、**サブコマンド別に、副作用より前に**次を検証し、無ければ即エラー終了する:

| サブコマンド | 副作用前に検証するもの |
|---|---|
| dry-run / run-trial / run-study | 解決済み workspace root と `config/_main.txt`(必須) |
| cleanup-running | 実際に使う storage のみ(workspace config は読まないので不要) |
| summarize-study | source/target の storage・artifact のみ(workspace config 不要) |

cleanup-running / summarize-study は明示 `--storage` 等だけで完結する呼び出しが、
無関係な workspace(`_default` 等)の不在で失敗してはならない(workspace は storage 等の
既定値導出にだけ使う)。ハーネスは非対話バッチなので**箱の自動生成はしない**
(typo した workspace 名で黙って新箱ができる事故を防ぐ。`_default` も Runner が生成済みなら通る)。

**preflight の 4 段階分離**: 「検証」の意味は source(既存必須)と target(作成可)で異なる。
現行の `storage_url_from_text` はパス解決時に `parent.mkdir` まで実行するため、
これを分離し、次の順序を守る:

1. **パス解決**(mkdir 等の副作用なし)
2. **source / target の存在・型検証**(すべて副作用なしで実施):
   - cleanup-running の storage: **既存必須**。存在しない DB や親ディレクトリを作成せずに失敗
   - summarize-study: 下表の型検証を行い、**一方が不正なら他方を生成しない**

   | 対象 | 検証 |
   |---|---|
   | source storage | 既存の regular file であること |
   | source artifact | 既存の directory であること |
   | target storage | 不存在、または regular file であること |
   | target artifact | 不存在、または directory であること |
3. **すべての preflight 成功**(run 系は workspace root / `config/_main.txt` の検証を含む)
4. **target 系の作成**: summarize-study の target storage / artifact、run 系の出力先
   (runs dir / trial フォルダ)、harness.log の生成はここで初めて行う

- run-trial / run-study の出力先作成は、事前に判定可能な入力(workspace / config 等)の
  検証完了後に限る
- **artifact store の初期化失敗は fail-fast**: 現行実装の「初期化失敗を WARN して学習継続」を
  廃止し、trial 採番・Optuna DB 更新より前にエラー終了する(D10/D19 の fail-fast 方針に統一)
- dry-run は storage / artifact へ**接続も作成もしない**(runs root の走査と config / manifest
  書き込みのみ、現行どおり)

新引数の同期が必要な既存箇所: `build_run_study_copy_args`(Dashboard 貼り付け用 args に
`--workspace` を含める)、`build_study_user_attrs`(`last_workspace` を追加)、
`make_manifest`。usage 文字列のパス例も更新。

### 4.2 生成 config への workspace include(D9)

runner へ `--workspace` は**素通ししない**。理由: runner の workspace 後読みは
trial の NN override より後に workspace config(env include=NN 既定値を含む)を適用してしまい、
探索パラメータを潰す。正しい合成順はハーネスが生成 config 内で表現する:

```text
$include <_main.txt>                                  # 共通部(env 行は無い)
$include "<abs path>/workspaces/<ws>/config/_main.txt" # workspace config(env 選択)
$include <DropMerge_optuna.txt>                        # --extra-config
app.run_name / app.runs_dir / train.seed / exp_exit_step / NN override 群   # trial 固有(最後勝ち)
```

`render_config` の include 2 行目として挿入する(絶対パスは既存の
`config_include_line` が `""` 囲みで出力する)。runner は `--config` 起動なので
workspace 解決は走らず(D6)、二重適用は起きない。

env 選択が workspace config に移ることで、ハーネスの「`--base-config` で env が選択済み」
という暗黙前提が「`--workspace` の config で env が選択済み」に変わる。`--base-config` /
`--extra-config` の意味は不変。

### 4.3 dashboard / 移行時の注意

- `apps/23_optuna_dashboard.bat`: workspace path を引数に取り、**runner と同じパス解決規則**
  (相対=`workspaces/` 基準、絶対可)で解決した。相対値の基準は**呼出元 cwd ではなく
  bat 自身の位置**とする — `%~f1` 単独では cwd 基準になるため、相対値は
  `%~dp0runner\workspaces\` へ結合してから正規化する
  `sqlite:///<ws>/optuna/optuna.db --artifact-dir <ws>/optuna/artifacts` で起動する形へ変更。
  **bat の契約**(現行の `mkdir` と「DB 不在でも WARN 起動」は廃止):
  - workspace 引数は**必須**(無引数は usage を表示して非 0 終了)
  - `#`・`//`・UNC(`\\` 始まり)は起動前に拒否(D18 の入力境界)
  - workspace root・`optuna.db`・`artifacts/` は**既存必須**(dashboard は閲覧ツールであり
    何も生成しない)。不足時は**何も生成せず非 0 終了**
  - これにより typo した workspace 名で空 DB や箱が作られる事故を構造的に防ぐ
- **旧 study の再開**: Dashboard の `00_last_run_study_args` に保存された旧引数は
  **そのままでは再利用不可** — `--runs-dir` は引数自体が廃止(§4.1)で未知引数エラー、
  `--storage` 等の外部出力 override も `<ws>/optuna` 配下限定によりエラーになり、
  env include も `_main.txt` から消えている。
  再開手順 = **`--workspace` を追加し、`--runs-dir` を除去、`--storage` 等の外部出力
  override は削除(または `<ws>/optuna` 配下へ更新)**する。optuna.md に注記する。
- optuna.db を workspace 単位に分けることで Dashboard の study 横断閲覧は workspace 内に
  限定される。跨ぎたい場合は `summarize-study --target-storage` で集約 db を作る(既存機構)。

## 5. 移行・文書(D12)

- 既存 `runs/`・`runs_optuna/`・`runs_optina/`・`runs_DropMerge4`・`runs_ImageCls` 等:
  手動フォルダ移動のみ。コード migration・自動リネームは行わない
  (optuna.md の「既存 `runs_optina/` 生成物の migration はしない」前例踏襲)。
  例: `runs/*` → `workspaces/_default/runs/`、`runs/apx-longrun/*` → `workspaces/apx-longrun/runs/`。
- **optuna 系フォルダ(`runs_optuna` 等)の分割マッピング**: 旧フォルダは Run と harness 生成物が
  混在しているため、移行先を分割する —
  - seed run / trial 代表フォルダ(`<study>_<trial>*`) → `workspaces/<ws>/runs/`
  - `optuna.db` / `harness.log`(`.1`/`.2` 含む) / `artifacts/` → `workspaces/<ws>/optuna/`
  - 移行後、Dashboard に保存済みの再開引数(`00_last_run_study_args`)は新形式
    (`--workspace` 付き・外部出力 override 除去)へ更新して使う(§4.3)。
  - **静止条件**: 移動前に harness(run-study/run-trial)・optuna-dashboard・実行中 study を
    すべて停止し、SQLite を clean shutdown させる。`optuna.db-wal` / `optuna.db-shm` が
    残っている場合は `optuna.db` と**同時に**移動する(片方だけの移動は DB 破損の元)。
- 正本 docs の更新対象:
  - `docs/design/020_user_guide_run.jp.md`(`app.runs_dir` 表、出力先の説明)
  - `docs/design/100_runtime_and_configuration.jp.md`(設定解決 6 ステップに workspace 後読みを追加)
  - `docs/design/160_applications_and_tools.jp.md`(entry point 一覧、bat)
  - `docs/design/210_metrics_viewer.jp.md`(§7 設定表、§9 HTTP API、runs-dir → workspaces-dir)
  - `docs/design/030_user_guide_analysis.jp.md`(Viewer 利用手順、セレクタ)
  - `docs/design/optuna.md`(出力レイアウト、共通引数、`--workspace`、再開時の注意)
- `CONTEXT.md`: 用語 3 件(ワークスペース / workspace config / Run作業セット改訂)。
- リリースパッケージング(ADR 0017 のミラー方式)への影響: `workspaces/` は実行時生成のため
  コピー対象追加は不要。テンプレート自動生成(§2.1)により配布物の初回起動も従来どおり動く。

## 6. 段階分割と受け入れ基準

**リリース単位**: PH1〜PH3 は**同一クリーンブレーク変更として一括完了が前提**。
段階は実装・レビューの順序であって、個別リリースの単位ではない
(PH1 完了時点で `_main.txt` の env 移管が入るため、旧前提の Viewer 既定値や optuna は
そのままでは噛み合わない)。中間状態での暫定運用: Viewer は既存の
`--metricsviewer.runs-dir` 系起動引数で任意の runs ディレクトリを直接指せば動作、
optuna は PH3 完了まで実行しない。

**PH1: Runner / config 機構**(§2)

- `--workspace` / ダイアログ(履歴コンボ+スキャン一覧のワンクリック起動) / 履歴 / `_default` 自動生成 /
  `history.txt`/`prefs.txt` の読み書き(ファイル削除=リセットの挙動含む)が動作
- パス指定の両形式(相対=workspaces/ 基準、絶対=別ドライブ等)で Run 出力・config 後読みが機能
- `ConfigData::ToPropertiesString` の round-trip 単体テスト(Properties で読み戻して一致。
  履歴の空白入りパス・MRU 更新・上限切り捨てを含む)、MetricsLogger 置換後の config_data.txt ダンプ不変
- `SaveProperties` の**既存ファイルへの 2 回目保存**テスト(既存 `history.txt`/`prefs.txt` を
  replace-existing で上書きできること。Windows で実行)
- `GetAppDataDir()` のポータブル判定(appdata/ 有無で切替)の単体テスト
- workspace config の後勝ち merge と `$include` search dir 解決の単体テスト
  (config_test.cpp に追加。2 ファイル目 merge、workspace 側 override、AutoMerge との順序)
- `--config` 時に workspace 解決が走らないこと、フラグ競合 3 組(`--config`×`--workspace` /
  `--config`×`--select-workspace` / `--workspace`×`--select-workspace`)が全てエラーになること
- workspace モードで実効 `app.runs_dir` が導出値とずれると起動エラーになること(D16)。
  直接キー(workspace config / CLI override)に加え、**間接経路(`app.$ = app.online` +
  `app.online.runs_dir`)のケースを必ず含める**
- workspace path の禁止文字(`#`・`//`)が受理時点と `SaveProperties` の両方で fail-fast すること、
  **UNC パス(`\\server\share`・`//server/share` の両表記)が受理時点で拒否されること**(D18)
- ダイアログ: Cancel でアプリ終了、空選択で OK が無効、初回起動で `_default` プリセット(D17)
- env 未選択 workspace で起動 → 「Failed to create env.」+ヒントで Run が始まらないこと
- 既存 Run(手動で `workspaces/_default/runs/` へ移動)の config_data.txt ダンプが従来と同等であること

**PH2: MetricsViewer**(§3)

- workspaces.json 列挙 / workspace 切替 / 切替後の Run リスト・メトリクス表示、
  API contract(400/404/no-op 204)どおりの応答
- 異 workspace 同名 Run で切替してもデータが混ざらないこと(epoch 検証)
- **遅延 query 中に切り替える結合テスト**: fixture は「**Run フォルダを別 workspace へ
  コピーした同名・同 generation の Run**」で固定(generation では区別できない最悪ケース)。
  旧 snapshot の query が新 workspace のキャッシュページを読まず、旧インスタンスに閉じて
  完走すること(D14 snapshot 所有)
- **切替ゲートの結合テスト**: latch で ingest 処理を停止した状態で切替 POST →
  POST が進行中 ingest の完了を待つこと / POST 成功後の次 cycle が新 workspace を使うこと /
  同時に複数の切替 POST を投げて直列化されること /
  **同一 target への同時 POST で swap と epoch 増加がちょうど 1 回であること**
- **initial-workspace の入力検証テスト**: traversal(`..`)・絶対パス・パス区切り入り・
  ドライブ相対(`C:`)・`.`・空・空白のみの値で起動時エラーになること、
  構文妥当で不在の値では WARN+空リストで起動すること
- **UNC workspaces-dir の起動時拒否**: `metricsviewer.workspaces-dir` に
  `\\server\share`・`//server/share` のいずれの表記を指定しても起動時エラーになること
- **A→B→A 復帰の競合テスト**: workspace A の旧 snapshot query が継続中に A→B→A と切り替えて
  A の ingest を開始 → process-global の DB lifecycle lock により同一 Run の DB アクセスが
  正しく直列化されること(lock が snapshot 化されていないことの検証)
- **lease 競合テスト**: snapshot 参照の取得直後〜`acquireLease()` 前でスレッドを停止し、
  その間に切替(retire)を完了させる → 再開したスレッドが retire を検知して現行 snapshot を
  取り直すこと、close-on-zero の実行がちょうど 1 回であること(Sp1 の同期境界検証)
- **リソース解放**: 切替後に旧 workspace のファイル(gzip stream の `metrics.jsonl.gz` 等)が
  Windows 上でロックされ続けないこと(旧 workspace フォルダの移動・削除が成功すること)
- `initial-workspace` 不在での起動(WARN+空リスト)、localStorage 保存値が列挙に無い場合の
  サイレントフォールバック
- 既存単体テスト(RunScannerTest 等)と Playwright テストの追随、全テスト緑

**PH3: optuna ハーネス + bat + docs**(§4, §5)

- `--workspace` 導出値で dry-run → run-trial → run-study が新レイアウトに出力されること
- 生成 config に workspace include が入り、seed run が Viewer の optuna workspace に見えること
- 明示引数 override が導出に勝つこと(**`--storage` / `--optuna-artifact-dir` 系のみ**。
  いずれも `<ws>/optuna` 配下限定)
- **`--runs-dir` の削除確認**: run 系サブコマンドに `--runs-dir` を渡すと未知引数エラーに
  なること、runs dir が常に `<ws>/runs` に導出されること
- **workspace 不在時の fail-fast**: run 系サブコマンドが trial 採番・DB 更新・artifact 作成の
  前にエラー終了し、中途生成物(採番済みフォルダ・DB エントリ)が残らないこと(D19)
- cleanup-running / summarize-study が明示 `--storage` 等のみで workspace 不在でも成功すること、
  run 系で `<ws>/optuna` 外へ出る明示 override が**エラー**になること(D19)
- **包含判定のテスト**: sibling-prefix(`workspaces/dm_opt2` を `dm_opt` 選択時に指定)・
  `..` 入りパス・symlink/junction 経由の箱外 escape・`--storage` 等の値の禁止文字(`#`/`//`)、
  がいずれも拒否されること
- **wrong-bucket のテスト**: `--storage=<ws>/runs/x.db` / `--optuna-artifact-dir=<ws>/config`
  等が workspace root 配下でも拒否されること
- **dashboard bat のテスト**: 無引数で usage+非 0 終了 / typo した workspace 名
  (root・optuna.db・artifacts 不足)で何も生成せず非 0 終了すること /
  **別 cwd から起動しても同じ workspace に解決されること**(相対基準が bat 位置であること)
- **旧 study 再開の契約テスト**: 旧引数(`--runs-dir runs_optuna` 等)のままの再開が失敗すること /
  §5 の分割移行 + 引数更新(`--workspace` 付き・`--runs-dir` 除去・外部 override 削除)後の
  再開が成功すること
- dry-run / cleanup-running / summarize-study が workspace ディレクトリや harness.log を
  生成しないこと(harness.log の生成は run-trial / run-study のみ)
- **preflight の非作成保証**(D19 4 段階): 存在しない storage を cleanup-running へ渡しても
  DB・ディレクトリが作成されないこと / summarize-study の source が不正な場合に target 側へ
  何も作成されないこと / run-study の workspace/config 検証が失敗した場合に
  storage・artifact・harness.log が作成されないこと
- **storage URI 解析のテスト**: 正常な既定値 `sqlite:///...` が拒否されないこと /
  SQLite URL と bare filesystem path の指定が等価に解決されること /
  変換後 path 部分の禁止文字(`#`・重複 separator・UNC)が拒否されること /
  非 SQLite scheme が拒否されること
- **source/target 型検証のテスト**: source storage が directory・source artifact が file 等の
  型違反で失敗し、target 側に何も生成されないこと / target 省略時に source が継承されること
- **artifact 初期化失敗の fail-fast テスト**: artifact store 初期化を失敗させたとき、
  trial 採番・DB 更新の前にエラー終了すること(WARN 継続しないこと)
- **ハーネスの禁止パス検証**: `--workspace` に `#`・`//`・UNC(`\\server\share` と
  `//server/share` の両表記)を渡すと、DB・artifact・出力先のいかなる生成よりも前に
  拒否されること(D18)
- bat 2 件(22 削除 / 23 引数化=runner と同じパス解決規則)と docs 更新

## 7. スコープ外(後続 PRD 候補)

- Viewer の任意パス「アタッチ」(退避先 workspace を移動せず開く)
- Runner GUI での実行時 workspace 切替(現状は再起動で切替)
- workspace 単位の UI パースペクティブ保存等、`workspaces/<ws>/settings/` の活用
- タグ・メタデータによる Run 横断検索(方針として不採用、ADR 0021)
- 複数 workspace の同時閲覧 UI(当面は手動の別ポート起動)
