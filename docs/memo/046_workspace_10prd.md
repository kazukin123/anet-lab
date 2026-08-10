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

**テンプレート**: 自動生成される `config/_main.txt` は、移管前の `_main.txt` にあった
env 選択ブロック(コメントアウト一覧+有効 env 1 行)をそのまま持つ。これにより
checkout 直後や release 配布物の初回起動でも従来どおり起動できる(現状維持)。
有効 env は移管時点の `_main.txt` の有効行に従う。

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
  UTF-8・temp ファイル→rename の置換書きで保存。値に `#` または `//` を含むキーは
  `LOG::warn()`(Properties の読み側がコメント扱いして round-trip が壊れるため。
  パスに `#` を含む履歴等の検出用)。
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
3. workspaces/<ws>/config/_main.txt を後勝ち merge で読む
   (env include はここで解決される。明示的な app.runs_dir 上書きも可能=導出値より勝つ)
4. ApplyCmdLineOverrides(1回目) → AutoMerge(.$ 展開) → ApplyCmdLineOverrides(2回目)
   (既存順序を維持。AutoMerge は workspace merge の後に走ることが必須 —
    workspace config が env.$ 等の bind を書けるため)
```

実装は「後勝ちで上書き merge する 2 ファイル目の受け口」を ConfigManager に追加する
(`Properties::Load` の公開化でも、`ConfigData` への Overwrite merge 追加でも良い。
OrderedMap::Set は既存キー上書き+初出順保持なので後者が素直)。

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
  `$include <DropMerge.txt>`)を削除。`_main.txt` は共通部
  (common / metrics_scalar / metrics_image / agent / nn)のみになる。
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
- サーバは「現在の workspace 名」を 1 つ保持する(初期値: プロパティ
  `metricsviewer.initial-workspace`、既定 `_default`)。`RunScanner` は
  `<workspaces-dir>/<current>/runs` を走査するよう改修(final フィールドの可変化、
  または current を保持する上位コンポーネント経由で走査先を解決)。
- Run 判定(`metrics.jsonl` / `.jsonl.gz` を持つ直下ディレクトリ)は不変。
  optuna trial 代表フォルダはこのフィルタで自然に除外される(既存テストで固定済みの挙動)。
- Viewer の列挙・切替対象は `workspaces/` 直下のみ。Runner 側で絶対パス指定した
  外部 workspace の閲覧は後続 PRD の「アタッチ」(§7)で扱う(当面は `workspaces/` 配下へ
  置くかフォルダを戻して見る)。

### 3.2 API 追加

| メソッド | パス | 動作 |
|---|---|---|
| GET | `/api/workspaces.json` | `workspaces/` 直下で `runs/` または `config/` を持つディレクトリ名の列挙 + 現在の workspace 名 |
| POST | `/api/workspace` | `{name}` で切替。列挙に含まれる名前のみ受理(パストラバーサル防止)。204 |

切替時のサーバ処理:

1. IngestScheduler / LoadingThread の対象を新 workspace へ切替(進行中の取り込みブロック完了を待つ)
2. キャッシュ世代(generation)をバンプし、LodPageCache・RunWarningRegistry 等の
   runId 文字列キーの in-memory 状態をクリア(異 workspace の同名 Run 衝突対策)
3. 以後の `/api/runs.json` / `/api/metrics.json` は新 workspace を返す

Run フォルダ内の `metrics_cache.db` はパス非依存(同一性検証は kind/size/mtime/SHA-256)なので、
workspace を移動・切替してもそのまま有効。

### 3.3 フロントエンド

- workspace セレクタ(`<select>`)を global-controls 領域に追加。起動時に
  `/api/workspaces.json` を取得し、localStorage の第 5 キー
  `anet.metricsviewer.workspace`(既存 4 キーのパターン踏襲)に選択を永続化、
  保存値があれば起動時に POST で復元する。
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
以下を導出する。**既存引数の明示指定が個別に勝つ**:

| 引数 | 導出既定値(`<ws>` = 解決済み workspace root。相対指定なら runner root 相対のまま) |
|---|---|
| `--runs-dir` | `<ws>/runs` |
| `--storage` | `sqlite:///<ws>/optuna/optuna.db` |
| `--optuna-artifact-dir` | `<ws>/optuna/artifacts` |
| harness.log | `<ws>/optuna/harness.log`(引数化しない。`--runs-dir` override の影響を受けず optuna/ 固定) |
| `--source-storage` / `--target-storage` / `--source-artifact-dir` / `--target-artifact-dir`(summarize-study) | 同上の storage / artifacts |

対象サブコマンド: dry-run / run-trial / run-study / summarize-study / cleanup-running。
`{study}` 等の placeholder 展開、trial 採番のディレクトリ走査(`scan_existing_trial_numbers`)、
`storage_url_from_text` の `parent.mkdir` は既存機構のまま新パスで動く。

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

- `apps/23_optuna_dashboard.bat`: workspace 名を引数に取り
  `sqlite:///workspaces/<ws>/optuna/optuna.db --artifact-dir workspaces/<ws>/optuna/artifacts`
  で起動する形へ変更。
- **旧 study の再開**: Dashboard の `00_last_run_study_args` に保存された旧引数
  (`--runs-dir runs_optuna` 等)は明示指定として引き続き有効。ただし env include が
  `_main.txt` から消えるため、**移管後に旧 study を再開する場合は `--workspace` の追加が必要**
  (生成 config に workspace include が入らないと env 未選択で fail する)。optuna.md に注記する。
- optuna.db を workspace 単位に分けることで Dashboard の study 横断閲覧は workspace 内に
  限定される。跨ぎたい場合は `summarize-study --target-storage` で集約 db を作る(既存機構)。

## 5. 移行・文書(D12)

- 既存 `runs/`・`runs_optuna/`・`runs_optina/`・`runs_DropMerge4`・`runs_ImageCls` 等:
  手動フォルダ移動のみ。コード migration・自動リネームは行わない
  (optuna.md の「既存 `runs_optina/` 生成物の migration はしない」前例踏襲)。
  例: `runs/*` → `workspaces/_default/runs/`、`runs/apx-longrun/*` → `workspaces/apx-longrun/runs/`。
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

**PH1: Runner / config 機構**(§2)

- `--workspace` / ダイアログ(履歴コンボ+スキャン一覧のワンクリック起動) / 履歴 / `_default` 自動生成 /
  `history.txt`/`prefs.txt` の読み書き(ファイル削除=リセットの挙動含む)が動作
- パス指定の両形式(相対=workspaces/ 基準、絶対=別ドライブ等)で Run 出力・config 後読みが機能
- `ConfigData::ToPropertiesString` の round-trip 単体テスト(Properties で読み戻して一致。
  履歴の空白入りパス・MRU 更新・上限切り捨てを含む)、MetricsLogger 置換後の config_data.txt ダンプ不変
- `GetAppDataDir()` のポータブル判定(appdata/ 有無で切替)の単体テスト
- workspace config の後勝ち merge と `$include` search dir 解決の単体テスト
  (config_test.cpp に追加。2 ファイル目 merge、workspace 側 override、AutoMerge との順序)
- `--config` 時に workspace 解決が走らないこと、`--config`+`--workspace` がエラーになること
- env 未選択 workspace で起動 → 「Failed to create env.」+ヒントで Run が始まらないこと
- 既存 Run(手動で `workspaces/_default/runs/` へ移動)の config_data.txt ダンプが従来と同等であること

**PH2: MetricsViewer**(§3)

- workspaces.json 列挙 / workspace 切替 / 切替後の Run リスト・メトリクス表示
- 異 workspace 同名 Run で切替してもデータが混ざらないこと(世代バンプの検証)
- 既存単体テスト(RunScannerTest 等)と Playwright テストの追随、全テスト緑

**PH3: optuna ハーネス + bat + docs**(§4, §5)

- `--workspace` 導出値で dry-run → run-trial → run-study が新レイアウトに出力されること
- 生成 config に workspace include が入り、seed run が Viewer の optuna workspace に見えること
- 明示引数 override が導出に勝つこと
- bat 2 件(22 削除 / 23 引数化)と docs 更新

## 7. スコープ外(後続 PRD 候補)

- Viewer の任意パス「アタッチ」(退避先 workspace を移動せず開く)
- Runner GUI での実行時 workspace 切替(現状は再起動で切替)
- workspace 単位の UI パースペクティブ保存等、`workspaces/<ws>/settings/` の活用
- タグ・メタデータによる Run 横断検索(方針として不採用、ADR 0021)
- 複数 workspace の同時閲覧 UI(当面は手動の別ポート起動)
