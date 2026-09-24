# PRD 052: eval 定義とスケジュールの namespace 分離

- 起票日: 2026-08-14
- 状態: implementation ready
- 対象: `core/anet-core` RunManager（trainer.cpp）、`apps/runner/config` 全 config
- 関連: ADR 0027（本 PRD で新設）、CONTEXT.md 用語改訂（configured eval tag / dormant / eval schedule）
- 設計文書: `docs/design/100_runtime_and_configuration.jp.md`、`docs/design/010_framework_overview.jp.md`

## Context（背景・目的）

`train.eval.[tag].*` はキーが 1 つでも存在するとタグが実 eval として列挙され、`interval` 省略時は既定 100 で背景 EvalRunner が起動する。このため、EvalPanel 用の設定キャリアとしてだけ `[eval_panel]` タグを書いていた LunarLander.txt（`env.limit_step` と `eval_config_tag` のみ、interval 無し）で、**全 Run に意図しない背景 eval が寄生**していた。Atari では同構造が SDL 音声デバイスの取り合い（先に生成された背景 eval 側がデバイスを握り EvalPanel が無音）として顕在化し発見された。

根本原因は schema にある。`interval` / `use_background` の消費者は EpisodeEvalObserver（スケジューラ）ただ一つであり、Env / Runner の定義とは無関係なのに、定義ブロック `train.eval.[tag]` に間借りしている。このコードベースの他の消費者は既に全員「自分の namespace で宣言し、タグ / runner を名前で参照」する慣例に従っている（metrics の `$eval.[name]`、EvalPanel の `app.online.eval_panel.eval_config_tag`）。EpisodeEvalObserver だけが例外で、これが「宣言＝起動」もつれの正体である。

本 PRD は定義（`train.eval.[tag]`）と定期駆動（`train.eval_schedule.[tag]`）を別 namespace に分離し、Env + Runner + Observer の生成を schedule エントリ駆動へ変える。dormant は「interval=0 で宣言する状態」から「定義済みだが未スケジュール」の導出状態になる。挙動等価な移行であり、全 config で同じ eval が同じ周期・同じ seed 系で走る。唯一の挙動差は LunarLander の寄生 eval が消えることで、これは修正そのものである。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | スケジューラの namespace は `train.eval_schedule.[tag].*`。定義 `train.eval.[tag].*` と兄弟階層 |
| D2 | schedule のキーはタグ名そのもの。同一タグへの複数 schedule は config キー一意性により構造的に不可能で、禁止コードは書かない。common.txt → 環境別 config の上書きは既存のオーバーレイ意味論のまま |
| D3 | `interval` は**必須キー（既定なし）**。schedule エントリがあるのに `interval` 未記載は `ANET_SYSTEM_ERROR`。発端バグの原因＝暗黙の既定 100 を新 namespace に持ち込まない。全現行 config は interval 明示済みで影響ゼロ |
| D4 | `interval = 0` は**明示 OFF** として許可する。挙動はエントリ無しと同一（Env / Runner / Observer 非生成、dormant 扱い）。common 層の schedule を環境別 config で打ち消すオーバーライド（ImageCls eval2、GridMaze_muzero eval1 で現用）と、1 行トグル運用（LunarLander test1/test2 のコメント候補群）の両方に必要。`interval < 0` はエラー（既存踏襲） |
| D5 | `use_background` は schedule 側。既定 true（既存踏襲） |
| D6 | `run_mode` / `eval_batch_size` / `clone_model` / `env.*` は定義側に残す。消費者はそれぞれ BatchEnvBuilder（定義の実体化）と EvalRunner であり、「いつ走らせるか」の語彙ではない。EvalPanel も run_mode / env overlay を鏡写し参照するため定義側にあるべき |
| D7 | schedule が未定義タグ（`train.eval` に無いタグ）を参照したら `ANET_SYSTEM_ERROR` |
| D8 | dormant は宣言物から**導出状態**へ: 定義済み ∧（schedule エントリ無し ∨ interval=0）。魔法値 0 での宣言は不要化。dormant タグの処遇（name 予約 + `ValidateConfig` のみ、生成なし、metrics 参照は WARN 1 回 + skip）は現行と同一 |
| D9 | metrics が未スケジュールタグを `$eval.[name]` 参照したときの WARN 文言は活性化方法を案内する（§2.5）。WARN 1 回 + skip の振舞い自体は維持 |
| D10 | 起動時にタグごとのロールを 1 行ログ出力する（scheduled / definition-only。§2.6）。取り残し設定や書き忘れの可視化手段を兼ねる |
| D11 | 旧位置キー（`train.eval.[tag].interval` / `use_background`）の検出は **NoCare**。クリーンブレーク方針（AGENTS.md）に従い、旧キー専用の検出コード・WARN・tripwire は追加しない。リポジトリ管理下の config は本変更内で全移行する。取り残しは D10 のロールログで definition-only として可視化される |
| D12 | dead key 削除: `RunManager::Config::eval_interval`（trainer.cpp:708,719。読むだけで未使用）と、その唯一の現用記述 `CartPole.txt:88` の `train.eval_interval = 10` |
| D13 | Atari の `[eval_panel]` は interval 行削除で唯一の宣言キーが消えるため、`train.eval.[eval_panel].run_mode = eval1` を明示して宣言キーを残す（既定値と同じで挙動不変。`Atari.txt:502` の `eval_config_tag = eval_panel` が参照中） |
| D14 | seed domain は `"eval_env/" + tag` のまま維持（挙動等価の要） |
| D15 | 成果物は本 PRD、ADR 0027、CONTEXT.md 用語 3 件（本 PRD 作成時に実施済み）。実装時に `docs/design/` の関連記述を同一変更内で更新（§3.2） |

## 1. 現状の事実（コード確認済み）

2026-08-14 時点、branch `main`（HEAD 23e66f3 + 未コミット差分あり）で実測済み。

| 事実 | 根拠 |
|---|---|
| `MakeSubConfigData("train.eval")` でタグ列挙。**キー存在＝タグ宣言**。name 予約検証（"train" / "EvalPanel" 予約含む） | `trainer.cpp:740`, `:755-760` |
| `interval` 既定 **100**（`int interval = 100;` → `Read`）。負値はエラー | `trainer.cpp:846-851` |
| `interval == 0` → dormant: `dormant_eval_tags_` 登録 + `RegisterEnvName` + `ValidateConfig` のみ。Env / Runner / Observer 非生成 | `trainer.cpp:868-874` |
| 非 dormant: seed domain `"eval_env/"+tag` → `CreateBatchEnv(tag, eval_batch_size, run_mode)` → `EvalRunner(clone_model, tag)` → `AttachScoped<EpisodeEvalObserver>(train_runner_, eval_runner, interval, use_background)` | `trainer.cpp:876-896` |
| `interval` の消費者は EpisodeEvalObserver ただ一つ。LEARN 軸 step で `step % interval == 0` 判定、`use_background` は専用 1 スレッドプール + 前回未完なら `WaitBackgroundEval()` | `observers.cpp:487-573` |
| resolve_runner: dormant タグ参照は WARN 1 回 + skip（文言に旧語彙 `interval=0` が埋まっている）、未宣言タグ参照はエラー | `trainer.cpp:899-913` |
| dead key: `RunManager::Config::eval_interval = 50` は読むだけで未使用。現用記述は CartPole.txt:88 の 1 箇所のみ | `trainer.cpp:708,719`、`CartPole.txt:88` |
| キー→消費者の対応: `env.*` / `run_mode` / `eval_batch_size` → BatchEnvBuilder、`clone_model` → EvalRunner、`interval` / `use_background` → EpisodeEvalObserver | `trainer.cpp:840-896` |
| EvalPanel の mirror EvalRunner はパネル側が自前生成（RunManager 非関与）。本変更の影響なし | `trainer.cpp:991` 付近 |
| 意図的に走らせるタグは全 config で interval 明示、意図的 dormant も全て明示 0。**既定 100 に落ちているのは LunarLander `[eval_panel]` ただ一つ＝事故そのもの。既定値の意図的利用者はゼロ** | config 全数 grep（§4 の表） |
| ImageCls `train.eval.[eval2].interval = 0` は定義本体なしの単独行＝ common.txt の eval2（interval=100）を環境別 config で打ち消すオーバーライド。GridMaze_muzero eval1 も同型 | `ImageCls.txt:637`, `GridMaze_muzero.txt:247` |
| Atari `[eval_panel]` の非コメント宣言キーは `interval = 0` ただ一つ（env.* は全てコメントアウト）。`Atari.txt:502` の `eval_config_tag = eval_panel` が参照中 | `Atari.txt:512-519`, `:502` |

## 2. 契約

### 2.1 namespace 構造

```ini
# 定義（純粋。書いただけでは何もインスタンス化しない）
train.eval.[eval1].run_mode = eval1
train.eval.[eval1].clone_model = true
train.eval.[eval1].eval_batch_size = 1
train.eval.[eval1].env.…                      # Env overlay

# 定期駆動 = スケジューラが名前で参照
train.eval_schedule.[eval1].interval = 100    # 必須。0 = 明示 OFF
train.eval_schedule.[eval1].use_background = true
```

- **定義** `train.eval.[tag]`: キー存在＝タグ宣言（現行踏襲）。name 予約（"train" / "EvalPanel" / タグ間重複）と `ValidateConfig`（宣言時 schema 検証）は全定義タグに対して維持。許容キーは `run_mode` / `eval_batch_size` / `clone_model` / `env.*`（消費者は §1 の対応表どおり）。
- **スケジュール** `train.eval_schedule.[tag]`: キーは `interval`（必須、int、`>= 0`。`0` = 明示 OFF、負値エラー）と `use_background`（省略可、既定 true）。消費者は EpisodeEvalObserver のみ。
- Env + Runner + Observer の生成は**有効な schedule エントリ（interval > 0）が駆動**する。生成手順・引数・seed domain（`"eval_env/" + tag`）・`AttachScoped<EpisodeEvalObserver>` の interval / use_background 受け渡しは現行と同一。EpisodeEvalObserver の実行機構（LEARN 軸判定、背景プール、`WaitBackgroundEval`）は変更しない。

### 2.2 検証（fail-fast）

| 状態 | 挙動 |
|---|---|
| schedule が `train.eval` に無いタグを参照 | `ANET_SYSTEM_ERROR`。タグ名と「train.eval.[tag] に定義が必要」の旨を含める |
| schedule エントリに `interval` キーが無い | `ANET_SYSTEM_ERROR`。「schedule エントリには interval が必須（0 = 明示 OFF）」の旨を含める |
| `interval < 0` | `ANET_SYSTEM_ERROR`（現行踏襲） |
| `eval_batch_size <= 0`（定義側） | `ANET_SYSTEM_ERROR`（現行踏襲） |
| 旧位置キー `train.eval.[tag].interval` / `use_background` の残存 | **NoCare**（D11。検出コード無し） |

### 2.3 dormant（導出状態）

定義済みタグのうち、schedule エントリが無いか `interval = 0` のものが dormant。処遇は現行 dormant と同一:

- name 予約（`RegisterEnvName`）+ `ValidateConfig` は行う。Env / Runner / Observer / background worker は生成しない。
- 内部の dormant 集合（現 `dormant_eval_tags_`）は「定義済み ∧ 有効 schedule 無し」で構築する。

### 2.4 config dump / metrics への記録

定義タグごとの設定 JSON 記録（`MetricsLogger::Log(config_prefix, ...)`）は現行どおり全定義タグに対して行う。schedule は `ConfigData` の一部として通常の config dump（`config/config_data.txt`）に自然に含まれるため、追加の記録機構は作らない。

### 2.5 metrics 参照解決（resolve_runner）

- 未定義タグの `$eval.[name]` 参照: エラー（現行踏襲）。
- dormant（定義済み・未スケジュール）タグの参照: タグごと 1 回 WARN + skip（現行踏襲）。文言は旧語彙 `interval=0` を排し、状態と活性化手段を案内する:

```text
Skipping metrics for unscheduled eval tag. tag='<tag>'. The tag is defined in train.eval but has no active train.eval_schedule entry.
```

### 2.6 起動時ロールログ

RunManager 構築時、定義タグごとに 1 行、ロールを `LOG::info()` で出力する:

```text
eval tag 'eval1': scheduled (interval=100, background=true)
eval tag 'eval_panel': definition-only
```

意図しない背景 eval の寄生（本 PRD の発端）と、schedule 書き忘れ・取り残し設定（D11 の NoCare を選んだ代償）の両方を起動ログで即座に可視化する。

## 3. 実装範囲

### 3.1 コード変更

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/src/trainer.cpp` | ①タグ列挙（:740）は定義 ns のまま維持。②`MakeSubConfigData("train.eval_schedule")` で schedule を列挙し、未定義タグ参照を fail-fast（D7）。③`interval` / `use_background` の Read（:846-851, :860-862）を schedule 側へ移し、`interval` を必須化（D3。既定値付き Read をやめる）。④生成ループを「定義列挙（name 予約 + `ValidateConfig` + 設定 JSON 記録）→ 有効 schedule のみ Env/Runner/Observer 生成」に再編。dormant 集合は導出（§2.3）。⑤resolve_runner の WARN 文言差し替え（§2.5）。⑥ロールログ追加（§2.6）。⑦`Config::eval_interval` 削除（D12） |

EpisodeEvalObserver（observers.cpp / observers.hpp）は変更しない。コンストラクタ引数 `eval_interval` の意味・消費は不変。

### 3.2 文書（実装と同一変更内で更新）

| ファイル | 変更内容 |
|---|---|
| `docs/design/100_runtime_and_configuration.jp.md` | :152 の「`interval=0` は dormant 宣言」を導出 dormant + eval_schedule 契約へ書き換え。:267 のキー表を定義/スケジュールの 2 行に分割（`train.eval.[tag].*` = RunMode / `eval_batch_size` / Env override / model clone、`train.eval_schedule.[tag].*` = interval / use_background） |
| `docs/design/010_framework_overview.jp.md` | :294 の「設定された評価は…interval に達したとき起動」を schedule 駆動の記述へ更新 |
| `CONTEXT.md` | 用語 3 件（本 PRD 作成時に実施済み: configured eval tag 改訂 / dormant 改訂 / eval schedule 新設） |
| `docs/adr/0027-eval-definition-schedule-separation.md` | 本 PRD 作成時に新設済み |

## 4. config 移行インベントリ（挙動等価、機械的移動）

行番号は 2026-08-14 時点。コメントアウトされた候補行（トグルイディオム）も同じ位置関係で新 namespace へ移す。

| ファイル | 変更 |
|---|---|
| `common.txt:6,8-9,14,16-17` | eval1 / eval2 の `interval = 100`・`use_background = true`（+ コメント候補 `use_background = false`）→ `train.eval_schedule.[eval1/eval2].*` へ移動。定義側（run_mode / clone_model / eval_batch_size）は残す |
| `DropMerge.txt:1583-1584` | `interval = 100` ×2 → eval_schedule |
| `GridMaze.txt:244-245` | `interval = 10` ×2 → eval_schedule |
| `GridMaze_muzero.txt:247,249-252` | eval1 `= 0`（common 打ち消し）・eval2 `= 5` + コメント候補群 → eval_schedule。:255-257 のコメントブロック内の `train.eval.[test1].interval` 表記も新 ns へ追随 |
| `ImageCls.txt:624,632,637` | eval1 `= 50`・eval_full `= 0`・eval2 `= 0`（common 打ち消し）→ eval_schedule。eval_full / eval2 の明示 0 は維持（eval2 は common 打ち消しに必須、eval_full はトグル温存） |
| `LunarLander.txt:269-271,275-277` | test1 / test2 の `= 0` + コメント候補群 → eval_schedule。**`[eval_panel]` ブロック（:96-97）は無変更**——本設計により定義だけの状態が正しい形になり、寄生 eval が消える |
| `Atari.txt:458-459` | eval1 / eval2 `= 100` → eval_schedule |
| `Atari.txt:512-519` | `interval = 0` 行（:517）を削除し、宣言キーとして `train.eval.[eval_panel].run_mode = eval1` を置く（D13）。:512-516 のコメント（「interval = 0 は必須」の旧契約説明）を新契約の説明（定義だけなら何も走らない）へ書き換え。SDL 音声・WASAPI 関連のコメント（:505-510）は有効なまま残す |
| `CartPole.txt:88` | dead key `train.eval_interval = 10` を削除（D12） |

移行後、リポジトリ管理下に `train.eval.[tag].interval` / `use_background` / `train.eval_interval` の非コメント記述が残らないこと（grep で確認）。

補足: 既発行の暫定止血タスク「LunarLander に interval=0 を 1 行追加」は本設計の実装で不要化される。

## 5. 受け入れ基準

### 5.1 挙動等価（移行の正しさ）

- 全リポジトリ config で、移行前に走っていたタグが同じ interval・同じ `use_background`・同じ seed 系（`eval_env/` + tag）で走る。Run の `config/config_data.txt` に新 namespace が記録される。
- LunarLander で背景 eval「eval_panel」が生成されない（発端バグの解消）。起動ロールログで `eval tag 'eval_panel': definition-only` が出る。
- Atari で EvalPanel の `eval_config_tag = eval_panel` 参照が従来どおり解決される（D13 の宣言キー維持）。

### 5.2 fail-fast

- schedule が未定義タグを参照 → `ANET_SYSTEM_ERROR`（タグ名を含む）。
- schedule エントリに `interval` 無し → `ANET_SYSTEM_ERROR`。
- `interval < 0` → `ANET_SYSTEM_ERROR`（非退行）。

### 5.3 dormant / metrics

- 定義のみ（schedule 無し）のタグ: name 予約 + `ValidateConfig` が行われ、Env / Runner / Observer 非生成。
- `train.eval_schedule.[tag].interval = 0`: 上記と同一挙動。
- metrics が dormant タグを参照 → WARN 1 回（新文言）+ skip。未宣言タグ参照 → エラー（非退行）。

### 5.4 テスト

- 既存テスト全緑（`anet-core-test.exe`。ビルドは AGENTS.md 記載の `VsDevCmd.bat` 経由）。
- 追加テスト（RunManager 構築レベルで検証可能な範囲）:
  1. 定義のみのタグで Env / Runner / Observer が生成されないこと（旧契約では既定 100 で生成されていた状態の RED 化に相当）
  2. schedule の未定義タグ参照で構築が失敗すること
  3. `interval` 欠落で構築が失敗すること
  4. `interval = 0` がエントリ無しと同一挙動であること
  5. 有効 schedule で従来どおり EvalRunner / EpisodeEvalObserver が interval / use_background 付きで生成されること

## 6. スコープ外

- EpisodeEvalObserver の実行機構（LEARN 軸判定、背景スレッドプール、`WaitBackgroundEval` の同期挙動）。eval1 / eval2 同位相による Train 定期停止の問題は既診断・別件。
- EvalPanel の mirror EvalRunner 生成経路（パネル側が自前生成、RunManager 非関与）。
- eval 定義セットの汎化（curriculum / mixture 等の将来展開）。
- 旧キー検出・移行案内の仕組み（D11 で NoCare と裁定）。

## 7. Further Notes

- 引継ぎ時の案「旧位置キーはエラー + 移行案内」は、AGENTS.md クリーンブレーク方針の「旧キー専用の検出コード・tripwire を残さない」と矛盾するため NoCare へ裁定した（D11）。閉スキーマ化（未知キー一律 fail-fast）による一般化も検討したが不採用。取り残しの可視化はロールログ（D10）が担う。
- `interval = 0` の明示 OFF を残した決め手は、トグルイディオム温存に加え、**common 層の schedule を環境別 config で打ち消すにはエントリ削除では表現できない**こと（ImageCls eval2 / GridMaze_muzero eval1 が現用）。config のオーバーレイ構造がある限り「無効化を値で表現する」手段は必須。
- `interval` を必須化した（D3）ため、AGENTS.md Fail-Fast 原則の表の「意図された休止（`interval=0` 等）」の実例は本契約では「schedule エントリの interval=0」が該当する。エントリを書いて interval を書き忘れた状態は「意図された休止」ではなく書き忘れであり fail-fast 対象。
