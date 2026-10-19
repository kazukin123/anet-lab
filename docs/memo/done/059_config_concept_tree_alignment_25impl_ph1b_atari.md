# PRD 059 Phase 1 素材化 + Phase 2 幹化 実装メモ: Atari.txt

- 作業日: 2026-08-24
- 対象: `apps/runner/config/Atari.txt` のみ（共通ファイル・他枠ファイルは未変更）
- 正本: `059_config_concept_tree_alignment_30mat_guide.md`（手順） / `..._10prd.md`（設計）
- 見本: `apps/runner/config/LunarLander.txt`

## 結果サマリ

| 項目 | 結果 |
|---|---|
| dump の `@` なし素材 | **48 行 → 0 行** |
| 等価性（値の差分 / 新規キー） | **0 件 / 0 件**（default / qr / classic の 3 変種で確認） |
| 消滅キー | **50 件**（全て意図分。内訳は下記） |
| `[config]` テスト | 全緑（89 test cases / 602 assertions） |
| 全 core テスト | 453 中 451 pass。失敗は既知の `replay_buffer_test.cpp(984)` のみ |
| `inspect_run_test.py` | 53 tests OK |
| `resolve_workspace_test.ps1` | passed |
| `git diff --check` | クリーン |
| smoke（既定 = IQN） | 起動確認 OK。warn/error 0 件 |
| smoke（幹有効化 `run.@nature_dqn`） | 起動確認 OK（PER 検証の修正が前提。§不具合 2 参照） |

## rename 内訳

| 旧 | 新 | 行数 |
|---|---|---|
| `AtariEnv.v5.*` | `AtariEnv.@v5.*` | 4 |
| `AtariEnv.classic.*` | `AtariEnv.@classic.*` | 4 |
| `AtariEnv.100k.*` | `AtariEnv.@100k.*` | 4 |
| `AtariEnv.test1.*` | `AtariEnv.@test1.*` | 1（コメント行） |
| `net.qr.*` | `net.@qr : *` | 2 |
| `net.iqn.*` | `net.@iqn : *` | 7 |
| `net.branch.AtariNature/AtariImpala/AtariImpalaX2/AtariImpalaViT.*` | `net.branch.@<同名>.*` | 8 |
| `NatureDQN.*` | `DefaultDQNAgent.@nature.*` | 13 |
| `B.*` | `DefaultDQNAgent.@random.*` | 6 |

チェーン参照・コメントアウト済みラダー（`#train.eval.[test1].env.$` 等）も追随済み。

## 消滅行（dump から消えた 50 キー）

| prefix | 件数 | 種別 |
|---|---|---|
| `AtariEnv.` | 12 | `@` 化した env プリセット |
| `NatureDQN.` | 13 | `@` 化した再現素材 |
| `net.branch.` | 8 | `@` 化した BODY 素材 |
| `net.iqn.` | 7 | `@` 化した配線素材 |
| `B.` | 6 | `@` 化した Random Policy 素材 |
| `net.qr.` | 2 | `@` 化した配線素材 |
| `A.quantile_mode` | 1 | ALGO 素材が供給するため削除 |
| `R.quantile_mode` | 1 | 同上 |

削除した旧 2 行同期は `DefaultDQNAgent.net.$` ラダー 2 行 / `R.quantile_mode` ラダー 2 行 / 「agent 側と揃えること」コメント 1 行 / `A.quantile_mode` 1 行。

## 新規に入れた機構

### ALGO 素材（059 D12）

```text
DefaultDQNAgent.@qr  : quantile_mode = qr
DefaultDQNAgent.@qr  : net.$ = net.@qr
DefaultDQNAgent.@iqn : quantile_mode = iqn
DefaultDQNAgent.@iqn : net.$ = net.@iqn
```

`@none` は**作らなかった**。`quantile_mode = none` の配線は QR と同一（V/A を main_feature 直結。fusion が要るのは IQN だけ）なので、`DefaultDQNAgent.@nature` 自身に `quantile_mode = none` と `net.$ = net.@qr` を持たせた。素材が 1 つ減り、チェーンに `@qr` を置いて直後に打ち消す誤読も避けられる。

### RUN_BUDGET の一点化（案 A を採用）

```text
app.@vars : max_exp_step  =  50,000,000
app.@vars : half_exp_step =  25,000,000
app.online.exp_pause_step = ${app.@vars.max_exp_step}
A.learner.per_beta_step   = ${app.@vars.half_exp_step}
```

**案 A を採った理由**: この枠の予算変更は 8M / 10M / 50M と毎回任意値で、固定メニュー化（案 B の予算幹）の根拠が無い。痛みの実体は「2 行が 15 行離れた別セクションにある」ことなので、隣接ペア化で解消する。案 B は「固定予算メニューが定着したら」の遅延ゲートに置く。

`app.batchrun.exp_exit_step = 2,500,000` は**リテラルのまま据置**。online（50M）と値が異なり、`${}` で束ねると等価性が壊れるため。bat から予算を変える場合は `exp_exit_step` に加えて `--app.@vars.max_exp_step` / `--app.@vars.half_exp_step` を渡す必要がある旨をコメントに明記した（CLI 第 1 相は素材キーを受け付け、`${}` 展開は leaf override の後なので波及する）。

### TauProj 次元の値参照化

```text
net.block.[AtariIQNTauProj512].linear.out_features = ${net.block.[AtariLinear512].linear.out_features}
```

「main_feature 最終次元と一致必須」というコメントだけの手動同期を解消。両者 512 で同値のため等価性は保たれる。resolution.json の `references` に記録されることを smoke で確認済み。

### named 幹 3 本（全て既定 OFF）

`run.@v5_iqn_impala_x2`（BTR 比較の主軸） / `run.@classic_iqn_impala_x2`（デモ・長時間） / `run.@nature_dqn`（文献再現）。

## 等価性検証

一時テスト（`config_test.cpp` へ `[tempdump]` タグ、**検証後に削除済み**）で `ConfigManager::GetConfigData().ToPropertiesString()` を before/after で取得しキー単位比較。ConfigManager は app と同じ構成（base `_main.txt` + workspace `_main.txt` を `overwrite_config_paths`）で構築した。

| 変種 | 値の差分 | 新規キー | 消滅キー |
|---|---|---|---|
| default（IQN / v5 / ImpalaX2） | 0 | 0 | 50 |
| qr（ALGO を @qr へ） | 0 | 0 | 50 |
| classic（プロトコル切替） | 0 | 0 | 50 |

幹の等価性:

| 比較 | 結果 |
|---|---|
| `run.@nature_dqn` vs 同じ選択を手で並べた overlay | **完全一致** |
| `run.@v5_iqn_impala_x2` vs 既定（root ラダー） | **完全一致** |
| `run.@classic_iqn_impala_x2` vs classic のみ切替 | 差分は意図した eval 設定のみ（ε 0.05、上限 18,000 ×2） |

## 不具合 1: NatureDQN 旧手順の `quantile_mode` 握り潰し（本改修で解消）

旧運用のコメントに従って `DefaultDQNAgent.$ = @baseline > A > @bf16 > NatureDQN > R > X` を有効化すると、dump の `DefaultDQNAgent.quantile_mode` が **`none` ではなく `iqn`** になっていた（2026-08-24 に before dump で実測）。チェーン上 `R` が `NatureDQN` より右にあり、root の `R.quantile_mode = iqn` が握り潰していたため。他のキー（`n_step=1` / `use_per=false` / `use_double_dqn=false` / `use_dueling_net=false` / `gamma=0.99`）は正しく効いていた。

ALGO 素材化で `R.quantile_mode` を撤去したため、この不具合は副作用で解消した。`run.@nature_dqn` の dump で `quantile_mode = none` を確認済み。

## 不具合 2: PER 無効構成が起動できない（本作業で修正）

`run.@nature_dqn` を有効化して起動すると、Run フォルダを作らずに終了していた。原因は `ValidateReplayPriorityConfig`（`dqn_based_agent.cpp`）の下記チェック。

```
learner.per_initial_priority_mode=max requires learner.use_per=true; set the mode to fixed or enable PER.
```

`@nature` は `use_per = false` だが、`per_initial_priority_mode` は baseline の `max` のまま。**PER を切るだけの構成が、無関係な設定値の書き換えを強要されていた。**

### 裁定と修正

`per_initial_priority_mode` は **PER 有効化に備えた設定値**であり、PER 無効時にどの正常値が入っていても**何の問題も無い正常ケース**である。したがって:

- **不正値（`fixed` / `max` / `actor_approx` 以外）はシステムエラー** — `ParseReplayInitialPriorityMode` が従来どおり担う
- **正常値 + PER 無効はエラーにも WARN にもしない** — チェックそのものを削除

あわせて、`use_per` を見ずにモードを読んでいた 3 箇所を揃え、PER 無効時の mode を**完全に不活性**にした（従来は `actor_approx` だと PER 無効でも未使用の estimator を生成し、Actor が未使用の Q ヒントを出していた）。

| ファイル | 変更 |
|---|---|
| `dqn_based_agent.cpp` `ValidateReplayPriorityConfig` | `!use_per && mode != FIXED` のエラーを削除 |
| 同上 | `actor_approx` + `per_alpha=0` の WARN を `use_per &&` で限定 |
| 同上 `SetupReplayBuffer` | estimator 生成を `use_per &&` で限定 |
| `default_dqn_agent.cpp` `CreateActor` | `emit_actor_q_hint` を `use_per &&` で限定 |
| `rainbow_agent.cpp` `CreateActor` | 同上 |
| `dqn_based_agent_test.cpp` | 該当 `CHECK_THROWS` を `CHECK_NOTHROW` へ |

`[per]` テスト全緑（57 test cases / 492 assertions）。全 core テストは 453 中 451 pass（既知の ReplayBuffer 2 件のみ失敗）。

なお本修正は `Atari.txt` 素材化とは独立した core の変更であり、コミットを分けた方がよい。

### 幹の動作確認

修正後、`run.$ = run.@nature_dqn` を有効化した Release ビルドが**正常に起動し、classic / `quantile_mode = none` / `AtariNature` で走り出すことを確認した**。幹の機構自体に問題は無い。

なお作業中に「`run.@v5_iqn_impala_x2` でも起動しない」と観測したが、これは**別の幹（`run.@nature_dqn`）が同時に有効なままで、後勝ちでそちらが選ばれ PER エラーに落ちていた**ものと考えられる（`run.$` の重複有効化は最後の行が勝つ）。修正後は再現しない。

### 残: Release ビルド

最終版（WARN も削除した状態）での Release 再ビルドは未実施。50M Run が実行中で exe がロックされるため見送った。実行中の構成は `use_per = true` のため、警告版と最終版で**挙動差は無い**。Run 終了後に再ビルドすること。

## 段階 2（ViT-lite）は未着手

段階 1 の等価性証明を汚さないため分離した。`net.block.[AtariViTProj96]` / `[AtariImpalaTransLite]` はカタログなので dump に残り、新規キーが発生する。段階 1 は完了したのでいつでも着手できる。

## 複雑性監査（grill 2026-08-24 の裁定）

| 機構 | 裁定 | 理由 |
|---|---|---|
| env プリセット `@` 化 | keep | dump 汚染 12 行 |
| ALGO 素材 + 配線 `@` 化 | keep | ガイド必須。`A.quantile_mode` の削除は必須（残すと IQN 選択が壊れる） |
| BODY `@` 化 | keep | dump 汚染 8 行 |
| `B` の `@` 化 | keep | dump 汚染 6 行 |
| `DefaultDQNAgent.@none` | **shrink（作らない）** | 痛みは可読性のみ。`@nature` に配線を持たせて解消 |
| `app.@vars` 2 定数 | keep | 予算 × per_beta の同期忘れが 2026-08-22 / 08-23 に実発生 |
| 幹 3 本 | keep | 3 箇所・4 箇所の同時編集契約が実在 |
| ViT-lite + 幹 1 本 | keep（段階 2 の門の後ろ） | 痛みは未発生（実験対象） |
| 予算幹（案 B） | **defer** | 固定予算メニューが定着した時点で再検討 |

## 上位（横断整合チェック）への申し送り

### 1. `B` → `DefaultDQNAgent.@random` の波及

`B.*` は `Atari.txt` / `DropMerge.txt` / `GridMaze.txt` の 3 ファイルに**同名で各 6 行**存在する。互いに参照は無く（DropMerge / GridMaze はチェーン参照 0 件）ファイルローカル定義なので、Atari 単独改名で機能的破綻は無いが、**命名が 3 ファイルで不揃いになった**。

判断が要る点: `B` は「上書き層は無印のまま維持」（D20）の対象リスト（A/E/R/X/M/O/P）に入っていないが、単文字スコープという見た目は上書き層と同じである。「単文字だが実体はプリセット」という中間物をどう扱うかの規約が要る。本枠は D5（チェーンから選ばれる=素材）を優先して `@` 化した。

### 2. BODY 素材の命名規約

`net.branch.@<Name>`（名前を保存）で `@` 化した。`net.branch.[main_feature]` などの `[...]` 形はインスタンス（カタログ）なので無印のまま。

**LunarLander には BODY 軸が存在しない**（`structure` を直書きしてコメントラダーで切替）ため見本が無く、本枠が事実上の先例になっている。DropMerge も `net.branch.Suika*` で同じ構造を持つので、全 env 共通の規約として追認するか判断が要る。

なお `HasMaterialSegment`（`config_impl.cpp:87-95`）は全セグメントを走査するため、`net.branch.@X.bind` のような深さ 3 の `@` も正しく素材と判定される（確認済み）。

### 3. ALGO 素材・配線素材の共通化

`DefaultDQNAgent.@qr / @iqn` と `net.@qr / net.@iqn` を Atari.txt が所有している（ガイドの「env ファイル所有・重複は現段階では正」に従った）。配線実体（`AtariIQNTauProj512` の次元等）は env 依存だが、**ALGO 素材の 4 行自体は LunarLander と完全に同一**なので、agent.txt への共通化余地がある。

### 4. run プロファイル（named 幹）の選定基準

本枠で採った基準は「**定着 + 実験対象 + 今後の比較・デモで使う可能性が高い組み合わせ**」で、フル Run 署名（`<protocol>_<algo>_<body>`）を名前にした。

- **採った 3 本**: `v5_iqn_impala_x2`（定着＝現行の実動作）、`classic_iqn_impala_x2`（デモ・長時間）、`nature_dqn`（文献再現）
- **段階 2 で 1 本**: `classic_iqn_vitlite`（実験対象）
- **作らなかったもの**: protocol 2 × ALGO 2 × BODY 4 の残り。D4（素材直積の事前定義禁止）

判断が要る点: **フル署名の命名は名前が座標になるため、埋めたくなる圧力がかかる**。幹のネストは fail-fast なので共通部を括り出せず、`v5_*` と `classic_*` で 2 行が丸ごと重複する。16 通り全部を書くと 100 行超になる。全 env 横断で「幹を増やす規律」を明文化するか、`run.$` がチェーンであること（PH2 実装記録で確認済み。`run.$ = run.@a > run.@b` が合成可能）を使って軸ごとに分ける方針にするかは、横断で決めた方がよい。

### 5. `AtariIQNTauProj512` の名前

次元を値参照化したため、参照先（`AtariLinear512`）が変わると**名前の `512` が実態とずれる**。カタログ名なので structure 文字列からも引かれており、改名は影響範囲が広い。

### 6. bat の予算駆動

`apps/12_batch_run_atari5.bat` は `app.batchrun.exp_exit_step` のみを CLI で渡しており、`per_beta_step` が追随しない（2026-08-22 に発見した地雷）。`@vars` 化により `--app.@vars.max_exp_step` / `--app.@vars.half_exp_step` を渡せば波及するようになったが、**bat 自体は本枠で未変更**（rename 対象キーに触れていないため追随不要だった）。bat の更新は別途。

### 7. CLI 第 1 相の判定が「源プレフィクス + leaf」を取りこぼす

**症状**: `--app.online.exp_exit_step=200000` を CLI で渡しても効かない（Run が終わらない）。dump には

```
972:app.online.exp_exit_step = 200000     ← CLI で書かれたが誰も読まない
（app.exp_exit_step は存在しない）
865:app.exp_pause_step = 50,000,000       ← こちらは選択で降りてきている
```

**原因**: §5.3 の第 1 相の判定が「`.$` または `@` を含むキー」。`app.<源>.<leaf>` 形は**どちらも含まないので第 2 相**へ回り、選択の**後**にそのままのキー名で置かれる。`app.online.*` は選択元でしかないため、書かれた値を読む者がいない。エラーにも WARN にもならず、**静かに無効化される**。

**回避**: 実効 leaf キーで渡せば効く。`--app.exp_exit_step=50000` で正常終了を実測（`run_20260824-080048_smoke_exitkey`、warn 0 件）。

**波及**: `apps/12_batch_run_atari5.bat:61-63` の

```bat
SET BUDGET=app.batchrun.exp_exit_step=50000000
```

が同じ形なので、**現状は無効化されているはず**（同 bat の `app.$=app.batchrun`（58 行目）は `.$` を含むので第 1 相で正しく効く）。申し送り 6 の「`per_beta_step` が追随しない」より上位の問題で、予算指定そのものが通っていない。

**旧挙動との関係（推定）**: 2 相化の前は「AutoMerge 前後に同一 override を 2 回適用」だったため、AutoMerge 前の適用で源キーに書かれ、merge で `app.*` へ複写されて効いていたはず。つまりこれは **2 相化に伴う挙動変化**であり、Atari 固有ではなく `<源プレフィクス>.<leaf>` を CLI で渡す全ての呼び出し側に効く。

**判断が要る点**（横断で決めるべき、本枠では未変更）:

| 案 | 内容 | コスト / 副作用 |
|---|---|---|
| a | 呼び出し側を実効 leaf キーへ直す（`app.exp_exit_step=...`） | 最小。ただし「どの源を上書きしているか」が CLI から読めなくなる |
| b | 第 1 相の判定を拡張し、選択元プレフィクス（`app.$` が選ぶ `app.batchrun.*` 等）も第 1 相で受ける | 表現力は保つ。判定に選択の事前解決が要る |
| c | 第 2 相で「誰も読まないキー」を作る書き込みを検出して fail-fast / WARN | 静かな無効化を潰す最強の網。ただしコード側 default しか持たないキーの上書きは正当なので、単純な「未定義キー禁止」は不可。未読キー検出機構は現状 **無い**（`config_impl.cpp` / `config.hpp` に該当実装なし） |

なお本枠の `@vars`（`app.@vars.max_exp_step`）は `@` を含むため第 1 相で正しく効く（今回の Run で `${}` 経由の波及を実測済み）。bat を案 a で直す場合も `@vars` 経由（`--app.@vars.max_exp_step=...`）にすれば `per_beta_step` の追随（申し送り 6）と同時に解決できる。
