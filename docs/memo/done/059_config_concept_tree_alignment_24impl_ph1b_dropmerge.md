# PRD 059 Phase 1b DropMerge 素材化・主要 Run 幹 実装メモ

## 概要

`apps/runner/config/DropMerge.txt` を PRD 059 の素材体系へ移行し、既存の実効設定を維持したまま、定着済み構成・比較対照・次期実験・再現検証を表す named 幹を追加する。

共通設定、batch launcher、Run artifact、実験記録は変更しない。実装前後の golden dump、named 幹の個別指定との一致、Config テストと smoke により設定契約を検証する。

## 主な変更

- `DropMergeEnv` の named 設定、QR/IQN 配線、DropMerge 所有 BODY、Random Policy を `@` 素材へ移行する。
- `DefaultDQNAgent.@qr / @iqn` に `quantile_mode` と NN 配線を束ね、Agent チェーン 1 項で切り替える。
- `A.quantile_mode`、`R.quantile_mode`、旧 `DefaultDQNAgent.net.$` 同期ラダーを削除する。
- 次の named 幹を既定 OFF で追加する。
  - `run.@iqn32_stratified`
  - `run.@qr51_control`
  - `run.@iqn32_antithetic`
  - `run.@iqn32_actor_approx`
  - `run.@iqn32_repro`
- `train.seed`、Run 予算、`app.run_name` は幹へ含めない。online 50M と batch 100M は意図的に異なるため一点化しない。
- 重複した `app.run_name` は現在の最終実効値を維持して 1 箇所へ整理し、未定義素材を参照する古いコメント例を削除する。

## テスト

- Public interface / surface: `ConfigManager::GetConfigData()`、`GetResolutionJson()`、Runner の QR/IQN 起動経路。
- 優先 behavior:
  1. 既定 IQN、QR51、Random Policy の素材化前後で実効値が不変である。
  2. 各 named 幹が、同じ選択を個別指定した構成と一致する。
  3. `config_data.txt` 相当の dump から未選択素材が消え、resolution には幹・ALGO・Env・BODY・配線が残る。
  4. QR/IQN が終了 step 1 相当で NN 構築まで到達する。
- TDD 順序: 一時 `[tempdump]` テストで before dump を取得し、素材群ごとに最小変更と focused 検証を行う。after 比較と幹検証後、一時テストを削除する。

## 検証

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "[config]"
core\anet-core\bin\Debug\anet-core-test.exe
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
git diff --check
```

QR/IQN smoke は Release Runner を `--workspace dm-iqn`、終了 step 1 相当の CLI override で起動し、NN ヘッダログ到達と warn/error の有無を確認する。

## 前提

- BODY 素材と Random Policy は添付依頼の列挙外だが、PRD 059 Phase 1 の「選択肢として使う無印素材を残さない」完了条件に含める。
- `net.block.*`、`DefaultDQNAgent.net.branch.[*]`、上書き層 A/E/R/X/M/O/P は素材化しない。
- tau 3 箇所を束ねる独立素材と部分幹合成は、横断規約が必要なため今回導入しない。
- 既存の未コミット変更を保持し、stage、commit、push は行わない。

## 実装結果

### 素材化と幹

- `DropMergeEnv` は 10 素材、DropMerge 所有 BODY は `net.body` 19 素材と `net.branch` 12 素材を `@` 化した。
- ALGO は `DefaultDQNAgent.@qr / @iqn`、Random Policy は `DefaultDQNAgent.@random` に集約した。
- 5 本の named 幹を追加し、`run.$` は未選択のまま維持した。既定 IQN は `run.@iqn32_stratified` と完全一致する。
- `app.run_name` は、従来最後勝ちしていた `run_{t}_dm_iqn-k32-n32-m32` を維持して有効行を 1 箇所へ整理した。
- online の 50M と batch の 100M、`train.seed`、`app.run_name` は幹へ含めていない。
- 未定義の `DropMergeEnv.G20`、`BF16.env`、`DropMergeEnv.standard > DropMergeEnv.test1 > E1` を参照するコメント例を削除した。

### Golden dump

素材化前の既定 IQN、QR51、Random Policy を一時 `[tempdump]` テストで保存し、素材化後と key/value map で比較した。一時テストは検証後に削除した。

| 構成 | 値変更 | 新規実効キー | 消滅キー |
|---|---:|---:|---:|
| 既定 IQN | 0 | 0 | 100 |
| QR51 | 0 | 0 | 100 |
| Random Policy | 0 | 0 | 106 |

既定 IQN / QR51 の 100 キーの内訳は、`DropMergeEnv` 素材 41、BODY 素材 43、旧 `net.iqn / net.qr` 素材 8、旧 Random 素材 `B.*` 6、旧 quantile 同期行 2 である。すべて未選択素材または削除対象の同期行で、選択済み leaf の値変更はない。

Random Policy の追加 6 キーは、旧グローバル `DefaultDQNAgent.net.$=net.iqn` が Random チェーンにも流入させていた `tau_embedding / iqn_fusion / value_stream / adv_stream` の IQN 配線である。ALGO 選択を Agent チェーン内へ移した結果、Random Policy では選択されなくなった。値が別値へ変化したキーや新規実効キーはない。

5 本の幹は、同じ R/X overlay または leaf を個別指定した dump と文字列単位で完全一致した。

- `run.@iqn32_stratified` = 既定構成
- `run.@qr51_control` = `@qr` 選択 + QR51
- `run.@iqn32_antithetic` = X の policy/current/target 3 箇所を antithetic
- `run.@iqn32_actor_approx` = R の初期 priority mode を actor_approx
- `run.@iqn32_repro` = deterministic backend

`config_resolution.json` では、各幹について `run.$` から幹、`DefaultDQNAgent.$` から ALGO、`DefaultDQNAgent.net.$` から `net.@iqn / @qr`、`DropMergeEnv.$` から Env 素材列、main branch から `net.branch.@SuikaImpala2_ViT128`、`backend.$` から backend 素材への解決を確認した。

### 検証結果

- Debug `anet-core-test` build: 成功。
- `[config]`: 91 test cases / 608 assertions、全通過。
- 全 core: 455 test cases 中 453 通過、4665 assertions 中 4663 通過。既存の ReplayBuffer `episode_start without done` 2 ケースだけ失敗した。
  - `ReplayBuffer n-step returns stop at episode_start without done`
  - `ReplayBuffer frame stacking starts a new stack at episode_start without done`
- workspace resolver: `Workspace launcher tests passed.`
- `git diff --check`: 対象ファイルで whitespace error なし。
- IQN smoke: `app.exp_exit_step=1`、`Network Head: IQN Dueling`、model shape dump、停止、正常 serialize を確認。
- QR smoke: `app.exp_exit_step=1`、`Network Head: Quantile Dueling (N=51)`、model shape dump、停止、正常 serialize を確認。
- smoke 用 Run 3 件は、対象絶対パスが `apps/runner/workspaces/dm-iqn/runs` 配下であることを確認して削除した。既存 Run artifact は変更していない。

### 実装時に判明した点と改善候補

- BODY 素材と Random Policy 素材は当初の作業列挙から漏れていたが、素材化ガイドの「選択肢として使う無印素材を残さない」という完了条件に含まれるため本変更で対応した。
- tau 方式 3 箇所を束ねる独立素材や、`run.$` チェーンによる部分幹合成は組合せ爆発を抑えられる。ただし命名と責務を全 Env で揃える必要があるため、横断整合チェック側の改善候補として保留する。
- R/X の素材値を幹から上書きすると、dump には素材側キーと解決済み Agent leaf の両方が現れる。幹の同期契約は source 行数ではなく、個別指定との dump 完全一致と最終 Agent leaf で確認する。
- batch smoke の終了値は、素材側の `app.batchrun.exp_exit_step` ではなく選択後 leaf の `app.exp_exit_step=1` を CLI 指定した。Run 予算を幹へ含めない方針は維持する。
