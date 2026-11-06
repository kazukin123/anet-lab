# PRD091：checkpoint 引き継ぎ時の optimizer 設定適用

## 概要

DefaultDQN の checkpoint 読み込み後、optimizer の全 options を現在の設定で置き換える。moment と optimizer の step は checkpoint から保持し、差分を WARN、適用後の値を INFO に出す。2026-10-15 承認済み計画を正本とする。既存の未コミット変更は保持する。

## 主な変更

- Learner に非公開の BuildOptimizerOptions() const を追加し、group 順の std::vector<torch::optim::AdamWOptions> を返す。Noisy OFF は通常 group、ON は通常・sigma の順。
- SetupOptimizer と Load で生成関数を共用する。parameter 分類・順序、通常／Fused の選択、構築時 verbose は維持する。
- Load の読み込み成功後、lr、weight_decay、eps、betas、amsgrad を完全一致で比較し、差分通知後 options 全体を代入する。optimizer は再生成せず、復元済み state を保持する。
- WARN は異なる項目ごとに1行、group 番号、option 名、checkpoint 値、設定値、対応キーを含める。lr は learner.alpha、eps は learner.adam_eps、weight_decay は通常 group が learner.weight_decay、sigma が learner.noisy_weight_decay。betas／amsgrad はキーなし。
- INFO は適用後 group ごとに1行、lr・weight_decay・eps を出す。learner.enabled=false にも同じ規則を適用する。
- 公開 API、設定キー、archive 形式は変更しない。group 数・parameter 数の不一致検証は LibTorch に委ねる。

## テストと TDD 順序

Public surface は Learner::Save/Load、TDLearner の公開更新、DefaultDQNAgent の auto_load_file とログ。dqn_based_agent_test.cpp で各挙動を1件ずつ RED→GREEN に進める。

1. 通常 AdamW・Noisy OFF で非ゼロ勾配による1回の更新後に Save。異なる設定へ Load し、新設定の options と exp_avg・exp_avg_sq・step の保持を確認。
2. WARN 件数、キー・両値、group ごとの INFO を検証。同一設定は WARN 0件。テスト側で betas／amsgrad の差分も作り、既定値への置換とキーなし通知を検証。
3. Noisy 2 group の sigma decay、通常／Fused 間の読み込み、enabled=false を検証。
4. checkpoint の options 復元を期待する既存 Noisy 往復テストを新契約へ更新。同一設定での次回更新の等価性を維持し、異なる設定の適用は専用テストで検証。
5. Save→auto_load_file fixture で alpha を変更した Agent 構築の WARN・INFO を検証。

TDLearner の final は変えない。state 検査は既存 Learner 派生テストクラス、実際の TDLearner は公開 Save／Load・更新経路で被覆。本体にテスト専用 API は追加しない。

## 検証と文書同期

- MSVC 初期化経由で Debug ビルド。追加テスト、DQN／Noisy／serialization 関連、通常スイートを実行。
- 既存 Run の agent_close.anet と互換構成・同一 device で CLI の alpha を変更して短い smoke。WARN・INFO、実効 config と INFO の lr の一致、終了コード0を確認し受入証拠を保持。
- 設計文書200の日英版に options／state の引き継ぎ規則と sigma group への適用を記載。
- Run ガイド020の日英版の checkpoint「再開」を「引き継ぎ」へ統一。DefaultDQN に限定し新しい Run の optimizer 設定が効く旨を追記。ガイドから PRD・ADR は参照しない。

## 前提

ブロッカー0件。PRD091・ADR0053の確定事項に従う。ImageCls・Rainbow・MuZero、Run 全体の再開、異なる device への Load、policy 選択、schedule・メトリクス追加は対象外。

## 実装・検証結果（2026-10-15）

- 実装完了。options 生成は BuildOptimizerOptions に集約し、Load は state の復元後に全 options を現在の設定へ置換する。構築・Load の group 順序、archive 形式、公開 API は維持した。
- テスト側の SetupAdamW / Optimizer accessor は既存 TestLearner のみに追加した。TDLearner の final、本体の公開 API は変更していない。
- 日英の設計文書200・Runガイド020を同期した。CONTEXT、ADR0053、PRD091本文の既存変更は保持した。

### TDD と回帰検証

リポジトリルートの PowerShell で実行したビルド・既定スイートのコマンド:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe
```

| 検証 | 結果 | 証拠（リポジトリルート相対） |
|---|---|---|
| 変更前の既存 Noisy optimizer 往復 | 70 assertions 成功 | 実行時出力 |
| 最初の RED | lr / weight_decay / eps の3不一致、state の検査は成功 | out/prd091/01-red-options.log |
| 最小修正の GREEN | 28 assertions 成功 | out/prd091/04-green-options.log |
| 共通化と既存テスト移行 | 4 cases / 266 assertions 成功 | out/prd091/05-green-refactor.log |
| ログ契約の RED → GREEN | 未実装ログの21失敗 → 154 assertions 成功 | out/prd091/06-red-logs.log、07-green-logs.log |
| Noisy / 通常・Fused / enabled / 同一設定 / 既定options差分 | 64条件 / 3,056 assertions 成功 | out/prd091/08-matrix.log |
| Agent 経路を含む対象回帰 | 5 cases / 3,305 assertions 成功 | out/prd091/09-agent-and-regression.log |
| 全 Debug ビルド | 成功、終了コード0 | out/prd091/09-build-debug.log |
| anet-core-test 既定スイート | 677 cases 中676成功・1 skip、37,961 assertions 成功、終了コード0 | out/prd091/10-default-suite.log |

スキップは PRD070 の CUDA 利用不可環境向け検査で、この環境では CUDA が利用できるため。初回本体ビルドでは既存 PCH と MSVC のバージョン不一致（C1853）が発生したため、生成済み PCH と生成元 object を再生成して解消した。ソース・CMake の回避変更は加えていない。

### 同一 device smoke

- 親 Run: run_20261014-080249_dmfb_31e_21_b512rr2 の agent_close.anet。
- 保存時・読み込み時とも CUDA。親 Run の保存済み実効 config を --config で読み、auto_load_file と learner.alpha=5e-5 を CLI で指定した。
- 短時間検証用に num_envs=8、replay_capacity=512、replay_batch_size=32、update_warmup_steps=0、exp_exit_step=256 とし、background eval の interval を0にした。Network 構成は維持した。
- 結果 Run: run_20261015-085156_prd091_smoke。終了コード0、終了時 agent_close.anet 保存成功。
- group 0 / 1 の双方で checkpoint=0.0001 → config=5e-05 の WARN と lr=5e-05 の INFO を確認。実効 config の learner.alpha=5e-5 と一致した。
- loss / grad_norm は exp_step 48–240 に13点記録され、読み込み後の学習更新も確認した。
- 証拠: out/prd091/smoke-verify.json、smoke-config.json、smoke-tags.json、smoke-arguments.json。実行・照合スクリプトは out/prd091/smoke.ps1 と smoke_verify.py、Run 本体は out/prd091/runs/run_20261015-085156_prd091_smoke/ に保持した。
