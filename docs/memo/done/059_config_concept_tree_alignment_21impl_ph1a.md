# PRD 059 Phase 1a `<agent>.net` rename・DQN 契約検証 実装計画

## 概要

- DefaultDQN / ImageCls / Rainbow の最終 NN ツリー読込先を `<class_id>.net` へ移す。
- DefaultDQN Factory で `quantile_mode` と `taus` bind の局所契約を fail-fast 検証する。
- MuZero のコード、root の `net.rep` / `net.dyn` / `net.pred`、`GridMaze_muzero.txt` は変更しない。
- 本文書を Phase 1a の実装正本とし、RED → GREEN の結果と最終検証結果を追記する。

## 実装変更

- DefaultDQN / ImageCls / Rainbow の各 Factory は `GetTargetAgentClassId() + ".net"` を `NetworkConfig` に渡す。`NetworkConfig` の公開 API と既定 prefix は変更しない。
- `net.block.[*]` と `net.config_profile` はグローバル共有のまま維持し、既存の global → agent-local merge を利用する。
- `DefaultDQNAgentFactory` は両 Config 構築直後、Agent・NetworkModel 構築前に次を検証する。
  - 全 branch の解析済み `bind_terms` から、factor が完全一致で `taus` のものを列挙する。
  - `iqn` は `taus` bind を 1 件以上必要とする。
  - `qr` / `none` は `taus` bind を禁止する。
  - エラーは英語の `ANET_SYSTEM_ERROR` とし、`DefaultDQNAgent.quantile_mode` と、欠落時は `DefaultDQNAgent.net.branch.[*].bind`、余剰時は該当する完全な bind key を含める。
  - branch の到達性や最終出力への意味的寄与は検証しない。
- validator は `DefaultDQNAgentFactory` の private static member とし、匿名 namespace や新規ソースファイルを追加しない。既存の無関係な helper は変更しない。

## 設定移行

- 最終ツリーへの直書き 107 行を rename する。
  - Atari / CartPole / DropMerge / GridMaze / LunarLander: `DefaultDQNAgent.net.*`
  - ImageCls: `ImageClsAgent.net.*`
  - `nn.txt` の汎用 3 例: `<agent>.net.body.$`
  - `nn.txt` の IQN 固有 7 例: `DefaultDQNAgent.net.*`
- bat の 5 行も LHS だけを同じ規則で変更する。
  - tracked の `apps/12_batch_run_atari5.bat`
  - workspace に存在する ignored bat 2 ファイル・4 行
  - CP932 + CRLF を維持し、ignored ファイルはローカル移行として記録する。
- RHS の `net.qr` / `net.iqn` / named branch / body 素材、`net.block`、`net.config_profile` は変更しない。
- Phase 1b の素材 `@` 化・`:` 移行、Rainbow validator、Phase 2 機能は実施しない。

## ドキュメント

- `docs/design/100_runtime_and_configuration.jp.md` に `<class_id>.net` の読込、グローバルカタログとの merge、DefaultDQN Factory 検証、MuZero の Phase 1a 対象外を反映する。
- ユーザー変更中の PRD 059、ADR 0030、`CONTEXT.md` は編集しない。

## TDD と回帰検証

- tracer bullet: agent 配下の IQN ツリーとグローバル `net.block` を用い、`DefaultDQNAgentFactory::CreateAgent()` 経由で Agent 構築と action 生成が成功することを確認する。
- 続いて `iqn` + tau なし、`iqn` + selection なし、`qr` + tau あり、`none` + tau ありを 1 behavior ずつ RED → GREEN で追加する。
- validator は直接テストせず Factory の公開経路とエラー文を検証する。
- global block/profile と agent-local branch/output の merge、ImageCls / Rainbow Factory の class-id 由来 prefix を回帰テストする。
- golden は新 prefix を旧 `net.*` へ正規化し、凍結旧 AutoMerge の結果と key/value・`Properties` 順序を比較する。`GridMaze_muzero` は正規化なしで完全一致させる。
- QR / IQN の一時 workspace Runner smoke で Factory 検証と NN 構築を確認する。

## 最終検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[config]"
core\anet-core\bin\Debug\anet-core-test.exe "[workspace]"
core\anet-core\bin\Debug\anet-core-test.exe
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
git diff --check
```

- 既知の ReplayBuffer 2 件だけが同じ理由で失敗する場合は Phase 1a と分離して記録する。
- ユーザーの未コミット PRD・ADR 修正、ignored workspace、無関係な未追跡ファイルを保持する。
- `git add`、commit、push は実行しない。

## 実装記録

- `DefaultDQNAgentFactory` の tracer bullet は、agent-local IQN tree を root `net` として読んだため `No branches defined in NetworkConfig for prefix 'net'` で RED を確認した。Factory の prefix を `DefaultDQNAgent.net` へ変更し、global `net.block` と組み合わせた Agent 構築・action 生成が GREEN になった。
- `iqn` + tau なし、`iqn` + selection なし、`qr` + tau あり、`none` + tau ありは Factory 公開経路で順に RED を確認し、`DefaultDQNAgentFactory` の private static validator で GREEN にした。欠落エラーは `DefaultDQNAgent.net.branch.[*].bind`、余剰エラーは該当する完全な bind key を含む。
- ImageCls / Rainbow の tracer bullet も root `net` 読込で RED を確認し、`GetTargetAgentClassId() + ".net"` への変更で GREEN にした。global block/profile と agent-local branch/output の merge テストも追加した。
- runner config の最終ツリー 107 行を、Atari 11、CartPole 9、DropMerge 41、GridMaze 4、ImageCls 16、LunarLander 16、`nn.txt` 10 の内訳で移行した。root の素材・カタログと RHS は維持した。
- bat は tracked の `apps/12_batch_run_atari5.bat` 1 行と、ignored workspace の `10_live_run.bat` / `11_live_rehearsal.bat` 4 行を LHS のみ移行した。ignored 2 ファイルはローカル移行として CP932 + CRLF を維持した。
- frozen AutoMerge golden は 7 overlay について 7,040 assertions が通過した。DQN / ImageCls は新 prefix を旧 root `net.*` へ正規化して key/value と `Properties` 順序を比較し、`GridMaze_muzero` は正規化せず完全一致を確認した。MuZero の production code と `GridMaze_muzero.txt` に差分はない。
- Debug の `anet-core-test` / `AnetRLRunner` build、`[config]` 81 cases / 7,611 assertions、`[workspace]` 16 cases / 91 assertions、DQN・ImageCls の focused tests、`resolve_workspace_test.ps1` はすべて成功した。
- QR / IQN Runner smoke は CPU・1 env・`app.exp_exit_step=1` で正常終了し、それぞれ `Network Head: Quantile Dueling (N=32)` と `Network Head: IQN Dueling` まで通過した。
- 全 core test は 445 cases 中 443 passed / 2 failed、11,660 assertions 中 11,658 passed / 2 failed だった。失敗は既知の `ReplayBuffer n-step returns stop at episode_start without done` と `ReplayBuffer frame stacking starts a new stack at episode_start without done` の 2 件だけで、Phase 1a とは分離する。
- `git diff --check` は成功した。ユーザーの未コミット PRD / ADR と無関係な未追跡ファイルは保持し、`git add`、commit、push は実行していない。
