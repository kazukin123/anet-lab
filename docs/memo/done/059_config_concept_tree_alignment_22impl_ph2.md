# PRD 059 Phase 2 幹前段・resolution subcommand 実装メモ

## 概要

- ConfigResolverへ`run.$`のnamed幹をroot展開する前段を追加する。
- `inspect_run.py resolution`を追加し、新旧形式の`config_resolution.json`をMetrics master/cacheに触れず表示する。
- 公開C++ API、resolution payloadの`schema_version: 1`、inspect_runのtop-level`schema_version: 2`は変更しない。

## 主な変更

### ConfigResolver

- `ResolutionEngine::Resolve()`のCLI第1相適用後、selectionスナップショット前にprivateな幹前段を呼ぶ。
- `run.$`があればチェーンを左から右へ解決し、相対`@name`は`run.@name`、それ以外はroot絶対参照として扱う。素材配下はprefixを剥がしてrootの`working_map_`へ後書きする。
- 前段は`effective_map_`とnested selectionを処理しない。`run.$`を通常selectionから除外し、resolutionの先頭へ1 entryだけ記録する。
- 幹がrootの`run.$`を生成した場合は素材名と`named trunk must not select another trunk`を含む英語エラーでfail-fastする。未定義素材と下流の循環・深さ制限は既存診断を使う。
- `run.foo`は通常leafとして残し、幹供給leaf < selection結果 < CLI leaf overrideの既存順位を維持する。

### inspect_run resolution

- sourceは`json/config_resolution.json`のenvelope、`config/config_resolution.json`の素payload、missingの順で直接読む。
- Run nodeの`resolution`は`status`、`source`、`schema_version`、`trunk`、`selections`、`references`を持つ。statusは`ok | missing | source_error`とする。
- 未知schema versionはwarning付きでbest-effort表示する。優先sourceが壊れている場合は下位sourceへfallbackせず、そのRunを`source_error`、command exitを1とする。
- MarkdownはRunごとのsource/status/schema、任意のtrunk要約、selection表、reference表を出す。
- `open_run()`、Metrics master、gzip、`metrics_cache.db`、`runs` subcommandは変更しない。

### ドキュメント

- `docs/design/100_runtime_and_configuration.jp.md`へ幹前段、末尾追記相当、通常selectionへの引継ぎ、幹ネスト禁止を追記する。
- `docs/design/030_user_guide_analysis.jp.md` §6へ`resolution`を追加し、後続節を6.8/6.9へ繰り下げる。
- PRD 059、ADR、`CONTEXT.md`は編集しない。

## TDD

- Public surfaceはC++の`ConfigManager`とPythonの`inspect_run.py` CLIだけを使い、private helperを直接テストしない。
- C++は絶対named幹からroot leafと下流selectionを解決するtracer bulletから始め、相対幹、後勝ち、CLI幹切替、CLI leaf優先、幹ネスト、未定義幹、no-op互換、`run.foo`を1 behaviorずつRED→GREENにする。
- Pythonはenvelope＋幹表示のCLI tracerから始め、素payload、missing、未知schema、幹なし、複数Run、primary破損時の非fallback、Metrics source非参照を1 behaviorずつRED→GREENにする。
- refactorはGREEN後だけ行い、その都度関連テストを再実行する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[config]"
core\anet-core\bin\Debug\anet-core-test.exe
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run_test.py
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
git diff --check
```

## 前提

- 今回はPhase 2のコード部だけを実装し、実設定への`run.@...`導入、NatureDQNコメント運用の幹化、`runs`への幹列は後続作業とする。
- 既知のReplayBuffer 2件だけが全coreテストで失敗する場合はPhase 2と分離して記録する。
- 既存のPH1a、LunarLander素材化、staged/unstaged変更を保持し、`git add`、commit、pushは実行しない。

## 実装記録

### ConfigResolver

- tracer bulletでは、絶対named幹から生成された下流selectionが幹前段なしでは`run.AtariEnv.$`として解釈され、`material selection target not found`になるREDを確認した。`ExpandNamedTrunk()`をCLI第1相後へ追加し、root leaf、下流selection、resolution先頭の`run.$`を同時に確認してGREENにした。
- 相対幹、チェーンの左から右の後勝ち、root leaf/selectionへの末尾追記相当、CLI幹切替、CLI leaf優先、未定義幹、`run.$`なし互換、`run.foo`維持を`ConfigManager`公開経路で追加した。
- 幹ネストは当初そのままrootへ展開されるREDを確認し、生成targetが`run.$`なら素材名と完全なsource keyを含む英語エラーでfail-fastするようにしてGREENにした。
- 通常selectionと幹でresolution chain生成を共有するprivate helperへGREEN後に整理し、`[config][resolver]`は20 test cases / 79 assertionsで成功した。

### inspect_run resolution

- CLI tracer bulletでは`resolution`がargparseで未知subcommandになるREDを確認し、現行envelopeの直接読込、Run node、named幹要約、selection/reference表示を追加してGREENにした。
- 旧raw payload fallback、missing、未知schema、幹なし、複数Run、primary破損時の非fallback、不正payload、Metrics source/cache非参照をCLI公開経路で追加した。
- `json/config_resolution.json`と`config/config_resolution.json`を持つ既存Runをそれぞれ実読込し、両形式ともMarkdown表示とexit 0を確認した。
- `inspect_run_test.py`は53 testsで成功した。

### ドキュメント

- `docs/design/100_runtime_and_configuration.jp.md`へCLI第1相後の幹前段、rootへの後書き、通常selectionへの引継ぎ、resolution順、幹ネスト禁止を反映した。
- `docs/design/030_user_guide_analysis.jp.md`へ`resolution`のsource優先順、表示内容、missing、未知schema、primary破損、Metrics非参照を§6.7として追加し、旧§6.7/6.8を§6.8/6.9へ繰り下げた。

### 最終検証

- MSVC Debugで`anet-core-test`と`AnetRLRunner`をビルドし、成功した。
- `[config]`: 89 test cases / 602 assertions、全成功。
- `[workspace]`: 16 test cases / 91 assertions、全成功。
- `resolve_workspace_test.ps1`: 成功。
- 全core: 453 test cases中451成功、2失敗。失敗は既知の`ReplayBuffer n-step returns stop at episode_start without done`と`ReplayBuffer frame stacking starts a new stack at episode_start without done`だけで、Phase 2とは分離する。
- `git diff --check`: 成功。
- `git add`、commit、pushは実行していない。
