# PRD 059 CLI override 両相適用 修正実装メモ

## 概要

- CLIの源プレフィクス形がselectionへ反映されない回帰を修正し、「CLIはすべての設定ファイルより優先」を復元する。
- 公開API、selection、named幹、`${}`、resolution JSONの契約は変更しない。

## 実装変更

- `ResolutionEngine`構築時に、`IsResolverInputKey`による第1相フィルタを外し、全CLI overrideを`working_map_`へ先出しする。
- 第2相は現状どおり、selection・素材キーを除く実効leafだけを`effective_map_`へ最終適用する。
- `IsResolverInputKey`は第2相の篩に引き続き使用し、CLI指定の`.$`や`@`素材をdumpへ残さない。
- `ExpandNamedTrunk`、selection DFS、`ExpandReferences`には変更を加えない。
- コードコメントを「全CLI overrideを第1相、実効leafを第2相で再適用」という契約へ更新する。

## テスト

- 最初のREDとして、`app.batchrun.exp_exit_step=100`を`app.$=app.batchrun`で選択し、CLIで源キーを`200`へ上書きするテストを追加する。修正後は`app.exp_exit_step`と残置される源キーの双方が`200`になることを確認する。
- 既存テストと必要な追記で、通常selection・named幹に対するCLI実効leafの優先、CLI selectionのdump除外、`@vars`上書きの`${}`波及、参照元へのCLIリテラル上書きを固定する。
- 各ケースは`ConfigManager`公開経路で検証し、private helper用APIは追加しない。

## ドキュメント

- `docs/design/100_runtime_and_configuration.jp.md`のCLI第1相を「全CLI overrideをsource mapへ先出しし、実効leafは第2相で最終上書き」に更新する。
- PRD 059、25impl、ADR、`CONTEXT.md`、既存のPH1a／PH2／素材化／PER変更は編集・巻き戻ししない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[config]"
core\anet-core\bin\Debug\anet-core-test.exe
powershell -NoProfile -ExecutionPolicy Bypass -File apps\runner\tools\resolve_workspace_test.ps1
git diff --check
```

- 全coreで既知のReplayBuffer 2件だけが失敗する場合は本修正と分離して記録する。それ以外の新規失敗は完了扱いにしない。
- `git add`、commit、pushは実行しない。

## 前提

- `apps/12_batch_run_atari5.bat`のキー変更は行わず、現行の`app.batchrun.exp_exit_step`指定をResolver側で再び有効にする。
- 源プレフィクス形は、対応するselectionがその源を選んだ場合だけ実効値へ反映される。無条件の最終上書きが必要な指定は実効leaf形を使う既存契約を維持する。

## 実装記録

### RED→GREEN

- `ConfigManager applies a CLI source-prefix leaf before selection`をtracer bulletとして追加した。production変更前は、CLIで`app.batchrun.exp_exit_step=200`を指定してもselection後の`app.exp_exit_step`がファイル値`100`のままになり、1 test case / 1 assertion failureのREDを確認した。
- `ResolutionEngine`の第1相から`IsResolverInputKey`条件を外し、全CLI overrideを`working_map_`へ先出しした。同じtracerは3 assertionsでGREENになり、残置される源キーも`200`であることを確認した。
- `${}`を供給するselectionへCLI実効leafのリテラル値を重ねるテストを追加し、最終値がリテラルのまま、不要なreference記録が生成されないことを確認した。
- 既存の通常selection、named幹、CLI selectionのdump除外、`@vars`上書きの波及と合わせ、`[config][resolver][cli]`は6 test cases / 26 assertionsで成功した。

### 実装・ドキュメント

- CLI第1相は全override、第2相はselection・素材を除く実効leafの再適用とした。`IsResolverInputKey`は第2相の篩に維持した。
- `ExpandNamedTrunk`、selection DFS、`ExpandReferences`、公開API、resolution schemaは変更していない。
- `docs/design/100_runtime_and_configuration.jp.md`の手順3を両相適用契約へ更新した。
- `apps/12_batch_run_atari5.bat`、PRD 059、25impl、ADR、`CONTEXT.md`には本作業による変更を加えていない。

### 最終検証

- MSVC Debugで`anet-core-test`と`AnetRLRunner`をビルドし、成功した。
- `[config]`: 91 test cases / 608 assertions、全成功。
- `resolve_workspace_test.ps1`: 成功。
- 全core: 455 test cases中453成功、2失敗。失敗は既知の`ReplayBuffer n-step returns stop at episode_start without done`と`ReplayBuffer frame stacking starts a new stack at episode_start without done`だけで、本修正とは分離する。
- `git diff --check`: 成功。
- 既存のstaged/unstaged/untracked変更を保持し、`git add`、commit、pushは実行していない。
