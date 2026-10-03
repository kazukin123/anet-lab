# WeightInitConfig.mode 文字列化 実装メモ

## 概要

`WeightInitConfig::mode` を `int` から `std::string` へ変更し、既存5値を `default` / `xavier` / `he` / `orthogonal` / `constant` に移行する。既定値は現行動作を維持して `"xavier"` とする。

旧数値との後方互換、fallback、WARN は入れない。未知の mode 文字列や旧数値文字列が実行時に渡された場合は `ANET_SYSTEM_ERROR` で fail-fast させる。

## 主な変更

- `core/anet-core/include/anet/nn.hpp` の `WeightInitConfig.mode` を `std::string mode = "xavier"` に変更し、コメントを文字列規約へ更新する。
- `core/anet-core/src/nn_impl.hpp` の `WeightInitializer::Initialize` を文字列比較へ変更する。
  - `"default"` は weight / bias とも触らず return する。
  - `"xavier"` / `"he"` / `"orthogonal"` / `"constant"` は現行数値 mode と同じ初期化を行う。
  - 未知値は bias 初期化へ進む前に `ANET_SYSTEM_ERROR` で失敗させる。
- `core/anet-core/src/nn_modules.cpp` と `core/anet-core/include/anet/muzero_proto_agent.hpp` のコード内数値 mode を対応文字列へ置換する。
- `core/anet-core/src/nn_test.cpp` など既存テスト内の数値 mode を文字列へ更新し、`MakeResBlockTestModule` の `init2.mode = 2` は `"he"` にする。
- `apps/runner/config/*.txt` と `apps/runner/tools/dropmerge_optuna.py` の `init.mode = N` / `head_init.mode = N` を文字列へ置換する。
- `docs/adr/0008-weight-init-mode-string.md` は既に存在する未追跡ファイルを採用し、必要な場合のみ PRD と整合する最小文言修正を行う。
- `CONTEXT.md` は更新しない。これは強化学習ドメイン用語ではなく実装設定名の変更であるため。

## テスト

- `core/anet-core/src/nn_test.cpp` に、各文字列 mode が従来の torch 初期化関数と same seed で一致することを直接比較するテストを追加する。
- 同じテスト群で、既定値 `"xavier"`、`"default"` no-op、`"constant"` の bias 動作、未知値と旧数値文字列 `"2"` の `CHECK_THROWS` を確認する。
- 既存の NN / DQN head 関連テストが文字列 mode で引き続き通ることを確認する。

## 検証

```powershell
rg -n "init\.mode\s*=\s*[0-9]|head_init\.mode\s*=\s*[0-9]|\{\s*[0-9]\s*," core/anet-core apps/runner/config apps/runner/tools
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

runner smoke は既定では実行しない。run artifact を作るため、必要になった場合だけ別途実施する。

## 前提

- `trunc_normal` の追加は `028_cnblock_convnext_10prd.md` 側で扱い、この実装では追加しない。
- `GetNonlinearityType` の未知文字列 fallback は PRD 外なので変更しない。
- `apps/runner/11_batch_run.bat` の REM コメント内の旧表記は実行に影響しないため、必須変更対象にしない。
- 無関係な未コミット変更と未追跡ファイルは保持する。
