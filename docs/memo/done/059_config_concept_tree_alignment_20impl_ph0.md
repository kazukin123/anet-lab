# 設定体系再設計 Phase 0 実装メモ

## 概要

`ConfigManager::GetConfigData()` の契約を維持したまま、既存 `AutoMerge()` を private deep module `ConfigResolver` へ置き換える。既存 config を変更せず同じ実効設定を生成し、新しい素材宣言・参照構文を追加する。解決記録は構造化 Run metadata として既存 JSON 出力経路へ統合し、`json/config_resolution.json` と Metrics master の両方へ記録する。

## 主な変更

- 設定読込時に、最初の `=` より左を key として空白除去・`:` 糖衣のドット正規化を行う。`:` の複数指定・空片は fail-fast、`$include` 欠落は既存どおり WARN 継続とする。
- resolver は base → injected → overwrite file → CLI 第1相を source map に統合し、selection DFS → CLI leaf override → `${full.key}` 展開 → effective config・resolution 記録生成の順に処理する。
- selection は単独の `@name` を LHS 所有者配下の相対参照、それ以外を root 絶対参照として扱う。素材参照の未定義、循環、深さ10超過は経路付きで fail-fast、無印 prefix の0件解決は従来どおり no-op とする。
- デフォルト直書き < selection の解決結果 < CLI leaf override の全順序を固定する。素材から生成された nested `.$` も再帰解決し、独立した selection 記録を残す。
- `${full.key}` token は CLI leaf override 後の値で置換し、各 token を `references` に記録する。未定義参照、参照先の再参照、実効値に残る未解決 token は fail-fast とし、式評価は導入しない。
- effective config から `.$` と `@` セグメントを含む素材定義を除外する。無印素材・上書き層は Phase 0 では従来どおり残す。
- `ConfigManager::GetResolutionJson() const` を追加し、serialize 済み文字列ではなく `anet::json` を値で返す。公開 record 型は作らない。payload は `schema_version: 1`、決定的な解決順の `selections`、`${}` ごとの `references` を持つ。
- Runner は `config_data.txt` と同じ初期化タイミングで既存 `MetricsLogger::Log("config_resolution", json)` を呼ぶ。`json/config_resolution.json` は `type` / `tag` / `data` envelope を持ち、同じ record を Metrics master に記録する。Config 固有の artifact API は追加しない。
- 設定解決順と新 artifact を `docs/design/100_runtime_and_configuration.jp.md` に反映する。

## テスト

- Public interface / surface: `ConfigManager::GetConfigData()`、`ConfigManager::GetResolutionJson()`、Properties 設定構文、CLI `key=value`、既存 `MetricsLogger::Log(tag, json)`、Runner 起動時 artifact。
- 優先 behavior: 相対素材 selection の tracer bullet → 既存絶対チェーン → nested selection → 右勝ち・CLI 2相 → 素材除外 → `:` 正規化 → `${}` override 波及 → 未定義素材・循環・深さ超過・未解決参照 → golden comparison → JSON metadata artifact。
- TDD 順序: 各 behavior で1テストを追加して失敗を確認し、最小実装で GREEN にしてから次へ進む。refactor は GREEN 後だけ行う。
- golden comparison: test-local の凍結した旧 AutoMerge を oracle とし、`_main.txt` に CartPole、GridMaze、GridMaze_muzero、ImageCls、Atari、DropMerge、LunarLander の現行 overlay を重ね、key/value と Properties 出力順が完全一致することを確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[config]"
core\anet-core\bin\Debug\anet-core-test.exe "[metrics][config]"
core\anet-core\bin\Debug\anet-core-test.exe "[workspace]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- Phase 1 の `<agent>.net` rename、Factory ドメイン検証、設定ファイルの `@`/`:` 移行は行わない。
- Phase 2 の root 持ち上げ、named 幹、`inspect_run.py` 対応は行わない。
- resolution 詳細モードの `sources` / `writers`、`$include` fail-fast 化、式評価は追加しない。
- 既存 JSON 経路の数値丸め、pretty-print、timestamp 付与を共通契約として受け入れる。resolution は分析・診断用 metadata であり、再読込対象は `config_data.txt` のみとする。
- Metrics Viewer の専用表示と `inspect_run.py` 対応は行わない。
- `CONTEXT.md` と ADR 0030 は変更しない。
- 未コミットの PRD、ADR、Atari 設定および無関係な変更は保持する。
