# PRD082 Atari RAM メトリクス実装メモ

## 概要

PRD082 と ADR 0046 に従い、ゲーム別の RAM 定義を番号付き `ram_metric.[n]` に結び、実ゲーム完了値・進行中の表示・完了ログを同じ定義から出す。既存の報酬と `game_score` 契約は維持する。

## 主な変更

- `AtariEnvConfig` は既定 `AtariEnv.ram_metric.[game]` の全ブロックを ROM 不要で読み、ゲーム名・ラベル・バス番地・閉じた 5 語の畳み方・番号付けを検証する。生の設定キー一覧で不正な RAM 定義キーと `run.eval.[tag].env.ram_metric.*` を検出する。番号のない定義は評価しないが、受理した定義と `metrics` 行は Module Config の実効設定と JSON に記録する。
- `AtariEnv` は番号付き定義のみを保持し、全 `ale_->act()` 後の RAM を観測する。全 `reset_game()` 直後の値を v0 として仕切り直し、episodic-life の soft reset では継続する。`RecordGameCompletion` で番号順に値を確定し、`GetScalar` は未完了・現在ゲームに番号なしなら NaN、全ゲームに番号なしなら `nullopt`、不正な `ram_metric.` 書式なら fail-fast にする。
- `MakeAuxData` に現在値 `ram_metric.<label>` の int64 tensor を載せる。AtariView はキー順で既存のオーバーレイ末尾に表示する。完了ログには番号順のラベル付き確定値を追加する。番号のないゲームでは RAM 読み取り・表示・ログ追加をしない。
- `Atari.txt` に kung_fu_master、qbert、phoenix、breakout の定義を既定葉の `?=` で置き、trace 4 行の番号 1～3、train scalar 3 行を追加する。`docs/design/220_atari_env.jp.md` の設定・accessor・RAM 表・表示を更新する。PRD の未番号付け `floor_max` 表示例は契約に合わせて修正する。

## テスト

- Public surface: `AtariEnvConfig`、`AtariEnvFactory` から得る Env の `GetSpec` / `Reset` / `Step` / `GetScalar`、結果の AuxData、完了ログ、解決済み `Atari.txt`。
- 最初の縦スライス: Pong の定義 `0x8D max_seen` を公開設定から読み、Reset/Step の AuxData 現在値と未完了 NaN、truncation での確定値を 1 ケースで確認する。RED を確認して最小実装で GREEN にする。
- 続いて 1 挙動ずつ RED→GREEN: 5 種の畳み方と v0/遷移、全 `act()`/`reset_game()` 経路と soft reset、番号の有無と不正参照、全ブロックの設定 fail-fast、ログ、オーバーレイ用データ。内部実装の置換に耐える公開面のテストを優先する。
- ROM が必要なケースは既存の `FindRom`/SKIP 規約を使う。Pong の deterministic seed と kung_fu_master の短い truncation を使い、RAM の実測値は PRD の期待値と照合する。

## 検証

- 編集前 `[atari]` baseline: 2026-09-24、`AtariEnv-test.exe '[atari]' --reporter compact`、27 ケース・525 アサーション成功。既存 Pong truncation の seed 123 は score 0、game_len 2、game_frames 2。
- 編集前に trace/scalar の同 seed 比較に使う実行条件と出力を記録し、RAM キーなしで編集後と比較する。取得不能な場合は checksum pass と呼ばず、理由と代替 evidence を明記する。
- MSVC `VsDevCmd.bat` を初期化して x64-Debug の `AtariEnv-test` をビルドし、`[atari]` を実行する。設定解決後の既存キー・文字列値は RAM 追加分以外に差がないことを確認する。`git diff --check` も実行する。

### 実測結果と baseline の扱い

- 編集前の `[atari]` は 27 ケース・525 アサーションで成功。編集後は seed 123 で全ケースを再実行し、36 ケース・601 アサーションが成功した。既存の Pong truncation / log / reward / episodic-life テストも含む。
- `Atari.txt` の解決後に 4 ゲームの番号表、trace のキー列、scalar の `source_key` を確認した。Pong の 1 skip 窓で CPU 得点 RAM の 21 回の増加、kung_fu_master の短い truncation と soft reset / hard reset の境界を実 ROM で確認した。
- `anet-core-test.exe '[trace]' --rng-seed 123` は 9 ケース・449 アサーション、`'[metrics][observers]' --rng-seed 123` は 13 ケース・67 アサーション成功。trace テストは生成済み `out/test-tmp` の fixture 削除が読み取り専用 sandbox に拒まれたため、同じコマンドを許可付きで再実行した。
- **waiver:** 編集前に同 seed の trace/scalar JSONL 行を保存できなかったため、ADR 0037 型の「編集前 baseline との行単位 checksum pass」は主張しない。代替 evidence は編集前・編集後の既存 Atari suite、RAM 定義なしでの既存ゲーム完了ログの厳密一致テスト、ObserverFactory の解決済み trace/scalar 定義検査、既存 trace/scalar observer suite である。実 Run の行単位比較は未実施。

### 2026-09-24 修正ラウンド（D17 と設定監査）

- D17 の公開面テストで、受理した番号付き定義・`metrics` 行・番号なし定義が `AtariEnvConfig::GetScopedConfigData()` に入り、JSON にも記録されることを確認した。Pong の ROM テストでは Env 自身の `GetConfigData()` に定義と `metrics` 行が載ることも確認した。修正前は新テストの 5 アサーションすべてが失敗し、修正後に通過した。
- 最終ソースを MSVC x64-Debug で `AtariEnv-test` にビルドし、`[atari] --rng-seed 123 --reporter compact` は **37 ケース・609 アサーション成功**。これには D17、ROM 依存の Pong / kung_fu_master、設定拒否条件、既存の報酬・reset 回帰を含む。
- `check_default_leaves.py` の初回実行は追加した RAM 宣言 15 行の `=` を既定葉として検出し、15 エラーだった。該当 15 行だけを `?=` へ直して再実行し、**23,706 assignments / 3,286 audited / 0 errors** となった。
- 全キー・文字列値比較は、Git `HEAD` の `Atari.txt` を `.scratch/prd082-config-compare/Atari-before.txt` に採り、現行 `Atari.txt` と同じ `_main.txt`・同じ他の設定ファイル・同じ CLI 選択で解決した。`v5_noop30` と `classic` の各プリセットで、RAM 定義 15 キー、scalar 宣言と展開結果の計 6 キーが追加され、trace 4 キーへ `ram_metric.[1] ram_metric.[2] ram_metric.[3]` が末尾追加された。それ以外は**全キーの存在と文字列値が一致**し、旧キーの欠落もなかった。一時比較テストは 1 ケース・6,657 アサーション成功後に除去し、上記の最終ビルドと suite を再実行した。この比較は設定解決の比較であり、前項の trace/scalar JSONL checksum waiver を取り消すものではない。

### 2026-09-24 修正ラウンド 2（D19）

- PRD の D19 に従い、`Atari.txt` の RAM scalar を番号由来の tag に改め、train・eval1・eval2・evalg の 4 群へ mean 12 本・EMA 9 本・max 12 本の計 33 本を配置した。RAM 定義と trace 4 行は変更していない。設計文書 220 §4.9 には、eval env の Module Config ダンプに出る `run.eval.[tag].env.ram_metric.*` は実効値の記録であり、同形の入力指定は禁止されることを追記した。
- 新 tag `42_env/50_ram_metric_1_mean` の設定・解決済み定義を期待するテストが旧設定に対して 2 アサーション失敗する RED を確認した。設定変更後、33 本すべての tag・`source_key`・EMA 有無と係数・実行条件を解決済み `ScalarMetricDef` で検証した。旧 3 tag と `53_evalg` の EMA 3 tag の不在も確認した。
- 一時比較テストで修正前の作業中 `Atari.txt`（`.scratch/prd082-21-config-compare/Atari-before.txt`）と修正後を同じ `_main.txt`・CLI 選択で解決した。`v5_noop30` と `classic` の各プリセットで、旧 scalar の宣言・展開結果 6 キーが消え、新 33 本の宣言・展開結果 66 キーが増えた。それ以外の**全キーの存在と文字列値は一致**した（1 ケース・6,855 アサーション成功）。一時比較テストを除去して最終ビルドを行った。
- MSVC x64-Debug の最終 `AtariEnv-test` で `[atari] --rng-seed 123 --reporter compact` は **37 ケース・992 アサーション成功**。`ATARI_ROM_DIR` は `.venv/Lib/site-packages/ale_py/roms` を指し、Pong と kung_fu_master の ROM 依存ケースも実行した。`check_default_leaves.py` は **23,755 assignments / 3,316 audited / 0 errors**。`Atari.txt` は UTF-8 BOM と LF を維持し、`git diff --check` は空白エラーなし。前項の trace/scalar JSONL checksum waiver は引き続き適用する。

## 前提

- 未決事項監査のユーザー判断ブロッカーは 0。ADR 0046 と `CONTEXT.md` の現行定義を使い、PRD/ADR/用語集を作り直さない。
- `ConfigData::MakeSubConfigData` は角括弧入り override prefix を正しく扱えないため、禁止キーの検出には生の設定キー一覧を使う。
- `Atari.txt` の既存の作業中の選択・コメントを保持する。旧 API・旧キーへの互換層は追加しない。
