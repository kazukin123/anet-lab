# PRD194 設定キー診断の確認

[PRD194](../../../../../docs/memo/0194_config_key_full_names_in_messages_10rfd.md) §8.2 の条件で、エラー・WARN・INFO と診断へ設定キーを渡す呼び出しを調べる。リポジトリルートで実行する。

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0194/audit_config_messages.py --expected src/core/base/testdata/rfd0194/expected.jsonl
```

期待一覧は、ファイル・種類・名前の並びを1行ずつ持つ。集計結果と各組の件数が一致すれば終了コードは0になる。差がある場合は余分な組と足りない組を表示する。期待を変更するときは PRD §8.7 の規則と根拠を実装記録に残す。

行番号付きで起草時の集計を確認する。

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/rfd0194/audit_config_messages.py --ref a6e920ed
```

初期の期待一覧は、起草時の303文と64呼び出しから §8.3 の移行対象を除き、§8.7 の27文は残る名前に絞って作った。動的に組み立てるキーや固定のルートを見分けるため、診断経路のテストと §8.5・§8.6 の確認も行う。
