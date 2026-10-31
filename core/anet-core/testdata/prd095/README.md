# PRD095の検証

repoルートの.venvを使う。台帳CLIのテストは一時ディレクトリへ台帳・分類・地図を作り、正本の台帳には試験用レコードを追加しない。wiki補助処理のテストはGit object DBを読み取る一時worktreeと、終了時に除去する一意名の材料を使う。全Git呼び出しはGIT_INDEX_FILEで指定した一時indexを使い、git read-tree HEADで初期化する。本物のindexとrefには書き込まない。

```powershell
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd095/registry_test.py -v
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd095/wiki_test.py -v
```

- 台帳CLI: list --category、拡張子によらない_test除去、110とアプリケーション地図の範囲の違い、未知key、open/stale、区画外保持、不正な目印で全派生ファイル不変。
- wiki補助: 作業ツリーの材料とblob、fix除外、同blobの移動とdone移動、未追跡ファイル・テスト集約、再生成の冪等性、PRD候補0本・複数本でリンク無しの注記と報告を出して全地図を更新すること、Gitが専用indexを選んでHEADのファイル集合を読めること。
- エージェントの試走（受入8〜14・17）は機械テストと区別し、取り込み・撤回・消えた名前・リンク修正・doneの2節・housekeeping選定・監査の結果を実装計画へ記録する。意味判断を文字列置換テストだけで成功扱いしない。
- 試験用のページ・PRD・行は最後に作成した分だけ消し、log.mdは追記を保持する。正本の材料を試験のために書き換えない。
- statsの分類集計は変更前後で比較する。skillの実体コピー・LF・BOM、出典リンク、目録、公開できない値を確認する。

- 段階3: scan CLIは開始revision以前から未変更の材料も検出し、ingestのblobが一致すると候補から除く。全体走査でも既存の移動・更新優先・派生物除外の契約を保つ。
