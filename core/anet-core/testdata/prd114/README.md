# PRD114 の検証

repoルートの `.venv` を使う。Python依存は `reports/tools/requirements.txt` にある。テストは独立した一時Git repoへ入力と出力を作り、本物のindex・ref・仕事一覧へ書かない。

```powershell
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd114/issues_test.py -v
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd095/wiki_test.py -v
```

- maps: issueからINDEXと地図の課題区画を生成。本版の既定、依存とWSJF継承、複数areaとアプリケーション自身の範囲、全入力・目印を検証してからの書き込み、区画外保持、内容と更新時刻の冪等性。
- 日付: raised・resolved・ai_judgedはYYYY-MM-DDを受け、整数・週日付・実在しない暦日を拒否する。任意の日付欄の空欄とYAMLの日付型は受け付け、引用符なしの不正暦日も欄と指定値を示す。入力エラーでは派生出力を変えない。
- scan: PRDを`prd`で指す課題の判定と状態からPRDの実装済みを導く表（PRD120 §5.2）と、本文が同じまま実装済みになったPRDの検出、`implemented:`を添えたingest/writebackによる取り込み済みの記録（パスとblobで突き合わせ、日付は使わない）。過去のdoneの扱いは既存テストでも確認する。
- stats: 単票の状態・種類・分類・完了の理由の集計、未追跡を含む直下PRDの状態をそれを指す課題から導くこと（PRD120 D16）、課題の無い直下PRDで止まること、過去の保管場所の件数を公開CLIで確認する。
- 移行の検証: 元のF番号と単票の一対一対応、旧台帳の退役、改名の番号の非再利用、すべての直下PRDと索引issueの一対一対応、skillコピー・LF・BOM・リンクと共有コアの不変を確認する。
- [migration.json](migration.json) は開始コミット、F番号→issue、PRD→索引issue、旧名→新名の固定した対応を持つ。移行の検証ではこの対応の過不足と実ファイルを突き合わせる。
- skill試走A・B・Dは実データの一時コピーで行い、意味判断を文字列の検査だけで成功扱いしない。Aは起票から独立レビューまで実施する。
- C・C2・Eと最初の5回の工程末の成功指標は、PRDが定める実際の場面がなければ未実施として実装メモに残す。
