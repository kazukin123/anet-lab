---
name: anet-audit
description: Integrated improvement audit for anet-lab. One lens (refactor, defect, architecture, drift, test-gap, backlog, process) times one scope per run, read-only, with findings accumulated in reports/registry/findings.jsonl through fingerprint dedupe and a status lifecycle, and derived views rebuilt. Use when the user invokes $anet-audit or /anet-audit, optionally with lens=<lens> scope=<path or category key or design doc> max=<n>, or from anet-housekeeping.
---

# ANET Audit

コード、文書、開発プロセスの状況を観点ごとに定点で見て、改善候補を台帳に蓄積する。毎回新しいレポートは書かない。成果物は台帳の差分だけで、読む側は「最近の変化」と上位数件だけ読めばよい。

## 引数

`$anet-audit [lens=<観点>] [scope=<パス | カテゴリ key | 設計文書>] [max=<新規件数の上限。既定 8>]`

## 観点

| 観点 | 見るもの | 基準 | 典型的な分類タグ |
|---|---|---|---|
| refactor | 可読性、1 関心 1 機構、重複の生成源、既存の重複と死にコード、肥大化、命名、規約違反、変更の波及、依存の複雑さ、テスト容易性 | `docs/リファクタリング観点.txt` を全文読む。無ければ本表の項目で代用 | 観点ファイルの見出し名 |
| defect | 所有権・寿命・スレッド境界、未定義動作、例外安全、境界条件、TODO/FIXME のうち実害があるもの | 具体的な失敗シナリオが書けるものだけ | 所有権 / 寿命 / 並行 / 境界 / 例外 |
| architecture | モジュールの深さと seam、レイヤ違反、循環依存、ボイラープレートやアダプタの生成源 | `improve-codebase-architecture` スキルの DEEPENING 語彙があれば使う。CONTEXT.md の用語で書く | seam / 依存 / 重複生成源 |
| drift | 設計文書 1xx・2xx、CONTEXT.md、ADR とコードの不一致。設定キーの 3 集合差(コードが読むキー、設定ファイルのキー、文書のキー) | 文書 1 本を読み、対応コードと突き合わせる | 文書 stale / 未文書 / 死にキー / 用語ずれ |
| test-gap | 未テストの prod ファイルに対する特性テストの提案 | `reports/stats/trend.md` の「テスト負債」を入力に、LOC × churn 順、年代 pre 優先 | test-gap |
| backlog | `docs/memo` 直下の 9xx PRD、`frozen/`、`docs/*.txt` の散在メモの前提が今も成り立つか。実装済みなのに直下に残る PRD | 前提をコードで確認する | 前提崩れ / 実装済み / 重複 PRD |
| process | 未コミット差分の規模と滞留、直下 0xx の WIP、docs 同時更新率、AGENTS.md 規約の遵守(抜き取り) | `reports/stats/summary.md` と `trend.md` | WIP / 滞留 / 規約 |

ガードレール(観点ファイルの運用ルールを引き継ぐ):

- 観点は排他ではなくタグ。複数に当たる候補は最も根本の観点で 1 件にし、他は付随として summary に書く。
- 挙動が変わる修正はリファクタ候補にしない。defect として扱う。
- hot path(Sample、Push、Learn、per-step 経路)や RNG 呼び出し順に触る提案は、対処方針に perf-neutral 確認と同 seed 再現性チェックを含める。
- ホットスポット補正: `reports/stats/stats.json` の `flow.churn90` で上位のファイルは優先度を 1 段上げ、今後触る予定の無い場所は 1 段下げる。

## 範囲(scope)の決め方

1. 引数があればそれを使う。
2. 無ければ `reports/registry/rotation.json` を読む。`next_lens` を観点にし、`scopes` のうちその観点の `last_seen` が最も古いもの(未実施が最優先、同点なら churn90 合計が大きい方)を選ぶ。コード観点(refactor、defect、architecture、test-gap)は `kind: code` の scope、drift は `kind: doc`、backlog と process は scope を持たず全体を見る。
3. 実行後、該当 scope の `last_seen[<lens>]` を今日にし、`next_lens` を次に進める。順は refactor → defect → architecture → drift → test-gap → backlog → process → refactor。

## 手順

1. lens と scope を確定し、`run_id = <YYYY-MM-DD>-<claude|codex>` を決める。
2. 基準を読む(観点ファイル、設計文書、summary.md のうち該当するもの)。
3. 範囲のコードと文書を読む。読み取り専用のコマンド(rg、git log、git blame、`python reports/tools/registry.py list`)は使ってよい。ビルドとテスト実行はしない。
4. 既存レコードを再検証する。

   ```
   python reports/tools/registry.py list --status open --path <scope のパス>
   ```

   各件の根拠が今も存在するか確認し、存在すれば `touch --ids ... --run <run_id>`、消えていれば `set --id ... --status fixed --note "<根拠>"`、判断できなければ `set --id ... --status stale`。
5. 新規の指摘を 1 件 1 行の JSON にまとめ、一時ファイルから登録する。fingerprint(観点 + ファイル + シンボル)で既知のものは自動で弾かれ、last_seen だけ更新される。

   ```
   python reports/tools/registry.py add --file <tmp>.jsonl --run <run_id>
   ```

   件数は `max` 以内、優先度の高いものから。
6. 派生ビューを再生成し、検証する。

   ```
   python reports/tools/registry.py build
   python reports/tools/registry.py validate
   ```

7. `rotation.json` を更新する。
8. 完了報告: 観点と範囲、再検証の結果(touched / fixed / stale の件数)、新規(id と優先度)、上位 3 件の要旨、次回の観点と範囲。

## レコードの品質基準

- `file` と `evidence`(path と line)を必ず付ける。`scenario` は「入力・状態 → 何が起きる」か「具体的な smell」を書く。「可能性がある」だけの推測は書かない。
- `priority`: P0 = 学習結果や再現性を壊す、クラッシュ。P1 = 実害のあるバグ、同型バグを生み続ける構造。P2 = 変更コストを上げる構造、規約違反。P3 = 命名、コメント。ホットスポット補正を適用する。
- `confidence`: high = 根拠を読んで確定。med = 状況証拠。low = 要確認。
- 既存レコードの本文(title、summary、scenario)は書き換えない。変化は `status` と `history` で表す。
- レコードの全項目と CLI は `reports/tools/registry.py` 冒頭の docstring が正本。

## rotation.json

```json
{"next_lens": "refactor",
 "scopes": [
   {"key": "318_rl_core", "kind": "code", "last_seen": {"refactor": "2026-09-20"}},
   {"key": "docs/design/150_replay_buffer.jp.md", "kind": "doc", "last_seen": {}}
 ]}
```

`kind: code` の key は `reports/tools/categories.json` のカテゴリ key(パスの集合はそこから引く)か、ディレクトリのパス。

## 禁止

- コードを変えない。テストを書かない。PRD を作らない。提案は台帳に残すだけ。人が PRD に昇格させたら `set --status promoted --note "<PRD 番号>"` で閉じる。
- git の書き込み操作をしない。
- 質問して止まらない。観点や範囲の判断は本書の規則で決め、完了報告に書く。
