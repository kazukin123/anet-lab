---
name: anet-audit
description: Audit one lens and one scope of anet-lab, record findings as one issue per file, and update only AI judgments when rechecking existing issues. Use $anet-audit or /anet-audit, or from anet-housekeeping.
---

# ANET Audit

コード、文書、開発プロセスを定点で見て、改善候補を `docs/management/issues/` に 1 課題 1 ファイルで記録する。起動（housekeeping の了承を含む）は、新しい指摘を記録する指示である。

## 引数

`$anet-audit [lens=<観点>] [scope=<パス | カテゴリ key | 設計文書>] [max=<新規件数の上限。既定 8>]`

## 観点

| 観点 | 見るもの | 基準 | 典型的な分類タグ |
|---|---|---|---|
| refactor | 可読性、1 関心 1 機構、重複の生成源、既存の重複と死にコード、肥大化、命名、規約違反、変更の波及、依存の複雑さ、テスト容易性 | `docs/quality/refactor.md` を全文読む。無ければ本表の項目で代用。範囲の ADR を先に読む | 観点ファイルの観点名 |
| defect | 所有権・寿命・スレッド境界、未定義動作、例外安全、境界条件、TODO/FIXME のうち実害があるもの | 具体的な失敗シナリオが書けるものだけ | 所有権 / 寿命 / 並行 / 境界 / 例外 |
| architecture | モジュールの深さと seam、レイヤ違反、循環依存、ボイラープレートやアダプタの生成源 | `improve-codebase-architecture` スキルの DEEPENING 語彙があれば使う。CONTEXT.md の用語で書く。範囲の ADR を先に読む | seam / 依存 / 重複生成源 |
| drift | 設計文書 1xx・2xx、CONTEXT.md とコードの不一致。範囲の ADR は (a) 決定と実装の不一致 (b) 後発 ADR で改訂済みなのに現役に見える (c) 設計文書から参照されていない(入口なし) の 3 点を見る。設定キーの 3 集合差(コードが読むキー、設定ファイルのキー、文書のキー) | 文書 1 本を読み、対応コードと突き合わせる | 文書 stale / 未文書 / 死にキー / 用語ずれ / ADR ずれ / ADR 入口なし |
| test-gap | 未テストの prod ファイルに対する特性テストの提案 | `reports/stats/trend.md` の「テスト負債」を入力に、LOC × churn 順、年代 pre 優先 | test-gap |
| backlog | issue の status と AI の判定のずれ（未着手なのに残件無し、など）、`docs/*.txt` の散在メモ | 本文とコードで確認する | 前提崩れ / 実装済み / 重複 |
| process | 未コミット差分の規模と滞留、PRD の課題の status から導いた PRD の WIP、docs 同時更新率、AGENTS.md 規約の遵守(抜き取り) | `reports/stats/summary.md` と `trend.md` | WIP / 滞留 / 規約 |

ガードレール(観点ファイルの運用ルールを引き継ぐ):

- 観点は排他ではなくタグ。複数に当たる候補は最も根本の観点で 1 件にし、他は付随として summary に書く。
- 挙動が変わる修正はリファクタ候補にしない。defect として扱う。
- ADR と矛盾する提案は出さない。ADR が意図と記録している形は候補にしない。当時の前提が今は崩れていると判断した場合だけ、ADR 再訪の候補として ADR 番号と前提が崩れた根拠を書く。
- hot path(Sample、Push、Learn、per-step 経路)や RNG 呼び出し順に触る提案は、対処方針に perf-neutral 確認と同 seed 再現性チェックを含める。
- refactor・architecture では、地図の経緯の ADR・PRD が意図した形と「不採用」の形を候補にしない。前提が崩れた根拠がある場合だけ、出典と根拠を添えて再訪の候補にする。「先送り」「範囲外」は再開条件が揃った根拠を添えれば候補にしてよい。候補の対処方針には地図の「変えるときの条件」を引く。
- wiki の書く部分は編集しない。食い違いは報告する。`wiki_support.py maps` による課題区画の再生成だけは派生一覧として扱う。
- ホットスポット補正: `reports/stats/stats.json` の `flow.churn90` で上位のファイルは `urgency` を 1 段上げ、今後触る予定の無い場所は 1 段下げる。上げ下げの幅と、上げ下げする前の値は `anet-create-issue` の「anet での当てはめ」による。補正したときは、そのことと理由を課題の背景に書く。

## 範囲(scope)の決め方

1. 引数があればそれを使う。
2. 無ければ `reports/registry/rotation.json` を読む。`next_lens` を観点にし、`scopes` のうちその観点の `last_seen` が最も古いもの(未実施が最優先、同点なら churn90 合計が大きい方)を選ぶ。コード観点(refactor、defect、architecture、test-gap)は `kind: code` の scope、drift は `kind: doc`、backlog と process は scope を持たず全体を見る。
3. 実行後、該当 scope の `last_seen[<lens>]` を今日にし、`next_lens` を次に進める。順は refactor → defect → architecture → drift → test-gap → backlog → process → refactor。

## 手順

1. lens と scope を確定する。観点ファイル、設計文書、必要な ADR を読む。refactor・architecture は実装の地図と、選ばなかった形・変えるときの条件も先に読む。
2. コードと文書を読み取りだけで確かめる。ビルドとテストは実行しない。
3. 範囲に area が当たる、次版・解決でない既存 issue（PRD の課題を含む）を本文まで読む。パスの範囲は本文の根拠位置でも絞る。根拠が残るなら「残件有り」、消えたなら「残件無し」、判断できなければ「判定不能」を `anet-update-issue` の「判定」で記録する。status と監理の項目は変えない。新しい根拠は本文へ追記できる。
4. 新しい指摘は `anet-create-issue` の「分け方」で既存との重複を判定し、同じなら追記し、違えば起票する。phase は監査。type は defect→不具合、drift→不整合、refactor・architecture・test-gap→リファクタリング、process→機能改善、backlog→調査。kind は process だけハーネス、ほかは成果物。area は範囲に当たる `categories.json` の key の並びとする。domain は `anet-create-issue` の「各項目の書き方」のとおり area の先頭と本文から選ぶ（process はハーネスなので開発）。
5. 新規は max 以内（既定 8 件）で、WSJF の大きいものから記録する。WSJF の 4 つの値と priority の置き方、本文の構成は `anet-create-issue` に従う。
6. repo の `.venv` から `reports/tools/wiki_support.py maps` を実行して INDEX と地図の課題区画を再生成する。入力の不備で失敗したら成功扱いしない。
7. `rotation.json` の該当 scope の日付と次の観点を更新する。
8. 観点・範囲、読んだ地図、AI の判定別の再検証件数、新規 issue の ID と WSJF、上位 3 件の要旨、次の観点と範囲を報告する。refactor・architecture は、不採用の形を根拠なく候補にした件数と、変えるときの条件を引いた件数も示す。

## issue の品質基準

- 事象にファイル・行・symbol を、影響に具体的な失敗シナリオを書く。推測だけの指摘は書かない。
- 起票時の WSJF の 4 つの値は本文を読んで見立て、priority は中を置く。既存 issue の監理項目は AI の判断で変えない。
- 本文の既存の事実は書き換えず、新しい根拠は日付とともに追記する。

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

- コードを変えない。テストを書かない。PRD を作らない。提案は issue に残す。
- git の書き込み操作をしない。
- 質問して止まらない。観点や範囲の判断は本書の規則で決め、完了報告に書く。
