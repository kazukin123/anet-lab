---
name: anet-translate-docs
description: Translate anet-lab's Japanese design documents (docs/design/*.jp.md) into English .en.md siblings one file at a time (long files chapter by chapter), tracking the source blob hash in a header comment and keeping CONTEXT.md terminology consistent through reports/translate/glossary.json. Use when the user invokes $anet-translate-docs or /anet-translate-docs, optionally with a file name or max=<n>, or from anet-housekeeping.
---

# ANET Translate Docs

`docs/design` の日本語設計文書を英語にする。`docs/**` は pages.yml で GitHub Pages に配信されるので、訳文はそのまま公開される。訳は構造を変えず、内容を足さない。

## 対象と順序

- 対象は `docs/design/*.jp.md` だけ。出力は同じ場所に `*.en.md`(`010_framework_overview.jp.md` → `010_framework_overview.en.md`)。
- 順序: README, 010, 020, 030, 040, 100, 110, 120, 130, 140, 150, 160, 200, 210, 220。引数でファイル名があればそれを優先する。
- 状態はファイル自身が持つ。`.en.md` の 1 行目に次のコメントを置く。

  ```
  <!-- translated-from: 010_framework_overview.jp.md blob:<sha1> date:2026-09-20 progress:done -->
  ```

  `sha1` は `git hash-object docs/design/<jp ファイル>` の値で、作業ツリーの内容で計算する。`progress` は `done` か `chapter <N>/<M>`。
- 対象の決め方: jp ごとに hash を計算し、en が無ければ「未訳」、en の `progress` が done でなければ「続き」、en の blob が今の hash と違えば「更新あり」。優先は 続き → 未訳 → 更新あり。

## 手順

1. 対象を決める。`max` があればその本数まで、無ければ 1 本。
2. CONTEXT.md の Language 節と `reports/translate/glossary.json` を読む。
3. jp ファイルを全文読む。
4. 訳す。
   - 見出し構造、表、コードブロック、設定キー、識別子、メトリクス tag、パス、コマンドは原文のまま。コードブロックの中は訳さない。
   - 用語は CONTEXT.md の canonical term を使う。canonical term が日本語のものは glossary.json の対応英語を使い、無ければ決めて glossary.json に追記する。1 語 1 訳。
   - 相対リンク `xxx.jp.md` は、対応する `xxx.en.md` が存在すれば `.en.md` へ向け、無ければそのまま残す。
   - 内容を足さない、直さない。原文の誤りや古い記述に気付いたら訳はそのままにし、完了報告に「drift 候補」として書く。
   - 文体は簡潔な技術英語。
5. 1,000 行を超える文書は `## ` の章単位で進める。章ごとに en ファイルへ書き足し、1 行目の `progress` を更新してから次の章へ進む。
6. 検査。全部満たすまで直す。
   - 見出しの数と階層が一致する。
   - コードブロックの数が一致する。
   - 表の行数が一致する。
   - コードブロックの外にひらがな、カタカナ、漢字が残っていない。
   - リンク先のファイルが存在する。
7. 完了報告: 訳したファイルと章、glossary に足した語、drift 候補、残りの未訳と更新あり件数。

## glossary.json

```json
{"terms": {"価値ストリーム": "value stream"}}
```

キーは CONTEXT.md の canonical term(日本語)、値は英訳。決めた訳は変えない。

## 禁止

- jp ファイルを変えない。`docs/design` と `reports/translate/glossary.json` 以外に書かない。
- git の書き込み操作をしない。
- 質問して止まらない。用語の判断は本書の規則で決め、完了報告に書く。
