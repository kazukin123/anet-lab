---
name: anet-ingest-wiki
description: Ingest a source document into anet-lab wiki pages with citations, retractions, index and blob records. Use when the user invokes $anet-ingest-wiki or /anet-ingest-wiki with an optional material path, or after an authorized Run-analysis writeback. Without a path, use only materials written or changed in this conversation.
---

# Wikiへの取り込み

`$anet-ingest-wiki [材料のパス]`。共通契約は[wikiの規約](../../../docs/wiki/README.md)の「原則と正本」「置き場と書式」「取り込み・点検」。開始前に読む。

1. 対象を決め、存在と材料の集合への所属を検証する。引数なしなら、この会話で書いた・直した材料を会話とstaged・unstaged・untrackedの差分から決める。作業ツリーにあるだけの別作業の材料は選ばない。不正なパスは何も直さず、指定パスと材料の集合を示して止まる。対象が無ければ「この会話の材料なし」と報告して終わる。
2. 正本を全文読み、要約時の blob を持つサマリを書く。主要な概念を挙げ、当たるページをすべて読んで既存の項目と比べ、規約に従って主張・撤回・食い違いとトピック別まとめの書く部分を更新する。全文は主のページ、ほかの該当ページには要旨と出典を書く。横断的知見を見直し、既存の見立てに支える事実・合わない事実を足す。新しい見立てを作るのは点検である。知見ページではRFDの除外節を守る。トピック別まとめの2節だけは経緯に載せたRFD全文から中身で拾い、採った決定と新機能の合否を写さない。scanのpending行にimplementedがあるRFDは、実装後の内容をトピック別まとめへ取り込む。ingest/writebackのblobの直後へ `implemented:YYYY-MM-DD`（scanが示した日）を添える。取り込み済みかはパスとblobで突き合わせ、日付は使わないので、同じ本文の取り込みを繰り返さない。RFDの実装済みは、そのRFDを`rfd`で指す課題の判定と状態からscanが導く（解決の課題は済みとして数え、次版・対応しない・不要になったの課題が残るRFDは実装済みにしない）。
3. トピック別まとめを新設した場合は課題の目印（issues:begin / issues:end）を置き、repoの.venvからwiki_support.py mapsで機械的な節とINDEX・課題区画を生成する。課題区画を手で直さない。既存トピック別まとめの機械的な2節は点検に委ねる。
4. 目録とingestのblob記録を更新し、記録にはサマリと直した概念のページを列挙する。規約の機械的な確かめと公開できない値の走査を行う。
5. 直したページ、足した主張と撤回、分かっていないことに書いたものを報告する。

正本を書き換えない。gitの書き込みをしない。質問して止まらず、決めきれないことは「分かっていないこと」へ書く。Plan Modeでは調査だけを行い、ページ・ログ・派生ビューを編集しない。
