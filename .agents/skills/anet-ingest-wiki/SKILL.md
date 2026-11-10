---
name: anet-ingest-wiki
description: Ingest a source document into anet-lab wiki pages with citations, retractions, index and blob records. Use when the user invokes $anet-ingest-wiki or /anet-ingest-wiki with an optional material path, or after an authorized Run-analysis writeback. Without a path, use only materials written or changed in this conversation.
---

# Wikiへの取り込み

`$anet-ingest-wiki [材料のパス]`。共通契約は[wikiの規約](../../../docs/wiki/README.md)の「原則と正本」「置き場と書式」「取り込み・点検」。開始前に読む。

1. 対象を決め、存在と材料の集合への所属を検証する。引数なしなら、この会話で書いた・直した材料を会話とstaged・unstaged・untrackedの差分から決める。作業ツリーにあるだけの別作業の材料は選ばない。不正なパスは何も直さず、指定パスと材料の集合を示して止まる。対象が無ければ「この会話の材料なし」と報告して終わる。
2. 正本と関係するページを読み、規約に従って主張・撤回・食い違いと実装の地図の書く部分を更新する。知見ページではPRDの除外節を守る。地図の2節だけは経緯に載せたPRD全文から中身で拾い、採った決定と新機能の合否を写さない。scanのpending行にimplementedがあるPRDは、実装後の内容を地図へ取り込む。ingest/writebackのblobの直後へ `implemented:YYYY-MM-DD`（scanが示した日）を添える。取り込み済みかはパスとblobで突き合わせ、日付は使わないので、同じ本文の取り込みを繰り返さない。PRDの実装済みは、そのPRDを`prd`で指す課題の判定と状態からscanが導く（解決の課題は済みとして数え、次版・対応しない・不要になったの課題が残るPRDは実装済みにしない）。
3. 地図を新設した場合は課題の目印（issues:begin / issues:end）を置き、repoの.venvからwiki_support.py mapsで機械的な節とINDEX・課題区画を生成する。課題区画を手で直さない。既存地図の機械的な2節は点検に委ねる。
4. 目録とingestのblob記録を更新し、規約の機械的な確かめと公開できない値の走査を行う。
5. 直したページ、足した主張と撤回、分かっていないことに書いたものを報告する。

正本を書き換えない。gitの書き込みをしない。質問して止まらず、決めきれないことは「分かっていないこと」へ書く。Plan Modeでは調査だけを行い、ページ・ログ・派生ビューを編集しない。
