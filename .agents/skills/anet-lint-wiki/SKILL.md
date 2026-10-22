---
name: anet-lint-wiki
description: Check and repair anet-lab wiki ingestion gaps, retractions, source links, index coverage and obsolete names; regenerate mechanical map sections. Use when the user invokes $anet-lint-wiki or /anet-lint-wiki, or when anet-housekeeping selects wiki-lint. Accepts max=n, default 3.
---

# Wikiの点検

`$anet-lint-wiki [max=n]`。共通契約は[wikiの規約](../../../docs/wiki/README.md)の「原則と正本」「置き場と書式」「取り込み・点検」。開始前に読む。定常整備の[共通の約束](../README.md)も適用する。

1. maxを検証する。既定3、1以上の整数だけを受け付け、それ以外は何も直さず指定値と期待値を示して止まる。
2. repoの.venvから `reports/tools/wiki_support.py scan` を読み取り専用で実行し、起点・blob・候補を得る。段階3まではstart以後に変わった材料だけを対象にする。
3. 規約の7項目を点検する: 取り込み漏れ、変わったページの撤回の反映漏れ、出典リンク、目録漏れ、孤立ページ、消えた名前、地図の機械的な部分。取り込み漏れは更新ありを先にmax本まで、[取り込み](../anet-ingest-wiki/SKILL.md)と同じ手順で処理する。全部の知見ページを読み直さない。
4. リンクは同じファイル名、次に番号を除いた名前で探す。一意なら修正し、解決できなければ主張末尾に「（出典のリンク切れ）」を付けて報告する。注記済みの件数は毎回報告するが前提条件には数えない。消えた名前の照合は規約の対象だけをcore・appsの現行コードと設定で確認する。旧名を推測で新名へ置き換えない。
5. doneへ移ったPRDは、変更対象と分類・置き場から既存地図の経緯と2節へ取り込む。カテゴリページの無い範囲は作らず、段階3へ残す。地図新設時だけ目印を置いてregistry.py buildを回す。
6. repoの.venvから `reports/tools/wiki_support.py maps` で全地図のファイルとPRD一覧を再生成する。台帳の区画には触れない。設計文書に記述なしのページ名、2節が記述なしのページの数を数える。
7. 目録とページを確認し、ingestとlintの記録（現在revision、修正ページ、未解決）を追記する。規約の機械的な確かめと公開できない値の走査を行う。
8. 点検・修正・取り込みの結果、未解決、注記済みリンク切れの数、記述なしの集計、mapsの `unresolved_prds` にあるPRDを一意に決められないページ・番号・候補数と件数（ページと番号の組の数）を報告する。

正本を書き換えない。gitの書き込みをしない。質問して止まらず、出典どうしの食い違いは「分かっていないこと」へ書く。Plan Modeとhousekeepingのdry-runではscanと調査だけを行い、ページ・ログ・派生ビューを編集しない。
