---
name: anet-survey-queue
description: Run the next item in anet-lab's survey queue (reports/INDEX.md) with the survey or paper-details skill, save the report under reports/, and move the item to the done list. Every queued item is runnable. Use when the user invokes $anet-survey-queue or /anet-survey-queue, optionally with a topic, or from anet-housekeeping when the queue is not empty.
---

# ANET Survey Queue

実験に関連する論文・文献サーベイをキューから 1 件ずつ回す。調査そのものは `survey` スキル(テーマ横断)か `paper-details` スキル(1 論文)が行い、このスキルはキューの選択と INDEX の更新だけを担う。

## INDEX の形(`reports/INDEX.md`)

- `## 済み`: 表。列は 日付 | ファイル | テーマ | 関連。
- `## キュー`: 表。列は 優先 | 種別 | テーマ | 関連 | 論点メモ。
  - キューの行はすべて実行対象。優先の数が小さいものから回す。
  - 種別は `survey`(テーマ横断)か `paper`(1 論文)。
  - キューへの追加は人が行う。このスキルはキューに項目を足さない。

## 手順

1. 引数でテーマがあればそれ。無ければキューのうち優先の数が最も小さい 1 件。キューが空なら「キューが空」と報告して終わる。
2. 種別に応じて `survey` または `paper-details` スキルの `SKILL.md` を探し、全文読む。見つからなければ次を返して終わる。

   ```text
   ERROR: anet-survey-queue requires the "survey" (or "paper-details") skill, but it is unavailable. No artifacts were created.
   ```

3. そのスキルの契約(`documenting-with-sources`、`writing-quotation` を含む)に従って調査し、`reports/<slug>_survey_<YYYY-MM-DD>.md`(paper は `reports/<slug>_paper_<YYYY-MM-DD>.md`)に書く。既存 reports の様式(`# Survey: ...`、Date、Scope、Table of Contents)に揃える。論点メモに書かれた問いに必ず答える。
4. `reports/INDEX.md` を更新する。キューから行を消し、済みに 日付 | ファイル | テーマ | 関連 を追加する。
5. 完了報告: テーマ、出力パス、主要な結論 3 行、関連 RFD へ反映すべき点(あれば。RFD 自体は変えない)。

## 禁止

- RFD と実験記録を変えない。キューに項目を足さない。
- git の書き込み操作をしない。
- 質問して止まらない。
