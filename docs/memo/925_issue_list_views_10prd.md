# issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表）暫定 PRD

> 2026-11-03 番号を 999 から 925 へ振り直した（PRD114）。

> 2026-11-03 起票（暫定。質疑を省き、既定を置いて、決めることを §6 に残した）。
> 起点: [PRD114](114_findings_intake_and_fix_prd_10prd.md) の再グリル（2026-11-02〜03）で、課題の置き場を issue（1 課題 1 ファイル）に替えるとき、ユーザーが一覧の見せ方について「ここはこだわりたい。依存関係をツリーで見えるようにしたり、何らかの形で Spreadsheet で見れるようにしたい。一覧は HTML でもいいのかもだけど見やすく工夫したい」と述べ、場合によっては一覧の skill にそうした機能を付ける案も挙げた。PRD114 は、見せ方は別の関心で試作で見比べないと決めにくいとして、最小の一覧だけを作り、本書へ分けた（PRD114 §8）。
> 関連: [PRD114](114_findings_intake_and_fix_prd_10prd.md)（issue の形、`area`・`type`・AI の判定の欄・`docs`、最小の一覧）、[ADR 0068](../adr/0068-work-state-and-order-live-in-issues-prd-stays-an-instruction-document.md)、[skills README](../../.agents/skills/README.md)。

## 1. 目的と問題

issue の一覧を、人が開いて眺め、次に何をするかを決めやすい形で見せる。

- PRD114 のあと、すべての仕事（PRD の索引の issue を含む）が issue になり、件数は 100 件を超える見込みである（台帳から 44 件、直下の PRD の索引の issue 64 件、以後の工程末の分）。
- 一覧の skill（`anet-list-issues`）は会話の中で表を出すが、開けばいつでも見られるファイルは、PRD114 の最小の一覧（`docs/management/issues/INDEX.md` と wiki の地図の「課題」の区画）だけである。
- 親子（`parent`）と依存（`depends`）の関係は、表では読み取りにくい。
- 表計算のソフトで並べ替え・絞り込み・書き込みをしながら見たい場面がある。

## 2. 現状（PRD114 の実装のあと）

1. issue は `docs/management/issues/issue-YYYYMMDD-xxxxxx.md` に 1 課題 1 ファイルで、frontmatter に状態・監理の項目（`priority`・`depends`・`parent`・WSJF）・`area`・`type`・`ai_judge`・`docs` を持つ（PRD114 §5.1）。
2. 一覧の skill は、依存と WSJF の順に並べ、`area`・`type` で絞り、「束ね」と「棚卸し」で提案する（PRD114 D12・§5.1）。
3. 最小の一覧は、`reports/tools/wiki_support.py` が `INDEX.md`（一覧の skill と同じ並び）と wiki の地図の区画を作り直す（PRD114 D21）。
4. 課題の skill と発番は、anet の skill と文書として持つ（PRD114 D2）。
5. `docs/` 配下は GitHub Pages に配信される（skills README の anet- スキル共通の約束）。

## 3. ゴール・非ゴール

- G1: 依存と親子を、木や図で見られる。
- G2: 一覧を、表計算のソフトで開いて並べ替え・絞り込みができる。
- G3: 一覧の表を、状態・優先度・WSJF・`area`・`type`・AI の判定で絞り込み、並べ替えられる見やすい形で見られる。
- 非ゴール: issue の記録を一覧の道具から書き換えること（一覧は派生物で、記録を変えるのは課題の skill だけ）。

## 4. 確定事項（暫定の既定）

本節の各行は暫定の既定で、決定ではない（暫定の起票）。再グリルですべて見直す。

| # | 既定 | 理由 |
|---|---|---|
| D1 | 見せ方は 4 つにする。表（HTML。絞り込み・並べ替え・各行から issue を開ける）、親子の木（折り畳める）、依存の図（Mermaid。GitHub の上でもそのまま描ける）、スプレッドシート（CSV） | 起点のユーザーの要望を、それぞれ 1 つの形に当てた |
| D2 | 出力を作る道具は、発番と同じく Node.js で `docs/process/scripts/` に置く | 発番の道具と同じ置き場と言語に揃える（PRD114 の実装で、発番の道具が Node.js でここに置かれる） |
| D3 | 一覧の skill に、引数「ツリー」（会話の中で文字の木を出す）と「書き出し」（道具を呼んでファイルを作る）を足す | ユーザーが「一覧の skill に機能を付ける」案を挙げた。会話の中で見る形と、開いて見るファイルの両方を、同じ入口から出す |
| D4 | 出力は `docs/management/issues/` に置く派生物（`index.html`・`tree.md`・`issues.csv`）とし、冒頭に「道具が作り直す。手で直さない」と書く。PRD114 の `INDEX.md` は `index.html` と並べて残す | 正本（issue）の隣に置くと、リンクが短く済む |
| D5 | wiki の地図の「課題」の区画は、PRD114 の anet 側の道具（`wiki_support.py`）のままにする | 地図は anet にしか無い |

## 5. 実装契約

（再グリルで決める。§6 の未決を決めてから書く）

## 6. 未決

| # | 事項 | 今の既定 | 決めること |
|---|---|---|---|
| 1 | 表の見た目 | 絞り込みと並べ替えのできる 1 枚の HTML | 試作で 2〜3 案を見比べて決める（列、色、行の折り畳み、WSJF の引き継ぎの見せ方） |
| 2 | スプレッドシートの形 | CSV | CSV で足りるか、xlsx か、Google スプレッドシートへ送るか（送るなら外部への書き込みになる） |
| 3 | 道具の置き場 | Node.js で `docs/process/scripts/`（発番と同じ） | Python で `reports/tools/`（`wiki_support.py` と同じ）に置く形にするか |
| 4 | 更新のきっかけ | 一覧の skill の「書き出し」を呼んだとき | housekeeping の候補にするか、課題の skill が書くたびに作り直すか |
| 5 | GitHub Pages | `docs/` 配下なので配信される | 一覧の HTML を Pages で見せたいか、配信から外すか |
| 6 | 依存の図の大きさ | すべての未解決の issue を 1 枚に描く | 件数が多いとき、`area` ごとに分けるか |

## 7. 依存と順序

- PRD114 の実装（issue の形、課題の skill、最小の一覧）が先に要る。
- 再グリルの前に、PRD114 の最小の一覧を使って、足りない点を確かめるとよい。
