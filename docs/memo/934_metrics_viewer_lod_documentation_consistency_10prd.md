# Metrics ViewerのLOD集約数の記述整合 PRD

> 2026-11-09 工程末の「この場で直す」（ユーザーの選択）で、範囲外の不備として設計160 jp §7.1（379 行）の「最後の1つ」を「最後の主pane」に直した（同じ文書の 31・382 行に合わせた。英訳は修正のレビューの指摘で直し済み）。D2 の「ほかの行は変えない」と受入 4 は本 PRD の直しについての決まりで、この直しは含まない。英訳の翻訳元の記録は、この直しのあとの日本語版の hash にした。

> 2026-11-09 再グリル（anet-create-prd、Claude Code。別モデル Fable の検討を含む）。暫定の印を外し、題から「暫定」を外した。§1 を書き直し、節番号の誤り（§2.1・§7.1 → §2.2・§7.2）を直した。範囲に、英語版の設計160（今の日本語版へそろえ、翻訳元の記録も更新する）、ADR 0015 の先頭の注記、wiki の問いを加えた。§7.2 には LOD の集約数（親 1 件に集める子の件数）を書かず、集約数は §2.2 の 1 か所だけに書くことにした。旧「ゴールと範囲」を §2 ゴールと非ゴールに、旧「対応時の確認と受入基準」を §6 受入（8 項目）に置き換え、確定事項（§3、D1〜D8。担当は D7）、直す箇所（§4）、成功の測り方（§5）、未決表（§7、なし）を新設した。

> 2026-11-03 番号を 999 から 934 へ振り直した（PRD114）。

> 2026-11-01: [PRD110 pin対応の実装記録](done/110_runner_main_pane_dynamic_centre_22impl.md)の独立レビューで見つかった範囲外の指摘を、ユーザーの「課題として記録する」の回答により別件として記録した。今回は記録のみで、文書の修正には着手していない。

> 関連: 索引 issue [issue-20261103-55b5f1](../management/issues/issue-20261103-55b5f1.md)、[ADR 0052](../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md)、[ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)、[設計160](../design/160_applications_and_tools.jp.md)、[設計210](../design/210_metrics_viewer.jp.md)

## 1. 問題

2026-11-09 に確かめた現用の文書のうち、次の記述が実装の LOD の集約数と食い違っている。行番号はその日のものである。

| 場所 | 今の記述 | 実装との関係 |
|---|---|---|
| [設計160 jp](../design/160_applications_and_tools.jp.md) §2.2 Metrics Viewer（48行） | 「factor 4 LOD」 | 合っている |
| 同 §7.2 Metrics Viewer（393行） | 「LODは完成した16子だけから作り」 | 食い違う |
| [設計160 en](../design/160_applications_and_tools.en.md) §2.2（49行） | `factor-16 LOD` | 食い違う |
| 同 §7.2（394行） | `LOD consists only of completed sets of 16 children` | 食い違う |
| [ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)（3行） | 「factor16 序数バケット LOD」。先頭に ADR 0052 への注記が無い | 判断は ADR 0052 で改められたが、ADR 0015 から ADR 0052 へたどれない |
| [wiki の表示粒度のページ](../wiki/mechanisms/metrics_resolution.md)「分かっていないこと」（15・16行） | この食い違いを問いとして持つ。その下に余分な「なし」が 1 行ある | 文書を直すと問いが解ける |

実装（2026-11-09 に読み直した）:

- [LodBucket.java:21](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodBucket.java#L21) の `LOD_FACTOR` は `4` である。
- [LodIngestWriter.java:109](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java#L109) の `appendChild` は、同じ level の子が `LOD_FACTOR` 件そろうと親を合成して書く。そろうまでは書かない。
- 同 120〜122 行の `combine` は、子がちょうど `LOD_FACTOR` 件であることを確かめ、違えば例外を投げる。

関係する決定と、すでに合っている文書:

- [ADR 0052](../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md) が、保存 LOD を factor16 から factor4 に改めた。ADR 0052 は「ADR 0015のfactor16のみを更新する」と書く。
- [設計210 jp](../design/210_metrics_viewer.jp.md) の 170・184・262・726 行と、wiki の [Metrics Viewer の地図](../wiki/apps/120_metrics_viewer.md)「概要・責務」は、すでに 4 で書いている。設計210 jp 163 行の「連続する16点」は序数で数えた幅の例（4² = 16）で、食い違いではない。

経緯:

- 2026-11-01、PRD110 の実装記録の独立レビューで範囲外の指摘として見つかり、本 PRD に記録した。そのときの記録は節番号を §2.1・§7.1 としていたが、正しくは §2.2・§7.2 である（§2.1・§7.1 は Runner の節）。
- 日本語版の §2.2 は [PRD095 の実装記録](done/095_llm_wiki_22impl.md)（202行）で factor4 に直したが、§7.2 と英語版は直さなかった。
- 英語版は、[anet-translate-docs](../../.agents/skills/anet-translate-docs/SKILL.md) の規則で、1 行目の `translated-from` に訳元の日本語版の blob を持つ。2026-11-09 の記録は `blob:3d5bf542629044a246d551cc448c1a956fe1bb1e`（2026-11-04）で、そのあとの日本語版の 2 か所の変更（270 行の Runner の View data を作る時機、379 行の Runner の最大化の扱い。どちらもコミット済み）は英語版に訳されていない。`git diff 3d5bf542629044a246d551cc448c1a956fe1bb1e e049b63407960f11e6d6f4ec91aced213aa48b61`（後者は 2026-11-09 の日本語版の blob）の hunk は、`@@ -267,7 +267,7 @@` と `@@ -376,7 +376,7 @@` の 2 つだけである。

## 2. ゴールと非ゴール

ゴール:

- 現用の文書（設計160の日英、wiki）で、LOD の集約数を実装（同じ level の子 4 件で親 1 件）にそろえる。
- ADR 0015 から、factor を改めた ADR 0052 へたどれるようにする。
- 設計160の英語版を今の日本語版にそろえ、翻訳元の記録も合わせる。
- 古い記述が現用の範囲に残っていないことを、検索で確かめる。

非ゴール（退けた案と、それが本当の制約か）:

- `LOD_FACTOR` を 16 に戻して文書に合わせる案 — 退ける。ADR 0052 が測定に基づいて 4 を選んだ。実装と ADR が正であり、本当の制約である。
- LOD の集約の仕方・保存形式・schema を変える案 — 範囲外。本件は説明の整合だけを扱う。制約ではなく別の関心であり、変えるなら別の PRD で扱う。
- 過去の記録の「16」を直す案 — 退ける。対象は done/041・042 の PRD と実装計画、done/089・095・110 の記録、frozen/929、ADR 0015・0052 の本文、`reports/metrics_viewer_resolution_investigation_2026-10-10.md`、wiki の作業の記録（`docs/wiki/log.md`）の既存の行、issue（`docs/management/issues`）の既存の記述である（D5・D7 が足す行は含まない）。[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針は、過去の ADR・履歴資料を当時の記録として残すと定めており、本当の制約である。
- 設計160 jp の 393 行のほかの行を直す案 — 退ける。§2.2 はすでに正しく、ほかの行は LOD の集約数を書いていない。制約ではなく、直す対象が無い（工程末の「この場で直す」で直した 379 行は本 PRD の案ではない。冒頭の改訂行）。
- PRD110 の pin 実装を変える案 — 範囲外。食い違いは pin 対応の前からあり、関係しない。制約ではなく別の関心である。

## 3. 確定事項

| D | 決定 | 理由（依存するもの） |
|---|---|---|
| D1 | 正とする集約数は 4 とする。同じ level の子が 4 件そろうと親 1 件を書き、そろわない末尾は行にしない | §1 の実装（LodBucket.java の 21 行、LodIngestWriter.java の 109・120 行）と ADR 0052 |
| D2 | 設計160 jp §7.2（393行）の「LODは完成した16子だけから作り」を「LODは子がそろった完成bucketだけから作り」に改め、§7.2 には LOD の集約数（親 1 件に集める子の件数）を書かない。ほかの数（`L0` などの識別子に含まれる数字、§7.2 のほかの行の「最大30秒」など）はこの決定の対象外である。設計160 の中で集約数を書くのは §2.2 の「factor 4 LOD」の 1 か所だけにし、§2.2 とほかの行は変えない（工程末の「この場で直す」で直した 379 行は本決定の外。冒頭の改訂行） | D1。同じ集約数を文書の 2 か所に書くと、片方だけ直して食い違う（§1 の経緯の PRD095）。§7.2 の文の役割は、完成した bucket だけを書いて viewport query から独立に持つことで、集約数はその本質ではない。集約数の仕組みの正本は設計210（jp 170・184・726 行）である。2026-11-09 の再グリルで、別モデルの検討の論点を受けてユーザーが選んだ。§7.2 にも「子4件」と書く案は、§7.2 だけで集約数が分かる利点より、集約数の二重化を残す弱点が大きいので退けた |
| D3 | 設計160 en を、D2 を当てたあとの日本語版にそろえる。§2.2（49行）は `factor-4 LOD` に改める。訳元の版（blob `3d5bf542…`）から D2 を当てたあとの日本語版までの差分の hunk すべてについて、当たる英語版の段落を訳し直す（2026-11-09 の時点では日本語版の 270・379・393 行の 3 つ。393 行に当たる英語版は §7.2 の 394 行）。あわせて日英の全文を段落ごとに突き合わせ、hunk の外で日本語版と合わない英語版の段落（49 行のように、記録した訳元の版とも合っていなかったもの）も訳し直す。1 行目の `translated-from` の blob は、この全文の突き合わせを終えたあとに、D2 のあとの日本語版の `git hash-object` の値にし（工程末の「この場で直す」で 379 行を直したあとは、その直しも当てた日本語版の値。冒頭の改訂行）、date を作業日にし、progress は `done` のままとする。訳は anet-translate-docs の手順 4（構造を変えない、内容を足さない、`CONTEXT.md` と `reports/translate/glossary.json` の訳語）に従う | D1・D2。全文を突き合わせるのは、blob を更新すると英語版の全体が今の日本語版の訳だと記録することになり、確かめずにハッシュだけ更新しないという決まり（[実装 品質確認観点](../quality/impl.md)「検証と報告」の規約）に当たるためである。49 行は、訳元の版の日本語版 §2.2 がすでに「factor 4 LOD」だったのに `factor-16` のままで（`git cat-file -p 3d5bf542629044a246d551cc448c1a956fe1bb1e` の 48 行）、hunk だけでは拾えない食い違いがあると分かっている。翻訳元の記録が確かめた範囲を正しく表すという決まり（[実装 品質確認観点](../quality/impl.md)「検証と報告」の規約と項番50）と、作業ツリーの内容で hash を計算するという決まり（anet-translate-docs）の両方に、例外なしで合わせるため。2026-11-09 の再グリルでユーザーが選んだ。2 行だけ直して記録を据え置く案は、項番50の例外を本 PRD に抱えるので退けた（その案の利点だった「別件の未コミットの変更に触れない」は、別件が `2fce286e` でコミットされて無くなった） |
| D4 | ADR 0015 の題（1 行目）と空行（2 行目）の次に、注記 1 行と空行を足す。注記は既存の ADR の先例（[ADR 0010](../adr/0010-actor-priority-mean-q-approx.md) の 3 行など）の `> **後発決定**:` の形とし、保存 LOD の factor16 を ADR 0052 で factor4 に改めたこと、ADR 0052 が改めたのはこの点だけであること、注記を足した日と本 PRD を書き、ADR 0052 へリンクする。本文は変えない | ADR は本文を書き換えず、後の決定で変わったときは古い ADR の先頭に注記を足す決まり（anet-create-prd の Offer ADRs sparingly）。ADR 0052 は「ADR 0015のfactor16のみを更新する」と書くが、ADR 0015 の側に注記が無かった。「そのほかの判断は維持する」とは書かない。ADR 0015 のほかの判断が今も有効かは、ADR 0052 のあとの決定にもよるためである。2026-11-09 の再グリルでユーザーが選んだ |
| D5 | wiki の表示粒度のページの「分かっていないこと」から、本件の問いと余分な「なし」を消し、節の本文を、同じページの「撤回・更新」と同じ「なし。」の 1 行にする。4 子という事実は、このページに足さない。wiki の作業の記録に、材料を本 PRD とする `writeback` の行を 1 行足す。設計160 の直しの取り込みは、点検（anet-lint-wiki）の取り込み漏れの検出に任せる | D2・D3 で出典どうしの食い違いが解けるため。機構のページは仕組みを書かない（[wiki の規約](../wiki/README.md)「置き場と書式」）。ページの直しは本 PRD の決定によるので、記録の種別は PRD づくりの書き戻し（`writeback`）である（同「置き場と書式」の作業の記録の行）。設計160 の blob が変われば、点検が「更新あり」として拾う（同「anet-lint-wiki」の見るもの 1） |
| D6 | 過去の記録（§2 の非ゴールの 3 つ目に挙げたもの）は直さない | [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針 |
| D7 | 担当: 精査（2026-11-09 の再グリル）と修正を、同じ枠（Claude Code）で行う。起票（2026-11-01）は PRD110 の実装の枠（Codex。[PRD110 の実装記録](done/110_runner_main_pane_dynamic_centre_22impl.md) 5 行の Codex への引き継ぎ）で行ったので、起票と精査は分かれている。§6 の受入を確かめた結果（コマンドと結果）は、索引 issue の対応内容の「確かめ」と、修正のレビューの依頼に書く。修正は本 PRD のレビューが OK になったあと、レジストリの「この場で直す」で行う。直した分は成果物キー `impl`、入力元を本 PRD として独立レビューに出し、実装計画は作らない。本 PRD を `docs` で指す索引 issue の対応内容に、直した内容と変更の場所を書き足し、AI の判定を「実装済み」にする | ユーザーの起動時の指示（「修正もこの枠で実施したい」）。直すのはビルドもテストも要らない文書だけで、[レジストリ](../process/registry.md)「この場で直す」の 1 に当たる。枠をまとめる判断は確定事項に「担当」として書く（[memo README](README.md) §工程と担当）。入力元は、本件を本 PRD の受入で確かめるので「なし」にせず、レジストリの `impl` の行の入力元（`prd`）に従う。判定の値は anet-update-issue の「判定」（PRD の実装を終えた索引 issue は「実装済み」）。2026-11-09 の再グリルで、別モデルの検討の反対意見を受けてユーザーが選んだ。「本 PRD の実装を実装計画なしで行う」と書く案は、anet-create-prd が「この場で直す」を除いて実装へ進まないと定めるので退けた。実装の工程（implement-prd-with-docs）へ切り替える案は、文書の数行の直しに実装計画を足して重いので退けた |
| D8 | 索引 issue の `title`・`scope` の「Metrics ViewerのLOD集約数の記述整合 暫定PRD」から「暫定」を外し、本 PRD の題とそろえる。D7 で索引 issue に書き足すときに一緒に直す | 本 PRD の題から「暫定」を外したので、issue の一覧と食い違わないようにするため。`title`・`scope` は anet-update-issue の変える先（状態と監理の項目）に入らず、AI の判断で変える欄ではないので、2026-11-09 の再グリルの未決のまとめの問いで、ユーザーが推奨案を指示した |

## 4. 直す箇所と直さない箇所

| 場所 | 扱い | 決定 |
|---|---|---|
| 設計160 jp 393 行 | 直す | D2 |
| 設計160 jp 48 行とほかの行（工程末の「この場で直す」で直した 379 行は除く。冒頭の改訂行） | 変えない | D2 |
| 設計160 en 1・49・394 行と、日本語版 270・379 行に当たる段落（2026-11-09 の en 272・380 行）。全文の突き合わせで日本語版と合わないと分かった段落 | 直す | D3 |
| ADR 0015 の題の次 | 注記を足す | D4 |
| wiki `mechanisms/metrics_resolution.md` の「分かっていないこと」 | 問いと余分な行を消す | D5 |
| wiki の作業の記録（`docs/wiki/log.md`）の末尾 | 行を足す | D5 |
| 索引 issue の対応内容と判定 | 書き足す | D7 |
| 索引 issue の `title`・`scope` | 「暫定」を外す | D8 |
| 設計210 の日英、wiki の Metrics Viewer の地図、ADR 0052 | 変えない（すでに 4） | D1 |
| 設計210 jp 163 行「連続する16点」 | 変えない（序数で数えた幅の例） | D1 |
| §2 の非ゴールの 3 つ目に挙げた過去の記録 | 変えない | D6 |

## 5. 成功の測り方

§6 の受入 2 の検索が 0 件になること。修正前（2026-11-09）は 4 件（設計160 en の 49・394 行、設計160 jp の 393 行、wiki の表示粒度のページの 15 行）である。

## 6. 受入

受入 1 は D7 の修正に入る前に、ほかの項目は D7 の修正を終えたあとに、リポジトリの根で確かめる。

1. 根拠の再確認（修正に入る前）: `git grep -n -F "LOD_FACTOR = " -- apps/metrics-viewer/src/main` の結果が、`LodBucket.java` の `static final int LOD_FACTOR = 4;` の 1 件だけである。あわせて、`git grep -n -F "LodBucket.LOD_FACTOR" -- apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java` の結果に、`pending.size() < LodBucket.LOD_FACTOR` と `children.size() != LodBucket.LOD_FACTOR` を含む行がある。どちらかが違えば修正に入らず、本 PRD を再グリルする。
2. 古い記述の不在: `git grep -n -i -E "factor[ =-]?16|16 ?子|16 children|sets of 16" -- ':!docs/adr' ':!docs/memo/done' ':!docs/memo/frozen' ':!docs/memo/934_metrics_viewer_lod_documentation_consistency_10prd.md' ':!docs/wiki/log.md' ':!reports' ':!docs/management/issues'` が 0 件である。範囲は追跡しているファイルの全体である。除くものと理由: `docs/adr`・`docs/memo/done`・`docs/memo/frozen`・`docs/wiki/log.md`・`reports`・`docs/management/issues` は記録であり（D6）、issue の対応内容には D7 で直す前後の文言を書くことがある。本 PRD は古い記述を引用する。
3. 日英の記述:
   - 次の 3 つの検索が、それぞれ 1 である。
     - `git grep -c -F "LODは子がそろった完成bucketだけから作り" -- docs/design/160_applications_and_tools.jp.md`
     - `git grep -c -F "factor 4 LOD" -- docs/design/160_applications_and_tools.jp.md`
     - `git grep -c -F "factor-4 LOD" -- docs/design/160_applications_and_tools.en.md`
   - `git grep -n -i -E "[0-9]+ ?child" -- docs/design/160_applications_and_tools.en.md` が 0 件である（LOD の集約数を子の件数として書いた英語の記述が無い。D2。修正前の 2026-11-09 は 394 行の 1 件）。あわせて、設計160 en の §7.2 の `TagStats` で始まる行（日本語版 393 行の訳）が、子がそろった完成 bucket だけから LOD を作ることを書いている。
4. 日本語版の変更の範囲: `git diff HEAD -U0 -- docs/design/160_applications_and_tools.jp.md` の変更が、393 行の 1 行の削除と 1 行の追加だけである（工程末の「この場で直す」で直した 379 行は除く。冒頭の改訂行）。
5. 英語版の翻訳元の記録と訳:
   - 訳し直した範囲: `git cat-file -p 3d5bf542629044a246d551cc448c1a956fe1bb1e` を一時ファイルへ書き出し、`git diff --no-index -U0 <一時ファイル> docs/design/160_applications_and_tools.jp.md` の hunk のそれぞれについて、当たる英語版の段落が D3 で訳し直されている（2026-11-09 の見込みは、日本語版の 270・379・393 行の 3 つ）。
   - 日英の全文（D3）: 設計160 en の、コードブロックの外の各段落が、対応する日本語版の段落の内容を足し引きなく訳している。段落とは、空行で区切られた本文のまとまり、表の 1 行、箇条の 1 項目、見出しの 1 行である。対応のさせ方は、同じ見出しの下で、出てくる順に 1 対 1 とする（下の手順 6 の検査で、見出し・表の行・コードブロックの数が日英で一致することを前提にする）。英語版の識別子・設定キー・パス・コードの書き方は原文のままでよく、訳語は `CONTEXT.md` と `reports/translate/glossary.json` に従う（anet-translate-docs の手順 4）。
   - 日英の全文の突き合わせの結果（訳元と合っていなかった英語版の段落と直した内容。無ければ無い旨）が、索引 issue の対応内容の「確かめ」に書かれている。上の項目を補う記録である。
   - 設計160 en の 1 行目の `blob:` の値が、`git hash-object docs/design/160_applications_and_tools.jp.md` の出力と一致し、`progress:done` で、date が作業日である。
   - 設計160 en の、日本語版 270 行に当たる段落が、View data を作る時機（timer が断面を取得した後の最初の Train event と、前回から 200ms 経った Train event だけ）と、Train View を隠している間と FPS が 0 の間も 200ms ごとの作成が続くことを書いている。
   - 設計160 en の、日本語版 379 行に当たる段落が、menu・toolbar と Eval の Step・再開で pane を開閉するときと Reset Layout では先に最大化を解くこと、最大化 pane の ✕ と pin は wx の restore 経路に任せること、float した pane を dock へ落としたときに既存の 200ms UPDATE_UI で解けたことを拾ってポリシーを当て直すことを書いている。
   - anet-translate-docs の手順 6 の検査がすべて通る: 見出しの数と階層、コードブロックの数、表の行数が日英で一致し、設計160 en のコードブロックの外にひらがな・カタカナ・漢字が無く、リンク先のファイルが存在する（2026-11-09 の修正前の時点で、すべて通っている）。
6. ADR 0015: `git diff HEAD -- docs/adr/0015-metrics-cache-disposable-derivative.md` の変更が、2 行目（空行）の次への注記 1 行と空行の追加だけで、削除行が無い。足した注記は `> **後発決定**:` で始まり、`0052-metrics-display-uses-common-step-bins-and-factor4-storage.md` へのリンクを持つ（PRD 品質確認観点の項番28 と同じ見方）。
7. wiki: `docs/wiki/mechanisms/metrics_resolution.md` の「分かっていないこと」の節の本文が「なし。」の 1 行だけである。`docs/wiki/log.md` の末尾に、本件の書き戻しの行（`writeback`）がある。wiki の規約「書いたあとの機械的な確かめ」のうち共通の項目（リンク先の実在、目録への掲載、見出しの次の戻る導線）が通る。
8. 索引 issue: 本 PRD を `docs` で指す索引 issue の「対応内容」に、直した内容と変更の場所の行がある。`ai_judge` が「実装済み」、`ai_judged` が作業日で、`status` は変わっていない。`title` と `scope` がどちらも「Metrics ViewerのLOD集約数の記述整合 PRD」である（D8）。

## 7. 未決表

なし。
