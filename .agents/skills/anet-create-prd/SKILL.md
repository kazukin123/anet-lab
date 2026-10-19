---
name: anet-create-prd
description: Create or revise an anet-lab PRD through a docs-aware grilling session with questions, drafting, self-check, and review, then stop without implementation. Use /anet-create-prd or $anet-create-prd with a topic or an existing PRD filename (including a re-grill from a separate session), or when the user wants to grill or stress-test a plan against the project's documented decisions.
---

# PRD の作成・改訂

[プロセス契約](../../../docs/process/protocol.md) の Step 1〜6 に従う入口。anet での対応は [レジストリ](../../../docs/process/registry.md)、自己点検は [PRD 品質確認観点](../../../docs/quality/prd.md) による。プロセス契約とレジストリを先に読み、工程固有の手順だけを本書で補う。PRD のテンプレートは作らない。

引数は題材か既存 PRD のファイル名。既存 PRD は改訂として扱う。成果物は PRD、必要な用語集・ADR と wiki の更新であり、実装へは進まない。

## 1. 対象を決める

新規か改訂かを決める。既存 PRD は `docs/memo/` のサブフォルダも含めて完全一致で探し、本文と引用する ADR を読む。複数一致または未発見なら対象を尋ねる。

最初の応答の前に、`docs/wiki/index.md` を題材の語（別名を含む）で引き、関係する知見のページと、触る部分のアプリケーション・機能カテゴリのページ（実装の地図）を読む。読んだページ名を1行で示す（関係するページが無ければ「なし」）。既存 PRD なら要点提示、新規なら goal anchor を示す応答に置く。出典と食い違う箇所は起草で直すために控える。

### 既存 PRD の要点提示

既存 PRD や題材に対応する設計メモがある場合、最初に1画面程度で次を平易に示す。章ごとの転記ではなく、共有する前提の説明にする。

- 問題、ゴール、非ゴール。
- 設計の形を決める主要な決定と、その根拠の ADR。触る既存部品の働きも1行ずつ示す。
- 未決、暫定値、要確認事項。
- 過去のグリルと改訂時期。主要な決定の前提だけを現行コード・用語集・後続の判断と照合し、ずれが無ければそう示す。
- 今回検討する論点の見込み。

主要な決定の前提の照合には wiki の知見を使い、触る既存部品の働きには実装の地図を使う。wiki はまとめなので、PRD が依存する事実は元の出典を確認して PRD 本文に出典つきで書く。

同じ応答に goal anchor を示し、既存内容への質問に答えてから質疑へ入る。説明と決め直しを混ぜない。

### 再グリルで開き直す条件

確定事項を再び問うのは、前提が変わった、コードや新しい証拠と矛盾した、あとの決定で不要になった、ユーザーが求めた、別モデルが結論に反対した、のいずれかだけである。理由や前提を問うだけの論点は開き直さず、本文を補って示す。

## 2. 質疑

共通理解に達するまで、設計の決定木を枝ごとにたどり、決定どうしの依存を1つずつ解く。ほかの決定が依存する決定から先に扱う。たどった論点は下の「論点の仕分け」で分け、①②④⑤⑥はその区分に従って調査・確かめ方の整理・実装計画や未決表への記録を行う。ユーザーに問う論点は③だけにする。

### goal anchor

解く問題、ゴール、明示された非ゴールを短く示し、合意を得る。以後の判断はこのゴールに照らす。途中で方向がずれた場合は、ゴール自体が変わったかを確認する。

### scope screen

新しい機構・概念ごとに、ゴールへの寄与を確認する。仮説を確かめる PRD では、仮説・比べ方・終わり方を確認する。「今は要らない」を理由に削らない。実装の負荷は AI が持つため、後で要るものは同時に作る。器や抽象は、確定した要求から導ける最小の形で提案する。

先送りできるのは、実験結果が無いと決められない場合と、別の関心なので別 PRD にする場合だけ。先送りの理由と、何が揃ったら作るかを PRD 本文に記録する。

後で要るので同時に作るものは、次の境界で詰める。

| 区分 | 対象 | 決め方 |
|---|---|---|
| 契約 | 外から見える振る舞い・設定キー・公開 API・メトリクス名、既存部品との受け渡し・持ち主・失敗時の挙動など、PRD が設計として決めるもの全部 | 1問ずつ |
| 細部 | 契約の枠の中の値や方式で、推奨の既定が契約や他の決定から出るもの。既定値、閾値、表示・ログ書式など | 契約を詰めたあと、推奨の既定と理由の一覧の扱いを1問で聞く |
| 実装の詳細（⑤） | PRD の契約に効かない関数分割やファイル配置など | 実装計画へ送る |

細部の一覧は `<番号>. <事項>: <推奨の既定>（理由: 1 行）` とする。プロセス契約1.2節の切り上げと同じ形で「推奨で埋める／後で決める／1件ずつ聞く」を問う。切り上げ以外にもこの形を使う読み替えはレジストリにある。答えは PRD に書き、「後で決める」は未決表へ残す。追加の会話なしで実装へ入れる状態にするため、細部を実装計画へ送らない。細部の一覧の1問は「一覧の扱い」という1つの論点を問うもので、別々の判断を並べて各々への回答を要求しない。

クラス構成と名前は細部にも⑤にも入れず、起草のクラス一覧に書く。

### 論点の仕分け

| 区分 | 扱い |
|---|---|
| ① コードや文書で分かる事実 | 調べ、根拠のリンクを添えて示す。聞かない |
| ② 規約・ADR・CONTEXT.md・現行コードから一意に導ける事項 | 聞かずに一覧で示す。導いた根拠の所在を少なくとも1つ示せるものに限る |
| ③ ゴール・優先度・許容する不利益の選択 | 推奨案を添えて1問ずつ聞き、回答を待つ |
| ④ 実験・試作でしか分からない事項 | 確かめ方と、結果ごとの分かれ方を決める |
| ⑤ PRD の契約に効かない実装の詳細 | 実装計画へ送る。クラス構成と名前は除く |
| ⑥ 範囲外 | 999 の暫定 PRD か未決表へ送る。起票はプロセス契約7節に従い、ユーザーが選んだものだけ行う |

①と②の根拠は、人がすぐ開けるリンクで示す。文書は該当節の行、ADR はファイル、コードは行番号付きにする。

②の一覧は、scope screen 後の最初の個別の問いと同じ応答で示す。goal anchor の合意と scope screen の確認はこの問いに数えない。当たる事項が無ければ一覧は出さない。各行は `<番号>. <事項>: <導いた内容>（根拠: <リンク>）` とし、ユーザーが番号を挙げれば③へ戻す。根拠の所在を示せないものは③として聞く。推奨案があることは、一意に導けることではない。実質の選択を「既定値」の名目で②へ入れない。

### 選択肢の示し方

各案を正常・境界・失敗時の具体的な挙動で比較する。該当する API の呼び出し例や設定例（コロン記法）を添え、性能と保守性のどちらで有利かを書く。未共有の概念を名前だけで出さない。用語、具体例、コードとの照合は後述の Domain awareness と During the session に従う。

質問は1問ずつとし、明示的な回答を待つ。無回答や時間の経過を同意と扱わない。コードや文書で解ける疑問は探索して解く。

### 問いを出し直すとき

説明を求められて答えたあと、回答が問いに合っていなかったとき、ほかの論点を挟んで戻るとき、別モデルの検討の論点として聞くときなど、前に出した問いをもう一度出すときも、その応答だけで答えられるように書く。

- 論点、決めると何が変わるか、選択肢ごとの具体的な挙動、性能と保守性の比較、推奨とその理由を、前の応答と同じだけ省かずに示す。
- 「上の表」「前と同じ」「先ほどの A」のように、前の応答を参照させて省かない。チャットが長くなると、ユーザーは前の応答を探すことになる。
- あいだの説明や回答で変わった点（選択肢の増減、推奨の変更、前提の訂正）は反映し、何が変わったかを1行で添える。
- 選択式の問い（質問欄）を使う場合も、各選択肢の説明を、ほかの応答を見なくても意味が分かる文にする。

### 暫定・999 と Plan Mode

「暫定」「999」と指示された起票では質疑を省き、決めきれない点を未決表へ書く。

Plan Mode は前提にしない。Plan Mode で起動された場合は、質疑が終わった時点で決まったことを計画として示し、承認を受けて Plan Mode を抜けてから起草へ進む。モード中は PRD・用語集・ADR・wiki を編集しない。どちらのモードで始めても、最後に実装へ進まない。

## 3. 起草

PRD 本文、必要な `CONTEXT.md` と ADR、および関係する wiki のページを書く。PRD は [memo README](../../../docs/memo/README.md) の工程・採番・リンク規約に従い、会話を知らなくても読めるものにする。作業記録は PRD 本文の確定事項、未決表、冒頭の改訂行が兼ねる。未決の扱いはプロセス契約1節・7節による。

### wiki への書き戻し

グリルで分かった知見を、まず PRD 本文に出典つきで書き、その節を出典として wiki の関係するページへまとめる。wiki だけにある知見を作らず、読んで気づいた食い違いもこの段で直す。暫定・999 の起票（質疑を省く場合）でも行う。[wiki の規約](../../../docs/wiki/README.md)に従い、目録を更新し、書いた側が機械的な確かめと公開できない値の走査を行う。最終 blob の記録は段7で行う。

### 決定の理由と前提

確定事項の各決定に理由を書く。その理由が他の決定や事実に依存するときは、何に依存しているかを名指す。再グリルでの前提の変化と、自己点検での決定の残滓を本文から判定できるようにする。

### 成功の測り方

効果が出たことを確かめる指標を書く。scope screen で先送りしたものには、理由と作る条件を書く。

### クラス一覧

足す・変える・消すクラスがあれば、実装契約の節に、名前・区分（新規／変更／削除）・責務（1行）の一覧を AI が書く。消すだけの PRD でも書く。グリルでは問わない。ユーザーは起草後の問いや実装前に修正でき、実装レビューでも指摘できる。

一覧の直前に必ず次の1行を添える。

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

実装は一覧に従うが、上の許可も PRD 自身の契約である。この扱いを実装スキルや読み替え表へ追加しない。

## 4. 別モデルの検討（任意、Claude Code だけ）

Codex ではこの段を行わない。Claude Code では起草後、自己点検前に、実施するかを1問で聞く。新しい機構・概念か ADR を含む PRD なら「入れる」、含まなければ「入れない」を推奨する。問いには費用の目安を添える。目安は、Fable で約10分（PRD087 の受入の試走での実測。PRD094 の草案1本で、道具の呼び出し75回・出力約3.4万トークン）。PRD の大きさで増減する。

- セッションが Opus なら Fable、Fable なら Opus を、呼び出し時に指定する。
- どちらでもなければ「Fable で入れる」「Opus で入れる」「入れない」から選んでもらう。実施を勧めるときは Fable を先頭にする。
- 入れると答えた場合だけ `anet-other-model` を起動する。依頼は [OTHER-MODEL.md](OTHER-MODEL.md) の雛形で組み、会話の中身は渡さない。定義は [.claude/agents/anet-other-model.md](../../../.claude/agents/anet-other-model.md)。
- 返った論点は6区分で仕分け、③だけを質疑の続きとしてその場で1問ずつ聞く。会話で決めた論点でも、結論への反対なら両方の意見を示して聞く。理由や前提の確認だけなら本文を補い、①として示す。
- 答えを草案へ反映して自己点検へ進む。設計の妥当性を見るこの検討は、観点の充足を見るレビューとも、ユーザーが選ぶ別枠の再グリルとも別であり、置き換えない。

## 5. 自己点検

PRD 品質確認観点の全項目を、各項目の適用範囲に従って通す。加えて次を確認する。

- **目的のずれ**：goal anchor 後に足した機構・要求がゴールに寄与しない、またはゴール自体が動いた場合は、「ゴールを変える／ずれた部分を戻す」を別の1問で聞く。未決をまとめた問いの前に聞き、主要な流れを止める事項の選択肢へ読み替えない。
- **決定の残滓**：後の決定で前提が変わった決定を直し、何を直したかを示す。

自己点検の結果を示す応答には、次の3行を、該当が無い場合も含めて必ずこの順で置く。機械的な検査の結果だけで済ませない。

```text
観点: 全項目を通した（直したもの: …／なし）
目的のずれ: なし（理由を1行）／あり（このあと別の1問で聞く）
決定の残滓: なし／直したもの: …
```

総量の過剰、要件の実在、最小解との差分、段階の独立性を問う段は持たない。成功の測り方は観点13で確認する。

そのあと、残る未決をプロセス契約1節に従って推奨案とともにまとめて1問で聞き、回答を反映して関連観点を再点検する。未決が無ければその旨を示してレビューへ進む。

## 6. レビューと差し戻し対応

レビューは必ず独立したコンテキストの `an-review` で受ける。依頼の組み立てと起動・続けての依頼は [anet-review](../anet-review/SKILL.md) の「2. 依頼を組み立てる」「3. 呼ぶ」を参照する。同じ手順を書き写さない。応答原文の保存は同スキルの「原文を保存する」に従う。

レビュー専用入口の「成果物を直さず停止する」は、この工程の差し戻し対応へ適用しない。プロセス契約 Step 4・5、1.3節に従い、指摘を仕分け、自分で直せるものをまとめて直してから同じレビューエージェントへ出し直す。往復のたびにユーザーへ返さない。判断待ち・停止、レビュー後に残る未決への質問はプロセス契約による。

## 7. 未決を示して停止する

完了条件は、対象の PRD と関連文書の起草・自己点検が済み、レビューが OK になり、残る未決とその扱いが記録されていること。

プロセス契約 Step 6・6節・7節に従い、未決の起票候補を示して停止し、ユーザーが選んだものだけ記録する。候補が無ければ「なし」と示して完了報告へ進む。完了報告は同3.2節・3.3節に従い平易に書く。次の入口はレジストリから示すが、実装は始めない。

起票の候補を記録したあと（未決表と「未解決」の ID への置き換えを含む）、完了報告の直前に、書き戻した内容とその時点の PRD 本文を照合し、食い違えば wiki を直す。候補が無ければ「なし」を示したあとに行う。その時点の PRD を `git hash-object` で計算し、`docs/wiki/log.md` に `writeback` の行を、是正があれば `fix` の行も足す（直したページが無ければ `→ なし`）。暫定・999 でも行う。途中停止で記録が残らなかったものは、`anet-lint-wiki` が入ったら取り込み漏れとして拾う。Plan Mode の間は記録も編集しない。

<supporting-info>

## Domain awareness

During codebase exploration, also look for existing documentation:

### File structure

Most repos have a single context:

```
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-event-sourced-orders.md
│       └── 0002-postgres-for-write-model.md
└── src/
```

If a `CONTEXT-MAP.md` exists at the root, the repo has multiple contexts. The map points to where each one lives:

```
/
├── CONTEXT-MAP.md
├── docs/
│   └── adr/                          ← system-wide decisions
├── src/
│   ├── ordering/
│   │   ├── CONTEXT.md
│   │   └── docs/adr/                 ← context-specific decisions
│   └── billing/
│       ├── CONTEXT.md
│       └── docs/adr/
```

Create files lazily — only when you have something to write. If no `CONTEXT.md` exists, create one when the first term is resolved. If no `docs/adr/` exists, create it when the first ADR is needed.

## During the session

### Challenge against the glossary

When the user uses a term that conflicts with the existing language in `CONTEXT.md`, call it out immediately. "Your glossary defines 'cancellation' as X, but you seem to mean Y — which is it?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term. "You're saying 'account' — do you mean the Customer or the User? Those are different things."

### Discuss concrete scenarios

When domain relationships are being discussed, stress-test them with specific scenarios. Invent scenarios that probe edge cases and force the user to be precise about the boundaries between concepts.

### Cross-reference with code

When the user states how something works, check whether the code agrees. If you find a contradiction, surface it: "Your code cancels entire Orders, but you just said partial cancellation is possible — which is right?"

### Update CONTEXT.md inline

When a term is resolved, update `CONTEXT.md` right there. Don't batch these up — capture them as they happen. Use the format in [CONTEXT-FORMAT.md](./CONTEXT-FORMAT.md).

`CONTEXT.md` should be totally devoid of implementation details. Do not treat `CONTEXT.md` as a spec, a scratch pad, or a repository for implementation decisions. It is a glossary and nothing else.

Keep each definition to one or two sentences. If a term needs a paragraph to pin down, it is a spec and not a term — put the detail in the design doc and leave the glossary entry short enough to remember.

### Offer ADRs sparingly

Only offer to create an ADR when all three are true:

1. **Hard to reverse** — the cost of changing your mind later is meaningful
2. **Surprising without context** — a future reader will wonder "why did they do it this way?"
3. **The result of a real trade-off** — there were genuine alternatives and you picked one for specific reasons

If any of the three is missing, skip the ADR. Use the format in [ADR-FORMAT.md](./ADR-FORMAT.md).

**Keep it short.** An ADR is a title plus 1-3 sentences. `Status`, `Considered Options` and `Consequences` are optional — most ADRs do not need them. Do not mirror the PRD: the PRD says what to build, the ADR says why this shape was chosen and what was rejected. An ADR long enough to be skimmed instead of read has failed at its only job.

**Never rewrite an existing ADR.** An ADR records the judgement made at that time; rewriting it destroys the reason the decision looked right then. When a later decision changes one, append a short note at the top of the old ADR — which ADR revised it, what was replaced, what still stands — and then write the new one. Partial revision is the normal case, so always say which parts survive.

</supporting-info>
