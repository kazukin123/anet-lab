# grill スキル拡張: goal anchor と簡素化パス PRD

> 番号 056。Topic: 311_開発プロセス / 開発ツール (#7)。
> 状態: 設計確定・**同一変更内で実装済み**(SKILL.md 4 ファイル編集)。本書は変更の記録であり、挿入ブロックの仕様の正本。
> 分担: 設計・SKILL.md 編集=Claude、commit=ユーザー。
> 起点: 059(設定体系再設計 PRD)の検討過程で、grill 型質疑の構造的盲点が実証されたこと。

---

## 1. 背景

開発フロー「チャットで大枠・方針を詰める → grill で具体化して PRD 化」における grill(質問による設計の具体化セッション)には、構造的な弱点が 2 系統ある。

### 1.1 簡素化方向のチェックが漏れる

grill は「この場合どうなるか?」を積む**単調増加のプロセス**であり、次の 6 観点が質問形式から自然には出ない。

| # | 観点 | grill で出ない理由 |
|---|---|---|
| 1 | 総和としての過剰 | 各部分は個別質問に正当化で答えられてしまう。「丸ごと不要では」は具体化質問から出ない |
| 2 | 要件の実在性 | 想像上の将来ニーズも「こう使う」と答えられるため具体化質問を生き残る |
| 3 | 決定の残骸 | grill は前へ進むプロセスで、後続決定によって前提が消えた過去の結論を再訪しない |
| 4 | 最小解との差分 | 採用案は磨かれるが「pin された要件だけから再導出した最小構成との diff」は誰も問わない |
| 5 | 段階の独立価値 | 「全部やる前提」で各段階を詰めると、途中で止めても正味プラスかが問われない |
| 6 | 成功指標の測定可能性 | 仕様は詰まるが「成功したとどう分かるか」は聞き忘れる |

実証例(059 の検討過程): RUN_BUDGET 軸は当初 root 横断素材として設計されたが、後続決定 D11(RUN_BUDGET / LEARNING_SCHEDULE 分離)で横断性の根拠だった `per_beta_step` が別軸へ移り、横断機構(root 持ち上げ)の Phase 0 必須性が消えていた。この「決定の残骸」(観点 3)は具体化質疑では検出されず、逆方向の監査パスで検出された。

### 1.2 長丁場で目的・ゴールを見失う

grill セッションはたいてい長くなり、個別質疑が「何のためだったか」から漂流する。

## 2. 要求

grill セッションを次の 4 要素構成にする。

| 順序 | 要素 | 目的 | 実施条件 |
|---|---|---|---|
| ① | **goal anchor** | 問題・ゴール・非目標を数行で合意し、以後の全判定(簡素化パス含む)の基準点にする。途中で回答がアンカーから逸れたら指摘し「ゴール自体が変わったのか」を確認する | 常に |
| ② | **scope screen** | 入口の足切り。plan が導入する新機構・概念を列挙し、ゴールに照らした実痛 pin(実際に起きた事故・実測されたコスト・現存する workaround)を確認。pin できないものはその場で deferred gate 化または削除し、**その機構への詳細質問をスキップ**する | plan が新機構・概念を導入する場合 |
| ③ | 主質疑(既存のツリー walk) | 具体化 | 常に |
| ④ | **final simplification pass** | 出口の監査。積み上がった決定の総和を、goal anchor に照らして §1.1 の 6 観点で逆方向から検査する。裁定は keep / shrink / defer-behind-gate / cut の 4 値で取り、成果物(PRD 等)に「複雑性監査」として記録する | 常に(最終 YES/NO 質問の前) |

役割分担の根拠: 「決定の残骸」「最小解 diff」は全決定が揃わないと機能しないため終盤専用。「実痛 pin」は序盤が主戦場だが、grill 中に機構が育つため終盤でも再確認する(序盤で足切りしても、その後の質疑で機構は増える)。序盤の足切りがないと、最後に「丸ごと不要」と判明した機構への質疑ラウンドが全て無駄になる。

## 3. 仕様(挿入ブロック)

以下 2 ブロックを、対象 4 ファイル(§4)へ**完全同一のテキスト**で挿入した。言語は既存 SKILL.md に合わせ英語。質問形式は grill 本体と同じ(1 問ずつ・推奨付き)とし、新しい進行様式を増やさない。frontmatter(name / description)は変更しない(トリガ条件不変)。

### 3.1 序盤ブロック

挿入位置: 冒頭段落(`Interview me relentlessly ...`)の直後、`Ask the questions one at a time ...` の前。

```markdown
Open the session with a **goal anchor**: state, in a few lines, the problem being solved, the goal of the plan, and any explicit non-goals, and get my agreement before the first question. Long grill sessions drift; this anchor is the reference that every later judgement — including the simplification passes below — is measured against. If mid-session answers start pulling away from the anchor, say so and ask whether the goal itself has changed.

If the plan introduces new mechanisms or concepts, follow with a **scope screen** before descending into detail: enumerate them, and for each confirm what real pain (an incident, a measured cost, an existing workaround) pins it with respect to the agreed goal. Anything pinned only to an imagined future need is — right there — demoted to a deferred gate ("build when X first hurts") or dropped, and its detail questions are skipped entirely. Do not spend grill rounds specifying something that should not be built.
```

### 3.2 終盤ブロック

挿入位置: `If a question can be answered by exploring the codebase ...` の直後、`At the end of the grilling session, do not automatically ...` の前。

```markdown
Before the closing question, run a **final simplification pass**, measured against the goal anchor agreed at the start. Grilling is a monotonic process — every question adds specification — so finish by interrogating the accumulated design in the opposite direction. Announce the switch, then ask these one at a time like any other grill question, each with your recommended verdict:

1. **Aggregate excess** — re-enumerate every mechanism/concept the plan now introduces, including anything added during this session; for each, ask "if we cut this entirely, which *real* pain returns?" A part can survive every detail question and still not deserve to exist.
2. **Requirement reality** — for requirements added or reshaped during the session: pinned to a pain that actually happened, or to an imagined future need? Demote imagined needs to a deferred gate instead of building now.
3. **Decision residue** — re-check earlier conclusions whose premises were changed by later decisions in this same session (e.g. a mechanism justified by a concern that a later split already resolved).
4. **Minimal-solution diff** — re-derive the smallest design that satisfies only the pinned requirements and the agreed goal, and present its diff against the current plan; every item in the diff must re-justify itself or be shrunk, deferred, or cut.
5. **Phase independence** — does each phase deliver net value on its own, and can the effort stop after any phase without leaving things worse?
6. **Success measurability** — how will we know the claimed benefit materialised? Prefer mechanically checkable indicators (lines to edit for a routine task, recurrence of a named incident, workaround code deleted).

Record the verdicts (keep / shrink / defer-behind-gate / cut) in the session's output artifact — e.g. a "complexity audit" section in the PRD — so future readers can see why something was deliberately not built.
```

## 4. 対象ファイルと同期契約

grill 系スキルは 2 スキル×2 置き場の 4 ファイル。`.agents/skills/` が正本、`.claude/skills/` は Claude Code 用の同内容ミラー(AGENTS.md「LLM コーディング規律」節に記載の構造)。

- `.agents/skills/grill-me/SKILL.md` / `.claude/skills/grill-me/SKILL.md`
- `.agents/skills/grill-with-docs/SKILL.md` / `.claude/skills/grill-with-docs/SKILL.md`(挿入は `<what-to-do>` タグ内の対応位置)

2 スキル間・2 置き場間で挿入ブロックを**完全同一**に保つ。今後ブロックを改訂する場合も 4 ファイル同時に行う。

## 5. 検証

1. 編集後、各スキルの `.agents` 版と `.claude` 版の完全一致を `git diff --no-index` で確認する(本変更で実施済み)。
2. 実地確認: 次回の grill セッション(059 設定体系再設計 PRD の最終化を予定)で、goal anchor → scope screen → 主質疑 → final simplification pass の順に発動し、裁定が複雑性監査として PRD に記録されることを確認する。
3. 運用後、観点の増減・文言調整が必要になったら本 PRD を改訂の正本として更新する。

## 6. Out of Scope

- リポジトリ外の `grilling` スキル(プラグイン由来。リポジトリ管理下にない)。
- `to-prd` / `implement-prd-with-docs` 等、他スキルへの同種パスの展開(必要が実証されたら。本 PRD の scope screen の精神に従い、先回りしない)。
- 簡素化パス自体の自動化・チェックリスト外部ファイル化(SKILL.md 内で完結する現形で開始し、肥大したら分離を検討)。
