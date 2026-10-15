---
name: grill-with-docs
description: Grilling session that challenges your plan against the existing domain model, sharpens terminology, and updates documentation (CONTEXT.md, ADRs) inline as decisions crystallise. When the plan already exists as a document (a PRD, design memo), it opens with a digest of that document so the grill starts from a shared reading of it. Use when user wants to stress-test a plan against their project's language and documented decisions.
---

<what-to-do>

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

If the plan already exists as a document — a PRD, a design memo, an issue, or any file the user names or that the conversation clearly points at — start with an **existing-document digest** before anything else. Most grills in a repo start this way, and a grill run on a half-read document only re-asks what the document already settled. Read the document in full, then follow its references far enough to understand it: the documents it links, the `CONTEXT.md` terms it uses, the ADRs it cites, and the code paths it names. Then present the digest to the user and wait for their response before the goal anchor. The digest is for the user as much as for you: it shows what you took from the document, so they can correct your reading before it steers the questions. Keep it to roughly 15–25 lines, in the language the session is being held in, and cover:

- **What the document is** — path, kind (PRD, memo, …), and its own status line and revision notes if it carries them (e.g. "暫定メモ, 案 A〜D 未確定").
- **Problem, goal, non-goals** as the document states them — these become the draft goal anchor below.
- **What it already decided** — the choices it has fixed, each with the section heading it lives under. These are not re-asked unless they conflict with code, `CONTEXT.md`, or a later decision.
- **What it leaves open** — the options it enumerates without choosing, the decision points marked undecided, the questions it defers "to the grill". These are the first grill rounds.
- **Facts it asserts about the current code**, with a one-line verdict each from a quick check against the codebase: holds / has drifted (say how) / could not verify.
- **Related documents and code** it depends on or supersedes, so later questions can cite them.

Close the digest by asking whether the reading is right and whether anything has changed since the document was written. Do not start questioning until the user confirms or corrects it. If the user corrects the digest, restate the corrected part before moving on.

When the plan exists only in the user's head or in the conversation, skip the digest and begin with the goal anchor.

Next, set a **goal anchor** (it opens the session when there was no document to digest): state, in a few lines, the problem being solved, the goal of the plan, and any explicit non-goals, and get my agreement before the first question. When a digest was presented, derive the anchor from the digest's problem / goal / non-goals and say so; the user is confirming that the document's stated goal is still the goal. Long grill sessions drift; this anchor is the reference that every later judgement — including the simplification passes below — is measured against. If mid-session answers start pulling away from the anchor, say so and ask whether the goal itself has changed.

If the plan introduces new mechanisms or concepts, follow with a **scope screen** before descending into detail: enumerate them, and for each confirm what real pain (an incident, a measured cost, an existing workaround) pins it with respect to the agreed goal. Anything pinned only to an imagined future need is — right there — demoted to a deferred gate ("build when X first hurts") or dropped, and its detail questions are skipped entirely. Do not spend grill rounds specifying something that should not be built.

Ask the questions one at a time, waiting for feedback on each question before continuing.

If a question can be answered by exploring the codebase, explore the codebase instead.

Before the closing question, run a **final simplification pass**, measured against the goal anchor agreed at the start. Grilling is a monotonic process — every question adds specification — so finish by interrogating the accumulated design in the opposite direction. Announce the switch, then ask these one at a time like any other grill question, each with your recommended verdict:

1. **Aggregate excess** — re-enumerate every mechanism/concept the plan now introduces, including anything added during this session; for each, ask "if we cut this entirely, which *real* pain returns?" A part can survive every detail question and still not deserve to exist.
2. **Requirement reality** — for requirements added or reshaped during the session: pinned to a pain that actually happened, or to an imagined future need? Demote imagined needs to a deferred gate instead of building now.
3. **Decision residue** — re-check earlier conclusions whose premises were changed by later decisions in this same session (e.g. a mechanism justified by a concern that a later split already resolved).
4. **Minimal-solution diff** — re-derive the smallest design that satisfies only the pinned requirements and the agreed goal, and present its diff against the current plan; every item in the diff must re-justify itself or be shrunk, deferred, or cut.
5. **Phase independence** — does each phase deliver net value on its own, and can the effort stop after any phase without leaving things worse?
6. **Success measurability** — how will we know the claimed benefit materialised? Prefer mechanically checkable indicators (lines to edit for a routine task, recurrence of a named incident, workaround code deleted).

Record the verdicts (keep / shrink / defer-behind-gate / cut) in the session's output artifact — e.g. a "complexity audit" section in the PRD — so future readers can see why something was deliberately not built.

At the end of the grilling session, do not automatically request plan approval or start implementation.

If the current collaboration mode is Plan Mode, ask one final YES/NO question asking whether to request approval for the agreed plan. Only if the user selects or answers YES, present the plan for approval in a `<proposed_plan>` block. If the user selects or answers NO, gives an ambiguous answer, or does not answer, do not request plan approval.

If the current collaboration mode is not Plan Mode, ask one final YES/NO question asking whether to execute the agreed plan. Only if the user selects or answers YES, start implementation. If the user selects or answers NO, gives an ambiguous answer, or does not answer, do not edit files or run implementation commands.

</what-to-do>

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
