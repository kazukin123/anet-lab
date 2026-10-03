---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Open the session with a **goal anchor**: state, in a few lines, the problem being solved, the goal of the plan, and any explicit non-goals, and get my agreement before the first question. Long grill sessions drift; this anchor is the reference that every later judgement — including the simplification passes below — is measured against. If mid-session answers start pulling away from the anchor, say so and ask whether the goal itself has changed.

If the plan introduces new mechanisms or concepts, follow with a **scope screen** before descending into detail: enumerate them, and for each confirm what real pain (an incident, a measured cost, an existing workaround) pins it with respect to the agreed goal. Anything pinned only to an imagined future need is — right there — demoted to a deferred gate ("build when X first hurts") or dropped, and its detail questions are skipped entirely. Do not spend grill rounds specifying something that should not be built.

Ask the questions one at a time.

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
