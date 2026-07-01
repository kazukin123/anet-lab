---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Ask the questions one at a time.

If a question can be answered by exploring the codebase, explore the codebase instead.

At the end of the grilling session, do not automatically request plan approval or start implementation.

If the current collaboration mode is Plan Mode, ask one final YES/NO question asking whether to request approval for the agreed plan. Only if the user selects or answers YES, present the plan for approval in a `<proposed_plan>` block. If the user selects or answers NO, gives an ambiguous answer, or does not answer, do not request plan approval.

If the current collaboration mode is not Plan Mode, ask one final YES/NO question asking whether to execute the agreed plan. Only if the user selects or answers YES, start implementation. If the user selects or answers NO, gives an ambiguous answer, or does not answer, do not edit files or run implementation commands.
