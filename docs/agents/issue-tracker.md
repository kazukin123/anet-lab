# Issue Tracker: Local Markdown

Implementation issues for this repo live as Markdown files under `.scratch/`.
PRDs and implementation plans remain in `docs/memo/` as the project's existing
documentation source of truth.

## Artifact Responsibilities

- A PRD defines the requirements and design contract for one change. It lives at
  `docs/memo/NNN_<slug>_10prd.md`.
- An implementation plan defines the overall implementation approach, order, and
  validation for its PRD. It lives beside the PRD as
  `docs/memo/NNN_<slug>_2ximpl.md`.
- `docs/memo/` carries the state of each PRD as a subfolder: active PRDs sit
  directly under `docs/memo/`, while completed, frozen, and dropped ones move
  into `done/`, `frozen/`, and `dropped/`. Filenames stay unique across all of
  them, so resolving a PRD by exact filename still works, but the search must
  recurse into the subfolders. See `docs/memo/README.md`.
- An implementation issue is an independently claimable execution unit with its
  own status and dependencies. It lives at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`.

Implementation issues reference their source PRD and implementation plan when
those artifacts exist. Do not copy their bodies into `.scratch/`.

## PRDs and Implementation Plans

- `to-prd` creates only the corresponding `docs/memo/*_10prd.md`. It does not
  create a tracking issue under `.scratch/`.
- Use the established formal number when one is known. Use `999` for an
  unnumbered backlog draft.
- `implement-prd-with-docs` creates the next available implementation-plan slot
  beside the PRD: `20impl`, then `21impl`, and so on.
- Never overwrite an existing PRD or implementation-plan slot.
- PRDs and implementation plans do not carry the local issue `Status:` field.

## Implementation Issues

- Create implementation issues only when the user explicitly asks to use
  `to-issues` or otherwise requests ticket decomposition.
- Use one directory per implementation effort:
  `.scratch/<feature-slug>/issues/`.
- Store one issue per file, numbered from `01`; do not create a combined ticket
  file.
- Put a `Status:` line near the top of each issue file using the values in
  `triage-labels.md`.
- Record dependencies as `Blocked by: NN, NN`.
- Append conversation history under a `## Comments` heading.
- When an issue is resolved, append the result under `## Answer` and set
  `Status: resolved`.

## Skill Routing

When a skill says "publish to the issue tracker":

- For a PRD or spec, create the canonical `docs/memo/*_10prd.md`.
- For an implementation issue, create a file under
  `.scratch/<feature-slug>/issues/`.
- Keep each artifact in only one canonical location.

When a skill says "fetch the relevant ticket", read the referenced `.scratch/`
issue file. When the request names a PRD or implementation plan, resolve it
under `docs/memo/` by exact path or exact filename.

## Wayfinding Operations

Used by `wayfinder`. The map is a file with one child file per ticket.

- **Map**: `.scratch/<effort>/map.md`.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`.
- **Blocking**: record `Blocked by: NN, NN` near the top.
- **Frontier**: choose the first numbered open, unblocked, and unclaimed ticket.
- **Claim**: set `Status: claimed` before starting work.
- **Resolve**: append `## Answer`, set `Status: resolved`, and add a context
  pointer to the map.
