# 0001: Record architecture decisions

## Context
The project makes non-trivial modeling and engineering choices whose rationale
is easy to lose once the code lands. Reviewers and future maintainers need the
reasoning, not just the result.

## Decision
Record every non-obvious decision as a short ADR under `docs/decisions/`, named
`NNNN-title.md`. Each ADR states the context, the decision, the options that
were rejected, and the consequence. ADRs are numbered sequentially and are
immutable once merged; a later ADR supersedes an earlier one rather than
editing it.

## Rejected options
- No ADRs, rely on commit messages: decisions get buried and lose their
  alternatives and trade-offs.
- A single running design doc: harder to review per-decision and to reference
  from code or reports.

## Consequence
Decisions are reviewable in isolation and linkable from phase reports and the
eventual README.
