# 0002: Packaging, environment and project layout

## Status
The dependency-group decision below is superseded by
[0008](0008-dependency-extras.md), which records the four extras the project
actually ships. The `uv`, `src/` layout and Python-pin decisions still stand.

## Context
The repository must be reproducible, keep CI light, and avoid accidental
imports of the package from the working directory during tests.

## Decision
- Manage the environment with `uv` and a locked `uv.lock`; Python is pinned to
  `>=3.11,<3.12`.
- Use a `src/` layout with the package `ticketintel`, so tests import the
  installed package rather than the source tree by accident.
- Split dependencies into three groups: `runtime` (default: config, CLI, schema
  handling), `ml` (torch, transformers, scikit-learn and friends), and `dev`
  (lint, typecheck, test). CI installs only `dev` plus runtime, keeping the
  default workflow fast; heavy ML tests run behind the `ml` marker.

## Rejected options
- Flat layout: invites shadowing the package with the source directory on
  `sys.path`.
- Poetry / pip-tools: `uv` resolves and installs faster and produces a single
  lockfile that also drives CI.
- One dependency set: pulls torch into every CI run for no benefit on the
  lint/typecheck/test path.

## Consequence
`pip install -e ".[dev]"` and `uv sync --extra dev` both yield a working
development environment; ML-heavy work is opt-in.
