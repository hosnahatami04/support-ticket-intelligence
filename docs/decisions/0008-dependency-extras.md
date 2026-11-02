# 0008: Four dependency extras instead of three

## Status
Accepted. Supersedes the dependency-group part of
[0002](0002-packaging-and-layout.md); the `uv`, `src/` layout and Python-pin
decisions in 0002 still stand.

## Context
0002 split dependencies into three groups: default runtime, `ml` and `dev`. Two
later phases broke that split.

The data pipeline needs pandas, pyarrow, scikit-learn and datasets. Under the
three-group plan those belonged in `ml`, which also carries torch and
transformers. That made the data tests un-runnable in CI without a multi-hundred
megabyte torch download, for no benefit: nothing in the data pipeline imports
torch.

The service layer then added a third shape of dependency. FastAPI, uvicorn,
streamlit and prometheus-client are needed to serve the model but not to train
it, and not to run a unit test.

Keeping both inside `ml` would have meant the lint/typecheck/test job installing
a web stack and a deep-learning stack to check types.

## Decision
Use four extras, each the smallest set that makes its own tests runnable:

- `data` — numpy, pandas, pyarrow, scikit-learn, scipy, datasets, matplotlib.
- `service` — `ticketintel[data]` plus fastapi, uvicorn, prometheus-client,
  streamlit, httpx, requests.
- `ml` — `ticketintel[data]` plus torch, transformers, sentence-transformers.
- `dev` — pytest, ruff, mypy, stubs and the audit tooling.

`service` and `ml` both depend on `data` rather than restating it, so the
numeric stack is pinned in one place.

CI uses this split directly: the `quality` job installs `dev` only and runs
`pytest -m "not ml and not data"`, while `data-tests` installs
`dev,data,service` and runs `pytest -m "not ml"`. The transformer tier stays
behind the `ml` marker, and every heavy import sits behind
`pytest.importorskip` so a missing extra skips the test rather than erroring the
run.

## Rejected options
- Keep three groups and widen `ml`: makes the data tests depend on torch and the
  type-check job depend on streamlit.
- A single `extras = all` set: the light CI job becomes the slow one.
- Dependency groups (PEP 735) instead of extras: `uv` supports them, but extras
  stay installable with plain `pip install ".[data]"`, which keeps the README
  instructions tool-agnostic.

## Consequence
Four extras to keep in sync instead of three, and a contributor must pick the
right one. In exchange each CI job installs only what it runs, and the
`data`-marked tests run on every push without a torch download.
