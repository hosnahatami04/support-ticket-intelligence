# Build stage: resolve dependencies into a virtualenv that the runtime copies.
FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd AS build

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv venv "$VIRTUAL_ENV" \
 && uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache ".[service]"

# Runtime stage: only the virtualenv and the application, run as a non-root user.
FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

RUN useradd --create-home --uid 10001 ticketintel
WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts

USER ticketintel
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "ticketintel.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
