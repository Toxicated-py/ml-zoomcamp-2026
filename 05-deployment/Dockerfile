FROM python:3.14.6-slim-bookworm

# RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked --no-install-project

ENV PATH="/code/.venv/bin:$PATH"

COPY "predict.py" "model.bin" ./

EXPOSE 9696

ENTRYPOINT ["/app/.venv/bin/uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]