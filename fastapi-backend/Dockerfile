FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

CMD ["uv", "run", "python", "-Xfrozen_modules=off", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]