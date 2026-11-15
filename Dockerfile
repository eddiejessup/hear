FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOST=0.0.0.0 \
    PORT=8080 \
    HEAR_NO_RELOAD=1

# Install Python 3.15 and project dependencies
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --python 3.15

COPY config.yaml ./

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "hear"]
