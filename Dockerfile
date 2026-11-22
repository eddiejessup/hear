FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOST=0.0.0.0 \
    PORT=8080 \
    HEAR_NO_RELOAD=1 \
    HEAR_DATA_DIR=/data

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=litestream/litestream:0.5 /usr/local/bin/litestream /usr/local/bin/litestream

# Install Python 3.15 and project dependencies
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --python 3.15

COPY config.yaml ./
COPY deploy/litestream.yml /etc/litestream.yml
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /data

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
