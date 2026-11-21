#!/bin/sh
set -eu

export HEAR_DATA_DIR="${HEAR_DATA_DIR:-/data}"
DB_PATH="${HEAR_DATA_DIR}/hear.db"
mkdir -p "${HEAR_DATA_DIR}"

if [ -x "/app/.venv/bin/hear" ]; then
  APP_CMD="/app/.venv/bin/hear"
else
  APP_CMD="uv run --no-dev hear"
fi

if [ -n "${HEAR_LITESTREAM_BUCKET:-}" ]; then
  echo "▸ Litestream enabled (replica: gs://${HEAR_LITESTREAM_BUCKET}/hear.db)"
  litestream restore -config /etc/litestream.yml -if-db-not-exists -if-replica-exists "${DB_PATH}"
  exec litestream replicate -config /etc/litestream.yml -exec "${APP_CMD}"
fi

echo "▸ Starting hear without Litestream (HEAR_LITESTREAM_BUCKET not set)"
# shellcheck disable=SC2086
exec ${APP_CMD}
