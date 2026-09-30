#!/usr/bin/env bash
# CPU tests in every env. Extra pytest args are passed through (e.g. -m repro).
set -euo pipefail
cd "$(dirname "$0")/.."
for env in main legacy; do
  echo "== $env"
  (export TSFM_ENV=$env; source .envrc; "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -q -p no:cacheprovider "$@")
done
