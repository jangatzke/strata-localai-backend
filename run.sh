#!/bin/sh
# Starts the Strata LocalAI gRPC backend. Run from /srv/strata-localai-backend.
set -eu
cd "$(dirname "$0")"
exec ./venv/bin/python strata_grpc_backend.py --config strata-backend.json --host "${HOST:-0.0.0.0}" --port "${PORT:-50053}"
