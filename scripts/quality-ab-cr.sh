#!/usr/bin/env bash
# CACHE_ROUTE quality A/B with live progress (tok/s, running accuracy).
# Stop anytime with Ctrl-C — partial results stay under ~/quality-ab-cr/latest/
#
# Examples:
#   ~/quality-ab-cr.sh                     # hellaswag × 20 (default, faster)
#   ~/quality-ab-cr.sh --limit 40 --tasks hellaswag,arc_challenge,mmlu
#   ~/quality-ab-cr.sh --only off          # stock only
#   ~/quality-ab-cr.sh --only on           # CACHE_ROUTE=1 only
#
# Watch progress in another terminal:
#   tail -f ~/quality-ab-cr/latest/cr_off/STATUS.txt
#   tail -f ~/quality-ab-cr/latest/cr_on/STATUS.txt
#   ls ~/quality-ab-cr/latest/*/checkpoint_*.json

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export COLI_MODEL="${COLI_MODEL:-${HOME}/glm52-colibri-int4}"
export GLM_BIN="${GLM_BIN:-${HOME}/colibri/c/glm}"
export BENCH_DATA="${BENCH_DATA:-${HOME}/colibri/c/bench}"

# sensible GB10-ish defaults if user hasn't set them (do not force CACHE_ROUTE here)
export MTP="${MTP:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer repo script; fall back to home copy if present
if [[ -f "${SCRIPT_DIR}/quality-ab-cr.py" ]]; then
  exec python3 "${SCRIPT_DIR}/quality-ab-cr.py" "$@"
fi
exec python3 "${HOME}/quality-ab-cr.py" "$@"
