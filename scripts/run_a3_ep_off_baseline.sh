#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible name. The migration baseline is now the reduced-model,
# EP-off, single-card path.
exec bash "$SCRIPT_DIR/run_a3_single_card_baseline.sh"
