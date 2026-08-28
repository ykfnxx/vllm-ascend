#!/usr/bin/env bash

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -r "$BUNDLE_DIR/SHA256SUMS" ]] || {
    echo "SHA256SUMS is missing from the A5 bundle." >&2
    exit 1
}
(
    cd "$BUNDLE_DIR"
    if command -v sha256sum >/dev/null; then
        sha256sum -c SHA256SUMS
    else
        shasum -a 256 -c SHA256SUMS
    fi
) >/dev/null
echo "A5_KVGATHER_BUNDLE_CHECKSUM_OK"
