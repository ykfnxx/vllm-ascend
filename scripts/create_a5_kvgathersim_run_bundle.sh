#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_NAME="${A5_BUNDLE_NAME:-dsa-offload-0.23-graph-a5}"
OUTPUT="${1:-$REPO_ROOT/../$BUNDLE_NAME.zip}"
TMP_DIR="$(mktemp -d)"
STAGE="$TMP_DIR/$BUNDLE_NAME"
PAYLOAD="$STAGE/payload/vllm-ascend-0.23.0"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Commit the requested source changes before creating the bundle." >&2
    exit 1
fi
mkdir -p "$PAYLOAD"
git -C "$REPO_ROOT" archive --format=tar HEAD | tar -xf - -C "$PAYLOAD"
printf '%s\n' "$commit" >"$PAYLOAD/SNAPSHOT_COMMIT"
cp "$PAYLOAD/scripts/a5_kvgather_bundle/apply_and_run_on_host.sh" "$STAGE/"
cp "$PAYLOAD/scripts/a5_kvgather_bundle/verify_bundle.sh" "$STAGE/"
cp "$PAYLOAD/scripts/a5_kvgather_bundle/README_CN.md" "$STAGE/"
chmod 700 "$STAGE/apply_and_run_on_host.sh" "$STAGE/verify_bundle.sh"

(
    cd "$STAGE"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
        xargs -0 shasum -a 256 >SHA256SUMS
)
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
(
    cd "$TMP_DIR"
    zip -qr "$OUTPUT" "$BUNDLE_NAME"
)
unzip -tq "$OUTPUT" >/dev/null
echo "A5_KVGATHER_RUN_BUNDLE_READY: $OUTPUT"
echo "bundle_name=$BUNDLE_NAME"
echo "source_commit=$commit"
echo "sha256=$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
