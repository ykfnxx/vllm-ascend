# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PROFILE_FILENAMES = {
    "kernel_details.csv",
    "operator_details.csv",
    "op_statistic.csv",
}
GRAPH_MARKER = (
    "DSA_OFFLOAD_KVGATHER_SIM_GRAPH_ACTIVE "
    "lookup=dsa_sparse_turbo_resolve_update_batch_v2 "
    "gather=asu_kv_gather_direct_v2 mtp=1 graph_mode=FULL_DECODE_ONLY"
)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _profile_evidence(profile_dir: Path) -> dict[str, bool]:
    profile_files = [path for path in profile_dir.rglob("*") if path.is_file() and path.name in PROFILE_FILENAMES]
    if not profile_files:
        raise RuntimeError(f"No analyzed operator CSV exists under {profile_dir}")
    evidence = {
        "resolve_update_batch_v2": False,
        "asu_kv_gather_direct_v2": False,
        "sparse_flash_attention": False,
    }
    for path in profile_files:
        with path.open(
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as profile_file:
            reader = csv.reader(profile_file)
            for row in reader:
                normalized = _normalize(" ".join(row))
                evidence["resolve_update_batch_v2"] |= (
                    "dsasparseturboresolveupdatebatchv2" in normalized
                )
                evidence["asu_kv_gather_direct_v2"] |= (
                    "asukvgatherdirectv2" in normalized
                )
                evidence["sparse_flash_attention"] |= "sparseflashattention" in normalized
    missing = [name for name, found in evidence.items() if not found]
    if missing:
        raise RuntimeError("Decode profile is missing required operators: " + ", ".join(missing))
    return evidence


def validate(
    decode_log: Path,
    response_json: Path,
    profile_dir: Path,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    response = json.loads(response_json.read_text(encoding="utf-8"))
    choices = response.get("choices")
    usage = response.get("usage")
    if not isinstance(choices, list) or len(choices) != batch_size:
        actual = len(choices) if isinstance(choices, list) else choices
        raise RuntimeError(f"Expected {batch_size} choices, got {actual!r}")
    if not isinstance(usage, dict):
        raise RuntimeError("Response does not contain usage metadata")
    total_prompt_tokens = batch_size * prompt_tokens
    total_completion_tokens = batch_size * output_tokens
    if usage.get("prompt_tokens") != total_prompt_tokens:
        raise RuntimeError(
            f"prompt_tokens mismatch: expected={total_prompt_tokens} actual={usage.get('prompt_tokens')}"
        )
    if usage.get("completion_tokens") != total_completion_tokens:
        raise RuntimeError(
            f"completion_tokens mismatch: expected={total_completion_tokens} actual={usage.get('completion_tokens')}"
        )

    log_text = decode_log.read_text(encoding="utf-8", errors="replace")
    if GRAPH_MARKER not in log_text:
        raise RuntimeError("Decode log does not prove MTP kvgather_sim FULL_DECODE_ONLY graph execution")
    if "Replaying aclgraph" not in log_text:
        raise RuntimeError("Decode log does not prove that the captured ACL graph replayed")
    forbidden_markers = (
        "DMP_A5_ACTIVE",
        "DMP_A5_SCHEME5_ACTIVE",
        "DMP_A5_SCHEME55_ACTIVE",
    )
    mixed = [marker for marker in forbidden_markers if marker in log_text]
    if mixed:
        raise RuntimeError("Decode log unexpectedly contains DMP route markers: " + ", ".join(mixed))

    evidence = _profile_evidence(profile_dir)
    trace_dirs = sorted(str(path) for path in profile_dir.rglob("*_ascend_pt") if path.is_dir())
    if not trace_dirs:
        raise RuntimeError(f"No Ascend profiler trace exists under {profile_dir}")
    return {
        "batch_size": batch_size,
        "prompt_tokens_per_request": prompt_tokens,
        "output_tokens_per_request": output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "mtp_speculative_tokens": 1,
        "decode_graph": "FULL_DECODE_ONLY",
        "decode_graph_replayed": True,
        "dmp": False,
        "lookup_operator": "dsa_sparse_turbo_resolve_update_batch_v2",
        "gather_operator": "asu_kv_gather_direct_v2",
        "gather_source_payload": "synthetic_zero",
        "profile_scope": "decode_only",
        "profile_evidence": evidence,
        "trace_dirs": trace_dirs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--response-json", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    args = parser.parse_args()
    summary = validate(
        args.decode_log,
        args.response_json,
        args.profile_dir,
        args.batch_size,
        args.prompt_tokens,
        args.output_tokens,
    )
    print("A5_DSA_OFFLOAD_MTP_KVGATHER_SIM_GRAPH_VALIDATED: " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
