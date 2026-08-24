# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "examples" / "dsa_sparse_pd_shm_validate.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "dsa_sparse_pd_shm_validate",
    VALIDATOR_PATH,
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)


def _write_events(
    path: Path,
    prefix: str,
    events: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(
            f"INFO {prefix}{json.dumps(event)}\n"
            for event in events
        ),
        encoding="utf-8",
    )


def test_validates_shared_memory_content_and_release(tmp_path):
    object_name = "vllm_ascend_dsa_sparse_payload"
    content_sha256 = "a" * 64
    payload = {
        "name": object_name,
        "size": 8,
        "cache_kind": "indexer",
        "cache_layer_name": "model.layers.0.self_attn.indexer.k_cache",
        "content_sha256": content_sha256,
        "cache_planes": [{"offset": 0, "nbytes": 4}],
        "tail_planes": [{"offset": 4, "nbytes": 4}],
    }
    handoff = {
        "protocol_version": 5,
        "remote_request_id": "request-a",
        "stored_token_count": 3,
        "block_size": 2,
        "layer_topk_by_rank": {"0": {"layer.0": [0]}},
        "shared_memory_payloads_by_rank": {
            "0": {"layer.0": payload}
        },
    }
    prefill_log = tmp_path / "prefill.log"
    decode_log = tmp_path / "decode.log"
    handoff_sha256 = "b" * 64
    _write_events(
        prefill_log,
        validator.PD_LOG_PREFIX,
        [
            {
                "event": "handoff_send",
                "handoff_sha256": handoff_sha256,
                "handoff": handoff,
            }
        ],
    )
    with prefill_log.open("a", encoding="utf-8") as log_file:
        log_file.write(
            "INFO "
            + validator.PROBE_LOG_PREFIX
            + json.dumps(
                {
                    "event": "shared_memory_publish",
                    "object_name": object_name,
                    "payload_bytes": 8,
                    "cache_kind": "indexer",
                    "content_sha256": content_sha256,
                }
            )
            + "\n"
        )
    _write_events(
        decode_log,
        validator.PD_LOG_PREFIX,
        [
            {
                "event": "handoff_receive",
                "handoff_sha256": handoff_sha256,
                "handoff": handoff,
            }
        ],
    )
    with decode_log.open("a", encoding="utf-8") as log_file:
        for event_name in (
            "shared_memory_content_verified",
            "shared_memory_consume",
        ):
            log_file.write(
                "INFO "
                + validator.PROBE_LOG_PREFIX
                + json.dumps(
                    {
                        "event": event_name,
                        "object_name": object_name,
                        "payload_bytes": 8,
                        "cache_kind": "indexer",
                        "content_sha256": content_sha256,
                    }
                )
                + "\n"
            )

    summary = validator.validate(
        prefill_log=prefill_log,
        decode_log=decode_log,
        shared_memory_root=tmp_path,
        require_mtp=False,
    )
    assert summary == {
        "payload_count": 1,
        "indexer_payload_count": 1,
        "mtp_payload_count": 0,
        "tail_payload_count": 1,
        "verified_payload_count": 1,
        "released_payload_count": 1,
    }

    _write_events(
        decode_log,
        validator.PD_LOG_PREFIX,
        [
            {
                "event": "handoff_receive",
                "handoff_sha256": handoff_sha256,
                "handoff": handoff,
            }
        ],
    )
    with decode_log.open("a", encoding="utf-8") as log_file:
        for event_name, event_sha256 in (
            ("shared_memory_content_verified", "c" * 64),
            ("shared_memory_consume", content_sha256),
        ):
            log_file.write(
                "INFO "
                + validator.PROBE_LOG_PREFIX
                + json.dumps(
                    {
                        "event": event_name,
                        "object_name": object_name,
                        "payload_bytes": 8,
                        "cache_kind": "indexer",
                        "content_sha256": event_sha256,
                    }
                )
                + "\n"
            )
    with pytest.raises(RuntimeError, match="content_sha256 mismatch"):
        validator.validate(
            prefill_log=prefill_log,
            decode_log=decode_log,
            shared_memory_root=tmp_path,
            require_mtp=False,
        )
