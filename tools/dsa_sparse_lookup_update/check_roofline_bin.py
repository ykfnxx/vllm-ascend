#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BLOCK_HEADER = struct.Struct("<QBBBB")
CODE_PATH_BYTES = 4096
ROOFLINE_BLOCK_TYPE = 0x0D
BLOCK_TYPE_NAMES = {
    0x01: "Code",
    0x02: "Trace",
    0x03: "FileApi",
    0x04: "InstrApi",
    0x05: "OpBasicInfo",
    0x06: "ComputeLoadFigure",
    0x07: "ComputeLoadTable",
    0x08: "StorageAccessHeatMap",
    0x09: "StorageAccessTable",
    0x0C: "OccupancyMap",
    ROOFLINE_BLOCK_TYPE: "RoofLine",
    0x0E: "CacheLineHeatMap",
    0x0F: "TopStallReason",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Roofline payload in an msOpProf "
            "visualize_data.bin file."
        )
    )
    parser.add_argument(
        "visualize_data",
        type=Path,
        help="Path to the visualize_data.bin produced by msOpProf.",
    )
    return parser.parse_args()


def _parse_blocks(data: bytes) -> list[tuple[int, bytes]]:
    blocks: list[tuple[int, bytes]] = []
    offset = 0
    while offset < len(data):
        remaining = len(data) - offset
        if remaining < BLOCK_HEADER.size:
            raise ValueError(
                f"trailing {remaining} byte(s) at offset {offset}; "
                "the file is truncated"
            )

        content_size, block_type, padding, _, _ = (
            BLOCK_HEADER.unpack_from(data, offset)
        )
        offset += BLOCK_HEADER.size

        if block_type == 0x01:
            if len(data) - offset < CODE_PATH_BYTES:
                raise ValueError(
                    f"truncated Code header at offset {offset}"
                )
            offset += CODE_PATH_BYTES

        if padding > content_size:
            raise ValueError(
                f"block 0x{block_type:02x} has padding {padding} "
                f"larger than content size {content_size}"
            )

        block_end = offset + content_size
        if block_end > len(data):
            raise ValueError(
                f"block 0x{block_type:02x} at offset {offset} "
                f"requires {content_size} byte(s), but only "
                f"{len(data) - offset} remain; the file is truncated"
            )

        payload_end = block_end - padding
        blocks.append((block_type, data[offset:payload_end]))
        offset = block_end

    return blocks


def _parse_roofline(payload: bytes, index: int) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"RoofLine block {index} contains invalid JSON: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(
            f"RoofLine block {index} must contain a JSON object"
        )
    return decoded


def main() -> int:
    args = _parse_args()
    path = args.visualize_data.expanduser().resolve()
    if not path.is_file():
        print(f"FAIL: file does not exist: {path}", file=sys.stderr)
        return 2

    data = path.read_bytes()
    try:
        blocks = _parse_blocks(data)
        roofline_payloads = [
            _parse_roofline(payload, index)
            for index, (block_type, payload) in enumerate(blocks)
            if block_type == ROOFLINE_BLOCK_TYPE
        ]
    except ValueError as error:
        print(f"file: {path}")
        print(f"file_size: {len(data)}")
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    block_counts = Counter(block_type for block_type, _ in blocks)
    print(f"file: {path}")
    print(f"file_size: {len(data)}")
    print(f"blocks: {len(blocks)}")
    for block_type in sorted(block_counts):
        name = BLOCK_TYPE_NAMES.get(block_type, "Unknown")
        print(
            f"  0x{block_type:02x} {name}: "
            f"{block_counts[block_type]}"
        )

    print(f"roofline_blocks: {len(roofline_payloads)}")
    group_count = 0
    line_count = 0
    for block_index, roofline in enumerate(roofline_payloads):
        groups = roofline.get("multiple_rooflines", [])
        if not isinstance(groups, list):
            print(
                f"FAIL: RoofLine block {block_index} has a non-list "
                "multiple_rooflines field",
                file=sys.stderr,
            )
            return 2
        print(f"roofline_block[{block_index}] groups: {len(groups)}")
        group_count += len(groups)
        for group in groups:
            if not isinstance(group, dict):
                print(
                    f"FAIL: RoofLine block {block_index} contains an "
                    "invalid group",
                    file=sys.stderr,
                )
                return 2
            title = group.get("title", "<untitled>")
            rooflines = group.get("rooflines", [])
            if not isinstance(rooflines, list):
                print(
                    f"FAIL: Roofline group {title!r} has a non-list "
                    "rooflines field",
                    file=sys.stderr,
                )
                return 2
            line_count += len(rooflines)
            print(f"  {title}: {len(rooflines)} line(s)")

    print(f"roofline_groups: {group_count}")
    print(f"roofline_lines: {line_count}")
    if not roofline_payloads:
        print(
            "FAIL: no RoofLine block was written; the replay was "
            "incomplete or Roofline collection was not enabled",
            file=sys.stderr,
        )
        return 1
    if group_count == 0 or line_count == 0:
        print(
            "FAIL: the RoofLine block is present but contains no "
            "analysis data; check the A5 SIMT operand-record and PMU "
            "replay results",
            file=sys.stderr,
        )
        return 1

    print("PASS: non-empty Roofline analysis data is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
