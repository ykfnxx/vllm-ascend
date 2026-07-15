#!/usr/bin/env python3
"""Parse raw Ascend PyTorch profiler data outside vLLM worker processes.

vLLM workers normally run as daemon processes, where torch_npu refuses to
start the subprocesses required by profiler analysis. Run this helper after
profiling to export a compact DB from the raw ``FRAMEWORK``/``PROF_*`` data.

Example:

    python3 examples/dsa_sparse/parse_glm5_dsa_profile.py \
        /data/dsa-profiles/run/timestamp/bs1/trace \
        --output-dir /data/dsa-profiles-parsed/bs1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

PROFILE_SEARCH_DEPTH = 3
RANK_IN_NAME = re.compile(r"(?:^|_)rank_?(\d+)(?:_|$)", re.IGNORECASE)
RANK_INFO_FILE = re.compile(r"profiler_info_(\d+)\.json$")


def _is_raw_profile_dir(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    return any(child.is_dir() and child.name == "FRAMEWORK" for child in children) or any(
        child.is_dir() and child.name.startswith("PROF_") for child in children
    )


def _discover_profile_dirs(root: Path) -> list[Path]:
    profiles: list[Path] = []
    pending: deque[tuple[Path, int]] = deque([(root, 0)])
    while pending:
        path, depth = pending.popleft()
        if _is_raw_profile_dir(path):
            profiles.append(path)
            continue
        if depth >= PROFILE_SEARCH_DEPTH:
            continue
        try:
            children = sorted(child for child in path.iterdir() if child.is_dir())
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in children)
    return sorted(set(profiles))


def _find_rank_value(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("rank_id", "global_rank", "rankId", "rank"):
            rank = value.get(key)
            if isinstance(rank, int) and rank >= 0:
                return rank
            if isinstance(rank, str) and rank.isdigit():
                return int(rank)
        for child in value.values():
            rank = _find_rank_value(child)
            if rank is not None:
                return rank
    elif isinstance(value, list):
        for child in value:
            rank = _find_rank_value(child)
            if rank is not None:
                return rank
    return None


def _profile_rank(path: Path) -> int | None:
    match = RANK_IN_NAME.search(path.name)
    if match:
        return int(match.group(1))

    for info_path in sorted(path.glob("profiler_info*.json")):
        match = RANK_INFO_FILE.fullmatch(info_path.name)
        if match:
            return int(match.group(1))
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rank = _find_rank_value(info)
        if rank is not None:
            return rank
    return None


def _output_files(profile_dir: Path, export_type: str) -> list[Path]:
    output_dir = profile_dir / "ASCEND_PROFILER_OUTPUT"
    if not output_dir.is_dir():
        return []
    if export_type == "db":
        return sorted(output_dir.glob("*.db"))
    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix in {".csv", ".json", ".db"}
    )


def _copy_export(
    profile_dir: Path,
    outputs: list[Path],
    output_root: Path,
) -> list[Path]:
    exported_profile = output_root / profile_dir.name
    exported_output = exported_profile / "ASCEND_PROFILER_OUTPUT"
    exported_output.mkdir(parents=True, exist_ok=True)

    for info_path in profile_dir.glob("profiler_info*.json"):
        shutil.copy2(info_path, exported_profile / info_path.name)
    metadata_path = profile_dir / "profiler_metadata.json"
    if metadata_path.is_file():
        shutil.copy2(metadata_path, exported_profile / metadata_path.name)

    exported: list[Path] = []
    for output in outputs:
        destination = exported_output / output.name
        shutil.copy2(output, destination)
        exported.append(destination)
    return exported


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-parse raw Ascend PyTorch profiler data. By default only "
            "rank 0 is exported as a compact MindStudio Insight DB."
        )
    )
    parser.add_argument(
        "profile_path",
        type=Path,
        help="trace/case directory or one raw *_ascend_pt profile directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="separate directory for compact parsed results",
    )
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument(
        "--rank",
        type=int,
        action="append",
        help="rank to parse; may be specified more than once (default: 0)",
    )
    rank_group.add_argument(
        "--all-ranks",
        action="store_true",
        help="parse every discovered rank",
    )
    parser.add_argument(
        "--export-type",
        choices=("db", "text"),
        default="db",
        help="torch_npu export format (default: db)",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="maximum torch_npu parser processes (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.profile_path.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"FAIL: profile path is not a directory: {root}", file=sys.stderr)
        return 2
    if args.max_processes < 1:
        print("FAIL: --max-processes must be positive", file=sys.stderr)
        return 2
    if args.rank is not None and any(rank < 0 for rank in args.rank):
        print("FAIL: --rank cannot be negative", file=sys.stderr)
        return 2

    profile_dirs = _discover_profile_dirs(root)
    if not profile_dirs:
        print(
            f"FAIL: no raw FRAMEWORK/PROF_* profiler directory found under {root}",
            file=sys.stderr,
        )
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {output_root}")

    ranked_profiles = [(path, _profile_rank(path)) for path in profile_dirs]
    print("Discovered raw profiles:")
    for path, rank in ranked_profiles:
        rank_label = "unknown" if rank is None else str(rank)
        print(f"  rank={rank_label}: {path}")

    if args.all_ranks:
        selected = ranked_profiles
    else:
        requested_ranks = set(args.rank if args.rank is not None else [0])
        selected = [
            (path, rank)
            for path, rank in ranked_profiles
            if rank in requested_ranks
        ]
        missing_ranks = requested_ranks - {
            rank for _, rank in selected if rank is not None
        }
        if missing_ranks:
            ranks = ", ".join(str(rank) for rank in sorted(missing_ranks))
            print(f"FAIL: requested rank(s) not found: {ranks}", file=sys.stderr)
            return 1

    try:
        from torch_npu.profiler.profiler import analyse
    except ImportError as error:
        print(
            "FAIL: torch_npu profiler is unavailable; run this script in the "
            f"CANN/vLLM Ascend container: {error}",
            file=sys.stderr,
        )
        return 1

    failures: list[Path] = []
    for profile_dir, rank in selected:
        print(
            f"Parsing rank={rank if rank is not None else 'unknown'} as "
            f"{args.export_type}: {profile_dir}"
        )
        analyse(
            profiler_path=str(profile_dir),
            max_process_number=args.max_processes,
            export_type=args.export_type,
        )
        outputs = _output_files(profile_dir, args.export_type)
        if not outputs:
            failures.append(profile_dir)
            print(
                "FAIL: parser produced no expected output; inspect "
                f"{profile_dir / 'logs'}",
                file=sys.stderr,
            )
            continue
        exported = _copy_export(profile_dir, outputs, output_root)
        for output in exported:
            print(f"  output: {output} ({output.stat().st_size} bytes)")

    if failures:
        print(
            f"FAIL: {len(failures)} of {len(selected)} profile(s) failed to parse",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: parsed {len(selected)} profile(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
