#!/usr/bin/env python3
"""Extract vLLM distributed-initialization latency from an engine log."""

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

TIMESTAMP = r"(?P<month>\d{2})-(?P<day>\d{2}) (?P<clock>\d{2}:\d{2}:\d{2})"
START_PATTERN = re.compile(
    rf"{TIMESTAMP}.*world_size=\d+.*distributed_init_method=(?P<method>\S+)"
)
DONE_PATTERN = re.compile(rf"{TIMESTAMP}.*rank \d+ in world size \d+ is assigned as")


def parse_timestamp(match: re.Match[str]) -> datetime:
    return datetime.strptime(
        f"2000-{match.group('month')}-{match.group('day')} {match.group('clock')}",
        "%Y-%m-%d %H:%M:%S",
    )


def summarize(log_path: Path) -> str:
    start_match: re.Match[str] | None = None
    done_match: re.Match[str] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if start_match is None:
            start_match = START_PATTERN.search(line)
            continue
        done_match = DONE_PATTERN.search(line)
        if done_match is not None:
            break

    if start_match is None or done_match is None:
        return "distributed_init_seconds=unavailable"

    start = parse_timestamp(start_match)
    done = parse_timestamp(done_match)
    if done < start:
        done += timedelta(days=366)
    seconds = int((done - start).total_seconds())
    return "\n".join(
        (
            f"distributed_init_method={start_match.group('method')}",
            f"distributed_init_start={start_match.group('month')}-"
            f"{start_match.group('day')} {start_match.group('clock')}",
            f"distributed_init_done={done_match.group('month')}-"
            f"{done_match.group('day')} {done_match.group('clock')}",
            f"distributed_init_seconds={seconds}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(summarize(args.log))


if __name__ == "__main__":
    main()
