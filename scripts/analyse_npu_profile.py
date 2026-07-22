
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root")
    return parser.parse_args()


def looks_like_profile_dir(path: Path) -> bool:
    name = path.name
    if name.startswith("PROF_"):
        return True
    markers = [
        "profiler_info.json",
        "profiler_metadata.json",
        "trace_view.json",
        "kernel_details.csv",
    ]
    if any((path / marker).exists() for marker in markers):
        return True
    # torch_npu sometimes creates raw rank/device subtrees; keep this loose.
    children = {p.name for p in path.iterdir()} if path.exists() and path.is_dir() else set()
    return bool(children & {"device", "host", "FRAMEWORK", "ASCEND_PROFILER_OUTPUT"})


args = parse_args()
root = Path(args.profile_root).resolve()
print(f"profile_root={root}")
print(f"exists={root.exists()}")

if not root.exists():
    raise SystemExit(2)

print("directory tree preview:")
for p in list(root.rglob("*"))[:120]:
    print(p)

from torch_npu.profiler.profiler import analyse  # noqa: E402

candidates: list[Path] = []
if root.is_dir():
    candidates.append(root)
    candidates.extend(p for p in root.rglob("*") if p.is_dir() and looks_like_profile_dir(p))

deduped: list[Path] = []
seen = set()
for c in candidates:
    s = str(c)
    if s not in seen:
        seen.add(s)
        deduped.append(c)

ok = []
failed = []
for c in deduped:
    print(f"\n[analyse] trying {c}")
    try:
        analyse(str(c))
    except Exception as exc:  # noqa: BLE001
        print(f"[analyse] failed {c}: {type(exc).__name__}: {exc}")
        failed.append((c, repr(exc)))
    else:
        print(f"[analyse] success {c}")
        ok.append(c)

print("\n==== analyse summary ====")
print("success:")
for c in ok:
    print(c)
print("failed:")
for c, exc in failed:
    print(c, exc)

if not ok:
    raise SystemExit(1)
