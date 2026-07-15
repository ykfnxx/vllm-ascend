#!/usr/bin/env python3
"""Diagnose why a GLM-5 DSA service does not invoke the ASU index ops.

This script does not import torch, initialize an NPU, or execute either custom
operator. It inspects the running ``vllm serve`` process, the Python package
used by that process, the model config, and an optional server log.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DSA_INDEX_CAPACITY = 128 * 1024
DSA_QUERY_TOKENS = 2 * 1024
DSA_RESIDENT_TOKENS = 8 * 1024
DSA_BLOCK_SIZE = 128
DSA_SPARSE_THRESHOLD = DSA_QUERY_TOKENS + DSA_RESIDENT_TOKENS + DSA_BLOCK_SIZE
VERIFY_PROMPT_TOKENS = 10_600
SUPPORTED_ARCHITECTURE = "GlmMoeDsaForCausalLM"
SUPPORTED_MODEL_TYPE = "glm_moe_dsa"

OPERATOR_LOG_MARKERS = (
    "DSA sparse invoking asu_hbm_index_lookup",
    "DSA sparse completed asu_hbm_index_lookup",
    "DSA sparse invoking asu_hbm_index_maintain_aicpu",
    "DSA sparse completed asu_hbm_index_maintain_aicpu",
)
FRAMEWORK_LOG_MARKERS = (
    "DSA sparse general-plugin bootstrap installed",
    "DSA sparse platform patch installed",
    "DSA sparse EngineCore bootstrap verified",
    "DSA sparse EngineCore entry patch active",
    "DSA sparse runtime patches installed",
    "DSA sparse scheduler manager enabled",
    "DSA sparse worker manager enabled",
    "DSA DECODE REACHED SPARSE THRESHOLD",
    "DSA sparse worker forward mode active",
    "DSA sparse SFA path active",
    "DSA sparse indexer completed",
    "DSA sparse after_indexer entered",
    "DSA sparse layer batch ready",
    "DSA sparse invoking resident initialization",
    "DSA sparse completed resident initialization",
)
DENSE_SFA_LOG_MARKER = "DSA sparse SFA dense path active"
SOURCE_MARKERS = {
    "dsa_sparse/dsa_config.py": (
        SUPPORTED_ARCHITECTURE,
        '"dsa_hbm_sparse_budget": DSA_LOOKUP_QUERY_TOKENS',
    ),
    "dsa_sparse/dsa_ascend_ops_backend.py": (
        "torch.ops._C_ascend.asu_hbm_index_lookup",
        "torch.ops._C_ascend.asu_hbm_index_maintain_aicpu",
    ),
    "attention/sfa_v1.py": (
        "dsa_mgr.after_indexer",
        "attn_metadata.dsa_score_topk_k",
    ),
    "patch/dsa_sparse/patch_scheduler.py": (
        "plan_decode_resident_slots",
        "dsa_alloc_slots_wrap",
    ),
    "patch/dsa_sparse/patch_engine_process.py": (
        "ensure_dsa_engine_core_entrypoint",
        "CoreEngineProcManager",
        "verify_dsa_runtime_patches_installed",
    ),
    "platform.py": (
        "ASCEND_CUSTOM_OPP_PATH",
        '"aicpu_transformer"',
    ),
}


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    argv: list[str]
    executable: Path | None
    cwd: Path | None
    environ: dict[str, str]


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def pass_(self, message: str) -> None:
        print(f"[PASS] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a running GLM-5 DSA service without executing NPU ops."
        )
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="PID of the vllm serve process; auto-detected when omitted",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="model directory; inferred from the serve command when omitted",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        help="server stdout/stderr log used by the verify script",
    )
    return parser.parse_args()


def _read_proc_items(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def _read_proc_link(path: Path) -> Path | None:
    try:
        return Path(os.readlink(path))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _load_process(pid: int) -> ProcessInfo | None:
    proc_dir = Path("/proc") / str(pid)
    argv = _read_proc_items(proc_dir / "cmdline")
    if not argv:
        return None
    env_items = _read_proc_items(proc_dir / "environ")
    environ = {}
    for item in env_items:
        key, separator, value = item.partition("=")
        if separator:
            environ[key] = value
    return ProcessInfo(
        pid=pid,
        argv=argv,
        executable=_read_proc_link(proc_dir / "exe"),
        cwd=_read_proc_link(proc_dir / "cwd"),
        environ=environ,
    )


def _looks_like_vllm_serve(argv: list[str]) -> bool:
    if "serve" not in argv:
        return False
    serve_index = argv.index("serve")
    return any("vllm" in Path(arg).name.lower() for arg in argv[:serve_index])


def _find_serve_processes() -> list[ProcessInfo]:
    processes = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        process = _load_process(int(proc_dir.name))
        if process is not None and _looks_like_vllm_serve(process.argv):
            processes.append(process)
    return sorted(processes, key=lambda process: process.pid)


def _arg_value(argv: list[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        prefix = f"{name}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _has_arg(argv: list[str], name: str) -> bool:
    return name in argv or any(arg.startswith(f"{name}=") for arg in argv)


def _parse_int_arg(
    reporter: Reporter,
    argv: list[str],
    name: str,
) -> int | None:
    value = _arg_value(argv, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        reporter.fail(f"{name} is not an integer: {value!r}")
        return None


def _false_or_unset(value: str | None) -> bool:
    return value is None or value.strip().lower() in {"", "0", "false", "no"}


def _check_process_args(reporter: Reporter, process: ProcessInfo) -> Path | None:
    reporter.info(f"serve pid={process.pid}")
    reporter.info(f"serve command={shlex.join(process.argv)}")
    reporter.info(f"serve executable={process.executable}")
    reporter.info(f"serve cwd={process.cwd}")

    additional_raw = _arg_value(process.argv, "--additional-config")
    additional_config: dict[str, Any] = {}
    if additional_raw is None:
        reporter.fail("serve command has no --additional-config")
    else:
        try:
            parsed = json.loads(additional_raw)
        except json.JSONDecodeError as error:
            reporter.fail(f"--additional-config is invalid JSON: {error}")
        else:
            if isinstance(parsed, dict):
                additional_config = parsed
            else:
                reporter.fail("--additional-config is not a JSON object")

    dsa_config = additional_config.get("dsa_sparse_config")
    if isinstance(dsa_config, dict) and dsa_config.get("enabled") is True:
        reporter.pass_("dsa_sparse_config.enabled=true")
    else:
        reporter.fail("dsa_sparse_config.enabled=true is missing")

    if _has_arg(process.argv, "--enforce-eager"):
        reporter.pass_("--enforce-eager is enabled")
    else:
        reporter.fail("--enforce-eager is missing")
    if _has_arg(process.argv, "--no-async-scheduling"):
        reporter.pass_("--no-async-scheduling is enabled")
    else:
        reporter.fail("--no-async-scheduling is missing")
    if _has_arg(process.argv, "--speculative-config"):
        reporter.fail("speculative decoding disables the current DSA path")
    else:
        reporter.pass_("speculative decoding is disabled")
    max_batched_tokens = _parse_int_arg(
        reporter, process.argv, "--max-num-batched-tokens"
    )
    if _has_arg(process.argv, "--enable-chunked-prefill"):
        reporter.pass_("--enable-chunked-prefill is enabled")
    elif (
        max_batched_tokens is not None
        and max_batched_tokens >= VERIFY_PROMPT_TOKENS
    ):
        reporter.pass_(
            "chunked prefill is disabled, but max batched tokens can hold the "
            f"{VERIFY_PROMPT_TOKENS}-token verification prompt"
        )
    else:
        reporter.warn(
            "chunked prefill is not a DSA requirement, but the verification "
            f"prompt has {VERIFY_PROMPT_TOKENS} tokens while "
            f"--max-num-batched-tokens={max_batched_tokens!r}; enable chunked "
            "prefill or raise the batch token limit"
        )
    quantization = _arg_value(process.argv, "--quantization")
    if quantization == "ascend":
        reporter.pass_("--quantization ascend is enabled")
    else:
        reporter.fail(f"--quantization must be 'ascend', got {quantization!r}")

    block_size = _parse_int_arg(reporter, process.argv, "--block-size")
    if block_size == DSA_BLOCK_SIZE:
        reporter.pass_(f"block size is {DSA_BLOCK_SIZE}")
    else:
        reporter.fail(
            f"block size must be {DSA_BLOCK_SIZE}, got {block_size!r}"
        )

    max_model_len = _parse_int_arg(reporter, process.argv, "--max-model-len")
    if max_model_len is None:
        reporter.fail("--max-model-len is missing")
    elif max_model_len > DSA_INDEX_CAPACITY:
        reporter.fail(
            f"max model length {max_model_len} exceeds DSA index capacity "
            f"{DSA_INDEX_CAPACITY}"
        )
    elif max_model_len <= DSA_SPARSE_THRESHOLD:
        reporter.fail(
            f"max model length {max_model_len} cannot cross sparse threshold "
            f"{DSA_SPARSE_THRESHOLD}"
        )
    else:
        reporter.pass_(f"max model length is {max_model_len}")

    if additional_config.get("enable_sparse_c8") is True:
        reporter.fail("additional_config.enable_sparse_c8 must be disabled")
    else:
        reporter.pass_("Sparse C8 is disabled")

    environment_checks = {
        "VLLM_ASCEND_BALANCE_SCHEDULING": _false_or_unset,
        "VLLM_USE_V2_MODEL_RUNNER": _false_or_unset,
    }
    for name, validator in environment_checks.items():
        value = process.environ.get(name)
        if validator(value):
            reporter.pass_(f"{name} is disabled")
        else:
            reporter.fail(f"{name} must be disabled, got {value!r}")

    for name in (
        "VLLM_ASCEND_ENABLE_FLASHCOMM1",
        "VLLM_ASCEND_ENABLE_FLASHCOMM",
    ):
        value = process.environ.get(name)
        if _false_or_unset(value):
            reporter.pass_(f"{name} is disabled")
        else:
            reporter.warn(
                f"{name}={value!r} enables the DSA context-parallel path; "
                "lookup/maintain still run, but this is not the minimal "
                "operator-verification path"
            )

    compile_custom_kernels = process.environ.get("COMPILE_CUSTOM_KERNELS")
    if compile_custom_kernels is not None and _false_or_unset(
        compile_custom_kernels
    ):
        reporter.fail(
            f"COMPILE_CUSTOM_KERNELS must be enabled, got "
            f"{compile_custom_kernels!r}"
        )
    else:
        reporter.pass_("COMPILE_CUSTOM_KERNELS is enabled or uses its default")

    reporter.info(
        "VLLM_LOGGING_LEVEL="
        f"{process.environ.get('VLLM_LOGGING_LEVEL', '<unset: INFO default>')}"
    )
    reporter.info(
        "initial ASCEND_CUSTOM_OPP_PATH="
        f"{process.environ.get('ASCEND_CUSTOM_OPP_PATH', '<unset>')}"
        " (vllm-ascend may set it later in NPUPlatform.import_kernels)"
    )

    serve_index = process.argv.index("serve")
    if serve_index + 1 >= len(process.argv):
        reporter.fail("serve command has no model path")
        return None
    model_arg = process.argv[serve_index + 1]
    if model_arg.startswith("-"):
        reporter.fail(f"cannot infer model path after 'serve': {model_arg!r}")
        return None
    model_path = Path(model_arg).expanduser()
    if not model_path.is_absolute() and process.cwd is not None:
        model_path = process.cwd / model_path
    return model_path


PACKAGE_PROBE = r"""
import importlib.metadata
import importlib.util
import json
import sys

if sys.path and sys.path[0] == "":
    sys.path[0] = sys.argv[1]

result = {}
for module_name, distribution_name in (
    ("vllm", "vllm"),
    ("vllm_ascend", "vllm-ascend"),
    ("transformers", "transformers"),
):
    try:
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
    except Exception as error:
        origin = f"ERROR: {error}"
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    result[module_name] = {"origin": origin, "version": version}
result["vllm_general_plugins"] = {
    entry_point.name: entry_point.value
    for entry_point in importlib.metadata.entry_points(
        group="vllm.general_plugins"
    )
}
print(json.dumps(result))
"""


def _find_service_entrypoint_dir(process: ProcessInfo) -> Path:
    serve_index = process.argv.index("serve")
    for argument in reversed(process.argv[:serve_index]):
        if "vllm" not in Path(argument).name.lower():
            continue
        entrypoint = Path(argument)
        if entrypoint.is_absolute():
            return entrypoint.parent
        for path_entry in process.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(path_entry) / entrypoint
            if candidate.is_file():
                return candidate.parent
    if process.executable is not None:
        return process.executable.parent
    return Path("/nonexistent-vllm-entrypoint")


def _probe_service_python(
    reporter: Reporter,
    process: ProcessInfo,
) -> dict[str, Any] | None:
    if process.executable is None:
        reporter.fail("cannot resolve the service Python executable")
        return None
    environment = dict(process.environ) if process.environ else None
    entrypoint_dir = _find_service_entrypoint_dir(process)
    try:
        result = subprocess.run(
            [
                str(process.executable),
                "-c",
                PACKAGE_PROBE,
                str(entrypoint_dir),
            ],
            cwd=process.cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        reporter.fail(f"cannot probe the service Python: {error}")
        return None
    if result.returncode != 0:
        reporter.fail(
            "service Python package probe failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
        return None
    try:
        package_info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        reporter.fail(f"service Python package probe returned invalid JSON: {error}")
        return None

    reporter.info(f"service Python import entrypoint={entrypoint_dir}")
    for module_name in ("vllm", "vllm_ascend", "transformers"):
        module_info = package_info.get(module_name, {})
        reporter.info(
            f"service {module_name}: version={module_info.get('version')!r}, "
            f"origin={module_info.get('origin')!r}"
        )
        origin = module_info.get("origin")
        if not origin or (
            isinstance(origin, str) and origin.startswith("ERROR:")
        ):
            reporter.fail(f"service Python cannot find {module_name}")

    vllm_version = package_info.get("vllm", {}).get("version")
    if vllm_version == "0.18.0":
        reporter.pass_("service uses vllm 0.18.0")
    else:
        reporter.fail(
            f"service must use vllm 0.18.0, got {vllm_version!r}"
        )
    ascend_version = package_info.get("vllm_ascend", {}).get("version")
    if isinstance(ascend_version, str) and "rc" in ascend_version.lower():
        reporter.fail(
            "service still uses a vllm-ascend release candidate: "
            f"{ascend_version!r}"
        )
    elif isinstance(ascend_version, str):
        reporter.pass_(f"service vllm-ascend is not an rc build: {ascend_version}")
    else:
        reporter.warn("vllm-ascend distribution version is unavailable")

    general_plugins = package_info.get("vllm_general_plugins", {})
    dsa_plugin = general_plugins.get("ascend_dsa_sparse")
    if dsa_plugin == "vllm_ascend:register_dsa_sparse":
        reporter.pass_("service metadata contains the DSA general plugin")
    else:
        reporter.fail(
            "service metadata does not contain the DSA general plugin; "
            "reinstall vllm-ascend after rebuilding, got "
            f"{dsa_plugin!r}"
        )
    return package_info


def _check_installed_dsa_source(
    reporter: Reporter,
    package_info: dict[str, Any] | None,
) -> None:
    if package_info is None:
        return
    origin = package_info.get("vllm_ascend", {}).get("origin")
    if not isinstance(origin, str) or origin.startswith("ERROR:"):
        return
    package_root = Path(origin).parent
    for relative_path, expected_markers in SOURCE_MARKERS.items():
        source_path = package_root / relative_path
        if not source_path.is_file():
            reporter.fail(f"installed DSA source is missing: {source_path}")
            continue
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as error:
            reporter.fail(f"cannot read installed source {source_path}: {error}")
            continue
        missing = [marker for marker in expected_markers if marker not in source]
        if missing:
            reporter.fail(
                f"installed source is not the integration branch: {source_path}; "
                f"missing markers={missing!r}"
            )
        else:
            reporter.pass_(f"installed source contains DSA integration: {source_path}")


def _check_installed_custom_opp(
    reporter: Reporter,
    package_info: dict[str, Any] | None,
) -> None:
    if package_info is None:
        return
    origin = package_info.get("vllm_ascend", {}).get("origin")
    if not isinstance(origin, str) or origin.startswith("ERROR:"):
        return

    package_root = Path(origin).parent
    vendor_opp = package_root / "_cann_ops_custom/vendors/vllm-ascend"
    aicpu_opp = vendor_opp / "op_impl" / "aicpu_transformer"
    required_paths = (
        vendor_opp,
        aicpu_opp,
        aicpu_opp / "op_impl/cpu/config/cust_aicpu_kernel.json",
        aicpu_opp
        / "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so",
    )
    for path in required_paths:
        if path.exists():
            reporter.pass_(f"service package contains custom OPP path: {path}")
        else:
            reporter.fail(f"service package custom OPP path is missing: {path}")

    extension_paths = sorted(package_root.glob("vllm_ascend_C*.so"))
    if extension_paths:
        for path in extension_paths:
            reporter.pass_(f"service package contains custom-op binding: {path}")
    else:
        reporter.fail(
            f"service package contains no vllm_ascend_C extension: {package_root}"
        )

    reporter.info(
        "runtime custom OPP path expected from NPUPlatform.import_kernels: "
        f"{aicpu_opp}:{vendor_opp}"
    )


def _load_json(reporter: Reporter, path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        reporter.fail(f"file does not exist: {path}")
        return None
    except (OSError, json.JSONDecodeError) as error:
        reporter.fail(f"cannot read {path}: {error}")
        return None
    if not isinstance(value, dict):
        reporter.fail(f"JSON root is not an object: {path}")
        return None
    return value


def _check_model_config(reporter: Reporter, model_path: Path) -> None:
    try:
        model_path = model_path.expanduser().resolve()
    except OSError:
        model_path = model_path.expanduser()
    reporter.info(f"model path={model_path}")
    config = _load_json(reporter, model_path / "config.json")
    if config is None:
        return

    architectures = config.get("architectures")
    if isinstance(architectures, list) and SUPPORTED_ARCHITECTURE in architectures:
        reporter.pass_(f"model architecture includes {SUPPORTED_ARCHITECTURE}")
    else:
        reporter.fail(
            f"model architecture must include {SUPPORTED_ARCHITECTURE}, "
            f"got {architectures!r}"
        )
    model_type = config.get("model_type")
    if model_type == SUPPORTED_MODEL_TYPE:
        reporter.pass_(f"model_type={SUPPORTED_MODEL_TYPE}")
    else:
        reporter.fail(
            f"model_type must be {SUPPORTED_MODEL_TYPE}, got {model_type!r}"
        )
    index_topk = config.get("index_topk")
    if index_topk == DSA_QUERY_TOKENS:
        reporter.pass_(f"index_topk={DSA_QUERY_TOKENS}")
    else:
        reporter.fail(
            f"index_topk must be {DSA_QUERY_TOKENS}, got {index_topk!r}"
        )
    max_position_embeddings = config.get("max_position_embeddings")
    if (
        isinstance(max_position_embeddings, int)
        and max_position_embeddings > DSA_SPARSE_THRESHOLD
    ):
        reporter.pass_(
            f"model context limit is {max_position_embeddings}, above the "
            f"sparse threshold {DSA_SPARSE_THRESHOLD}"
        )
    else:
        reporter.fail(
            "model max_position_embeddings cannot cross the sparse threshold: "
            f"{max_position_embeddings!r}"
        )


def _check_server_log(reporter: Reporter, server_log: Path | None) -> None:
    if server_log is None:
        reporter.warn(
            "--server-log was not provided; operator and stage logs were not checked"
        )
        return
    try:
        log_size = server_log.stat().st_size
        log_file = server_log.open(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError) as error:
        reporter.fail(f"cannot read server log {server_log}: {error}")
        return
    reporter.info(f"server log={server_log} bytes={log_size}")

    threshold_marker = "DSA DECODE REACHED SPARSE THRESHOLD"
    tokenized_marker = "DSA TOKENIZED PROMPT"
    error_markers = (
        "BinaryGetFunction failed",
        "Aicpu kernel execute failed",
        "AICPU kernel execute failed",
        "No Available shared memory broadcast",
        "Poller timed out",
    )
    found_operator_markers: set[str] = set()
    found_framework_markers: set[str] = set()
    found_errors: set[str] = set()
    threshold_reached = False
    tokenized_prompt_seen = False
    dense_sfa_seen = False
    with log_file:
        for line in log_file:
            found_operator_markers.update(
                marker for marker in OPERATOR_LOG_MARKERS if marker in line
            )
            found_framework_markers.update(
                marker for marker in FRAMEWORK_LOG_MARKERS if marker in line
            )
            found_errors.update(
                marker for marker in error_markers if marker in line
            )
            threshold_reached |= threshold_marker in line
            tokenized_prompt_seen |= tokenized_marker in line
            dense_sfa_seen |= DENSE_SFA_LOG_MARKER in line

    framework_present = [
        marker for marker in FRAMEWORK_LOG_MARKERS
        if marker in found_framework_markers
    ]
    framework_missing = [
        marker for marker in FRAMEWORK_LOG_MARKERS
        if marker not in found_framework_markers
    ]
    for marker in framework_present:
        reporter.pass_(f"server log contains framework marker: {marker}")
    if framework_missing:
        reporter.warn(
            "server log is missing framework markers in the expected call "
            f"chain: {framework_missing!r}"
        )

    present = [
        marker for marker in OPERATOR_LOG_MARKERS
        if marker in found_operator_markers
    ]
    if len(present) == len(OPERATOR_LOG_MARKERS):
        reporter.pass_("all lookup/maintain invocation and completion logs exist")
    elif present:
        missing = [
            marker for marker in OPERATOR_LOG_MARKERS
            if marker not in found_operator_markers
        ]
        reporter.fail(
            "only part of the operator sequence was logged; "
            f"missing={missing!r}"
        )
    else:
        reporter.warn("server log contains no lookup/maintain invocation markers")

    if dense_sfa_seen:
        reporter.warn(
            "SFA executed a DecodeOnly batch without sparse score control and "
            "used the original indexer at least once"
        )

    if threshold_reached:
        reporter.pass_("server log confirms ENTER_SPARSE_DECODE was reached")
    elif tokenized_prompt_seen:
        reporter.warn(
            "DSA request logging is active, but no request reached the sparse threshold"
        )
    else:
        reporter.warn(
            "no DSA stage markers are visible; the sparse-threshold marker "
            "is logged at INFO, so no request reached ENTER_SPARSE_DECODE or "
            "the scheduler patch is not active"
        )

    if found_errors:
        reporter.warn(
            "server log contains runtime error markers: "
            f"{sorted(found_errors)!r}"
        )


def main() -> int:
    args = _parse_args()
    reporter = Reporter()

    if args.pid is not None:
        process = _load_process(args.pid)
        if process is None:
            reporter.fail(f"cannot read process {args.pid}")
            process_candidates = []
        elif not _looks_like_vllm_serve(process.argv):
            reporter.fail(f"process {args.pid} is not a vllm serve process")
            process_candidates = []
        else:
            process_candidates = [process]
    else:
        process_candidates = _find_serve_processes()
        if not process_candidates:
            reporter.fail("no running vllm serve process was found")
        elif len(process_candidates) > 1:
            dsa_candidates = [
                process
                for process in process_candidates
                if "dsa_sparse_config" in " ".join(process.argv)
            ]
            if len(dsa_candidates) == 1:
                process_candidates = dsa_candidates
            else:
                reporter.fail(
                    "multiple vllm serve processes were found; rerun with --pid: "
                    + ", ".join(str(process.pid) for process in process_candidates)
                )
                process_candidates = []

    if process_candidates:
        process = process_candidates[0]
        inferred_model_path = _check_process_args(reporter, process)
        package_info = _probe_service_python(reporter, process)
        _check_installed_dsa_source(reporter, package_info)
        _check_installed_custom_opp(reporter, package_info)
        model_path = args.model_path or inferred_model_path
        if model_path is None:
            reporter.fail("model path is unavailable; pass --model-path")
        else:
            _check_model_config(reporter, model_path)

    _check_server_log(reporter, args.server_log)

    print(
        f"[SUMMARY] failures={reporter.failures} warnings={reporter.warnings} "
        f"sparse_threshold={DSA_SPARSE_THRESHOLD}"
    )
    if reporter.failures:
        return 1
    reporter.pass_(
        "static service environment checks passed; run the verify script to "
        "exercise decode"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
