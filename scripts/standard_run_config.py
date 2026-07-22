
import ctypes
import json
import os
import sys
from pathlib import Path

DEFAULT_BATCH_SIZE = 64
FALLBACK_BATCH_SIZE = 48
DEFAULT_OUTPUT_TOKENS = 10
DEFAULT_PROMPT_TOKENS = 128 * 1024
DEFAULT_MAX_MODEL_LEN = 132000
DEFAULT_VISIBLE_DEVICES = "0"
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_DMP_STREAM_MODE = "two"
DEFAULT_DMP_KV_BACKEND = "local"
DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parent / "dmp-runtime"
DEFAULT_FUSED_INDEXER_ROOT = (
    Path(__file__).resolve().parent / "dmp-fused-indexer-kv-select"
)
DEFAULT_FUSED_INDEXER_OPP_PATH = str(
    DEFAULT_FUSED_INDEXER_ROOT / "opp/vendors/customize"
)
DEFAULT_FUSED_INDEXER_PYTHON_PATH = str(DEFAULT_FUSED_INDEXER_ROOT / "torch_extension")
DEFAULT_LOOKUP_MAINTAIN_ROOT = Path(__file__).resolve().parent / "dmp-lookup-maintain"
DEFAULT_LOOKUP_MAINTAIN_OPP_PATH = str(
    DEFAULT_LOOKUP_MAINTAIN_ROOT / "opp/vendors/customize"
)
DEFAULT_LOOKUP_MAINTAIN_PYTHON_PATH = str(
    DEFAULT_LOOKUP_MAINTAIN_ROOT / "torch_extension"
)
DEFAULT_DUAL_ATTENTION_OPP_PATH = str(DEFAULT_RUNTIME_ROOT / "opp/vendors/customize")
DEFAULT_DUAL_ATTENTION_PYTHON_PATH = str(DEFAULT_RUNTIME_ROOT / "python")
DEFAULT_HIXL_PYTHON_PATH = str(DEFAULT_RUNTIME_ROOT / "hixl-python")
DEFAULT_HIXL_CONFIG = str(Path(__file__).resolve().parent / "dmp_hixl_config.json")
LEGACY_DUAL_ATTENTION_OPP_PATH = "/usr/local/Ascend/cann-8.5.1/opp/vendors/customize"
DEFAULT_VLLM_ASCEND_OP_API_PATH = (
    "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/"
    "vendors/vllm-ascend/op_api/lib"
)
_PRELOADED_OPERATOR_LIBRARIES: list[ctypes.CDLL] = []


def configure_dmp_runtime(visible_devices: str) -> str:
    """Configure the selected DMP path before torch_npu or vLLM import."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    os.environ["VLLM_ASCEND_ENABLE_DMP"] = "1"
    fused_indexer_enabled = (
        os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT", "0")
        == "1"
    )
    lookup_maintain_enabled = (
        os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN", "1") == "1"
    )
    dual_attention_enabled = (
        os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION", "0") == "1"
    )
    if (
        sum((fused_indexer_enabled, lookup_maintain_enabled, dual_attention_enabled))
        > 1
    ):
        raise ValueError(
            "DMP Fused Indexer+KVSelect, Lookup/Maintain, and "
            "Dual-Attention modes are mutually exclusive"
        )
    # Scheme 4 uses S0=LI/Lookup/hit SFA/miss SFA, S1=Gather, S2=Maintain.
    # The following topology switch applies to scheme 2:
    # two: S0=A/B indexer + hit/miss SFA + merge + MLP;
    #      S1=A/B KVSelect + KVGather.
    # four: S0=main compute, S1=A indexer, plus separate Select/Gather streams.
    stream_mode = os.environ.setdefault(
        "VLLM_ASCEND_DMP_STREAM_MODE", DEFAULT_DMP_STREAM_MODE
    ).lower()
    if stream_mode not in ("2", "4", "two", "four"):
        raise ValueError("VLLM_ASCEND_DMP_STREAM_MODE must be one of: two, four, 2, 4")
    kv_backend = os.environ.setdefault(
        "VLLM_ASCEND_DMP_KV_BACKEND", DEFAULT_DMP_KV_BACKEND
    ).lower()
    if kv_backend not in ("local", "hixl"):
        raise ValueError("VLLM_ASCEND_DMP_KV_BACKEND must be local or hixl")
    os.environ.setdefault("VLLM_ASCEND_DMP_HIXL_CONFIG", DEFAULT_HIXL_CONFIG)
    vendor_root = ""
    runtime_python_path = ""
    if fused_indexer_enabled:
        vendor_root = os.getenv(
            "DMP_FUSED_INDEXER_OPP_PATH", DEFAULT_FUSED_INDEXER_OPP_PATH
        )
        runtime_python_path = os.getenv(
            "DMP_FUSED_INDEXER_PYTHON_PATH",
            DEFAULT_FUSED_INDEXER_PYTHON_PATH,
        )
        for dependency_python_path in (
            os.getenv(
                "DMP_LOOKUP_MAINTAIN_PYTHON_PATH",
                DEFAULT_LOOKUP_MAINTAIN_PYTHON_PATH,
            ),
            os.getenv(
                "DMP_DUAL_ATTENTION_PYTHON_PATH",
                DEFAULT_DUAL_ATTENTION_PYTHON_PATH,
            ),
        ):
            if dependency_python_path not in sys.path:
                sys.path.insert(0, dependency_python_path)
    elif lookup_maintain_enabled:
        vendor_root = os.getenv(
            "DMP_LOOKUP_MAINTAIN_OPP_PATH",
            DEFAULT_LOOKUP_MAINTAIN_OPP_PATH,
        )
        runtime_python_path = os.getenv(
            "DMP_LOOKUP_MAINTAIN_PYTHON_PATH",
            DEFAULT_LOOKUP_MAINTAIN_PYTHON_PATH,
        )
        dual_python_path = os.getenv(
            "DMP_DUAL_ATTENTION_PYTHON_PATH",
            DEFAULT_DUAL_ATTENTION_PYTHON_PATH,
        )
        if dual_python_path not in sys.path:
            sys.path.insert(0, dual_python_path)
    elif dual_attention_enabled:
        vendor_root = os.getenv(
            "DMP_DUAL_ATTENTION_OPP_PATH", DEFAULT_DUAL_ATTENTION_OPP_PATH
        )
        runtime_python_path = os.getenv(
            "DMP_DUAL_ATTENTION_PYTHON_PATH",
            DEFAULT_DUAL_ATTENTION_PYTHON_PATH,
        )

    if runtime_python_path and runtime_python_path not in sys.path:
        sys.path.insert(0, runtime_python_path)
    if dual_attention_enabled and kv_backend == "hixl":
        hixl_python_path = os.getenv("DMP_HIXL_PYTHON_PATH", DEFAULT_HIXL_PYTHON_PATH)
        if hixl_python_path not in sys.path:
            sys.path.insert(0, hixl_python_path)

    removable_vendor_roots = (
        os.path.normpath(DEFAULT_FUSED_INDEXER_OPP_PATH),
        os.path.normpath(DEFAULT_LOOKUP_MAINTAIN_OPP_PATH),
        os.path.normpath(
            os.path.join(
                DEFAULT_LOOKUP_MAINTAIN_OPP_PATH,
                "op_impl",
                "aicpu_transformer",
            )
        ),
        os.path.normpath(DEFAULT_DUAL_ATTENTION_OPP_PATH),
        os.path.normpath(LEGACY_DUAL_ATTENTION_OPP_PATH),
    )
    selected_opp_paths = []
    if vendor_root:
        selected_opp_paths.append(vendor_root)
        if fused_indexer_enabled:
            selected_opp_paths.extend(
                (
                    os.getenv(
                        "DMP_LOOKUP_MAINTAIN_OPP_PATH",
                        DEFAULT_LOOKUP_MAINTAIN_OPP_PATH,
                    ),
                    os.getenv(
                        "DMP_DUAL_ATTENTION_OPP_PATH",
                        DEFAULT_DUAL_ATTENTION_OPP_PATH,
                    ),
                )
            )
        elif lookup_maintain_enabled:
            # CANN 8.5 requires the suffixed AICPU repository before the
            # surrounding vendor OPP in ASCEND_CUSTOM_OPP_PATH.
            selected_opp_paths.insert(
                0,
                os.path.join(vendor_root, "op_impl", "aicpu_transformer"),
            )
            selected_opp_paths.append(
                os.getenv(
                    "DMP_DUAL_ATTENTION_OPP_PATH",
                    DEFAULT_DUAL_ATTENTION_OPP_PATH,
                )
            )
    removable_vendor_roots = (
        *removable_vendor_roots,
        *(os.path.normpath(path) for path in selected_opp_paths),
    )
    current_paths = [
        path
        for path in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if path and os.path.normpath(path) not in removable_vendor_roots
    ]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        [*selected_opp_paths, *current_paths]
    )

    # The bundled extension opens libcust_opapi.so by name. Keep its library
    # first and keep the selected custom copy out of LD_LIBRARY_PATH.
    selected_op_api_path = (
        os.path.normpath(os.path.join(vendor_root, "op_api", "lib"))
        if vendor_root
        else ""
    )
    legacy_dual_op_api_path = os.path.normpath(
        os.path.join(LEGACY_DUAL_ATTENTION_OPP_PATH, "op_api", "lib")
    )
    persistent_dual_op_api_path = os.path.normpath(
        os.path.join(
            os.getenv(
                "DMP_DUAL_ATTENTION_OPP_PATH",
                DEFAULT_DUAL_ATTENTION_OPP_PATH,
            ),
            "op_api",
            "lib",
        )
    )
    vllm_op_api_path = os.path.normpath(
        os.getenv("VLLM_ASCEND_OP_API_PATH", DEFAULT_VLLM_ASCEND_OP_API_PATH)
    )
    ld_paths = [
        path
        for path in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
        and os.path.normpath(path)
        not in (
            selected_op_api_path,
            legacy_dual_op_api_path,
            persistent_dual_op_api_path,
            vllm_op_api_path,
        )
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([vllm_op_api_path, *ld_paths])
    return vendor_root


def preload_dmp_operator_libraries(vendor_root: str) -> None:
    """Load and validate the selected custom-op library separately."""
    vllm_op_api_path = os.getenv(
        "VLLM_ASCEND_OP_API_PATH", DEFAULT_VLLM_ASCEND_OP_API_PATH
    )
    old_library_path = os.path.join(vllm_op_api_path, "libcust_opapi.so")
    try:
        # vllm_ascend_C links against this base opapi. Make its symbols global
        # before loading the isolated DMP libraries with RTLD_DEEPBIND.
        old_library = ctypes.CDLL(old_library_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load the bundled vllm-ascend operator library: {old_library_path}"
        ) from exc
    if not hasattr(old_library, "aclnnAddRmsNormBias"):
        raise RuntimeError(
            f"Bundled operator library does not contain aclnnAddRmsNormBias: {old_library_path}"
        )

    if not vendor_root:
        _PRELOADED_OPERATOR_LIBRARIES.append(old_library)
        return

    new_library_path = os.path.join(vendor_root, "op_api", "lib", "libcust_opapi.so")
    try:
        new_library_mode = ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
        new_library = ctypes.CDLL(new_library_path, mode=new_library_mode)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load the selected DMP operator library: {new_library_path}"
        ) from exc
    lookup_maintain_enabled = (
        os.getenv("VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN", "0") == "1"
    )
    fused_indexer_enabled = (
        os.getenv("VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT", "0") == "1"
    )
    if lookup_maintain_enabled:
        required_symbols = [
            "aclnnAsuHbmIndexLookup",
            "aclnnAsuHbmIndexMaintainAicpu",
            "aclnnDmpLookupKvGather",
        ]
    elif fused_indexer_enabled:
        required_symbols = ["aclnnLightningIndexerDecodeUpdatePool"]
    else:
        required_symbols = ["aclnnDmpSparseFlashAttention"]
        if os.getenv("VLLM_ASCEND_DMP_KV_BACKEND", "local").lower() == "local":
            required_symbols.append("aclnnKVSelect")
    missing_symbols = [
        symbol for symbol in required_symbols if not hasattr(new_library, symbol)
    ]
    if missing_symbols:
        raise RuntimeError(
            "Selected DMP operator library is stale; missing "
            f"{', '.join(missing_symbols)}: {new_library_path}"
        )
    loaded_libraries = [old_library, new_library]
    if lookup_maintain_enabled:
        dual_vendor_root = os.getenv(
            "DMP_DUAL_ATTENTION_OPP_PATH",
            DEFAULT_DUAL_ATTENTION_OPP_PATH,
        )
        dual_library_path = os.path.join(
            dual_vendor_root, "op_api", "lib", "libcust_opapi.so"
        )
        try:
            dual_library = ctypes.CDLL(
                dual_library_path,
                mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
            )
        except OSError as exc:
            raise RuntimeError(
                "Lookup/Maintain scheme 4 also requires the persistent "
                f"Dual-Attention operator library: {dual_library_path}"
            ) from exc
        dual_symbols = (
            "aclnnDmpSparseFlashAttention",
            "aclnnDaAttentionMerge",
        )
        missing_dual_symbols = [
            symbol for symbol in dual_symbols if not hasattr(dual_library, symbol)
        ]
        if missing_dual_symbols:
            raise RuntimeError(
                "Dual-Attention library required by Lookup/Maintain is stale; "
                f"missing {', '.join(missing_dual_symbols)}: {dual_library_path}"
            )
        loaded_libraries.append(dual_library)
    elif fused_indexer_enabled:
        dependency_libraries = (
            (
                "KVIO",
                os.getenv(
                    "DMP_LOOKUP_MAINTAIN_OPP_PATH",
                    DEFAULT_LOOKUP_MAINTAIN_OPP_PATH,
                ),
                ("aclnnDmpLookupKvGather",),
            ),
            (
                "Dual-Attention SFA",
                os.getenv(
                    "DMP_DUAL_ATTENTION_OPP_PATH",
                    DEFAULT_DUAL_ATTENTION_OPP_PATH,
                ),
                ("aclnnDmpSparseFlashAttention",),
            ),
        )
        for label, dependency_root, symbols in dependency_libraries:
            dependency_library_path = os.path.join(
                dependency_root, "op_api", "lib", "libcust_opapi.so"
            )
            try:
                dependency_library = ctypes.CDLL(
                    dependency_library_path,
                    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Fused scheme 3 requires the {label} operator library: "
                    f"{dependency_library_path}"
                ) from exc
            missing_dependency_symbols = [
                symbol
                for symbol in symbols
                if not hasattr(dependency_library, symbol)
            ]
            if missing_dependency_symbols:
                raise RuntimeError(
                    f"{label} library required by fused scheme 3 is stale; "
                    f"missing {', '.join(missing_dependency_symbols)}: "
                    f"{dependency_library_path}"
                )
            loaded_libraries.append(dependency_library)
    _PRELOADED_OPERATOR_LIBRARIES.extend(loaded_libraries)


def validate_context_length(
    prompt_tokens: int,
    output_tokens: int,
    max_model_len: int,
) -> None:
    required = prompt_tokens + output_tokens
    if required > max_model_len:
        raise ValueError(
            f"max_model_len={max_model_len} is smaller than the required "
            f"context length {prompt_tokens}+{output_tokens}={required}"
        )


def load_seed_texts(prompt_file: Path) -> tuple[str, str]:
    """Load two different source texts for warmup and measured requests."""
    texts: list[str] = []
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            texts.append(item["question"])
            if len(texts) == 2:
                break

    if not texts:
        raise RuntimeError(f"No prompt found in {prompt_file}")
    if len(texts) == 1:
        texts.append(texts[0] + "\nMeasured request.")
    return texts[0], texts[1]


def build_token_prompts(
    tokenizer,
    seed_text: str,
    batch_size: int,
    prompt_tokens: int,
    label: str,
):
    """Build unique requests with exactly prompt_tokens token IDs each."""
    from vllm import TokensPrompt

    body_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not body_ids:
        raise RuntimeError("Tokenizer produced an empty seed prompt")

    prompts = []
    for request_idx in range(batch_size):
        prefix = f"{label} request {request_idx}\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(prefix_ids) >= prompt_tokens:
            prompt_ids = prefix_ids[:prompt_tokens]
        else:
            remaining = prompt_tokens - len(prefix_ids)
            repeats = (remaining + len(body_ids) - 1) // len(body_ids)
            prompt_ids = prefix_ids + (body_ids * repeats)[:remaining]

        if len(prompt_ids) != prompt_tokens:
            raise RuntimeError(
                f"Expected {prompt_tokens} prompt tokens, got {len(prompt_ids)}"
            )
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
    return prompts
