
import os
import time
from pathlib import Path

from standard_run_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_PROMPT_TOKENS,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_VISIBLE_DEVICES,
    activate_model_runtime,
    build_token_prompts,
    configure_dmp_runtime,
    load_seed_texts,
    preload_dmp_operator_libraries,
    validate_context_length,
)

# =============================================================================
# Runtime environment
# =============================================================================

# Use one physical NPU card by default.
# This must be set before importing torch_npu / vLLM.
VISIBLE_DEVICES = os.getenv("VISIBLE_DEVICES", DEFAULT_VISIBLE_DEVICES)
SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMERS_RUNTIME = activate_model_runtime(SCRIPT_DIR)
# Defaults to scheme 4. The wrapper can select scheme 3 with DMP_SCHEME=3:
# S0=fused Indexer+Select update, S1=one local mock KVIO per microbatch,
# then S0 runs selected-cache SFA. The linked fused branch has no separate
# AICPU Maintain because token-to-slot update is already fused into S0.
os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN", "1")
os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT", "0")
os.environ.setdefault("VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION", "0")
CUSTOM_OPP_PATH = configure_dmp_runtime(VISIBLE_DEVICES)
preload_dmp_operator_libraries(CUSTOM_OPP_PATH)

# Profiling output.
PROFILE_DIR = SCRIPT_DIR / "vllm_profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["VLLM_TORCH_PROFILER_DIR"] = str(PROFILE_DIR)

# Ascend log level. Keep this as the original script used.
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"

# Optional graph dump, keep disabled unless debugging graph capture.
# os.environ["DUMP_GE_GRAPH"] = "2"
# os.environ["DUMP_GRAPH_LEVEL"] = "3"
# os.environ["DUMP_GRAPH_PATH"] = str(SCRIPT_DIR / "dump_graph")
# (SCRIPT_DIR / "dump_graph").mkdir(parents=True, exist_ok=True)

# Optional debug knobs.
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
# os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "0"
# os.environ["VLLM_DISABLE_FUSION"] = "1"

import torch_npu  # noqa: E402,F401
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import CompilationConfig, CompilationMode  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402

# =============================================================================
# Prompt setup
# =============================================================================

PROMPT_FILE = SCRIPT_DIR / "prompts/datasets/GSM8K-in2000-bs1408.jsonl"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
PROMPT_TOKENS = int(os.getenv("PROMPT_TOKENS", str(DEFAULT_PROMPT_TOKENS)))
OUTPUT_TOKENS = int(os.getenv("MAX_TOKENS", str(DEFAULT_OUTPUT_TOKENS)))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))
TENSOR_PARALLEL_SIZE = int(
    os.getenv("TENSOR_PARALLEL_SIZE", str(DEFAULT_TENSOR_PARALLEL_SIZE))
)
MODEL_PATH = os.getenv("MODEL_PATH", "/models/GLM-5.1-w4a8")
MODEL_QUANTIZATION = os.getenv("MODEL_QUANTIZATION", "ascend").strip().lower()
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "auto")
ENABLE_EXPERT_PARALLEL = os.getenv("ENABLE_EXPERT_PARALLEL", "0") == "1"
quantization = None if MODEL_QUANTIZATION in ("", "none") else MODEL_QUANTIZATION
_, measured_seed = load_seed_texts(PROMPT_FILE)
validate_context_length(PROMPT_TOKENS, OUTPUT_TOKENS, MAX_MODEL_LEN)


# =============================================================================
# vLLM setup
# =============================================================================

sampling_params = SamplingParams(
    temperature=0,
    top_p=1,
    max_tokens=OUTPUT_TOKENS,
    ignore_eos=True,
)

compilation_config = CompilationConfig(
    mode=CompilationMode.VLLM_COMPILE,
    backend="",
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    cudagraph_capture_sizes=[BATCH_SIZE],
    compile_sizes=[BATCH_SIZE],
)

llm = LLM(
    model=MODEL_PATH,
    served_model_name="glm-5",
    quantization=quantization,
    dtype=MODEL_DTYPE,
    disable_log_stats=True,
    load_format="safetensors",
    skip_tokenizer_init=False,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    pipeline_parallel_size=1,
    enable_expert_parallel=ENABLE_EXPERT_PARALLEL,
    enforce_eager=False,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=BATCH_SIZE,
    max_num_batched_tokens=4096,
    speculative_config=None,
    compilation_config=compilation_config,
    additional_config={
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
        "fuse_muls_add": True,
        "ascend_compilation_config": {
            "mode": "CompilationMode.VLLM_COMPILE",
            "backend": "",
            "cudagraph_mode": "CUDAGraphMode.FULL_DECODE_ONLY",
        },
        "topo_sorting_mode": "stable",
    },
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": str(PROFILE_DIR),
    },
    trust_remote_code=True,
)

tokenizer = llm.get_tokenizer()
run_prompt = build_token_prompts(
    tokenizer, measured_seed, BATCH_SIZE, PROMPT_TOKENS, "measured"
)


# =============================================================================
# Run
# =============================================================================

# Prime prefix cache with the exact measured requests before profiling.
_ = llm.generate(run_prompt, sampling_params)

llm.start_profile()
outputs = llm.generate(run_prompt, sampling_params)
llm.stop_profile()

# Uncomment if you need to inspect generated text.
# for output in outputs:
#     print(f"Prompt: {output.prompt!r}")
#     print(f"Generated text: {output.outputs[0].text!r}")

# Give profiler background work a little time to flush.
time.sleep(10)
