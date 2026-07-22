
import argparse
import json
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
    build_token_prompts,
    configure_dmp_runtime,
    load_seed_texts,
    preload_dmp_operator_libraries,
    validate_context_length,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--visible-devices", default=DEFAULT_VISIBLE_DEVICES)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE
    )
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--enable-npugraph-ex", action="store_true")
    parser.add_argument(
        "--prompt-file", default="prompts/datasets/GSM8K-in2000-bs1408.jsonl"
    )
    return parser.parse_args()


args = parse_args()
validate_context_length(args.prompt_tokens, args.max_tokens, args.max_model_len)

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = Path(args.profile_dir).resolve()
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_FILE = Path(args.prompt_file)
if not PROMPT_FILE.is_absolute():
    PROMPT_FILE = SCRIPT_DIR / PROMPT_FILE

# Must be set before torch_npu / vLLM imports.
CUSTOM_OPP_PATH = configure_dmp_runtime(args.visible_devices)
preload_dmp_operator_libraries(CUSTOM_OPP_PATH)
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"

import torch_npu  # noqa: E402,F401
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import CompilationConfig, CompilationMode  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402

_, profile_seed = load_seed_texts(PROMPT_FILE)

sampling_params = SamplingParams(
    temperature=0,
    top_p=1,
    max_tokens=args.max_tokens,
    ignore_eos=True,
)

compilation_config = CompilationConfig(
    mode=CompilationMode.VLLM_COMPILE,
    backend="",
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    cudagraph_capture_sizes=[args.batch_size],
    compile_sizes=[args.batch_size],
)

print(
    json.dumps(
        {
            "kind": "profile_config",
            "mode": args.mode,
            "profile_dir": str(PROFILE_DIR),
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "prompt_tokens": args.prompt_tokens,
            "max_model_len": args.max_model_len,
            "visible_devices": args.visible_devices,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_npugraph_ex": args.enable_npugraph_ex,
            "dmp_dual_attention": True,
            "custom_opp_path": CUSTOM_OPP_PATH,
            "prime_prefix_cache_before_profile": True,
        },
        ensure_ascii=False,
    ),
    flush=True,
)

llm = LLM(
    model="/models/GLM-5.1",
    served_model_name="glm-5",
    quantization=None,
    dtype="bfloat16",
    disable_log_stats=True,
    load_format="safetensors",
    skip_tokenizer_init=False,
    tensor_parallel_size=args.tensor_parallel_size,
    pipeline_parallel_size=1,
    enforce_eager=args.mode == "eager",
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    max_model_len=args.max_model_len,
    max_num_seqs=args.batch_size,
    max_num_batched_tokens=4096,
    speculative_config=None,
    compilation_config=compilation_config,
    additional_config={
        "enable_npugraph_ex": args.enable_npugraph_ex,
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
        # These fields are accepted by vLLM ProfilerConfig. vLLM-Ascend's
        # current worker profiler mostly relies on start/stop, but keeping
        # a one-active-iteration intent makes the run configuration explicit.
        "wait_iterations": 0,
        "warmup_iterations": 0,
        "active_iterations": 1,
        "delay_iterations": 0,
        "max_iterations": 1,
    },
    trust_remote_code=True,
)

tokenizer = llm.get_tokenizer()
profile_prompts = build_token_prompts(
    tokenizer,
    profile_seed,
    args.batch_size,
    args.prompt_tokens,
    "profile",
)

print(
    json.dumps(
        {"kind": "prefix_cache_prime_start", "mode": args.mode},
        ensure_ascii=False,
    ),
    flush=True,
)
# Use exactly the same requests as the measured run. This expensive 128K
# prefill happens before profiling and leaves its KV blocks in prefix cache.
_ = llm.generate(profile_prompts, sampling_params)
print(
    json.dumps(
        {"kind": "prefix_cache_prime_done", "mode": args.mode},
        ensure_ascii=False,
    ),
    flush=True,
)

# Give asynchronous graph-capture bookkeeping a short gap before profiling.
time.sleep(2)

print(
    json.dumps({"kind": "profile_start", "mode": args.mode}, ensure_ascii=False),
    flush=True,
)
llm.start_profile()
outputs = llm.generate(profile_prompts, sampling_params)

# In normal vLLM use, generate() returns after worker execution is complete.
# The short wait avoids stopping exactly on the profiler daemon boundary.
time.sleep(2)
llm.stop_profile()
print(
    json.dumps({"kind": "profile_stop", "mode": args.mode}, ensure_ascii=False),
    flush=True,
)

output_tokens = 0
for output in outputs:
    if output.outputs:
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is not None:
            output_tokens += len(token_ids)

print(
    json.dumps(
        {
            "kind": "profile_summary",
            "mode": args.mode,
            "num_outputs": len(outputs),
            "output_tokens": output_tokens,
            "profile_dir": str(PROFILE_DIR),
        },
        ensure_ascii=False,
    ),
    flush=True,
)

# Let torch_npu profiler background work flush raw files before process exit.
time.sleep(20)
