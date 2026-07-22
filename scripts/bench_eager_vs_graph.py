
import argparse
import json
import os
import statistics
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
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--visible-devices", default=DEFAULT_VISIBLE_DEVICES)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE
    )
    parser.add_argument("--enable-npugraph-ex", action="store_true")
    parser.add_argument(
        "--prompt-file", default="prompts/datasets/GSM8K-in2000-bs1408.jsonl"
    )
    return parser.parse_args()


args = parse_args()
validate_context_length(args.prompt_tokens, args.max_tokens, args.max_model_len)

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = Path(args.prompt_file)
if not PROMPT_FILE.is_absolute():
    PROMPT_FILE = SCRIPT_DIR / PROMPT_FILE

# These must be set before importing torch_npu / vLLM.
CUSTOM_OPP_PATH = configure_dmp_runtime(args.visible_devices)
preload_dmp_operator_libraries(CUSTOM_OPP_PATH)
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"

import torch_npu  # noqa: E402,F401
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import CompilationConfig, CompilationMode  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402


def count_output_tokens(outputs):
    total = 0
    for output in outputs:
        if not output.outputs:
            continue
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
    return total


warmup_seed, run_seed = load_seed_texts(PROMPT_FILE)

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

init_start = time.perf_counter()
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
    trust_remote_code=True,
)
init_seconds = time.perf_counter() - init_start

tokenizer = llm.get_tokenizer()
warmup_prompts = build_token_prompts(
    tokenizer,
    warmup_seed,
    args.batch_size,
    args.prompt_tokens,
    "warmup",
)
run_prompts = build_token_prompts(
    tokenizer,
    run_seed,
    args.batch_size,
    args.prompt_tokens,
    "measured",
)

for i in range(args.warmup):
    _ = llm.generate(warmup_prompts, sampling_params)
    print(
        json.dumps(
            {"kind": "warmup_done", "mode": args.mode, "iter": i}, ensure_ascii=False
        ),
        flush=True,
    )

rows = []
for i in range(args.repeat):
    start = time.perf_counter()
    outputs = llm.generate(run_prompts, sampling_params)
    elapsed = time.perf_counter() - start
    output_tokens = count_output_tokens(outputs)
    row = {
        "kind": "bench_iter",
        "mode": args.mode,
        "iter": i,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "output_tokens_per_s": output_tokens / elapsed if elapsed > 0 else None,
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

times = [r["elapsed_s"] for r in rows]
speeds = [
    r["output_tokens_per_s"] for r in rows if r["output_tokens_per_s"] is not None
]
summary = {
    "kind": "bench_summary",
    "mode": args.mode,
    "visible_devices": args.visible_devices,
    "enable_npugraph_ex": args.enable_npugraph_ex,
    "batch_size": args.batch_size,
    "max_tokens": args.max_tokens,
    "prompt_tokens": args.prompt_tokens,
    "max_model_len": args.max_model_len,
    "tensor_parallel_size": args.tensor_parallel_size,
    "warmup": args.warmup,
    "repeat": args.repeat,
    "init_seconds": init_seconds,
    "mean_elapsed_s": statistics.mean(times),
    "median_elapsed_s": statistics.median(times),
    "mean_output_tokens_per_s": statistics.mean(speeds),
    "median_output_tokens_per_s": statistics.median(speeds),
}
print("BENCH_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
