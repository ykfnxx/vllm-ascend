import argparse
import os
import sys
from dataclasses import asdict

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from gather_selection_kv_cache import register_torch_ops
except Exception:  # pragma: no cover - optional bridge
    register_torch_ops = None


ESS_KEYS = {
    "device",
    "batch_size",
    "seq_len",
    "kv_max_seq_len",
    "block_size",
    "index_topk",
    "gpu_sparse_capacity_tokens",
    "warmup_steps",
    "decode_steps",
    "seed",
    "enable_overlap",
    "num_heads",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "indexer_head_dim",
    "layout_query",
    "layout_key",
    "layout_kv",
    "enable_custom_indexer",
    "enable_custom_sparse_attn",
}
BASELINE_KEYS = {
    "device",
    "batch_size",
    "seq_len",
    "kv_max_seq_len",
    "block_size",
    "index_topk",
    "seed",
    "indexer_num_heads",
    "sparse_attn_num_heads",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "indexer_head_dim",
    "layout_query",
    "layout_key",
    "layout_kv",
    "enable_custom_indexer",
    "enable_custom_sparse_attn",
    "topk_reuse_rate",
}

DUAL_ATTENTION_KEYS = BASELINE_KEYS


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--mode", choices=["dbo", "baseline", "dual_attention"], default="dbo")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-csv", default="outputs/csv/dbo_steps.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--kv-max-seq-len", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--index-topk", type=int, default=None)
    parser.add_argument("--gpu-sparse-capacity-tokens", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enable-overlap", type=int, choices=[0, 1], default=None)
    parser.add_argument("--num-heads", type=int, default=None, help="ESS/dbo indexer heads")
    parser.add_argument("--indexer-num-heads", type=int, default=None)
    parser.add_argument("--sparse-attn-num-heads", type=int, default=None)
    parser.add_argument("--kv-lora-rank", type=int, default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=None)
    parser.add_argument("--indexer-head-dim", type=int, default=None)
    parser.add_argument("--layout-query", default=None)
    parser.add_argument("--layout-key", default=None)
    parser.add_argument("--layout-kv", default=None)
    parser.add_argument("--enable-custom-indexer", type=int, choices=[0, 1], default=None)
    parser.add_argument("--enable-custom-sparse-attn", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--topk-reuse-rate",
        type=float,
        default=None,
        help="Gather selection pool hit ratio target (0=cold/reinit each step, 1=keep status)",
    )
    parser.add_argument("--ess-debug-stages", action="store_true")
    return parser.parse_args()


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def _cli_overrides(args) -> dict:
    return {
        "device": args.device,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "kv_max_seq_len": args.kv_max_seq_len,
        "block_size": args.block_size,
        "index_topk": args.index_topk,
        "gpu_sparse_capacity_tokens": args.gpu_sparse_capacity_tokens,
        "warmup_steps": args.warmup_steps,
        "decode_steps": args.decode_steps,
        "seed": args.seed,
        "enable_overlap": bool(args.enable_overlap) if args.enable_overlap is not None else None,
        "num_heads": args.num_heads,
        "indexer_num_heads": args.indexer_num_heads,
        "sparse_attn_num_heads": args.sparse_attn_num_heads,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "indexer_head_dim": args.indexer_head_dim,
        "layout_query": args.layout_query,
        "layout_key": args.layout_key,
        "layout_kv": args.layout_kv,
        "enable_custom_indexer": bool(args.enable_custom_indexer) if args.enable_custom_indexer is not None else None,
        "enable_custom_sparse_attn": bool(args.enable_custom_sparse_attn) if args.enable_custom_sparse_attn is not None else None,
        "topk_reuse_rate": args.topk_reuse_rate,
    }


def _build_config(config_cls, valid_keys: set[str], args):
    cfg_dict = asdict(config_cls())
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)
        unknown = [k for k in yaml_cfg.keys() if k not in valid_keys]
        if unknown:
            raise ValueError(f"Unknown keys in config {args.config}: {unknown}")
        cfg_dict.update(yaml_cfg)
    for key, value in _cli_overrides(args).items():
        if key in valid_keys and value is not None:
            cfg_dict[key] = value
    return config_cls(**cfg_dict)


def main():
    args = parse_args()
    if args.ess_debug_stages:
        os.environ["ESS_DEBUG_STAGES"] = "1"
    if register_torch_ops is not None:
        register_torch_ops()

    if args.mode == "baseline":
        from baseline import BaselineConfig, _print_step, run_baseline_pipeline

        config = _build_config(BaselineConfig, BASELINE_KEYS, args)
        for row in run_baseline_pipeline(config):
            _print_step(row)
        return

    if args.mode == "dual_attention":
        from baseline import BaselineConfig
        from dual_attention import print_dual_attention_step, run_dual_attention_pipeline

        config = _build_config(BaselineConfig, DUAL_ATTENTION_KEYS, args)
        for row in run_dual_attention_pipeline(config):
            print_dual_attention_step(row)
        return

    from dbo.pipeline import ESSConfig, export_csv, init_cache_state, print_summary, run_decode_pipeline_overlap, summarize

    config = _build_config(ESSConfig, ESS_KEYS, args)
    state = init_cache_state(config.gpu_sparse_capacity_tokens)
    runtime_status = {}
    metrics = run_decode_pipeline_overlap(config, state, runtime_status=runtime_status)
    summary = summarize(metrics)
    if config.enable_custom_sparse_attn or config.enable_da_overlap or config.enable_dba_overlap:
        summary.update(runtime_status)
    print_summary(summary)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    export_csv(metrics, args.out_csv)
    print(f"\nstep metrics exported to: {args.out_csv}")


if __name__ == "__main__":
    main()
