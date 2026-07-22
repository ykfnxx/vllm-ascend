#!/usr/bin/env bash

# Resolve the offline GLM runtime wheels from either the new host mount or the
# legacy paths used by the original A2 container scripts.
dmp_resolve_model_runtime_wheels() {
    local wheel_dir="${DMP_WHEEL_DIR:-/dmp-host}"
    local transformers="${DMP_TRANSFORMERS_WHEEL:-}"
    local huggingface="${DMP_HUGGINGFACE_HUB_WHEEL:-}"
    local search_dir candidate
    local -a search_dirs=(
        "$wheel_dir"
        "$wheel_dir/wheels"
        /workspace
        /workspace/scripts/wheels
    )

    if [[ -z "$transformers" ]]; then
        for search_dir in "${search_dirs[@]}"; do
            [[ -d "$search_dir" ]] || continue
            candidate="$(find "$search_dir" -maxdepth 1 -type f \
                -name 'transformers-5.2.0*.whl' -print \
                | LC_ALL=C sort | tail -1)"
            if [[ -n "$candidate" ]]; then
                transformers="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$huggingface" ]]; then
        for search_dir in "${search_dirs[@]}"; do
            [[ -d "$search_dir" ]] || continue
            candidate="$(find "$search_dir" -maxdepth 1 -type f \
                -name 'huggingface_hub-1.22.0*.whl' -print \
                | LC_ALL=C sort | tail -1)"
            if [[ -n "$candidate" ]]; then
                huggingface="$candidate"
                break
            fi
        done
    fi

    DMP_RESOLVED_TRANSFORMERS_WHEEL="$transformers"
    DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL="$huggingface"
}
