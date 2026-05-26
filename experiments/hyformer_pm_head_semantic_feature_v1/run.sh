#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- H-SemanticFeature-v1: 3 user group tokens + 4 item group tokens ----
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --d_model 68 \
    --user_ns_tokens 3 \
    --item_ns_tokens 4 \
    --num_queries 2 \
    --pm_head_enabled \
    --time_token_enabled \
    --pm_feature_dim 64 \
    --pm_feature_dropout 0.05 \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups_h_semantic_feature_v1.json" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
