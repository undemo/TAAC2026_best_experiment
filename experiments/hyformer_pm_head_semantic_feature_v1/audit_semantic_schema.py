"""Read-only schema/fid audit for H-SemanticFeature-v1.

This script inspects a flat parquet file directly. It does not require
schema.json and does not participate in training.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


USER_GROUPS_V3 = {
    "U_low_context": [1, 48, 49, 50, 51, 52, 55, 58, 59, 60],
    "U_profile_stat_shared": [3, 4, 15, 53, 54, 56, 57, 62, 63, 64, 65, 66, 86],
    "U_compact_tail_flags": [
        80, 82, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
        102, 103, 104, 105, 106, 107, 108, 109,
    ],
}

ITEM_GROUPS_V3 = {
    "I_low_card": [9, 13, 81, 83, 84, 85],
    "I_mid_behavior_a": [10, 11],
    "I_mid_behavior_b": [12, 16],
    "I_high_id_like": [5, 6, 7, 8],
}

DENSE_EMBEDDING_LIKE_FIDS = [61, 87, 89, 90, 91]
DENSE_STAT_LIKE_FIDS = [62, 63, 64, 65, 66]


def _is_list_type(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        return all(_value_is_missing(v) for v in value)
    try:
        return int(value) == -1
    except (TypeError, ValueError, OverflowError):
        return False


def _flatten_values(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(value)
        else:
            out.append(value)
    return out


def _safe_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if not isinstance(value, (list, tuple)):
        value = [value]
    arr = np.asarray(value, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _extract_columns(schema: pa.Schema) -> Dict[str, Dict[int, str]]:
    patterns = {
        "user_int": re.compile(r"^user_int_feats_(\d+)$"),
        "item_int": re.compile(r"^item_int_feats_(\d+)$"),
        "user_dense": re.compile(r"^user_dense_feats_(\d+)$"),
        "item_dense": re.compile(r"^item_dense_feats_(\d+)$"),
    }
    result: Dict[str, Dict[int, str]] = {k: {} for k in patterns}
    for name in schema.names:
        for family, pattern in patterns.items():
            match = pattern.match(name)
            if match:
                result[family][int(match.group(1))] = name
                break
    return result


def _feature_stats(table: pa.Table, column_name: str) -> Dict[str, Any]:
    col = table[column_name]
    field_type = col.type
    py_values = col.to_pylist()
    n = len(py_values)
    missing = sum(1 for v in py_values if _value_is_missing(v))
    flat = _flatten_values(py_values)
    zero = 0
    minus1 = 0
    nan = 0
    for value in flat:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            nan += 1
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if numeric == 0:
            zero += 1
        if numeric == -1:
            minus1 += 1
    denom = max(len(flat), 1)
    kind = "list" if _is_list_type(field_type) else "scalar"
    if "float" in str(field_type).lower() or "double" in str(field_type).lower():
        kind = "dense_list" if _is_list_type(field_type) else "dense_scalar"
    stats = {
        "column": column_name,
        "arrow_type": str(field_type),
        "kind": kind,
        "num_rows": n,
        "missing_ratio_demo": missing / max(n, 1),
        "zero_ratio_demo": zero / denom,
        "minus1_ratio_demo": minus1 / denom,
        "nan_ratio_demo": nan / denom,
    }
    if kind.startswith("dense"):
        norms = []
        max_abs = []
        log_norms = []
        for value in py_values:
            arr = _safe_float_array(value)
            if arr.size == 0:
                continue
            signed_log = np.sign(arr) * np.log1p(np.abs(arr))
            norms.append(float(np.linalg.norm(arr)))
            max_abs.append(float(np.max(np.abs(arr))))
            log_norms.append(float(np.linalg.norm(signed_log)))
        if norms:
            stats.update({
                "dense_raw_norm_mean_demo": float(np.mean(norms)),
                "dense_raw_norm_max_demo": float(np.max(norms)),
                "dense_max_abs_demo": float(np.max(max_abs)),
                "dense_signed_log1p_norm_mean_demo": float(np.mean(log_norms)),
                "dense_signed_log1p_norm_max_demo": float(np.max(log_norms)),
            })
        else:
            stats.update({
                "dense_raw_norm_mean_demo": None,
                "dense_raw_norm_max_demo": None,
                "dense_max_abs_demo": None,
                "dense_signed_log1p_norm_mean_demo": None,
                "dense_signed_log1p_norm_max_demo": None,
            })
    return stats


def _load_v2_fids(path: Optional[str]) -> Dict[str, set]:
    if not path:
        return {"user_int": set(), "item_int": set()}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    user = set()
    item = set()
    for fids in cfg.get("user_ns_groups", {}).values():
        user.update(int(fid) for fid in fids)
    for fids in cfg.get("item_ns_groups", {}).values():
        item.update(int(fid) for fid in fids)
    return {"user_int": user, "item_int": item}


def _audit_group(
    group_type: str,
    group_name: str,
    fids: List[int],
    columns: Dict[int, str],
    v2_fids: set,
) -> Dict[str, Any]:
    missing_demo = [fid for fid in fids if fid not in columns]
    missing_v2 = [fid for fid in missing_demo if fid not in v2_fids]
    duplicates = sorted({fid for fid in fids if fids.count(fid) > 1})
    return {
        "type": "semantic_group_audit_demo",
        "group_name": group_name,
        "group_type": group_type,
        "fids": fids,
        "num_fids": len(fids),
        "all_fids_present_in_demo": len(missing_demo) == 0,
        "missing_fids_in_demo": missing_demo,
        "missing_fids_not_in_v2": missing_v2,
        "duplicated_fids": duplicates,
        "constructible_from_demo_or_v2": len(missing_v2) == 0 and not duplicates,
    }


def _coverage_error(groups: Dict[str, List[int]], available_fids: set, v2_fids: set) -> List[int]:
    required = set()
    for fids in groups.values():
        required.update(fids)
    return sorted(fid for fid in required if fid not in available_fids and fid not in v2_fids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--v2-groups-json", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.parquet)
    table = parquet.read()
    schema = table.schema
    columns_by_family = _extract_columns(schema)
    v2_fids = _load_v2_fids(args.v2_groups_json)

    feature_stats = {
        family: {
            str(fid): _feature_stats(table, col)
            for fid, col in sorted(columns.items())
        }
        for family, columns in columns_by_family.items()
    }

    group_events = []
    for name, fids in USER_GROUPS_V3.items():
        group_events.append(_audit_group(
            "user_int", name, fids, columns_by_family["user_int"], v2_fids["user_int"]))
    for name, fids in ITEM_GROUPS_V3.items():
        group_events.append(_audit_group(
            "item_int", name, fids, columns_by_family["item_int"], v2_fids["item_int"]))

    dense_group_audit = {
        "type": "dense_group_audit_demo",
        "embedding_like_fids": DENSE_EMBEDDING_LIKE_FIDS,
        "stat_like_fids": DENSE_STAT_LIKE_FIDS,
        "missing_embedding_like_fids_in_demo": [
            fid for fid in DENSE_EMBEDDING_LIKE_FIDS
            if fid not in columns_by_family["user_dense"]
        ],
        "missing_stat_like_fids_in_demo": [
            fid for fid in DENSE_STAT_LIKE_FIDS
            if fid not in columns_by_family["user_dense"]
        ],
    }
    can_construct_ns_groups = (
        not _coverage_error(USER_GROUPS_V3, set(columns_by_family["user_int"]), v2_fids["user_int"])
        and not _coverage_error(ITEM_GROUPS_V3, set(columns_by_family["item_int"]), v2_fids["item_int"])
        and all(not event["duplicated_fids"] for event in group_events)
    )

    seq_columns = [
        name for name in schema.names
        if re.match(r"^seq_[a-z]_", name) or re.match(r"^.*_seq_.*", name)
    ]
    result = {
        "type": "semantic_schema_audit_demo",
        "parquet": args.parquet,
        "num_rows": table.num_rows,
        "num_columns": table.num_columns,
        "all_columns": schema.names,
        "identified_columns": {
            family: {str(fid): col for fid, col in sorted(columns.items())}
            for family, columns in columns_by_family.items()
        },
        "meta_columns": {
            "user_id": "user_id" if "user_id" in schema.names else None,
            "item_id": "item_id" if "item_id" in schema.names else None,
            "timestamp": "timestamp" if "timestamp" in schema.names else None,
            "seq_columns": seq_columns,
        },
        "feature_stats": feature_stats,
        "group_audit": group_events,
        "dense_group_audit": dense_group_audit,
        "can_construct_ns_groups_h_semantic_feature_v1": can_construct_ns_groups,
        "warning": (
            "demo_1000 statistics are for implementation feasibility only; "
            "full training must log train-row-group stats again."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
