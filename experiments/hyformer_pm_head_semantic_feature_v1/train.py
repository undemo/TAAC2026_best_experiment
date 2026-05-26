"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')
    parser.add_argument('--tf_events_dir', type=str, default=None,
                        help='TensorBoard event directory (env: TRAIN_TF_EVENTS_PATH; defaults to <log_dir>/tf_events)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='Stop training after N optimizer steps for smoke tests (0 = unlimited)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups_h_semantic_feature_v1.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # PM head-only experiment. This only concatenates PM features before the
    # classifier and does not modify any HyFormer backbone component.
    parser.add_argument('--pm_head_enabled', action='store_true', default=False,
                        help='Enable head-only PM features before the classifier')
    parser.add_argument('--pm_feature_dim', type=int, default=64,
                        help='Width of the PM feature vector concatenated to the HyFormer representation')
    parser.add_argument('--pm_feature_dropout', type=float, default=0.05,
                        help='Dropout applied inside and after the PM head feature extractor')
    parser.add_argument('--pm_feature_norm_enabled', dest='pm_feature_norm_enabled',
                        action='store_true', default=True,
                        help='Enable LayerNorm on the exported PM feature vector')
    parser.add_argument('--no_pm_feature_norm', dest='pm_feature_norm_enabled',
                        action='store_false',
                        help='Disable LayerNorm on the exported PM feature vector')

    # MissingAware + grouped dense semantic feature experiment.
    parser.add_argument('--missing_aware_enabled', action='store_true', default=True,
                        help='Enable H-SemanticFeature-v1 MissingAware residuals')
    parser.add_argument('--no_missing_aware', dest='missing_aware_enabled',
                        action='store_false',
                        help='Disable all MissingAware residuals')
    parser.add_argument('--sparse_missing_indicator_enabled', action='store_true', default=True,
                        help='Enable sparse per-fid missing indicators')
    parser.add_argument('--no_sparse_missing_indicators', dest='sparse_missing_indicator_enabled',
                        action='store_false',
                        help='Disable sparse per-fid missing indicators')
    parser.add_argument('--zero_is_missing_fids', type=str, default='',
                        help='Comma-separated fids for which zero should be treated as missing')
    parser.add_argument('--missing_indicator_project_to_group', action='store_true', default=True,
                        help='Project group missing indicators to residual vectors')
    parser.add_argument('--missing_residual_alpha_init', type=float, default=0.1,
                        help='Initial scale for missing residuals')
    parser.add_argument('--missing_residual_alpha_learnable', action='store_true', default=True,
                        help='Make missing residual alpha learnable')
    parser.add_argument('--dense_missing_aware_enabled', action='store_true', default=True,
                        help='Enable dense missing-aware branches')
    parser.add_argument('--dense_stat_transform', type=str, default='signed_log1p',
                        choices=['none', 'signed_log1p'],
                        help='Transform for stat-like dense values')
    parser.add_argument('--dense_grouped_encoder_enabled', action='store_true', default=True,
                        help='Use grouped user dense encoder while keeping one dense token')
    parser.add_argument('--no_dense_grouped_encoder', dest='dense_grouped_encoder_enabled',
                        action='store_false',
                        help='Use the legacy single Linear user dense projection')
    parser.add_argument('--dense_embedding_like_fids', type=str, default='61,87,89,90,91',
                        help='Comma-separated embedding-like dense fids')
    parser.add_argument('--dense_stat_like_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated stat-like dense fids')
    parser.add_argument('--dense_missing_indicator_enabled', action='store_true', default=True,
                        help='Enable dense missing indicator branch')
    parser.add_argument('--dense_value_clip_abs', type=float, default=0.0,
                        help='Optional abs clip for dense values; 0 disables clipping')
    parser.add_argument('--dense_encoder_dropout', type=float, default=0.01,
                        help='Dropout inside grouped dense encoder')

    # TimeToken experiment: sample wall-clock + sequence recency token inserted
    # into the NS-token stream.
    parser.add_argument('--time_token_enabled', action='store_true', default=False,
                        help='Enable H-TimeToken-v1 and append one TimeToken to NS tokens')
    parser.add_argument('--time_token_dim', type=int, default=0,
                        help='Internal TimeToken feature width (0 = d_model)')
    parser.add_argument('--time_token_dropout', type=float, default=0.01,
                        help='Dropout inside TimeTokenEncoder')
    parser.add_argument('--time_token_norm_enabled', dest='time_token_norm_enabled',
                        action='store_true', default=True,
                        help='Enable LayerNorm inside TimeTokenEncoder')
    parser.add_argument('--no_time_token_norm', dest='time_token_norm_enabled',
                        action='store_false',
                        help='Disable TimeTokenEncoder LayerNorm')
    parser.add_argument('--time_token_insert_position', type=str, default='ns_tokens',
                        choices=['ns_tokens'],
                        help='Where to insert TimeToken; H-TimeToken-v1 supports ns_tokens')
    parser.add_argument('--time_bucket_vocab_size', type=int, default=0,
                        help='Sample time bucket vocab size (0 = NUM_TIME_BUCKETS)')
    parser.add_argument('--time_gap_bucket_vocab_size', type=int, default=0,
                        help='Sequence gap bucket vocab size (0 = NUM_TIME_BUCKETS)')
    parser.add_argument('--use_sample_time_features', dest='use_sample_time_features',
                        action='store_true', default=True,
                        help='Use sample timestamp-derived Beijing wall-clock features')
    parser.add_argument('--no_sample_time_features', dest='use_sample_time_features',
                        action='store_false',
                        help='Disable sample timestamp-derived TimeToken features')
    parser.add_argument('--use_seq_recency_features', dest='use_seq_recency_features',
                        action='store_true', default=True,
                        help='Use per-sequence recency gap summaries in TimeToken')
    parser.add_argument('--no_seq_recency_features', dest='use_seq_recency_features',
                        action='store_false',
                        help='Disable per-sequence recency TimeToken features')
    parser.add_argument('--use_seq_time_summary', dest='use_seq_time_summary',
                        action='store_true', default=True,
                        help='Use per-sequence mean/max gap summaries in TimeToken')
    parser.add_argument('--no_seq_time_summary', dest='use_seq_time_summary',
                        action='store_false',
                        help='Disable per-sequence time summary TimeToken features')
    parser.add_argument('--use_time_of_day_features', dest='use_time_of_day_features',
                        action='store_true', default=True,
                        help='Use Beijing cyclic time-of-day features')
    parser.add_argument('--no_time_of_day_features', dest='use_time_of_day_features',
                        action='store_false',
                        help='Disable Beijing cyclic time-of-day features')
    parser.add_argument('--time_tz_offset_hours', type=int, default=8,
                        help='Timezone offset applied to timestamp before wall-clock features')
    parser.add_argument('--use_hour_embedding', action='store_true', default=False,
                        help='Use learned hour embedding; off by default for time-cold generalization')
    parser.add_argument('--time_daypart_vocab_size', type=int, default=7,
                        help='Coarse Beijing daypart vocab size')

    # Logging / validation controls.
    parser.add_argument('--disable_tqdm', action='store_true', default=True,
                        help='Disable tqdm progress bars and use fixed-frequency JSONL logs')
    parser.add_argument('--enable_tqdm', dest='disable_tqdm', action='store_false',
                        help='Accepted for compatibility; trainer still uses JSONL logs')
    parser.add_argument('--train_log_every_steps', type=int, default=100,
                        help='Emit train_step JSONL every N steps')
    parser.add_argument('--time_log_enabled', dest='time_log_enabled',
                        action='store_true', default=False,
                        help='Emit TimeToken shape/stat/health JSONL logs')
    parser.add_argument('--no_time_log', dest='time_log_enabled',
                        action='store_false',
                        help='Disable TimeToken JSONL health/stat logs')
    parser.add_argument('--time_debug_first_n_batches', type=int, default=3,
                        help='Emit TimeToken shape logs for the first N training batches')
    parser.add_argument('--ns_group_debug_first_n_batches', type=int, default=3,
                        help='Emit NS group shape logs for the first N batches')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH', args.tf_events_dir)
    if args.tf_events_dir is None and args.log_dir is not None:
        args.tf_events_dir = os.path.join(args.log_dir, 'tf_events')

    return args


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def main() -> None:
    args = parse_args()
    missing_paths = [
        name for name in ('data_dir', 'ckpt_dir', 'log_dir', 'tf_events_dir')
        if getattr(args, name) is None
    ]
    if missing_paths:
        raise ValueError(
            "Missing required path(s): "
            + ", ".join(missing_paths)
            + ". Provide CLI flags or TRAIN_DATA_PATH/TRAIN_CKPT_PATH/TRAIN_LOG_PATH."
        )

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        logging.warning(
            "schema file not found at %s; dataset will infer schema from parquet columns",
            schema_path,
        )

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        zero_is_missing_fids=_parse_int_list(args.zero_is_missing_fids),
        collect_seen_ids=False,
    )
    if not os.path.exists(schema_path) and hasattr(pcvr_dataset, "raw_schema"):
        inferred_schema_path = os.path.join(args.ckpt_dir, "inferred_schema.json")
        with open(inferred_schema_path, "w", encoding="utf-8") as f:
            json.dump(pcvr_dataset.raw_schema, f, ensure_ascii=False, indent=2)
        schema_path = inferred_schema_path
        logging.info("Persisted inferred schema to %s", schema_path)

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_ns_group_names = list(ns_groups_cfg['user_ns_groups'].keys())
        item_ns_group_names = list(ns_groups_cfg['item_ns_groups'].keys())
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {user_ns_group_names}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {item_ns_group_names}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]
        user_ns_group_names = [f"U{i}" for i in range(len(user_ns_groups))]
        item_ns_group_names = [f"I{i}" for i in range(len(item_ns_groups))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "pm_head_enabled": args.pm_head_enabled,
        "pm_feature_dim": args.pm_feature_dim,
        "pm_feature_dropout": args.pm_feature_dropout,
        "pm_feature_norm_enabled": args.pm_feature_norm_enabled,
        "time_token_enabled": args.time_token_enabled,
        "time_token_dim": args.time_token_dim,
        "time_token_dropout": args.time_token_dropout,
        "time_token_norm_enabled": args.time_token_norm_enabled,
        "time_token_insert_position": args.time_token_insert_position,
        "time_bucket_vocab_size": args.time_bucket_vocab_size,
        "time_gap_bucket_vocab_size": args.time_gap_bucket_vocab_size,
        "use_sample_time_features": args.use_sample_time_features,
        "use_seq_recency_features": args.use_seq_recency_features,
        "use_seq_time_summary": args.use_seq_time_summary,
        "use_time_of_day_features": args.use_time_of_day_features,
        "time_tz_offset_hours": args.time_tz_offset_hours,
        "use_hour_embedding": args.use_hour_embedding,
        "time_daypart_vocab_size": args.time_daypart_vocab_size,
        "missing_aware_enabled": args.missing_aware_enabled,
        "sparse_missing_indicator_enabled": args.sparse_missing_indicator_enabled,
        "missing_indicator_project_to_group": args.missing_indicator_project_to_group,
        "missing_residual_alpha_init": args.missing_residual_alpha_init,
        "missing_residual_alpha_learnable": args.missing_residual_alpha_learnable,
        "dense_missing_aware_enabled": args.dense_missing_aware_enabled,
        "dense_stat_transform": args.dense_stat_transform,
        "dense_grouped_encoder_enabled": args.dense_grouped_encoder_enabled,
        "dense_embedding_like_fids": _parse_int_list(args.dense_embedding_like_fids),
        "dense_stat_like_fids": _parse_int_list(args.dense_stat_like_fids),
        "dense_missing_indicator_enabled": args.dense_missing_indicator_enabled,
        "dense_value_clip_abs": args.dense_value_clip_abs,
        "dense_encoder_dropout": args.dense_encoder_dropout,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    model.user_ns_group_names = user_ns_group_names
    model.item_ns_group_names = item_ns_group_names
    model.user_ns_group_fids = list(ns_groups_cfg['user_ns_groups'].values()) if args.ns_groups_json and os.path.exists(args.ns_groups_json) else [
        [fid] for fid, _, _ in pcvr_dataset.user_int_schema.entries
    ]
    model.item_ns_group_fids = list(ns_groups_cfg['item_ns_groups'].values()) if args.ns_groups_json and os.path.exists(args.ns_groups_json) else [
        [fid] for fid, _, _ in pcvr_dataset.item_int_schema.entries
    ]
    model.user_int_fids = pcvr_dataset.user_int_schema.feature_ids
    model.item_int_fids = pcvr_dataset.item_int_schema.feature_ids
    model.user_dense_fids = pcvr_dataset.user_dense_schema.feature_ids
    if args.ns_tokenizer_type == 'group':
        if len(user_ns_groups) != 3 or len(item_ns_groups) != 4:
            raise ValueError(
                "H-SemanticFeature-v1 expects exactly 3 user groups and 4 item groups; "
                f"got user={len(user_ns_groups)}, item={len(item_ns_groups)}")
        if model.num_ns != 9:
            raise ValueError(
                "H-SemanticFeature-v1 expects total NS tokens to remain 9 "
                "(3 user int + 1 user dense + 4 item int + 1 time); "
                f"got num_ns={model.num_ns}")

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    if args.ns_tokenizer_type == 'group' and T != 17:
        raise ValueError(
            "H-NSGroup-v2-item3 expects RankMixer input token count T=17 "
            f"(num_queries * num_sequences + num_ns); got T={T}")
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(
        "PM head config: enabled=%s, feature_dim=%s, dropout=%s, norm_enabled=%s",
        args.pm_head_enabled,
        args.pm_feature_dim,
        args.pm_feature_dropout,
        args.pm_feature_norm_enabled,
    )
    logging.info(
        "TimeToken config: enabled=%s, insert_position=%s, time_token_dim=%s, dropout=%s, "
        "sample_time=%s, seq_recency=%s, seq_summary=%s, time_of_day=%s, hour_embedding=%s",
        args.time_token_enabled,
        args.time_token_insert_position,
        model.time_token_dim,
        args.time_token_dropout,
        args.use_sample_time_features,
        args.use_seq_recency_features,
        args.use_seq_time_summary,
        args.use_time_of_day_features,
        args.use_hour_embedding,
    )
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        disable_tqdm=args.disable_tqdm,
        train_log_every_steps=args.train_log_every_steps,
        time_log_enabled=args.time_log_enabled,
        time_debug_first_n_batches=args.time_debug_first_n_batches,
        ns_group_debug_first_n_batches=args.ns_group_debug_first_n_batches,
        max_train_steps=args.max_train_steps,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
