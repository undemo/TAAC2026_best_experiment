"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates validation AUC on the main model
path only.
"""

import os
import glob
import shutil
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Validation metric: Binary AUC.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        disable_tqdm: bool = True,
        train_log_every_steps: int = 100,
        time_log_enabled: bool = True,
        time_debug_first_n_batches: int = 3,
        ns_group_debug_first_n_batches: int = 3,
        max_train_steps: int = 0,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.disable_tqdm: bool = disable_tqdm
        self.train_log_every_steps: int = max(1, int(train_log_every_steps))
        self.time_log_enabled: bool = time_log_enabled
        self.time_debug_first_n_batches: int = max(0, int(time_debug_first_n_batches))
        self.ns_group_debug_first_n_batches: int = max(0, int(ns_group_debug_first_n_batches))
        self.max_train_steps: int = max(0, int(max_train_steps))
        self._time_shape_logs_emitted: int = 0
        self._ns_group_shape_logs_emitted: int = 0
        self._grad_clip_total: int = 0
        self._grad_clip_hits: int = 0
        self._latest_validation_metrics: Dict[str, Any] = {}
        self._final_val_auc: Optional[float] = None
        self._best_epoch: Optional[int] = None

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")

    def _json_safe(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 1:
                item = value.detach().float().cpu().item()
                return float(item)
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

    def _log_json(self, event: Dict[str, Any]) -> None:
        logging.info(json.dumps(self._json_safe(event), ensure_ascii=False, sort_keys=True))

    def _scalar_mean(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu().item())
        return value

    def _log_run_start(self) -> None:
        user_group_names = list(getattr(self.model, "user_ns_group_names", []))
        item_group_names = list(getattr(self.model, "item_ns_group_names", []))
        self._log_json({
            "type": "run_start",
            "experiment": "H-SemanticFeature-v1",
            "model": "HyFormer-PMHead-TimeToken-NSGroup",
            "time_token_enabled": bool(getattr(self.model, "time_token_enabled", False)),
            "pm_head_enabled": bool(getattr(self.model, "pm_head_enabled", False)),
            "disable_tqdm": bool(self.disable_tqdm),
        })
        self._log_json({
            "type": "semantic_feature_config",
            "experiment": "H-SemanticFeature-v1",
            "base": "H-NSGroup-v2-item3",
            "ns_tokenizer_type": getattr(self.model, "ns_tokenizer_type", None),
            "user_int_tokens": int(getattr(self.model, "num_user_int_tokens", 0)),
            "user_dense_token": 1,
            "item_int_tokens": int(getattr(self.model, "num_item_int_tokens", 0)),
            "time_token": 1 if bool(getattr(self.model, "time_token_enabled", False)) else 0,
            "total_ns_tokens": int(getattr(self.model, "num_ns", 0)),
            "rankmixer_T": int(getattr(self.model, "num_queries", 0)) * int(getattr(self.model, "num_sequences", 0)) + int(getattr(self.model, "num_ns", 0)),
            "d_model": int(getattr(self.model, "d_model", 0)),
            "missing_aware_enabled": bool(getattr(self.model, "missing_aware_enabled", False)),
            "dense_grouped_encoder_enabled": bool(getattr(self.model, "dense_grouped_encoder_enabled", False)),
            "time_token_enabled": bool(getattr(self.model, "time_token_enabled", False)),
            "pm_head_enabled": bool(getattr(self.model, "pm_head_enabled", False)),
        })
        for group_type, names, fids_list, available in [
            ("user_int", user_group_names, getattr(self.model, "user_ns_group_fids", []), set(getattr(self.model, "user_int_fids", []))),
            ("item_int", item_group_names, getattr(self.model, "item_ns_group_fids", []), set(getattr(self.model, "item_int_fids", []))),
        ]:
            for name, fids in zip(names, fids_list):
                missing = [fid for fid in fids if fid not in available]
                self._log_json({
                    "type": "semantic_group_audit",
                    "group_name": name,
                    "group_type": group_type,
                    "fids": fids,
                    "num_fids": len(fids),
                    "all_fids_present": len(missing) == 0,
                    "missing_fids": missing,
                    "duplicated_fids": sorted({fid for fid in fids if fids.count(fid) > 1}),
                    "estimated_cardinality": None,
                    "zero_ratio": None,
                    "missing_ratio": None,
                    "minus1_ratio": None,
                })
        self._log_json({
            "type": "time_config",
            "time_token_enabled": bool(getattr(self.model, "time_token_enabled", False)),
            "time_token_insert_position": getattr(self.model, "time_token_insert_position", None),
            "time_token_dim": getattr(self.model, "time_token_dim", None),
            "time_token_dropout": getattr(self.model, "time_token_dropout", None),
            "use_sample_time_features": getattr(self.model, "use_sample_time_features", None),
            "use_seq_recency_features": getattr(self.model, "use_seq_recency_features", None),
            "use_seq_time_summary": getattr(self.model, "use_seq_time_summary", None),
            "use_time_of_day_features": getattr(self.model, "use_time_of_day_features", None),
            "use_hour_embedding": getattr(self.model, "use_hour_embedding", None),
        })

    def _log_time_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        step: int,
        batch_idx: int,
    ) -> None:
        if not self.time_log_enabled or not diagnostics:
            return
        if self._time_shape_logs_emitted < self.time_debug_first_n_batches:
            self._log_json({
                "type": "semantic_feature_shape",
                "batch": batch_idx,
                "user_group_tokens_shape": diagnostics.get("user_int_tokens_shape"),
                "item_group_tokens_shape": diagnostics.get("item_int_tokens_shape"),
                "user_dense_token_shape": diagnostics.get("user_dense_token_shape"),
                "time_token_shape": diagnostics.get("time_token_shape"),
                "ns_tokens_shape": diagnostics.get("ns_tokens_shape"),
                "rankmixer_input_shape": diagnostics.get("rankmixer_input_shape"),
                "classifier_input_shape": diagnostics.get("classifier_input_shape"),
            })
            self._time_shape_logs_emitted += 1
        if self._ns_group_shape_logs_emitted < self.ns_group_debug_first_n_batches:
            self._log_json({
                "type": "ns_group_shape",
                "batch": batch_idx,
                **{
                    f"user_group_{idx}_shape": diagnostics.get(f"user_group_{idx}_shape")
                    for idx in range(int(getattr(self.model, "num_user_int_tokens", 0)))
                },
                **{
                    f"item_group_{idx}_shape": diagnostics.get(f"item_group_{idx}_shape")
                    for idx in range(int(getattr(self.model, "num_item_int_tokens", 0)))
                },
                "ns_tokens_shape": diagnostics.get("ns_tokens_shape"),
                "rankmixer_input_shape": diagnostics.get("rankmixer_input_shape"),
            })
            self._ns_group_shape_logs_emitted += 1

        if step % self.train_log_every_steps != 0:
            return

        feature_stats = diagnostics.get("time_feature_stats") or {}
        self._log_json({
            "type": "time_feature_stats",
            "step": step,
            "sample_time_available": feature_stats.get("sample_time_available"),
            "seq_time_available": feature_stats.get("seq_time_available"),
            "time_bucket_min": feature_stats.get("time_bucket_min"),
            "time_bucket_max": feature_stats.get("time_bucket_max"),
            "time_bucket_unique": feature_stats.get("time_bucket_unique"),
            "seq_a_last_gap_mean": feature_stats.get("seq_a_last_gap_mean"),
            "seq_b_last_gap_mean": feature_stats.get("seq_b_last_gap_mean"),
            "seq_c_last_gap_mean": feature_stats.get("seq_c_last_gap_mean"),
            "seq_d_last_gap_mean": feature_stats.get("seq_d_last_gap_mean"),
            "seq_a_empty_ratio": feature_stats.get("seq_a_empty_ratio"),
            "seq_b_empty_ratio": feature_stats.get("seq_b_empty_ratio"),
            "seq_c_empty_ratio": feature_stats.get("seq_c_empty_ratio"),
            "seq_d_empty_ratio": feature_stats.get("seq_d_empty_ratio"),
        })
        nan_count = diagnostics.get("time_token_nan_count", 0)
        inf_count = diagnostics.get("time_token_inf_count", 0)
        self._log_json({
            "type": "semantic_token_health",
            "step": step,
            "user_low_context_norm": self._scalar_mean(diagnostics.get("user_low_context_norm")),
            "user_profile_stat_shared_norm": self._scalar_mean(diagnostics.get("user_profile_stat_shared_norm")),
            "user_compact_tail_flags_norm": self._scalar_mean(diagnostics.get("user_compact_tail_flags_norm")),
            "item_low_card_norm": self._scalar_mean(diagnostics.get("item_low_card_norm")),
            "item_mid_behavior_a_norm": self._scalar_mean(diagnostics.get("item_mid_behavior_a_norm")),
            "item_mid_behavior_b_norm": self._scalar_mean(diagnostics.get("item_mid_behavior_b_norm")),
            "item_high_id_like_norm": self._scalar_mean(diagnostics.get("item_high_id_like_norm")),
            "user_dense_token_norm": self._scalar_mean(diagnostics.get("user_dense_token_norm")),
            "time_token_norm": self._scalar_mean(diagnostics.get("time_token_norm")),
            "pm_feature_norm": self._scalar_mean(diagnostics.get("pm_feature_norm")),
            "hyformer_repr_norm": self._scalar_mean(diagnostics.get("hyformer_repr_norm")),
            "nan_count": diagnostics.get("nan_count", nan_count),
            "inf_count": diagnostics.get("inf_count", inf_count),
        })
        group_names = list(getattr(self.model, "user_ns_group_names", [])) + list(getattr(self.model, "item_ns_group_names", []))
        for idx, group_name in enumerate(group_names):
            key = f"user_group_{idx}_missing_ratio" if idx < len(getattr(self.model, "user_ns_group_names", [])) else f"item_group_{idx - len(getattr(self.model, 'user_ns_group_names', []))}_missing_ratio"
            self._log_json({
                "type": "missing_feature_stats",
                "step": step,
                "group_name": group_name,
                "missing_ratio_mean": self._scalar_mean(diagnostics.get(key)),
                "zero_ratio_mean": None,
                "minus1_ratio_mean": None,
                "empty_multivalue_ratio": None,
                "missing_residual_norm": self._scalar_mean(diagnostics.get("missing_residual_norm")),
                "group_token_norm": self._scalar_mean(diagnostics.get("hyformer_ns_token_norm")),
                "missing_to_token_norm_ratio": self._scalar_mean(diagnostics.get("missing_to_token_norm_ratio")),
            })
        self._log_json({
            "type": "dense_group_stats",
            "step": step,
            "embedding_like_raw_norm_mean": self._scalar_mean(diagnostics.get("embedding_like_raw_norm")),
            "embedding_like_raw_norm_max": self._scalar_mean(diagnostics.get("embedding_like_raw_abs_max")),
            "stat_like_raw_abs_mean": self._scalar_mean(diagnostics.get("stat_like_raw_abs_mean")),
            "stat_like_raw_abs_max": self._scalar_mean(diagnostics.get("stat_like_raw_abs_max")),
            "stat_like_log_abs_mean": self._scalar_mean(diagnostics.get("stat_like_log_abs_mean")),
            "stat_like_log_abs_max": self._scalar_mean(diagnostics.get("stat_like_log_abs_max")),
            "dense_missing_ratio": self._scalar_mean(diagnostics.get("dense_missing_ratio")),
            "dense_token_norm": self._scalar_mean(diagnostics.get("dense_token_norm")),
            "stat_branch_norm": self._scalar_mean(diagnostics.get("stat_branch_norm")),
            "embedding_branch_norm": self._scalar_mean(diagnostics.get("embedding_branch_norm")),
            "missing_branch_norm": self._scalar_mean(diagnostics.get("missing_branch_norm")),
        })

    def _grad_clip_hit_rate(self) -> float:
        if self._grad_clip_total <= 0:
            return 0.0
        return float(self._grad_clip_hits / self._grad_clip_total)

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, os.path.join(ckpt_dir, "schema.json"))

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        logging.info("Start training (PCVRHyFormer)")
        self._log_run_start()
        self.model.train()
        total_step = 0
        last_epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            last_epoch = epoch
            loss_sum = 0.0
            step_count = 0

            for step, batch in enumerate(self.train_loader):
                train_metrics = self._train_step(batch)
                loss = float(train_metrics["loss"])
                total_step += 1
                step_count += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                self._log_time_diagnostics(
                    train_metrics.get("diagnostics", {}),
                    total_step,
                    batch_idx=total_step,
                )

                if total_step % self.train_log_every_steps == 0:
                    avg_loss = loss_sum / max(step_count, 1)
                    self._log_json({
                        "type": "train_step",
                        "epoch": epoch,
                        "step": total_step,
                        "loss": loss,
                        "avg_loss": avg_loss,
                        "lr": self.dense_optimizer.param_groups[0]["lr"],
                        "grad_norm": train_metrics.get("grad_norm"),
                        "grad_clip_hit_rate": self._grad_clip_hit_rate(),
                        "time_token_enabled": bool(getattr(self.model, "time_token_enabled", False)),
                    })

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc = self._validate_for_checkpoint(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | val_auc: {val_auc}")
                    self._maybe_record_best_epoch(epoch, val_auc)
                    self._log_validation_json(epoch, val_auc)

                    if self.writer:
                        self.writer.add_scalar('val_auc', val_auc, total_step)

                    self._handle_validation_result(total_step, val_auc)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        self._log_run_digest(epoch)
                        return

                if self.max_train_steps > 0 and total_step >= self.max_train_steps:
                    logging.info("Stopping at max_train_steps=%d", self.max_train_steps)
                    val_auc = self._validate_for_checkpoint(epoch=epoch)
                    self._maybe_record_best_epoch(epoch, val_auc)
                    self._log_validation_json(epoch, val_auc)
                    self._log_run_digest(epoch)
                    return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / max(step_count, 1)}")

            val_auc = self._validate_for_checkpoint(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | val_auc: {val_auc}")
            self._maybe_record_best_epoch(epoch, val_auc)
            self._log_validation_json(epoch, val_auc)

            if self.writer:
                self.writer.add_scalar('val_auc', val_auc, total_step)

            self._handle_validation_result(total_step, val_auc)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

        self._log_run_digest(last_epoch)

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            timestamp=device_batch.get('timestamp'),
            user_int_missing_mask=device_batch.get('user_int_missing_mask'),
            user_int_zero_mask=device_batch.get('user_int_zero_mask'),
            user_int_minus1_mask=device_batch.get('user_int_minus1_mask'),
            item_int_missing_mask=device_batch.get('item_int_missing_mask'),
            item_int_zero_mask=device_batch.get('item_int_zero_mask'),
            item_int_minus1_mask=device_batch.get('item_int_minus1_mask'),
            user_dense_missing_mask=device_batch.get('user_dense_missing_mask'),
            user_dense_zero_mask=device_batch.get('user_dense_zero_mask'),
        )

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single training step and return scalar training diagnostics."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        model_result = self.model(model_input, return_diagnostics=True)
        logits = model_result["logits"]  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
        grad_norm_value = float(grad_norm.detach().float().cpu().item())
        self._grad_clip_total += 1
        if grad_norm_value > 1.0:
            self._grad_clip_hits += 1

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm_value,
            "diagnostics": model_result.get("diagnostics", {}),
        }

    def _compute_binary_auc(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> float:
        """Compute validation AUC after filtering NaN predictions."""
        all_labels = all_labels.long()
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            probs = probs[~nan_mask]
            labels_np = labels_np[~nan_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            return 0.0
        return float(roc_auc_score(labels_np, probs))

    def _validate_for_checkpoint(self, epoch: Optional[int] = None) -> float:
        """Validation entry used by training.

        The strong baseline is monitored only by the normal-path validation AUC.
        """
        return self.evaluate(epoch=epoch)

    def _log_validation_json(self, epoch: int, auc: float) -> None:
        self._final_val_auc = auc
        self._log_json({
            "type": "validation",
            "epoch": epoch,
            "val_auc": auc,
            "best_auc": self.early_stopping.best_score,
            "best_epoch": self._best_epoch,
        })

    def _maybe_record_best_epoch(self, epoch: int, auc: float) -> None:
        best = self.early_stopping.best_score
        if best is None or auc > best + self.early_stopping.delta:
            self._best_epoch = epoch

    def _log_run_digest(self, epoch: int) -> None:
        self._log_json({
            "type": "run_digest",
            "experiment": "H-SemanticFeature-v1",
            "best_epoch": self._best_epoch,
            "best_val_auc": self.early_stopping.best_score,
            "final_val_auc": self._final_val_auc,
            "epoch": epoch,
        })

    def evaluate(self, epoch: Optional[int] = None) -> float:
        """Run validation over ``self.valid_loader`` and return AUC.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing the metric.
        """
        logging.info("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        auc = self._compute_binary_auc(all_logits, all_labels)
        self._latest_validation_metrics = {
            "val_auc": auc,
        }
        return auc

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
