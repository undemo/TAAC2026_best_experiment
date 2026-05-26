"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    timestamp: Optional[torch.Tensor] = None  # tensor [B], Unix seconds
    user_int_missing_mask: Optional[torch.Tensor] = None
    user_int_zero_mask: Optional[torch.Tensor] = None
    user_int_minus1_mask: Optional[torch.Tensor] = None
    item_int_missing_mask: Optional[torch.Tensor] = None
    item_int_zero_mask: Optional[torch.Tensor] = None
    item_int_minus1_mask: Optional[torch.Tensor] = None
    user_dense_missing_mask: Optional[torch.Tensor] = None
    user_dense_zero_mask: Optional[torch.Tensor] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class UserDenseGroupedEncoder(nn.Module):
    """Encode heterogeneous user_dense fids into one dense NS token."""

    def __init__(
        self,
        user_dense_feature_specs: List[Tuple[int, int, int]],
        d_model: int,
        embedding_like_fids: List[int],
        stat_like_fids: List[int],
        dense_stat_transform: str = "signed_log1p",
        dense_missing_indicator_enabled: bool = True,
        dense_missing_aware_enabled: bool = True,
        dense_value_clip_abs: float = 0.0,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.feature_specs = [(int(fid), int(offset), int(length)) for fid, offset, length in user_dense_feature_specs]
        self.fid_to_spec = {fid: (idx, offset, length) for idx, (fid, offset, length) in enumerate(self.feature_specs)}
        self.embedding_like_fids = [int(fid) for fid in embedding_like_fids]
        self.stat_like_fids = [int(fid) for fid in stat_like_fids]
        self.dense_stat_transform = dense_stat_transform
        self.dense_missing_indicator_enabled = dense_missing_indicator_enabled
        self.dense_missing_aware_enabled = dense_missing_aware_enabled
        self.dense_value_clip_abs = float(dense_value_clip_abs or 0.0)

        missing = [
            fid for fid in self.embedding_like_fids + self.stat_like_fids
            if fid not in self.fid_to_spec
        ]
        if missing:
            raise ValueError(f"dense grouped encoder references missing user_dense fids: {missing}")

        self.embedding_dim = sum(self.fid_to_spec[fid][2] for fid in self.embedding_like_fids)
        self.stat_dim = sum(self.fid_to_spec[fid][2] for fid in self.stat_like_fids)
        self.num_dense_fids = len(self.feature_specs)
        summary_dim = 4

        self.embedding_branch = nn.Sequential(
            nn.LayerNorm(max(self.embedding_dim, 1)),
            nn.Linear(max(self.embedding_dim, 1), d_model),
            nn.SiLU(),
        )
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(max(self.stat_dim, 1)),
            nn.Linear(max(self.stat_dim, 1), d_model),
            nn.SiLU(),
        )
        self.missing_branch = nn.Sequential(
            nn.Linear(max(self.num_dense_fids, 1), d_model),
            nn.SiLU(),
        )
        self.summary_branch = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, d_model),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def _slice_fids(self, dense: torch.Tensor, fids: List[int], min_dim: int) -> torch.Tensor:
        parts = []
        for fid in fids:
            _, offset, length = self.fid_to_spec[fid]
            parts.append(dense[:, offset:offset + length])
        if parts:
            return torch.cat(parts, dim=1)
        return dense.new_zeros(dense.size(0), min_dim)

    def forward(
        self,
        dense: torch.Tensor,
        dense_missing_mask: Optional[torch.Tensor] = None,
        eval_missing_mode: str = "normal",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.dense_value_clip_abs > 0:
            dense = dense.clamp(min=-self.dense_value_clip_abs, max=self.dense_value_clip_abs)

        embedding_values = self._slice_fids(dense, self.embedding_like_fids, max(self.embedding_dim, 1))
        stat_values = self._slice_fids(dense, self.stat_like_fids, max(self.stat_dim, 1))
        stat_raw = stat_values
        use_log = self.dense_stat_transform == "signed_log1p" and eval_missing_mode != "without_dense_stat_log_transform"
        if use_log:
            stat_values = stat_values.sign() * torch.log1p(stat_values.abs())

        if dense_missing_mask is None:
            dense_missing_mask = dense.new_zeros(dense.size(0), self.num_dense_fids)
        dense_missing_mask = dense_missing_mask.to(dtype=dense.dtype)
        use_dense_missing = (
            self.dense_missing_aware_enabled
            and self.dense_missing_indicator_enabled
            and eval_missing_mode not in {
                "without_missing_aware",
                "without_dense_missing_indicators",
                "sparse_missing_only",
            }
        )
        if not use_dense_missing:
            dense_missing_mask = torch.zeros_like(dense_missing_mask)

        abs_dense = dense.detach().float().abs()
        log_abs = torch.log1p(abs_dense)
        summary = torch.stack([
            log_abs.mean(dim=1),
            log_abs.max(dim=1).values,
            torch.linalg.vector_norm(log_abs, dim=1),
            dense_missing_mask.detach().float().mean(dim=1),
        ], dim=1).to(dtype=dense.dtype)

        embedding_branch = self.embedding_branch(embedding_values)
        stat_branch = self.stat_branch(stat_values)
        missing_branch = self.missing_branch(dense_missing_mask)
        summary_branch = self.summary_branch(summary)
        token = self.fusion(torch.cat([
            embedding_branch, stat_branch, missing_branch, summary_branch
        ], dim=1))
        diagnostics = {
            "embedding_like_raw_norm": embedding_values.detach().float().norm(dim=1, keepdim=True),
            "embedding_like_raw_abs_max": embedding_values.detach().float().abs().max(dim=1, keepdim=True).values,
            "stat_like_raw_abs_mean": stat_raw.detach().float().abs().mean(dim=1, keepdim=True),
            "stat_like_raw_abs_max": stat_raw.detach().float().abs().max(dim=1, keepdim=True).values,
            "stat_like_log_abs_mean": torch.log1p(stat_raw.detach().float().abs()).mean(dim=1, keepdim=True),
            "stat_like_log_abs_max": torch.log1p(stat_raw.detach().float().abs()).max(dim=1, keepdim=True).values,
            "dense_missing_ratio": dense_missing_mask.detach().float().mean(dim=1, keepdim=True),
            "dense_token_norm": token.detach().float().norm(dim=1, keepdim=True),
            "stat_branch_norm": stat_branch.detach().float().norm(dim=1, keepdim=True),
            "embedding_branch_norm": embedding_branch.detach().float().norm(dim=1, keepdim=True),
            "missing_branch_norm": missing_branch.detach().float().norm(dim=1, keepdim=True),
        }
        return token, diagnostics


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class HyFormerPMHeadFeatureExtractor(nn.Module):
    """Head-only predictive-memory feature extractor.

    This branch consumes already-built HyFormer token representations and
    exports fixed-width features for the classifier head. It deliberately does
    not feed anything back into the HyFormer backbone.
    """

    def __init__(
        self,
        d_model: int,
        pm_feature_dim: int,
        pm_feature_dropout: float,
        pm_feature_norm_enabled: bool,
        recent_k: int = 32,
    ) -> None:
        super().__init__()
        if pm_feature_dim <= 0:
            raise ValueError("pm_feature_dim must be positive when PM head is enabled")
        self.recent_k = max(1, int(recent_k))

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(pm_feature_dropout),
        )
        self.recent_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(pm_feature_dropout),
        )
        raw_dim = d_model * 4 + 1
        hidden_dim = max(pm_feature_dim, d_model)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(pm_feature_dropout),
            nn.Linear(hidden_dim, pm_feature_dim),
        )
        self.feature_norm = nn.LayerNorm(pm_feature_dim) if pm_feature_norm_enabled else nn.Identity()
        self.feature_dropout = nn.Dropout(pm_feature_dropout)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (tokens * valid).sum(dim=1) / denom

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        all_pools = []
        recent_pools = []
        for tokens, mask in zip(seq_tokens_list, seq_masks_list):
            all_pools.append(self._masked_mean(tokens, mask))
            recent_len = min(self.recent_k, tokens.size(1))
            recent_pools.append(
                self._masked_mean(tokens[:, :recent_len], mask[:, :recent_len])
            )

        all_pool = torch.stack(all_pools, dim=1).mean(dim=1)
        recent_pool = torch.stack(recent_pools, dim=1).mean(dim=1)

        memory = self.memory_proj(all_pool)
        recent = self.recent_proj(recent_pool)
        error = recent - memory
        ns_summary = ns_tokens.mean(dim=1)
        error_norm = error.float().norm(dim=1, keepdim=True).to(dtype=error.dtype)

        raw_features = torch.cat([memory, recent, error, ns_summary, error_norm], dim=1)
        features = self.feature_proj(raw_features)
        features = self.feature_norm(features)
        return self.feature_dropout(features)


class TimeTokenEncoder(nn.Module):
    """Encode sample wall-clock time and sequence recency summaries as one token."""

    def __init__(
        self,
        d_model: int,
        seq_domains: List[str],
        time_token_dim: int = 0,
        time_token_dropout: float = 0.01,
        time_token_norm_enabled: bool = True,
        time_bucket_vocab_size: int = 65,
        time_gap_bucket_vocab_size: int = 65,
        use_sample_time_features: bool = True,
        use_seq_recency_features: bool = True,
        use_seq_time_summary: bool = True,
        use_time_of_day_features: bool = True,
        use_hour_embedding: bool = False,
        time_daypart_vocab_size: int = 7,
        time_tz_offset_hours: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.seq_domains = list(seq_domains)
        self.time_token_dim = int(time_token_dim) if int(time_token_dim) > 0 else int(d_model)
        self.time_bucket_vocab_size = max(1, int(time_bucket_vocab_size))
        self.time_gap_bucket_vocab_size = max(1, int(time_gap_bucket_vocab_size))
        self.use_sample_time_features = bool(use_sample_time_features)
        self.use_seq_recency_features = bool(use_seq_recency_features)
        self.use_seq_time_summary = bool(use_seq_time_summary)
        self.use_time_of_day_features = bool(use_time_of_day_features)
        self.use_hour_embedding = bool(use_hour_embedding)
        self.time_daypart_vocab_size = max(2, int(time_daypart_vocab_size))
        self.time_tz_offset_seconds = int(time_tz_offset_hours) * 3600

        emb_count = 0
        if self.use_sample_time_features and self.use_time_of_day_features:
            self.daypart_embedding = nn.Embedding(
                self.time_daypart_vocab_size, self.time_token_dim, padding_idx=0)
            emb_count += 1
            if self.use_hour_embedding:
                self.hour_embedding = nn.Embedding(24, self.time_token_dim)
                emb_count += 1
            else:
                self.hour_embedding = None
        else:
            self.daypart_embedding = None
            self.hour_embedding = None

        if self.use_seq_recency_features or self.use_seq_time_summary:
            self.gap_embedding = nn.Embedding(
                self.time_gap_bucket_vocab_size, self.time_token_dim, padding_idx=0)
            if self.use_seq_recency_features:
                emb_count += len(self.seq_domains)
            if self.use_seq_time_summary:
                emb_count += 2 * len(self.seq_domains)
        else:
            self.gap_embedding = None

        continuous_dim = 1
        if self.use_sample_time_features and self.use_time_of_day_features:
            continuous_dim += 8
        per_domain_dim = 0
        if self.use_seq_recency_features:
            per_domain_dim += 4
        if self.use_seq_time_summary:
            per_domain_dim += 3
        continuous_dim += len(self.seq_domains) * per_domain_dim

        self.continuous_proj = nn.Sequential(
            nn.Linear(continuous_dim, self.time_token_dim),
            nn.LayerNorm(self.time_token_dim),
            nn.SiLU(),
        )

        input_dim = (emb_count + 1) * self.time_token_dim
        hidden_dim = max(self.time_token_dim, d_model)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim) if time_token_norm_enabled else nn.Identity(),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(time_token_dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in [self.daypart_embedding, self.hour_embedding, self.gap_embedding]:
            if emb is None:
                continue
            nn.init.xavier_normal_(emb.weight.data)
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx, :] = 0

    @staticmethod
    def _cyclic_pair(values: torch.Tensor, period: float) -> Tuple[torch.Tensor, torch.Tensor]:
        radians = values.float() * (2.0 * math.pi / period)
        return torch.sin(radians), torch.cos(radians)

    def _sample_time_features(
        self,
        timestamp: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        B = timestamp.size(0)
        device = timestamp.device
        continuous: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []

        ts = timestamp.long()
        available = ts > 0
        local_ts = ts + self.time_tz_offset_seconds
        second_of_day = torch.remainder(local_ts, 86400).clamp(min=0)
        minute_of_day = torch.div(second_of_day, 60, rounding_mode='floor')
        hour = torch.div(second_of_day, 3600, rounding_mode='floor').clamp(0, 23)
        day = torch.div(local_ts, 86400, rounding_mode='floor')
        weekday = torch.remainder(day + 3, 7).clamp(0, 6)

        daypart = torch.zeros(B, dtype=torch.long, device=device)
        daypart = torch.where(hour < 5, torch.ones_like(daypart), daypart)
        daypart = torch.where((hour >= 5) & (hour < 10), torch.full_like(daypart, 2), daypart)
        daypart = torch.where((hour >= 10) & (hour < 14), torch.full_like(daypart, 3), daypart)
        daypart = torch.where((hour >= 14) & (hour < 18), torch.full_like(daypart, 4), daypart)
        daypart = torch.where((hour >= 18) & (hour < 22), torch.full_like(daypart, 5), daypart)
        daypart = torch.where(hour >= 22, torch.full_like(daypart, 6), daypart)
        daypart = torch.where(available, daypart.clamp(max=self.time_daypart_vocab_size - 1), torch.zeros_like(daypart))

        if self.daypart_embedding is not None:
            embeddings.append(self.daypart_embedding(daypart))
        if self.hour_embedding is not None:
            embeddings.append(self.hour_embedding(hour))

        minute_sin, minute_cos = self._cyclic_pair(minute_of_day, 1440.0)
        hour_sin, hour_cos = self._cyclic_pair(hour, 24.0)
        weekday_sin, weekday_cos = self._cyclic_pair(weekday, 7.0)
        is_late_night = ((hour < 5) & available).float()
        continuous.extend([
            available.float().unsqueeze(1),
            minute_sin.unsqueeze(1),
            minute_cos.unsqueeze(1),
            hour_sin.unsqueeze(1),
            hour_cos.unsqueeze(1),
            weekday_sin.unsqueeze(1),
            weekday_cos.unsqueeze(1),
            is_late_night.unsqueeze(1),
        ])

        stats = {
            "sample_time_available": available.float().mean(),
            "sample_hour_min": hour.float().min() if B > 0 else timestamp.new_tensor(0.0),
            "sample_hour_max": hour.float().max() if B > 0 else timestamp.new_tensor(0.0),
            "minute_of_day_min": minute_of_day.float().min() if B > 0 else timestamp.new_tensor(0.0),
            "minute_of_day_max": minute_of_day.float().max() if B > 0 else timestamp.new_tensor(0.0),
            "late_night_ratio": is_late_night.mean() if B > 0 else timestamp.new_tensor(0.0),
            "time_bucket_min": daypart.float().min() if B > 0 else timestamp.new_tensor(0.0),
            "time_bucket_max": daypart.float().max() if B > 0 else timestamp.new_tensor(0.0),
            "time_bucket_unique": torch.tensor(float(daypart.unique().numel()), device=device),
        }
        return continuous, embeddings, stats

    def _seq_gap_features(
        self,
        domain: str,
        time_bucket: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        B, L = time_bucket.shape
        device = time_bucket.device
        tb = time_bucket.long().clamp(min=0, max=self.time_gap_bucket_vocab_size - 1)
        positions = torch.arange(L, device=device).unsqueeze(0)
        valid = (positions < seq_len.long().unsqueeze(1)) & (tb > 0)
        valid_count = valid.sum(dim=1)
        empty = valid_count == 0

        max_id = self.time_gap_bucket_vocab_size - 1
        min_fill = torch.full_like(tb, max_id)
        min_gap = torch.where(valid, tb, min_fill).min(dim=1).values
        min_gap = torch.where(empty, torch.zeros_like(min_gap), min_gap).clamp(0, max_id)
        max_gap = torch.where(valid, tb, torch.zeros_like(tb)).max(dim=1).values.clamp(0, max_id)

        denom = valid_count.clamp(min=1).float()
        mean_gap_float = (tb.float() * valid.float()).sum(dim=1) / denom
        mean_gap = torch.round(mean_gap_float).long().clamp(0, max_id)
        mean_gap = torch.where(empty, torch.zeros_like(mean_gap), mean_gap)

        continuous: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []
        norm_denom = float(max(max_id, 1))
        if self.use_seq_recency_features:
            embeddings.append(self.gap_embedding(min_gap) if self.gap_embedding is not None else tb.new_zeros(B, self.time_token_dim, dtype=torch.float))
            recency_proxy = torch.where(
                empty,
                torch.zeros_like(mean_gap_float),
                1.0 / (min_gap.float() + 1.0),
            )
            valid_len_norm = torch.log1p(valid_count.float()) / math.log1p(max(L, 1))
            continuous.extend([
                (min_gap.float() / norm_denom).unsqueeze(1),
                recency_proxy.unsqueeze(1),
                valid_len_norm.unsqueeze(1),
                empty.float().unsqueeze(1),
            ])
        if self.use_seq_time_summary:
            if self.gap_embedding is not None:
                embeddings.append(self.gap_embedding(mean_gap))
                embeddings.append(self.gap_embedding(max_gap))
            spread = (max_gap.float() - min_gap.float()).clamp(min=0.0)
            continuous.extend([
                (mean_gap_float / norm_denom).unsqueeze(1),
                (max_gap.float() / norm_denom).unsqueeze(1),
                (spread / norm_denom).unsqueeze(1),
            ])

        stats = {
            f"{domain}_last_gap_mean": min_gap.float().mean() if B > 0 else tb.new_tensor(0.0, dtype=torch.float),
            f"{domain}_empty_ratio": empty.float().mean() if B > 0 else tb.new_tensor(0.0, dtype=torch.float),
            f"{domain}_valid_len_mean": valid_count.float().mean() if B > 0 else tb.new_tensor(0.0, dtype=torch.float),
        }
        return continuous, embeddings, stats

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ref = next(iter(seq_time_buckets.values()))
        B = ref.size(0)
        device = ref.device
        if timestamp is None:
            timestamp = torch.zeros(B, dtype=torch.long, device=device)
        else:
            timestamp = timestamp.to(device=device)

        continuous_parts: List[torch.Tensor] = [
            torch.ones(B, 1, dtype=torch.float, device=device)
        ]
        embedding_parts: List[torch.Tensor] = []
        stats: Dict[str, Any] = {}

        if self.use_sample_time_features and self.use_time_of_day_features:
            cont, embs, sample_stats = self._sample_time_features(timestamp)
            continuous_parts.extend(cont)
            embedding_parts.extend(embs)
            stats.update(sample_stats)
        else:
            stats["sample_time_available"] = torch.zeros((), dtype=torch.float, device=device)

        seq_time_available = []
        for domain in self.seq_domains:
            tb = seq_time_buckets[domain].to(device=device)
            seq_time_available.append((tb > 0).float().mean())
            if self.use_seq_recency_features or self.use_seq_time_summary:
                cont, embs, seq_stats = self._seq_gap_features(domain, tb, seq_lens[domain].to(device=device))
                continuous_parts.extend(cont)
                embedding_parts.extend(embs)
                stats.update(seq_stats)

        if seq_time_available:
            stats["seq_time_available"] = torch.stack(seq_time_available).mean()
        else:
            stats["seq_time_available"] = torch.zeros((), dtype=torch.float, device=device)

        continuous = torch.cat(continuous_parts, dim=1)
        embedding_parts.append(self.continuous_proj(continuous))
        token_input = torch.cat(embedding_parts, dim=1)
        time_token = self.feature_proj(token_input).unsqueeze(1)
        return time_token, stats


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        user_dense_feature_specs: List[Tuple[int, int, int]],
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # PM head-only experiment
        pm_head_enabled: bool = False,
        pm_feature_dim: int = 64,
        pm_feature_dropout: float = 0.05,
        pm_feature_norm_enabled: bool = True,
        # TimeToken experiment
        time_token_enabled: bool = False,
        time_token_dim: int = 0,
        time_token_dropout: float = 0.01,
        time_token_norm_enabled: bool = True,
        time_token_insert_position: str = 'ns_tokens',
        time_bucket_vocab_size: int = 0,
        time_gap_bucket_vocab_size: int = 0,
        use_sample_time_features: bool = True,
        use_seq_recency_features: bool = True,
        use_seq_time_summary: bool = True,
        time_token_ablation_mode: str = 'normal',
        use_time_of_day_features: bool = True,
        time_tz_offset_hours: int = 8,
        use_hour_embedding: bool = False,
        time_daypart_vocab_size: int = 7,
        # MissingAware + grouped dense semantic feature experiment
        missing_aware_enabled: bool = True,
        sparse_missing_indicator_enabled: bool = True,
        missing_indicator_project_to_group: bool = True,
        missing_residual_alpha_init: float = 0.1,
        missing_residual_alpha_learnable: bool = True,
        dense_missing_aware_enabled: bool = True,
        dense_stat_transform: str = "signed_log1p",
        dense_grouped_encoder_enabled: bool = True,
        dense_embedding_like_fids: Optional[List[int]] = None,
        dense_stat_like_fids: Optional[List[int]] = None,
        dense_missing_indicator_enabled: bool = True,
        dense_value_clip_abs: float = 0.0,
        dense_encoder_dropout: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.pm_head_enabled = pm_head_enabled
        self.pm_feature_dim = pm_feature_dim
        self.pm_feature_dropout = pm_feature_dropout
        self.pm_feature_norm_enabled = pm_feature_norm_enabled
        self.time_token_enabled = time_token_enabled
        self.time_token_insert_position = time_token_insert_position
        self.time_token_ablation_mode = time_token_ablation_mode
        self.time_token_dim = int(time_token_dim) if int(time_token_dim) > 0 else d_model
        self.time_token_dropout = time_token_dropout
        self.time_token_norm_enabled = time_token_norm_enabled
        self.use_sample_time_features = use_sample_time_features
        self.use_seq_recency_features = use_seq_recency_features
        self.use_seq_time_summary = use_seq_time_summary
        self.use_time_of_day_features = use_time_of_day_features
        self.time_tz_offset_hours = time_tz_offset_hours
        self.use_hour_embedding = use_hour_embedding
        self.time_daypart_vocab_size = time_daypart_vocab_size
        self.missing_aware_enabled = missing_aware_enabled
        self.sparse_missing_indicator_enabled = sparse_missing_indicator_enabled
        self.missing_indicator_project_to_group = missing_indicator_project_to_group
        self.missing_residual_alpha_init = float(missing_residual_alpha_init)
        self.missing_residual_alpha_learnable = missing_residual_alpha_learnable
        self.dense_missing_aware_enabled = dense_missing_aware_enabled
        self.dense_stat_transform = dense_stat_transform
        self.dense_grouped_encoder_enabled = dense_grouped_encoder_enabled
        self.dense_embedding_like_fids = dense_embedding_like_fids or [61, 87, 89, 90, 91]
        self.dense_stat_like_fids = dense_stat_like_fids or [62, 63, 64, 65, 66]
        self.dense_missing_indicator_enabled = dense_missing_indicator_enabled
        self.dense_value_clip_abs = dense_value_clip_abs
        self.dense_encoder_dropout = dense_encoder_dropout

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")
        self.num_user_int_tokens = int(num_user_ns)
        self.num_item_int_tokens = int(num_item_ns)
        self.ns_group_enabled = ns_tokenizer_type == 'group'

        self.user_missing_projs = nn.ModuleList([
            nn.Linear(len(group), d_model) for group in user_ns_groups
        ])
        self.item_missing_projs = nn.ModuleList([
            nn.Linear(len(group), d_model) for group in item_ns_groups
        ])
        alpha = torch.tensor(float(missing_residual_alpha_init))
        if missing_residual_alpha_learnable:
            self.missing_residual_alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("missing_residual_alpha", alpha)

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            if dense_grouped_encoder_enabled:
                self.user_dense_proj = UserDenseGroupedEncoder(
                    user_dense_feature_specs=user_dense_feature_specs or [],
                    d_model=d_model,
                    embedding_like_fids=self.dense_embedding_like_fids,
                    stat_like_fids=self.dense_stat_like_fids,
                    dense_stat_transform=dense_stat_transform,
                    dense_missing_indicator_enabled=dense_missing_indicator_enabled,
                    dense_missing_aware_enabled=dense_missing_aware_enabled,
                    dense_value_clip_abs=dense_value_clip_abs,
                    dropout=dense_encoder_dropout,
                )
            else:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        self.base_num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                            + num_item_ns + (1 if self.has_item_dense else 0))
        if time_token_enabled and time_token_insert_position != 'ns_tokens':
            raise ValueError(
                f"time_token_insert_position={time_token_insert_position!r} is not supported in H-TimeToken-v1; "
                "use 'ns_tokens'.")
        self.num_ns = self.base_num_ns + (1 if time_token_enabled else 0)

        if time_token_enabled:
            resolved_time_bucket_vocab_size = (
                int(time_bucket_vocab_size) if int(time_bucket_vocab_size) > 0 else max(num_time_buckets, 1)
            )
            resolved_time_gap_bucket_vocab_size = (
                int(time_gap_bucket_vocab_size) if int(time_gap_bucket_vocab_size) > 0 else max(num_time_buckets, 1)
            )
            self.time_token_encoder = TimeTokenEncoder(
                d_model=d_model,
                seq_domains=self.seq_domains,
                time_token_dim=self.time_token_dim,
                time_token_dropout=time_token_dropout,
                time_token_norm_enabled=time_token_norm_enabled,
                time_bucket_vocab_size=resolved_time_bucket_vocab_size,
                time_gap_bucket_vocab_size=resolved_time_gap_bucket_vocab_size,
                use_sample_time_features=use_sample_time_features,
                use_seq_recency_features=use_seq_recency_features,
                use_seq_time_summary=use_seq_time_summary,
                use_time_of_day_features=use_time_of_day_features,
                use_hour_embedding=use_hour_embedding,
                time_daypart_vocab_size=time_daypart_vocab_size,
                time_tz_offset_hours=time_tz_offset_hours,
            )
        else:
            self.time_token_encoder = None

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        if pm_head_enabled:
            self.pm_head = HyFormerPMHeadFeatureExtractor(
                d_model=d_model,
                pm_feature_dim=pm_feature_dim,
                pm_feature_dropout=pm_feature_dropout,
                pm_feature_norm_enabled=pm_feature_norm_enabled,
            )
            classifier_input_dim = d_model + pm_feature_dim
        else:
            self.pm_head = None
            classifier_input_dim = d_model

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _apply_time_token(
        self,
        ns_tokens: torch.Tensor,
        inputs: ModelInput,
        eval_time_mode: str = "normal",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "ns_tokens_before_shape": list(ns_tokens.shape),
            "ns_tokens_after_shape": list(ns_tokens.shape),
        }
        if not self.time_token_enabled or self.time_token_encoder is None:
            return ns_tokens, diagnostics

        if eval_time_mode not in {
            "normal",
            "without_time_token",
            "zero_time_token",
            "shuffle_time_token",
            "random_time_token",
        }:
            raise ValueError(f"unsupported eval_time_mode={eval_time_mode!r}")

        time_token, feature_stats = self.time_token_encoder(
            inputs.timestamp, inputs.seq_time_buckets, inputs.seq_lens)
        if eval_time_mode in {"without_time_token", "zero_time_token"}:
            time_token = torch.zeros_like(time_token)
        elif eval_time_mode == "shuffle_time_token" and time_token.size(0) > 1:
            order = torch.randperm(time_token.size(0), device=time_token.device)
            time_token = time_token[order]
        elif eval_time_mode == "random_time_token":
            scale = time_token.detach().float().std().clamp(min=1e-3).to(dtype=time_token.dtype)
            time_token = torch.randn_like(time_token) * scale

        ns_after = torch.cat([ns_tokens, time_token], dim=1)
        ns_norm = ns_tokens.detach().float().norm(dim=2).mean().view(1, 1)
        time_norm = time_token.detach().float().norm(dim=2).mean().view(1, 1)
        diagnostics.update({
            "time_token_shape": list(time_token.shape),
            "ns_tokens_after_shape": list(ns_after.shape),
            "time_token_norm": time_norm,
            "time_token_std": time_token.detach().float().std().view(1, 1),
            "hyformer_ns_token_norm": ns_norm,
            "time_to_ns_norm_ratio": time_norm / ns_norm.clamp(min=1e-6),
            "time_token_nan_count": torch.isnan(time_token).sum().detach(),
            "time_token_inf_count": torch.isinf(time_token).sum().detach(),
            "time_feature_stats": feature_stats,
        })
        return ns_after, diagnostics

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def _build_ns_tokens(
        self,
        inputs: ModelInput,
        eval_ns_group_mode: str = "normal",
        eval_missing_mode: str = "normal",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Build NS tokens and optionally mask one semantic group for eval.

        The default normal path is exactly the same tensor flow as the main
        PM+TimeToken model. Mask modes are used only for H-NSGroup-v2-item3
        validation ablations.
        """
        valid_modes = {
            "normal",
            "mask_item_low_card_token",
            "mask_item_mid_behavior_a_token",
            "mask_item_mid_behavior_b_token",
            "mask_item_high_id_like_token",
            "mask_user_low_context_token",
            "mask_user_profile_stat_shared_token",
            "mask_user_compact_tail_flags_token",
        }
        valid_modes.update({f"mask_user_group_{i}" for i in range(self.num_user_int_tokens)})
        if eval_ns_group_mode not in valid_modes:
            raise ValueError(f"unsupported eval_ns_group_mode={eval_ns_group_mode!r}")

        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)
        missing_diagnostics: Dict[str, Any] = {}
        use_sparse_missing = (
            self.missing_aware_enabled
            and self.sparse_missing_indicator_enabled
            and self.missing_indicator_project_to_group
            and eval_missing_mode not in {
                "without_missing_aware",
                "without_sparse_missing_indicators",
                "dense_missing_only",
            }
        )
        if use_sparse_missing:
            user_residuals = []
            item_residuals = []
            if inputs.user_int_missing_mask is not None:
                user_missing = inputs.user_int_missing_mask.to(dtype=user_ns.dtype, device=user_ns.device)
                for group_idx, (group, proj) in enumerate(zip(self.user_ns_tokenizer.groups, self.user_missing_projs)):
                    group_mask = user_missing[:, group]
                    residual = proj(group_mask)
                    user_ns[:, group_idx, :] = F.layer_norm(
                        user_ns[:, group_idx, :] + self.missing_residual_alpha * residual,
                        (user_ns.size(-1),),
                    )
                    user_residuals.append(residual.detach().float().norm(dim=1, keepdim=True))
                    missing_diagnostics[f"user_group_{group_idx}_missing_ratio"] = group_mask.detach().float().mean(dim=1, keepdim=True)
            if inputs.item_int_missing_mask is not None:
                item_missing = inputs.item_int_missing_mask.to(dtype=item_ns.dtype, device=item_ns.device)
                for group_idx, (group, proj) in enumerate(zip(self.item_ns_tokenizer.groups, self.item_missing_projs)):
                    group_mask = item_missing[:, group]
                    residual = proj(group_mask)
                    item_ns[:, group_idx, :] = F.layer_norm(
                        item_ns[:, group_idx, :] + self.missing_residual_alpha * residual,
                        (item_ns.size(-1),),
                    )
                    item_residuals.append(residual.detach().float().norm(dim=1, keepdim=True))
                    missing_diagnostics[f"item_group_{group_idx}_missing_ratio"] = group_mask.detach().float().mean(dim=1, keepdim=True)
            if user_residuals or item_residuals:
                residual_norm = torch.cat(user_residuals + item_residuals, dim=1).mean(dim=1, keepdim=True)
                token_norm = torch.cat([user_ns.detach().float().norm(dim=2), item_ns.detach().float().norm(dim=2)], dim=1).mean(dim=1, keepdim=True)
                missing_diagnostics["missing_residual_norm"] = residual_norm
                missing_diagnostics["missing_to_token_norm_ratio"] = residual_norm / token_norm.clamp(min=1e-6)
        else:
            missing_diagnostics["missing_residual_norm"] = user_ns.new_zeros(user_ns.size(0), 1)
            missing_diagnostics["missing_to_token_norm_ratio"] = user_ns.new_zeros(user_ns.size(0), 1)

        if eval_ns_group_mode.startswith("mask_user_group_"):
            group_idx = int(eval_ns_group_mode.rsplit("_", 1)[1])
            if group_idx >= user_ns.size(1):
                raise ValueError(
                    f"{eval_ns_group_mode} requested but user_ns has only {user_ns.size(1)} tokens")
            user_ns = user_ns.clone()
            user_ns[:, group_idx, :] = 0
        elif eval_ns_group_mode == "mask_item_low_card_token":
            if item_ns.size(1) < 1:
                raise ValueError("mask_item_low_card_token requested but item_ns has no tokens")
            item_ns = item_ns.clone()
            item_ns[:, 0, :] = 0
        elif eval_ns_group_mode == "mask_item_mid_behavior_a_token":
            if item_ns.size(1) < 2:
                raise ValueError("mask_item_mid_behavior_a_token requested but item_ns has fewer than 2 tokens")
            item_ns = item_ns.clone()
            item_ns[:, 1, :] = 0
        elif eval_ns_group_mode == "mask_item_mid_behavior_b_token":
            if item_ns.size(1) < 3:
                raise ValueError("mask_item_mid_behavior_b_token requested but item_ns has fewer than 3 tokens")
            item_ns = item_ns.clone()
            item_ns[:, 2, :] = 0
        elif eval_ns_group_mode == "mask_item_high_id_like_token":
            if item_ns.size(1) < 4:
                raise ValueError("mask_item_high_id_like_token requested but item_ns has fewer than 4 tokens")
            item_ns = item_ns.clone()
            item_ns[:, 3, :] = 0
        elif eval_ns_group_mode == "mask_user_low_context_token":
            user_ns = user_ns.clone()
            user_ns[:, 0, :] = 0
        elif eval_ns_group_mode == "mask_user_profile_stat_shared_token":
            user_ns = user_ns.clone()
            user_ns[:, 1, :] = 0
        elif eval_ns_group_mode == "mask_user_compact_tail_flags_token":
            user_ns = user_ns.clone()
            user_ns[:, 2, :] = 0

        ns_parts = [user_ns]
        dense_diagnostics: Dict[str, Any] = {}
        if self.has_user_dense:
            if self.dense_grouped_encoder_enabled:
                user_dense_tok, dense_diagnostics = self.user_dense_proj(
                    inputs.user_dense_feats,
                    dense_missing_mask=inputs.user_dense_missing_mask,
                    eval_missing_mode=eval_missing_mode,
                )
                user_dense_tok = user_dense_tok.unsqueeze(1)
            else:
                user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        diagnostics: Dict[str, Any] = {
            "ns_group_mode": eval_ns_group_mode,
            "missing_mode": eval_missing_mode,
            "user_int_tokens_shape": list(user_ns.shape),
            "item_int_tokens_shape": list(item_ns.shape),
            "user_dense_token_shape": list(user_dense_tok.shape) if self.has_user_dense else None,
            "ns_tokens_before_time_shape": list(ns_tokens.shape),
            "user_low_context_norm": user_ns[:, 0, :].detach().float().norm(dim=1, keepdim=True) if user_ns.size(1) > 0 else None,
            "user_profile_stat_shared_norm": user_ns[:, 1, :].detach().float().norm(dim=1, keepdim=True) if user_ns.size(1) > 1 else None,
            "user_compact_tail_flags_norm": user_ns[:, 2, :].detach().float().norm(dim=1, keepdim=True) if user_ns.size(1) > 2 else None,
            "item_low_card_norm": item_ns[:, 0, :].detach().float().norm(dim=1, keepdim=True) if item_ns.size(1) > 0 else None,
            "item_mid_behavior_a_norm": item_ns[:, 1, :].detach().float().norm(dim=1, keepdim=True) if item_ns.size(1) > 1 else None,
            "item_mid_behavior_b_norm": item_ns[:, 2, :].detach().float().norm(dim=1, keepdim=True) if item_ns.size(1) > 2 else None,
            "item_high_id_like_norm": item_ns[:, 3, :].detach().float().norm(dim=1, keepdim=True) if item_ns.size(1) > 3 else None,
            "user_dense_token_norm": user_dense_tok.detach().float().norm(dim=2).mean(dim=1, keepdim=True) if self.has_user_dense else None,
        }
        diagnostics.update(missing_diagnostics)
        diagnostics.update(dense_diagnostics)
        for i in range(self.num_user_int_tokens):
            diagnostics[f"user_group_{i}_shape"] = (
                list(user_ns[:, i:i + 1, :].shape) if i < user_ns.size(1) else None
            )
        for i in range(self.num_item_int_tokens):
            diagnostics[f"item_group_{i}_shape"] = (
                list(item_ns[:, i:i + 1, :].shape) if i < item_ns.size(1) else None
            )
        return ns_tokens, diagnostics

    def _build_classifier_input(
        self,
        output: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        eval_pm_mode: str = "normal",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if eval_pm_mode not in {"normal", "without_pm_head", "shuffle_pm_head"}:
            raise ValueError(f"unsupported eval_pm_mode={eval_pm_mode!r}")

        hyformer_repr_norm = output.detach().float().norm(dim=1, keepdim=True)
        diagnostics: Dict[str, torch.Tensor] = {
            "hyformer_repr_norm": hyformer_repr_norm,
        }
        if self.pm_head is None:
            diagnostics["pm_feature_norm"] = output.new_zeros(output.size(0), 1)
            diagnostics["pm_to_repr_norm_ratio"] = output.new_zeros(output.size(0), 1)
            return output, diagnostics

        if eval_pm_mode == "without_pm_head":
            pm_features = output.new_zeros(output.size(0), self.pm_feature_dim)
        else:
            pm_features = self.pm_head(ns_tokens, seq_tokens_list, seq_masks_list)
            if eval_pm_mode == "shuffle_pm_head" and pm_features.size(0) > 1:
                order = torch.randperm(pm_features.size(0), device=pm_features.device)
                pm_features = pm_features[order]

        pm_features = pm_features.to(dtype=output.dtype)
        pm_feature_norm = pm_features.detach().float().norm(dim=1, keepdim=True)
        diagnostics["pm_feature_norm"] = pm_feature_norm
        diagnostics["pm_to_repr_norm_ratio"] = pm_feature_norm / hyformer_repr_norm.clamp(min=1e-6)
        return torch.cat([output, pm_features], dim=1), diagnostics

    def forward(
        self,
        inputs: ModelInput,
        eval_pm_mode: str = "normal",
        eval_time_mode: str = "normal",
        eval_ns_group_mode: str = "normal",
        eval_missing_mode: str = "normal",
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: semantic groups + existing user dense token
        ns_tokens, ns_group_diagnostics = self._build_ns_tokens(inputs, eval_ns_group_mode, eval_missing_mode)
        ns_tokens, time_diagnostics = self._apply_time_token(ns_tokens, inputs, eval_time_mode)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        classifier_input, diagnostics = self._build_classifier_input(
            output, ns_tokens, seq_tokens_list, seq_masks_list, eval_pm_mode
        )
        diagnostics.update(ns_group_diagnostics)
        diagnostics.update(time_diagnostics)
        diagnostics["ns_tokens_shape"] = list(ns_tokens.shape)
        diagnostics["rankmixer_input_shape"] = [
            ns_tokens.size(0),
            self.num_queries * self.num_sequences + ns_tokens.size(1),
            ns_tokens.size(2),
        ]
        diagnostics["classifier_input_shape"] = list(classifier_input.shape)
        diagnostics["nan_count"] = (
            torch.isnan(ns_tokens).sum().detach()
            + torch.isnan(classifier_input).sum().detach()
        )
        diagnostics["inf_count"] = (
            torch.isinf(ns_tokens).sum().detach()
            + torch.isinf(classifier_input).sum().detach()
        )
        if "time_token_norm" in diagnostics:
            diagnostics["time_to_repr_norm_ratio"] = (
                diagnostics["time_token_norm"] / diagnostics["hyformer_repr_norm"].clamp(min=1e-6)
            )
            diagnostics["time_to_pm_norm_ratio"] = (
                diagnostics["time_token_norm"] / diagnostics["pm_feature_norm"].clamp(min=1e-6)
            )
        logits = self.clsfier(classifier_input)  # (B, action_num)
        if return_diagnostics:
            return {
                "logits": logits,
                "hyformer_repr": output,
                "diagnostics": diagnostics,
            }
        return logits

    def predict(
        self,
        inputs: ModelInput,
        eval_pm_mode: str = "normal",
        eval_time_mode: str = "normal",
        eval_ns_group_mode: str = "normal",
        eval_missing_mode: str = "normal",
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        ns_tokens, ns_group_diagnostics = self._build_ns_tokens(inputs, eval_ns_group_mode, eval_missing_mode)
        ns_tokens, time_diagnostics = self._apply_time_token(ns_tokens, inputs, eval_time_mode)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        classifier_input, diagnostics = self._build_classifier_input(
            output, ns_tokens, seq_tokens_list, seq_masks_list, eval_pm_mode
        )
        diagnostics.update(ns_group_diagnostics)
        diagnostics.update(time_diagnostics)
        diagnostics["ns_tokens_shape"] = list(ns_tokens.shape)
        diagnostics["rankmixer_input_shape"] = [
            ns_tokens.size(0),
            self.num_queries * self.num_sequences + ns_tokens.size(1),
            ns_tokens.size(2),
        ]
        diagnostics["classifier_input_shape"] = list(classifier_input.shape)
        diagnostics["nan_count"] = (
            torch.isnan(ns_tokens).sum().detach()
            + torch.isnan(classifier_input).sum().detach()
        )
        diagnostics["inf_count"] = (
            torch.isinf(ns_tokens).sum().detach()
            + torch.isinf(classifier_input).sum().detach()
        )
        if "time_token_norm" in diagnostics:
            diagnostics["time_to_repr_norm_ratio"] = (
                diagnostics["time_token_norm"] / diagnostics["hyformer_repr_norm"].clamp(min=1e-6)
            )
            diagnostics["time_to_pm_norm_ratio"] = (
                diagnostics["time_token_norm"] / diagnostics["pm_feature_norm"].clamp(min=1e-6)
            )
        logits = self.clsfier(classifier_input)
        if return_diagnostics:
            return logits, output, diagnostics
        return logits, output
