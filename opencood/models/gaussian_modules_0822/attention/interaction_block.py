"""Pre-Norm multi-head interaction block for Gaussian query features."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class GaussianInteractionBlock(nn.Module):
    """Pre-Norm attention + FFN. Self-attn when ``context`` is omitted.

    ``query`` is ``[N, C]`` or ``[B, Lq, C]``. ``context`` is ``[N, M, C]``
    or ``[B, Lkv, C]``. ``mask`` is bool ``[N, M]`` / ``[B, Lkv]`` with
    True = valid token.
    """

    def __init__(self, dim: int, heads: int = 4, ffn_dim: Optional[int] = None) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = int(heads)
        self.head_dim = int(dim) // int(heads)
        width = int(ffn_dim) if ffn_dim is not None else 2 * int(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return query
        squeeze = query.dim() == 2
        if squeeze:
            query = query.unsqueeze(1)
        if context is None:
            context = query
        elif context.dim() == 2:
            context = context.unsqueeze(1)

        q_in = self.query_norm(query)
        kv_in = self.context_norm(context)
        batch, n_q, dim = q_in.shape
        n_kv = kv_in.shape[1]
        q = self.q_proj(q_in).view(batch, n_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(batch, n_kv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(batch, n_kv, self.heads, self.head_dim).transpose(1, 2)

        keep = None
        if mask is None:
            attn = F.scaled_dot_product_attention(q, k, v)
        else:
            valid = mask.bool()
            keep = valid.any(dim=-1)
            # Dummy key only for all-invalid rows. A bool all-ignore mask
            # NaNs on torch 2.1; an always-on ignored sentinel also blocks
            # QKV gradients.
            k = torch.cat([k, k.new_zeros(batch, self.heads, 1, self.head_dim)], dim=2)
            v = torch.cat([v, v.new_zeros(batch, self.heads, 1, self.head_dim)], dim=2)
            valid_ext = torch.cat([valid, ~keep[:, None]], dim=-1)
            bias = q.new_zeros(batch, 1, 1, n_kv + 1)
            bias = bias.masked_fill(~valid_ext[:, None, None, :], torch.finfo(q.dtype).min)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

        x1 = query + self.out_proj(attn.transpose(1, 2).reshape(batch, n_q, dim))
        out = x1 + self.ffn(self.ffn_norm(x1))
        if keep is not None:
            out = torch.where(keep[:, None, None], out, query)
        if squeeze:
            out = out.squeeze(1)
        return out
