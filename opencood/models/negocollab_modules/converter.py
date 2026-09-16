# -*- coding: utf-8 -*-
"""Window-grid Converter / CrossDomainConverter vendored from official NegoCollab.

Source: NegoCollab `opencood/models/fuse_modules/wg_fusion_modules.py`.
`feature_show` debug imports are removed. Class names match the official
modules so ComminPub can be a close port.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def sc_padding(x, window_size: int):
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0
    _, _, height, width = x.size()
    h_sc, w_sc = height // window_size, width // window_size
    res_h = height % window_size
    if res_h > 0:
        h_sc += 1
        padding_bottom = window_size - res_h
    res_w = width % window_size
    if res_w > 0:
        w_sc += 1
        padding_right = window_size - res_w
    return [h_sc, w_sc], [padding_left, padding_right, padding_top, padding_bottom]


def sc_unpadding(x, padding: List[int]):
    if x.dim() == 4:
        if padding[1] > 0:
            x = x[:, :, :, :-padding[1]]
        if padding[3] > 0:
            x = x[:, :, :-padding[3], :]
    return x


class Attention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, dropout: float = 0.0, window_size: int = 7) -> None:
        super().__init__()
        assert dim % dim_head == 0, "dimension should be divisible by dimension per head"
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, heads = (
            *x.shape,
            x.device,
            self.heads,
        )
        del batch, device
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")
        attn = self.attend(sim)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class SwapFusionBlock(nn.Module):
    def __init__(self, input_dim: int, mlp_dim: int, dim_head: int, window_size: int, drop_out: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Rearrange("b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, window_size)),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            Rearrange("b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, window_size)),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
        )

    def forward(self, x, mask=None):
        del mask
        return self.block(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, qkv_bias: bool = False, rel_pos_emb: bool = False, norm=nn.LayerNorm) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None):
        assert k.shape == v.shape
        _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        q = rearrange(q, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        k = rearrange(k, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        v = rearrange(v, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        dot = self.scale * torch.einsum("b l Q d, b l K d -> b l Q K", q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)
        a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)
        a = rearrange(
            a,
            "b (x y) (w1 w2) d -> b x y w1 w2 d",
            x=q_height,
            y=q_width,
            w1=q_win_height,
            w2=q_win_width,
        )
        z = self.proj(a)
        if skip is not None:
            z = z + skip
        return z


class CrossDomainSwapFusionBlock(nn.Module):
    def __init__(self, dim: int, dim_heads: int, heads: int, qkv_bias: bool, win_size: int) -> None:
        super().__init__()
        self.win_size = win_size
        self.prenorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.cross_win = CrossAttention(dim, heads, dim_heads, qkv_bias)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, ego, cav_feature):
        query = rearrange(
            cav_feature, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.win_size, w2=self.win_size
        )
        key = rearrange(
            ego, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.win_size, w2=self.win_size
        )
        value = rearrange(
            ego, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.win_size, w2=self.win_size
        )
        query = rearrange(self.cross_win(query, key, value, skip=query), "b x y w1 w2 d -> b (x w1) (y w2) d")
        query = self.prenorm(query)
        query = query + self.ff(query)
        query = self.post_norm(query)
        return rearrange(query, "b h w d -> b d h w")


class Converter(nn.Module):
    """Sender-side window-grid converter (official NegoCollab)."""

    def __init__(self, args) -> None:
        super().__init__()
        self.depth = args["num_of_blocks"]
        input_dim = args["dim"]
        mlp_dim = args["dim"]
        window_size = args["window_size"]
        drop_out = args["drop_out"]
        heads = args["heads"]
        dim_head = input_dim // heads
        self.window_size = window_size
        self.layers = nn.ModuleList(
            [
                SwapFusionBlock(input_dim, mlp_dim, dim_head, window_size, drop_out)
                for _ in range(self.depth)
            ]
        )
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, x, mask=None):
        _, padding_pos = sc_padding(x, self.window_size)
        x = F.pad(x, padding_pos)
        for stage in self.layers:
            x = stage(x, mask=mask)
        x = sc_unpadding(x, padding_pos)
        return self.mlp_head(x)


class CrossdomianConverter(nn.Module):
    """Receiver-side ego-conditioned converter. Official class name is kept."""

    def __init__(self, args) -> None:
        super().__init__()
        self.depth = args["num_of_blocks"]
        input_dim = args["dim"]
        heads = args["heads"]
        dim_head = input_dim // heads
        window_size = args["window_size"]
        self.window_size = window_size
        self.layers = nn.ModuleList(
            [
                CrossDomainSwapFusionBlock(input_dim, dim_head, heads, True, window_size)
                for _ in range(self.depth)
            ]
        )
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, ego_feature, cav_feature):
        _, padding_pos = sc_padding(ego_feature, self.window_size)
        ego_feature = F.pad(ego_feature, padding_pos)
        _, padding_pos = sc_padding(cav_feature, self.window_size)
        cav_feature = F.pad(cav_feature, padding_pos)
        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
        x = sc_unpadding(x, padding_pos)
        return self.mlp_head(x)
