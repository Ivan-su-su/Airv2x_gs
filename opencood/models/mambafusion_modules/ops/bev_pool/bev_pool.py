import torch

from . import bev_pool_ext

__all__ = ["bev_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        # 空数据保护：如果输入为空，返回零张量
        if x.shape[0] == 0:
            out = torch.zeros((B, D, H, W, x.shape[1] if len(x.shape) > 1 else 1), device=x.device, dtype=x.dtype)
            ctx.save_for_backward(
                torch.zeros(0, dtype=torch.int32, device=x.device),
                torch.zeros(0, dtype=torch.int32, device=x.device),
                torch.zeros(0, geom_feats.shape[1] if len(geom_feats.shape) > 1 else 4, dtype=torch.int32, device=x.device)
            )
            ctx.saved_shapes = B, D, H, W
            return out
        
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        
        # 空数据保护：如果 interval_starts 为空，返回零张量
        if interval_starts.shape[0] == 0:
            out = torch.zeros((B, D, H, W, x.shape[1] if len(x.shape) > 1 else 1), device=x.device, dtype=x.dtype)
            ctx.save_for_backward(interval_starts, torch.zeros_like(interval_starts), geom_feats.int())
            ctx.saved_shapes = B, D, H, W
            return out
        
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        # 空数据保护：如果 interval_starts 为空，返回零梯度
        if interval_starts.shape[0] == 0:
            # 从 out_grad 推断特征维度
            C = out_grad.shape[1] if len(out_grad.shape) > 1 else 1
            x_grad = torch.zeros(0, C, device=out_grad.device, dtype=out_grad.dtype)
            return x_grad, None, None, None, None, None, None

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
