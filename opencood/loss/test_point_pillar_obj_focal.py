"""Unit tests for the new objectness binary focal loss math.

Only tests the obj focal semantics (ignore exclusion, focal suppression,
gradient flow, Npos normalization). No dataset dependency.
"""

import torch
import torch.nn.functional as F


def _obj_focal(obj_preds: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, obj_alpha: float = 0.25, obj_gamma: float = 2.0) -> torch.Tensor:
    """Mirror of the active objectness loss in PointPillarLossMultiClass.

    Kept in sync intentionally: this is the exact math under test, in a
    batch-of-1 friendly layout.
    """
    pos_mask = pos_mask.to(device=obj_preds.device, dtype=obj_preds.dtype)
    neg_mask = neg_mask.to(device=obj_preds.device, dtype=obj_preds.dtype)
    valid_mask = (pos_mask + neg_mask).clamp(max=1.0)

    target = pos_mask
    bce = F.binary_cross_entropy_with_logits(
        obj_preds, target, reduction="none"
    )
    p = torch.sigmoid(obj_preds)
    alpha_t = (
        target * obj_alpha
        + (1.0 - target) * (1.0 - obj_alpha)
    )
    pt_error = (
        target * (1.0 - p)
        + (1.0 - target) * p
    )
    focal_weight = (
        alpha_t
        * pt_error.pow(obj_gamma)
    )

    obj_loss_raw = (
        focal_weight
        * bce
        * valid_mask
    )

    num_pos = pos_mask.sum(
        dim=(1, 2, 3)
    ).clamp(min=1.0)

    obj_loss = (
        obj_loss_raw.sum(dim=(1, 2, 3))
        / num_pos
    ).mean()
    return obj_loss


def test_ignore_anchor_contributes_zero() -> None:
    """Anchors with pos=0, neg=0 must contribute exactly 0 to the loss."""
    pos = torch.zeros(1, 2, 2, 1)
    neg = torch.zeros(1, 2, 2, 1)
    logits = torch.full((1, 2, 2, 1), 1.0e4, requires_grad=True)
    loss = _obj_focal(logits, pos, neg)
    # Only ignore anchors exist; num_pos clamps to 1, raw sum is 0.
    assert float(loss.item()) == 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) == 0.0


def test_easy_negative_suppressed_by_focal() -> None:
    """gamma=2 focal must down-weight an easy negative (p=0.01) hard."""
    neg = torch.ones(1, 1, 1, 1)
    pos = torch.zeros(1, 1, 1, 1)
    # p = 0.01 -> logit = logit(0.01); p = 0.5 -> logit = 0.
    easy = torch.log(torch.tensor(0.01 / 0.99))
    hard = torch.tensor(0.0)
    loss_easy = _obj_focal(easy.reshape(1, 1, 1, 1), pos, neg)
    loss_hard = _obj_focal(hard.reshape(1, 1, 1, 1), pos, neg)
    ratio = float(loss_easy.item()) / float(loss_hard.item())
    # (1-p)^2 at p=0.01 is 1e-4 vs 0.25 at p=0.5 -> ratio ~4e-4.
    assert ratio < 0.01, f"easy/hard focal ratio {ratio:.6f} not suppressed"
    assert float(loss_easy.item()) < 0.05 * float(loss_hard.item())


def test_hard_negative_has_nonzero_gradient() -> None:
    """A hard negative (p=0.5) must still receive a finite, nonzero grad."""
    neg = torch.ones(1, 1, 1, 1)
    pos = torch.zeros(1, 1, 1, 1)
    logits = torch.zeros(1, 1, 1, 1, requires_grad=True)  # p=0.5
    loss = _obj_focal(logits, pos, neg)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_positive_gradient_pushes_logit_up() -> None:
    """A positive at low p must get a gradient that increases its logit."""
    pos = torch.ones(1, 1, 1, 1)
    neg = torch.zeros(1, 1, 1, 1)
    # p = 0.1 -> logit = log(0.1/0.9)
    logits = torch.log(torch.tensor(0.1 / 0.9)).reshape(1, 1, 1, 1).clone().requires_grad_(True)
    loss = _obj_focal(logits, pos, neg)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum().item()) > 0.0
    # Gradient descent (logit -= lr * grad) must increase the logit.
    assert float(logits.grad.item()) < 0.0


def test_normalization_by_num_pos_not_num_neg() -> None:
    """Adding easy negatives must not change the loss: denominator is Npos."""
    torch.manual_seed(0)
    # One hard-ish positive at p=0.3, target=1.
    pos_logit = torch.log(torch.tensor(0.3 / 0.7))
    H = W = 4
    pos = torch.zeros(1, H, W, 1)
    neg = torch.zeros(1, H, W, 1)
    pos[0, 0, 0, 0] = 1.0
    # Baseline: one positive only.
    logits_a = torch.full((1, H, W, 1), float(pos_logit))
    loss_a = _obj_focal(logits_a, pos, neg)
    # Variant: same positive + many easy negatives (p=0.01).
    neg[0, 0, 1, 0] = 1.0
    neg[0, 1, 0, 0] = 1.0
    neg[0, 1, 1, 0] = 1.0
    easy_logit = float(torch.log(torch.tensor(0.01 / 0.99)))
    logits_b = torch.full((1, H, W, 1), easy_logit)
    logits_b[0, 0, 0, 0] = float(pos_logit)
    loss_b = _obj_focal(logits_b, pos, neg)
    # Denominator is num_pos=1 in both; the extra easy negatives add a
    # tiny (focal-suppressed) amount but do NOT rescale the whole loss.
    diff = abs(float(loss_a.item()) - float(loss_b.item()))
    base = float(loss_a.item())
    assert diff / base < 0.05, (
        f"loss changed by {diff:.6f} ({diff / base * 100:.1f}%) after adding "
        "easy negatives; normalization must be num_pos only"
    )


if __name__ == "__main__":
    test_ignore_anchor_contributes_zero()
    test_easy_negative_suppressed_by_focal()
    test_hard_negative_has_nonzero_gradient()
    test_positive_gradient_pushes_logit_up()
    test_normalization_by_num_pos_not_num_neg()
    print("obj focal tests ok")
