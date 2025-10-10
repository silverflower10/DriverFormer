#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
losses/nhpp.py — NHPP negative log-likelihood (SAFE)

- lam, y, dt 모두 NaN/Inf → 수치 치환
- lam/log(λ) 수치안정: lam ∈ [LAM_MIN, LAM_MAX]
- dt 음수 금지, 분모 0 방지(EPS)
- 세그먼트 길이 정규화(per kb × 30) 동일 유지
"""

from __future__ import annotations
import torch

# ===== 수치 안전 상수 =====
LAM_MIN: float = 1e-9
LAM_MAX: float = 1e6
EPS:     float = 1e-12

__all__ = [
    "trapezoid_nhpp_loss",
    "trapezoid_nhpp_loss_segment_weighted",
]


def _sanitize(lam: torch.Tensor, y: torch.Tensor, dt: torch.Tensor):
    """NaN/Inf 및 음수 길이 방지."""
    lam = torch.nan_to_num(lam, nan=LAM_MIN, posinf=LAM_MAX, neginf=LAM_MIN)
    y   = torch.nan_to_num(y,   nan=0.0,     posinf=0.0,     neginf=0.0).clamp_min(0.0)
    dt  = torch.nan_to_num(dt,  nan=0.0,     posinf=0.0,     neginf=0.0).clamp_min(0.0)
    return lam, y, dt


def trapezoid_nhpp_loss(
    lam: torch.Tensor,          # (B,T)
    y:   torch.Tensor,          # (B,T)
    len_kb: torch.Tensor,       # (B,T)  Δt
    reduction: str = "mean",    # "mean" | "none"
) -> torch.Tensor:
    """
    Bin-level NHPP 음의 로그우도:
        -ℓ = - Σ_i [ y_i log λ_i  -  λ_i * Δt_i ]
    """
    lam, y, dt = _sanitize(lam, y, len_kb)
    lam_safe = lam.clamp(LAM_MIN, LAM_MAX)

    # NLL = -( y·logλ - λΔt )
    sum_log = (y * torch.log(lam_safe)).sum(dim=1)   # (B,)
    integ   = (lam_safe * dt).sum(dim=1)             # (B,)
    neg_ll  = -(sum_log - integ)                     # (B,)

    neg_ll = torch.nan_to_num(neg_ll, nan=0.0, posinf=1e6, neginf=1e6)
    if reduction == "none":
        return neg_ll
    elif reduction == "mean":
        return neg_ll.mean()
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


def trapezoid_nhpp_loss_segment_weighted(
    lam: torch.Tensor,          # (B,T)
    y:   torch.Tensor,          # (B,T)
    len_kb: torch.Tensor,       # (B,T)
    w_seg: torch.Tensor,        # (B,)
) -> torch.Tensor:
    """
    세그먼트별 NLL(bin) → (per-kb 정규화 × 30) → 세그먼트 가중 평균 스칼라 손실.
    """
    lam, y, dt = _sanitize(lam, y, len_kb)
    lam_safe = lam.clamp(LAM_MIN, LAM_MAX)

    # bin-level NLL per segment
    per_seg = trapezoid_nhpp_loss(lam_safe, y, dt, reduction="none")  # (B,)

    # 길이 정규화(per kb) 후 ×30
    seg_len = dt.sum(dim=1).clamp_min(EPS)        # (B,)
    per_seg = (per_seg / seg_len) * 30.0          # (B,)

    # 가중 평균
    per_seg = torch.nan_to_num(per_seg, nan=0.0, posinf=1e6, neginf=1e6)
    w = torch.nan_to_num(w_seg, nan=1.0, posinf=1.0, neginf=1.0).to(per_seg.dtype)
    w_sum = w.sum().clamp_min(EPS)
    loss  = (w * per_seg).sum() / w_sum
    return loss
