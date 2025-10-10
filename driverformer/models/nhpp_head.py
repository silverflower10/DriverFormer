#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/nhpp_head.py — NHPP rate head (safe version)

Updated on Mon Jul  7 19:20:00 2025  ← attention-save policy + safety clamps + grad sanitizer

- Transformer 출력 x(B,T,D) → per-kb rate λ(B,T)
- softplus로 양수 보장
- exp(log_c) 오버플로우 방지(clamp)
- 최종 λ는 [rate_min, rate_max]로 클램프 + NaN/Inf 치환
- backward 시 λ-그래디언트 NaN/Inf를 0으로 살균(hook)해 MulBackward0 NaN 전파 차단
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NHPPHead"]


class NHPPHead(nn.Module):
    """
    NHPPHead(hidden_dim=768, rate_min=1e-9, rate_max=1e6, ln_eps=1e-5, softplus_threshold=20.0)

    Args
    ----
    hidden_dim : int
        입력 임베딩 차원(D).
    rate_min : float
        λ 하한 (log(λ) 계산 안정성; 0 금지).
    rate_max : float
        λ 상한 (수치 폭주 방지).
    ln_eps : float
        LayerNorm eps.
    softplus_threshold : float
        softplus 안정 임계값(큰 입력에서 선형 근사 사용).
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        rate_min: float = 1e-9,
        rate_max: float = 1e6,
        ln_eps: float = 1e-5,
        softplus_threshold: float = 20.0,
    ):
        super().__init__()
        self.ln   = nn.LayerNorm(hidden_dim, eps=ln_eps)
        self.lin  = nn.Linear(hidden_dim, 1)
        self.log_c = nn.Parameter(torch.zeros(()))  # 전역 스케일(log)
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.sp_th    = float(softplus_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D) → λ: (B,T)
        """
        # LayerNorm + 선형 → softplus로 양수 rate
        x = self.ln(x.contiguous())
        rate = F.softplus(self.lin(x), beta=1.0, threshold=self.sp_th).squeeze(-1)  # (B,T) ≥ 0

        # exp(log_c) 안정화 (overflow 방지)
        scale = torch.exp(self.log_c.clamp(-20, 20))

        lam = rate * scale  # (B,T)

        # NaN/Inf 치환 후 범위 클램프
        lam = torch.nan_to_num(lam, nan=self.rate_min, posinf=self.rate_max, neginf=self.rate_min)
        lam = lam.clamp(self.rate_min, self.rate_max)

        # ★ Gradient sanitizer: 손실에서 내려오는 ∂L/∂λ가 NaN/Inf면 0으로 정리
        if lam.requires_grad:
            lam.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))

        return lam


if __name__ == "__main__":  # quick sanity (옵션)
    B, T, D = 2, 5, 768
    head = NHPPHead(D)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T)
    assert torch.isfinite(out).all() and (out > 0).all()
    print("NHPPHead sanity OK:", out.min().item(), out.max().item())
