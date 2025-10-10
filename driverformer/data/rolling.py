#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rolling.py — NHPP 정합 롤링(합) 안전 구현

- cuDNN 1D-conv 경로를 비활성화하여 backward NaN 전파 방지
- 입력/커널의 dtype/device 일치
- 분모(Δt 합) 0 방지용 클램프
- NaN/Inf → 수치 치환

API
---
rolling_sum_nhpp(lam, y, len_kb, width=2) -> (lam_roll, y_roll, len_roll)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import Tuple

# 수치 안전 상수
_RATE_MIN  = 1e-9
_RATE_MAX  = 1e6
_DEN_MIN   = 1e-12


def _nan_to_num_(t: torch.Tensor) -> torch.Tensor:
    """NaN/Inf를 안전한 수치로 치환."""
    return torch.nan_to_num(t, nan=0.0, posinf=_RATE_MAX, neginf=0.0)


def rolling_sum_nhpp(
    lam: torch.Tensor,        # (B,T)  per-kb rate λ
    y: torch.Tensor,          # (B,T)  관측 카운트
    len_kb: torch.Tensor,     # (B,T)  구간 길이(Δt; kb)
    *,
    width: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NHPP-정합 롤링(합), causal/right-aligned(좌측 pad):

      y'(t)   = Σ_{j=t-width+1..t} y(j)
      μ'(t)   = Σ_{j=t-width+1..t} [ λ(j) * Δt(j) ]
      Δt'(t)  = Σ_{j=t-width+1..t} Δt(j)
      λ'(t)   = μ'(t) / Δt'(t)

    width <= 1이면 원본을 그대로 반환합니다.
    """
    if width <= 1:
        return lam, y, len_kb

    # ---- 수치 안전화 ----
    lam = _nan_to_num_(lam).clamp(_RATE_MIN, _RATE_MAX)
    y   = _nan_to_num_(y).clamp(min=0.0)
    dt  = _nan_to_num_(len_kb).clamp(min=0.0)

    # 커널: 입력과 dtype/device 일치
    k = _make_kernel_box(width, device=lam.device, dtype=lam.dtype)

    # 기대값 μ = λ · Δt
    mu   = lam * dt

    # causal box-sum (좌측 zero-pad), cuDNN 비활성화로 math 경로만 사용
    y_r  = _conv1d_causal_sum(y,  k)   # (B,T)
    mu_r = _conv1d_causal_sum(mu, k)   # (B,T)
    dt_r = _conv1d_causal_sum(dt, k)   # (B,T)

    # 분모 0 방지 + λ' 범위 고정
    dt_r  = dt_r.clamp_min(_DEN_MIN)
    lam_r = (mu_r / dt_r).clamp(_RATE_MIN, _RATE_MAX)

    return lam_r, y_r, dt_r


def _make_kernel_box(width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """길이 width의 box 커널 (1,1,K)."""
    w = int(max(1, width))
    return torch.ones(1, 1, w, device=device, dtype=dtype)


def _conv1d_causal_sum(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x: (B,T), k: (1,1,K)
    왼쪽(K-1) zero-pad → 출력 길이 T 유지.
    cuDNN을 끄고 수학 경로(F.conv1d)로만 수행해 backward NaN을 방지.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B,T), got {tuple(x.shape)}")
    if k.ndim != 3 or k.size(0) != 1 or k.size(1) != 1:
        raise ValueError(f"k must be (1,1,K), got {tuple(k.shape)}")

    B, T = x.shape
    K    = int(k.size(-1))

    x1   = x.unsqueeze(1).contiguous()           # (B,1,T)
    xpad = F.pad(x1, (K-1, 0)).contiguous()      # (B,1,T+K-1)

    # ★ 중요: cuDNN 비활성화 → math 경로로 강제
    with cudnn.flags(enabled=False, benchmark=False, deterministic=True):
        s = F.conv1d(xpad, k, padding=0)         # (B,1,T)
    return s.squeeze(1).contiguous()             # (B,T)


__all__ = ["rolling_sum_nhpp"]
