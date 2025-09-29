#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025    ← NEW  (attention‑save policy)

@author: silverflo

Changes in this revision
------------------------
* 학습(epoch) 중 : Val‑loss 개선 시 **마지막 레이어 head‑mean 1장** 저장
* 학습 완료(final): 베스트 모델로 **전체 레이어·헤드** 스택 저장
* 그 외 로직(길이‑가중 residual sampler, Huber/IRLS 등)은 이전 버전과 동일
"""

# --------------------------------------------------------------------------- #
# Imports & setup                                                             #
# --------------------------------------------------------------------------- #
import os, sys, math, random, pickle, argparse, gc, itertools
from functools import partial
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from torch.nn.functional import scaled_dot_product_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from rotary_embedding_torch import RotaryEmbedding
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import fdrcorrection_twostage
from tqdm import tqdm
import warnings
from scipy.special import logsumexp
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
DEBUG_NAN = True  # NaN 디버깅

# --------------------------------------------------------------------------- #

def rolling_sum_nhpp(lam: torch.Tensor, y: torch.Tensor, len_kb: torch.Tensor,
                     *, width: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NHPP-정합 롤링(합), 길이 보존(causal/right-aligned):
      y'  = Σ_{j=t-width+1..t} y_j
      μ'  = Σ (λ_j·Δt_j)
      Δt' = Σ Δt_j
      λ'  = μ'/Δt'
    width<=1이면 변화 없음.
    """
    if width <= 1:
        return lam, y, len_kb
    eps = 1e-9
    k = _make_kernel_box(width, y.device)

    mu    = lam * len_kb                         # 기대카운트 μ = λ·Δt
    y_r   = _conv1d_causal_sum(y,      k)        # (B,T)
    mu_r  = _conv1d_causal_sum(mu,     k)        # (B,T)
    dt_r  = _conv1d_causal_sum(len_kb, k)        # (B,T)
    lam_r = mu_r / (dt_r + eps)                  # 길이로 가중한 λ 평균

    return lam_r, y_r, dt_r

def _make_kernel_box(width: int, device):
    w = int(max(1, width))
    return torch.ones(1, 1, w, device=device, dtype=torch.float32)  # (out=1,in=1,K)

def _conv1d_causal_sum(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T), k: (1,1,K)
    왼쪽만 (K-1) 만큼 0-패딩 → 출력 길이 = T 유지
    각 위치 t에서 윈도우 [t-K+1, ..., t] 합(부족분은 0으로 보충)
    """
    K = k.size(-1)
    x_pad = F.pad(x.unsqueeze(1), (K-1, 0))   # (B,1,T+K-1)
    return torch.conv1d(x_pad, k, padding=0).squeeze(1) if hasattr(torch, "conv1d") else F.conv1d(x_pad, k, padding=0).squeeze(1)

