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

def trapezoid_nhpp_loss(lam, y, len_kb, reduction="mean"):
    """
    Bin-level(사각형) NHPP 음의 로그우도:
      -ℓ = - Σ_i [ y_i log λ_i  -  λ_i * Δt_i ]
    """
    lam_safe = lam.clamp(1e-9, 1e4)
    # Σ y_i log λ_i
    sum_log = (y * torch.log(lam_safe)).sum(dim=1)
    # Σ λ_i * Δt_i   ← (사다리꼴이 아니라 각 bin 자체 길이 사용)
    integ = (lam_safe * len_kb).sum(dim=1)
    ll = sum_log - integ
    return (-ll if reduction == "none" else -ll.mean())


def trapezoid_nhpp_loss_segment_weighted(lam, y, len_kb, w_seg):
    """
    세그먼트별 NLL(bin) → (per-kb 정규화 × 30) → 세그먼트 허버가중 평균
    (내부의 기본 손실은 bin-level NLL을 호출)
    """
    per_seg = trapezoid_nhpp_loss(lam, y, len_kb, reduction="none")  # (B,)
    seg_len = (len_kb.sum(dim=1) + 1e-9)
    per_seg = (per_seg / seg_len) * 30.0
    return (w_seg * per_seg).sum() / (w_seg.sum() + 1e-9)
