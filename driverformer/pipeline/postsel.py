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

def estimate_pi0_storey_bootstrap(p_values,
                                  lambdas=None,
                                  B=200,
                                  seed=None,
                                  pi0_floor=0.01,
                                  pi0_ceil=1.0):
    """
    Storey(2002/2003): λ-grid에서 π0(λ)=#{p>λ}/((1-λ)m) 추정 후,
    부트스트랩으로 MSE 최소 λ* 선택.
    """
    p = _sanitize_pvals(p_values); m = p.size
    if m == 0: return 1.0, 0.5, np.array([1.0]), np.array([0.0])
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.01, dtype=float)
    lambdas = lambdas[(lambdas>=0.0) & (lambdas<1.0)]
    if lambdas.size == 0: lambdas = np.array([0.5], float)

    with np.errstate(divide='ignore', invalid='ignore'):
        pi0_grid = np.array([np.mean(p > lam)/max(1e-12, 1.0-lam) for lam in lambdas], float)
    pi0_grid = np.clip(pi0_grid, pi0_floor, pi0_ceil)
    pi0_min = float(np.min(pi0_grid))

    rng = np.random.default_rng(seed)
    mse_grid = np.zeros_like(lambdas, float)
    B = max(1, int(B))
    for _ in range(B):
        pb = rng.choice(p, size=m, replace=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            pi0_b = np.array([np.mean(pb > lam)/max(1e-12, 1.0-lam) for lam in lambdas], float)
        pi0_b = np.clip(pi0_b, pi0_floor, pi0_ceil)
        mse_grid += (pi0_b - pi0_min)**2
    mse_grid /= float(B)

    j = int(np.argmin(mse_grid))
    return float(pi0_grid[j]), float(lambdas[j]), pi0_grid, mse_grid


def qvalues_storey(p_values, pi0):
    """
    Storey q-values with monotone adjustment.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    # p는 [0,1]로 클립(극단값/음수 방지)
    p = np.clip(p, 0.0, 1.0)

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]

    # ❗ dtype로 지정 (세 번째 인자 아님)
    ranks = np.arange(1, m + 1, dtype=np.float64)

    q_sorted = pi0 * m * p_sorted / ranks
    # 단조 감소로 조정 (뒤에서 앞으로 누적 최소)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def qvalues_bh(p_values):
    """
    Benjamini–Hochberg q-values (monotone).
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    p = np.clip(p, 0.0, 1.0)

    order = np.argsort(p)
    p_sorted = p[order]

    ranks = np.arange(1, m + 1, dtype=np.float64)

    q_sorted = (m * p_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def _sanitize_pvals(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return np.clip(p, 0.0, 1.0) if p.size else p
