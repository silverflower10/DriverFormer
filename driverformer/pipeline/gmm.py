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

def fit_gmm(sample, k, seed):
    logx = np.log1p(sample).reshape(-1, 1)
    g    = GaussianMixture(k, random_state=seed).fit(logx)
    w, mu = g.weights_, g.means_.ravel()
    sig   = np.sqrt([np.diag(c)[0] for c in g.covariances_])
    keep  = ~((mu < 0.05) & (sig < 0.05))  # 스파이크 제거
    if keep.sum() == 0: keep[:] = True
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w /= w.sum()
    return dict(w=w, mu=mu, sig=sig)


def fit_gmm_auto(sample,
                 k_min: int = 1,
                 k_max: int = 8,
                 seed: int = 0,
                 n_init: int = 3,
                 max_iter: int = 500):
    """
    BIC로 최적 k 선택. (log1p 변환 공간), 스파이크 성분 제거 후 재정규화.
    """
    x = np.asarray(sample, float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return {"w": np.array([1.0]), "mu": np.array([0.0]), "sig": np.array([0.5]),
                "k_init": 1, "k_final": 1}
    logx = np.log1p(x).reshape(-1, 1)
    best_gmm, best_bic = None, float("inf")
    k_max_eff = max(k_min, min(k_max, int(max(1, x.size // 10))))
    for k in range(max(1, k_min), max(1, k_max_eff)+1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=n_init, max_iter=max_iter).fit(logx)
        bic = g.bic(logx)
        if bic < best_bic:
            best_bic, best_gmm = bic, g
    w, mu = best_gmm.weights_.copy(), best_gmm.means_.ravel().copy()
    sig   = np.sqrt([np.diag(c)[0] for c in best_gmm.covariances_])
    keep  = ~((mu < 0.05) & (sig < 0.05))
    if keep.sum() == 0: keep[:] = True
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w = w / w.sum()
    return {"w": w, "mu": mu, "sig": sig,
            "k_init": best_gmm.n_components, "k_final": len(w)}


def mix_sf(x, g):
    """
    Mixture survival function: p = P[X >= x] on log1p-scale Gaussian mixture.
    x: 1D array-like of llr_norm
    g: dict(w, mu, sig) from fit_gmm
    """
    z = np.log1p(np.maximum(np.asarray(x, dtype=np.float64), 0.0))[:, None]  # (N,1)
    t = (z - g["mu"]) / g["sig"]                                             # (N,K)
    comp_log = np.log(g["w"])[None, :] + norm.logsf(t)                       # log(w_k * SF_k)
    return np.exp(logsumexp(comp_log, axis=1))                               # (N,)


def mix_neglog10p_from_gmm(x, g, min_p=1e-300):
    """
    Mixture SF p = P(X>=x) on log1p-space.
    반환: (-log10 p, p)
    """
    z = np.log1p(np.maximum(np.asarray(x, float), 0.0))[:, None]
    t = (z - g["mu"]) / g["sig"]
    comp_log = np.log(g["w"])[None, :] + norm.logsf(t)
    logp = logsumexp(comp_log, axis=1)
    logp = np.maximum(logp, math.log(min_p))
    neglog10p = -logp / math.log(10.0)
    return neglog10p, np.exp(logp)
