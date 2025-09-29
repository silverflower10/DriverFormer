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

def qq_plot(p, title, pdf):
    """
    Draw a -log10 QQ-plot with independent x/y axis limits.
    - xlim, ylim을 각각 데이터 최대값에 맞춰 따로 잡음
    - 대각선(y=x)은 두 축 범위의 공통 구간까지만 표시
    """
    p = np.asarray(p, dtype=float)

    # p=0 처리: 최소 양수 p 주변으로 지터 부여
    mask_zero = (p <= 0)
    if mask_zero.any():
        min_pos = p[~mask_zero].min() if (~mask_zero).any() else 1e-30
        p[mask_zero] = np.random.uniform(
            low=min_pos * 0.5,
            high=min_pos * 0.9,
            size=mask_zero.sum()
        )

    # 안정성: [1e-300, 1] 클리핑 후 정렬
    p = np.clip(p, 1e-300, 1.0)
    p.sort()
    m = len(p)
    theo = (np.arange(1, m + 1) - 0.5) / max(m, 1)
    x = -np.log10(theo)
    y = -np.log10(p)

    # 축 한쪽씩 범위 계산 (여유 20%)
    x_max = float(x.max()) if m else 1.0
    y_max = float(y.max()) if m else 1.0
    x_lim = (0.0, max(1.0, x_max * 1.2))
    y_lim = (0.0, max(1.0, y_max * 1.2))

    plt.figure(figsize=(5, 5))
    if m:
        plt.scatter(x, y, s=8, fc="white", ec="k", lw=.4)

    # 기준선 y=x는 공통 구간까지만
    diag_end = min(x_lim[1], y_lim[1])
    plt.plot([0, diag_end], [0, diag_end], 'r--', lw=1.2)

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel("Theoretical -log10 p")
    plt.ylabel("Observed -log10 p")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pdf, dpi=300)
    plt.close()
