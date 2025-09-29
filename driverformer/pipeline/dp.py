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

def _pred_idx(lst, i, gap_bp):
    s = lst[i]["start_bp"]
    lo, hi, res = 0, i - 1, -1
    while lo <= hi:
        m = (lo + hi) // 2
        # 이전 구간의 end + gap < 현재 start 여야 호환됨 (gap=0이면 기존 '<' 유지)
        if lst[m]["end_bp"] + gap_bp < s:
            res, lo = m, m + 1
        else:
            hi = m - 1
    return res


def dp_select(iv_list, gap_bp=0):
    by_chr = defaultdict(list)
    for iv in iv_list:
        by_chr[iv["chrom"]].append(iv)

    chosen = []
    for ch, lst in by_chr.items():
        lst.sort(key=lambda d: d["end_bp"])
        n      = len(lst)
        pred   = [_pred_idx(lst, i, gap_bp) for i in range(n)]
        dp     = np.zeros(n + 1)
        keep   = np.zeros(n + 1, bool)

        for i in range(1, n + 1):
            cand = lst[i-1]["LLR_weighted"] + dp[pred[i-1] + 1]
            if cand > dp[i-1]:
                dp[i], keep[i] = cand, True
            else:
                dp[i] = dp[i-1]

        i = n
        while i > 0:
            if keep[i]:
                chosen.append(lst[i-1])
                i = pred[i-1] + 1
            else:
                i -= 1
    return chosen
