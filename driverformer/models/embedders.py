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

class FeatureEmbedder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=768):
        super().__init__(); self.proj = nn.Linear(feature_dim, hidden_dim)
    def forward(self, x): return self.proj(x)


class FeatClsFusion(nn.Module):
    """
    CLS + Feature 를 같은 hidden 차원(d_model)에서
    LayerNorm 뒤 element-wise sum.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.ln_cls  = nn.LayerNorm(hidden_dim)
        self.ln_feat = nn.LayerNorm(hidden_dim)
    def forward(self, cls_emb, feat_emb):
        # 입력 두 텐서 shape: (B, L, hidden_dim)
        return self.ln_cls(cls_emb) + self.ln_feat(feat_emb)


class ChromosomeEmbedder(nn.Module):
    def __init__(self, n, d=768):
        super().__init__()
        self.emb = nn.Embedding(n, d)
        # 하드코딩 스케일(여기만 바꾸면 전체 반영됨)
        self.register_buffer("scale", torch.tensor(0.2, dtype=torch.float32))

    def forward(self, ids):
        return self.emb(ids) * self.scale
