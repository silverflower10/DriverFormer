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

import re

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


CANON_CHROMS = set(CHROM_LIST_24)


def _norm_chr(x: str) -> str:
    """'chr1', '1', 'X' 등을 'chr1','chrX'로 통일. 기타는 앞의 'chr' 유지."""
    s = str(x).strip()
    m = re.match(r'(?i)^(?:chr)?([0-9]+|x|y|m|mt)$', s)
    if not m:
        # 'chr11_gl000202_random' 같은 건 그대로 둔다
        return s if s.lower().startswith("chr") else ("chr" + s)
    body = m.group(1).upper()
    if body in ("M", "MT"): return "chrM"
    return "chr" + body


def _apply_chr_norm(df: pd.DataFrame, col: str = "chrom") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].map(_norm_chr)
    return df


def _chr_key(c: str):
    c = c.lower().removeprefix("chr")
    if c.isdigit():
        return (0, int(c))
    return {"x": (1, 0), "y": (2, 0), "m": (3, 0), "mt": (3, 0)}.get(c, (4, c))
