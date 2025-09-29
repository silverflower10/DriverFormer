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

def _set_seed(s=0):
    random.seed(s); np.random.seed(s)


def _move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_cpu(v) for v in obj)
    return obj


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def efficient_load_ckpt(path: str):
    """
    • PyTorch ≥2.6 : torch.load 기본값 weights_only=True  → 반드시 False 로 덮어써야 함
    • mmap=True     옵션은 일부 버전에서 weights_only 인자를 무시하는 버그가 있어 제거
    • 하위 버전(≤2.5)은 weights_only 인자를 그냥 무시하고 동작하므로 호환 유지
    """
    kw = dict(map_location="cpu", weights_only=False)   # 핵심!

    try:
        # 우선 일반 로드
        return torch.load(path, **kw)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        # 문제가 있으면 마지막으로 안전하게 재시도
        del kw["weights_only"]          # 구버전 호환
        return torch.load(path, **kw)


def efficient_save_ckpt(path, **named_objs):
    torch.save({k: _move_to_cpu(v) for k, v in named_objs.items()},
               path, pickle_protocol=4, _use_new_zipfile_serialization=False)
    gc.collect()


def check_pretrained_model_exists(path): return path and os.path.isfile(path)


def build_chrom_id_map(_):
    return {c: i for i, c in enumerate(CHROM_LIST_24)}
