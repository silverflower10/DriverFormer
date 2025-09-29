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

import numpy as np, torch
from ..utils.stats import compute_mad, compute_iqr, huber_weight

def eval_objective_weighted(model_c, loader, device, huber_factor=3.0, use_mad=False):
    """
    bin-level NLL → (세그 길이로 나눠)×30 → Huber 세그가중 평균 (1-pass)
    - 세그별 누적: sum_log = Σ y·logλ, sum_ldt = Σ λ·Δt, len_kb = Σ Δt
    - residual r_seg = mean(y - λ) 로 허버 가중 계산
    """
    for m in model_c.values():
        m.eval()

    res   = {}  # seg_id -> r_seg
    stats = {}  # seg_id -> {"sum_log":..., "sum_ldt":..., "len_kb":...}
    eps = 1e-9

    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = b["y_array"].to(device)
        len_b  = b["length_array"].to(device)            # ★
        cid_b  = b["chrom_id"].to(device)

        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

        key_pad  = (len_b <= 0)                          # ★
        lam      = model_c["nhpp_head"](
            model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        ).clamp(1e-9, 1e4)

        # 세그 단위로 유효 길이 L만 집계
        for i, seg in enumerate(b["raw_segments"]):
            L   = seg["cls_array"].shape[0]
            sid = seg["global_idx"]

            y_i   = y_b[i, :L]
            dt_i  = len_b[i, :L]
            lam_i = lam[i, :L]
            mu_i  = lam_i * dt_i 

            # residual (허버 가중 계산용)
            res[sid] = float((y_i - mu_i).mean().item())

            # 통계 누적
            if sid not in stats:
                stats[sid] = {"sum_log": 0.0, "sum_ldt": 0.0, "len_kb": 0.0}
            s = stats[sid]
            s["sum_log"] += float((y_i * lam_i.log()).sum().item())
            s["sum_ldt"] += float((lam_i * dt_i).sum().item())
            s["len_kb"]  += float(dt_i.sum().item())

    if not stats:
        return 0.0

    # Huber 세그 가중
    rs = np.array(list(res.values()), dtype=np.float64)
    if rs.size == 0:
        w = {sid: 1.0 for sid in stats.keys()}
    else:
        scale = compute_mad(rs) if use_mad else compute_iqr(rs)
        delta = max(huber_factor * scale, 1e-9)
        w = {sid: huber_weight(res[sid], delta) for sid in stats.keys()}

    # 가중 평균 (per-seg/kb × 30)
    tot_num, tot_den = 0.0, 0.0
    for sid, s in stats.items():
        per_seg = (-(s["sum_log"] - s["sum_ldt"])) / max(s["len_kb"], eps) * 30.0
        wi = float(w.get(sid, 1.0))
        tot_num += wi * per_seg
        tot_den += wi

    return tot_num / max(tot_den, eps)


def _perseg_bin_nll30k(lam, y, len_kb):
    """bin-level NLL을 세그 단위로 (per-kb×30) 정규화한 벡터(B,) 반환."""
    lam_safe = lam.clamp(1e-9, 1e4)
    sum_log  = (y * torch.log(lam_safe)).sum(dim=1)
    integ    = (lam_safe * len_kb).sum(dim=1)
    neg_ll   = -(sum_log - integ)                     # (B,)
    seg_len  = (len_kb.sum(dim=1) + 1e-9)             # (B,)
    return (neg_ll / seg_len) * 30.0                  # (B,)


def _to_huber_weights_from_res(res_dict, huber_factor, use_mad):
    rs = np.array(list(res_dict.values()), dtype=np.float64)
    scale = compute_mad(rs) if use_mad else compute_iqr(rs)
    delta = max(huber_factor * scale, 1e-9)
    w = {sid: huber_weight(r, delta) for sid, r in res_dict.items()}
    return w, float(delta)
