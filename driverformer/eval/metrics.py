#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025    ← NEW  (attention-save policy + NaN guards)

@author: silverflo

Changes in this revision
------------------------
* 학습(epoch) 중 : Val-loss 개선 시 **마지막 레이어 head-mean 1장** 저장
* 학습 완료(final): 베스트 모델로 **전체 레이어·헤드** 스택 저장
* NaN 방지: len=0 세그먼트/마스크 보정, log/분모/가중치 클램프, nan_to_num 적용
"""

# --------------------------------------------------------------------------- #
# Imports & setup                                                             #
# --------------------------------------------------------------------------- #
import os, sys, math, random, pickle, argparse, gc, itertools, warnings
from functools import partial
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from rotary_embedding_torch import RotaryEmbedding

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import fdrcorrection_twostage
from tqdm import tqdm
from scipy.special import logsumexp

# ---- project utils ----
from ..utils.stats import compute_mad, compute_iqr, huber_weight
# (필요시) from ..utils.io import unwrap  # 다른 곳에서 사용된다면 유지

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
DEBUG_NAN = True  # NaN 디버깅

# --------------------------------------------------------------------------- #
# NaN/Inf 안전 가드                                                           #
# --------------------------------------------------------------------------- #
_EPS = 1e-8

def _assert_finite(name: str, x: torch.Tensor):
    """입력 텐서의 유한성 검사. NaN/Inf가 있으면 즉시 예외."""
    if not torch.is_tensor(x):
        return
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:8].tolist()
        raise RuntimeError(f"[NaN/Inf in {name}] examples: {bad}")

def _nan_to_num_(x: torch.Tensor, nan=0.0, posinf=1e6, neginf=0.0):
    """in-place NaN/Inf 정리."""
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=x)

def _safe_log(x: torch.Tensor) -> torch.Tensor:
    """log 연산 안전화: log(clamp(x, min=_EPS)) + NaN/Inf 정리."""
    y = torch.log(x.clamp_min(_EPS))
    return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def _fix_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    key padding mask 보정: (B,T)에서 한 row가 전부 pad(True)이면
    첫 토큰을 keep(False)로 변경해 all-pad → NaN softmax를 방지.
    """
    mask = mask.bool()
    if mask.ndim != 2:
        return mask
    all_pad = mask.all(dim=1)  # (B,)
    if all_pad.any():
        idx = torch.nonzero(all_pad, as_tuple=False).squeeze(1)
        mask[idx, 0] = False  # 최소 1토큰은 유효하게
    return mask

# --------------------------------------------------------------------------- #
# 평가 목적함수 (세그 가중 Huber, per-kb×30)                                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_objective_weighted(model_c, loader, device,
                            huber_factor: float = 3.0,
                            use_mad: bool = False) -> float:
    """
    bin-level NLL → (세그 길이로 나눠)×30 → Huber 세그가중 평균 (1-pass)
    - 세그별 누적: sum_log = Σ y·logλ, sum_ldt = Σ λ·Δt, len_kb = Σ Δt
    - residual r_seg = mean(y - λ·Δt)  (※ μ=λ·Δt)
    - 모든 log/분모/가중치에 NaN 방지 가드 포함
    """
    for m in model_c.values():
        if isinstance(m, nn.Module):
            m.eval()

    res   = {}  # seg_id -> r_seg
    stats = {}  # seg_id -> {"sum_log":..., "sum_ldt":..., "len_kb":...}
    eps = _EPS

    for b in loader:
        cls_b  = b["cls_array"].to(device, non_blocking=True)
        feat_b = b["feat_array"].to(device, non_blocking=True)
        y_b    = b["y_array"].to(device, non_blocking=True)
        len_b  = b["length_array"].to(device, non_blocking=True)   # Δt
        cid_b  = b["chrom_id"].to(device, non_blocking=True)

        if DEBUG_NAN:
            _assert_finite("cls_array", cls_b)
            _assert_finite("feat_array", feat_b)
            _assert_finite("y_array", y_b)
            _assert_finite("length_array", len_b)

        # padding mask 및 보정
        key_pad  = _fix_key_padding_mask(len_b <= 0)

        # forward → λ (rate)
        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

        tr_out   = model_c["global_transformer"](fused + chr_emb,
                                                 key_padding_mask=key_pad)
        lam_raw  = model_c["nhpp_head"](tr_out)

        # λ는 항상 양수: softplus + EPS, 상한은 과도한 exp 폭주 방지용
        lam = F.softplus(lam_raw) + eps            # (B,T)
        lam = lam.clamp_max(1e6)
        _nan_to_num_(lam)

        # 세그 단위 집계
        for i, seg in enumerate(b["raw_segments"]):
            L   = int(seg["cls_array"].shape[0])
            if L <= 0:
                continue
            sid = seg["global_idx"]

            y_i   = y_b[i, :L]
            dt_i  = len_b[i, :L]
            lam_i = lam[i, :L]

            # 유효 위치(Δt>0)만 사용
            valid = (dt_i > 0)
            if not torch.any(valid):
                continue

            y_i   = y_i[valid]
            dt_i  = dt_i[valid].clamp_min(eps)
            lam_i = lam_i[valid]

            # μ = λ·Δt
            mu_i  = lam_i * dt_i
            _nan_to_num_(mu_i)

            # residual r_seg = mean(y - μ)
            r_seg = (y_i - mu_i).mean()
            r_val = float(torch.nan_to_num(r_seg, nan=0.0, posinf=0.0, neginf=0.0).item())
            res[sid] = r_val

            # 통계 누적 (log λ는 safe_log)
            if sid not in stats:
                stats[sid] = {"sum_log": 0.0, "sum_ldt": 0.0, "len_kb": 0.0}

            sum_log  = (y_i * _safe_log(lam_i)).sum()
            sum_ldt  = (lam_i * dt_i).sum()
            len_sum  = dt_i.sum()

            # 숫자화 + NaN 방지
            s = stats[sid]
            s["sum_log"] += float(torch.nan_to_num(sum_log, nan=0.0, posinf=0.0, neginf=0.0).item())
            s["sum_ldt"] += float(torch.nan_to_num(sum_ldt, nan=0.0, posinf=0.0, neginf=0.0).item())
            s["len_kb"]  += float(torch.nan_to_num(len_sum, nan=0.0, posinf=0.0, neginf=0.0).item())

    if not stats:
        return 0.0

    # Huber 세그 가중
    rs = np.asarray(list(res.values()), dtype=np.float64)
    if rs.size == 0:
        w = {sid: 1.0 for sid in stats.keys()}
        delta_val = 1.0
    else:
        scale = compute_mad(rs) if use_mad else compute_iqr(rs)
        # scale NaN/0 방지
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        delta_val = max(huber_factor * scale, 1e-6)
        w = {}
        for sid in stats.keys():
            w_i = float(huber_weight(res.get(sid, 0.0), delta_val))
            if not np.isfinite(w_i):
                w_i = 0.0
            # 가중치 폭주/음수 방지
            w[sid] = max(0.0, min(w_i, 1e3))

    # 가중 평균 (per-seg/kb × 30)
    tot_num, tot_den = 0.0, 0.0
    for sid, s in stats.items():
        denom = max(s["len_kb"], eps)
        per_seg = (-(s["sum_log"] - s["sum_ldt"])) / denom * 30.0
        if not np.isfinite(per_seg):
            continue
        wi = float(w.get(sid, 1.0))
        tot_num += wi * per_seg
        tot_den += wi

    return float(tot_num / max(tot_den, eps))

# --------------------------------------------------------------------------- #
# 보조: per-seg bin NLL (per-kb×30)                                           #
# --------------------------------------------------------------------------- #
def _perseg_bin_nll30k(lam: torch.Tensor, y: torch.Tensor, len_kb: torch.Tensor) -> torch.Tensor:
    """
    bin-level NLL을 세그 단위로 (per-kb×30) 정규화한 벡터(B,) 반환.
    모든 연산은 NaN 방지 가드를 포함.
    """
    eps = _EPS
    lam = F.softplus(lam) + eps          # λ>0
    lam = lam.clamp_max(1e6)
    _nan_to_num_(lam)

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0)
    len_kb = torch.nan_to_num(len_kb, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(eps)

    sum_log = (y * _safe_log(lam)).sum(dim=1)       # (B,)
    integ   = (lam * len_kb).sum(dim=1)             # (B,)
    neg_ll  = -(sum_log - integ)                    # (B,)
    seg_len = len_kb.sum(dim=1).clamp_min(eps)      # (B,)

    out = (neg_ll / seg_len) * 30.0
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

# --------------------------------------------------------------------------- #
# 보조: Huber 가중 생성                                                       #
# --------------------------------------------------------------------------- #
def _to_huber_weights_from_res(res_dict: dict, huber_factor: float, use_mad: bool):
    """
    residual dict -> {sid: weight}, delta
    - scale=0/NaN 방지
    - weight 범위/NaN 방지
    """
    rs = np.asarray(list(res_dict.values()), dtype=np.float64)
    if rs.size == 0:
        return {sid: 1.0 for sid in res_dict.keys()}, 1.0

    scale = compute_mad(rs) if use_mad else compute_iqr(rs)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    delta = max(huber_factor * scale, 1e-6)

    w = {}
    for sid, r in res_dict.items():
        w_i = float(huber_weight(float(r), delta))
        if not np.isfinite(w_i):
            w_i = 0.0
        w[sid] = max(0.0, min(w_i, 1e3))  # 상한으로 폭주 방지
    return w, float(delta)
