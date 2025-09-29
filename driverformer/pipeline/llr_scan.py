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

def make_per_bin(all_pred_csv: str, out_csv: str) -> str:
    """
    all_prediction.csv → per-bin 집계:
      (chrom,start,end)별 lam_pred 평균, obs_count 첫 값(원본 유지)
    """
    df = pd.read_csv(all_pred_csv)
    gb = (df.groupby(["chrom", "start", "end"], sort=False)
            .agg(lam_pred=("lam_pred", "mean"),
                 obs_count=("obs_count", "first"))
            .reset_index())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    gb.to_csv(out_csv, index=False)
    return out_csv


def _build_chunks(lam, obs, st, en, win, ov):
    L = len(lam)
    out, cid, idx, cur = [], 0, 0, st[0]
    while cur <= en[-1]:
        stop = cur + win - 1
        sel = []
        for i in range(idx, L):
            if en[i] < cur:
                idx = i + 1
                continue
            if st[i] > stop:
                break
            sel.append(i)
        if sel:
            sl = np.array(sel, int)
            out.append({"chunk_id": cid,
                        "lam": lam[sl], "obs": obs[sl],
                        "st":  st[sl],  "en":  en[sl]})
            cid += 1
        cur += (win - ov)
    return out


def _psums(lam, obs, st, en):
    kb = (en - st + 1) / 1_000.0
    psC = np.concatenate([[0], np.cumsum(obs,          dtype=float)])
    psE = np.concatenate([[0], np.cumsum(lam * kb,     dtype=float)])
    return psC, psE


def _llr(psC, psE, i, j):
    y  = psC[j+1] - psC[i]
    ex = psE[j+1] - psE[i]
    if y <= ex or ex <= 1e-12:
        return None
    a = (y + 1) / (ex + 1)
    return y * math.log(a) + (1 - a) * ex


def _scan_chunk(ck, min_bp, max_bp):
    lam, obs, st, en = (ck[k] for k in ("lam", "obs", "st", "en"))
    psC, psE = _psums(lam, obs, st, en)

    out, i_ptr, L = [], 0, len(lam)
    for j in range(L):
        while i_ptr <= j and (en[j] - st[i_ptr]) > max_bp:
            i_ptr += 1
        for i in range(i_ptr, j + 1):
            seg_len = int(en[j] - st[i] + 1)
            if seg_len < min_bp:
                continue
            v = _llr(psC, psE, i, j)
            if v is not None:
                out.append({"start_bp": int(st[i]),
                            "end_bp":   int(en[j]),
                            "len_bp":   seg_len,
                            "LLR_raw":  float(v)})
    return out


def _presmooth_nhpp_numpy(lam, obs, st, en, W_bins: int):
    """
    NHPP-일관 사전 스무딩(박스 W):
      ỹ  = box(W)*y,
      μ̃  = box(W)*(λ·Δt),
      Δt̃ = box(W)*Δt,
      λ̃  = μ̃/Δt̃
    반환: (lam_tilde, obs_tilde, st, en)
    """
    W = int(max(1, W_bins))
    if W == 1:
        return lam, obs, st, en
    k  = np.ones(W, dtype=np.float64)
    kb = (en - st + 1).astype(np.float64) / 1_000.0
    mu = lam.astype(np.float64) * kb

    y_s  = np.convolve(obs.astype(np.float64), k, mode="same")
    mu_s = np.convolve(mu,                     k, mode="same")
    dt_s = np.convolve(kb,                     k, mode="same")
    lam_s = mu_s / np.maximum(dt_s, 1e-12)

    return lam_s.astype(np.float32), y_s.astype(np.float32), st, en
