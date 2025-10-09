#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025    ← NEW  (attention-save policy)

@author: silverflo

Changes in this revision
------------------------
* 학습(epoch) 중 : Val-loss 개선 시 **마지막 레이어 head-mean 1장** 저장
* 학습 완료(final): 베스트 모델로 **전체 레이어·헤드** 스택 저장
* 그 외 로직(길이-가중 residual sampler, Huber/IRLS 등)은 이전 버전과 동일
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

from ..utils.io import unwrap

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

def _fix_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    key padding mask 보정: (B,T)에서 한 row가 전부 pad(True)이면
    첫 토큰을 keep(False)로 변경해 all-pad → NaN softmax를 방지.
    """
    mask = mask.bool()
    all_pad = mask.all(dim=1)  # (B,)
    if all_pad.any():
        idx = torch.nonzero(all_pad, as_tuple=False).squeeze(1)
        mask[idx, 0] = False  # 최소 1토큰은 유효하게
    return mask

def _sanitize_attn(a: torch.Tensor) -> torch.Tensor:
    """
    Attention 텐서를 안전화:
    - NaN/Inf → 0
    - 음수 0으로 클램프
    - 마지막 축(Tk)으로 합=1이 되도록 재정규화
    """
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a.clamp_min_(0.0)
    denom = a.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    a = a / denom
    return a

# --------------------------------------------------------------------------- #
# Attention 저장 유틸                                                         #
# --------------------------------------------------------------------------- #
def save_last_layer_attention(gt, epoch: int, step: int, out_dir: str):
    """
    마지막 레이어의 head-mean(또는 사전 준비된) attention을 저장.
    gt: unwrap(model_components["global_transformer"])
    """
    a = getattr(gt, "last_attn_cpu", None)  # forward 에서 CPU로 복사해 둔 텐서 기대
    if a is None:
        print(f"[WARN] epoch {epoch}: no attention captured", flush=True)
        return

    # 안전화(유한성/정규화)
    a = _sanitize_attn(a.detach().cpu())

    save_dir = os.path.join(out_dir, "attn")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"epoch": epoch, "step_global": step, "attn": a.half()},
        os.path.join(save_dir, f"attn_epoch_{epoch:03d}.pt"),
    )

    # 메모리 정리
    try:
        delattr(gt, "last_attn_cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()
    print(f"[INFO] last-layer attention saved (epoch {epoch})", flush=True)

@torch.no_grad()
def dump_full_attention(model_c: dict, loader: DataLoader, device: torch.device, out_path: str):
    """
    학습 완료 후: 전체 레이어·헤드 attention 스택 저장.
    model_c: {'feature_embedder', 'feat_cls_fusion', 'chrom_embedder', 'global_transformer', ...}
    """
    gt = unwrap(model_c["global_transformer"])

    # eval 모드 권장
    for m in model_c.values():
        if isinstance(m, nn.Module):
            m.eval()

    layer_buf = [[] for _ in range(len(gt.layers))]

    for batch in loader:
        # 이전 배치 잔여값 초기화
        for ly in gt.layers:
            if hasattr(ly, "attn_weight"):
                ly.attn_weight = None

        cls_b  = batch["cls_array"].to(device, non_blocking=True)
        feat_b = batch["feat_array"].to(device, non_blocking=True)
        len_b  = batch["length_array"].to(device, non_blocking=True)
        cid_b  = batch["chrom_id"].to(device, non_blocking=True)

        # 입력 유한성 검사
        if DEBUG_NAN:
            _assert_finite("cls_array", cls_b)
            _assert_finite("feat_array", feat_b)
            _assert_finite("length_array", len_b)

        # 길이 기반 패딩 마스크 구성 및 보정
        key_pad = (len_b <= 0)
        key_pad = _fix_key_padding_mask(key_pad)

        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

        # forward 시 각 레이어가 ly.attn_weight에 (B,H,T,T)로 저장되도록 설계되어 있어야 함
        _ = model_c["global_transformer"](
            fused + chr_emb,
            key_padding_mask=key_pad,
            return_attn=True
        )

        # 레이어별 attention 수집 + 안전화
        for i, ly in enumerate(gt.layers):
            attn_i = getattr(ly, "attn_weight", None)
            if attn_i is not None:
                attn_i = _sanitize_attn(attn_i.detach().cpu())
                layer_buf[i].append(attn_i.half())
            ly.attn_weight = None

    stacked = [torch.cat(buf, 0) if buf else None for buf in layer_buf]

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"attn": stacked}, out_path)
    torch.cuda.empty_cache()
    print(f"[INFO] full-stack attention saved → {out_path}", flush=True)
