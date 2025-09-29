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

class SegmentDataset(Dataset):
    def __init__(self, segments): self.segments = segments
    def __len__(self): return len(self.segments)
    def __getitem__(self, idx): return self.segments[idx]


def segment_cls_embeddings_fixed_lengths(
    cls_list, feature_dict, seg_len_list=(10, 50, 100),
    discard_leftover=False, overlap_factor=0.0):
    chrom_map = defaultdict(list)
    any_feat = next(iter(feature_dict.values()))
    feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else any_feat.shape[0]

    for chrom, w_idx, sbp, ebp, cls_vec, y_val in cls_list:
        feat_vec = feature_dict.get((chrom, w_idx),
                                    np.zeros(feature_dim, dtype=np.float32))
        chrom_map[chrom].append((w_idx, sbp, ebp, cls_vec, y_val, feat_vec))

    all_segments = []
    for seg_len in seg_len_list:
        for chrom in chrom_map:
            items = sorted(chrom_map[chrom], key=lambda x: x[0])
            n, i = len(items), 0
            step = max(1, int(seg_len * (1.0 - overlap_factor)))
            while i < n:
                end = i + seg_len
                if end > n:
                    if discard_leftover: break
                    end = n
                all_segments.append(_make_segment_dict(chrom, items[i:end]))
                i += step
    for gid, seg in enumerate(all_segments): seg["global_idx"] = gid
    return all_segments


def segment_collate_fn(batch, *, chrom_id_map=None, cutmix_p=0.2):
    if random.random() < cutmix_p and len(batch) >= 2:
        i, j = random.sample(range(len(batch)), 2)
        seg_i, seg_j = batch[i], batch[j]
        L = min(seg_i["cls_array"].shape[0], seg_j["cls_array"].shape[0])
        cut = random.randint(1, max(1, int(0.5 * L)))
        s = random.randint(0, L - cut)
        for key in ("cls_array", "feat_array", "y_array", "start_array", "end_array"):
            tmp = seg_i[key][s:s+cut].copy()
            seg_i[key][s:s+cut] = seg_j[key][s:s+cut]
            seg_j[key][s:s+cut] = tmp

    lens = [b["cls_array"].shape[0] for b in batch]; max_len = max(lens)

    def _pad(arr, val=0):
        pad = max_len - arr.shape[0]
        if pad > 0:
            if arr.ndim == 2:
                arr = np.pad(arr, ((0, pad), (0, 0)), constant_values=val)
            else:
                arr = np.pad(arr, (0, pad), constant_values=val)
        return arr

    cls_list, feat_list, y_list, s_list, e_list, cid_list = [], [], [], [], [], []
    for seg in batch:
        cls_list.append(_pad(seg["cls_array"]))
        feat_list.append(_pad(seg["feat_array"]))
        y_list.append(_pad(seg["y_array"]))
        s_list.append(_pad(seg["start_array"]))
        e_list.append(_pad(seg["end_array"]))
        cid_list.append(chrom_id_map.get(seg["chrom"], 0) if chrom_id_map else 0)

    cls_b  = torch.tensor(np.stack(cls_list), dtype=torch.float32)
    feat_b = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    y_b    = torch.tensor(np.stack(y_list),  dtype=torch.float32)
    s_b    = torch.tensor(np.stack(s_list),  dtype=torch.long)
    e_b    = torch.tensor(np.stack(e_list),  dtype=torch.long)
    len_b  = (e_b - s_b + 1).clamp_min(0).float() / 1000.0
    B, T = len_b.shape
    valid = torch.arange(T).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
    len_b = len_b * valid.to(len_b.dtype)  # 또는: len_b[~valid] = 0

    
    cid_b  = torch.tensor(cid_list, dtype=torch.long)

    return dict(cls_array=cls_b, feat_array=feat_b, y_array=y_b,
                start_array=s_b, end_array=e_b, length_array=len_b,
                chrom_id=cid_b, raw_segments=batch)


def _make_segment_dict(chrom, slice_):
    idx_array   = np.array([s[0] for s in slice_], dtype=int)
    start_array = np.array([s[1] for s in slice_], dtype=np.int64)
    end_array   = np.array([s[2] for s in slice_], dtype=np.int64)
    cls_array   = np.stack([s[3] for s in slice_], axis=0)
    y_array     = np.array([s[4] for s in slice_], dtype=np.float32)
    feat_array  = np.stack([s[5] for s in slice_], axis=0)
    return dict(chrom=chrom, idx_array=idx_array, start_array=start_array,
                end_array=end_array, cls_array=cls_array,
                feat_array=feat_array, y_array=y_array)
