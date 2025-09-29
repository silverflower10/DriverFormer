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

import os, torch
from pathlib import Path
from ..utils.io import unwrap

def save_last_layer_attention(gt, epoch, step, out_dir):
    """
    gt: unwrap(model_components["global_transformer"])
    """
    a = getattr(gt, "last_attn_cpu", None)      # forward 에서 복사해 둔 텐서
    if a is None:
        print(f"[WARN] epoch {epoch}: no attention captured", flush=True)
        return

    os.makedirs(os.path.join(out_dir, "attn"), exist_ok=True)
    torch.save(
        {"epoch": epoch, "step_global": step, "attn": a.half()},   # 이미 CPU tensor
        os.path.join(out_dir, "attn", f"attn_epoch_{epoch:03d}.pt")
    )
    delattr(gt, "last_attn_cpu")          # 메모리 해제
    torch.cuda.empty_cache()              # GPU 캐시도 정리
    print(f"[INFO] last-layer attention saved (epoch {epoch})", flush=True)


def dump_full_attention(model_c, loader, device, out_path):
    gt = unwrap(model_c["global_transformer"])
    for m in model_c.values():  # ★ 권장: eval 모드
        m.eval()
    layer_buf = [[] for _ in range(len(gt.layers))]

    for batch in loader:
        # (선택) 이전 배치 잔여값 초기화
        for ly in gt.layers:
            ly.attn_weight = None

        cls_b  = batch["cls_array"].to(device)
        feat_b = batch["feat_array"].to(device)
        len_b  = batch["length_array"].to(device)
        cid_b  = batch["chrom_id"].to(device)
        key_pad  = (len_b <= 0)

        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

        
        _ = model_c["global_transformer"](fused + chr_emb,
                                          key_padding_mask=key_pad,
                                          return_attn=True)

        for i, ly in enumerate(gt.layers):
            if ly.attn_weight is not None:
                layer_buf[i].append(ly.attn_weight.cpu().half())
            ly.attn_weight = None

    stacked = [torch.cat(buf, 0) if buf else None for buf in layer_buf]
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"attn": stacked}, out_path)
    torch.cuda.empty_cache()
    print(f"[INFO] full-stack attention saved → {out_path}", flush=True)
