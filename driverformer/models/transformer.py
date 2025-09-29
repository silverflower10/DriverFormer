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

class RoPE(nn.Module):
    def __init__(self, rot_dim: int, max_len: int = 32768):
        super().__init__()
        from rotary_embedding_torch import RotaryEmbedding
        self.rot = RotaryEmbedding(dim = rot_dim,
                                   cache_max_seq_len = max_len)

        # 버전 플래그
        self._has_xpos  = getattr(self.rot, "use_xpos", False)
        self._has_split = hasattr(self.rot, "rotate_queries_or_keys")

    # ------------------------------------------------------------------ #
    def forward(self, q, k):
        """
        q, k : (B, h, L, d_head)  – input & output 모두 동일 shape
        """
        if self._has_xpos:
            # 0.5.x 이상 & XPOS 사용
            return self.rot.rotate_queries_and_keys(q, k)

        if self._has_split:
            # 0.4.x 이하  (쿼리·키 각각 회전)
            q_rot = self.rot.rotate_queries_or_keys(q)
            k_rot = self.rot.rotate_queries_or_keys(k)
            return q_rot, k_rot

        # 최후: sin/cos 테이블 직접 적용
        from rotary_embedding_torch import apply_rotary_pos_emb
        sin, cos = self.rot.sin, self.rot.cos     # 버퍼는 replica 로 복제됨
        return (apply_rotary_pos_emb(q, sin, cos),
                apply_rotary_pos_emb(k, sin, cos))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w12   = nn.Linear(d_model, hidden * 2, bias=False)
        self.proj  = nn.Linear(hidden,  d_model,  bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.proj(F.silu(gate) * x))


class AttnEncoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048,
                 dropout=0.1, max_seq_len=8192, batch_first=True):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj_o= nn.Linear(d_model, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.ffn   = SwiGLUFFN(d_model, dim_feedforward, dropout)
        self.res_scale = 0.3
        self.head_dim  = d_model // nhead
        self.nhead = nhead
        self.rope = RoPE(rot_dim=min(self.head_dim, 64), max_len=max_seq_len)
        self.attn_weight = None

    def _qkv_proj(self, x):
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        new_shape = q.size()[:-1] + (self.nhead, self.head_dim)
        return (q.view(new_shape).transpose(1,2),
                k.view(new_shape).transpose(1,2),
                v.view(new_shape).transpose(1,2))

    def forward(self, src, *, need_weights=False, key_padding_mask=None):
        """
        src: (B, L, D)
        key_padding_mask: (B, L) bool, True = PAD(가려야 함)
        """
        h = self.ln1(src)
        q, k, v = self._qkv_proj(h)
        q, k = self.rope(q, k)

        attn_mask = None
        if key_padding_mask is not None:
            # (B,T)→(B,1,1,T)로 브로드캐스트 가능하게
            if key_padding_mask.dim() != 2 or key_padding_mask.shape[1] != src.shape[1]:
                raise ValueError(f"key_padding_mask must be (B,T)={src.shape[:2]}, got {tuple(key_padding_mask.shape)}")
            attn_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool, device=src.device)

        attn_out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False
        )
        attn_out = attn_out.transpose(1,2).contiguous().view_as(src)
        attn_out = self.proj_o(attn_out)
        src = src + self.res_scale * attn_out

        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,L,L)
            if key_padding_mask is not None:
                # (B,L) -> (B,1,1,L)
                mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool, device=scores.device)
                neg_inf = torch.finfo(scores.dtype).min  # dtype-safe
                scores = scores.masked_fill(mask, neg_inf)
            wei = scores.softmax(-1)
            self.attn_weight = wei.detach()
        else:
            self.attn_weight = None

        h2  = self.ln2(src)
        src = src + self.res_scale * self.ffn(h2)
        return src


class GlobalTransformerEncoder(nn.Module):
    """
    key_padding_mask: (B, T) bool, True = PAD
    """
    def __init__(self, d_model=768, nhead=6, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.pos_encoder = nn.Identity()
        self.layers = nn.ModuleList([
            AttnEncoderLayer(d_model, nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout, batch_first=True,
                             max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

    def forward(self, src, key_padding_mask=None, *, return_attn=False):
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != src.shape[:2]:
                raise ValueError(f"key_padding_mask must be (B,T)={src.shape[:2]}, got {tuple(key_padding_mask.shape)}")
            key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=src.device)

        out = self.pos_encoder(src)
        attn_buf = []
        for layer in self.layers:
            out = layer(out, need_weights=return_attn, key_padding_mask=key_padding_mask)
            if return_attn and layer.attn_weight is not None:
                attn_buf.append(layer.attn_weight.mean(1))
        if return_attn:
            return out, attn_buf[::-1]
        return out
