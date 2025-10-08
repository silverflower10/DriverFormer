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
        B, T, D = src.shape
        h = self.ln1(src)

        q, k, v = self._qkv_proj(h)                           # (B,h,T,dh)
        q, k = self.rope(q, k)

        # ----- Key 마스크(키를 가리는 용도) & Query 마스크(쿼리 위치 자체를 무효화) -----
        attn_mask_keys = None
        qpad_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape[1] != T:
                raise ValueError(f"key_padding_mask must be (B,T)={src.shape[:2]}, got {tuple(key_padding_mask.shape)}")
            kpad = key_padding_mask.to(dtype=torch.bool, device=src.device)      # (B,T)
            attn_mask_keys = kpad[:, None, None, :]                               # (B,1,1,T)  -> 키만 가림
            qpad_mask = kpad[:, None, :, None]                                    # (B,1,T,1)  -> 쿼리 행 표시

        # ----- SDPA: 연산은 안정적으로 하고 dtype은 원래대로 -----
        q_sdpa = q if q.dtype == torch.float32 else q.float()
        k_sdpa = k if k.dtype == torch.float32 else k.float()
        v_sdpa = v if v.dtype == torch.float32 else v.float()

        attn_out = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask_keys,                    # True == mask
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False
        ).to(src.dtype)

        attn_out = attn_out.transpose(1,2).contiguous().view_as(src)
        # 패딩된 쿼리 위치의 출력은 0으로 강제 (잔여연결에 쓰레기 안 섞이게)
        if qpad_mask is not None:
            attn_out = attn_out.masked_fill(qpad_mask.squeeze(1).unsqueeze(-1), 0.0)

        attn_out = self.proj_o(attn_out)
        src = src + self.res_scale * attn_out

        if need_weights:
            # --- 가중치 저장: float32로 softmax, 완전-마스크 행(모두 -inf) 처리 ---
            # (주의) softmax([-inf,...])는 NaN → 0으로 대체
            scale = self.head_dim ** -0.5
            scores = (q.float() @ k.float().transpose(-2, -1)) * scale          # (B,h,T,T)

            if attn_mask_keys is not None:
                scores = scores.masked_fill(attn_mask_keys, float("-inf"))

            wei = F.softmax(scores, dim=-1)                                      # float32
            # 모든 키가 가려진 쿼리 행 → softmax가 NaN이 되므로 0으로
            bad_rows = ~torch.isfinite(wei).all(dim=-1, keepdim=True)            # (B,h,T,1)
            wei = torch.where(bad_rows, torch.zeros_like(wei), wei)
            # 쿼리 자체가 패딩인 행도 0으로
            if qpad_mask is not None:
                wei = wei.masked_fill(qpad_mask, 0.0)

            wei = torch.nan_to_num(wei, nan=0.0, posinf=0.0, neginf=0.0).to(scores.dtype)
            # 헤드 평균 후 저장
            self.attn_weight = wei.mean(dim=1).detach()                           # (B,T,T)
        else:
            self.attn_weight = None

        h2  = self.ln2(src)
        src = src + self.res_scale * self.ffn(h2)

        # 디버그(선택): 어텐션 출력이 유한한지 최종 체크
        if DEBUG_NAN and not torch.isfinite(src).all():
            raise RuntimeError("[AttnEncoderLayer] non-finite value detected")

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
