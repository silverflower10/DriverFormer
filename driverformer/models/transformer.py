#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-ready Transformer encoder (drop-in)

- 기존 코드의 RoPE/SDPA/NaN-guard/패딩처리를 그대로 유지
- 반환 인터페이스 확장:
  * (hidden, pooled) 또는 (hidden, pooled, attn_list)
- 풀링 옵션: 'none' | 'mean' | 'cls'
- GlobalTransformerEncoder 는 FoundationEncoder 상속(하위호환용)
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

    def forward(self, q, k):
        """
        q, k : (B, h, L, d_head)  – input & output 모두 동일 shape
        """
        if self._has_xpos:
            # 0.5.x 이상 & XPOS 사용
            return self.rot.rotate_queries_and_keys(q, k)

        if self._has_split:
            # 0.4.x 이하 (쿼리·키 각각 회전)
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
        self.w12  = nn.Linear(d_model, hidden * 2, bias=False)
        self.proj = nn.Linear(hidden,  d_model,  bias=False)
        self.drop = nn.Dropout(dropout)

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
        new_shape = x.size()[:-1] + (self.nhead, self.head_dim)
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
        if key_padding_mask is not None:
            # --- key_padding_mask → (B,T,1)로 정규화 후 masked_fill ---
            pad = key_padding_mask
            if pad.dim() == 3 and pad.size(1) == 1:      # (B,1,T) → (B,T)
                pad = pad.squeeze(1)
            elif pad.dim() == 2:
                pass
            else:
                raise RuntimeError(f"Unexpected key_padding_mask shape: {pad.shape}")

            # 드문 케이스: (T,B)로 들어왔을 때 교정
            if pad.size(0) != attn_out.size(0) and pad.size(1) == attn_out.size(0):
                pad = pad.transpose(0, 1)  # (T,B)→(B,T)

            if pad.size(1) != attn_out.size(1):
                raise RuntimeError(f"Key padding length {pad.size(1)} != seq len {attn_out.size(1)}")

            mask_bt1 = pad.unsqueeze(-1)                      # (B,T,1)
            attn_out = attn_out.masked_fill(mask_bt1, 0.0)    # (B,T,D)

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


# ================================
# Foundation Encoder (새 인터페이스)
# ================================
class FoundationEncoder(nn.Module):
    """
    - pool: 'none' | 'mean' | 'cls'
    - forward:
        return_attn=False → (hidden, pooled)
        return_attn=True  → (hidden, pooled, attn_list[::-1])
          * attn_list의 각 원소는 (B,T)  ← 기존 코드의 mean(1)을 유지해 호환성 보장
    - key_padding_mask: (B,T) (True==PAD)
    """
    def __init__(self, d_model=768, nhead=6, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
                 pool: str = 'none'):
        super().__init__()
        self.pool = pool
        self.layers = nn.ModuleList([
            AttnEncoderLayer(d_model, nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout, batch_first=True,
                             max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, src, key_padding_mask=None, *,
                return_hidden: bool = True,
                return_pooled: Optional[str] = None,
                return_attn: bool = False):
        """
        src: (B,L,D)
        """
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != src.shape[:2]:
                raise ValueError(f"key_padding_mask must be (B,T)={src.shape[:2]}, got {tuple(key_padding_mask.shape)}")
            key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=src.device)

        out = src
        attn_buf = []
        for layer in self.layers:
            out = layer(out, need_weights=return_attn, key_padding_mask=key_padding_mask)
            if return_attn and layer.attn_weight is not None:
                # 기존 GlobalTransformerEncoder가 하던 대로 (B,T,T) -> mean(1) → (B,T)
                attn_buf.append(layer.attn_weight.mean(1))
                layer.attn_weight = None

        out = self.out_norm(out)

        # ---- pooling ----
        pooled = None
        pool_mode = (return_pooled or self.pool or 'none')
        if pool_mode == 'mean':
            if key_padding_mask is None:
                pooled = out.mean(dim=1)
            else:
                valid = (~key_padding_mask).to(out.dtype).unsqueeze(-1)  # (B,L,1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                pooled = (out * valid).sum(dim=1) / denom
        elif pool_mode == 'cls':
            pooled = out[:, 0, :]

        if return_attn:
            return (out if return_hidden else None,
                    pooled,
                    attn_buf[::-1])  # 마지막 레이어를 [0]으로
        return (out if return_hidden else None, pooled)


# ============================================
# CondFiLM (per-layer FiLM for organ-aware tuning)
# ============================================
class CondFiLM(nn.Module):
    def __init__(self, num_organs:int, d_model:int, hidden:int=128):
        super().__init__()
        self.emb = nn.Embedding(num_organs, hidden)
        self.proj_g = nn.Linear(hidden, d_model)  # gamma
        self.proj_b = nn.Linear(hidden, d_model)  # beta
        # 보수적 초기화(처음엔 영향 거의 없도록)
        nn.init.zeros_(self.proj_g.weight); nn.init.zeros_(self.proj_g.bias)
        nn.init.zeros_(self.proj_b.weight); nn.init.zeros_(self.proj_b.bias)

    def forward(self, x: torch.Tensor, organ_ids: torch.Tensor):
        # x: (B,T,D), organ_ids: (B,)
        h = self.emb(organ_ids)               # (B,H)
        g = self.proj_g(h).unsqueeze(1)       # (B,1,D)
        b = self.proj_b(h).unsqueeze(1)       # (B,1,D)
        return x * (1.0 + g) + b              # FiLM: scale/shift


# ============================================
# (하위호환) GlobalTransformerEncoder: 상속 방식
#  - 기존 호출부가 기대하는 반환형 유지:
#       return_attn=False → out
#       return_attn=True  → (out, attn_list)
#  - 변경점: per-layer CondFiLM 적용 (organ_ids 필요)
# ============================================
class GlobalTransformerEncoder(FoundationEncoder):
    """
    key_padding_mask: (B, T) bool, True = PAD
    """
    def __init__(self, d_model=768, nhead=6, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
                 num_organs: Optional[int] = None):
        super().__init__(d_model=d_model, nhead=nhead, num_layers=num_layers,
                         dim_feedforward=dim_feedforward, dropout=dropout,
                         max_seq_len=max_seq_len, pool='none')
        # per-layer FiLM (옵션): num_organs가 주어지면 활성화
        self.cond = CondFiLM(num_organs, d_model, hidden=128) if num_organs is not None else None

    def forward(self, src, key_padding_mask=None, *, organ_ids: Optional[torch.Tensor] = None, return_attn: bool = False):
        """
        src: (B,L,D)
        organ_ids: (B,) long or None
        """
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != src.shape[:2]:
                raise ValueError(f"key_padding_mask must be (B,T)={src.shape[:2]}, got {tuple(key_padding_mask.shape)}")
            key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=src.device)

        out = src
        attn_buf = []
        for layer in self.layers:
            out = layer(out, need_weights=return_attn, key_padding_mask=key_padding_mask)
            # ★ 변경점: 레이어마다 CondFiLM 적용
            if (self.cond is not None) and (organ_ids is not None):
                out = self.cond(out, organ_ids)
            if return_attn and layer.attn_weight is not None:
                attn_buf.append(layer.attn_weight.mean(1))
                layer.attn_weight = None

        out = self.out_norm(out)

        if return_attn:
            return out, attn_buf[::-1]
        return out
