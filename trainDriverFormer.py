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
# Utilities                                                                   #
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


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def check_pretrained_model_exists(path): return path and os.path.isfile(path)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_chrom_id_map(_):
    return {c: i for i, c in enumerate(CHROM_LIST_24)}


def compute_mad(arr):
    med = np.median(arr); mad = np.median(np.abs(arr - med))
    return max(1.4826 * mad, 1e-9)


def compute_iqr(arr):
    q1, q3 = np.percentile(arr, [25, 75]); return max(q3 - q1, 1e-9)


def huber_weight(r, delta):
    a = abs(r); return 1.0 if a <= delta else delta / (a + 1e-9)

# --------------------------------------------------------------------------- #
# Label helpers (canonical-chrom only, anchor-aware binning, flexible sample) #
# --------------------------------------------------------------------------- #
import re

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CANON_CHROMS = set(CHROM_LIST_24)

def _norm_chr(x: str) -> str:
    """'chr1', '1', 'X' 등을 'chr1','chrX'로 통일. 기타는 앞의 'chr' 유지."""
    s = str(x).strip()
    m = re.match(r'(?i)^(?:chr)?([0-9]+|x|y|m|mt)$', s)
    if not m:
        # 'chr11_gl000202_random' 같은 건 그대로 둔다
        return s if s.lower().startswith("chr") else ("chr" + s)
    body = m.group(1).upper()
    if body in ("M", "MT"): return "chrM"
    return "chr" + body

def _apply_chr_norm(df: pd.DataFrame, col: str = "chrom") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].map(_norm_chr)
    return df

def _infer_bin_size_and_anchor_from_bins(cls_bins: pd.DataFrame) -> tuple[int, int]:
    """CLS bin으로부터 bin_size, anchor(start%bin_size 최빈값) 추정"""
    lens = (cls_bins["end"] - cls_bins["start"] + 1)
    bin_size = int(lens.mode().iat[0])
    anchor   = int((cls_bins["start"] % bin_size).mode().iat[0])
    return bin_size, anchor

def _bins_from_cls_list(cls_list):
    rows = []
    for chrom, w_idx, sbp, ebp, cls_vec, y_val in cls_list:
        rows.append((str(chrom), int(sbp), int(ebp)))
    df = pd.DataFrame(rows, columns=["chrom","start","end"]).drop_duplicates()
    df = _apply_chr_norm(df, "chrom").sort_values(["chrom","start"]).reset_index(drop=True)
    return df

def _guess_sample_col(df: pd.DataFrame) -> str:
    """sample 컬럼명이 제각각일 때 자동 감지."""
    lower = {c.lower(): c for c in df.columns}
    # 1) 흔한 이름 우선
    for k in ["sample","sample_id","donor","donor_id","tumor","tumour","tumor_id","vcf","file","tumor_sample"]:
        if k in lower: return lower[k]
    # 2) 내용 패턴으로 추정(값에 .vcf / purple / CPCT / WIDE 등 포함)
    cand = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            hits = s.str.contains(r"(purple|\.vcf|CPCT|WIDE|OBC)", case=False, regex=True, na=False).mean()
            if hits > 0.2:  # 20% 이상 해당 패턴
                cand.append((hits, c))
    if cand:
        cand.sort(reverse=True)
        return cand[0][1]
    # 3) 마지막 수단: 유니크도가 높은 object 컬럼 사용
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        uniq = sorted(((df[c].nunique(dropna=True), c) for c in obj_cols), reverse=True)
        return uniq[0][1]
    raise KeyError("sample-like column not found")

def _load_mutations_events(path: str,
                           use_midpoint: bool = True,
                           require_pass: bool = True) -> pd.DataFrame:
    """
    CSV/TSV에서 (chrom, pos, sample)만 추출.
    - filter 있으면 PASS/./...;PASS;... 허용
    - sample 컬럼명 자동 추정
    - 비표준 컨티그 제외: chr1..chr22, chrX, chrY만 사용
    """
    try:
        raw = pd.read_csv(path)
    except Exception:
        raw = pd.read_csv(path, sep="\t")

    cols = {c.lower(): c for c in raw.columns}
    def pick(cands, what):
        for k in cands:
            if k in cols: return cols[k]
        raise KeyError(f"{path}: required column '{what}' not found")

    c_chrom = pick(["chrom","chr","chromosome","seqnames"], "chrom")
    c_start = pick(["start","pos","bp"], "start")
    c_end   = cols.get("end", c_start)
    c_filt  = cols.get("filter", None)
    c_samp  = _guess_sample_col(raw)

    df = pd.DataFrame({
        "chrom":  raw[c_chrom].astype(str).values,
        "start":  pd.to_numeric(raw[c_start], errors="raise").astype(np.int64).values,
        "end":    pd.to_numeric(raw[c_end],   errors="raise").astype(np.int64).values,
        "sample": raw[c_samp].astype(str).values,
        "filter": raw[c_filt].astype(str).values if c_filt else "PASS"
    })
    _apply_chr_norm(df, "chrom")

    # 필터 처리
    if require_pass:
        f = df["filter"].astype(str).str.upper().fillna("")
        pass_mask = (f == "PASS") | (f == ".") | f.str.contains("PASS", regex=False)
        df = df[pass_mask].copy()

    # 표준 염색체만 사용
    before = len(df)
    df = df[df["chrom"].isin(CANON_CHROMS)].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[LABEL] dropped non-canonical contigs: {dropped} rows", flush=True)

    pos = ((df["start"] + df["end"]) // 2).astype(np.int64) if use_midpoint else df["start"].astype(np.int64)
    return pd.DataFrame({"chrom": df["chrom"].values,
                         "pos":   pos.values,
                         "sample":df["sample"].values})

def _build_y_map_from_mutations(cls_bins: pd.DataFrame,
                                events_df: pd.DataFrame,
                                *, verbose: bool = True) -> dict:
    """CLS 그리드와 같은 bin_size/anchor로 이벤트 binning → (chrom,start,end)별 unique sample 수."""
    cls_bins = _apply_chr_norm(cls_bins.copy(), "chrom")
    events_df = _apply_chr_norm(events_df.copy(), "chrom")

    bin_size, anchor = _infer_bin_size_and_anchor_from_bins(cls_bins)
    ev = events_df.copy()
    ev["bin_start"] = ((ev["pos"] - anchor) // bin_size) * bin_size + anchor
    ev["bin_end"]   = ev["bin_start"] + (bin_size - 1)

    ev = ev.merge(cls_bins, left_on=["chrom","bin_start"], right_on=["chrom","start"],
                  how="inner", suffixes=("","_grid"))
    if verbose:
        print(f"[LABEL] events={len(events_df):,}  matched_rows={len(ev):,}  "
              f"matched_bins={ev[['chrom','start','end']].drop_duplicates().shape[0]:,}  "
              f"(bin_size={bin_size}, anchor={anchor})", flush=True)

    dedup = ev.drop_duplicates(["chrom","start","end","sample"])
    agg   = (dedup.groupby(["chrom","start","end"]).size()
                  .rename("y").reset_index())

    y_map = {(r["chrom"], int(r["start"]), int(r["end"])): int(r["y"])
             for _, r in agg.iterrows()}
    return y_map

def _attach_labels_from_y_map(all_segments, y_map: dict):
    miss = 0
    for seg in all_segments:
        s_arr = seg["start_array"]; e_arr = seg["end_array"]
        y_new = np.zeros_like(s_arr, dtype=np.float32)
        chrom = _norm_chr(seg["chrom"])
        for j, (s, e) in enumerate(zip(s_arr, e_arr)):
            y_new[j] = float(y_map.get((chrom, int(s), int(e)), 0))
            if y_new[j] == 0: miss += 1
        seg["y_array"] = y_new
    print(f"[LABEL] attached to segments (zeros on {miss} bins with no match).", flush=True)
    return all_segments


# ───────── NHPP-정합 rolling-sum(box) helpers ─────────

def _make_kernel_box(width: int, device):
    w = int(max(1, width))
    return torch.ones(1, 1, w, device=device, dtype=torch.float32)  # (out=1,in=1,K)

def _conv1d_causal_sum(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T), k: (1,1,K)
    왼쪽만 (K-1) 만큼 0-패딩 → 출력 길이 = T 유지
    각 위치 t에서 윈도우 [t-K+1, ..., t] 합(부족분은 0으로 보충)
    """
    K = k.size(-1)
    x_pad = F.pad(x.unsqueeze(1), (K-1, 0))   # (B,1,T+K-1)
    return F.conv1d(x_pad, k, padding=0).squeeze(1)  # (B,T)

def rolling_sum_nhpp(lam: torch.Tensor, y: torch.Tensor, len_kb: torch.Tensor,
                     *, width: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NHPP-정합 롤링(합), 길이 보존(causal/right-aligned):
      y'  = Σ_{j=t-width+1..t} y_j
      μ'  = Σ (λ_j·Δt_j)
      Δt' = Σ Δt_j
      λ'  = μ'/Δt'
    width<=1이면 변화 없음.
    """
    if width <= 1:
        return lam, y, len_kb
    eps = 1e-9
    k = _make_kernel_box(width, y.device)

    mu    = lam * len_kb                         # 기대카운트 μ = λ·Δt
    y_r   = _conv1d_causal_sum(y,      k)        # (B,T)
    mu_r  = _conv1d_causal_sum(mu,     k)        # (B,T)
    dt_r  = _conv1d_causal_sum(len_kb, k)        # (B,T)
    lam_r = mu_r / (dt_r + eps)                  # 길이로 가중한 λ 평균

    return lam_r, y_r, dt_r


# --------------------------------------------------------------------------- #
# Model blocks                                                                #
# --------------------------------------------------------------------------- #
class FeatureEmbedder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=768):
        super().__init__(); self.proj = nn.Linear(feature_dim, hidden_dim)
    def forward(self, x): return self.proj(x)


# ── NEW (sum + LayerNorm 2개) ──────────────────────
class FeatClsFusion(nn.Module):
    """
    CLS + Feature 를 같은 hidden 차원(d_model)에서
    LayerNorm 뒤 element-wise sum.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.ln_cls  = nn.LayerNorm(hidden_dim)
        self.ln_feat = nn.LayerNorm(hidden_dim)
    def forward(self, cls_emb, feat_emb):
        # 입력 두 텐서 shape: (B, L, hidden_dim)
        return self.ln_cls(cls_emb) + self.ln_feat(feat_emb)

class ChromosomeEmbedder(nn.Module):
    def __init__(self, n, d=768):
        super().__init__()
        self.emb = nn.Embedding(n, d)
        # 하드코딩 스케일(여기만 바꾸면 전체 반영됨)
        self.register_buffer("scale", torch.tensor(0.2, dtype=torch.float32))

    def forward(self, ids):
        return self.emb(ids) * self.scale

# ------------------------------------------------------------------ #
# Rotary-PE wrapper  (멀티-GPU 안전 버전)                             #
# ------------------------------------------------------------------ #
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


# ── ① SwiGLU FFN ----------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w12   = nn.Linear(d_model, hidden * 2, bias=False)
        self.proj  = nn.Linear(hidden,  d_model,  bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.proj(F.silu(gate) * x))


# ── ② Pre-LN + 0.1-Residual Attn-Layer ------------------------------
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


class NHPPHead(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__(); self.ln = nn.LayerNorm(hidden_dim); self.lin = nn.Linear(hidden_dim, 1)
        self.log_c  = nn.Parameter(torch.zeros(())) 
    def forward(self, x):
        rate = F.softplus(self.lin(self.ln(x))).squeeze(-1)  # (B,L)
        return (rate * torch.exp(self.log_c)).clamp_min(1e-6)


# --------------------------------------------------------------------------- #
# Dataset / collate / loss                                                    #
# --------------------------------------------------------------------------- #
class SegmentDataset(Dataset):
    def __init__(self, segments): self.segments = segments
    def __len__(self): return len(self.segments)
    def __getitem__(self, idx): return self.segments[idx]


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


def trapezoid_nhpp_loss(lam, y, len_kb, reduction="mean"):
    """
    Bin-level(사각형) NHPP 음의 로그우도:
      -ℓ = - Σ_i [ y_i log λ_i  -  λ_i * Δt_i ]
    """
    lam_safe = lam.clamp(1e-9, 1e4)
    # Σ y_i log λ_i
    sum_log = (y * torch.log(lam_safe)).sum(dim=1)
    # Σ λ_i * Δt_i   ← (사다리꼴이 아니라 각 bin 자체 길이 사용)
    integ = (lam_safe * len_kb).sum(dim=1)
    ll = sum_log - integ
    return (-ll if reduction == "none" else -ll.mean())



def trapezoid_nhpp_loss_segment_weighted(lam, y, len_kb, w_seg):
    """
    세그먼트별 NLL(bin) → (per-kb 정규화 × 30) → 세그먼트 허버가중 평균
    (내부의 기본 손실은 bin-level NLL을 호출)
    """
    per_seg = trapezoid_nhpp_loss(lam, y, len_kb, reduction="none")  # (B,)
    seg_len = (len_kb.sum(dim=1) + 1e-9)
    per_seg = (per_seg / seg_len) * 30.0
    return (w_seg * per_seg).sum() / (w_seg.sum() + 1e-9)

@torch.no_grad()
def calibrate_log_c_huber_like_training(model_c, loader, device,
                                        huber_factor=3.0, use_mad=False,
                                        label_roll=False, roll_width=1):
    for m in model_c.values():
        m.eval()
    nh = unwrap(model_c["nhpp_head"])
    c_cur = float(torch.exp(nh.log_c).cpu())

    # 1) 세그 residual 수집 → 허버 가중  (★ y - μ 로 변경)
    seg_residual = {}
    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = b["y_array"].to(device)
        len_b  = b["length_array"].to(device)
        cid_b  = b["chrom_id"].to(device)

        key_pad = (len_b <= 0)
        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        lam      = model_c["nhpp_head"](
            model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        ).clamp(1e-9, 1e4)

        if label_roll and roll_width > 1:
            lam, y_b, len_b = rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)

        # ★ μ = λ·Δt
        mu_b = lam * len_b

        y_np, mu_np = y_b.cpu().numpy(), mu_b.cpu().numpy()
        for i, seg in enumerate(b["raw_segments"]):
            L = seg["cls_array"].shape[0]
            seg_residual[seg["global_idx"]] = float((y_np[i, :L] - mu_np[i, :L]).mean())

    rs = np.array(list(seg_residual.values()), dtype=np.float64)
    scale = compute_mad(rs) if use_mad else compute_iqr(rs)
    delta = max(huber_factor * scale, 1e-9)
    def _hw(r, d):
        a = abs(r); return 1.0 if a <= d else d / (a + 1e-9)
    w_seg = {sid: _hw(r, delta) for sid, r in seg_residual.items()}

    # 2) 허버 가중 합으로 c* 추정 (기존 공식 유지)
    num, den = 0.0, 0.0
    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = b["y_array"].to(device)
        len_b  = b["length_array"].to(device)
        cid_b  = b["chrom_id"].to(device)

        key_pad = (len_b <= 0)
        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        lam      = model_c["nhpp_head"](
            model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        ).clamp(1e-9, 1e4)

        if label_roll and roll_width > 1:
            lam, y_b, len_b = rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)

        base = lam / max(c_cur, 1e-12)

        for i, seg in enumerate(b["raw_segments"]):
            L     = seg["cls_array"].shape[0]
            sid   = seg["global_idx"]
            w     = float(w_seg.get(sid, 1.0))
            y_i   = y_b[i, :L]
            dt_i  = len_b[i, :L]
            base_i= base[i, :L]
            num  += w * y_i.sum().item()
            den  += w * (base_i * dt_i).sum().item()

    eps = 1e-12
    if num <= eps or den <= eps:
        print(f"[CAL-HUBER-TRAIN] skip (num={num:.3g}, den={den:.3g}) keep c={c_cur:.6g}", flush=True)
        return

    c_star = num / den
    nh.log_c.copy_(torch.tensor(math.log(max(c_star, eps)), device=nh.log_c.device))
    print(f"[CAL-HUBER-TRAIN] delta={delta:.4g}  c_prev={c_cur:.6g}  c_new={c_star:.6g}  "
          f"ratio={c_star/(c_cur+eps):.4f}", flush=True)

@torch.no_grad()
def _compute_seg_residuals_for_loader(model_c, loader, device, *,
                                      label_roll: bool = False,
                                      roll_width: int = 1):
    """세그 residual r = mean(y - μ) (pad 제외).
    label_roll/roll_width로 훈련과 동일한 롤링(합)을 적용할 수 있음.
    """
    for m in model_c.values():
        m.eval()

    res = {}
    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = b["y_array"].to(device)
        len_b  = b["length_array"].to(device)
        cid_b  = b["chrom_id"].to(device)
        key_pad = (len_b <= 0)

        # forward
        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        lam      = model_c["nhpp_head"](
            model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        ).clamp(1e-9, 1e4)   # per-bin rate

        # ★ 훈련과 동일한 롤링(합) 적용: λ′, y′, Δt′
        if label_roll and roll_width > 1:
            lam, y_b, len_b = rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)

        # μ′ = λ′ · Δt′  (또는 롤링 미적용시 μ = λ·Δt)
        mu_b = lam * len_b

        # 세그 유효 길이만 사용하여 r = mean(y − μ)
        for i, seg in enumerate(b["raw_segments"]):
            L = seg["cls_array"].shape[0]
            res[seg["global_idx"]] = float((y_b[i, :L] - mu_b[i, :L]).mean().item())

    return res



def _to_huber_weights_from_res(res_dict, huber_factor, use_mad):
    rs = np.array(list(res_dict.values()), dtype=np.float64)
    scale = compute_mad(rs) if use_mad else compute_iqr(rs)
    delta = max(huber_factor * scale, 1e-9)
    w = {sid: huber_weight(r, delta) for sid, r in res_dict.items()}
    return w, float(delta)

@torch.no_grad()
def _perseg_bin_nll30k(lam, y, len_kb):
    """bin-level NLL을 세그 단위로 (per-kb×30) 정규화한 벡터(B,) 반환."""
    lam_safe = lam.clamp(1e-9, 1e4)
    sum_log  = (y * torch.log(lam_safe)).sum(dim=1)
    integ    = (lam_safe * len_kb).sum(dim=1)
    neg_ll   = -(sum_log - integ)                     # (B,)
    seg_len  = (len_kb.sum(dim=1) + 1e-9)             # (B,)
    return (neg_ll / seg_len) * 30.0                  # (B,)

@torch.no_grad()
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


# --------------------------------------------------------------------------- #
# Attention‑save helpers                                                      #
# --------------------------------------------------------------------------- #

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

@torch.no_grad()
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


# ────────────────────────────────────────────────────────────────────────────
# Post pipeline (LLR → GMM → DP)
# ────────────────────────────────────────────────────────────────────────────

def _chr_key(c: str):
    c = c.lower().removeprefix("chr")
    if c.isdigit():
        return (0, int(c))
    return {"x": (1, 0), "y": (2, 0), "m": (3, 0), "mt": (3, 0)}.get(c, (4, c))

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

# ─────── DP (weighted interval scheduling) with gap option ────────────────
def _pred_idx(lst, i, gap_bp):
    s = lst[i]["start_bp"]
    lo, hi, res = 0, i - 1, -1
    while lo <= hi:
        m = (lo + hi) // 2
        # 이전 구간의 end + gap < 현재 start 여야 호환됨 (gap=0이면 기존 '<' 유지)
        if lst[m]["end_bp"] + gap_bp < s:
            res, lo = m, m + 1
        else:
            hi = m - 1
    return res

def dp_select(iv_list, gap_bp=0):
    by_chr = defaultdict(list)
    for iv in iv_list:
        by_chr[iv["chrom"]].append(iv)

    chosen = []
    for ch, lst in by_chr.items():
        lst.sort(key=lambda d: d["end_bp"])
        n      = len(lst)
        pred   = [_pred_idx(lst, i, gap_bp) for i in range(n)]
        dp     = np.zeros(n + 1)
        keep   = np.zeros(n + 1, bool)

        for i in range(1, n + 1):
            cand = lst[i-1]["LLR_weighted"] + dp[pred[i-1] + 1]
            if cand > dp[i-1]:
                dp[i], keep[i] = cand, True
            else:
                dp[i] = dp[i-1]

        i = n
        while i > 0:
            if keep[i]:
                chosen.append(lst[i-1])
                i = pred[i-1] + 1
            else:
                i -= 1
    return chosen

# ─────── GMM helpers ───────────────────────────────────────────────────────
def _resolve_presmooth_bins(args) -> int:
    pb = getattr(args, "pipeline_presmooth_bins", None)
    if pb is not None:
        # 사용자가 준 값을 그대로 사용 (0/1 ⇒ off)
        return max(0, int(pb))
    if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
        return int(args.label_roll_width)
    return 1


def fit_gmm_auto(sample,
                 k_min: int = 1,
                 k_max: int = 8,
                 seed: int = 0,
                 n_init: int = 3,
                 max_iter: int = 500):
    """
    log1p(sample) 위에서 GaussianMixture(k)의 BIC를 비교해 최적 k 선택.
    원 코드의 '스파이크 성분(μ<0.05 & σ<0.05)' 제거 규칙을 동일 적용.
    반환: dict(w,mu,sig,k_init,k_final)
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return {"w": np.array([1.0]), "mu": np.array([0.0]), "sig": np.array([0.5]),
                "k_init": 1, "k_final": 1}

    logx = np.log1p(x).reshape(-1, 1)
    best_gmm, best_bic = None, float("inf")
    k_max_eff = max(k_min, min(k_max, int(max(1, x.size // 10))))  # 표본 너무 작으면 k 상한 축소

    for k in range(max(1, k_min), max(1, k_max_eff) + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=n_init, max_iter=max_iter).fit(logx)
        bic = g.bic(logx)
        if bic < best_bic:
            best_bic, best_gmm = bic, g

    w = best_gmm.weights_.copy()
    mu = best_gmm.means_.ravel().copy()
    sig = np.sqrt([np.diag(c)[0] for c in best_gmm.covariances_])

    # 스파이크 제거 후 재정규화 (원 규칙 유지)
    keep = ~((mu < 0.05) & (sig < 0.05))
    if keep.sum() == 0:
        keep[:] = True
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w = w / w.sum()

    return {"w": w, "mu": mu, "sig": sig,
            "k_init": best_gmm.n_components, "k_final": len(w)}
# ===========================================================================

def fit_gmm(sample, k, seed):
    logx = np.log1p(sample).reshape(-1, 1)
    g    = GaussianMixture(k, random_state=seed).fit(logx)
    w, mu = g.weights_, g.means_.ravel()
    sig   = np.sqrt([np.diag(c)[0] for c in g.covariances_])
    keep  = ~( (mu < 0.05) & (sig < 0.05) )
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w /= w.sum()
    return dict(w=w, mu=mu, sig=sig)

def mix_sf(x, g):
    """
    Mixture survival function: p = P[X >= x] on log1p-scale Gaussian mixture.
    x: 1D array-like of llr_norm
    g: dict(w, mu, sig) from fit_gmm
    """
    z = np.log1p(np.maximum(np.asarray(x, dtype=np.float64), 0.0))[:, None]  # (N,1)
    t = (z - g["mu"]) / g["sig"]                                             # (N,K)
    comp_log = np.log(g["w"])[None, :] + norm.logsf(t)                       # log(w_k * SF_k)
    return np.exp(logsumexp(comp_log, axis=1))                               # (N,)

def mix_neglog10p_from_gmm(x, g, min_p=1e-300):
    """
    Stable -log10 p using mixture survival in log-domain.
    반환: (-log10 p, p)
    """
    z = np.log1p(np.maximum(np.asarray(x, dtype=np.float64), 0.0))[:, None]
    t = (z - g["mu"]) / g["sig"]
    comp_log = np.log(g["w"])[None, :] + norm.logsf(t)
    logp = logsumexp(comp_log, axis=1)                                       # ln p
    logp = np.maximum(logp, np.log(min_p))                                   # floor
    neglog10p = -logp / np.log(10.0)
    return neglog10p, np.exp(logp)

def _sanitize_pvals(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    return np.clip(p, 0.0, 1.0) if p.size else p

def estimate_pi0_storey_bootstrap(p_values,
                                  lambdas=None,
                                  B=200,
                                  seed=None,
                                  pi0_floor=0.01,
                                  pi0_ceil=1.0):
    """
    Storey(2002/2003) 부트스트랩 기반 λ 선택.
    반환: pi0_hat, lambda_star, pi0_grid, mse_grid
    """
    p = _sanitize_pvals(p_values); m = p.size
    if m == 0:
        return 1.0, 0.5, np.array([1.0]), np.array([0.0])

    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.01, dtype=np.float64)
    lambdas = lambdas[(lambdas >= 0.0) & (lambdas < 1.0)]
    if lambdas.size == 0:
        lambdas = np.array([0.5], dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        pi0_grid = np.array([np.mean(p > lam) / max(1e-12, 1.0 - lam) for lam in lambdas], dtype=np.float64)
    pi0_grid = np.clip(pi0_grid, pi0_floor, pi0_ceil)
    pi0_min = float(np.min(pi0_grid))

    rng = np.random.default_rng(seed)
    mse_grid = np.zeros_like(lambdas, dtype=np.float64)
    B = max(1, int(B))
    for _ in range(B):
        pb = rng.choice(p, size=m, replace=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            pi0_b = np.array([np.mean(pb > lam) / max(1e-12, 1.0 - lam) for lam in lambdas], dtype=np.float64)
        pi0_b = np.clip(pi0_b, pi0_floor, pi0_ceil)
        mse_grid += (pi0_b - pi0_min) ** 2
    mse_grid /= float(B)

    j = int(np.argmin(mse_grid))
    lambda_star = float(lambdas[j])
    pi0_hat = float(np.clip(pi0_grid[j], pi0_floor, pi0_ceil))
    return pi0_hat, lambda_star, pi0_grid, mse_grid

def qvalues_storey(p_values, pi0):
    """
    Storey q-values with monotone adjustment.
    """
    p = np.asarray(p_values, dtype=np.float64); m = p.size
    if m == 0: return p
    order = np.argsort(p, kind="mergesort"); p_sorted = p[order]
    ranks = np.arange(1, m+1, dtype=np.float64)
    q_sorted = pi0 * m * p_sorted / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted); q[order] = q_sorted
    return q

def qvalues_bh(p_values):
    """
    Benjamini–Hochberg q-values (monotone).
    """
    p = np.asarray(p_values, dtype=np.float64); m = p.size
    if m == 0: return p
    order = np.argsort(p); p_sorted = p[order]
    q_sorted = (m * p_sorted) / np.arange(1, m+1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted); q[order] = q_sorted
    return q
# ========================================================================== #


def qq_plot(p, title, pdf):
    """
    Draw a -log10 QQ-plot with independent x/y axis limits.
    - xlim, ylim을 각각 데이터 최대값에 맞춰 따로 잡음
    - 대각선(y=x)은 두 축 범위의 공통 구간까지만 표시
    """
    p = np.asarray(p, dtype=float)

    # p=0 처리: 최소 양수 p 주변으로 지터 부여
    mask_zero = (p <= 0)
    if mask_zero.any():
        min_pos = p[~mask_zero].min() if (~mask_zero).any() else 1e-30
        p[mask_zero] = np.random.uniform(
            low=min_pos * 0.5,
            high=min_pos * 0.9,
            size=mask_zero.sum()
        )

    # 안정성: [1e-300, 1] 클리핑 후 정렬
    p = np.clip(p, 1e-300, 1.0)
    p.sort()
    m = len(p)
    theo = (np.arange(1, m + 1) - 0.5) / max(m, 1)
    x = -np.log10(theo)
    y = -np.log10(p)

    # 축 한쪽씩 범위 계산 (여유 20%)
    x_max = float(x.max()) if m else 1.0
    y_max = float(y.max()) if m else 1.0
    x_lim = (0.0, max(1.0, x_max * 1.2))
    y_lim = (0.0, max(1.0, y_max * 1.2))

    plt.figure(figsize=(5, 5))
    if m:
        plt.scatter(x, y, s=8, fc="white", ec="k", lw=.4)

    # 기준선 y=x는 공통 구간까지만
    diag_end = min(x_lim[1], y_lim[1])
    plt.plot([0, diag_end], [0, diag_end], 'r--', lw=1.2)

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel("Theoretical -log10 p")
    plt.ylabel("Observed -log10 p")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pdf, dpi=300)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Events loader & summarizer for DP intervals
# ────────────────────────────────────────────────────────────────────────────
def _load_events_for_dp(path: str, *, use_midpoint: bool = True, require_pass: bool = True) -> pd.DataFrame:
    # ① 읽기: UUID는 문자열로, low_memory=False로 타입 흔들림 방지
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"UUID": "string"})
    except Exception:
        df = pd.read_csv(path, sep="\t", low_memory=False, dtype={"UUID": "string"})

    # ② BOM/공백 제거 후 소문자 매핑
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    def pick(names, what):
        for k in names:
            if k in cols: 
                return cols[k]
        raise KeyError(f"events file needs '{what}' (found columns: {list(df.columns)})")

    # 컬럼 찾기(동의어 넓힘)
    c_chrom = pick(["chrom","chr","seqnames","chromosome"], "chrom")
    c_start = pick(["start","pos","bp"], "start")
    c_end   = cols.get("end", c_start)
    c_type  = pick([
        "event_type","type","svtype","sv_type",
        "varianttypes","variant_type","svclass","sv_class",
        "class","event","category"
    ], "event_type")
    c_samp  = pick(["sample","sample_id","donor","donor_id","tumor","tumour"], "sample")
    c_filt  = cols.get("filter", None)

    out = pd.DataFrame({
        "chrom"     : df[c_chrom].astype(str),
        "start"     : pd.to_numeric(df[c_start], errors="raise").astype(np.int64),
        "end"       : pd.to_numeric(df[c_end],   errors="raise").astype(np.int64),
        "event_type": df[c_type].astype(str),
        "sample"    : df[c_samp].astype(str),
        "filter"    : df[c_filt].astype(str) if c_filt else "PASS"
    })

    if require_pass:
        f = out["filter"].astype(str).str.upper().fillna("")
        out = out[(f == "PASS") | (f == ".") | (f.str.contains("PASS", regex=False))].copy()

    # 이벤트 타입 표준화(대소문자/표기 차이 흡수, BND→TRA 등)
    et = out["event_type"].astype(str).str.strip().str.lower()
    et = et.str.split(";").str[0]  # "DUP;INV" 같은 복합 표기가 있다면 첫 항만 사용
    map_et = {
        "duplication": "dup", "dup": "dup",
        "deletion": "del",     "del": "del",
        "inversion": "inv",    "inv": "inv",
        "translocation": "tra","tra": "tra",
        "bnd": "tra",          "t": "tra"
    }
    out["event_type"] = et.map(map_et).fillna(et)

    # 좌표를 단일 pos로(중점/시작 선택)
    pos = ((out["start"] + out["end"]) // 2).astype(np.int64) if use_midpoint else out["start"].astype(np.int64)
    return pd.DataFrame({"chrom": out["chrom"].astype(str),
                         "pos":   pos.values,
                         "event_type": out["event_type"].astype(str),
                         "sample": out["sample"].astype(str)})


def _summarize_events_for_intervals(events_df: pd.DataFrame, sel_intervals: list, *, dedup_sample: bool = True):
    """
    DP로 선택된 interval 목록(sel_intervals: dict에 chrom/start_bp/end_bp 포함)에 대해
    event_type별 개수 + 분류 지표를 요약하여, interval별 dict 리스트를 반환.
    반환: (list_of_dicts, sorted_event_types)
    """
    by_chr = {c: d[["pos","event_type","sample"]].reset_index(drop=True)
              for c, d in events_df.groupby("chrom", sort=False)}
    results, all_types = [], set()

    print(f"[events] summarizing for {len(sel_intervals)} intervals ...", flush=True)
    for iv in tqdm(sel_intervals, desc="Summarizing events", ncols=88):
        ch, s, e = iv["chrom"], int(iv["start_bp"]), int(iv["end_bp"])
        row = {
            "n_evt_all": 0,
            "n_samp_all": 0,
            # 분류 지표 기본값
            "max_junc_per_sample": 0,
            "median_junc_per_sample": 0.0,
            "gini_junc": 0.0,
            "event_type_diversity": 0,
            "frac_TRA": 0.0,
        }
        if ch in by_chr:
            sub = by_chr[ch]
            sub = sub[(sub["pos"] >= s) & (sub["pos"] <= e)].copy()
            if not sub.empty:
                # 총계
                row["n_evt_all"] = int(len(sub))
                row["n_samp_all"] = int(sub["sample"].nunique())

                # 타입별 카운트
                t = sub["event_type"].astype(str).str.lower()
                vc_evt = t.value_counts()
                for tname, n in vc_evt.items():
                    key = f"n_evt_{tname}"
                    row[key] = int(n); all_types.add(tname)
                row["event_type_diversity"] = int(vc_evt.size)

                # 샘플별 분포
                per_sample = sub.groupby("sample").size()
                row["max_junc_per_sample"] = int(per_sample.max())
                row["median_junc_per_sample"] = float(per_sample.median())
                row["gini_junc"] = float(_gini_from_counts(per_sample.to_numpy(dtype=np.int64)))

                # 전좌 비율
                n_tra = int(t.str.contains("tra").sum())
                row["frac_TRA"] = (n_tra / row["n_evt_all"]) if row["n_evt_all"] > 0 else 0.0

                if dedup_sample:
                    vc_samp = (sub.drop_duplicates(["event_type","sample"])
                                  .groupby("event_type").size())
                    for tname, n in vc_samp.items():
                        key = f"n_samp_{tname}"
                        row[key] = int(n); all_types.add(tname)
        results.append(row)

    print(f"[events] types found: {sorted(all_types)}", flush=True)
    return results, sorted(all_types)



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

# ────────────────────────────────────────────────────────────────────────────
# GMM helpers (auto-k + fixed-k)  ← 파일 내 '한 번만' 존재하도록
# ────────────────────────────────────────────────────────────────────────────
def fit_gmm(sample, k, seed):
    logx = np.log1p(sample).reshape(-1, 1)
    g    = GaussianMixture(k, random_state=seed).fit(logx)
    w, mu = g.weights_, g.means_.ravel()
    sig   = np.sqrt([np.diag(c)[0] for c in g.covariances_])
    keep  = ~((mu < 0.05) & (sig < 0.05))  # 스파이크 제거
    if keep.sum() == 0: keep[:] = True
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w /= w.sum()
    return dict(w=w, mu=mu, sig=sig)

def fit_gmm_auto(sample,
                 k_min: int = 1,
                 k_max: int = 8,
                 seed: int = 0,
                 n_init: int = 3,
                 max_iter: int = 500):
    """
    BIC로 최적 k 선택. (log1p 변환 공간), 스파이크 성분 제거 후 재정규화.
    """
    x = np.asarray(sample, float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return {"w": np.array([1.0]), "mu": np.array([0.0]), "sig": np.array([0.5]),
                "k_init": 1, "k_final": 1}
    logx = np.log1p(x).reshape(-1, 1)
    best_gmm, best_bic = None, float("inf")
    k_max_eff = max(k_min, min(k_max, int(max(1, x.size // 10))))
    for k in range(max(1, k_min), max(1, k_max_eff)+1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=n_init, max_iter=max_iter).fit(logx)
        bic = g.bic(logx)
        if bic < best_bic:
            best_bic, best_gmm = bic, g
    w, mu = best_gmm.weights_.copy(), best_gmm.means_.ravel().copy()
    sig   = np.sqrt([np.diag(c)[0] for c in best_gmm.covariances_])
    keep  = ~((mu < 0.05) & (sig < 0.05))
    if keep.sum() == 0: keep[:] = True
    w, mu, sig = w[keep], mu[keep], sig[keep]
    w = w / w.sum()
    return {"w": w, "mu": mu, "sig": sig,
            "k_init": best_gmm.n_components, "k_final": len(w)}

def mix_neglog10p_from_gmm(x, g, min_p=1e-300):
    """
    Mixture SF p = P(X>=x) on log1p-space.
    반환: (-log10 p, p)
    """
    z = np.log1p(np.maximum(np.asarray(x, float), 0.0))[:, None]
    t = (z - g["mu"]) / g["sig"]
    comp_log = np.log(g["w"])[None, :] + norm.logsf(t)
    logp = logsumexp(comp_log, axis=1)
    logp = np.maximum(logp, math.log(min_p))
    neglog10p = -logp / math.log(10.0)
    return neglog10p, np.exp(logp)
# ======== add near other helpers ========
def _gini_from_counts(arr: np.ndarray) -> float:
    """Gini coefficient for a vector of positive counts (zeros excluded)."""
    x = np.asarray(arr, dtype=float)
    x = x[x > 0]
    n = x.size
    if n == 0:
        return 0.0
    x.sort()
    cum = np.cumsum(x)
    S = cum[-1]
    if S <= 0:
        return 0.0
    # Equivalent closed form using cumulative sums
    g = (n + 1.0 - 2.0 * (cum / S).sum() / n)
    return float(max(0.0, min(1.0, g)))

# ────────────────────────────────────────────────────────────────────────────
# Storey π0 + q-values  ← 파일 내 '한 번만' 존재하도록
# ────────────────────────────────────────────────────────────────────────────
def _sanitize_pvals(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return np.clip(p, 0.0, 1.0) if p.size else p

def estimate_pi0_storey_bootstrap(p_values,
                                  lambdas=None,
                                  B=200,
                                  seed=None,
                                  pi0_floor=0.01,
                                  pi0_ceil=1.0):
    """
    Storey(2002/2003): λ-grid에서 π0(λ)=#{p>λ}/((1-λ)m) 추정 후,
    부트스트랩으로 MSE 최소 λ* 선택.
    """
    p = _sanitize_pvals(p_values); m = p.size
    if m == 0: return 1.0, 0.5, np.array([1.0]), np.array([0.0])
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.01, dtype=float)
    lambdas = lambdas[(lambdas>=0.0) & (lambdas<1.0)]
    if lambdas.size == 0: lambdas = np.array([0.5], float)

    with np.errstate(divide='ignore', invalid='ignore'):
        pi0_grid = np.array([np.mean(p > lam)/max(1e-12, 1.0-lam) for lam in lambdas], float)
    pi0_grid = np.clip(pi0_grid, pi0_floor, pi0_ceil)
    pi0_min = float(np.min(pi0_grid))

    rng = np.random.default_rng(seed)
    mse_grid = np.zeros_like(lambdas, float)
    B = max(1, int(B))
    for _ in range(B):
        pb = rng.choice(p, size=m, replace=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            pi0_b = np.array([np.mean(pb > lam)/max(1e-12, 1.0-lam) for lam in lambdas], float)
        pi0_b = np.clip(pi0_b, pi0_floor, pi0_ceil)
        mse_grid += (pi0_b - pi0_min)**2
    mse_grid /= float(B)

    j = int(np.argmin(mse_grid))
    return float(pi0_grid[j]), float(lambdas[j]), pi0_grid, mse_grid

def qvalues_storey(p_values, pi0):
    """
    Storey q-values with monotone adjustment.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    # p는 [0,1]로 클립(극단값/음수 방지)
    p = np.clip(p, 0.0, 1.0)

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]

    # ❗ dtype로 지정 (세 번째 인자 아님)
    ranks = np.arange(1, m + 1, dtype=np.float64)

    q_sorted = pi0 * m * p_sorted / ranks
    # 단조 감소로 조정 (뒤에서 앞으로 누적 최소)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q

def qvalues_bh(p_values):
    """
    Benjamini–Hochberg q-values (monotone).
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    p = np.clip(p, 0.0, 1.0)

    order = np.argsort(p)
    p_sorted = p[order]

    ranks = np.arange(1, m + 1, dtype=np.float64)

    q_sorted = (m * p_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q

# ==========================================================================
def run_llr_gmm_dp_pipeline(all_pred_path: str,
                            out_dir: Path,
                            chunk_size: int    = 1_000_000,
                            chunk_overlap: int = 100_000,
                            min_distance: int  = 0,
                            max_distance: int  = 100_000,
                            sample_frac: float = 0.01,
                            gmm_k: int         = 3,
                            beta: float        = 1.0,
                            gamma: float       = 0.0,
                            seed: int          = 0,
                            dp_gap_bp: int     = 0,
                            events_file: str   = None,
                            events_use_midpoint: bool = True,
                            events_dedup_sample: bool = True,
                            presmooth_bins: int = 1,
                            # ▼ 여기에 추가
                            gmm_auto: bool     = False,
                            # === post-selection ===
                            postsel_gmm_k_auto: bool = True,
                            postsel_gmm_k_max: int = 6,
                            postsel_gmm_n_init: int = 3,
                            postsel_gmm_max_iter: int = 500,
                            postsel_gmm_k: Optional[int] = None,
                            postsel_fdr_method: str = "storey",
                            postsel_bootstrap: int = 200,
                            postsel_lambda_start: float = 0.05,
                            postsel_lambda_end: float = 0.95,
                            postsel_lambda_step: float = 0.01,
                            postsel_pi0_floor: float = 0.01,
                            postsel_pi0_ceil: float = 1.0):
    """
    0) all_prediction → per-bin
    1) (옵션) presmoothing → variable-length LLR scan
    2) global GMM (√len LLR sample) → p계산에 사용
    3) p·-log10 p로 DP 가중 → DP (비중첩 집합)
    4) [핵심] 선택집합 재피팅(GMM, k 자동/BIC) → p_post
    5) 선택집합 내부 FDR(q) 계산(Storey 기본) → fdr_post (=fdr)
    """
    _set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PIPE] presmooth_bins={presmooth_bins} (variable scan)", flush=True)

    # 0) all_prediction → per-bin
    per_bin = out_dir / "prediction_per_bin.csv"
    make_per_bin(all_pred_path, per_bin.as_posix())
    print(f"[0] per-bin  → {per_bin}", flush=True)

    # 1) Poisson LLR scan
    bin_df = pd.read_csv(per_bin)
    bin_df["chrom"] = bin_df["chrom"].astype(str)
    chroms = sorted(bin_df["chrom"].unique(), key=_chr_key)

    llr_csv = out_dir / "llr_intervals.csv"
    pos_samp, n_pos, n_zero = [], 0, 0
    with llr_csv.open("w") as w:
        w.write("chrom,start_bp,end_bp,len_bp,LLR_raw\n")
        for ch in tqdm(chroms, desc="LLR scan"):
            sub = bin_df[bin_df["chrom"] == ch].sort_values("start")
            lam = sub["lam_pred"].to_numpy(np.float32)
            obs = sub["obs_count"].to_numpy(np.float32)
            st  = sub["start"].to_numpy(np.int64)
            en  = sub["end"].to_numpy(np.int64)

            if presmooth_bins and presmooth_bins > 1:
                lam, obs, st, en = _presmooth_nhpp_numpy(lam, obs, st, en, W_bins=presmooth_bins)

            ivs = []
            for ck in _build_chunks(lam, obs, st, en, chunk_size, chunk_overlap):
                ivs.extend(_scan_chunk(ck, min_distance, max_distance))

            for iv in ivs:
                w.write(f"{ch},{iv['start_bp']},{iv['end_bp']},{iv['len_bp']},{iv['LLR_raw']}\n")
                if iv["LLR_raw"] > 1e-6:
                    n_pos += 1
                    len_kb = max(iv["len_bp"] / 1_000.0, 1e-9)
                    if random.random() < sample_frac:
                        pos_samp.append(iv["LLR_raw"] / math.sqrt(len_kb))
                else:
                    n_zero += 1
    print(f"[1] interval csv saved   (+LLR={n_pos:,}, zero≈{n_zero:,})", flush=True)
    
    # ── (2) global GMM ──
    pos_arr = np.asarray(pos_samp, dtype=float)
    pos_arr = pos_arr[np.isfinite(pos_arr)]          # NaN/inf 제거
    if pos_arr.size == 0:
        print("[WARN] GMM sample empty; fallback.", flush=True)
        pos_arr = np.array([0.05, 0.10, 0.20], dtype=float)

    core = pos_arr  # 샘플 그대로 사용 (트리밍 없음)

    if gmm_auto:
        try:
            gmm = fit_gmm_auto(core, k_min=1, k_max=3, seed=seed, n_init=3, max_iter=500)
            if not len(gmm.get("w", [])):
                raise RuntimeError("Empty GMM after auto-fit")
            mode_str = "auto"
        except Exception:
            gmm = fit_gmm(core, k=int(gmm_k), seed=seed)
            mode_str = f"auto→fallback(k={int(gmm_k)})"
    else:
        gmm = fit_gmm(core, k=int(gmm_k), seed=seed)
        mode_str = f"fixed(k={int(gmm_k)})"

    print(f"[2] GMM mode={mode_str}, k={len(gmm['w'])}, π={np.round(gmm['w'],3)}", flush=True)


    # 3) p / weight / DP
    print("[3] DP selection …", flush=True)
    llr_df = pd.read_csv(llr_csv, dtype={"chrom": "category"})
    sel = []
    for ch in chroms:
        iv_chr = llr_df[llr_df["chrom"] == ch].copy()
        if iv_chr.empty: continue
        llr = iv_chr["LLR_raw"].to_numpy(float)
        len_kb = np.maximum(iv_chr["len_bp"].to_numpy(float)/1_000.0, 1e-9)
        llr_norm = llr / np.sqrt(len_kb)

        p = np.ones_like(llr, float)
        neglogp = np.zeros_like(llr, float)
        mask = (llr > 0)
        if mask.any():
            nl10, pv = mix_neglog10p_from_gmm(llr_norm[mask], gmm, min_p=1e-300)
            neglogp[mask] = nl10
            p[mask] = pv

        wgt = llr * np.power(neglogp, beta) * np.power(len_kb, gamma)
        iv_chr["p_val"] = np.clip(p, 1e-300, 1.0)
        iv_chr["neglog10_p"] = neglogp
        iv_chr["LLR_weighted"] = wgt

        chosen = dp_select(iv_chr.to_dict("records"), gap_bp=dp_gap_bp)
        sel.extend(chosen)
        print(f"    {ch:>4}: intervals={len(iv_chr):7,d} → selected={len(chosen):5,d}", flush=True)

    if len(sel) == 0:
        final_csv = out_dir / "final_result.csv"
        pd.DataFrame(columns=["chrom","start","end","len_bp","LLR_raw","obs_sum","exp_sum",
                              "p_post","fdr_post","fdr"]).to_csv(final_csv, index=False)
        print("[WARN] No intervals selected; wrote empty final_result.csv", flush=True)
        (out_dir / "qq_plot.pdf").write_bytes(b"")
        return final_csv.as_posix()

    # (옵션) events 요약
    evt_summaries, evt_types = None, []
    if events_file:
        try:
            _evt = _load_events_for_dp(events_file, use_midpoint=events_use_midpoint, require_pass=True)
            evt_summaries, evt_types = _summarize_events_for_intervals(_evt, sel, dedup_sample=events_dedup_sample)
            print(f"[events] loaded {len(_evt):,} rows; types={sorted(set(_evt['event_type']))}", flush=True)
        except Exception as e:
            print(f"[WARN] events summary skipped: {e}", flush=True)
            evt_summaries = None

    # 4) 선택집합(post-selection) – 전역 GMM을 그대로 사용 (하드코딩)
    llr_sel = np.array([iv["LLR_raw"] for iv in sel], float)
    len_kb_sel = np.array([max((iv["end_bp"] - iv["start_bp"] + 1) / 1_000.0, 1e-9) for iv in sel], float)
    llr_norm_sel = llr_sel / np.sqrt(len_kb_sel)

    # ★ 전역 GMM 재사용 (재피팅 금지)
    gmm_post = gmm

    # 선택집합 p (전역 GMM으로 계산)
    p_post = np.ones_like(llr_sel, float)
    msk = (llr_sel > 0)
    if msk.any():
        _, pv = mix_neglog10p_from_gmm(llr_norm_sel[msk], gmm_post, min_p=1e-300)
        p_post[msk] = np.clip(pv, 1e-300, 1.0)

        # 5) 선택집합 FDR
        method = postsel_fdr_method.lower()
        if method == "storey":
            lam_grid = np.arange(postsel_lambda_start, postsel_lambda_end + 1e-12,
                                postsel_lambda_step, float)
            pi0_hat, lam_star, _, _ = estimate_pi0_storey_bootstrap(
                p_post, lambdas=lam_grid, B=postsel_bootstrap, seed=seed,
                pi0_floor=postsel_pi0_floor, pi0_ceil=postsel_pi0_ceil
            )
            q_post = qvalues_storey(p_post, pi0_hat)
            print(f"[POST] Storey q: pi0={pi0_hat:.3f}, lambda*={lam_star:.2f}, B={postsel_bootstrap}", flush=True)
        else:
            q_post = qvalues_bh(p_post)
            print(f"[POST] BH q-values computed on selected set (m={len(p_post)})", flush=True)
        
    # ---- QQ plot (post-selection, DP-selected intervals only) ----
    
    qq_title = "QQ – Post-selection p (Storey)" if method == "storey" else "QQ – Post-selection p (BH)"
    qq_pdf   = out_dir / "qq_plot.pdf"
    if p_post.size and np.isfinite(p_post).any():
        qq_plot(p_post, qq_title, qq_pdf)   
    else:
        # 선택구간이 없으면 빈 파일로 남겨 UI/파이프라인이 깨지지 않게 함
        qq_pdf.write_bytes(b"")
    # --------------------------------------------------------------


    for iv, pp, qq in zip(sel, p_post, q_post):
        iv["p_post"] = float(pp)
        iv["fdr_post"] = float(qq)
        iv["fdr"] = float(qq)  # 최종 보고는 post-selection FDR
        
        # 6) obs/exp 집계 + CSV 저장  ← 여기부터 교체
    by_chr = {c: d for c, d in bin_df.groupby("chrom", observed=True)}
    rows = []
    for idx, iv in enumerate(sel):
        ch, s, e = iv["chrom"], iv["start_bp"], iv["end_bp"]
        sub = by_chr[ch]
        msk2 = (sub["start"] >= s) & (sub["end"] <= e)
        obs_sum = float(sub.loc[msk2, "obs_count"].sum())
        kb  = (sub.loc[msk2, "end"] - sub.loc[msk2, "start"] + 1) / 1_000.0
        exp_sum = float((sub.loc[msk2, "lam_pred"] * kb).sum())

        row = dict(chrom=ch, start=s, end=e, len_bp=e - s + 1,
                   LLR_raw=iv["LLR_raw"], obs_sum=obs_sum, exp_sum=exp_sum,
                   p_post=iv.get("p_post", np.nan), fdr_post=iv.get("fdr_post", np.nan), fdr=iv.get("fdr", np.nan))

        # 이벤트 요약/분류 지표가 있으면 병합
        if evt_summaries is not None:
            es = evt_summaries[idx]
            # 기본 수치
            row.update({
                "n_evt_all": es.get("n_evt_all", 0),
                "n_samp_all": es.get("n_samp_all", 0),
                "max_junc_per_sample": es.get("max_junc_per_sample", 0),
                "median_junc_per_sample": es.get("median_junc_per_sample", 0.0),
                "gini_junc": es.get("gini_junc", 0.0),
                "event_type_diversity": es.get("event_type_diversity", 0),
                "frac_TRA": es.get("frac_TRA", 0.0),
            })
            # 타입별 동적 컬럼(n_evt_*, n_samp_*)도 함께 병합
            for k, v in es.items():
                if k.startswith("n_evt_") or k.startswith("n_samp_"):
                    row[k] = v

    
        else:
            # 이벤트 파일이 없으면 분류 불가
            row.update({
                "n_evt_all": 0, "n_samp_all": 0,
                "max_junc_per_sample": 0,
                "median_junc_per_sample": 0.0,
                "gini_junc": 0.0,
                "event_type_diversity": 0,
                "frac_TRA": 0.0
             
            })

        rows.append(row)

    final_df = pd.DataFrame(rows).sort_values(["chrom", "start"]).reset_index(drop=True)
    # 타입별 동적 컬럼 정수화
    evt_cols = [c for c in final_df.columns if c.startswith("n_evt_") or c.startswith("n_samp_")]
    if evt_cols:
        final_df[evt_cols] = final_df[evt_cols].fillna(0).astype(int)

    final_csv = out_dir / "final_result.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[✓] final_result.csv → {final_csv}", flush=True)


# ---------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def train_and_predict(args):
    set_seed(args.seed)
    
        # ── CPU 스레드 수 제어 (옵션) ───────────────────────────────────────────
    if args.torch_threads is not None and args.torch_threads > 0:
        try:
            torch.set_num_threads(int(args.torch_threads))  # intra-op threads
            # (선택) inter-op도 조절하고 싶으면 아래 한 줄 추가:
            # torch.set_num_interop_threads(max(1, int(args.torch_threads) // 2))
            print(f"[INFO] torch num_threads = {torch.get_num_threads()}", flush=True)
        except Exception as e:
            print(f"[WARN] set_num_threads failed: {e}", flush=True)

    # ------------------------- 데이터 로딩 --------------------------------- #
    print("[INFO] Loading data ..", flush=True)
    with open(args.cls_file, "rb") as f:
        cls_list = pickle.load(f)
    with open(args.feat_file, "rb") as f:
        feature_dict = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"[INFO] device = {device}, #GPUs = {n_gpu}", flush=True)

    all_segments = segment_cls_embeddings_fixed_lengths(
        cls_list, feature_dict,
        seg_len_list=args.segment_lengths,
        discard_leftover=args.discard_leftover,
        overlap_factor=args.overlap_factor,
    )
    print(f"[INFO] #all_segments = {len(all_segments)}", flush=True)
    # ===== NEW: mutations CSV 로부터 1kb-bin 라벨 생성(하드 PASS) =====
    if args.mutations_file:
        print(f"[LABEL] building 1kb labels from mutations (PASS only): {args.mutations_file}", flush=True)
        cls_bins = _bins_from_cls_list(cls_list)
        ev    = _load_mutations_events(args.mutations_file, use_midpoint=True, require_pass=True)
        y_map = _build_y_map_from_mutations(cls_bins, ev)

        _attach_labels_from_y_map(all_segments, y_map)
        print(f"[LABEL] y_map bins = {len(y_map):,} (unique sample per 1kb bin)", flush=True)

    train_segs, val_segs, _ = [], [], []
    c_map = defaultdict(list)
    for seg in all_segments: c_map[seg["chrom"]].append(seg)
    for c in c_map:
        lst = c_map[c]; random.shuffle(lst)
        n = len(lst); n_train = int(n * 0.9)
        train_segs.extend(lst[:n_train]); val_segs.extend(lst[n_train:])
    ds_train, ds_val, ds_all = SegmentDataset(train_segs), SegmentDataset(val_segs), SegmentDataset(all_segments)

    chrom_id_map = build_chrom_id_map(None)
    collate_train = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=args.cutmix_p)
    collate_eval  = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=0.0)

    pin = device.type == "cuda"

    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_train,
                              num_workers=args.num_data_workers, pin_memory=pin)
    train_loader_infer = DataLoader(ds_train, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collate_eval,
                                    num_workers=args.num_data_workers, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin)
    all_loader = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin)

    # ------------------------- 모델 --------------------------------------- #
    any_feat = next(iter(feature_dict.values()))
    feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else any_feat.shape[0]

    feat_embedder = FeatureEmbedder(feature_dim, hidden_dim=args.d_model)
    feat_cls_fusion    = FeatClsFusion(hidden_dim=args.d_model)
    chrom_embedder     = ChromosomeEmbedder(len(CHROM_LIST_24), d=args.d_model)
    global_transformer = GlobalTransformerEncoder(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        max_seq_len=args.max_seq_len)
    nhpp_head          = NHPPHead()

    for m in (feat_embedder, feat_cls_fusion, chrom_embedder,
              global_transformer, nhpp_head):
        m.to(device)

    model_components = dict(feature_embedder=feat_embedder,
                            feat_cls_fusion=feat_cls_fusion,
                            chrom_embedder=chrom_embedder,
                            global_transformer=global_transformer,
                            nhpp_head=nhpp_head)

    if n_gpu > 1:
        print(f"[INFO] Using DataParallel on {n_gpu} GPUs", flush=True)
        for k in model_components:
            model_components[k] = nn.DataParallel(model_components[k])
            

    param_groups = [
        {  # 일반 파라미터 ― WD 적용
            "params": itertools.chain(
                model_components["feature_embedder"].parameters(),
                model_components["feat_cls_fusion"].parameters(),
                model_components["chrom_embedder"].parameters(),
                model_components["global_transformer"].parameters(),
            ),
            "weight_decay": 1e-3,
            "lr": args.lr,          # 필요하면 개별 lr 지정 가능
        },
        {  # NHPPHead ― WD 예외
            "params": model_components["nhpp_head"].parameters(),
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    #optimizer = AdamW(sum((list(m.parameters()) for m in model_components.values()), []),
                      #lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8)

    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=args.lr_sched_T0,
                                            T_mult=args.lr_sched_Tmult,
                                            eta_min=args.lr * 0.3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = out_dir / "trained_model.pt"
    
            # --------------------- 훈련 상태 복원 / 재개 --------------------------- #
    start_epoch, train_flag = 0, True
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        ckpt = efficient_load_ckpt(args.resume_checkpoint)

                                # ① 모델 파라미터
        for k in model_components:
            model_components[k].load_state_dict(ckpt[k], strict=False)

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "sched_state" in ckpt:
            scheduler.load_state_dict(ckpt["sched_state"])

        start_epoch     = ckpt.get("epoch",  -1) + 1
        best_val_loss   = ckpt.get("best_val_loss", float("inf"))
        del ckpt; gc.collect()
        print(f"[INFO] Resumed from {args.resume_checkpoint} (epoch {start_epoch})", flush=True)


    elif check_pretrained_model_exists(model_save_path):
        ckpt = efficient_load_ckpt(model_save_path)
        for k in model_components:
            model_components[k].load_state_dict(ckpt[k], strict=False)
        del ckpt; train_flag = False
        print("[INFO] Found trained_model.pt → skip training", flush=True)
    else:
        print("[INFO] Training from scratch", flush=True)

    # ----------------------------- Training loop -------------------------- #
    if train_flag:
        seg_weight_dict = {seg["global_idx"]: 1.0 for seg in all_segments}
        max_grad_norm, log_interval = 1.0, 1000
        best_val_loss, best_state_dict = float("inf"), None
        epochs_no_improve, step_global = 0, start_epoch * len(train_loader)
        tau, use_rw_sampler = 1.0, False
        train_hist, val_hist = [], []

        def weighted_loss_one_batch(batch, return_lam=False):
            cls_b, feat_b = batch["cls_array"].to(device), batch["feat_array"].to(device)
            y_b, len_b = batch["y_array"].to(device), batch["length_array"].to(device)
            cid_b = batch["chrom_id"].to(device)
            key_pad = (len_b <= 0)

            feat_emb = model_components["feature_embedder"](feat_b)
            fused = model_components["feat_cls_fusion"](cls_b, feat_emb)
            chr_emb = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

            if args.save_attention:
                out, attn_last = model_components["global_transformer"](
                    fused + chr_emb,key_padding_mask=key_pad, return_attn=True
                )
                # DataParallel → unwrap 해서 모듈 본체에 임시 저장
                unwrap(model_components["global_transformer"]).last_attn_cpu \
                    = attn_last[0].detach().cpu()
                torch.cuda.empty_cache()         # GPU 메모리 즉시 반납
            else:
                out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
            
            lam = model_components["nhpp_head"](out)
            if args.label_roll and args.label_roll_width > 1:
                lam, y_b, len_b = rolling_sum_nhpp(
                lam, y_b, len_b,
                width=args.label_roll_width
                )

            w_seg = torch.tensor([seg_weight_dict.get(seg["global_idx"], 1.0)
                                  for seg in batch["raw_segments"]],
                                 device=device, dtype=torch.float32)
            loss = trapezoid_nhpp_loss_segment_weighted(lam, y_b, len_b, w_seg)
            if return_lam:
                return loss, lam.detach()
            return loss

        for epoch in range(start_epoch, args.epochs):
            for m in model_components.values(): m.train()
            sum_loss, cnt = 0.0, 0

            for step, batch in enumerate(train_loader):
                step_global += 1
                if DEBUG_NAN:
                    loss_val, lam_dbg = weighted_loss_one_batch(batch, True)
                else:
                    loss_val = weighted_loss_one_batch(batch)

                if not torch.isfinite(loss_val):
                    print(f"[NaN] epoch={epoch} step={step_global}", flush=True)
                    raise RuntimeError("NaN detected")

                optimizer.zero_grad(); loss_val.backward()
                for g in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], max_grad_norm)

                optimizer.step(); scheduler.step()

                sum_loss += loss_val.item(); cnt += 1

                if step_global % log_interval == 0:
                    avg = sum_loss / cnt; sum_loss = 0; cnt = 0
                            # ---- 여기 ↓ 추가 --------------------------------
                    if DEBUG_NAN:                # lam_dbg 가 있을 때만
                        lam_mean = lam_dbg.mean().item()
                        lam_std  = lam_dbg.std().item()
                        y_full   = batch["y_array"].to(device)          # (B, L)
                        len_kb   = batch["length_array"].to(device)     # (B, L)  ← pad 위치는 0 KB
                        mask     = (len_kb > 0)                        # True = 실제 bin
                        
                        y_mean   = (y_full[mask]).mean().item() if mask.any() else 0.0
                        mu_dbg   = lam_dbg * len_kb
                        mu_mean  = (mu_dbg[mask]).mean().item() if mask.any() else 0.0
                        nhpp_head = model_components["nhpp_head"]          # ← 핵심
                        scale     = torch.exp(unwrap(nhpp_head).log_c).item()
                        ratio    = mu_mean  / max(y_mean, 1e-9)
  
                        print(f"[Epoch {epoch} | Step {step_global}] "
                               f"batch_avg_loss={avg:.4f}  μ_mean={mu_mean:.4g}  y_mean={y_mean:.4g}  μ/y={ratio:.3f}  scale={scale:.3f}",
                            flush=True)

                    else:
                        print(f"[Epoch {epoch} | Step {step_global}] "
                        f"batch_avg_loss={avg:.4f}",
                        flush=True)
                    
                    #print(f"[Epoch {epoch} | Step {step_global}] batch_avg_loss = {avg:.4f}", flush=True)

            # -------------------- epoch 평가 ------------------------------ #
            def eval_loader(loader):
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in loader:
                        cls_b  = b["cls_array"].to(device)
                        feat_b = b["feat_array"].to(device)
                        y_b    = b["y_array"].to(device)
                        cid_b  = b["chrom_id"].to(device)
                        len_b  = b["length_array"].to(device) 
                        key_pad  = (len_b <= 0)

                        feat_emb = model_components["feature_embedder"](feat_b)
                        fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                        chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        lam      = model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad))

                        
                        # --- (옵션) BOX 롤링 적용: lam, y_b, len_b 모두 변환 ---
                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, len_b = rolling_sum_nhpp(
                                lam, y_b, len_b, width=args.label_roll_width
                            )
                        mu_b = lam * len_b 
                        # -------------------------------------------------------

                        for i, seg in enumerate(b["raw_segments"]):
                            L = seg["cls_array"].shape[0]
                            res[seg["global_idx"]] = float((y_b[i, :L] - mu_b[i, :L]).mean().item())

                rs     = np.array(list(res.values()), dtype=np.float64)
                scale  = compute_mad(rs) if args.use_mad else compute_iqr(rs)
                delta  = max(args.huber_factor * scale, 1e-9)
                w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

                # 2) bin-NLL → per-seg/kb×30 → 세그 허버가중 평균
                tot_num, tot_den = 0.0, 0.0
                with torch.no_grad():
                    for b in loader:
                        cls_b  = b["cls_array"].to(device)
                        feat_b = b["feat_array"].to(device)
                        y_b    = b["y_array"].to(device)
                        len_b  = b["length_array"].to(device)
                        cid_b  = b["chrom_id"].to(device)
                        
                        key_pad = (len_b <= 0)

                        feat_emb = model_components["feature_embedder"](feat_b)
                        fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                        chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        lam      = model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad))
                        # --- (옵션) BOX 롤링 적용(평가도 동일) ---
                        if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
                            lam, y_b, len_b = rolling_sum_nhpp(
                                lam, y_b, len_b, width=args.label_roll_width
                            )
                        # ----------------------------------------
                        

                        lam_safe = lam.clamp(1e-9, 1e4)
                        sum_log  = (y_b * torch.log(lam_safe)).sum(dim=1)          # (B,)
                        integ    = (lam_safe * len_b).sum(dim=1)                    # (B,)
                        neg_ll   = -(sum_log - integ)                               # (B,)
                        seg_len  = (len_b.sum(dim=1) + 1e-9)                        # (B,)
                        per_seg  = (neg_ll / seg_len) * 30.0                        # (B,)

                        ids = [seg["global_idx"] for seg in b["raw_segments"]]
                        w   = torch.tensor([w_dict.get(i, 1.0) for i in ids],
                                        device=per_seg.device, dtype=per_seg.dtype)

                        tot_num += float((w * per_seg).sum().item())
                        tot_den += float(w.sum().item())

                return tot_num / max(tot_den, 1e-9)
            nh = unwrap(model_components["nhpp_head"])
            with torch.no_grad():
                _logc_backup = nh.log_c.detach().clone()

            calibrate_log_c_huber_like_training(
                model_components, train_loader_infer, device,
                huber_factor=args.huber_factor, use_mad=args.use_mad,
                label_roll=args.label_roll, roll_width=args.label_roll_width
            )
            
            train_loss = eval_loader(train_loader_infer)
            val_loss = eval_loader(val_loader)
            train_hist.append(train_loss); val_hist.append(val_loss)
            print(f"[Epoch {epoch}] Train_loss = {train_loss:.4f}, Val_loss = {val_loss:.4f}", flush=True)

            # ----------- Huber 가중치 갱신 (seg_weight_dict) --------------- #
            def compute_segment_residual():
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in train_loader_infer:
                        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                        y_b    = b["y_array"].to(device) 
                        cid_b = b["chrom_id"].to(device)
                        len_b  = b["length_array"].to(device)
                        seg_ids = [seg["global_idx"] for seg in b["raw_segments"]]
                        feat_emb = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat_emb)
                        chr_emb = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        key_pad  = (len_b <= 0)
                        lam      = model_components["nhpp_head"](
                            model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                        )
                        
                        
                        # NHPP-정합 롤링(합) 적용: λ, y를 동일 윈도우로 변환
                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, _ = rolling_sum_nhpp(
                                lam, y_b, len_b, width=args.label_roll_width
                            )
                            
                        mu_b = lam * len_b 

                        y_np = y_b.cpu().numpy()
                        mu_np  = mu_b.cpu().numpy()
                        for i, seg_id in enumerate(seg_ids):
                            valid = b["raw_segments"][i]["cls_array"].shape[0]
                            res[seg_id] = float((y_np[i, :valid] - mu_np[i, :valid]).mean())
                return res

            seg_res = compute_segment_residual()
            rs = np.array(list(seg_res.values()))
            delta = max(args.huber_factor * (compute_mad(rs) if args.use_mad else compute_iqr(rs)), 1e-9)
            seg_weight_dict = {sid: huber_weight(r, delta) for sid, r in seg_res.items()}

            # ---- residual-weighted sampler warm‑up ----------------------- #
            if epoch == 3 and not use_rw_sampler:
                use_rw_sampler = True
                def build_sampler(resid_dict, tau, alpha, beta):
                    abs_r = np.array([abs(resid_dict.get(seg["global_idx"], 0.0))
                                      for seg in ds_train.segments], np.float64)
                    len_kb = np.array([(seg["end_array"][-1] - seg["start_array"][0] + 1) / 1000.0
                                       for seg in ds_train.segments], np.float64)
                    p = np.exp(-beta*abs_r / (tau + 1e-9)) * (len_kb ** alpha); p /= p.sum()
                    return WeightedRandomSampler(torch.DoubleTensor(p), len(ds_train), replacement=True)
                train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                          sampler=build_sampler(seg_res, tau, args.len_alpha, args.res_beta),
                                          collate_fn=collate_train, num_workers=args.num_data_workers,
                                          pin_memory=pin)
                print(f"[INFO] Residual-weighted sampler enabled at epoch {epoch}", flush=True)
            elif use_rw_sampler:
                tau *= 0.999
                sampler = build_sampler(seg_res, tau, args.len_alpha, args.res_beta)
                train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                          sampler=sampler, collate_fn=collate_train,
                                          num_workers=args.num_data_workers, pin_memory=pin)

            # ---------------- Val‑loss 개선 여부 -------------------------- #
            improved = val_loss < best_val_loss - best_val_loss * args.min_delta_pct / 100.0 \
                       if best_val_loss != float("inf") else True
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0

                best_state_dict = {k: model_components[k].state_dict()
                                   for k in model_components}

                ckpt_common = {**best_state_dict,
                               "epoch": epoch,
                               "optimizer_state": optimizer.state_dict(),
                               "sched_state": scheduler.state_dict(),
                               "best_val_loss": best_val_loss}

                if args.save_each_best:
                    ep_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
                    efficient_save_ckpt(ep_path, **ckpt_common)
                efficient_save_ckpt(model_save_path, **ckpt_common)

                print(f"[INFO] New best Val_loss = {best_val_loss:.4f}"
                      + (f" & {ep_path}" if args.save_each_best else ""),
                      flush=True)
 

                if args.save_attention:
                    save_last_layer_attention(
                        unwrap(model_components["global_transformer"]),
                                              epoch, step_global, out_dir)
                    torch.cuda.empty_cache()       
              
            else:
                epochs_no_improve += 1
            with torch.no_grad():
                nh.log_c.copy_(_logc_backup)

            if args.early_stop and epochs_no_improve >= args.patience:
                print("[INFO] Early stopping: patience exhausted", flush=True)
                break

        # ---------- 학습곡선 저장 --------------------------------------- #
        plt.figure()
        rng = range(len(train_hist))
        plt.plot(rng, train_hist, marker="o", label="train")
        plt.plot(rng, val_hist, marker="x", label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(out_dir, "train_val_loss_per_epoch.png")); plt.close()

    # ------------------------------------------------------------------- #
    # Final evaluation & attention full‑dump                               #
    # ------------------------------------------------------------------- #
    print("[INFO] Loading best model for final evaluation ..", flush=True)
    best_ckpt = efficient_load_ckpt(model_save_path)
    for k in model_components:
        model_components[k].load_state_dict(best_ckpt[k], strict=False)
    del best_ckpt; gc.collect(); torch.cuda.empty_cache()
    
    calibrate_log_c_huber_like_training(
        model_components,
        all_loader,              # 또는 val_loader / train_loader_infer
        device,
        huber_factor=args.huber_factor,
        use_mad=args.use_mad,
        label_roll=args.label_roll, 
        roll_width=args.label_roll_width
    )
    
    def evaluate(loader):
        # 평가용 헬퍼
        for m in model_components.values():
            m.eval()

        res = {}
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = b["y_array"].to(device)
                cid_b  = b["chrom_id"].to(device)
                len_b  = b["length_array"].to(device)  
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                lam      = model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad))

                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = rolling_sum_nhpp(
                    lam, y_b, len_b,
                    width=args.label_roll_width
                )
                mu_b = lam * len_b

                for i, seg in enumerate(b["raw_segments"]):
                    L = seg["cls_array"].shape[0]
                    res[seg["global_idx"]] = float((y_b[i, :L] - mu_b[i, :L]).mean().item())

        rs     = np.array(list(res.values()), dtype=np.float64)
        scale  = compute_mad(rs) if args.use_mad else compute_iqr(rs)
        delta  = max(args.huber_factor * scale, 1e-9)
        w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

        # 2) bin-NLL → per-seg/kb×30 → 세그 허버가중 평균
        tot_num, tot_den = 0.0, 0.0
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = b["y_array"].to(device)
                len_b  = b["length_array"].to(device)
                cid_b  = b["chrom_id"].to(device)
                key_pad  = (len_b <= 0) 

                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

                lam      = model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad))
                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)

                lam_safe = lam.clamp(1e-9, 1e4)
                sum_log  = (y_b * torch.log(lam_safe)).sum(dim=1)     # (B,)
                integ    = (lam_safe * len_b).sum(dim=1)               # (B,)
                neg_ll   = -(sum_log - integ)                          # (B,)
                seg_len  = (len_b.sum(dim=1) + 1e-9)                   # (B,)
                per_seg  = (neg_ll / seg_len) * 30.0                   # (B,)

                ids = [seg["global_idx"] for seg in b["raw_segments"]]
                w   = torch.tensor([w_dict.get(i, 1.0) for i in ids],
                               device=per_seg.device, dtype=per_seg.dtype)

                tot_num += float((w * per_seg).sum().item())
                tot_den += float(w.sum().item())

        return tot_num / max(tot_den, 1e-9)


    print(f"final Train_loss = {evaluate(train_loader_infer):.4f}", flush=True)
    print(f"final Val_loss   = {evaluate(val_loader):.4f}", flush=True)

    # ----------- 예측 CSV 생성 (train / val / all) ----------------------- #
    @torch.no_grad()
    def predict_and_save(loader, name):
        # eval 모드
        for m in model_components.values():
            m.eval()

        rows, tot_ll, tot_bin = [], 0.0, 0
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = b["y_array"].to(device)
                len_b  = b["length_array"].to(device)
                s_bp   = b["start_array"].cpu().numpy()
                e_bp   = b["end_array"].cpu().numpy()
                cid_b  = b["chrom_id"].to(device)
                key_pad  = (len_b <= 0)  # pad mask

                # forward
                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                lam      = model_components["nhpp_head"](
                    model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                ).clamp(1e-9, 1e4)  # per-bin rate (원본)

                # --- lam 계산 직후에 추가 ---
                # 커널 스무딩이 켜졌다면 예측도 같은 커널로 스무딩된 1kb당 rate로 저장
                lam_save = lam
                if args.label_roll and args.label_roll_width > 1:
                    lam_r, _y_ignore, _dt_ignore = rolling_sum_nhpp(
                        lam, y_b, len_b, width=args.label_roll_width
                    )
                    lam_save = lam_r  # 저장용 lam은 스무딩된 1kb당 rate

                # numpy 변환
                lam_np_raw  = lam.detach().cpu().numpy()       # bin_loglike 계산용(원본)
                lam_np      = lam_save.detach().cpu().numpy()  # 저장용 lam_pred(스무딩)
                y_np        = y_b.detach().cpu().numpy()
                len_np      = len_b.detach().cpu().numpy()

                B, T = lam_np.shape  # 배치 내 공통 패딩 길이

                for i, seg in enumerate(b["raw_segments"]):
                    # 세그 실제 길이
                    L = seg["cls_array"].shape[0]

                    # 안전 길이: 모든 배열 차원을 최소값으로 맞춤
                    L_eff = min(
                        L, T,
                        y_np.shape[1], len_np.shape[1],
                        s_bp.shape[1], e_bp.shape[1]
                    )
                    if L_eff <= 0:
                        continue

                    # 슬라이스(길이 L_eff)
                    lam_i      = lam_np[i,      :L_eff]          # 저장용 (스무딩된 1kb당 lam)
                    lam_raw_i  = lam_np_raw[i,  :L_eff]          # 로그우도 계산용 (원본 lam)
                    y_i        = y_np[i,       :L_eff]
                    dt_i       = len_np[i,     :L_eff]
                    s_i        = s_bp[i,       :L_eff]
                    e_i        = e_bp[i,       :L_eff]

                    # bin 로그우도 (원본 lam으로 계산하는 게 정합함)
                    lam_raw_i_safe = np.clip(lam_raw_i, 1e-9, 1e4)
                    llb_i = y_i * np.log(lam_raw_i_safe) - lam_raw_i_safe * dt_i

                    # 행 추가
                    chrom = seg["chrom"]
                    for j in range(L_eff):
                        rows.append(dict(
                            chrom=chrom,
                            start=int(s_i[j]),
                            end=int(e_i[j]),
                            lam_pred=float(lam_i[j]),      # ← 스무딩된 1kb당 rate 저장
                            obs_count=float(y_i[j]),       # ← 원본 관측 카운트 저장
                            bin_loglike=float(llb_i[j]),   # ← 원본 lam 기반 로그우도
                        ))
                        tot_ll  += llb_i[j]
                        tot_bin += 1

        df = pd.DataFrame(rows).sort_values(["chrom", "start"])
        out_file = out_dir / f"{name}_prediction.csv"
        df.to_csv(out_file, index=False)
        print(f"[INFO] {name} saved → {out_file} (bins={len(df)})", flush=True)

  

    # 그대로 호출 (변경 없음)
    predict_and_save(train_loader_infer, "train")
    predict_and_save(val_loader, "val")
    predict_and_save(all_loader, "all")

    if args.save_attention:
        dump_full_attention(model_components, all_loader, device,
                            out_dir / "attn" / "final_full_attention.pt")

    print("[DONE] Training/validation complete, predictions & attentions saved.", flush=True)

    return (out_dir / "all_prediction.csv").as_posix()

    
# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description="NHPP robust training with Huber loss")
    # 필수 입력
    # build_argparser()
    p.add_argument("--cls-file", required=False, help="Pickle with CLS vectors")
    p.add_argument("--feat-file", required=False, help="Pickle with feature dict")

    p.add_argument("--mutations-file", type=str, default=None,
                   help="CSV/TSV with columns [chrom,start,end,event_type,sample,(filter)] for events")
    # 출력
    p.add_argument("--out-dir", default="result_robust")
    # Resume
    p.add_argument("--resume-checkpoint", type=str, default=None)
    # Segmentation
    p.add_argument("--segment-lengths", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--discard-leftover", action="store_true")
    p.add_argument("--overlap-factor", type=float, default=0.0)
    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--len-alpha", type=float, default=0.7,
                   help="length exponent α in residual sampler (0~1)")
    p.add_argument("--res-beta", type=float, default=1.0,
                   help="Residual weight scaling β (0 = no residual bias)")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-sched-T0", type=int, default=5000)
    p.add_argument("--lr-sched-Tmult", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--num-data-workers", type=int, default=0)
    # Transformer
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-seq-len", type=int, default=1024)
    # Early stop
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--min-delta-pct", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    # 기타
    p.add_argument("--label-roll", action="store_true",
                   help="Enable NHPP-consistent rolling-sum (box) on y, μ(=λΔt), Δt (for training space)")
    p.add_argument("--label-roll-width", type=int, default=2,
                   help="Rolling window width in bins (>=2 to enable) for training space")

    p.add_argument("--huber-factor", type=float, default=3.0)
    p.add_argument("--use-mad", action="store_true")
    p.add_argument("--cutmix-p", type=float, default=0.2)
    # Attention dump
    p.add_argument("--save-attention", action="store_true",
                   help="save last-layer attention (on Val-improve) and final full stack")
    # 체크포인트
    p.add_argument("--save-each-best", action="store_true")

    # ===== Pipeline toggles =====
    p.add_argument("--run-pipeline", action="store_true",
                   help="Run LLR→GMM→DP pipeline after training using OUT/all_prediction.csv")
    p.add_argument("--pipeline-only", action="store_true",
                   help="Skip training and run pipeline only (requires --all-pred)")

    # ===== Pipeline args =====
    p.add_argument("--all-pred", type=str, default=None,
                   help="Path to all_prediction.csv (used when --pipeline-only or to override)")
    p.add_argument("--pipeline-out-dir", type=str, default=None,
                   help="Where to write pipeline outputs (default: OUT/postproc)")
    # ... build_argparser() 내부, --pipeline-gmm-k 바로 아래에 추가
    p.add_argument(
        "--pipeline-gmm-auto",
        action="store_true",
        help="Global GMM을 BIC로 자동(k) 선택해서 1회 피팅합니다. 지정 시 --pipeline-gmm-k는 무시됩니다."
    )


    p.add_argument("--pipeline-chunk-size",    type=int, default=1_000_000)
    p.add_argument("--pipeline-chunk-overlap", type=int, default=100_000)
    p.add_argument("--pipeline-min-distance",  type=int, default=0)
    p.add_argument("--pipeline-max-distance",  type=int, default=100_000)
    p.add_argument("--pipeline-sample-frac",   type=float, default=0.01)
    p.add_argument("--pipeline-gmm-k",         type=int,   default=3)
    p.add_argument("--pipeline-beta",          type=float, default=1.5)
    p.add_argument("--pipeline-gamma",         type=float, default=0.5)
    p.add_argument("--pipeline-seed",          type=int,   default=None,
                   help="Seed for pipeline (default: trainer --seed)")

    # DP
    p.add_argument("--pipeline-dp-gap-bp",     type=int,   default=0,
                   help="Minimum gap (bp) required between DP-selected intervals (0 keeps original behavior).")

    # ===== Post-selection refit controls =====
    p.add_argument("--postsel-gmm-k-auto", action="store_true",
                   help="선택집합 재피팅 시 GMM k를 BIC로 자동선택")
    p.add_argument("--postsel-gmm-k-max", type=int, default=3)
    p.add_argument("--postsel-gmm-n-init", type=int, default=3)
    p.add_argument("--postsel-gmm-max-iter", type=int, default=500)
    p.add_argument("--postsel-gmm-k", type=int, default=None,
                   help="자동미사용 시 고정 k; 미지정시 전역 gmm_k")

    p.add_argument("--postsel-fdr-method", choices=["storey","bh"], default="storey")
    p.add_argument("--postsel-bootstrap", type=int, default=200)
    p.add_argument("--postsel-lambda-start", type=float, default=0.05)
    p.add_argument("--postsel-lambda-end",   type=float, default=0.95)
    p.add_argument("--postsel-lambda-step",  type=float, default=0.01)
    p.add_argument("--postsel-pi0-floor",    type=float, default=0.01)
    p.add_argument("--postsel-pi0-ceil",     type=float, default=1.0)


    # NEW: Presmoothing W (bins). None/<=1 이면 자동(학습 상속) 또는 off.
    p.add_argument("--pipeline-presmooth-bins", type=int, default=None,
                   help="Presmoothing box width in bins for y and λ·Δt before variable scan "
                        "(None/<=1 = inherit label_roll_width if >1 else off).")

    return p

def _resolve_scan_params_from_training(args):
    """
    학습에서 label_roll을 사용했다면 같은 W로 롤링-스캔하도록 자동 설정.
    사용자가 pipeline-scan-mode/roll-*를 명시로 준 경우 그 값을 우선한다.
    """
    # 사용자가 명시로 rolling을 지정
    if getattr(args, "pipeline_scan_mode", "variable") == "rolling":
        return ("rolling",
                int(getattr(args, "pipeline_roll_width_bins", 11)),
                int(getattr(args, "pipeline_roll_stride_bins", 1)))

    # 학습에서 label_roll 사용(W>1) → 동일 W로 rolling 스캔 상속
    if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
        return ("rolling", int(args.label_roll_width), 1)

    # 기본: variable 스캔
    return ("variable",
            int(getattr(args, "pipeline_roll_width_bins", 11)),
            int(getattr(args, "pipeline_roll_stride_bins", 1)))

def main():
    pa = build_argparser()
    args = pa.parse_args()

    # Decide mode
    if args.pipeline_only:
        if not args.all_pred:
            raise ValueError("--pipeline-only requires --all-pred to be set")
        pipeline_out_dir = Path(args.pipeline_out_dir or (Path(args.all_pred).parent / "postproc"))
        seed = args.pipeline_seed if args.pipeline_seed is not None else args.seed

        # NEW: presmooth W 자동 상속/결정
        presmooth_w = _resolve_presmooth_bins(args)
        print(f"[CFG] presmooth_bins={presmooth_w}  (label_roll_width={getattr(args,'label_roll_width',None)})", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            
        run_llr_gmm_dp_pipeline(
            all_pred_path=args.all_pred,
            out_dir=pipeline_out_dir,
            chunk_size=args.pipeline_chunk_size,
            chunk_overlap=args.pipeline_chunk_overlap,
            min_distance=args.pipeline_min_distance,
            max_distance=args.pipeline_max_distance,
            sample_frac=args.pipeline_sample_frac,
            gmm_k=args.pipeline_gmm_k,
            beta=args.pipeline_beta,
            gamma=args.pipeline_gamma,
            seed=seed,
            dp_gap_bp=args.pipeline_dp_gap_bp,
            events_file=args.mutations_file,
            events_use_midpoint=True,
            events_dedup_sample=True,
            presmooth_bins=presmooth_w,
            # post-selection options 그대로 전달
            gmm_auto=bool(args.pipeline_gmm_auto),
            postsel_gmm_k_auto=args.postsel_gmm_k_auto,
            postsel_gmm_k_max=args.postsel_gmm_k_max,
            postsel_gmm_n_init=args.postsel_gmm_n_init,
            postsel_gmm_max_iter=args.postsel_gmm_max_iter,
            postsel_gmm_k=args.postsel_gmm_k,
            postsel_fdr_method=args.postsel_fdr_method,
            postsel_bootstrap=args.postsel_bootstrap,
            postsel_lambda_start=args.postsel_lambda_start,
            postsel_lambda_end=args.postsel_lambda_end,
            postsel_lambda_step=args.postsel_lambda_step,
            postsel_pi0_floor=args.postsel_pi0_floor,
            postsel_pi0_ceil=args.postsel_pi0_ceil,
        )
        return

    # Training path
    if not args.cls_file or not args.feat_file:
            raise ValueError("--cls-file and --feat-file are required unless --pipeline-only is set")

    all_pred_path = train_and_predict(args)

    # Optional pipeline right after training
    if args.run_pipeline or args.pipeline_out_dir or args.all_pred:
        post_all_pred = args.all_pred or all_pred_path
        post_out_dir  = Path(args.pipeline_out_dir or (Path(args.out_dir) / "postproc"))
        seed = args.pipeline_seed if args.pipeline_seed is not None else args.seed

        presmooth_w = _resolve_presmooth_bins(args)
        print(f"[CFG] presmooth_bins={presmooth_w}  (label_roll_width={getattr(args,'label_roll_width',None)})", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

        run_llr_gmm_dp_pipeline(
            all_pred_path=post_all_pred,
            out_dir=post_out_dir,
            chunk_size=args.pipeline_chunk_size,
            chunk_overlap=args.pipeline_chunk_overlap,
            min_distance=args.pipeline_min_distance,
            max_distance=args.pipeline_max_distance,
            sample_frac=args.pipeline_sample_frac,
            gmm_k=args.pipeline_gmm_k,
            beta=args.pipeline_beta,
            gamma=args.pipeline_gamma,
            seed=seed,
            dp_gap_bp=args.pipeline_dp_gap_bp,
            events_file=args.mutations_file,
            events_use_midpoint=True,
            events_dedup_sample=True,
            presmooth_bins=presmooth_w,
            # post-selection options
            gmm_auto=bool(args.pipeline_gmm_auto),
            postsel_gmm_k_auto=args.postsel_gmm_k_auto,
            postsel_gmm_k_max=args.postsel_gmm_k_max,
            postsel_gmm_n_init=args.postsel_gmm_n_init,
            postsel_gmm_max_iter=args.postsel_gmm_max_iter,
            postsel_gmm_k=args.postsel_gmm_k,
            postsel_fdr_method=args.postsel_fdr_method,
            postsel_bootstrap=args.postsel_bootstrap,
            postsel_lambda_start=args.postsel_lambda_start,
            postsel_lambda_end=args.postsel_lambda_end,
            postsel_lambda_step=args.postsel_lambda_step,
            postsel_pi0_floor=args.postsel_pi0_floor,
            postsel_pi0_ceil=args.postsel_pi0_ceil,
        )


# ────────────────────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


