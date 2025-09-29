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

from ..utils.chrom import _apply_chr_norm, _norm_chr, CANON_CHROMS

def _bins_from_cls_list(cls_list):
    rows = []
    for chrom, w_idx, sbp, ebp, cls_vec, y_val in cls_list:
        rows.append((str(chrom), int(sbp), int(ebp)))
    df = pd.DataFrame(rows, columns=["chrom","start","end"]).drop_duplicates()
    df = _apply_chr_norm(df, "chrom").sort_values(["chrom","start"]).reset_index(drop=True)
    return df

def _infer_bin_size_and_anchor_from_bins(cls_bins: pd.DataFrame) -> tuple[int, int]:
    """CLS bin으로부터 bin_size, anchor(start%bin_size 최빈값) 추정"""
    lens = (cls_bins["end"] - cls_bins["start"] + 1)
    bin_size = int(lens.mode().iat[0])
    anchor   = int((cls_bins["start"] % bin_size).mode().iat[0])
    return bin_size, anchor

def _guess_sample_col(df: pd.DataFrame) -> str:
    """sample 컬럼명이 제각각일 때 자동 감지."""
    lower = {c.lower(): c for c in df.columns}
    # 1) 흔한 이름 우선
    for k in ["sample","sample_id","donor","donor_id",
              "tumor","tumour","tumor_id","vcf","file","tumor_sample"]:
        if k in lower:
            return lower[k]
    # 2) 내용 패턴으로 추정(값에 .vcf / purple / CPCT / WIDE / OBC 포함)
    cand = []
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            hits = s.str.contains(r"(purple|\.vcf|CPCT|WIDE|OBC)",
                                  case=False, regex=True, na=False).mean()
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
        raw = pd.read_csv(path, low_memory=False, dtype=str)
    except Exception:
        raw = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)

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
