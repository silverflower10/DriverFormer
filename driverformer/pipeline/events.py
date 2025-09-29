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

from tqdm import tqdm

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
