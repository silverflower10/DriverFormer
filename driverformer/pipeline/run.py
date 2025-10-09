#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025    ← NEW  (attention-save policy + NaN guards & debug)

@author: silverflo

Changes in this revision
------------------------
* 학습(epoch) 중 : Val-loss 개선 시 마지막 레이어 head-mean 1장 저장
* 학습 완료(final): 베스트 모델로 전체 레이어·헤드 스택 저장
* LLR→GMM→DP 구간에 NaN/Inf 방지 가드와 디버그 로그 대폭 강화
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

# ── project utils (존재 가정) ──
from ..pipeline.llr_scan import make_per_bin, _build_chunks, _psums, _llr, _scan_chunk, _presmooth_nhpp_numpy
from ..pipeline.gmm import fit_gmm, fit_gmm_auto, mix_neglog10p_from_gmm
from ..pipeline.dp import dp_select
from ..pipeline.postsel import estimate_pi0_storey_bootstrap, qvalues_storey, qvalues_bh
from ..utils.io import _set_seed
from ..pipeline.events import _load_events_for_dp, _summarize_events_for_intervals
from ..utils.chrom import _chr_key as chr_key
from ..utils.plotting import qq_plot
from ..utils.chrom import _chr_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
DEBUG_NAN = True  # NaN 디버깅

# --------------------------------------------------------------------------- #
# helpers: NaN/Inf guards                                                     #
# --------------------------------------------------------------------------- #
_EPS = 1e-8

def _dbg(msg: str):
    print(f"[DBG] {msg}", flush=True)

def _np_isfinite_all(name, arr):
    ok = np.isfinite(arr).all()
    if not ok:
        bad = np.where(~np.isfinite(arr))[0][:10]
        print(f"[WARN] non-finite in {name} (first idx): {bad}", flush=True)
    return ok

def _safe_div(n, d, eps=_EPS):
    d = np.clip(d, eps, None)
    return n / d

def _safe_log_np(x, eps=_EPS):
    return np.log(np.clip(x, eps, None))

def _sanitize_series_positive(s, name, minv=_EPS, clip_max=None):
    a = s.to_numpy(float)
    a = np.nan_to_num(a, nan=minv, posinf=(clip_max or 1e12), neginf=minv)
    a = np.clip(a, minv, (clip_max or np.inf))
    _np_isfinite_all(name, a)
    return a

def _sanitize_series_any(s, name, repl0=0.0):
    a = s.to_numpy(float)
    a = np.nan_to_num(a, nan=repl0, posinf=repl0, neginf=repl0)
    _np_isfinite_all(name, a)
    return a

def _safe_mix_neglog10p(llr_norm_pos, gmm, min_p=1e-300):
    """mix_neglog10p_from_gmm wrapper → 항상 finite 반환"""
    if llr_norm_pos.size == 0:
        return np.array([], float), np.array([], float)
    nl10, pv = mix_neglog10p_from_gmm(llr_norm_pos, gmm, min_p=min_p)
    nl10 = np.nan_to_num(nl10, nan=0.0, posinf=0.0, neginf=0.0)
    pv   = np.nan_to_num(pv,   nan=1.0, posinf=1.0, neginf=1.0)
    pv   = np.clip(pv, min_p, 1.0)
    return nl10, pv

def _safe_qq_plot(p, title, out_pdf: Path):
    try:
        if p.size and np.isfinite(p).any():
            qq_plot(p, title, out_pdf)
        else:
            out_pdf.write_bytes(b"")
    except Exception as e:
        print(f"[WARN] qq_plot skipped: {e}", flush=True)
        out_pdf.write_bytes(b"")

# --------------------------------------------------------------------------- #
# LLR → GMM → DP (NaN-safe)                                                   #
# --------------------------------------------------------------------------- #
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
                            gmm_auto: bool     = False,
                            # === post-selection ===
                            postsel_gmm_k_auto: bool = True,      # (사용 안함; 전역 GMM 재사용)
                            postsel_gmm_k_max: int = 6,           # (호환용)
                            postsel_gmm_n_init: int = 3,
                            postsel_gmm_max_iter: int = 500,
                            postsel_gmm_k: Optional[int] = None,  # (호환용)
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
    4) 선택집합 p (전역 GMM 재사용) → 5) FDR(q)
    * 모든 단계에 NaN/Inf 방지 가드를 삽입
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

    # ── 기본 정리/검증 ──
    # 필수 컬럼 체크
    for col in ["chrom", "start", "end", "lam_pred", "obs_count"]:
        if col not in bin_df.columns:
            raise RuntimeError(f"per-bin csv missing column: {col}")

    bin_df["chrom"] = bin_df["chrom"].astype(str)

    # lam_pred, obs_count 정리 (양수/유한)
    bin_df["lam_pred"]  = _sanitize_series_positive(bin_df["lam_pred"], "lam_pred", minv=1e-10, clip_max=1e8)
    bin_df["obs_count"] = np.clip(_sanitize_series_any(bin_df["obs_count"], "obs_count", repl0=0.0), 0.0, 1e12)

    # 길이(bp) 0/음수 방지
    lens = (bin_df["end"].to_numpy(np.int64) - bin_df["start"].to_numpy(np.int64) + 1)
    lens = np.clip(lens, 1, None)
    bin_df["len_bp"] = lens
    bin_df["len_kb"] = np.maximum(bin_df["len_bp"].to_numpy(float) / 1_000.0, _EPS)

    # 크롬 정렬
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

            # presmooth (필요시)
            if presmooth_bins and presmooth_bins > 1:
                lam, obs, st, en = _presmooth_nhpp_numpy(lam, obs, st, en, W_bins=presmooth_bins)

            # 안전성 점검
            lam = np.clip(np.nan_to_num(lam, nan=1e-10, posinf=1e8, neginf=1e-10), 1e-10, 1e8)
            obs = np.clip(np.nan_to_num(obs, nan=0.0,   posinf=1e12, neginf=0.0),   0.0,   1e12)

            ivs = []
            for ck in _build_chunks(lam, obs, st, en, chunk_size, chunk_overlap):
                ivs.extend(_scan_chunk(ck, min_distance, max_distance))

            for iv in ivs:
                LLR_raw = float(np.nan_to_num(iv["LLR_raw"], nan=0.0, posinf=0.0, neginf=0.0))
                len_bp  = int(max(iv["len_bp"], 1))
                w.write(f"{ch},{iv['start_bp']},{iv['end_bp']},{len_bp},{LLR_raw}\n")
                if LLR_raw > 1e-6:
                    n_pos += 1
                    len_kb = max(len_bp / 1_000.0, _EPS)
                    if random.random() < sample_frac:
                        pos_samp.append(LLR_raw / math.sqrt(len_kb))
                else:
                    n_zero += 1

    print(f"[1] interval csv saved   (+LLR={n_pos:,}, zero≈{n_zero:,})", flush=True)

    # 2) global GMM
    pos_arr = np.asarray(pos_samp, dtype=float)
    pos_arr = pos_arr[np.isfinite(pos_arr)]
    if pos_arr.size == 0:
        print("[WARN] GMM sample empty; fallback to small positives.", flush=True)
        pos_arr = np.array([0.05, 0.10, 0.20], dtype=float)

    core = pos_arr  # trimming 없음

    if gmm_auto:
        try:
            gmm = fit_gmm_auto(core, k_min=1, k_max=max(3, gmm_k), seed=seed, n_init=3, max_iter=500)
            if not len(gmm.get("w", [])):
                raise RuntimeError("Empty GMM after auto-fit")
            mode_str = "auto"
        except Exception as e:
            print(f"[WARN] GMM auto failed → fallback to k={int(gmm_k)} ({e})", flush=True)
            gmm = fit_gmm(core, k=int(gmm_k), seed=seed)
            mode_str = f"auto→fallback(k={int(gmm_k)})"
    else:
        gmm = fit_gmm(core, k=int(gmm_k), seed=seed)
        mode_str = f"fixed(k={int(gmm_k)})"

    print(f"[2] GMM mode={mode_str}, k={len(gmm['w'])}, π={np.round(gmm['w'],3)}", flush=True)

    # 3) p / weight / DP
    print("[3] DP selection …", flush=True)
    llr_df = pd.read_csv(llr_csv, dtype={"chrom": "category"})
    llr_df["len_bp"] = np.maximum(llr_df["len_bp"].to_numpy(int), 1)

    sel = []
    for ch in chroms:
        iv_chr = llr_df[llr_df["chrom"] == ch].copy()
        if iv_chr.empty:
            continue

        llr   = _sanitize_series_any(iv_chr["LLR_raw"], "LLR_raw", repl0=0.0)
        len_kb = np.maximum(iv_chr["len_bp"].to_numpy(float)/1_000.0, _EPS)
        llr_norm = _safe_div(llr, np.sqrt(len_kb), eps=_EPS)

        # 기본값
        p = np.ones_like(llr, float)
        neglogp = np.zeros_like(llr, float)

        mask = (llr > 0)
        if mask.any():
            nl10, pv = _safe_mix_neglog10p(llr_norm[mask], gmm, min_p=1e-300)
            neglogp[mask] = nl10
            p[mask]       = pv

        wgt = llr * np.power(neglogp, beta) * np.power(len_kb, gamma)
        iv_chr["p_val"]        = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 1e-300, 1.0)
        iv_chr["neglog10_p"]   = np.nan_to_num(neglogp, nan=0.0, posinf=0.0, neginf=0.0)
        iv_chr["LLR_weighted"] = np.nan_to_num(wgt, nan=0.0, posinf=0.0, neginf=0.0)

        chosen = dp_select(iv_chr.to_dict("records"), gap_bp=dp_gap_bp)
        sel.extend(chosen)
        print(f"    {ch:>4}: intervals={len(iv_chr):7,d} → selected={len(chosen):5,d}", flush=True)

    # 선택없음 처리
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

    # 4) 선택집합 p (전역 GMM 재사용)
    llr_sel     = np.array([iv["LLR_raw"] for iv in sel], float)
    len_kb_sel  = np.array([max((iv["end_bp"] - iv["start_bp"] + 1) / 1_000.0, _EPS) for iv in sel], float)
    llr_norm_sel= _safe_div(llr_sel, np.sqrt(len_kb_sel), eps=_EPS)

    p_post = np.ones_like(llr_sel, float)
    msk = (llr_sel > 0)
    if msk.any():
        _, pv = _safe_mix_neglog10p(llr_norm_sel[msk], gmm, min_p=1e-300)
        p_post[msk] = pv

    # 5) FDR
    method = postsel_fdr_method.lower()
    if method == "storey":
        lam_grid = np.arange(postsel_lambda_start, postsel_lambda_end + 1e-12,
                             postsel_lambda_step, float)
        try:
            pi0_hat, lam_star, _, _ = estimate_pi0_storey_bootstrap(
                p_post, lambdas=lam_grid, B=postsel_bootstrap, seed=seed,
                pi0_floor=postsel_pi0_floor, pi0_ceil=postsel_pi0_ceil
            )
            q_post = qvalues_storey(p_post, float(pi0_hat))
            print(f"[POST] Storey q: pi0={pi0_hat:.3f}, lambda*={lam_star:.2f}, B={postsel_bootstrap}", flush=True)
        except Exception as e:
            print(f"[WARN] Storey failed ({e}) → fallback to BH", flush=True)
            q_post = qvalues_bh(p_post)
    else:
        q_post = qvalues_bh(p_post)
        print(f"[POST] BH q-values computed on selected set (m={len(p_post)})", flush=True)

    # QQ plot (post-selection)
    qq_title = "QQ – Post-selection p (Storey)" if method == "storey" else "QQ – Post-selection p (BH)"
    _safe_qq_plot(p_post, qq_title, out_dir / "qq_plot.pdf")

    # obs/exp 집계 + CSV 저장
    by_chr = {c: d for c, d in bin_df.groupby("chrom", observed=True)}
    rows = []
    for idx, iv in enumerate(sel):
        ch, s, e = iv["chrom"], iv["start_bp"], iv["end_bp"]
        sub = by_chr[ch]
        msk2 = (sub["start"] >= s) & (sub["end"] <= e)

        obs_sum = float(np.nan_to_num(sub.loc[msk2, "obs_count"].sum(), nan=0.0, posinf=0.0, neginf=0.0))
        kb      = (sub.loc[msk2, "end"] - sub.loc[msk2, "start"] + 1) / 1_000.0
        kb      = np.maximum(np.nan_to_num(kb.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0), _EPS)
        exp_sum = float(np.nan_to_num((sub.loc[msk2, "lam_pred"] * kb).sum(), nan=0.0, posinf=0.0, neginf=0.0))

        row = dict(chrom=ch, start=s, end=e, len_bp=e - s + 1,
                   LLR_raw=float(np.nan_to_num(iv["LLR_raw"], nan=0.0, posinf=0.0, neginf=0.0)),
                   obs_sum=obs_sum, exp_sum=exp_sum,
                   p_post=float(np.clip(p_post[idx], 1e-300, 1.0)),
                   fdr_post=float(q_post[idx]), fdr=float(q_post[idx]))

        if evt_summaries is not None:
            es = evt_summaries[idx]
            row.update({
                "n_evt_all": es.get("n_evt_all", 0),
                "n_samp_all": es.get("n_samp_all", 0),
                "max_junc_per_sample": es.get("max_junc_per_sample", 0),
                "median_junc_per_sample": es.get("median_junc_per_sample", 0.0),
                "gini_junc": es.get("gini_junc", 0.0),
                "event_type_diversity": es.get("event_type_diversity", 0),
                "frac_TRA": es.get("frac_TRA", 0.0),
            })
            for k, v in es.items():
                if k.startswith("n_evt_") or k.startswith("n_samp_"):
                    row[k] = v
        else:
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
    evt_cols = [c for c in final_df.columns if c.startswith("n_evt_") or c.startswith("n_samp_")]
    if evt_cols:
        final_df[evt_cols] = final_df[evt_cols].fillna(0).astype(int)

    final_csv = out_dir / "final_result.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[✓] final_result.csv → {final_csv}", flush=True)
    return final_csv.as_posix()

# --------------------------------------------------------------------------- #
# scan param resolvers (기존 인터페이스 유지)                                   #
# --------------------------------------------------------------------------- #
def _resolve_scan_params_from_training(args):
    """
    학습에서 label_roll을 사용했다면 같은 W로 롤링-스캔하도록 자동 설정.
    사용자가 pipeline-scan-mode/roll-*를 명시로 준 경우 그 값을 우선한다.
    """
    if getattr(args, "pipeline_scan_mode", "variable") == "rolling":
        return ("rolling",
                int(getattr(args, "pipeline_roll_width_bins", 11)),
                int(getattr(args, "pipeline_roll_stride_bins", 1)))

    if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
        return ("rolling", int(args.label_roll_width), 1)

    return ("variable",
            int(getattr(args, "pipeline_roll_width_bins", 11)),
            int(getattr(args, "pipeline_roll_stride_bins", 1)))

def _resolve_presmooth_bins(args) -> int:
    pb = getattr(args, "pipeline_presmooth_bins", None)
    if pb is not None:
        return max(0, int(pb))
    if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
        return int(args.label_roll_width)
    return 1
