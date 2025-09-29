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

import os, math, random, warnings
import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from ..pipeline.llr_scan import make_per_bin, _build_chunks, _psums, _llr, _scan_chunk, _presmooth_nhpp_numpy
from ..pipeline.gmm import fit_gmm, fit_gmm_auto, mix_neglog10p_from_gmm
from ..pipeline.dp import dp_select
from ..pipeline.postsel import estimate_pi0_storey_bootstrap, qvalues_storey, qvalues_bh
from ..utils.io import _set_seed
from ..pipeline.events import _load_events_for_dp, _summarize_events_for_intervals
from ..utils.chrom import _chr_key as chr_key
from ..utils.plotting import qq_plot
from ..utils.chrom import _chr_key

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


def _resolve_presmooth_bins(args) -> int:
    pb = getattr(args, "pipeline_presmooth_bins", None)
    if pb is not None:
        # 사용자가 준 값을 그대로 사용 (0/1 ⇒ off)
        return max(0, int(pb))
    if getattr(args, "label_roll", False) and getattr(args, "label_roll_width", 1) > 1:
        return int(args.label_roll_width)
    return 1
