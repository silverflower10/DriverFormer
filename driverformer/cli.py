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

from pathlib import Path
import warnings
from .config import build_argparser
from .train.loop import train_and_predict
from .pipeline.run import run_llr_gmm_dp_pipeline, _resolve_scan_params_from_training, _resolve_presmooth_bins

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
