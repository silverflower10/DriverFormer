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

import argparse

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
