#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025    ← NEW (attention-save policy, NaN guards & dense debug)

* 학습(epoch) 중 : Val-loss 개선 시 마지막 레이어 head-mean 1장 저장
* 학습 완료(final): 베스트 모델로 전체 레이어·헤드 스택 저장
* NaN 방지 가드, 디버그/미니덤프
* GPU backend 안전 모드(Flash/메모리절약 SDPA off, cuDNN deterministic + disabled)
* 모든 lam 계산 전 out.contiguous()
* VERIFY 블록(실제 import 확인) + 안전 롤링 강제 패치(monkey-patch)
* λ-그래디언트 살균 hook (MulBackward0 NaN 전파 차단)
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- project modules -------------------------------------------------------
from ..models.embedders import FeatureEmbedder, FeatClsFusion, ChromosomeEmbedder
from ..models.transformer import GlobalTransformerEncoder
from ..models.nhpp_head import NHPPHead
from ..data.segments import SegmentDataset, segment_cls_embeddings_fixed_lengths, segment_collate_fn
from ..data.labels import (
    _bins_from_cls_list,
    _load_mutations_events,
    _build_y_map_from_mutations,
    _attach_labels_from_y_map,
)
# D) rolling: 함수 import 대신 모듈 import (폴백 패치/검증과 호환)
import driverformer.data.rolling as R
from ..utils.io import (
    build_chrom_id_map, unwrap, efficient_load_ckpt, efficient_save_ckpt,
    check_pretrained_model_exists, set_seed
)
from ..losses.nhpp import trapezoid_nhpp_loss, trapezoid_nhpp_loss_segment_weighted
from ..train.attention import save_last_layer_attention, dump_full_attention
from ..utils.stats import compute_mad, compute_iqr, huber_weight
from ..utils.chrom import CHROM_LIST_24

# ===== Globals ===============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

DEBUG_NAN = True
_EPS       = 1e-8
_RATE_MIN  = 1e-9
_RATE_MAX  = 1e6  # 여유 상한

# ===== A) GPU backend 안전 모드 ============================================
import torch.backends.cudnn as _cudnn
_cudnn.enabled = False            # ★ cuDNN 완전 비활성화(최후 안전)
_cudnn.benchmark = False
_cudnn.deterministic = True
try:
    from torch.backends.cuda import sdp_kernel
    # Flash/메모리절약 SDPA 비활성, math만 사용
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===== NaN/Inf safety helpers ===============================================
def _nan_to_num_(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=_RATE_MAX, neginf=0.0)

def _safe_rate_(lam: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(lam).clamp(_RATE_MIN, _RATE_MAX)

def _safe_len_(dt: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(dt).clamp(min=0.0)

def _safe_count_(y: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(y).clamp(min=0.0)

def _stat(t: torch.Tensor):
    t = _nan_to_num_(t.detach())
    return dict(
        shape=list(t.shape),
        min=float(t.min()),
        max=float(t.max()),
        mean=float(t.mean()),
        n_nan=int((~torch.isfinite(t)).sum().item()),
        n_le0=int((t <= 0).sum().item()),
    )

def _dump_batch_stats(tag: str, lam: torch.Tensor, y: torch.Tensor, dt: torch.Tensor):
    lam = _safe_rate_(lam); y = _safe_count_(y); dt = _safe_len_(dt)
    mu  = lam * dt
    info = {
        "tag": tag,
        "lam": _stat(lam),
        "y":   _stat(y),
        "dt":  _stat(dt),
        "mu":  _stat(mu),
        "y_on_pad": int(((dt <= 0) & (y > 0)).sum().item()),
        "lam_le0":  int((lam <= 0).sum().item()),
    }
    print(f"[DBG] {info}", flush=True)

# ===== Calibration (Huber-like) =============================================
@torch.no_grad()
def calibrate_log_c_huber_like_training(model_c, loader, device,
                                        huber_factor: float = 3.0,
                                        use_mad: bool = False,
                                        label_roll: bool = False,
                                        roll_width: int = 1):
    for m in model_c.values():
        m.eval()

    nh = unwrap(model_c["nhpp_head"])
    c_cur = float(torch.exp(nh.log_c.detach()).cpu())

    # residual 수집
    seg_residual = {}
    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = _safe_count_(b["y_array"].to(device))
        len_b  = _safe_len_(b["length_array"].to(device))
        cid_b  = b["chrom_id"].to(device)

        key_pad = (len_b <= 0)
        feat_emb = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat_emb)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        out      = model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        out      = out.contiguous()
        lam      = _safe_rate_(model_c["nhpp_head"](out))

        if label_roll and roll_width > 1:
            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)
            lam   = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

        mu_b = lam * len_b
        y_np, mu_np = y_b.cpu().numpy(), mu_b.cpu().numpy()
        for i, seg in enumerate(b["raw_segments"]):
            L = seg["cls_array"].shape[0]
            seg_residual[seg["global_idx"]] = float((y_np[i, :L] - mu_np[i, :L]).mean())

    rs = np.array(list(seg_residual.values()), dtype=np.float64)
    scale = compute_mad(rs) if use_mad else compute_iqr(rs)
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    delta = max(huber_factor * scale, 1e-6)

    def _hw(r, d):
        a = abs(r);  return 1.0 if a <= d else d/(a + _EPS)

    w_seg = {sid: _hw(r, delta) for sid, r in seg_residual.items()}

    # c* 추정
    num = den = 0.0
    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = _safe_count_(b["y_array"].to(device))
        len_b  = _safe_len_(b["length_array"].to(device))
        cid_b  = b["chrom_id"].to(device)
        key_pad = (len_b <= 0)

        feat     = model_c["feature_embedder"](feat_b)
        fused    = model_c["feat_cls_fusion"](cls_b, feat)
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        out      = model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
        out      = out.contiguous()
        lam      = _safe_rate_(model_c["nhpp_head"](out))

        if label_roll and roll_width > 1:
            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)
            lam   = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

        base = lam / max(c_cur, _EPS)

        for i, seg in enumerate(b["raw_segments"]):
            L     = seg["cls_array"].shape[0]
            sid   = seg["global_idx"]
            w     = float(w_seg.get(sid, 1.0))
            y_i   = y_b[i, :L]
            dt_i  = len_b[i, :L]
            base_i= base[i, :L]
            num  += w * float(y_i.sum().item())
            den  += w * float((base_i * dt_i).sum().item())

    if not (np.isfinite(num) and np.isfinite(den)) or num <= _EPS or den <= _EPS:
        print(f"[CAL-HUBER-TRAIN] skip (num={num:.3g}, den={den:.3g}) keep c={c_cur:.6g}", flush=True)
        return

    c_star = num / den
    nh.log_c.copy_(torch.tensor(math.log(max(c_star, _EPS)), device=nh.log_c.device))
    print(f"[CAL-HUBER-TRAIN] c_prev={c_cur:.6g} → c_new={c_star:.6g}  ratio={c_star/(c_cur+_EPS):.4f}", flush=True)


# ===== Training & prediction ================================================
def train_and_predict(args):
    set_seed(args.seed)

    # ── CPU 스레드 수 제어 (옵션) ───────────────────────────────────────────
    if getattr(args, "torch_threads", None):
        try:
            torch.set_num_threads(int(args.torch_threads))  # intra-op
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
        discard_leftover=getattr(args, "discard_leftover", False),
        overlap_factor=args.overlap_factor,
    )
    print(f"[INFO] #all_segments = {len(all_segments)}", flush=True)

    # ==== B) VERIFY actually loaded modules (debug, once) ====
    if os.environ.get("DF_VERIFY_IMPORTS", "1") == "1":
        import inspect as _ins
        import driverformer.data.rolling as _R
        import driverformer.models.nhpp_head as _H
        try:
            src_r = _ins.getsource(_R._conv1d_causal_sum)
            print("[WHERE] rolling.py  ->", _R.__file__, flush=True)
            print("[CHECK] cudnn.flags in rolling:", ("cudnn.flags" in src_r), flush=True)
        except Exception as e:
            print("[VERIFY] rolling getsource failed:", e, flush=True)
        try:
            src_h = _ins.getsource(_H.NHPPHead.forward)
            print("[WHERE] nhpp_head.py->", _H.__file__, flush=True)
            print("[CHECK] safe head (clamp & nan_to_num):",
                  ("clamp(" in src_h and "nan_to_num" in src_h), flush=True)
        except Exception as e:
            print("[VERIFY] nhpp_head getsource failed:", e, flush=True)

    # ==== C) monkey-patch fallback (if old rolling is loaded) ====
    try:
        import inspect as _ins
        import driverformer.data.rolling as _R
        import torch.backends.cudnn as cudnn
        src_r = _ins.getsource(_R._conv1d_causal_sum)
        if "cudnn.flags" not in src_r:
            print("[PATCH] rolling: monkey-patch safe conv/rolling", flush=True)

            def _safe_conv1d_causal_sum(x, k):
                K = int(k.size(-1))
                x1 = F.pad(x.unsqueeze(1), (K-1, 0)).contiguous()
                with cudnn.flags(enabled=False, benchmark=False, deterministic=True):
                    y = F.conv1d(x1, k, padding=0)
                return y.squeeze(1).contiguous()

            def _safe_rolling(lam, y, dt, *, width=2):
                if width <= 1: return lam, y, dt
                RATE_MIN, RATE_MAX, DEN_MIN = 1e-9, 1e6, 1e-12
                lam = torch.nan_to_num(lam, nan=0.0, posinf=RATE_MAX, neginf=0.0).clamp(RATE_MIN, RATE_MAX)
                y   = torch.nan_to_num(y,   nan=0.0, posinf=0.0,     neginf=0.0).clamp_min(0.0)
                dt  = torch.nan_to_num(dt,  nan=0.0, posinf=0.0,     neginf=0.0).clamp_min(0.0)
                k   = torch.ones(1,1,int(max(1,width)), device=lam.device, dtype=lam.dtype)
                mu  = lam * dt
                y_r  = _safe_conv1d_causal_sum(y,  k)
                mu_r = _safe_conv1d_causal_sum(mu, k)
                dt_r = _safe_conv1d_causal_sum(dt, k).clamp_min(DEN_MIN)
                lam_r= (mu_r/dt_r).clamp(RATE_MIN, RATE_MAX)
                return lam_r, y_r, dt_r

            _R._conv1d_causal_sum = _safe_conv1d_causal_sum
            _R.rolling_sum_nhpp   = _safe_rolling
    except Exception as e:
        print("[PATCH] monkey-patch skipped:", e, flush=True)

    # ===== 1kb bin 라벨 =====
    if args.mutations_file:
        print(f"[LABEL] building 1kb labels (PASS only): {args.mutations_file}", flush=True)
        cls_bins = _bins_from_cls_list(cls_list)
        ev       = _load_mutations_events(args.mutations_file, use_midpoint=True, require_pass=True)
        y_map    = _build_y_map_from_mutations(cls_bins, ev)
        _attach_labels_from_y_map(all_segments, y_map)
        print(f"[LABEL] y_map bins = {len(y_map):,}", flush=True)

    # ----- split ----- 
    train_segs, val_segs = [], []
    c_map = defaultdict(list)
    for seg in all_segments:
        c_map[seg["chrom"]].append(seg)
    for c, lst in c_map.items():
        random.shuffle(lst)
        n = len(lst); n_train = int(n * 0.9)
        train_segs.extend(lst[:n_train]);  val_segs.extend(lst[n_train:])

    ds_train = SegmentDataset(train_segs)
    ds_val   = SegmentDataset(val_segs)
    ds_all   = SegmentDataset(all_segments)

    chrom_id_map = build_chrom_id_map(None)
    collate_train = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=args.cutmix_p)
    collate_eval  = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=0.0)

    pin = (device.type == "cuda")
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin)
    train_loader_infer = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin)
    all_loader = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin)

    # ------------------------- 모델 --------------------------------------- #
    any_feat = next(iter(feature_dict.values()))
    feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else any_feat.shape[0]

    feat_embedder      = FeatureEmbedder(feature_dim, hidden_dim=args.d_model)
    feat_cls_fusion    = FeatClsFusion(hidden_dim=args.d_model)
    chrom_embedder     = ChromosomeEmbedder(len(CHROM_LIST_24), d=args.d_model)
    global_transformer = GlobalTransformerEncoder(d_model=args.d_model, nhead=args.nhead,
                                                  num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                                                  dropout=args.dropout, max_seq_len=args.max_seq_len)
    nhpp_head          = NHPPHead()

    for m in (feat_embedder, feat_cls_fusion, chrom_embedder, global_transformer, nhpp_head):
        m.to(device)

    model_components = dict(
        feature_embedder=feat_embedder, feat_cls_fusion=feat_cls_fusion,
        chrom_embedder=chrom_embedder, global_transformer=global_transformer,
        nhpp_head=nhpp_head
    )

    if n_gpu > 1:
        print(f"[INFO] Using DataParallel on {n_gpu} GPUs", flush=True)
        for k in model_components:
            model_components[k] = nn.DataParallel(model_components[k])

    # ----- Optimizer/Scheduler ----- 
    param_groups = [
        {"params": itertools.chain(
            model_components["feature_embedder"].parameters(),
            model_components["feat_cls_fusion"].parameters(),
            model_components["chrom_embedder"].parameters(),
            model_components["global_transformer"].parameters(),
        ), "weight_decay": 1e-3, "lr": args.lr},
        {"params": model_components["nhpp_head"].parameters(), "weight_decay": 0.0, "lr": args.lr},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_sched_T0,
                                            T_mult=args.lr_sched_Tmult, eta_min=args.lr*0.3)

    # ----- Checkpoint ----- 
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = out_dir / "trained_model.pt"

    start_epoch, train_flag = 0, True
    best_val_loss = float("inf")

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        ckpt = efficient_load_ckpt(args.resume_checkpoint)
        for k in model_components: model_components[k].load_state_dict(ckpt[k], strict=False)
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "sched_state"     in ckpt: scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch   = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        del ckpt; gc.collect()
        print(f"[INFO] Resumed from {args.resume_checkpoint} (epoch {start_epoch})", flush=True)
    elif check_pretrained_model_exists(model_save_path):
        ckpt = efficient_load_ckpt(model_save_path)
        for k in model_components: model_components[k].load_state_dict(ckpt[k], strict=False)
        del ckpt; train_flag = False
        print("[INFO] Found trained_model.pt → skip training", flush=True)
    else:
        print("[INFO] Training from scratch", flush=True)

    # ----------------------------- Training loop -------------------------- #
    if train_flag:
        # ===== local safety/dump helpers (loop-scoped) =====
        FORCE_DEBUG_FIRST_BATCH = True

        # ===== init states =====
        seg_weight_dict   = {seg["global_idx"]: 1.0 for seg in all_segments}
        max_grad_norm     = 1.0
        log_interval      = 1000
        best_state_dict   = None
        epochs_no_improve = 0
        step_global       = start_epoch * max(1, len(train_loader))
        tau, use_rw_sampler = 1.0, False
        train_hist, val_hist = [], []

        # ===== one-batch forward with full safety guards =====
        def weighted_loss_one_batch(batch, return_ctx=False):
            # to device + safety
            cls_b  = batch["cls_array"].to(device)
            feat_b = batch["feat_array"].to(device)
            y_b    = _safe_count_(batch["y_array"].to(device))
            len_b  = _safe_len_(batch["length_array"].to(device))
            cid_b  = batch["chrom_id"].to(device)

            # padding mask (B,T) True at PAD
            key_pad = (len_b <= 0)

            # forward
            feat_emb = model_components["feature_embedder"](feat_b)
            fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
            chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

            if args.save_attention:
                out, attn_last = model_components["global_transformer"](
                    fused + chr_emb, key_padding_mask=key_pad, return_attn=True
                )
                unwrap(model_components["global_transformer"]).last_attn_cpu = attn_last[0].detach().cpu()
                torch.cuda.empty_cache()
            else:
                out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)

            out = out.contiguous()
            lam = _safe_rate_(model_components["nhpp_head"](out))  # per-kb rate

            # ★ λ-그래디언트 살균: 손실에서 NaN/Inf가 내려오면 0으로 정리
            if lam.requires_grad:
                lam.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))

            # roll(창 합) 사용 시 lam,y,len을 동일 창으로 변환 후 다시 안전화
            if args.label_roll and args.label_roll_width > 1:
                lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                lam   = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

            # seg별 가중치
            w_seg = torch.tensor(
                [seg_weight_dict.get(seg["global_idx"], 1.0) for seg in batch["raw_segments"]],
                device=device, dtype=torch.float32
            )

            # --- pre-loss sanity ---
            if DEBUG_NAN:
                if (not torch.isfinite(lam).all()) or (not torch.isfinite(y_b).all()) or (not torch.isfinite(len_b).all()):
                    _dump_batch_stats("nonfinite-pre-loss", lam, y_b, len_b)
                pad_y = ((len_b <= 0) & (y_b > 0)).sum().item()
                if pad_y > 0:
                    print(f"[WARN] y>0 on PAD bins: {pad_y}", flush=True)

            # 최종 손실
            loss = trapezoid_nhpp_loss_segment_weighted(lam, y_b, len_b, w_seg)

            if return_ctx:
                return loss, lam.detach(), y_b.detach(), len_b.detach()
            return loss

        # ===== epoch loop =====
        for epoch in range(start_epoch, args.epochs):
            for m in model_components.values(): m.train()
            sum_loss, cnt = 0.0, 0

            for _step, batch in enumerate(train_loader):
                step_global += 1

                if DEBUG_NAN:
                    loss_val, lam_dbg, y_dbg, dt_dbg = weighted_loss_one_batch(batch, True)
                else:
                    loss_val = weighted_loss_one_batch(batch)

                # 첫 배치 강제 덤프
                if DEBUG_NAN and FORCE_DEBUG_FIRST_BATCH and step_global == 1:
                    _dump_batch_stats("pre-loss(first-batch)", lam_dbg, y_dbg, dt_dbg)

                # NaN 감지 → 통계/미니덤프 후 raise
                if not torch.isfinite(loss_val):
                    if DEBUG_NAN and 'lam_dbg' in locals():
                        _dump_batch_stats("NaN-before-raise", lam_dbg, y_dbg, dt_dbg)
                        lam_s  = torch.clamp(lam_dbg, min=_RATE_MIN, max=_RATE_MAX)
                        y_s    = _safe_count_(y_dbg)
                        dt_s   = _safe_len_(dt_dbg)
                        sumlog = (y_s * torch.log(lam_s)).sum().item()
                        integ  = (lam_s * dt_s).sum().item()
                        print(f"[DBG] quick-NLL: sum(y*log λ)={sumlog:.4e}, ∫λΔt={integ:.4e}", flush=True)
                        try:
                            dump_path = (Path(args.out_dir) / "debug_first_batch.pt").as_posix()
                            torch.save({
                                "lam": lam_dbg.cpu().float()[:1],
                                "y":   y_dbg.cpu().float()[:1],
                                "dt":  dt_dbg.cpu().float()[:1],
                                "raw_segments": batch["raw_segments"][:1],
                            }, dump_path)
                            print(f"[DBG] saved {dump_path}", flush=True)
                        except Exception as e:
                            print(f"[DBG] dump save failed: {e}", flush=True)
                    print(f"[NaN] epoch={epoch} step={step_global}", flush=True)
                    raise RuntimeError("NaN detected")

                # backward
                optimizer.zero_grad()
                loss_val.backward()
                for g in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], max_grad_norm)
                optimizer.step()
                scheduler.step()

                sum_loss += float(loss_val.item()); cnt += 1

                # logging
                if step_global % log_interval == 0:
                    avg = sum_loss / max(cnt, 1); sum_loss = 0.0; cnt = 0
                    if DEBUG_NAN and 'lam_dbg' in locals():
                        len_kb = _safe_len_(batch["length_array"].to(device))
                        mask   = (len_kb > 0)
                        mu_dbg = lam_dbg * len_kb
                        y_mean  = float((_safe_count_(batch["y_array"].to(device))[mask]).mean().item()) if mask.any() else 0.0
                        mu_mean = float((mu_dbg[mask]).mean().item()) if mask.any() else 0.0
                        nhpp_h  = model_components["nhpp_head"]
                        scale   = torch.exp(unwrap(nhpp_h).log_c).item() if hasattr(unwrap(nhpp_h), "log_c") else float("nan")
                        ratio   = mu_mean / max(y_mean, 1e-9)
                        print(f"[Epoch {epoch} | Step {step_global}] "
                              f"loss={avg:.4f}  μ_mean={mu_mean:.4g}  y_mean={y_mean:.4g}  μ/y={ratio:.3f}  scale={scale:.3f}",
                              flush=True)
                    else:
                        print(f"[Epoch {epoch} | Step {step_global}] loss={avg:.4f}", flush=True)

            # -------------------- epoch 평가 ------------------------------ #
            def eval_loader(loader):
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in loader:
                        cls_b  = b["cls_array"].to(device)
                        feat_b = b["feat_array"].to(device)
                        y_b    = _safe_count_(b["y_array"].to(device))
                        len_b  = _safe_len_(b["length_array"].to(device))
                        cid_b  = b["chrom_id"].to(device)
                        key_pad = (len_b <= 0)

                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                        out = out.contiguous()
                        lam = _safe_rate_(model_components["nhpp_head"](out))

                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                            lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                        mu = lam * len_b
                        y_np, mu_np = y_b.cpu().numpy(), mu.cpu().numpy()
                        for i, seg in enumerate(b["raw_segments"]):
                            L = seg["cls_array"].shape[0]
                            res[seg["global_idx"]] = float((y_np[i, :L] - mu_np[i, :L]).mean())

                rs = np.array(list(res.values()), dtype=np.float64)
                scale = compute_mad(rs) if args.use_mad else compute_iqr(rs)
                if not np.isfinite(scale) or scale <= 0: scale = 1.0
                delta = max(args.huber_factor * scale, 1.e-6)
                w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

                tot_num = tot_den = 0.0
                with torch.no_grad():
                    for b in loader:
                        cls_b  = b["cls_array"].to(device)
                        feat_b = b["feat_array"].to(device)
                        y_b    = _safe_count_(b["y_array"].to(device))
                        len_b  = _safe_len_(b["length_array"].to(device))
                        cid_b  = b["chrom_id"].to(device)
                        key_pad = (len_b <= 0)

                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                        out = out.contiguous()
                        lam = _safe_rate_(model_components["nhpp_head"](out))

                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                            lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                        # per-seg NLL (길이 정규화 후 kb*30)
                        sum_log = (y_b * torch.log(lam)).sum(dim=1)
                        integ   = (lam * len_b).sum(dim=1)
                        neg_ll  = -(sum_log - integ)
                        seg_len = (len_b.sum(dim=1) + _EPS)
                        per_seg = (neg_ll / seg_len) * 30.0

                        ids = [seg["global_idx"] for seg in b["raw_segments"]]
                        w   = torch.tensor([w_dict.get(i, 1.0) for i in ids],
                                           device=per_seg.device, dtype=per_seg.dtype)
                        tot_num += float((w * per_seg).sum().item());  tot_den += float(w.sum().item())
                return tot_num / max(tot_den, _EPS)

            # log_c 임시 조정 → 평가
            nh = unwrap(model_components["nhpp_head"])
            with torch.no_grad():
                _logc_backup = nh.log_c.detach().clone() if hasattr(nh, "log_c") else None

            calibrate_log_c_huber_like_training(
                model_components, train_loader_infer, device,
                huber_factor=args.huber_factor, use_mad=args.use_mad,
                label_roll=args.label_roll, roll_width=args.label_roll_width
            )

            train_loss = eval_loader(train_loader_infer)
            val_loss   = eval_loader(val_loader)
            train_hist.append(train_loss); val_hist.append(val_loss)
            print(f"[Epoch {epoch}] Train={train_loss:.4f}  Val={val_loss:.4f}", flush=True)

            # -------- residuals → seg weights / sampler --------
            def compute_segment_residual():
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in train_loader_infer:
                        cls_b  = b["cls_array"].to(device)
                        feat_b = b["feat_array"].to(device)
                        y_b    = _safe_count_(b["y_array"].to(device))
                        len_b  = _safe_len_(b["length_array"].to(device))
                        cid_b  = b["chrom_id"].to(device)
                        key_pad = (len_b <= 0)

                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                        out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                        out = out.contiguous()
                        lam = _safe_rate_(model_components["nhpp_head"](out))

                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                            lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                        mu = lam * len_b
                        y_np, mu_np = y_b.cpu().numpy(), mu.cpu().numpy()
                        for i, seg in enumerate(b["raw_segments"]):
                            L = seg["cls_array"].shape[0]
                            res[seg["global_idx"]] = float((y_np[i, :L] - mu_np[i, :L]).mean())
                return res

            seg_res = compute_segment_residual()
            rs = np.array(list(seg_res.values()))
            scale = compute_mad(rs) if args.use_mad else compute_iqr(rs)
            if not np.isfinite(scale) or scale <= 0: scale = 1.0
            delta = max(args.huber_factor * scale, 1e-6)
            seg_weight_dict = {sid: huber_weight(r, delta) for sid, r in seg_res.items()}

            def build_sampler(resid_dict, tau, alpha, beta):
                abs_r  = np.array([abs(resid_dict.get(seg["global_idx"], 0.0)) for seg in ds_train.segments], np.float64)
                len_kb = np.array([(seg["end_array"][-1] - seg["start_array"][0] + 1) / 1000.0 for seg in ds_train.segments], np.float64)
                p = np.exp(-beta * abs_r / (tau + _EPS)) * np.maximum(len_kb, _EPS) ** alpha
                s = float(np.nansum(p))
                if (not np.isfinite(p).all()) or s <= 0.0:
                    p = np.full_like(p, 1.0 / len(p))
                else:
                    p /= s
                return WeightedRandomSampler(torch.DoubleTensor(p), len(ds_train), replacement=True)

            if epoch == 3 and not use_rw_sampler:
                use_rw_sampler = True
                train_loader = DataLoader(
                    ds_train, batch_size=args.batch_size,
                    sampler=build_sampler(seg_res, tau, args.len_alpha, args.res_beta),
                    collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin
                )
                print(f"[INFO] Residual-weighted sampler enabled @epoch {epoch}", flush=True)
            elif use_rw_sampler:
                tau *= 0.999
                train_loader = DataLoader(
                    ds_train, batch_size=args.batch_size,
                    sampler=build_sampler(seg_res, tau, args.len_alpha, args.res_beta),
                    collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin
                )

            # -------- best checkpoint & attention snapshot --------
            improved = (val_loss < best_val_loss - best_val_loss * getattr(args, "min_delta_pct", 0.5) / 100.0) if best_val_loss != float("inf") else True
            if improved:
                best_val_loss = val_loss; epochs_no_improve = 0
                best_state_dict = {k: model_components[k].state_dict() for k in model_components}
                ckpt_common = {**best_state_dict, "epoch": epoch,
                               "optimizer_state": optimizer.state_dict(),
                               "sched_state": scheduler.state_dict(),
                               "best_val_loss": best_val_loss}
                if getattr(args, "save_each_best", False):
                    ep_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
                    efficient_save_ckpt(ep_path, **ckpt_common)
                efficient_save_ckpt(model_save_path, **ckpt_common)
                print(f"[INFO] New best Val={best_val_loss:.4f}", flush=True)
                if getattr(args, "save_attention", False):
                    save_last_layer_attention(unwrap(model_components["global_transformer"]), epoch, step_global, out_dir)
                    torch.cuda.empty_cache()
            else:
                epochs_no_improve += 1

            # restore log_c
            if _logc_backup is not None:
                with torch.no_grad(): unwrap(model_components["nhpp_head"]).log_c.copy_(_logc_backup)

            # early stop
            if getattr(args, "early_stop", False) and epochs_no_improve >= getattr(args, "patience", 5):
                print("[INFO] Early stopping (patience reached)", flush=True)
                break

        # ---- 학습곡선 저장 ----
        plt.figure()
        x = range(len(train_hist))
        plt.plot(x, train_hist, marker="o", label="train")
        plt.plot(x, val_hist, marker="x", label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(out_dir, "train_val_loss_per_epoch.png")); plt.close()

    # ------------------------------------------------------------------- #
    # Final evaluation & attention full-dump                               #
    # ------------------------------------------------------------------- #
    print("[INFO] Loading best model for final evaluation ..", flush=True)
    best_ckpt = efficient_load_ckpt(model_save_path)
    for k in model_components: model_components[k].load_state_dict(best_ckpt[k], strict=False)
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
        for m in model_components.values(): m.eval()
        res = {}
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = _safe_count_(b["y_array"].to(device))
                len_b  = _safe_len_(b["length_array"].to(device))
                cid_b  = b["chrom_id"].to(device)
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                out      = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                out      = out.contiguous()
                lam      = _safe_rate_(model_components["nhpp_head"](out))

                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                    lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                mu_b = lam * len_b
                for i, seg in enumerate(b["raw_segments"]):
                    L = seg["cls_array"].shape[0]
                    res[seg["global_idx"]] = float((y_b[i, :L] - mu_b[i, :L]).mean().item())

        rs     = np.array(list(res.values()), dtype=np.float64)
        scale  = compute_mad(rs) if args.use_mad else compute_iqr(rs)
        delta  = max(args.huber_factor * scale, 1e-9)
        w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

        tot_num, tot_den = 0.0, 0.0
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = _safe_count_(b["y_array"].to(device))
                len_b  = _safe_len_(b["length_array"].to(device))
                cid_b  = b["chrom_id"].to(device)
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)

                out      = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                out      = out.contiguous()
                lam      = _safe_rate_(model_components["nhpp_head"](out))
                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                    lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                sum_log  = (y_b * torch.log(lam)).sum(dim=1)
                integ    = (lam * len_b).sum(dim=1)
                neg_ll   = -(sum_log - integ)
                seg_len  = (len_b.sum(dim=1) + 1e-9)
                per_seg  = (neg_ll / seg_len) * 30.0

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
        for m in model_components.values(): m.eval()

        rows, tot_ll, tot_bin = [], 0.0, 0
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = _safe_count_(b["y_array"].to(device))
                len_b  = _safe_len_(b["length_array"].to(device))
                s_bp   = b["start_array"].detach().cpu().numpy()
                e_bp   = b["end_array"].detach().cpu().numpy()
                cid_b  = b["chrom_id"].to(device)
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused    = model_components["feat_cls_fusion"](cls_b, feat_emb)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                out      = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)
                out      = out.contiguous()
                lam_raw  = _safe_rate_(model_components["nhpp_head"](out))

                lam_save = lam_raw
                if args.label_roll and args.label_roll_width > 1:
                    lam_r, _y_ignore, _dt_ignore = R.rolling_sum_nhpp(
                        lam_raw, y_b, len_b, width=args.label_roll_width
                    )
                    lam_save = _safe_rate_(lam_r)

                lam_np_raw  = lam_raw.detach().cpu().numpy()
                lam_np      = lam_save.detach().cpu().numpy()
                y_np        = y_b.detach().cpu().numpy()
                len_np      = len_b.detach().cpu().numpy()

                B, T = lam_np.shape
                for i, seg in enumerate(b["raw_segments"]):
                    L = seg["cls_array"].shape[0]
                    L_eff = min(L, T, y_np.shape[1], len_np.shape[1], s_bp.shape[1], e_bp.shape[1])
                    if L_eff <= 0: continue

                    lam_i      = lam_np[i, :L_eff]
                    lam_raw_i  = lam_np_raw[i, :L_eff]
                    y_i        = y_np[i, :L_eff]
                    dt_i       = len_np[i, :L_eff]
                    s_i        = s_bp[i, :L_eff]
                    e_i        = e_bp[i, :L_eff]

                    lam_raw_i_safe = np.clip(lam_raw_i, _RATE_MIN, 1e4)
                    llb_i = y_i * np.log(lam_raw_i_safe) - lam_raw_i_safe * dt_i

                    chrom = seg["chrom"]
                    for j in range(L_eff):
                        rows.append(dict(
                            chrom=chrom,
                            start=int(s_i[j]),
                            end=int(e_i[j]),
                            lam_pred=float(lam_i[j]),      # 스무딩된 1kb당 rate
                            obs_count=float(y_i[j]),       # 원본 관측 카운트
                            bin_loglike=float(llb_i[j]),   # 원본 lam 기반 로그우도
                        ))
                        tot_ll  += llb_i[j];  tot_bin += 1

        df = pd.DataFrame(rows).sort_values(["chrom", "start"])
        out_file = out_dir / f"{name}_prediction.csv"
        df.to_csv(out_file, index=False)
        print(f"[INFO] {name} saved → {out_file} (bins={len(df)})", flush=True)

    # 그대로 호출
    predict_and_save(train_loader_infer, "train")
    predict_and_save(val_loader,   "val")
    predict_and_save(all_loader,   "all")

    if getattr(args, "save_attention", False):
        dump_full_attention(model_components, all_loader, device, out_dir / "attn" / "final_full_attention.pt")

    print("[DONE] Training/validation complete, predictions & attentions saved.", flush=True)
    return (out_dir / "all_prediction.csv").as_posix()
