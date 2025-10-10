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

import os, math, itertools, gc, random, warnings
from functools import partial
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
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

from ..utils.io import build_chrom_id_map, unwrap, efficient_load_ckpt, efficient_save_ckpt, check_pretrained_model_exists, set_seed
from ..data.rolling import rolling_sum_nhpp
from ..losses.nhpp import trapezoid_nhpp_loss, trapezoid_nhpp_loss_segment_weighted
from ..train.attention import save_last_layer_attention, dump_full_attention
from ..utils.stats import compute_mad, compute_iqr, huber_weight
from ..utils.plotting import qq_plot
from ..utils.chrom import CHROM_LIST_24
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

@torch.no_grad()
def calibrate_log_c_huber_like_training(model_c, loader, device,
                                        huber_factor=3.0, use_mad=False,
                                        label_roll=False, roll_width=1):
    for m in model_c.values():
        m.eval()
    nh = unwrap(model_c["nhpp_head"])
    c_cur = float(torch.exp(nh.log_c.detach()).cpu())

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

        y_np, mu_np = y_b.detach().cpu().numpy(), mu_b.detach().cpu().numpy()
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

                        y_np = y_b.detach().cpu().numpy()
                        mu_np  = mu_b.detach().cpu().numpy()
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
                s_bp   = b["start_array"].detach().cpu().numpy()
                e_bp   = b["end_array"].detach().cpu().numpy()
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
                lam_np_raw  = lam.detach().detach().cpu().numpy()       # bin_loglike 계산용(원본)
                lam_np      = lam_save.detach().detach().cpu().numpy()  # 저장용 lam_pred(스무딩)
                y_np        = y_b.detach().detach().cpu().numpy()
                len_np      = len_b.detach().detach().cpu().numpy()

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
