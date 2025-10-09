#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:41:23 2025
Updated on Mon Jul  7 19:20:00 2025  ← attention-save policy + NaN guards & debug

Changes
-------
* Val 개선 시 마지막 레이어 head-mean 저장 / 최종에 full-stack 저장
* NaN 방지: λ/y/Δt 안정화, log/div 안전화, sampler 확률 폴백
* 평가/예측 경로에도 동일 가드 적용(roll 켜졌을 때 저장값 일치)
* 첫 배치 강제 디버그 출력, NaN 직전 통계 + 미니 덤프(debug_first_batch.pt) 저장
* 예측 CSV에 lam_pred_raw(원본), lam_pred_rollW(창 W 스무딩 per-kb) 둘 다 저장
"""

# ===== Imports ===============================================================
import os, sys, math, random, pickle, argparse, gc, itertools
from functools import partial
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- project modules -------------------------------------------------------
from ..models.embedders import FeatureEmbedder, FeatClsFusion, ChromosomeEmbedder
from ..models.transformer import GlobalTransformerEncoder
from ..models.nhpp_head import NHPPHead
from ..data.segments import SegmentDataset, segment_cls_embeddings_fixed_lengths, segment_collate_fn
from ..data.labels import _bins_from_cls_list, _load_mutations_events, _build_y_map_from_mutations, _attach_labels_from_y_map
from ..data.rolling import rolling_sum_nhpp
from ..utils.io import build_chrom_id_map, unwrap, efficient_load_ckpt, efficient_save_ckpt, check_pretrained_model_exists, set_seed
from ..losses.nhpp import trapezoid_nhpp_loss_segment_weighted
from ..train.attention import save_last_layer_attention, dump_full_attention
from ..utils.stats import compute_mad, compute_iqr, huber_weight
from ..utils.chrom import CHROM_LIST_24

# ===== Globals ===============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

DEBUG_NAN = True
_EPS       = 1e-8
_RATE_MIN  = 1e-9
_RATE_MAX  = 1e6  # 상한 여유

# --- stdout 무버퍼(선택) ---
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ===== NaN/Inf safety helpers ===============================================
def _nan_to_num_(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=0.0)

def _safe_rate_(lam: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(lam).clamp(_RATE_MIN, _RATE_MAX)

def _safe_len_(dt: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(dt).clamp(min=0.0)

def _safe_count_(y: torch.Tensor) -> torch.Tensor:
    return _nan_to_num_(y).clamp(min=0.0)

def _assert_finite(name: str, t: torch.Tensor):
    if DEBUG_NAN and not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].tolist()
        raise RuntimeError(f"[non-finite] {name} examples={bad}")

# ==== DEBUG helpers ====
FORCE_DEBUG_FIRST_BATCH = True  # 첫 배치 강제 출력

def _stat(t: torch.Tensor):
    t = t.detach()
    return dict(
        shape=list(t.shape),
        min=float(torch.nan_to_num(t).min()),
        max=float(torch.nan_to_num(t).max()),
        mean=float(torch.nan_to_num(t).mean()),
        n_nan=int((~torch.isfinite(t)).sum().item()),
        n_le0=int((t <= 0).sum().item()),
    )

def _dump_batch_stats(tag: str, lam: torch.Tensor, y: torch.Tensor, dt: torch.Tensor):
    mu = lam * dt
    info = {"tag": tag, "lam": _stat(lam), "y": _stat(y), "dt": _stat(dt), "mu": _stat(mu)}
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
        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
        y_b  = _safe_count_(b["y_array"].to(device))
        dt_b = _safe_len_(b["length_array"].to(device))
        cid  = b["chrom_id"].to(device)
        key_pad = (dt_b <= 0)

        feat = model_c["feature_embedder"](feat_b)
        fused = model_c["feat_cls_fusion"](cls_b, feat)
        chr_emb = model_c["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
        lam = _safe_rate_(model_c["nhpp_head"](model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))

        if label_roll and roll_width > 1:
            lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=roll_width)

        mu = lam * dt_b
        y_np, mu_np = y_b.cpu().numpy(), mu.cpu().numpy()
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
        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
        y_b  = _safe_count_(b["y_array"].to(device))
        dt_b = _safe_len_(b["length_array"].to(device))
        cid  = b["chrom_id"].to(device)
        key_pad = (dt_b <= 0)

        feat = model_c["feature_embedder"](feat_b)
        fused = model_c["feat_cls_fusion"](cls_b, feat)
        chr_emb = model_c["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
        lam = _safe_rate_(model_c["nhpp_head"](model_c["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))

        if label_roll and roll_width > 1:
            lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=roll_width)

        base = lam / max(c_cur, _EPS)
        for i, seg in enumerate(b["raw_segments"]):
            L   = seg["cls_array"].shape[0]
            sid = seg["global_idx"]
            w   = float(w_seg.get(sid, 1.0))
            num += w * float(y_b[i, :L].sum().item())
            den += w * float((base[i, :L] * dt_b[i, :L]).sum().item())

    if not (np.isfinite(num) and np.isfinite(den)) or num <= _EPS or den <= _EPS:
        print(f"[CAL-HUBER] skip (num={num:.3g}, den={den:.3g}) keep c={c_cur:.6g}", flush=True);  return
    c_star = num/den
    if np.isfinite(c_star) and c_star > 0:
        nh.log_c.copy_(torch.tensor(math.log(max(c_star, _EPS)), device=nh.log_c.device))
        print(f"[CAL-HUBER] c_prev={c_cur:.6g} → c_new={c_star:.6g}  ratio={c_star/(c_cur+_EPS):.4f}", flush=True)
    else:
        print(f"[CAL-HUBER] invalid c*: {c_star}", flush=True)

# ===== Training & prediction ================================================
def train_and_predict(args):
    set_seed(args.seed)

    if getattr(args, "torch_threads", None):
        try:
            torch.set_num_threads(int(args.torch_threads))
            print(f"[INFO] torch num_threads = {torch.get_num_threads()}", flush=True)
        except Exception as e:
            print(f"[WARN] set_num_threads failed: {e}", flush=True)

    # ----- 데이터 로딩 -----
    print("[INFO] Loading data ..", flush=True)
    with open(args.cls_file, "rb") as f:   cls_list = pickle.load(f)
    with open(args.feat_file, "rb") as f:  feature_dict = pickle.load(f)

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

    # ----- 1kb bin 라벨 -----
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

    # ----- 모델 -----
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

    model_components = dict(feature_embedder=feat_embedder, feat_cls_fusion=feat_cls_fusion,
                            chrom_embedder=chrom_embedder, global_transformer=global_transformer,
                            nhpp_head=nhpp_head)

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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_sched_T0, T_mult=args.lr_sched_Tmult, eta_min=args.lr*0.3)

    # ----- Checkpoint -----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = out_dir / "trained_model.pt"

    start_epoch, train_flag = 0, True
    best_val_loss = float("inf")

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        ckpt = efficient_load_ckpt(args.resume_checkpoint)
        for k in model_components: model_components[k].load_state_dict(ckpt[k], strict=False)
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "sched_state" in ckpt:     scheduler.load_state_dict(ckpt["sched_state"])
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

    # ----- Train loop -----
    if train_flag:
        seg_weight_dict = {seg["global_idx"]: 1.0 for seg in all_segments}
        max_grad_norm, log_interval = 1.0, 1000
        best_state_dict, epochs_no_improve, step_global = None, 0, start_epoch * max(1, len(train_loader))
        tau, use_rw_sampler = 1.0, False
        train_hist, val_hist = [], []

        def weighted_loss_one_batch(batch, return_lam=False):
            cls_b, feat_b = batch["cls_array"].to(device), batch["feat_array"].to(device)
            y_b  = _safe_count_(batch["y_array"].to(device))
            dt_b = _safe_len_(batch["length_array"].to(device))
            cid  = batch["chrom_id"].to(device)
            key_pad = (dt_b <= 0)

            feat = model_components["feature_embedder"](feat_b)
            fused = model_components["feat_cls_fusion"](cls_b, feat)
            chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)

            if args.save_attention:
                out, attn_last = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad, return_attn=True)
                unwrap(model_components["global_transformer"]).last_attn_cpu = attn_last[0].detach().cpu()
                torch.cuda.empty_cache()
            else:
                out = model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)

            lam = _safe_rate_(model_components["nhpp_head"](out))

            if args.label_roll and args.label_roll_width > 1:
                lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)

            _assert_finite("lam", lam); _assert_finite("y_b", y_b); _assert_finite("len_b", dt_b)

            w_seg = torch.tensor([seg_weight_dict.get(seg["global_idx"], 1.0) for seg in batch["raw_segments"]],
                                 device=device, dtype=torch.float32)
            loss = trapezoid_nhpp_loss_segment_weighted(lam, y_b, dt_b, w_seg)
            return (loss, lam.detach()) if return_lam else loss

        for epoch in range(start_epoch, args.epochs):
            for m in model_components.values(): m.train()
            sum_loss, cnt = 0.0, 0

            for _step, batch in enumerate(train_loader):
                step_global += 1
                # forward
                loss_val, lam_dbg = weighted_loss_one_batch(batch, True) if DEBUG_NAN else (weighted_loss_one_batch(batch), None)

                # --- 첫 배치 통계 강제 출력 ---
                if FORCE_DEBUG_FIRST_BATCH and step_global == 1:
                    y_dbg  = _safe_count_(batch["y_array"].to(device))
                    dt_dbg = _safe_len_(batch["length_array"].to(device))
                    _dump_batch_stats("pre-loss(first-batch)", lam_dbg, y_dbg, dt_dbg)

                # NaN 감지 시 통계 + 미니 덤프 저장
                if not torch.isfinite(loss_val):
                    y_dbg  = _safe_count_(batch["y_array"].to(device))
                    dt_dbg = _safe_len_(batch["length_array"].to(device))
                    _dump_batch_stats("NaN-before-raise", lam_dbg, y_dbg, dt_dbg)
                    try:
                        dump_path = (Path(args.out_dir) / "debug_first_batch.pt").as_posix()
                        torch.save({
                            "lam": lam_dbg.detach().cpu().float()[:1],
                            "y":   y_dbg.detach().cpu().float()[:1],
                            "dt":  dt_dbg.detach().cpu().float()[:1],
                            "raw_segments": batch["raw_segments"][:1],
                        }, dump_path)
                        print(f"[DBG] saved {dump_path}", flush=True)
                    except Exception as e:
                        print(f"[DBG] dump save failed: {e}", flush=True)
                    print(f"[NaN] epoch={epoch} step={step_global}", flush=True)
                    raise RuntimeError("NaN detected")

                optimizer.zero_grad()
                loss_val.backward()
                for g in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], max_grad_norm)
                optimizer.step(); scheduler.step()

                sum_loss += float(loss_val.item()); cnt += 1

                if step_global % log_interval == 0:
                    avg = sum_loss / max(cnt, 1); sum_loss = 0.0; cnt = 0
                    if DEBUG_NAN and lam_dbg is not None:
                        y_full = _safe_count_(batch["y_array"].to(device))
                        dt_b   = _safe_len_(batch["length_array"].to(device))
                        mask   = (dt_b > 0)
                        mu_dbg = lam_dbg * dt_b
                        y_mean  = (y_full[mask]).mean().item() if mask.any() else 0.0
                        mu_mean = (mu_dbg[mask]).mean().item()  if mask.any() else 0.0
                        nhpp_h  = model_components["nhpp_head"];  scale = torch.exp(unwrap(nhpp_h).log_c).item()
                        print(f"[Epoch {epoch} | Step {step_global}] loss={avg:.4f}  μ_mean={mu_mean:.4g}  y_mean={y_mean:.4g}  scale={scale:.3f}", flush=True)
                    else:
                        print(f"[Epoch {epoch} | Step {step_global}] loss={avg:.4f}", flush=True)

            # ---- epoch 평가 ----
            def eval_loader(loader):
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in loader:
                        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                        y_b  = _safe_count_(b["y_array"].to(device))
                        dt_b = _safe_len_(b["length_array"].to(device))
                        cid  = b["chrom_id"].to(device)
                        key_pad = (dt_b <= 0)

                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
                        lam = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))
                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)

                        mu = lam * dt_b
                        for i, seg in enumerate(b["raw_segments"]):
                            L = seg["cls_array"].shape[0]
                            res[seg["global_idx"]] = float((y_b[i, :L] - mu[i, :L]).mean().cpu().item())

                rs = np.array(list(res.values()), dtype=np.float64)
                scale = compute_mad(rs) if args.use_mad else compute_iqr(rs)
                if not np.isfinite(scale) or scale <= 0: scale = 1.0
                delta = max(args.huber_factor * scale, 1e-6)
                w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

                tot_num = tot_den = 0.0
                with torch.no_grad():
                    for b in loader:
                        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                        y_b  = _safe_count_(b["y_array"].to(device))
                        dt_b = _safe_len_(b["length_array"].to(device))
                        cid  = b["chrom_id"].to(device)
                        key_pad = (dt_b <= 0)
                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
                        lam = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))
                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)

                        sum_log = (y_b * torch.log(lam)).sum(dim=1)
                        integ   = (lam * dt_b).sum(dim=1)
                        neg_ll  = -(sum_log - integ)
                        seg_len = (dt_b.sum(dim=1) + _EPS)
                        per_seg = (neg_ll / seg_len) * 30.0

                        ids = [seg["global_idx"] for seg in b["raw_segments"]]
                        w   = torch.tensor([w_dict.get(i, 1.0) for i in ids], device=per_seg.device, dtype=per_seg.dtype)
                        tot_num += float((w * per_seg).sum().item());  tot_den += float(w.sum().item())
                return tot_num / max(tot_den, _EPS)

            nh = unwrap(model_components["nhpp_head"])
            with torch.no_grad(): _logc_backup = nh.log_c.detach().clone()

            calibrate_log_c_huber_like_training(model_components, train_loader_infer, device,
                                                huber_factor=args.huber_factor, use_mad=args.use_mad,
                                                label_roll=args.label_roll, roll_width=args.label_roll_width)

            train_loss = eval_loader(train_loader_infer)
            val_loss   = eval_loader(val_loader)
            train_hist.append(train_loss); val_hist.append(val_loss)
            print(f"[Epoch {epoch}] Train={train_loss:.4f}  Val={val_loss:.4f}", flush=True)

            # ---- seg residual→ sampler ----
            def compute_segment_residual():
                for m in model_components.values(): m.eval()
                res = {}
                with torch.no_grad():
                    for b in train_loader_infer:
                        cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                        y_b  = _safe_count_(b["y_array"].to(device))
                        dt_b = _safe_len_(b["length_array"].to(device))
                        cid  = b["chrom_id"].to(device)
                        key_pad = (dt_b <= 0)
                        feat = model_components["feature_embedder"](feat_b)
                        fused = model_components["feat_cls_fusion"](cls_b, feat)
                        chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
                        lam = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))
                        if args.label_roll and args.label_roll_width > 1:
                            lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)
                        mu = lam * dt_b
                        for i, seg in enumerate(b["raw_segments"]):
                            L = seg["cls_array"].shape[0]
                            res[seg["global_idx"]] = float((y_b[i, :L] - mu[i, :L]).mean().cpu().item())
                return res

            seg_res = compute_segment_residual()
            rs = np.array(list(seg_res.values()), dtype=np.float64)
            scale = compute_mad(rs) if args.use_mad else compute_iqr(rs)
            if not np.isfinite(scale) or scale <= 0: scale = 1.0
            delta = max(args.huber_factor * scale, 1e-6)
            seg_weight_dict = {sid: huber_weight(r, delta) for sid, r in seg_res.items()}

            def build_sampler(resid_dict, tau, alpha, beta):
                abs_r = np.array([abs(resid_dict.get(seg["global_idx"], 0.0)) for seg in ds_train.segments], np.float64)
                len_kb = np.array([(seg["end_array"][-1] - seg["start_array"][0] + 1)/1000.0 for seg in ds_train.segments], np.float64)
                p = np.exp(-beta * abs_r / (tau + _EPS)) * (np.maximum(len_kb, 1e-6) ** alpha)
                s = float(p.sum())
                if (not np.isfinite(p).all()) or s <= 0.0:
                    p = np.full_like(p, 1.0/len(p))
                else:
                    p /= s
                return WeightedRandomSampler(torch.DoubleTensor(p), len(ds_train), replacement=True)

            if epoch == 3 and not use_rw_sampler:
                use_rw_sampler = True
                train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                          sampler=build_sampler(seg_res, tau, args.len_alpha, args.res_beta),
                                          collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin)
                print(f"[INFO] Residual-weighted sampler enabled @epoch {epoch}", flush=True)
            elif use_rw_sampler:
                tau *= 0.999
                train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                          sampler=build_sampler(seg_res, tau, args.len_alpha, args.res_beta),
                                          collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin)

            # ---- best 모델 저장 & 어텐션 스냅숏 ----
            improved = (val_loss < best_val_loss - best_val_loss * args.min_delta_pct / 100.0) if best_val_loss != float("inf") else True
            if improved:
                best_val_loss = val_loss; epochs_no_improve = 0
                best_state_dict = {k: model_components[k].state_dict() for k in model_components}
                ckpt_common = {**best_state_dict, "epoch": epoch,
                               "optimizer_state": optimizer.state_dict(),
                               "sched_state": scheduler.state_dict(),
                               "best_val_loss": best_val_loss}
                if args.save_each_best:
                    ep_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt"); efficient_save_ckpt(ep_path, **ckpt_common)
                efficient_save_ckpt(model_save_path, **ckpt_common)
                print(f"[INFO] New best Val={best_val_loss:.4f}", flush=True)
                if args.save_attention:
                    save_last_layer_attention(unwrap(model_components["global_transformer"]), epoch, step_global, out_dir)
                    torch.cuda.empty_cache()
            else:
                epochs_no_improve += 1

            with torch.no_grad(): unwrap(model_components["nhpp_head"]).log_c.copy_(_logc_backup)
            if args.early_stop and epochs_no_improve >= args.patience:
                print("[INFO] Early stopping (patience reached)", flush=True);  break

        # ---- 학습곡선 저장 ----
        plt.figure()
        x = range(len(train_hist))
        plt.plot(x, train_hist, marker="o", label="train")
        plt.plot(x, val_hist, marker="x", label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(out_dir, "train_val_loss_per_epoch.png")); plt.close()

    # ===== Final evaluation & prediction ===================================
    print("[INFO] Loading best model for final evaluation ..", flush=True)
    best_ckpt = efficient_load_ckpt(model_save_path)
    for k in model_components: model_components[k].load_state_dict(best_ckpt[k], strict=False)
    del best_ckpt; gc.collect(); torch.cuda.empty_cache()

    calibrate_log_c_huber_like_training(model_components, all_loader, device,
                                        huber_factor=args.huber_factor, use_mad=args.use_mad,
                                        label_roll=args.label_roll, roll_width=args.label_roll_width)

    def evaluate(loader):
        for m in model_components.values(): m.eval()
        res = {}
        with torch.no_grad():
            for b in loader:
                cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                y_b  = _safe_count_(b["y_array"].to(device))
                dt_b = _safe_len_(b["length_array"].to(device))
                cid  = b["chrom_id"].to(device)
                key_pad = (dt_b <= 0)

                feat = model_components["feature_embedder"](feat_b)
                fused = model_components["feat_cls_fusion"](cls_b, feat)
                chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
                lam = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))
                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)
                mu = lam * dt_b
                for i, seg in enumerate(b["raw_segments"]):
                    L = seg["cls_array"].shape[0]
                    res[seg["global_idx"]] = float((y_b[i, :L] - mu[i, :L]).mean().cpu().item())

        rs = np.array(list(res.values()), dtype=np.float64)
        scale = compute_mad(rs) if args.use_mad else compute_iqr(rs)
        if not np.isfinite(scale) or scale <= 0: scale = 1.0
        delta = max(args.huber_factor * scale, 1e-6)
        w_dict = {sid: huber_weight(r, delta) for sid, r in res.items()}

        tot_num = tot_den = 0.0
        with torch.no_grad():
            for b in loader:
                cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
                y_b  = _safe_count_(b["y_array"].to(device))
                dt_b = _safe_len_(b["length_array"].to(device))
                cid  = b["chrom_id"].to(device)
                key_pad = (dt_b <= 0)
                feat = model_components["feature_embedder"](feat_b)
                fused = model_components["feat_cls_fusion"](cls_b, feat)
                chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
                lam = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))
                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, dt_b = rolling_sum_nhpp(lam, y_b, dt_b, width=args.label_roll_width)

                sum_log = (y_b * torch.log(lam)).sum(dim=1)
                integ   = (lam * dt_b).sum(dim=1)
                neg_ll  = -(sum_log - integ)
                seg_len = (dt_b.sum(dim=1) + _EPS)
                per_seg = (neg_ll / seg_len) * 30.0

                ids = [seg["global_idx"] for seg in b["raw_segments"]]
                w   = torch.tensor([w_dict.get(i, 1.0) for i in ids], device=per_seg.device, dtype=per_seg.dtype)
                tot_num += float((w * per_seg).sum().item());  tot_den += float(w.sum().item())
        return tot_num / max(tot_den, _EPS)

    print(f"final Train_loss = {evaluate(train_loader_infer):.4f}", flush=True)
    print(f"final Val_loss   = {evaluate(val_loader):.4f}", flush=True)

    # ----- 예측 CSV 저장 ---------------------------------------------------
    @torch.no_grad()
    def predict_and_save(loader, name):
        """
        저장 규칙
        - lam_pred_raw   : 원본 per-kb rate (roll 미적용)
        - lam_pred_rollW : 창 W 스무딩 후 per-kb rate (W=1이면 raw와 동일)
        - bin_loglike    : 원본 rate 기준의 per-bin log-likelihood (상수항 제외)
        """
        for m in model_components.values(): m.eval()

        rows = []; tot_ll = tot_bin = 0
        W = int(getattr(args, "label_roll_width", 1))
        use_roll = bool(getattr(args, "label_roll", False) and W > 1)

        for b in loader:
            cls_b, feat_b = b["cls_array"].to(device), b["feat_array"].to(device)
            y_b  = _safe_count_(b["y_array"].to(device))
            dt_b = _safe_len_(b["length_array"].to(device))
            s_bp = b["start_array"].cpu().numpy()
            e_bp = b["end_array"].cpu().numpy()
            cid  = b["chrom_id"].to(device)
            key_pad = (dt_b <= 0)

            feat = model_components["feature_embedder"](feat_b)
            fused = model_components["feat_cls_fusion"](cls_b, feat)
            chr_emb = model_components["chrom_embedder"](cid).unsqueeze(1).expand_as(fused)
            lam_raw = _safe_rate_(model_components["nhpp_head"](model_components["global_transformer"](fused + chr_emb, key_padding_mask=key_pad)))

            if use_roll:
                lam_roll, _y_ig, _dt_ig = rolling_sum_nhpp(lam_raw, y_b, dt_b, width=W)  # lam_roll = μ_roll/Δt_roll
                lam_roll = _safe_rate_(lam_roll)
            else:
                lam_roll = lam_raw

            lam_np_raw  = np.clip(lam_raw.cpu().numpy(),  _RATE_MIN, _RATE_MAX)
            lam_np_roll = np.clip(lam_roll.cpu().numpy(), _RATE_MIN, _RATE_MAX)
            y_np  = np.nan_to_num(y_b.cpu().numpy(),  nan=0.0, posinf=0.0, neginf=0.0)
            dt_np = np.nan_to_num(dt_b.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)

            B, T = lam_np_raw.shape
            for i, seg in enumerate(b["raw_segments"]):
                L = seg["cls_array"].shape[0]
                L_eff = min(L, T, y_np.shape[1], dt_np.shape[1], s_bp.shape[1], e_bp.shape[1])
                if L_eff <= 0: continue

                lam_i_raw  = lam_np_raw[i, :L_eff]
                lam_i_roll = lam_np_roll[i, :L_eff]
                y_i, dt_i  = y_np[i, :L_eff], dt_np[i, :L_eff]
                s_i, e_i   = s_bp[i, :L_eff], e_bp[i, :L_eff]

                llb_i = y_i * np.log(lam_i_raw) - lam_i_raw * dt_i   # 원본 lam 기준

                chrom = seg["chrom"]
                for j in range(L_eff):
                    rows.append(dict(
                        chrom=chrom,
                        start=int(s_i[j]),
                        end=int(e_i[j]),
                        lam_pred_raw=float(lam_i_raw[j]),
                        lam_pred_rollW=float(lam_i_roll[j]),
                        obs_count=float(y_i[j]),
                        bin_loglike=float(llb_i[j]),
                    ))
                    tot_ll += float(llb_i[j]);  tot_bin += 1

        df = pd.DataFrame(rows).sort_values(["chrom", "start"])
        out_file = out_dir / f"{name}_prediction.csv"
        df.to_csv(out_file, index=False)
        suffix = f" (roll W={W})" if use_roll else ""
        print(f"[INFO] {name} saved → {out_file} (bins={len(df)}){suffix}", flush=True)

    predict_and_save(train_loader_infer, "train")
    predict_and_save(val_loader,   "val")
    predict_and_save(all_loader,   "all")   # → all_prediction.csv 생성

    if args.save_attention:
        dump_full_attention(model_components, all_loader, device, out_dir / "attn" / "final_full_attention.pt")

    print("[DONE] Training/validation complete, predictions & attentions saved.", flush=True)
    return (out_dir / "all_prediction.csv").as_posix()
