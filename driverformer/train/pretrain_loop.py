#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_loop.py — Pan-cancer pretraining loop (multi-organ)

(중략: 상단 주석/설명 동일)
"""

# --------------------------------------------------------------------------- #
# Imports & setup                                                             #
# --------------------------------------------------------------------------- #
import os, sys, math, random, pickle, argparse, gc, itertools, warnings, json
from functools import partial
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv  # [LOG] CSV 로깅

# ---- project modules ------------------------------------------------------- #
from ..models.embedders import FeatureEmbedder, ChromosomeEmbedder
from ..models.embedders import FeatClsFusion as FeatClsFusionCtx
from ..models.transformer import GlobalTransformerEncoder
from ..models.nhpp_head import ConditionalNHPPHead, CondCfg

# 세그/콜레이트 (classic + lite 모두 임포트)
from ..data.segments import (
    SegmentDataset,
    segment_cls_embeddings_fixed_lengths,
    segment_collate_fn,
    segment_indices_fixed_lengths,
    segment_collate_fn_lite,
)

from ..data.labels import (
    _bins_from_cls_list,
    _load_mutations_events,
    _build_y_map_from_mutations,
    _attach_labels_from_y_map,
)
import driverformer.data.rolling as R
from ..utils.io import (
    build_chrom_id_map, unwrap, efficient_load_ckpt, efficient_save_ckpt,
    check_pretrained_model_exists, set_seed
)
from ..losses.nhpp import trapezoid_nhpp_loss_segment_weighted
from ..train.attention import save_last_layer_attention, dump_full_attention
from ..utils.stats import compute_mad, compute_iqr, huber_weight
from ..utils.chrom import CHROM_LIST_24

# ===== Globals =============================================================== #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

DEBUG_NAN   = True
_EPS        = 1e-8
_RATE_MIN   = 1e-9
_RATE_MAX   = 1e6
STRICT_BACKWARD = os.environ.get("DF_STRICT_BACKWARD", "0") == "1"


# ===== A) GPU backend (속도 우선 모드) ===================================== #
import torch.backends.cudnn as _cudnn
_cudnn.enabled = True           # cuDNN 사용
_cudnn.benchmark = True         # 입력 크기 고정/유사하면 자동 커널 튜닝
_cudnn.deterministic = False    # 재현성 ↓, 속도 ↑

# sdpa 경고 억제(신 API→구 API 폴백)
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_new
    _sdpa_kernel_new(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    try:
        from torch.backends.cuda import sdp_kernel as _sdpa_kernel_old
        _sdpa_kernel_old(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===== NaN/Inf safety helpers =============================================== #
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

def _attach_grad_sanitizer(t: torch.Tensor) -> torch.Tensor:
    if t.requires_grad:
        t.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
    return t

# --------------------- per-seg NLL helper ---------------------------------- #
def _perseg_nll30k(lam: torch.Tensor, y: torch.Tensor, len_kb: torch.Tensor) -> torch.Tensor:
    lam_safe = lam.clamp(1e-9, 1e6)
    sum_log  = (y * torch.log(lam_safe)).sum(dim=1)
    integ    = (lam_safe * len_kb).sum(dim=1)
    neg_ll   = -(sum_log - integ)
    seg_len  = (len_kb.sum(dim=1) + 1e-9)
    return (neg_ll / seg_len) * 30.0

# ---------- NEW: Robust batch weights (Huber, organ-wise) ------------------ #
def _robust_seg_weights_from_batch(
    lam: torch.Tensor,
    y_b: torch.Tensor,
    len_b: torch.Tensor,
    *,
    huber_factor: float,
    use_mad: bool,
    organ_ids: torch.Tensor,
) -> torch.Tensor:
    """
    배치에서 organ별로 세그 평균 residual 분포 스케일(IQR/MAD)을 계산 → δ_org = huber_factor×scale_org
    → 각 세그(샘플)의 허버 가중을 organ-wise로 반환 (shape: (B,))
    """
    with torch.no_grad():
        lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)
        mu = lam * len_b

        # 세그먼트 평균 residual (per-seg scalar)
        L = (len_b > 0).float().sum(dim=1).clamp_min(1.0)  # (B,)
        r_seg = ((y_b - mu).sum(dim=1) / L).detach().cpu().numpy()  # (B,)
        oid_np = organ_ids.detach().cpu().numpy()

        # organ-wise scale (IQR/MAD)
        delta_by_org: Dict[int, float] = {}
        for oid in np.unique(oid_np):
            vals = r_seg[oid_np == oid]
            if vals.size == 0:
                continue
            if use_mad:
                med = np.median(vals); mad = np.median(np.abs(vals - med))
                scale = 1.4826 * mad
            else:
                q1, q3 = np.percentile(vals, [25, 75])
                scale = (q3 - q1)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            delta_by_org[int(oid)] = float(huber_factor) * float(scale)

        # 허버 가중 w_i = min(1, δ_org/|r_i|)
        w = np.ones_like(r_seg, dtype=np.float32)
        for i, (ri, oi) in enumerate(zip(r_seg, oid_np)):
            d = float(delta_by_org.get(int(oi), float(huber_factor)))
            a = abs(float(ri))
            w[i] = 1.0 if a <= d else d / (a + 1e-9)

        return torch.tensor(w, device=lam.device, dtype=lam.dtype)  # (B,)



# --------------------------------------------------------------------------- #
# Manifest loader                                                             #
# --------------------------------------------------------------------------- #
def _load_manifest(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("manifest JSON must be a list of entries")
        rows = data
    else:
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception:
            df = pd.read_csv(p, sep="\t", dtype=str)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        need = {"organ", "cls_file", "feat_file"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"manifest must contain columns {sorted(need)}")
        rows = df.fillna("").to_dict("records")

    normed = []
    for r in rows:
        organ = str(r["organ"]).strip()
        cls_f = str(r["cls_file"]).strip()
        fea_f = str(r["feat_file"]).strip()
        mut_f = str(r.get("mutations_file", "")).strip() or None
        oid   = r.get("organ_id", None)
        if oid is not None and str(oid).strip() != "":
            oid = int(str(oid).strip())
            if oid < 0: raise ValueError(f"organ_id must be >=0: {oid}")
        else:
            oid = None
        if not (organ and cls_f and fea_f):
            continue
        normed.append(dict(
            organ=organ, cls_file=cls_f, feat_file=fea_f,
            mutations_file=mut_f, organ_id=oid
        ))
    if not normed:
        raise ValueError("manifest is empty after normalization")
    return normed



def _build_organ_id_map_from_manifest(entries: list) -> dict:
    given = {e["organ"]: e["organ_id"] for e in entries if e.get("organ_id") is not None}
    if given:
        if len(set(given.values())) != len(given.values()):
            raise ValueError("organ_id duplicates found")
        # ★ NEW: UNK 추가
        if "__UNK__" not in given:
            given["__UNK__"] = max(given.values()) + 1
        return given

    m = {}
    for e in entries:
        o = e["organ"]
        if o not in m:
            m[o] = len(m)
    # ★ NEW: UNK 추가
    if "__UNK__" not in m:
        m["__UNK__"] = len(m)
    return m


# ===== Pretty per-organ logger ============================================= #
def _format_per_organ_table(po: dict, cols: int = 3, order: str = "desc",
                            name_max: int = 16, val_fmt: str = "{:>7.3f}",
                            max_items: int = None) -> str:
    if not po:
        return "(no per-organ stats)"
    items = sorted(po.items(), key=lambda x: x[1], reverse=(order == "desc"))
    if max_items:
        items = items[:max_items]
    def _trim(s: str, m: int) -> str: return s if len(s) <= m else s[:m-1] + "…"
    name_w = min(max(len(k) for k,_ in items), name_max)
    cells = [f"{_trim(k,name_max):<{name_w}} {val_fmt.format(v)}" for k,v in items]
    rows = ["  ".join(cells[i:i+cols]) for i in range(0,len(cells),cols)]
    return "\n".join(rows)

def _format_per_organ_topk(po: dict, k: int) -> str:
    if not po: return ""
    items = sorted(po.items(), key=lambda x: x[1], reverse=True)
    top = ", ".join(f"{k}={v:.2f}" for k,v in items[:k])
    bot = ", ".join(f"{k}={v:.2f}" for k,v in items[-k:])
    return f"top{k}: {top} | bottom{k}: {bot}"

# --------------------------------------------------------------------------- #
# Balanced & Mixed Batch Samplers                                            #
# --------------------------------------------------------------------------- #
class BalancedBatchSampler(Sampler[List[int]]):
    """
    한 배치에 organ별로 같은 개수(k)씩 담는 BatchSampler.
    - replacement=True 방식으로 뽑아 epoch 동안 고갈 문제를 피함.
    - 유효 배치 크기 = k * n_organs (batch_size 인자와 무관)
    """
    def __init__(self, organ_to_indices: Dict[int, List[int]], per_organ_k: int, epoch_size: int):
        super().__init__(None)
        self.org_ids = sorted(organ_to_indices.keys())
        self.per_organ_k = max(1, int(per_organ_k))
        self.maps = {oid: list(lst) for oid, lst in organ_to_indices.items()}
        for oid in self.maps:
            random.shuffle(self.maps[oid])
        self.epoch_size = int(epoch_size)
        self.batch_size = self.per_organ_k * len(self.org_ids)

    def __iter__(self):
        need = math.ceil(self.epoch_size / self.batch_size)
        pools = {oid: deque(self.maps[oid]) for oid in self.org_ids}
        for _ in range(need):
            batch = []
            for oid in self.org_ids:
                take = []
                for _k in range(self.per_organ_k):
                    if pools[oid]:
                        take.append(pools[oid].popleft())
                    else:
                        take.append(random.choice(self.maps[oid]))
                batch.extend(take)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)


# -------- NEW: Organ 내부에서 Chrom 라운드로빈으로 뽑는 균형 샘플러 -------- #
class BalancedOrgChrBatchSampler(Sampler[List[int]]):
    """
    Organ-균형 + Organ 내부 Chrom-층화 라운드로빈 샘플러.
    - 한 배치에 organ별 per_organ_k개씩.
    - organ 내부에서는 chrom을 순환하며 1개씩 뽑음.
    """
    def __init__(self,
                 organ_chr_to_indices: Dict[int, Dict[int, List[int]]],
                 per_organ_k: int,
                 epoch_size: int):
        super().__init__(None)
        self.org_ids = sorted(organ_chr_to_indices.keys())
        self.per_organ_k = max(1, int(per_organ_k))
        self.epoch_size = int(epoch_size)

        # 각 organ 내 chrom → deque
        self.maps: Dict[Tuple[int,int], List[int]] = {}
        self.pools: Dict[Tuple[int,int], deque] = {}
        self.org_cids: Dict[int, List[int]] = {}
        for oid, cid_map in organ_chr_to_indices.items():
            cids = sorted(cid_map.keys())
            self.org_cids[oid] = cids
            for cid in cids:
                lst = list(cid_map[cid])
                if not lst: continue
                random.shuffle(lst)
                self.maps[(oid,cid)]  = lst
                self.pools[(oid,cid)] = deque(lst)
        self.chr_pos = {oid: 0 for oid in self.org_ids}
        self.batch_size = self.per_organ_k * len(self.org_ids)

    def __iter__(self):
        need = math.ceil(self.epoch_size / self.batch_size)
        for _ in range(need):
            batch = []
            for oid in self.org_ids:
                cids = self.org_cids.get(oid, [])
                if not cids:
                    # 해당 organ에 인덱스가 없으면 skip
                    continue
                for _k in range(self.per_organ_k):
                    cid = cids[self.chr_pos[oid] % len(cids)]
                    self.chr_pos[oid] += 1
                    key = (oid, cid)
                    dq  = self.pools.get(key, None)
                    base= self.maps.get(key, None)
                    if dq is None or base is None or len(base) == 0:
                        continue
                    if dq:
                        idx = dq.popleft()
                    else:
                        # 고갈 시 다시 채움(replacement)
                        refill = list(base)
                        random.shuffle(refill)
                        dq.extend(refill)
                        idx = dq.popleft()
                    batch.append(idx)
            random.shuffle(batch)
            if batch:
                yield batch

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)


# -------- NEW: 중요도(라벨/난이도) 샘플러 (장기 무시, 전역) ---------------- #
class ImportanceBatchSampler(Sampler[List[int]]):
    """
    정적 라벨 중요도 × 동적 난이도 EMA 기반 가중치로 전역에서 샘플링.
    - weights_static: 길이 N의 numpy/torch list
    - shared_state["ema_loss"]: dict(train_idx -> ema_value)
    - ema_gamma: 동적 난이도 지수
    """
    def __init__(self,
                 num_items: int,
                 weights_static: List[float],
                 batch_size: int,
                 num_batches: int,
                 shared_state: Dict[str, Any],
                 ema_gamma: float = 0.5):
        super().__init__(None)
        self.N = int(num_items)
        self.w_static = np.asarray(weights_static, dtype=np.float64)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.shared = shared_state
        self.ema_gamma = float(ema_gamma)

    def __iter__(self):
        # 동적 난이도 EMA 읽어와서 가중치 구성
        ema = self.shared.get("ema_loss", {})
        if len(ema) == 0:
            w_dyn = np.ones(self.N, dtype=np.float64)
        else:
            arr = np.ones(self.N, dtype=np.float64)
            vals = np.array(list(ema.values()), dtype=np.float64)
            mean_ema = float(np.mean(vals)) if np.isfinite(vals).all() and len(vals) > 0 else 1.0
            mean_ema = max(mean_ema, 1e-6)
            for k, v in ema.items():
                if 0 <= k < self.N:
                    arr[k] = (float(v) / mean_ema) ** self.ema_gamma
            w_dyn = np.clip(arr, 0.5, 3.0)

        w = self.w_static * w_dyn
        w = np.clip(w, 1e-6, None)
        p = w / w.sum()

        total = self.num_batches * self.batch_size
        # 한 번에 샘플링하고 배치로 분할
        idx_all = np.random.choice(self.N, size=total, replace=True, p=p)
        for i in range(self.num_batches):
            s = i * self.batch_size
            e = s + self.batch_size
            yield idx_all[s:e].tolist()

    def __len__(self):
        return self.num_batches


# -------- NEW: 50/50 혼합 배치 샘플러 -------------------------------------- #
class CombinedBatchSampler(Sampler[List[int]]):
    """
    두 batch_sampler를 50/50로 섞어서 배치 단위로 내보냄.
    - len = len_bal + len_imp (대략 동일하도록 구성)
    - 내부 sampler는 소진 시 자동 순환
    """
    def __init__(self, sampler_a: Sampler, sampler_b: Sampler,
                 num_batches_a: int, num_batches_b: int):
        super().__init__(None)
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        self.num_batches_a = int(num_batches_a)
        self.num_batches_b = int(num_batches_b)

    def __iter__(self):
        it_a = iter(self.sampler_a)
        it_b = iter(self.sampler_b)
        na, nb = self.num_batches_a, self.num_batches_b
        total = na + nb
        # 번갈아가며 내보내되, 개수 차이는 뒤에서 채움
        a_left, b_left = na, nb
        for i in range(total):
            use_a = (i % 2 == 0 and a_left > 0) or b_left == 0
            if use_a:
                try:
                    batch = next(it_a)
                except StopIteration:
                    it_a = iter(self.sampler_a)
                    batch = next(it_a)
                a_left -= 1
                yield batch
            else:
                try:
                    batch = next(it_b)
                except StopIteration:
                    it_b = iter(self.sampler_b)
                    batch = next(it_b)
                b_left -= 1
                yield batch

    def __len__(self):
        return self.num_batches_a + self.num_batches_b


# --------------------------------------------------------------------------- #
# CLS Bank (lite mode) loader                                                 #
# --------------------------------------------------------------------------- #
def _load_common_cls_bank(path: str, fp16: bool = False) -> Dict[Tuple[str,int], np.ndarray]:
    print(f"[CLS] Loading common CLS bank: {path}", flush=True)
    with open(path, "rb") as f:
        cls_list = pickle.load(f)
    dtype = np.float16 if fp16 else np.float32
    bank: Dict[Tuple[str,int], np.ndarray] = {}
    for chrom, widx, _sbp, _ebp, vec, _y in cls_list:
        arr = np.asarray(vec, dtype=np.float32).astype(dtype, copy=False)
        bank[(str(chrom), int(widx))] = arr
    d = next(iter(bank.values())).shape[-1] if bank else None
    print(f"[CLS] entries={len(bank):,} dim={d} dtype={dtype}", flush=True)
    return bank

# --------------------------------------------------------------------------- #
# Hybrid data shaping helpers (hardcoded)                                     #
# --------------------------------------------------------------------------- #
_HYB_SUBSAMPLE_RATIO = {10: 0.10, 50: 0.4, 100: 0.50}
_HYB_REFINE_100K_FRAC = 0.10
_HYB_REFINE_MODE      = "half"
_HYB_REFINE_KEEP_ORIG = True

def _approx_span_bp(seg: dict) -> Optional[int]:
    try:
        s = seg.get("start_array", None)
        e = seg.get("end_array", None)
        if s is not None and e is not None:
            s = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
            e = e.cpu().numpy() if isinstance(e, torch.Tensor) else e
            L = min(len(s), len(e))
            if L > 0:
                return int(e[L-1]) - int(s[0]) + 1
    except Exception:
        pass
    return None

def _approx_size_kb(seg: dict) -> Optional[int]:
    span = _approx_span_bp(seg)
    if span is not None:
        kb = span / 1000.0
        for tgt in (10, 50, 100):
            if abs(kb - tgt) <= tgt * 0.20:
                return tgt
        return min((10, 50, 100), key=lambda t: abs(kb - t))
    if "cls_array" in seg and hasattr(seg["cls_array"], "shape"):
        T = seg["cls_array"].shape[0]
        if 8 <= T <= 12: return 10
        if 40 <= T <= 60: return 50
        if 80 <= T <= 120: return 100
    return None

def _is_100kb_segment(seg: dict) -> bool:
    return (_approx_size_kb(seg) == 100)

def _split_segment_even(seg: dict, n_parts: int) -> list:
    def _slice(obj, sl):
        if isinstance(obj, torch.Tensor): return obj[sl]
        if hasattr(obj, "shape"): return obj[sl]
        if isinstance(obj, (list, tuple)): return obj[sl]
        return obj
    if "cls_array" in seg and hasattr(seg["cls_array"], "shape"):
        T = seg["cls_array"].shape[0]
    elif "start_array" in seg and hasattr(seg["start_array"], "shape"):
        T = seg["start_array"].shape[0]
    else:
        return [seg]
    part = max(1, T // n_parts)
    idxs = [i*part for i in range(n_parts)] + [T]
    idxs[-1] = T
    pieces = []
    for k in range(n_parts):
        a, b = idxs[k], idxs[k+1]
        if b <= a: continue
        new_seg = {**seg}
        for key, val in list(seg.items()):
            if key.endswith("_array") and hasattr(val, "shape"):
                new_seg[key] = _slice(val, slice(a, b))
        for key in ("y_array","length_array"):
            if key in seg and hasattr(seg[key], "shape"):
                new_seg[key] = _slice(seg[key], slice(a, b))
        if "global_idx" in new_seg: del new_seg["global_idx"]
        pieces.append(new_seg)
    return pieces

def refine_100k_segments_for_organ(segs: list, frac: float, mode: str, keep_original: bool, rng: random.Random) -> list:
    if frac <= 0.0: return segs
    hundred = [i for i, s in enumerate(segs) if _is_100kb_segment(s)]
    if not hundred: return segs
    k = max(1, int(round(len(hundred) * min(1.0, max(0.0, frac)))))
    pick = set(rng.sample(hundred, k))
    n_parts = 2 if mode == "half" else 10
    out = []
    for i, s in enumerate(segs):
        if i in pick:
            parts = _split_segment_even(s, n_parts=n_parts)
            if keep_original: out.append(s)
            out.extend(parts)
        else:
            out.append(s)
    return out

def subsample_by_size_organ(segs: list, ratios: dict, rng: random.Random) -> list:
    groups = defaultdict(list)
    for s in segs:
        sz = _approx_size_kb(s)
        groups[sz].append(s)
    kept = []
    for sz, lst in groups.items():
        r = float(ratios.get(sz, 1.0))
        if r >= 1.0:
            kept.extend(lst); continue
        n_keep = max(1, int(round(len(lst) * max(0.0, min(1.0, r)))))
        kept.extend(rng.sample(lst, n_keep))
    return kept

def _print_size_hist(label: str, organ: str, segs: list):
    cnt = Counter([_approx_size_kb(s) for s in segs])
    total = len(segs)
    parts = []
    for sz in (10,50,100,None):
        c = cnt.get(sz, 0)
        parts.append(f"{('None' if sz is None else f'{sz}kb')}: {c:,} ({(100*c/total if total else 0):.1f}%)")
    print(f"[HYB] {label:<6s} organ={organ:<16s} total={total:,}  " + "  ".join(parts), flush=True)

# --------------------------------------------------------------------------- #
# Calibration (Huber-like) with organ embed                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def calibrate_log_c_huber_like_training(
    model_c, loader, device,
    huber_factor: float = 3.0,
    use_mad: bool = False,
    label_roll: bool = False,
    roll_width: int = 1
):
    """
    Organ-wise Huber IRLS + **organ별 c 보정**:
      base = lam / s(org)  (s = nh.scale_for_ids)
      c*_org = (∑ y) / (∑ ∫ base dt)
      log_c_org ← log(c_new_org)  (tanh-squash로 과한 이동 억제)
    """
    # 내부 상수(추가 CLI 없음)
    _MIN_SEG_PER_ORG = 20           # 표본 적은 organ 스킵
    _SQUASH_K_LOG    = float(np.log(1.15))  # 한 번에 ~±15% 제한(부드럽게)
    eps = _EPS

    for m in model_c.values(): m.eval()
    nh = unwrap(model_c["nhpp_head"])

    # organ-wise 누적자
    num_by_org, den_by_org, cnt_by_org = {}, {}, {}

    for b in loader:
        cls_b  = b["cls_array"].to(device)
        feat_b = b["feat_array"].to(device)
        y_b    = _safe_count_(b["y_array"].to(device))
        len_b  = _safe_len_(b["length_array"].to(device))
        cid_b  = b["chrom_id"].to(device)
        oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]],
                              device=device, dtype=torch.long)
        key_pad = (len_b <= 0)

        feat_emb = model_c["feature_embedder"](feat_b)
        fused, _ = model_c["feat_cls_fusion"](
            cls_b, feat_emb, chrom_id=cid_b, valid_mask=(len_b > 0),
            tissue_ids=oid_b, epoch_idx=None
        )
        chr_emb  = model_c["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        org_emb  = model_c["organ_embedder"](oid_b).unsqueeze(1).expand_as(fused)

        tok   = model_c["organ_token"](oid_b)
        delta = model_c["orgchr_lr"](oid_b, cid_b)
        x = fused + chr_emb + org_emb + delta.expand_as(fused)
        x = torch.cat([tok, x], dim=1)
        if key_pad is not None:
            pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
            key_pad = torch.cat([pad0, key_pad], dim=1)
        x = model_c["cond_film"](x, oid_b)

        out = model_c["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_b)
        out = model_c["cond_film"](out, oid_b)[:, 1:, :]

        lam = _safe_rate_(model_c["nhpp_head"](out.contiguous(), tissue_ids=oid_b))
        if label_roll and roll_width > 1:
            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=roll_width)
            lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

        # ★ 현재 organ 스케일로 나눠 base 복원
        s = nh.scale_for_ids(oid_b)              # (B,1)
        base = (lam / s).clamp_min(1e-12)        # (B,T)

        # organ별 num/den 누적
        for i, seg in enumerate(b["raw_segments"]):
            L   = seg["cls_array"].shape[0]
            oid = int(seg["organ_id"])
            y_sum   = float(y_b[i, :L].sum().item())
            den_int = float((base[i, :L] * len_b[i, :L]).sum().item())
            num_by_org[oid] = num_by_org.get(oid, 0.0) + y_sum
            den_by_org[oid] = den_by_org.get(oid, 0.0) + den_int
            cnt_by_org[oid] = cnt_by_org.get(oid, 0) + 1

    # organ별 log-c 업데이트
    updated = []
    for oid in sorted(num_by_org.keys()):
        if cnt_by_org.get(oid, 0) < _MIN_SEG_PER_ORG:
            continue
        num = num_by_org[oid]; den = den_by_org[oid]
        if not (np.isfinite(num) and np.isfinite(den)) or num <= eps or den <= eps:
            continue

        c_prev_org = math.exp(nh.get_logc_per_organ(oid))
        c_star_org = max(num / den, eps)

        # 부드러운 억제: log-space tanh squash (하이퍼 추가 없음)
        logr      = float(np.log(c_star_org / max(c_prev_org, eps)))
        ratio_eff = float(np.exp(_SQUASH_K_LOG * np.tanh(logr)))
        c_new_org = c_prev_org * ratio_eff

        nh.set_logc_per_organ(oid, new_logc=math.log(max(c_new_org, eps)))
        updated.append((oid, c_prev_org, c_star_org, c_new_org))

    if not updated:
        print("[CAL-HUBER-TRAIN] per-organ: no update (insufficient data or invalid c*)", flush=True)
        return

    msg = ", ".join([f"org#{oid}: {p:.3g}→{n:.3g} (star={s:.3g})" for oid, p, s, n in updated[:6]])
    more = f" +{len(updated)-6} more" if len(updated) > 6 else ""
    print(f"[CAL-HUBER-TRAIN] per-organ updated: {msg}{more}", flush=True)


# --------------------------------------------------------------------------- #
# Pretrain (multi-organ)                                                      #
# --------------------------------------------------------------------------- #
def pretrain_and_predict(args):
    set_seed(args.seed)

    # (옵션) torch 스레드
    if getattr(args, "torch_threads", None):
        try:
            torch.set_num_threads(int(args.torch_threads))
            print(f"[INFO] torch num_threads = {torch.get_num_threads()}", flush=True)
        except Exception as e:
            print(f"[WARN] set_num_threads failed: {e}", flush=True)

    # ------------------------- 데이터 로딩 (multi-organ) ------------------- #
    manifest = _load_manifest(args.manifest)
    print(f"[INFO] manifest entries = {len(manifest)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"[INFO] device={device}, #GPUs={n_gpu}", flush=True)

    # organ id map
    organ_id_map = _build_organ_id_map_from_manifest(manifest)
    organ_name_map = {v: k for k, v in organ_id_map.items()}
    organ_names_ordered = [organ_name_map[i] for i in range(len(organ_name_map))]  # 고정 열 순서
    organ_names_ordered = [n for n in organ_names_ordered if n != "__UNK__"]


    # [LOG] CSV 경로
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    step_alpha_csv = log_dir / "step_alpha.csv"
    step_loss_csv  = log_dir / "step_loss.csv"
    epoch_csv      = log_dir / "epoch_metrics.csv"
    
    step_alpha_len_csv = log_dir / "step_alpha_by_len.csv"   # 열: step, epoch, alpha_10kb, alpha_50kb, alpha_100kb
    final_alpha_len_csv = log_dir / "final_alpha_by_len.csv" # 열: split, alpha_10kb, alpha_50kb, alpha_100kb


    def _csv_append_row(path: Path, header: list, row: list):
        write_header = (not path.exists() or path.stat().st_size == 0)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow(row)

    # 공용 CLS 라이트 모드 분기
    use_lite   = bool(getattr(args, "segs_lite", False))
    cls_bank   = None
    if use_lite:
        if not getattr(args, "common_cls", None):
            raise ValueError("--segs-lite requires --common-cls")
        cls_bank = _load_common_cls_bank(args.common_cls, fp16=bool(getattr(args, "cls_fp16", False)))

    # feature dim consistency + build segments per organ
    feature_dim_ref = None
    per_organ_segments: Dict[str, List[Dict[str,Any]]] = {}
    for entry in manifest:
        organ = entry["organ"]; oid = organ_id_map[organ]
        with open(entry["cls_file"], "rb") as f:
            cls_list = pickle.load(f)
        with open(entry["feat_file"], "rb") as f:
            feature_dict = pickle.load(f)

        any_feat = next(iter(feature_dict.values()))
        feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else any_feat.shape[0]
        if feature_dim_ref is None: feature_dim_ref = feature_dim
        elif feature_dim_ref != feature_dim:
            raise ValueError(f"[{organ}] feature_dim mismatch: {feature_dim} vs ref {feature_dim_ref}")

        if use_lite:
            segs = segment_indices_fixed_lengths(
                cls_list, feature_dict,
                seg_len_list=tuple(args.segment_lengths),
                discard_leftover=getattr(args, "discard_leftover", False),
                overlap_factor=args.overlap_factor,
            )
        else:
            segs = segment_cls_embeddings_fixed_lengths(
                cls_list, feature_dict,
                seg_len_list=args.segment_lengths,
                discard_leftover=getattr(args, "discard_leftover", False),
                overlap_factor=args.overlap_factor,
            )

        # (옵션) 라벨 붙이기
        if entry.get("mutations_file"):
            print(f"[LABEL:{organ}] building 1kb labels (PASS only): {entry['mutations_file']}", flush=True)
            cls_bins = _bins_from_cls_list(cls_list)
            ev       = _load_mutations_events(entry["mutations_file"], use_midpoint=True, require_pass=True)
            y_map    = _build_y_map_from_mutations(cls_bins, ev)
            _attach_labels_from_y_map(segs, y_map)
            print(f"[LABEL:{organ}] y_map bins = {len(y_map):,}", flush=True)

        # 메타 부여
        for s in segs:
            s["organ"] = organ
            s["organ_id"] = oid

        # ★★★ NEW: 24개 염색체만 유지 (chr1-22, chrX, chrY) — 비정규 chr 드롭 ★★★
        _before = len(segs)
        segs = [s for s in segs if str(s.get("chrom")) in CHROM_LIST_24]
        _drop = _before - len(segs)
        if _drop > 0:
            print(f"[FILTER:{organ}] dropped {_drop:,} segments with non-24 chr (e.g., chrM/chrUn/alt)", flush=True)

        # Hybrid 샘플링
        _print_size_hist("BASE", organ, segs)
        rng_sub = random.Random(args.seed + oid * 131)
        segs = subsample_by_size_organ(segs, _HYB_SUBSAMPLE_RATIO, rng_sub)
        _print_size_hist("SUBSP", organ, segs)
        rng_ref = random.Random(args.seed + oid * 911)
        segs = refine_100k_segments_for_organ(
            segs,
            frac=_HYB_REFINE_100K_FRAC,
            mode=_HYB_REFINE_MODE,
            keep_original=_HYB_REFINE_KEEP_ORIG,
            rng=rng_ref
        )
        _print_size_hist("REFIN", organ, segs)

        per_organ_segments[organ] = segs
        print(f"[INFO] organ={organ:<16s} segs={len(segs):,}", flush=True)

    # 전체 세그 합치고 global_idx 재부여
    all_segments = list(itertools.chain.from_iterable(per_organ_segments.values()))
    for gid, seg in enumerate(all_segments):
        seg["global_idx"] = gid
    print(f"[INFO] #all_segments (multi-organ) = {len(all_segments):,}", flush=True)

    # 조직별/염색체별 split (chrom-level split)
    train_segs, val_segs = [], []
    for organ, segs in per_organ_segments.items():
        c_map = defaultdict(list)
        for seg in segs: c_map[seg["chrom"]].append(seg)
        for _, lst in c_map.items():
            random.shuffle(lst)
            n = len(lst); n_train = int(n * 0.9)
            train_segs.extend(lst[:n_train]);  val_segs.extend(lst[n_train:])

    ds_train = SegmentDataset(train_segs)
    ds_val   = SegmentDataset(val_segs)
    ds_all   = SegmentDataset(all_segments)

    # collate (classic / lite)
    chrom_id_map = build_chrom_id_map(None)
    if use_lite:
        collate_train = partial(segment_collate_fn_lite, chrom_id_map=chrom_id_map, cutmix_p=args.cutmix_p, cls_bank=cls_bank)
        collate_eval  = partial(segment_collate_fn_lite,  chrom_id_map=chrom_id_map, cutmix_p=0.0, cls_bank=cls_bank)
    else:
        collate_train = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=args.cutmix_p)
        collate_eval  = partial(segment_collate_fn, chrom_id_map=chrom_id_map, cutmix_p=0.0)

    pin = (device.type == "cuda")

    # ==== Balanced + Mixed Samplers 준비 =================================== #
    # organ-only map (기존)
    organ_to_indices: Dict[int, List[int]] = defaultdict(list)
    # organ×chrom map (NEW)
    organ_chr_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    for idx, seg in enumerate(ds_train.segments):
        oid = seg["organ_id"]
        organ_to_indices[oid].append(idx)
        # ★ 가드: 24개 외 염색체는 건너뜀
        chrom = str(seg["chrom"])
        if chrom not in CHROM_LIST_24:
            continue
        cid = chrom_id_map[chrom]
        organ_chr_to_indices[oid][cid].append(idx)

    # train_idx <-> global_idx 매핑 (동적 중요도 EMA 업데이트용)
    trainidx_by_global = {seg["global_idx"]: i for i, seg in enumerate(ds_train.segments)}

    # 정적 중요도 가중치 (라벨/이벤트/희귀 chrom)
    chrom_counts = Counter([chrom_id_map[str(seg["chrom"])] for seg in ds_train.segments if str(seg["chrom"]) in CHROM_LIST_24])
    chrom_inv_sqrt = {cid: (count ** -0.5) for cid, count in chrom_counts.items()}
    # 정규화 (평균≈1)
    mean_inv = sum(chrom_inv_sqrt.values()) / max(len(chrom_inv_sqrt), 1)
    if mean_inv <= 0: mean_inv = 1.0
    for cid in chrom_inv_sqrt:
        chrom_inv_sqrt[cid] /= mean_inv

    static_weights = []
    for seg in ds_train.segments:
        # 라벨 기반
        w_label = 1.0
        y = seg.get("y_array", None)
        try:
            if y is not None:
                if isinstance(y, torch.Tensor): y_sum = float(y.sum().item())
                else: y_sum = float(np.asarray(y).sum())
                pos = (y_sum > 0.0)
                w_label *= (1.0 + (3.0 if pos else 0.0))  # pos면 ×4
                w_label *= min(5.0, 1.0 + math.log1p(max(0.0, y_sum)))
        except Exception:
            pass
        # 희귀 chrom 보정(가벼움)
        chrom = str(seg["chrom"])
        if chrom in CHROM_LIST_24:
            cid = chrom_id_map[chrom]
            w_chr = chrom_inv_sqrt.get(cid, 1.0)
        else:
            w_chr = 1.0
        w = max(1e-3, w_label * w_chr)
        static_weights.append(w)

    # 혼합 비율 및 배치 크기
    if getattr(args, "balance_per_batch", False):
        n_org = len(organ_to_indices)
        if n_org == 0:
            raise RuntimeError("No organs found for balanced batching.")
        if getattr(args, "per_organ_samples", None):
            k_per = int(args.per_organ_samples)
        else:
            k_per = max(1, int(args.batch_size // n_org))
        eff_bs = k_per * n_org
        print(f"[INFO] BalancedOrgChrBatchSampler enabled: per_organ_k={k_per}, n_org={n_org}, effective_batch_size={eff_bs}", flush=True)

        # A) Organ×Chrom 균형 샘플러
        bb_sampler = BalancedOrgChrBatchSampler(
            organ_chr_to_indices, per_organ_k=k_per, epoch_size=len(ds_train)
        )
        # B) 중요도 샘플러 (eff_bs 크기로)
        # 전체 배치 수 동일하게 맞추고 30/70(중요도/균형) 비율
        total_batches = math.ceil(len(ds_train) / eff_bs)
        imp_ratio = 0.30
        num_b = max(1, int(round(total_batches * imp_ratio)))  # 중요도
        num_a = max(1, total_batches - num_b)                  # 균형

        # 동적 난이도 공유 상태
        shared_state = {"ema_loss": defaultdict(lambda: 1.0)}
        imp_sampler = ImportanceBatchSampler(
            num_items=len(ds_train),
            weights_static=static_weights,
            batch_size=eff_bs,
            num_batches=num_b,
            shared_state=shared_state,
            ema_gamma=float(getattr(args, "ema_gamma", 0.5))
        )

        train_loader = DataLoader(
            ds_train,
            batch_sampler=CombinedBatchSampler(bb_sampler, imp_sampler, num_batches_a=num_a, num_batches_b=num_b),
            collate_fn=collate_train,
            num_workers=args.num_data_workers,
            pin_memory=pin
        )
    else:
        shared_state = {"ema_loss": defaultdict(lambda: 1.0)}
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_train, num_workers=args.num_data_workers, pin_memory=pin
        )

    train_loader_infer = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin
    )
    all_loader = DataLoader(
        ds_all, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=args.num_data_workers, pin_memory=pin
    )

    # ------------------------- 모델 --------------------------------------- #
    feat_embedder      = FeatureEmbedder(feature_dim_ref, hidden_dim=args.d_model)
    feat_cls_fusion = FeatClsFusionCtx(
        hidden_dim=args.d_model,
        chrom_n=len(CHROM_LIST_24),
        w_tok=0.4, w_seg=1.25, w_dom=0.15,
        use_stream_dropout=True, stream_dropout_p=0.05,
        use_alpha_warmup=True, alpha_target=0.50, warmup_epochs=3, warmup_lambda=1e-4,
        alpha0_init=0.5,
        alpha_min=getattr(args, "alpha_min", 0.10),
        enforce_alpha_min=False,
        logit_temperature=1.0,
        logit_offset=0.20,
        num_organs=len(organ_id_map),
        use_per_organ_alpha_bias=True,
        organ_bias_init=0.0,
        organ_bias_l2=1e-4,
        use_bias_warmup=True,
        bias_warmup_epochs=3,
        alpha_share_mode="token",   # "token"(기본) | "segment" | "window"
        alpha_window=0,                # window 모드에서만 >1 사용
        
        # ★ 병렬 CLS-CNN + organ-specific CLS bias 하드코딩
        use_cls_bias=True,
        cls_bias_scale=0.2,            # CLS 바이어스 영향(보통 0.1~0.3 권장)

        use_cls_cnn=True,
        cls_cnn_branches=[(3,1), (3,2)],  # 다중 dilations: R ≈ 5
        cls_cnn_scale=0.2              # CNN residual 영향(0.1~0.3 권장)
        )

    _fc = getattr(feat_cls_fusion, "core", feat_cls_fusion)
    if not hasattr(_fc, "alpha_min"):
        raise RuntimeError("[CHECK] FeatClsFusionCtx.core.alpha_min이 없습니다. embedders 모듈을 최신본으로 업데이트하세요.")
    print(f"[CHECK] fusion alpha_min = {float(_fc.alpha_min):.3f} (enforce={getattr(_fc,'enforce_alpha_min',False)})", flush=True)
    print("[CHECK] fusion params:",
          f"w_tok={getattr(_fc,'w_tok','NA')}, w_seg={getattr(_fc,'w_seg','NA')}, w_dom={getattr(_fc,'w_dom','NA')}, ",
          f"drop_p={getattr(_fc,'stream_dropout_p','NA')}, ",
          f"T={getattr(_fc,'logit_temperature','NA')}, offset={getattr(_fc,'logit_offset','NA')}",
          flush=True)

    chrom_embedder     = ChromosomeEmbedder(len(CHROM_LIST_24), d=args.d_model)
    organ_embedder     = ChromosomeEmbedder(len(organ_id_map),  d=args.d_model)
    # GlobalTransformerEncoder는 업데이트된 버전(organ_ids 지원, 내부 per-layer FiLM 가능)
    global_transformer = GlobalTransformerEncoder(d_model=args.d_model, nhead=args.nhead,
                                                  num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                                                  dropout=args.dropout, max_seq_len=args.max_seq_len,
                                                  num_organs=len(organ_id_map))
    nhpp_head          = ConditionalNHPPHead(
        hidden_dim=args.d_model,
        cond=CondCfg(mode="per_head", num_tissues=len(organ_id_map), tie_ln=True),
        rate_min=1e-9, rate_max=1e6
    )

    # ---- CondFiLM & OrganToken (입출구 FiLM + 도메인 토큰) ----
    class CondFiLM(nn.Module):
        def __init__(self, num_organs:int, d_model:int, hidden:int=128):
            super().__init__()
            self.org_emb = nn.Embedding(num_organs, hidden)
            self.proj_g  = nn.Linear(hidden, d_model)
            self.proj_b  = nn.Linear(hidden, d_model)
            nn.init.zeros_(self.proj_g.weight); nn.init.zeros_(self.proj_g.bias)
            nn.init.zeros_(self.proj_b.weight); nn.init.zeros_(self.proj_b.bias)
        def forward(self, x: torch.Tensor, organ_ids: torch.Tensor):
            h = self.org_emb(organ_ids)
            gamma = self.proj_g(h).unsqueeze(1)
            beta  = self.proj_b(h).unsqueeze(1)
            return x * (1.0 + gamma) + beta

    class OrganToken(nn.Module):
        def __init__(self, num_organs:int, d_model:int):
            super().__init__()
            self.emb = nn.Embedding(num_organs, d_model)
        def forward(self, organ_ids: torch.Tensor):
            return self.emb(organ_ids).unsqueeze(1)

    # ---- NEW: Organ×Chrom 저랭크 보정 모듈 ----
    class OrgChrLowRank(nn.Module):
        def __init__(self, num_org: int, num_chr: int, d: int, r: int = 16):
            super().__init__()
            self.U = nn.Embedding(num_org, r)   # organ factor
            self.V = nn.Embedding(num_chr, r)   # chrom factor
            self.P = nn.Linear(r, d, bias=False)
        def forward(self, organ_ids: torch.Tensor, chrom_ids: torch.Tensor):
            # organ_ids, chrom_ids: (B,)
            h = self.U(organ_ids) * self.V(chrom_ids)    # (B, r)
            delta = self.P(h).unsqueeze(1)               # (B, 1, D)
            return delta

    cond_film   = CondFiLM(num_organs=len(organ_id_map), d_model=args.d_model, hidden=128)
    organ_token = OrganToken(len(organ_id_map), args.d_model)
    orgchr_lr   = OrgChrLowRank(num_org=len(organ_id_map), num_chr=len(CHROM_LIST_24), d=args.d_model, r=16)

    # ---- NEW: Org×Chr 보정 0-init (delta≈0로 시작) ----
    with torch.no_grad():
        orgchr_lr.P.weight.zero_()

    for m in (feat_embedder, feat_cls_fusion, chrom_embedder, organ_embedder, global_transformer, nhpp_head, cond_film, organ_token, orgchr_lr):
        m.to(device)

    model_components = dict(
        feature_embedder=feat_embedder, feat_cls_fusion=feat_cls_fusion,
        chrom_embedder=chrom_embedder, organ_embedder=organ_embedder,
        global_transformer=global_transformer, nhpp_head=nhpp_head,
        cond_film=cond_film, organ_token=organ_token, orgchr_lr=orgchr_lr
    )

    if n_gpu > 1:
        print(f"[INFO] Using DataParallel on {n_gpu} GPUs", flush=True)
        for k in model_components:
            model_components[k] = nn.DataParallel(model_components[k])

    # ----- Optimizer/Scheduler ----- #
    param_groups = [
        {"params": itertools.chain(
            model_components["feature_embedder"].parameters(),
            model_components["feat_cls_fusion"].parameters(),
            model_components["chrom_embedder"].parameters(),
            model_components["organ_embedder"].parameters(),
            model_components["global_transformer"].parameters(),
            model_components["cond_film"].parameters(),
            model_components["organ_token"].parameters(),
            model_components["orgchr_lr"].parameters(),
        ), "weight_decay": 1e-3, "lr": args.lr},
        {"params": model_components["nhpp_head"].parameters(), "weight_decay": 0.0, "lr": args.lr},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # 스케줄러 안전 폴백
    steps_per_epoch = max(1, len(train_loader))
    T0      = getattr(args, "lr_sched_T0", None) or steps_per_epoch
    Tmult   = getattr(args, "lr_sched_Tmult", 1)
    eta_min = getattr(args, "lr_sched_eta_min", None) or max(1e-6, args.lr * 0.3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tmult, eta_min=eta_min)

    # ----------------------------- Training loop -------------------------- #
    FORCE_DEBUG_FIRST_BATCH = True
    max_grad_norm     = 1.0
    log_interval      = 1000
    step_global       = 0

    organ_sizes = Counter([s["organ_id"] for s in ds_train.segments])
    organ_beta = float(getattr(args, "organ_beta", 0.5))
    organ_weight = {oid: (1.0 / max(1, organ_sizes[oid])) ** organ_beta for oid in organ_sizes}

    use_ema_rw = bool(getattr(args, "ema_reweight", False))
    ema_decay  = float(getattr(args, "ema_decay", 0.9))
    ema_gamma  = float(getattr(args, "ema_gamma", 0.5))
    ema_org_loss = {oid: 1.0 for oid in organ_weight}

    def _update_ema(oid: int, val: float, decay: float = ema_decay):
        ema_org_loss[oid] = decay * ema_org_loss.get(oid, 1.0) + (1.0 - decay) * max(val, 1e-6)

    def weighted_loss_one_batch(batch, return_ctx=False):
        cls_b  = batch["cls_array"].to(device)
        feat_b = batch["feat_array"].to(device)
        y_b    = _safe_count_(batch["y_array"].to(device))
        len_b  = _safe_len_(batch["length_array"].to(device))
        cid_b  = batch["chrom_id"].to(device)
        oid_b  = torch.tensor([seg["organ_id"] for seg in batch["raw_segments"]],
                              device=device, dtype=torch.long)
        UNK_OID = organ_id_map.get("__UNK__")
        organ_dropout_p = 0.15  # 내부 상수(옵션 추가 없이)
        if UNK_OID is not None and organ_dropout_p > 0:
            oid_eff = oid_b.clone()
            drop_mask = (torch.rand_like(oid_eff.float()) < organ_dropout_p)
            oid_eff[drop_mask] = UNK_OID
        else:
            oid_eff = oid_b
        key_pad = (len_b <= 0)

        feat_emb = model_components["feature_embedder"](feat_b)
        fused, alpha_map = model_components["feat_cls_fusion"](
            cls_b, feat_emb,
            chrom_id=cid_b,
            valid_mask=(len_b > 0),
            tissue_ids=oid_eff,
            epoch_idx=epoch
        )

        chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
        org_emb  = model_components["organ_embedder"](oid_eff).unsqueeze(1).expand_as(fused)


        # ---- Domain token prepend + OrgChr Low-Rank + pre-FiLM ----
        tok   = model_components["organ_token"](oid_eff)      # (B,1,D)
        delta = model_components["orgchr_lr"](oid_eff, cid_b) # (B,1,D)
        x = fused + chr_emb + org_emb + delta.expand_as(fused)  # ★ 저랭크 보정 추가
        x = torch.cat([tok, x], dim=1)                    # (B,T+1,D)
        if key_pad is not None:
            pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
            key_pad = torch.cat([pad0, key_pad], dim=1)   # (B,T+1)
        x = model_components["cond_film"](x, oid_eff)

        out = model_components["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_eff)

        # ---- Post-FiLM + drop [ORGAN] position ----
        out = model_components["cond_film"](out, oid_eff)
        out = out[:, 1:, :]                               # align to original (B,T,D)

        out = out.contiguous()
        lam = _safe_rate_(model_components["nhpp_head"](out, tissue_ids=oid_eff))
        lam = _attach_grad_sanitizer(lam)

        if args.label_roll and args.label_roll_width > 1:
            lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
            lam   = _safe_rate_(lam)
            lam   = _attach_grad_sanitizer(lam)
            y_b   = _safe_count_(y_b)
            len_b = _safe_len_(len_b)

        base_w = [1.0 for _ in batch["raw_segments"]]
        obal   = [organ_weight.get(seg["organ_id"], 1.0) for seg in batch["raw_segments"]]

        # per-organ batch nll (로깅 & EMA)
        per_seg_vec = _perseg_nll30k(lam, y_b, len_b)  # (B,)
        oids = torch.tensor([seg["organ_id"] for seg in batch["raw_segments"]], device=per_seg_vec.device)
        per_organ_batch: Dict[int, float] = {}
        for oid in torch.unique(oids).tolist():
            m = per_seg_vec[oids == oid].mean().item()
            per_organ_batch[int(oid)] = float(m)
            if use_ema_rw:
                _update_ema(int(oid), float(m))

        if use_ema_rw and len(ema_org_loss) > 0:
            mean_ema = sum(ema_org_loss.values()) / max(len(ema_org_loss), 1)
            adj = []
            for seg in batch["raw_segments"]:
                oid_i = seg["organ_id"]
                denom = max(ema_org_loss.get(oid_i, mean_ema), 1e-6)
                coef = (mean_ema / denom) ** ema_gamma
                adj.append(coef)
        else:
            adj = [1.0] * len(batch["raw_segments"])

        # ---- NEW: 가중 클리핑 [0.5, 2.0]
        adj = torch.tensor(adj, device=device, dtype=torch.float32).clamp_(0.5, 2.0)

        # 최종 세그 가중 (robust 곱 전)
        w_seg  = torch.tensor([b*w*a for b, w, a in zip(base_w, obal, adj.tolist())], device=device, dtype=torch.float32)

        # ---- NEW: Robust cap (Huber) : 이상치 영향 제한 ----
        w_robust = _robust_seg_weights_from_batch(
            lam, y_b, len_b,
            huber_factor=float(getattr(args, "huber_factor", 3.0)),
            use_mad=bool(getattr(args, "use_mad", False)),
            organ_ids=oid_b,
        )# (B,)

        w_seg = w_seg * w_robust

        loss = trapezoid_nhpp_loss_segment_weighted(lam, y_b, len_b, w_seg)

        # 탈상관 페널티(코사인>0)
        try:
            core = unwrap(model_components["feat_cls_fusion"]).core
            c_ln = core.ln_cls(cls_b)
            f_ln = core.ln_feat(feat_emb)
            cos_cf = F.cosine_similarity(c_ln, f_ln, dim=-1)
            loss = loss + 1e-4 * torch.relu(cos_cf).mean()
        except Exception:
            pass

        loss = loss + unwrap(model_components["feat_cls_fusion"]).core.alpha_regularizer(alpha_map, epoch)

        # ---- NEW: 중요도 샘플러용 동적 난이도 EMA 업데이트 ----
        if "ema_loss" in shared_state:
            with torch.no_grad():
                for i, seg in enumerate(batch["raw_segments"]):
                    gidx = seg["global_idx"]
                    tidx = trainidx_by_global.get(gidx, None)
                    if tidx is None: continue
                    old = float(shared_state["ema_loss"][tidx])
                    new = ema_decay * old + (1.0 - ema_decay) * float(per_seg_vec[i].item())
                    shared_state["ema_loss"][tidx] = max(1e-6, new)

        if return_ctx:
            return loss, lam.detach(), y_b.detach(), len_b.detach(), alpha_map.detach(), per_organ_batch
        return loss

    # ====== Training ======
    train_hist, val_hist = [], []
    for epoch in range(args.epochs):
        for m in model_components.values(): m.train()
        sum_loss, cnt = 0.0, 0

        for _step, batch in enumerate(train_loader):
            step_global += 1
            loss_val, lam_dbg, y_dbg, dt_dbg, alpha_dbg, per_organ_stats = weighted_loss_one_batch(batch, True)
            if DEBUG_NAN and FORCE_DEBUG_FIRST_BATCH and step_global == 1:
                _dump_batch_stats("pre-loss(first-batch)", lam_dbg, y_dbg, dt_dbg)

            if not torch.isfinite(loss_val):
                _dump_batch_stats("NaN-before-raise", lam_dbg, y_dbg, dt_dbg); raise RuntimeError("NaN detected")

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(False):
                loss_val.backward()

            for m in model_components.values():
                for p in unwrap(m).parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            for g in optimizer.param_groups:
                nn.utils.clip_grad_norm_(g["params"], max_grad_norm)

            optimizer.step(); scheduler.step()

            sum_loss += float(loss_val.item()); cnt += 1

            # ---------- STEP CSV 로그 (열=암종, 행=step) ----------
            if step_global % log_interval == 0:
                avg = sum_loss / max(cnt, 1); sum_loss = 0.0; cnt = 0

                # α per-organ mean
                po_alpha = {}
                with torch.no_grad():
                    len_kb = _safe_len_(batch["length_array"].to(alpha_dbg.device))
                    valid = (len_kb > 0).unsqueeze(-1)
                    B = alpha_dbg.shape[0]
                    alpha_sum = defaultdict(float); alpha_cnt = defaultdict(int)
                    for i in range(B):
                        oid_i = int(batch["raw_segments"][i]["organ_id"])
                        a_i = alpha_dbg[i][valid[i]].mean().item() if valid[i].any() else float('nan')
                        if np.isfinite(a_i):
                            alpha_sum[oid_i] += a_i; alpha_cnt[oid_i] += 1
                    for oid_i in alpha_sum:
                        po_alpha[organ_name_map.get(oid_i, oid_i)] = float(alpha_sum[oid_i] / max(alpha_cnt[oid_i], 1))
                        
                                # ---------- ★ NEW: seg-length별 α 평균 (10/50/100kb) ----------
                    with torch.no_grad():
                        # alpha_dbg: (B,T,1), valid: (B,T,1)
                        len_kb_bt = _safe_len_(batch["length_array"].to(alpha_dbg.device))  # (B,T)
                        valid_bt1 = (len_kb_bt > 0).unsqueeze(-1)                            # (B,T,1)

                        # 각 샘플(세그먼트)별 α 평균: pad 제외
                        denom_b1 = valid_bt1.sum(dim=1).clamp_min(1.0)                      # (B,1)
                        alpha_mean_b1 = (alpha_dbg * valid_bt1).sum(dim=1) / denom_b1       # (B,1)
                        alpha_mean_b = alpha_mean_b1.squeeze(-1)                            # (B,)

                        # 길이 버킷(10/50/100)별로 모으기
                        bins = {10: [], 50: [], 100: []}
                        for i, seg in enumerate(batch["raw_segments"]):
                            sz = _approx_size_kb(seg)
                            if sz in bins:
                                am = float(alpha_mean_b[i].item())
                                if np.isfinite(am):
                                    bins[sz].append(am)

                        def _m(arr): return float(np.mean(arr)) if (len(arr) > 0) else float('nan')

                        a10, a50, a100 = _m(bins[10]), _m(bins[50]), _m(bins[100])

                        # CSV 저장
                        _csv_append_row(
                            step_alpha_len_csv,
                            ["step","epoch","alpha_10kb","alpha_50kb","alpha_100kb"],
                            [step_global, epoch, round(a10,6), round(a50,6), round(a100,6)]
                        )

                    # 콘솔에도 짧게 출력
                    print(f"[α by seglen] 10kb={a10:.3f}  50kb={a50:.3f}  100kb={a100:.3f}", flush=True)


                # loss per-organ(mean) already in per_organ_stats (organ_id→loss)
                po_loss = {organ_name_map.get(k, k): float(v) for k, v in per_organ_stats.items()}

                # CSV header & row: ["step","epoch", <organ columns...>]
                alpha_header = ["step","epoch"] + organ_names_ordered
                loss_header  = ["step","epoch"] + organ_names_ordered
                alpha_row = [step_global, epoch] + [round(po_alpha.get(name, float("nan")), 6) for name in organ_names_ordered]
                loss_row  = [step_global, epoch] + [round(po_loss.get(name,  float("nan")), 6) for name in organ_names_ordered]

                _csv_append_row(step_alpha_csv, alpha_header, alpha_row)
                _csv_append_row(step_loss_csv,  loss_header,  loss_row)

                # 콘솔 표시는 유지
                vals = list(po_loss.values()); po_mean = float(np.mean(vals)) if vals else float('nan'); po_std = float(np.std(vals)) if vals else float('nan')
                po_table   = _format_per_organ_table(po_loss, cols=3, order="desc", name_max=16)
                po_alpha_t = _format_per_organ_table(po_alpha, cols=3, order="desc", name_max=16, val_fmt="{:>7.3f}")
                print(
                    f"[Epoch {epoch} | Step {step_global}] loss={avg:.4f}  "
                    f"(per-organ loss mean±sd={po_mean:.3f}±{po_std:.3f})\n"
                    f"{po_table}\n[α per-organ (mean)]\n{po_alpha_t}",
                    flush=True,
                )

        # -------------------- epoch 평가 ------------------------------ #
        def eval_loader(loader):
            for m in model_components.values(): m.eval()
            res: Dict[int, float] = {}
            res_org: Dict[int, int] = {}  # NEW: seg_id -> organ_id

            with torch.no_grad():
                for b in loader:
                    cls_b  = b["cls_array"].to(device)
                    feat_b = b["feat_array"].to(device)
                    y_b    = _safe_count_(b["y_array"].to(device))
                    len_b  = _safe_len_(b["length_array"].to(device))
                    cid_b  = b["chrom_id"].to(device)
                    oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]], device=device, dtype=torch.long)
                    key_pad = (len_b <= 0)

                    feat_emb = model_components["feature_embedder"](feat_b)
                    fused, _ = model_components["feat_cls_fusion"](cls_b, feat_emb, chrom_id=cid_b, valid_mask=(len_b > 0), tissue_ids=oid_b, epoch_idx=None)
                    chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                    org_emb  = model_components["organ_embedder"](oid_b).unsqueeze(1).expand_as(fused)

                    tok   = model_components["organ_token"](oid_b)
                    delta = model_components["orgchr_lr"](oid_b, cid_b)
                    x = fused + chr_emb + org_emb + delta.expand_as(fused)
                    x = torch.cat([tok, x], dim=1)
                    if key_pad is not None:
                        pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
                        key_pad = torch.cat([pad0, key_pad], dim=1)
                    x = model_components["cond_film"](x, oid_b)

                    out = model_components["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_b)
                    out = model_components["cond_film"](out, oid_b)[:, 1:, :]

                    lam = _safe_rate_(model_components["nhpp_head"](out.contiguous(), tissue_ids=oid_b))

                    if args.label_roll and args.label_roll_width > 1:
                        lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                        lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                    mu = lam * len_b
                    y_np, mu_np = y_b.cpu().numpy(), mu.cpu().numpy()
                    for i, seg in enumerate(b["raw_segments"]):
                        L   = seg["cls_array"].shape[0]
                        sid = int(seg["global_idx"])
                        res[sid]     = float((y_np[i, :L] - mu_np[i, :L]).mean())
                        res_org[sid] = int(seg["organ_id"])

            # --- NEW: organ-wise δ & w_dict ---
            from collections import defaultdict as _dd
            by_org_vals = _dd(list)
            for sid, r in res.items():
                by_org_vals[res_org[sid]].append(float(r))

            delta_by_org: Dict[int, float] = {}
            for oid, vals in by_org_vals.items():
                arr = np.asarray(vals, dtype=np.float64)
                if arr.size == 0:
                    delta_by_org[oid] = float(args.huber_factor)
                    continue
                if args.use_mad:
                    med = np.median(arr); mad = np.median(np.abs(arr - med))
                    scale = 1.4826 * mad
                else:
                    q1, q3 = np.percentile(arr, [25, 75])
                    scale = (q3 - q1)
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                delta_by_org[oid] = float(args.huber_factor) * float(scale)

            def _hw_org_eval(r, oid):
                d = float(delta_by_org.get(int(oid), float(args.huber_factor)))
                a = abs(float(r));  return 1.0 if a <= d else d/(a + _EPS)

            w_dict = {sid: _hw_org_eval(res[sid], res_org[sid]) for sid in res}

            # ---- 가중 합산 (기존 로직 유지) ----
            tot_num = tot_den = 0.0
            with torch.no_grad():
                for b in loader:
                    cls_b  = b["cls_array"].to(device)
                    feat_b = b["feat_array"].to(device)
                    y_b    = _safe_count_(b["y_array"].to(device))
                    len_b  = _safe_len_(b["length_array"].to(device))
                    cid_b  = b["chrom_id"].to(device)
                    oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]], device=device, dtype=torch.long)
                    key_pad = (len_b <= 0)

                    feat_emb = model_components["feature_embedder"](feat_b)
                    fused, _ = model_components["feat_cls_fusion"](cls_b, feat_emb, chrom_id=cid_b, valid_mask=(len_b > 0), tissue_ids=oid_b, epoch_idx=None)
                    chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                    org_emb  = model_components["organ_embedder"](oid_b).unsqueeze(1).expand_as(fused)

                    tok   = model_components["organ_token"](oid_b)
                    delta = model_components["orgchr_lr"](oid_b, cid_b)
                    x = fused + chr_emb + org_emb + delta.expand_as(fused)
                    x = torch.cat([tok, x], dim=1)
                    if key_pad is not None:
                        pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
                        key_pad = torch.cat([pad0, key_pad], dim=1)
                    x = model_components["cond_film"](x, oid_b)

                    out = model_components["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_b)
                    out = model_components["cond_film"](out, oid_b)[:, 1:, :]

                    lam = _safe_rate_(model_components["nhpp_head"](out.contiguous(), tissue_ids=oid_b))
                    if args.label_roll and args.label_roll_width > 1:
                        lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                        lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                    # per-seg loss
                    sum_log = (y_b * torch.log(lam)).sum(dim=1)
                    integ   = (lam * len_b).sum(dim=1)
                    neg_ll  = -(sum_log - integ)
                    seg_len = (len_b.sum(dim=1) + _EPS)
                    per_seg = (neg_ll / seg_len) * 30.0

                    ids = [seg["global_idx"] for seg in b["raw_segments"]]
                    w   = torch.tensor([w_dict.get(i, 1.0) for i in ids], device=per_seg.device, dtype=per_seg.dtype)
                    tot_num += float((w * per_seg).sum().item());  tot_den += float(w.sum().item())

            return tot_num / max(tot_den, _EPS)

       

        # calibrate (optional)
        calibrate_log_c_huber_like_training(
            model_components, train_loader_infer, device,
            huber_factor=args.huber_factor, use_mad=args.use_mad,
            label_roll=args.label_roll, roll_width=args.label_roll_width
        )

        train_loss = eval_loader(train_loader_infer)
        val_loss   = eval_loader(val_loader)

        # epoch-level CSV (참고)
        _csv_append_row(epoch_csv, ["epoch","train_loss","val_loss"], [epoch, round(train_loss,6), round(val_loss,6)])

        print(f"[Epoch {epoch}] Train={train_loss:.4f}  Val={val_loss:.4f}", flush=True)

        # 베스트 저장(간단)
        ckpt = {
            "feature_embedder": model_components["feature_embedder"].state_dict(),
            "feat_cls_fusion":  model_components["feat_cls_fusion"].state_dict(),
            "chrom_embedder":   model_components["chrom_embedder"].state_dict(),
            "organ_embedder":   model_components["organ_embedder"].state_dict(),
            "global_transformer": model_components["global_transformer"].state_dict(),
            "nhpp_head":        model_components["nhpp_head"].state_dict(),
            "cond_film":        model_components["cond_film"].state_dict(),
            "organ_token":      model_components["organ_token"].state_dict(),
            "orgchr_lr":        model_components["orgchr_lr"].state_dict(),
            "epoch": epoch,
            "best_val_loss": val_loss,
            "organ_map": organ_id_map,
        }
        efficient_save_ckpt(out_dir / "trained_model.pt", **ckpt)

    # ---------------- Final eval & prediction (기존과 동일) ---------------- #
    print("[INFO] Loading best model for final evaluation ..", flush=True)
    best_ckpt = efficient_load_ckpt(out_dir / "trained_model.pt")
    for k in ("feature_embedder","feat_cls_fusion","chrom_embedder","organ_embedder","global_transformer","nhpp_head","cond_film","organ_token","orgchr_lr"):
        model_components[k].load_state_dict(best_ckpt[k], strict=False)
    del best_ckpt; gc.collect(); torch.cuda.empty_cache()

    calibrate_log_c_huber_like_training(
        model_components, all_loader, device,
        huber_factor=args.huber_factor, use_mad=args.use_mad,
        label_roll=args.label_roll, roll_width=args.label_roll_width
    )
    
    @torch.no_grad()
    def compute_alpha_by_len_csv(loader, split_name: str):
        """
        loader 전체를 돌며 seg-length(10/50/100kb)별 α 평균을 계산해 CSV에 한 줄 저장.
        """
        for m in model_components.values(): m.eval()

        bins = {10: [], 50: [], 100: []}

        for b in loader:
            cls_b  = b["cls_array"].to(device)
            feat_b = b["feat_array"].to(device)
            cid_b  = b["chrom_id"].to(device)
            oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]],
                                  device=device, dtype=torch.long)
            len_b  = _safe_len_(b["length_array"].to(device))

            # fusion 호출에서 α map 얻기(훈련과 동일 경로)
            fused, alpha_map = model_components["feat_cls_fusion"](
                cls_b, model_components["feature_embedder"](feat_b),
                chrom_id=cid_b, valid_mask=(len_b > 0),
                tissue_ids=oid_b, epoch_idx=None
            )  # alpha_map: (B,T,1)

            # 세그 평균 α (pad 제외)
            valid_bt1 = (len_b > 0).unsqueeze(-1)  # (B,T,1)
            denom_b1 = valid_bt1.sum(dim=1).clamp_min(1.0)
            alpha_mean_b1 = (alpha_map * valid_bt1).sum(dim=1) / denom_b1
            alpha_mean_b = alpha_mean_b1.squeeze(-1)  # (B,)

            # 길이 버킷으로 축적
            for i, seg in enumerate(b["raw_segments"]):
                sz = _approx_size_kb(seg)
                if sz in bins:
                    am = float(alpha_mean_b[i].item())
                    if np.isfinite(am):
                        bins[sz].append(am)

        def _m(arr): return float(np.mean(arr)) if (len(arr) > 0) else float('nan')
        a10, a50, a100 = _m(bins[10]), _m(bins[50]), _m(bins[100])

        # CSV에 append (헤더 자동 작성)
        _csv_append_row(
            final_alpha_len_csv,
            ["split","alpha_10kb","alpha_50kb","alpha_100kb"],
            [split_name, round(a10,6), round(a50,6), round(a100,6)]
        )
        print(f"[FINAL α by seglen | {split_name}] 10kb={a10:.3f}  50kb={a50:.3f}  100kb={a100:.3f}", flush=True)

    
    def evaluate(loader):
        for m in model_components.values(): m.eval()
        res: Dict[int, float] = {}
        res_org: Dict[int, int] = {}

        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = _safe_count_(b["y_array"].to(device))
                len_b  = _safe_len_(b["length_array"].to(device))
                cid_b  = b["chrom_id"].to(device)
                oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]], device=device, dtype=torch.long)
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused, _ = model_components["feat_cls_fusion"](cls_b, feat_emb, chrom_id=cid_b, valid_mask=(len_b > 0), tissue_ids=oid_b, epoch_idx=None)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                org_emb  = model_components["organ_embedder"](oid_b).unsqueeze(1).expand_as(fused)

                tok   = model_components["organ_token"](oid_b)
                delta = model_components["orgchr_lr"](oid_b, cid_b)
                x = fused + chr_emb + org_emb + delta.expand_as(fused)
                x = torch.cat([tok, x], dim=1)
                if key_pad is not None:
                    pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
                    key_pad = torch.cat([pad0, key_pad], dim=1)
                x = model_components["cond_film"](x, oid_b)

                out = model_components["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_b)
                out = model_components["cond_film"](out, oid_b)[:, 1:, :]

                lam = _safe_rate_(model_components["nhpp_head"](out.contiguous(), tissue_ids=oid_b))

                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                    lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                mu_b = lam * len_b
                for i, seg in enumerate(b["raw_segments"]):
                    L   = seg["cls_array"].shape[0]
                    sid = int(seg["global_idx"])
                    res[sid]     = float((y_b[i, :L] - mu_b[i, :L]).mean().item())
                    res_org[sid] = int(seg["organ_id"])

        # organ-wise δ & w_dict
        from collections import defaultdict as _dd
        by_org_vals = _dd(list)
        for sid, r in res.items():
            by_org_vals[res_org[sid]].append(float(r))

        delta_by_org: Dict[int, float] = {}
        for oid, vals in by_org_vals.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                delta_by_org[oid] = float(args.huber_factor)
                continue
            if args.use_mad:
                med = np.median(arr); mad = np.median(np.abs(arr - med))
                scale = 1.4826 * mad
            else:
                q1, q3 = np.percentile(arr, [25, 75])
                scale = (q3 - q1)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            delta_by_org[oid] = float(args.huber_factor) * float(scale)

        def _hw_org_eval(r, oid):
            d = float(delta_by_org.get(int(oid), float(args.huber_factor)))
            a = abs(float(r));  return 1.0 if a <= d else d/(a + _EPS)

        w_dict = {sid: _hw_org_eval(res[sid], res_org[sid]) for sid in res}

        # 가중 합산
        tot_num, tot_den = 0.0, 0.0
        with torch.no_grad():
            for b in loader:
                cls_b  = b["cls_array"].to(device)
                feat_b = b["feat_array"].to(device)
                y_b    = _safe_count_(b["y_array"].to(device))
                len_b  = _safe_len_(b["length_array"].to(device))
                cid_b  = b["chrom_id"].to(device)
                oid_b  = torch.tensor([seg["organ_id"] for seg in b["raw_segments"]], device=device, dtype=torch.long)
                key_pad  = (len_b <= 0)

                feat_emb = model_components["feature_embedder"](feat_b)
                fused, _ = model_components["feat_cls_fusion"](cls_b, feat_emb, chrom_id=cid_b, valid_mask=(len_b > 0), tissue_ids=oid_b, epoch_idx=None)
                chr_emb  = model_components["chrom_embedder"](cid_b).unsqueeze(1).expand_as(fused)
                org_emb  = model_components["organ_embedder"](oid_b).unsqueeze(1).expand_as(fused)

                tok   = model_components["organ_token"](oid_b)
                delta = model_components["orgchr_lr"](oid_b, cid_b)
                x = fused + chr_emb + org_emb + delta.expand_as(fused)
                x = torch.cat([tok, x], dim=1)
                if key_pad is not None:
                    pad0 = torch.zeros((key_pad.size(0), 1), dtype=key_pad.dtype, device=key_pad.device)
                    key_pad = torch.cat([pad0, key_pad], dim=1)
                x = model_components["cond_film"](x, oid_b)

                out = model_components["global_transformer"](x, key_padding_mask=key_pad, organ_ids=oid_b)
                out = model_components["cond_film"](out, oid_b)[:, 1:, :]

                lam = _safe_rate_(model_components["nhpp_head"](out.contiguous(), tissue_ids=oid_b))
                if args.label_roll and args.label_roll_width > 1:
                    lam, y_b, len_b = R.rolling_sum_nhpp(lam, y_b, len_b, width=args.label_roll_width)
                    lam = _safe_rate_(lam); y_b = _safe_count_(y_b); len_b = _safe_len_(len_b)

                sum_log = (y_b * torch.log(lam)).sum(dim=1)
                integ   = (lam * len_b).sum(dim=1)
                neg_ll  = -(sum_log - integ)
                seg_len = (len_b.sum(dim=1) + _EPS)
                per_seg = (neg_ll / seg_len) * 30.0

                ids = [seg["global_idx"] for seg in b["raw_segments"]]
                w   = torch.tensor([w_dict.get(i, 1.0) for i in ids], device=per_seg.device, dtype=per_seg.dtype)
                tot_num += float((w * per_seg).sum().item())
                tot_den += float(w.sum().item())
  
        return tot_num / max(tot_den, 1e-9)


    print(f"final Train_loss = {evaluate(train_loader_infer):.4f}", flush=True)
    print(f"final Val_loss   = {evaluate(val_loader):.4f}", flush=True)
    compute_alpha_by_len_csv(train_loader_infer, "train")
    compute_alpha_by_len_csv(val_loader,   "val")
    compute_alpha_by_len_csv(all_loader,   "all")

    # (predict_and_save 등 이하 동일하면 생략 가능)
    # ... 생략 ...

# ------------------------------ CLI --------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description="Pan-cancer pretraining (multi-organ)")
    p.add_argument("--manifest", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--segment-lengths", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--discard-leftover", action="store_true")
    p.add_argument("--overlap-factor", type=float, default=0.3)
    p.add_argument("--num-data-workers", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--use-mad", action="store_true", default=False)
    p.add_argument("--huber-factor", type=float, default=3.0)
    p.add_argument("--cutmix-p", type=float, default=0.2)
    p.add_argument("--label-roll", action="store_true")
    p.add_argument("--label-roll-width", type=int, default=2)
    p.add_argument("--organ-beta", type=float, default=0.5)
    p.add_argument("--lr-sched-T0", type=int, default=None)
    p.add_argument("--lr-sched-Tmult", type=int, default=1)
    p.add_argument("--lr-sched-eta-min", type=float, default=None)
    p.add_argument("--common-cls", type=str, default=None)
    p.add_argument("--segs-lite", action="store_true")
    p.add_argument("--cls-fp16", action="store_true")
    p.add_argument("--save-each-best", action="store_true")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta-pct", type=float, default=0.5)
    p.add_argument("--resume-checkpoint", type=str, default=None)
    p.add_argument("--save-attention", action="store_true")
    p.add_argument("--alpha-min", dest="alpha_min", type=float, default=0.20)
    p.add_argument("--balance-per-batch", action="store_true")
    p.add_argument("--per-organ-samples", type=int, default=None)
    p.add_argument("--ema-reweight", action="store_true")
    p.add_argument("--ema-gamma", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.9)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    pretrain_and_predict(args)
