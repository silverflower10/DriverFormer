#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segments.py — fixed-length segment builder & collate (classic & lite with tissue_id)

- Dataset 하나가 하나의 조직일 때는 tissue_id_default만 넘겨주면 됨(기본 0).
- 필요 시 (chrom, w_idx) → tissue_id 콜백(tissue_resolver)로 세밀 지정 가능.
- Classic: 세그에 CLS/FEAT를 복사해서 저장 (메모리↑, 빠름)
- Lite   : 세그엔 widx/좌표/FEAT/Y만 저장, CLS는 collate 시 공용 CLS_BANK에서 가져옴 (메모리↓)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional, Callable, List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "SegmentDataset",
    "segment_cls_embeddings_fixed_lengths",
    "segment_collate_fn",
    "segment_indices_fixed_lengths",
    "segment_collate_fn_lite",
]

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class SegmentDataset(Dataset):
    def __init__(self, segments: List[Dict[str, Any]]):
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.segments[idx]


# --------------------------------------------------------------------------- #
# Classic builders                                                            #
# --------------------------------------------------------------------------- #
def _make_segment_dict(
    chrom: str,
    slice_: List[Tuple[int, int, int, np.ndarray, float, np.ndarray]],
    tissue_id: int = 0,
) -> Dict[str, Any]:
    """
    slice_: list of tuples (w_idx, sbp, ebp, cls_vec, y_val, feat_vec)
    """
    idx_array   = np.array([s[0] for s in slice_], dtype=np.int32)
    start_array = np.array([s[1] for s in slice_], dtype=np.int64)
    end_array   = np.array([s[2] for s in slice_], dtype=np.int64)
    cls_array   = np.stack([np.asarray(s[3], dtype=np.float32) for s in slice_], axis=0)
    y_array     = np.array([s[4] for s in slice_], dtype=np.float32)
    feat_array  = np.stack([np.asarray(s[5], dtype=np.float32) for s in slice_], axis=0)

    return dict(
        chrom=str(chrom),
        idx_array=idx_array,
        start_array=start_array,
        end_array=end_array,
        cls_array=cls_array,
        feat_array=feat_array,
        y_array=y_array,
        tissue_id=int(tissue_id),
    )


def segment_cls_embeddings_fixed_lengths(
    cls_list: List[Tuple[str, int, int, int, np.ndarray, float]],
    feature_dict: Dict[Tuple[str, int], np.ndarray],
    seg_len_list: Tuple[int, ...] = (10, 50, 100),
    discard_leftover: bool = False,
    overlap_factor: float = 0.0,
    *,
    tissue_id_default: int = 0,
    tissue_resolver: Optional[Callable[[str, int], Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Classic 세그 생성: CLS/FEAT를 세그에 복사해 저장 (메모리↑, collate가 가벼움).
    """
    chrom_map: Dict[str, List[Tuple[int, int, int, np.ndarray, float, np.ndarray, Optional[int]]]] = defaultdict(list)

    # feature_dim 파악
    try:
        any_feat = next(iter(feature_dict.values()))
        feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else int(any_feat.shape[0])
    except StopIteration:
        feature_dim = 0

    for chrom, w_idx, sbp, ebp, cls_vec, y_val in cls_list:
        feat_vec = feature_dict.get((chrom, w_idx))
        if feat_vec is None:
            feat_vec = np.zeros((feature_dim,), dtype=np.float32) if feature_dim > 0 else np.zeros((0,), dtype=np.float32)
        else:
            feat_vec = np.asarray(feat_vec, dtype=np.float32)
            if feat_vec.ndim == 2:
                if feat_vec.shape[0] == 1:
                    feat_vec = feat_vec[0]
                elif feat_vec.shape[1] == 1:
                    feat_vec = feat_vec[:, 0]
                else:
                    feat_vec = feat_vec.mean(axis=0).astype(np.float32, copy=False)
        ti_elem = tissue_resolver(chrom, w_idx) if tissue_resolver else None
        chrom_map[str(chrom)].append(
            (int(w_idx), int(sbp), int(ebp), np.asarray(cls_vec, dtype=np.float32), float(y_val), feat_vec, ti_elem)
        )

    all_segments: List[Dict[str, Any]] = []
    step_cache: Dict[int, int] = {}
    for seg_len in seg_len_list:
        step = step_cache.setdefault(seg_len, max(1, int(seg_len * (1.0 - overlap_factor))))
        for chrom in chrom_map:
            items = sorted(chrom_map[chrom], key=lambda x: x[0])
            n, i = len(items), 0
            while i < n:
                end = i + seg_len
                if end > n:
                    if discard_leftover:
                        break
                    end = n
                ti_seg = items[i][6] if items[i][6] is not None else tissue_id_default
                core = [t[:6] for t in items[i:end]]
                all_segments.append(_make_segment_dict(chrom, core, tissue_id=ti_seg))
                i += step

    for gid, seg in enumerate(all_segments):
        seg["global_idx"] = gid
    return all_segments


# --------------------------------------------------------------------------- #
# Classic collate                                                             #
# --------------------------------------------------------------------------- #
def segment_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    chrom_id_map: Optional[Dict[str, int]] = None,
    cutmix_p: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Classic collate:
      - cutmix (seg-level swap of spans) 옵션 지원
      - variable length padding
      - length_array: (B,T) in kb, PAD는 0
      - 반환 텐서:
          cls_array(B,T,D), feat_array(B,T,F), y_array(B,T),
          start_array(B,T), end_array(B,T), length_array(B,T),
          chrom_id(B,), tissue_id(B,)
    """
    if cutmix_p > 0 and random.random() < cutmix_p and len(batch) >= 2:
        i, j = random.sample(range(len(batch)), 2)
        seg_i, seg_j = batch[i], batch[j]
        L = min(seg_i["cls_array"].shape[0], seg_j["cls_array"].shape[0])
        cut = random.randint(1, max(1, int(0.5 * L)))
        s = random.randint(0, L - cut)
        for key in ("cls_array", "feat_array", "y_array", "start_array", "end_array"):
            tmp = seg_i[key][s:s+cut].copy()
            seg_i[key][s:s+cut] = seg_j[key][s:s+cut]
            seg_j[key][s:s+cut] = tmp
        # tissue_id는 세그먼트 메타라 그대로 둠

    lens = [b["cls_array"].shape[0] for b in batch]
    max_len = max(lens)

    def _pad(arr: np.ndarray, val=0):
        pad = max_len - arr.shape[0]
        if pad > 0:
            if arr.ndim == 2:
                arr = np.pad(arr, ((0, pad), (0, 0)), constant_values=val)
            else:
                arr = np.pad(arr, (0, pad), constant_values=val)
        return arr

    cls_list, feat_list, y_list, s_list, e_list = [], [], [], [], []
    cid_list, tid_list = [], []
    for seg in batch:
        cls_list.append(_pad(seg["cls_array"]))
        feat_list.append(_pad(seg["feat_array"]))
        y_list.append(_pad(seg["y_array"]))
        s_list.append(_pad(seg["start_array"]))
        e_list.append(_pad(seg["end_array"]))
        cid_list.append(chrom_id_map.get(seg["chrom"], 0) if chrom_id_map else 0)
        tid_list.append(int(seg.get("tissue_id", 0)))

    cls_b  = torch.tensor(np.stack(cls_list), dtype=torch.float32)
    feat_b = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    y_b    = torch.tensor(np.stack(y_list),  dtype=torch.float32)
    s_b    = torch.tensor(np.stack(s_list),  dtype=torch.long)
    e_b    = torch.tensor(np.stack(e_list),  dtype=torch.long)

    len_b  = (e_b - s_b + 1).clamp_min(0).float() / 1000.0
    B, T = len_b.shape
    valid = torch.arange(T).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
    len_b = len_b * valid.to(len_b.dtype)

    cid_b = torch.tensor(cid_list, dtype=torch.long)
    tid_b = torch.tensor(tid_list, dtype=torch.long)

    return dict(
        cls_array=cls_b,
        feat_array=feat_b,
        y_array=y_b,
        start_array=s_b,
        end_array=e_b,
        length_array=len_b,
        chrom_id=cid_b,
        tissue_id=tid_b,
        raw_segments=batch,
    )


# --------------------------------------------------------------------------- #
# Lite builders (indices only)                                                #
# --------------------------------------------------------------------------- #
def segment_indices_fixed_lengths(
    cls_list: List[Tuple[str, int, int, int, np.ndarray, float]],
    feature_dict: Dict[Tuple[str, int], np.ndarray],
    seg_len_list: Tuple[int, ...] = (10, 50, 100),
    discard_leftover: bool = False,
    overlap_factor: float = 0.0,
    *,
    tissue_id_default: int = 0,
    tissue_resolver: Optional[Callable[[str, int], Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    경량(Lite) 세그 생성: CLS는 저장하지 않고 인덱스/좌표/FEAT/Y만 보관.

    반환: dict 리스트
      {
        'chrom': str,
        'widx_array': np.ndarray(int32),
        'start_array': np.ndarray(int64),
        'end_array': np.ndarray(int64),
        'feat_array': np.ndarray(L, F),
        'y_array': np.ndarray(L, float32),
        'tissue_id': int,
        'global_idx': int (나중에 부여)
      }
    """
    chrom_map: Dict[str, List[Tuple[int, int, int, float, np.ndarray, Optional[int]]]] = defaultdict(list)

    # feature_dim 파악
    try:
        any_feat = next(iter(feature_dict.values()))
        feature_dim = any_feat.shape[-1] if any_feat.ndim > 1 else int(any_feat.shape[0])
    except StopIteration:
        feature_dim = 0

    # cls_list 형식: (chrom, w_idx, sbp, ebp, cls_vec, y_val)
    for chrom, w_idx, sbp, ebp, _cls, y_val in cls_list:
        feat_vec = feature_dict.get((chrom, w_idx),
                                    np.zeros(feature_dim, dtype=np.float32))
        feat_vec = np.asarray(feat_vec, dtype=np.float32)
        if feat_vec.ndim == 2:
            if feat_vec.shape[0] == 1:
                feat_vec = feat_vec[0]
            elif feat_vec.shape[1] == 1:
                feat_vec = feat_vec[:, 0]
            else:
                feat_vec = feat_vec.mean(axis=0).astype(np.float32, copy=False)
        ti_elem = tissue_resolver(chrom, w_idx) if tissue_resolver else None
        chrom_map[str(chrom)].append(
            (int(w_idx), int(sbp), int(ebp), float(y_val), feat_vec, ti_elem)
        )

    all_segments: List[Dict[str, Any]] = []
    step_cache: Dict[int, int] = {}
    for seg_len in seg_len_list:
        step = step_cache.setdefault(seg_len, max(1, int(seg_len * (1.0 - overlap_factor))))
        for chrom in chrom_map:
            items = sorted(chrom_map[chrom], key=lambda x: x[0])
            n, i = len(items), 0
            while i < n:
                end = i + seg_len
                if end > n:
                    if discard_leftover:
                        break
                    end = n
                sl = items[i:end]
                widx = np.array([t[0] for t in sl], dtype=np.int32)
                sarr = np.array([t[1] for t in sl], dtype=np.int64)
                earr = np.array([t[2] for t in sl], dtype=np.int64)
                yarr = np.array([t[3] for t in sl], dtype=np.float32)
                farr = np.stack([t[4] for t in sl], axis=0).astype(np.float32, copy=False)

                # 요소 단위 tissue_id 정보가 있으면 첫 요소 사용, 없으면 default
                ti_seg = next((t[5] for t in sl if t[5] is not None), None)
                if ti_seg is None:
                    ti_seg = tissue_id_default

                all_segments.append(dict(
                    chrom=str(chrom),
                    widx_array=widx,
                    start_array=sarr,
                    end_array=earr,
                    feat_array=farr,
                    y_array=yarr,
                    tissue_id=int(ti_seg),
                ))
                i += step

    for gid, seg in enumerate(all_segments):
        seg["global_idx"] = gid
    return all_segments


# --------------------------------------------------------------------------- #
# Lite collate (assemble CLS from CLS_BANK)                                   #
# --------------------------------------------------------------------------- #
def segment_collate_fn_lite(
    batch: List[Dict[str, Any]],
    *,
    chrom_id_map: Optional[Dict[str, int]] = None,
    cutmix_p: float = 0.0,
    cls_bank: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Lite collate:
      - 세그에 저장된 widx_array를 이용해 CLS_BANK에서 CLS를 조립
      - cutmix 옵션 지원
      - 반환 텐서는 classic과 동일 형태 + tissue_id 포함
    """
    # (선택) cutmix — widx/start/end/feat/y 동일 구간 스왑
    if cutmix_p > 0 and random.random() < cutmix_p and len(batch) >= 2:
        i, j = random.sample(range(len(batch)), 2)
        seg_i, seg_j = batch[i], batch[j]
        L = min(seg_i["widx_array"].shape[0], seg_j["widx_array"].shape[0])
        cut = random.randint(1, max(1, int(0.5 * L)))
        s = random.randint(0, L - cut)
        for key in ("widx_array", "feat_array", "y_array", "start_array", "end_array"):
            tmp = seg_i[key][s:s+cut].copy()
            seg_i[key][s:s+cut] = seg_j[key][s:s+cut]
            seg_j[key][s:s+cut] = tmp

    lens = [b["widx_array"].shape[0] for b in batch]
    max_len = max(lens)

    def _pad(arr: np.ndarray, val=0, is2d: bool=False):
        pad = max_len - (arr.shape[0] if not is2d else arr.shape[0])
        if pad > 0:
            if is2d:
                arr = np.pad(arr, ((0, pad), (0, 0)), constant_values=val)
            else:
                arr = np.pad(arr, (0, pad), constant_values=val)
        return arr

    # CLS 뱅크 확보(인자 우선, 없으면 pretrain_loop의 글로벌을 폴백)
    if cls_bank is None:
        try:
            from driverformer.train.pretrain_loop import CLS_BANK as _BANK  # type: ignore
            cls_bank = _BANK
        except Exception as e:
            raise RuntimeError("CLS_BANK not provided and cannot import from pretrain_loop") from e

    cls_list, feat_list, y_list, s_list, e_list, cid_list, tid_list = [], [], [], [], [], [], []
    for seg in batch:
        w = seg["widx_array"]
        chrom = seg["chrom"]

        try:
            cls_stack = np.stack([cls_bank[(chrom, int(wi))] for wi in w], axis=0)  # (L, D)
        except KeyError as ke:
            raise KeyError(f"CLS_BANK missing key: {(chrom, int(wi))}") from ke

        if cls_stack.dtype == np.float16:
            cls_stack = cls_stack.astype(np.float32)  # 모델 입력은 float32

        cls_list.append(_pad(cls_stack, is2d=True))
        feat_list.append(_pad(seg["feat_array"], is2d=True))
        y_list.append(_pad(seg["y_array"]))
        s_list.append(_pad(seg["start_array"]))
        e_list.append(_pad(seg["end_array"]))

        cid_list.append(chrom_id_map.get(chrom, 0) if chrom_id_map else 0)
        tid_list.append(int(seg.get("tissue_id", 0)))

    cls_b  = torch.tensor(np.stack(cls_list), dtype=torch.float32)
    feat_b = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    y_b    = torch.tensor(np.stack(y_list),  dtype=torch.float32)
    s_b    = torch.tensor(np.stack(s_list),  dtype=torch.long)
    e_b    = torch.tensor(np.stack(e_list),  dtype=torch.long)

    len_b  = (e_b - s_b + 1).clamp_min(0).float() / 1000.0
    B, T = len_b.shape
    valid = torch.arange(T).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
    len_b = len_b * valid.to(len_b.dtype)

    cid_b  = torch.tensor(cid_list, dtype=torch.long)
    tid_b  = torch.tensor(tid_list, dtype=torch.long)

    return dict(
        cls_array=cls_b,
        feat_array=feat_b,
        y_array=y_b,
        start_array=s_b,
        end_array=e_b,
        length_array=len_b,
        chrom_id=cid_b,
        tissue_id=tid_b,
        raw_segments=batch,
    )
