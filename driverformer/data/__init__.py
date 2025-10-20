#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
driverformer.data ? dataset builders, labeling helpers, and NHPP-consistent rolling

Exports (stable API)
--------------------
Segments
  - SegmentDataset
  - segment_cls_embeddings_fixed_lengths
  - segment_collate_fn

Rolling
  - rolling_sum_nhpp

Labels (public aliases; leading-underscore originals are hidden)
  - bins_from_cls_list
  - infer_bin_size_and_anchor_from_bins
  - guess_sample_col
  - load_mutations_events
  - build_y_map_from_mutations
  - attach_labels_from_y_map
"""

# ── Segments ────────────────────────────────────────────────────────────────
from .segments import (
    SegmentDataset,
    segment_cls_embeddings_fixed_lengths,
    segment_collate_fn,
)

# ── Rolling (NHPP-consistent) ───────────────────────────────────────────────
from .rolling import rolling_sum_nhpp

# ── Labels (re-export as public aliases) ────────────────────────────────────
from .labels import (
    _bins_from_cls_list as bins_from_cls_list,
    _infer_bin_size_and_anchor_from_bins as infer_bin_size_and_anchor_from_bins,
    _guess_sample_col as guess_sample_col,
    _load_mutations_events as load_mutations_events,
    _build_y_map_from_mutations as build_y_map_from_mutations,
    _attach_labels_from_y_map as attach_labels_from_y_map,
)

__all__ = [
    # segments
    "SegmentDataset",
    "segment_cls_embeddings_fixed_lengths",
    "segment_collate_fn",
    # rolling
    "rolling_sum_nhpp",
    # labels (public aliases)
    "bins_from_cls_list",
    "infer_bin_size_and_anchor_from_bins",
    "guess_sample_col",
    "load_mutations_events",
    "build_y_map_from_mutations",
    "attach_labels_from_y_map",
]
