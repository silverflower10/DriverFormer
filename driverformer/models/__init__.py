# -*- coding: utf-8 -*-
# models/__init__.py

# ── Embedding & Fusion
from .embedders import FeatureEmbedder, ChromosomeEmbedder
from .embedders import FeatClsFusion           # (스칼라 α)
from .embedders import HierAlphaFusion as FeatClsFusionCtx  # 하위호환 별칭

# ── Transformer
from .transformer import (
    RoPE, SwiGLUFFN, AttnEncoderLayer,
    FoundationEncoder, GlobalTransformerEncoder
)

# ── NHPP Heads
from .nhpp_head import (
    NHPPHead,                 # shared alias
    ConditionalNHPPHead,      # 실구현
    CondCfg, build_nhpp_head
)

__all__ = [
    # embedders/fusion
    "FeatureEmbedder", "ChromosomeEmbedder",
    "FeatClsFusion", "FeatClsFusionCtx",
    # transformer
    "RoPE", "SwiGLUFFN", "AttnEncoderLayer",
    "FoundationEncoder", "GlobalTransformerEncoder",
    # heads
    "NHPPHead", "ConditionalNHPPHead", "CondCfg", "build_nhpp_head",
]
