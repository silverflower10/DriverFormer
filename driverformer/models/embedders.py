import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from collections import defaultdict

CHROM_LIST_24 = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# ────────────────────────────────────────────────────────────────────────────
# Simple feature → hidden projector
# ────────────────────────────────────────────────────────────────────────────
class FeatureEmbedder(nn.Module):
    """입력 feature_dim -> hidden_dim 선형 투영"""
    def __init__(self, feature_dim, hidden_dim=768):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim)
    def forward(self, x):
        return self.proj(x)

# ────────────────────────────────────────────────────────────────────────────
# Utility small embedders (chrom / organ)
# ────────────────────────────────────────────────────────────────────────────
class ChromosomeEmbedder(nn.Module):
    def __init__(self, n, d=768):
        super().__init__()
        self.emb = nn.Embedding(n, d)
        self.register_buffer("scale", torch.tensor(0.2, dtype=torch.float32))
    def forward(self, ids):
        return self.emb(ids) * self.scale

class OrganEmbedder(nn.Module):
    """조직(암종) Id를 임베딩 (Global Transformer 입력에 더해 사용 가능)"""
    def __init__(self, n, d=768, scale: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n, d)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
    def forward(self, ids):
        return self.emb(ids) * self.scale

# ────────────────────────────────────────────────────────────────────────────
# Hierarchical Alpha Fusion
#   out_t = α_t · LN(cls_t) + (1-α_t) · LN(feat_t)
#   α_t   = σ( (logit_sum + logit_offset) / logit_temperature )
#          where logit_sum = logit(α₀)
#                         + w_tok·logit(α_tok)
#                         + w_seg·logit(α_seg)
#                         + w_dom·logit(α_dom)
#                         [+ b_organ]
#
#  ★ alpha_share_mode
#   - "token"  : 토큰별 α (기본)
#   - "segment": 세그먼트 평균 로짓으로 α를 만들어 모든 토큰에 공유
#   - "window" : 1D 평균풀링(k)로 스무스한 로짓을 만든 후 α 계산(마스크 보정)
#
#  ★ NEW: CLS stream에 얕은 depthwise 1D-CNN (병렬 분지 + 1×1 혼합)
# ────────────────────────────────────────────────────────────────────────────
class HierAlphaFusion(nn.Module):
    def __init__(self,
                 hidden_dim: int = 768,
                 chrom_n: int = 24,
                 # 게이트 조합 가중
                 w_tok: float = 1.0,
                 w_seg: float = 0.50,
                 w_dom: float = 0.25,
                 # 스트림 드롭아웃
                 use_stream_dropout: bool = True,
                 stream_dropout_p: float = 0.15,
                 # α warmup
                 use_alpha_warmup: bool = True,
                 alpha_target: float = 0.50,
                 warmup_epochs: int = 3,
                 warmup_lambda: float = 1e-4,
                 # 초기 글로벌 α₀
                 alpha0_init: float = 0.50,
                 # α 하한 옵션
                 alpha_min: float = 0.0,
                 enforce_alpha_min: bool = False,
                 # logit 보정(온도/오프셋)
                 logit_temperature: float = 1.0,
                 logit_offset: float = 0.0,
                 # per-organ α bias
                 num_organs: Optional[int] = None,
                 use_per_organ_alpha_bias: bool = False,
                 organ_bias_init: float = 0.0,
                 organ_bias_l2: float = 1e-4,
                 use_bias_warmup: bool = True,
                 bias_warmup_epochs: int = 3,
                 # α 공유/스무딩 모드
                 alpha_share_mode: str = "token",     # "token" | "segment" | "window"
                 alpha_window: int = 0,
                 # ── NEW: organ-specific CLS bias
                 use_cls_bias: bool = True,
                 cls_bias_scale: float = 0.2,
                 # ── NEW: shallow multi-dilation CLS CNN (병렬 고정)
                 use_cls_cnn: bool = True,
                 cls_cnn_branches: Optional[List[Tuple[int,int]]] = None,  # [(k,d), ...]
                 cls_cnn_scale: float = 0.2):
        super().__init__()
        self.w_tok = float(w_tok)
        self.w_seg = float(w_seg)
        self.w_dom = float(w_dom)

        self.use_stream_dropout = bool(use_stream_dropout)
        self.stream_dropout_p   = float(stream_dropout_p)

        # α warmup 설정
        self.use_alpha_warmup = bool(use_alpha_warmup)
        self.alpha_target     = float(alpha_target)
        self.warmup_epochs    = int(warmup_epochs)
        self.warmup_lambda    = float(warmup_lambda)

        # α 하한
        self.alpha_min = float(max(0.0, min(1.0, alpha_min)))
        self.enforce_alpha_min = bool(enforce_alpha_min)

        # logit 보정 파라미터
        self.logit_temperature = float(max(1e-6, logit_temperature))
        self.logit_offset      = float(logit_offset)

        # α 공유/스무딩 모드
        self.alpha_share_mode = str(alpha_share_mode).lower()
        self.alpha_window     = int(max(0, alpha_window))

        # LayerNorm 두 스트림
        self.ln_cls  = nn.LayerNorm(hidden_dim)
        self.ln_feat = nn.LayerNorm(hidden_dim)

        # token gate: α_tok(B,T,1)
        self.mlp_token = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//4),
            nn.GELU(),
            nn.Linear(hidden_dim//4, 1)
        )
        # segment gate: α_seg(B,1,1)
        seg_in = 8  # |cls| mean/std, |feat| mean/std, cos mean/std, pad2
        self.mlp_segment = nn.Sequential(
            nn.Linear(seg_in, max(16, hidden_dim//32)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim//32), 1)
        )
        # domain(chrom) gate: α_dom(B,1,1)
        self.chrom_emb = nn.Embedding(int(chrom_n), max(8, hidden_dim//64))
        self.mlp_chrom = nn.Linear(max(8, hidden_dim//64), 1)

        # global alpha0 (학습 스칼라)
        alpha0_init = float(min(max(alpha0_init, 1e-6), 1-1e-6))
        logit_init  = math.log(alpha0_init / (1.0 - alpha0_init))
        self.alpha0_logit = nn.Parameter(torch.tensor(logit_init, dtype=torch.float32))

        # per-organ α bias (logit 공간 가산)
        self.use_per_organ_alpha_bias = bool(use_per_organ_alpha_bias and (num_organs is not None) and (num_organs > 0))
        self.organ_bias_l2 = float(max(0.0, organ_bias_l2))
        self.use_bias_warmup   = bool(use_bias_warmup)
        self.bias_warmup_epochs= int(max(0, bias_warmup_epochs))
        if self.use_per_organ_alpha_bias:
            self.organ_bias = nn.Embedding(int(num_organs), 1)
            nn.init.constant_(self.organ_bias.weight, float(organ_bias_init))
        else:
            self.organ_bias = None

        # ── NEW) Organ-specific CLS bias (additive)
        self.use_cls_bias = bool(use_cls_bias and (num_organs is not None) and (num_organs > 0))
        if self.use_cls_bias:
            self.cls_bias = nn.Embedding(int(num_organs), hidden_dim)
            nn.init.zeros_(self.cls_bias.weight)
            self.register_buffer("cls_bias_scale", torch.tensor(float(cls_bias_scale), dtype=torch.float32))

        # ── NEW) Multi-dilation shallow depthwise 1D-CNN over CLS sequence (Parallel)
        self.use_cls_cnn  = bool(use_cls_cnn)
        if self.use_cls_cnn:
            # 기본 브랜치: (3,1) + (3,2)
            if not cls_cnn_branches:
                cls_cnn_branches = [(3,1), (3,2)]
            self.cls_cnn_branches = list(cls_cnn_branches)

            # 분지 depthwise conv 들
            self.cls_cnn = nn.ModuleList()
            for k, d in self.cls_cnn_branches:
                k = int(max(1, k)); d = int(max(1, d))
                pad = (k // 2) * d
                conv = nn.Conv1d(hidden_dim, hidden_dim,
                                 kernel_size=k, padding=pad,
                                 dilation=d, groups=hidden_dim, bias=False)
                # 안전 시작(영향 0)
                nn.init.zeros_(conv.weight)
                self.cls_cnn.append(conv)

            # 분지 concat 후 D로 축소할 pointwise 1×1 conv
            self.cls_pw = nn.Conv1d(hidden_dim * len(self.cls_cnn), hidden_dim, kernel_size=1, bias=False)
            nn.init.zeros_(self.cls_pw.weight)

            self.register_buffer("cls_cnn_scale", torch.tensor(float(cls_cnn_scale), dtype=torch.float32))

        # 모니터링 버퍼: 유효 토큰 평균 α
        self.register_buffer("_alpha_dbg", torch.tensor(0.5, dtype=torch.float32))

        # 초기화(게이트 마지막 층/도메인 로짓 0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            if isinstance(self.mlp_token[-1], nn.Linear):
                self.mlp_token[-1].weight.zero_()
                if self.mlp_token[-1].bias is not None:
                    self.mlp_token[-1].bias.zero_()
            if isinstance(self.mlp_segment[-1], nn.Linear):
                self.mlp_segment[-1].weight.zero_()
                if self.mlp_segment[-1].bias is not None:
                    self.mlp_segment[-1].bias.zero_()
            self.mlp_chrom.weight.zero_()
            if self.mlp_chrom.bias is not None:
                self.mlp_chrom.bias.zero_()

    @staticmethod
    def _segment_stats(c: torch.Tensor, f: torch.Tensor, mask: torch.Tensor):
        """
        c,f: (B,T,D), mask: (B,T) True=valid -> (B,8)
        통계: |c| mean/std, |f| mean/std, cos mean/std, pad zeros(2)
        """
        eps = 1e-6
        B, T, D = c.shape
        c = torch.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
        f = torch.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)
        maskf = mask.to(dtype=c.dtype)
        Teff  = maskf.sum(dim=1).clamp_min(1.0)

        cn  = torch.linalg.vector_norm(c, dim=-1)
        fn  = torch.linalg.vector_norm(f, dim=-1)
        den = (cn.clamp_min(eps) * fn.clamp_min(eps))
        cos = ((c * f).sum(-1) / den).clamp(-1, 1)

        def _moments(x):
            xm = (x * maskf).sum(1) / Teff
            xc = x - xm.unsqueeze(1)
            var = ((xc * xc) * maskf).sum(1) / Teff
            std = var.clamp_min(0.0).sqrt()
            return xm, std

        mean_cn,  std_cn  = _moments(cn)
        mean_fn,  std_fn  = _moments(fn)
        mean_cos, std_cos = _moments(cos)

        stats = torch.stack([mean_cn, std_cn, mean_fn, std_fn, mean_cos, std_cos], dim=-1)  # (B,6)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        if stats.shape[-1] < 8:
            pad = torch.zeros(B, 8 - stats.shape[-1], device=stats.device, dtype=stats.dtype)
            stats = torch.cat([stats, pad], dim=-1)
        return stats  # (B,8)

    @staticmethod
    def _logit(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-6, 1-1e-6)
        return torch.log(x) - torch.log1p(1 - x)

    def _bias_scale(self, epoch_idx: Optional[int]) -> float:
        """organ bias warmup 스케일 (0→1 선형)"""
        if (not self.use_bias_warmup) or (epoch_idx is None) or (self.bias_warmup_epochs <= 0):
            return 1.0
        e = int(epoch_idx)
        if e <= 0: return 1e-3
        if e >= self.bias_warmup_epochs: return 1.0
        return max(1e-3, float(e) / float(self.bias_warmup_epochs))

    def _mask_aware_avgpool1d(self, x_bt: torch.Tensor, mask_bt: torch.Tensor, k: int) -> torch.Tensor:
        """
        x_bt:    (B,T)  (logit or scalar)
        mask_bt: (B,T)  (1=valid, 0=pad)
        k: kernel size
        return:  (B,T)  mask-aware 평균풀링 (간단 보정)
        """
        B, T = x_bt.shape
        a = x_bt.unsqueeze(1)           # (B,1,T)
        m = mask_bt.unsqueeze(1)        # (B,1,T)
        y = F.avg_pool1d(a, kernel_size=k, stride=1, padding=k//2).squeeze(1)  # (B,T)
        w = F.avg_pool1d(m, kernel_size=k, stride=1, padding=k//2).squeeze(1).clamp_min(1e-6)  # (B,T)
        y = y / w
        # PAD 위치는 원본 유지
        y = torch.where(mask_bt > 0, y, x_bt)
        return y

    def _apply_cls_cnn(self, c_btD: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        c_btD: (B,T,D), valid_mask: (B,T) True=valid
        depthwise CNN 병렬 분지를 concat→pointwise로 섞어서 residual로 더한다.
        """
        if not self.use_cls_cnn or len(self.cls_cnn) == 0:
            return c_btD

        x = c_btD.transpose(1, 2)  # (B,D,T)
        # 병렬 분지
        outs = [conv(x) for conv in self.cls_cnn]               # list of (B,D,T)
        ycat = torch.cat(outs, dim=1)                           # (B, D*#branches, T)
        ymix = self.cls_pw(ycat)                                # (B, D, T)
        c_new = c_btD + self.cls_cnn_scale * ymix.transpose(1, 2)  # (B,T,D)

        # PAD 위치는 0으로 정리(컨볼루션 스필오버 방지)
        if valid_mask is not None:
            c_new = torch.where(valid_mask.unsqueeze(-1), c_new, torch.zeros_like(c_new))
        return c_new

    def forward(self,
                cls_emb: torch.Tensor,              # (B,T,D)
                feat_emb: torch.Tensor,             # (B,T,D)
                chrom_id: Optional[torch.Tensor] = None,   # (B,)
                valid_mask: Optional[torch.Tensor] = None, # (B,T) True=valid
                epoch_idx: Optional[int] = None,
                tissue_ids: Optional[torch.Tensor] = None  # per-organ α
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = cls_emb.shape
        device  = cls_emb.device

        # valid mask normalize (True=valid)
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            if valid_mask.dim() != 2 or valid_mask.shape != (B, T):
                valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        if chrom_id is None:
            chrom_id = torch.zeros(B, dtype=torch.long, device=device)

        # LayerNorm
        c = self.ln_cls(cls_emb)
        f = self.ln_feat(feat_emb)

        # ── Organ-specific CLS bias (additive)
        if self.use_cls_bias and (tissue_ids is not None):
            c = c + self.cls_bias_scale * self.cls_bias(tissue_ids).unsqueeze(1)

        # ── Multi-dilation shallow CLS CNN (parallel residual)
        if self.use_cls_cnn:
            c = self._apply_cls_cnn(c, valid_mask)

        # optional stream dropout on cls
        if self.training and self.use_stream_dropout and random.random() < self.stream_dropout_p:
            c = 0.0 * c

        # gates (원천 확률형)
        a_tok = torch.sigmoid(self.mlp_token(torch.cat([c, f], dim=-1)))  # (B,T,1)
        seg_stats = self._segment_stats(c.detach(), f.detach(), valid_mask)  # (B,8)
        a_seg = torch.sigmoid(self.mlp_segment(seg_stats)).unsqueeze(1)      # (B,1,1)
        a_dom = torch.sigmoid(self.mlp_chrom(self.chrom_emb(chrom_id))).unsqueeze(1)  # (B,1,1)

        # stabilize
        a_tok = torch.nan_to_num(a_tok, nan=0.5, posinf=1-1e-6, neginf=1e-6)
        a_seg = torch.nan_to_num(a_seg, nan=0.5, posinf=1-1e-6, neginf=1e-6)
        a_dom = torch.nan_to_num(a_dom, nan=0.5, posinf=1-1e-6, neginf=1e-6)

        # ── logit 공간으로 전환
        z_tok = self._logit(a_tok)                     # (B,T,1)
        z_seg = self._logit(a_seg)                     # (B,1,1)
        z_dom = self._logit(a_dom)                     # (B,1,1)

        # ── α 공유/스무딩 모드
        maskf = valid_mask.float()                     # (B,T)
        if self.alpha_share_mode == "segment":
            den = maskf.sum(dim=1, keepdim=True).clamp_min(1.0)
            z_tok_seg = ((z_tok.squeeze(-1) * maskf).sum(dim=1, keepdim=True) / den).unsqueeze(-1)  # (B,1,1)
            z_tok_eff = z_tok_seg.expand(B, T, 1)
        elif self.alpha_share_mode == "window" and self.alpha_window > 1:
            k = int(self.alpha_window)
            z_tok_eff_2d = self._mask_aware_avgpool1d(z_tok.squeeze(-1), maskf, k)  # (B,T)
            z_tok_eff = z_tok_eff_2d.unsqueeze(-1)
        else:
            z_tok_eff = z_tok

        # ── 최종 α 로짓 합성
        logit_sum = (
            self.alpha0_logit
            + self.w_tok * z_tok_eff
            + self.w_seg * z_seg.expand_as(z_tok_eff)
            + self.w_dom * z_dom.expand_as(z_tok_eff)
        )

        # per-organ α bias (logit 공간 가산)
        if self.use_per_organ_alpha_bias:
            assert tissue_ids is not None, "HierAlphaFusion: tissue_ids (organ ids) must be provided for per-organ alpha."
            bias = self.organ_bias(tissue_ids).unsqueeze(1).expand(B, T, 1)
            scale = self._bias_scale(epoch_idx)
            logit_sum = logit_sum + scale * bias

        # logit 보정(오프셋/온도) → α
        logit_sum = (logit_sum + self.logit_offset) / self.logit_temperature
        alpha = torch.sigmoid(logit_sum)  # (B,T,1)

        # α 하한 강제
        if self.enforce_alpha_min and self.alpha_min > 0.0:
            alpha = alpha.clamp(min=self.alpha_min)

        out = alpha * c + (1.0 - alpha) * f

        # 유효 토큰 평균 α 기록
        self._alpha_dbg = alpha[valid_mask.unsqueeze(-1)].mean().detach()

        return out, alpha

    def alpha_regularizer(self, alpha_map: torch.Tensor, epoch_idx: Optional[int]):
        """
        warmup_epochs 동안 평균 α를 alpha_target 근처로 유도:
          L_reg = λ · (mean(α) - target)^2  +  (organ_bias_l2 · ||b||^2)
        """
        reg = torch.tensor(0.0, device=alpha_map.device)
        # α mean warmup
        if self.training and self.use_alpha_warmup and (epoch_idx is not None) and (epoch_idx < self.warmup_epochs):
            reg = reg + self.warmup_lambda * (alpha_map.mean() - self.alpha_target).pow(2)
        # organ bias L2
        if self.use_per_organ_alpha_bias and (self.organ_bias_l2 > 0.0):
            reg = reg + self.organ_bias_l2 * (self.organ_bias.weight.pow(2).mean())
        return reg

# ────────────────────────────────────────────────────────────────────────────
# Adapter with backward-compatible signature
# ────────────────────────────────────────────────────────────────────────────
class FeatClsFusion(nn.Module):
    """
    (호환 어댑터) 다양한 키워드 인자를 허용하여 내부 HierAlphaFusion을 호출.
    - alpha_min / enforce_alpha_min / num_organs / use_per_organ_alpha_bias /
      logit_temperature / logit_offset / alpha_share_mode / alpha_window /
      use_cls_bias / use_cls_cnn / cls_cnn_branches / cls_cnn_scale 등 전달 가능.
    """
    def __init__(self, hidden_dim=768, chrom_n=None, **alpha_kwargs):
        super().__init__()
        chrom_n = int(chrom_n) if chrom_n is not None else len(CHROM_LIST_24)
        self.core = HierAlphaFusion(hidden_dim=hidden_dim, chrom_n=chrom_n, **alpha_kwargs)

    def forward(self, *args, **kwargs):
        # CLS
        cls_emb = kwargs.get("cls", None)
        if cls_emb is None:
            cls_emb = kwargs.get("cls_b", kwargs.get("cls_a", kwargs.get("cls_emb", None)))
        # FEAT
        feat_emb = kwargs.get("feat", None)
        if feat_emb is None:
            feat_emb = kwargs.get("feature", kwargs.get("feat_emb", kwargs.get("x", None)))
        # positional fallback
        if (cls_emb is None or feat_emb is None) and len(args) >= 2:
            if cls_emb is None:  cls_emb = args[0]
            if feat_emb is None: feat_emb = args[1]
        if cls_emb is None or feat_emb is None:
            raise TypeError("FeatClsFusion.forward expects (cls_emb, feat_emb) via args or kwargs "
                            "(e.g., cls_b=..., feat=...)")

        # others
        chrom_id    = kwargs.get("chrom", kwargs.get("chrom_id", kwargs.get("dom", kwargs.get("domain", None))))
        valid_mask  = kwargs.get("mask", kwargs.get("valid_mask", kwargs.get("attn_mask", None)))
        epoch_idx   = kwargs.get("epoch", kwargs.get("epoch_idx", None))
        tissue_ids  = kwargs.get("tissue_ids", kwargs.get("organ_ids", None))  # per-organ α용

        return self.core(cls_emb, feat_emb,
                         chrom_id=chrom_id,
                         valid_mask=valid_mask,
                         epoch_idx=epoch_idx,
                         tissue_ids=tissue_ids)
