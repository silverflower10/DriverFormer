#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/nhpp_head.py — NHPP rate head (safe version, multi-tissue ready)

Updated on Mon Jul  7 19:20:00 2025
- Transformer 출력 x(B,T,D) → per-kb rate λ(B,T)
- softplus로 양수 보장
- exp(log_c) 오버플로우 방지(clamp)
- 최종 λ는 [rate_min, rate_max]로 클램프 + NaN/Inf 치환
- backward 시 λ-그래디언트 NaN/Inf를 0으로 살균(hook)해 MulBackward0 NaN 전파 차단
- (NEW) ConditionalNHPPHead: 'shared'|'scale'|'film'|'per_head' 4가지 모드 지원
  * shared  : 단일(기존 NHPPHead와 동일 동작)
  * scale   : 공유 헤드 + tissue별 Δlog_c(스케일 보정)
  * film    : 공유 헤드 + FiLM(γ,β)로 히든 조절
  * per_head: tissue별 LN/Linear/log_c를 각각 보유(표현력 최대)
    - **주의**: per_head에서도 전역 log_c를 보유하여 보정 루틴과 호환되며,
                실제 스케일은 exp(log_c + log_cs[t])로 적용됨.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NHPPHead", "ConditionalNHPPHead", "CondCfg", "build_nhpp_head"]


# =========================
# 공통 설정/구조
# =========================
@dataclass
class CondCfg:
    mode: Literal["shared", "scale", "film", "per_head"] = "shared"
    num_tissues: int = 1
    film_hidden: int = 0
    tie_ln: bool = True   # per_head에서 LayerNorm 공유 여부


# =========================
# 실구현: 조건화 NHPP 헤드
# =========================
class ConditionalNHPPHead(nn.Module):
    """
    ConditionalNHPPHead(hidden_dim, cond=CondCfg(...),
                        rate_min=1e-9, rate_max=1e6,
                        ln_eps=1e-5, softplus_threshold=20.0)

    입력
    ----
    x           : (B, T, D)   인코더 히든
    tissue_ids  : (B,) 또는 (B,T) int (mode='shared'면 생략 가능)

    출력
    ----
    λ           : (B, T)      per-kb rate (양수, clamp/살균 적용)

    모드
    ----
    - shared   : 단일 헤드 (기존 NHPPHead와 동일)
    - scale    : 공유 헤드 + tissue별 Δlog_c 보정
    - film     : 공유 헤드 + FiLM(γ,β)로 히든 조절
    - per_head : tissue별 LN/Linear/log_c 개별 보유 + 전역 log_c(보정루틴 호환)
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        cond: CondCfg = CondCfg(),
        rate_min: float = 1e-9,
        rate_max: float = 1e6,
        ln_eps: float = 1e-5,
        softplus_threshold: float = 20.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cond = cond
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.sp_th = float(softplus_threshold)

        K = int(cond.num_tissues)
        m = cond.mode

        if m == "per_head":
            # 조직별 완전 분기 + 전역 log_c (보정 루틴 호환)
            self.tie_ln = bool(cond.tie_ln)
            if self.tie_ln:
                self.ln_shared = nn.LayerNorm(self.hidden_dim, eps=ln_eps)
            else:
                self.lns = nn.ModuleList(
                    [nn.LayerNorm(self.hidden_dim, eps=ln_eps) for _ in range(K)]
                )
            self.lins = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(K)])
            # 전역 log_c (스칼라) — calibrate_* 호환용 + 전체 공통 스케일
            self.log_c = nn.Parameter(torch.zeros(()))
            # 헤드별 log_cs (스칼라)
            self.log_cs = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(K)])
            # 초기화
            for lin in self.lins:
                nn.init.xavier_uniform_(lin.weight)

        else:
            # 공유 경로
            self.ln = nn.LayerNorm(self.hidden_dim, eps=ln_eps)
            self.lin = nn.Linear(self.hidden_dim, 1)
            self.log_c = nn.Parameter(torch.zeros(()))  # 전역 스케일(log)
            nn.init.xavier_uniform_(self.lin.weight)

            if m == "scale":
                # 조직별 Δlog_c (스칼라)
                self.tissue_logc = nn.Embedding(K, 1)
                nn.init.zeros_(self.tissue_logc.weight)

            elif m == "film":
                # one-hot(K) → (γ,β) in R^{2D}
                if cond.film_hidden and cond.film_hidden > 0:
                    self.film = nn.Sequential(
                        nn.Linear(K, cond.film_hidden),
                        nn.SiLU(),
                        nn.Linear(cond.film_hidden, 2 * self.hidden_dim),
                    )
                else:
                    self.film = nn.Linear(K, 2 * self.hidden_dim, bias=False)
                # 선형층들 일괄 초기화
                for mod in self.modules():
                    if isinstance(mod, nn.Linear):
                        nn.init.xavier_uniform_(mod.weight)

    # ---------- helpers ----------
    @staticmethod
    def _ensure_bt(tissue_ids: torch.Tensor, T: int) -> torch.Tensor:
        # (B,) → (B,T); (B,T)면 그대로
        if tissue_ids.dim() == 1:
            tissue_ids = tissue_ids[:, None].expand(-1, T)
        return tissue_ids

    @staticmethod
    def _one_hot(tissue_ids: torch.Tensor, K: int, T: int, device) -> torch.Tensor:
        # (B,) or (B,T) → (B,T,K) one-hot
        tissue_ids = ConditionalNHPPHead._ensure_bt(tissue_ids, T)
        B = tissue_ids.shape[0]
        oh = torch.zeros(B, T, K, device=device, dtype=torch.float32)
        oh.scatter_(2, tissue_ids.long().clamp(min=0, max=K - 1).unsqueeze(-1), 1.0)
        return oh

    # === NEW: 현재 모드에서 tissue_ids에 해당하는 스케일 s 반환 ===
    def scale_for_ids(self, tissue_ids: torch.Tensor) -> torch.Tensor:
        """
        tissue_ids: (B,) or (B,T)
        return    : (B,1) if input (B,), else (B,T)
                    s = exp(global_log_c [+ organ_offset])
        """
        m = self.cond.mode
        if tissue_ids.dim() == 2:
            ids_bt = tissue_ids.long()
            want_bt = True
        else:
            ids_bt = tissue_ids.long().unsqueeze(-1)  # (B,1)
            want_bt = False

        if m == "per_head":
            # head별 log_cs를 텐서로 스택 후 인덱싱
            log_cs_vec = torch.stack([p for p in self.log_cs], dim=0)  # (K,)
            log_eff = (self.log_c + log_cs_vec[ids_bt]).clamp(-20, 20)
            s = torch.exp(log_eff)
        elif m == "scale":
            K = int(self.cond.num_tissues)
            ids_bt = ids_bt.clamp(min=0, max=K - 1)
            # 전역 + Δlog_c
            log_eff = (self.log_c + self.tissue_logc(ids_bt).squeeze(-1)).clamp(-20, 20)
            s = torch.exp(log_eff)
        else:
            # shared / film : 전역 스케일만
            s_scalar = torch.exp(self.log_c.clamp(-20, 20))
            s = s_scalar.expand_as(ids_bt)

        # float32로 보장
        s = s.to(dtype=torch.float32, device=tissue_ids.device)
        if not want_bt:
            # 입력이 (B,)였으면 (B,1) 반환
            return s
        return s

    # === NEW: 조직별 log-c 조회 ===
    @torch.no_grad()
    def get_logc_per_organ(self, organ_idx: int) -> float:
        if self.cond.mode == "per_head":
            return float((self.log_c + self.log_cs[organ_idx]).detach().cpu())
        elif self.cond.mode == "scale":
            return float((self.log_c + self.tissue_logc.weight[organ_idx, 0]).detach().cpu())
        else:
            return float(self.log_c.detach().cpu())

    # === NEW: 조직별 log-c 설정 ===
    @torch.no_grad()
    def set_logc_per_organ(self, organ_idx: int, new_logc: float):
        """
        new_logc: 해당 조직의 목표 log-scale 값.
                  (전역 + 조직오프셋)의 합이 new_logc가 되도록 내부 파라미터 조정.
        """
        if self.cond.mode == "per_head":
            # log_cs[o] ← new_logc - log_c
            target = float(new_logc) - float(self.log_c.detach())
            self.log_cs[organ_idx].copy_(torch.tensor(target, device=self.log_cs[organ_idx].device))
        elif self.cond.mode == "scale":
            # tissue_logc[o] ← new_logc - log_c
            target = float(new_logc) - float(self.log_c.detach())
            self.tissue_logc.weight.data[organ_idx, 0] = torch.tensor(
                target, device=self.tissue_logc.weight.device
            )
        else:
            # shared/film: 전역만 존재
            self.log_c.copy_(torch.tensor(float(new_logc), device=self.log_c.device))

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, tissue_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,T,D), tissue_ids: (B,) or (B,T) (mode='shared'면 생략 가능)
        return: λ(B,T)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,D); got {tuple(x.shape)}")
        B, T, D = x.shape
        m = self.cond.mode

        # ======== Shared (단일) ========
        if m == "shared" or tissue_ids is None:
            x_ = self.ln(x.contiguous())
            rate = F.softplus(self.lin(x_), beta=1.0, threshold=self.sp_th).squeeze(-1)  # (B,T) ≥ 0
            lam = rate * torch.exp(self.log_c.clamp(-20, 20))

        # ======== Per-head (조직별 완전 분기) ========
        elif m == "per_head":
            if tissue_ids is None:
                raise ValueError("per_head mode requires tissue_ids")
            tissue_ids = self._ensure_bt(tissue_ids, T)
            lam = x.new_empty(B, T)

            # 전역 스케일 (필요시 참고용)
            # global_scale = torch.exp(self.log_c.clamp(-20, 20))

            for t in range(int(self.cond.num_tissues)):
                sel = (tissue_ids == t)  # (B,T) bool
                if not sel.any():
                    continue
                x_t = x[sel].view(-1, D)  # (N,D)
                ln = self.ln_shared if getattr(self, "tie_ln", True) else self.lns[t]
                rate_t = F.softplus(
                    self.lins[t](ln(x_t.contiguous())),
                    beta=1.0, threshold=self.sp_th
                ).squeeze(-1)  # (N,)

                # 헤드별 스케일 = exp(global log_c + head log_cs[t])
                head_scale = torch.exp((self.log_c + self.log_cs[t]).clamp(-20, 20))
                lam_t = rate_t * head_scale

                # 안전장치
                lam_t = torch.nan_to_num(lam_t, nan=self.rate_min, posinf=self.rate_max, neginf=self.rate_min)
                lam_t = lam_t.clamp(self.rate_min, self.rate_max)
                lam[sel] = lam_t

        # ======== Scale / FiLM (공유 경로) ========
        else:
            if tissue_ids is None:
                raise ValueError(f"{m} mode requires tissue_ids")
            x_ = self.ln(x.contiguous())

            if m == "film":
                K = int(self.cond.num_tissues)
                oh = self._one_hot(tissue_ids, K, T, x.device)   # (B,T,K)
                gam_beta = self.film(oh)                         # (B,T,2D)
                gamma, beta = gam_beta.split(D, dim=-1)
                x_ = gamma * x_ + beta

            rate = F.softplus(self.lin(x_), beta=1.0, threshold=self.sp_th).squeeze(-1)  # (B,T)
            lam = rate * torch.exp(self.log_c.clamp(-20, 20))

            if m == "scale":
                K = int(self.cond.num_tissues)
                tissue_ids = self._ensure_bt(tissue_ids, T)
                # Δlog_c 추가 → exp로 스케일
                lam = lam * torch.exp(
                    self.tissue_logc(tissue_ids.clamp(min=0, max=K - 1)).squeeze(-1).clamp(-20, 20)
                )

        # ======== 공통 안전장치 ========
        lam = torch.nan_to_num(lam, nan=self.rate_min, posinf=self.rate_max, neginf=self.rate_min)
        lam = lam.clamp(self.rate_min, self.rate_max)
        if lam.requires_grad:
            lam.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
        return lam


# =========================
# 하위호환: 단일 NHPPHead 별칭
# =========================
class NHPPHead(ConditionalNHPPHead):
    """
    Backward-compatible alias for a single shared NHPP head.
    내부적으로 ConditionalNHPPHead(mode='shared')를 사용한다.
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        rate_min: float = 1e-9,
        rate_max: float = 1e6,
        ln_eps: float = 1e-5,
        softplus_threshold: float = 20.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            cond=CondCfg(mode="shared", num_tissues=1),
            rate_min=rate_min,
            rate_max=rate_max,
            ln_eps=ln_eps,
            softplus_threshold=softplus_threshold,
        )


# =========================
# 팩토리
# =========================
def build_nhpp_head(
    hidden_dim: int,
    mode: Literal["shared", "scale", "film", "per_head"],
    num_tissues: int,
    *,
    film_hidden: int = 0,
    tie_ln: bool = True,
    rate_min: float = 1e-9,
    rate_max: float = 1e6,
    ln_eps: float = 1e-5,
    softplus_threshold: float = 20.0,
) -> nn.Module:
    """
    사용 예:
      head = build_nhpp_head(
          hidden_dim=args.d_model, mode=args.head_mode, num_tissues=args.num_tissues,
          film_hidden=args.film_hidden, tie_ln=args.tie_ln
      )
    """
    if mode == "shared":
        return NHPPHead(hidden_dim=hidden_dim, rate_min=rate_min, rate_max=rate_max,
                        ln_eps=ln_eps, softplus_threshold=softplus_threshold)
    cond = CondCfg(mode=mode, num_tissues=num_tissues, film_hidden=film_hidden, tie_ln=tie_ln)
    return ConditionalNHPPHead(
        hidden_dim=hidden_dim, cond=cond,
        rate_min=rate_min, rate_max=rate_max,
        ln_eps=ln_eps, softplus_threshold=softplus_threshold
    )


# =========================
# Quick sanity (optional)
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, K = 2, 5, 32, 3
    x = torch.randn(B, T, D)
    tissues_bt = torch.tensor([[0,0,1,1,2],
                               [2,2,2,0,0]])  # (B,T)

    # 1) shared == 기존 NHPPHead
    h0 = NHPPHead(D)
    y0 = h0(x)
    assert y0.shape == (B, T) and torch.isfinite(y0).all() and (y0 > 0).all()

    # 2) scale
    h1 = build_nhpp_head(D, mode="scale", num_tissues=K)
    y1 = h1(x, tissue_ids=tissues_bt)
    assert y1.shape == (B, T)

    # 3) film
    h2 = build_nhpp_head(D, mode="film", num_tissues=K, film_hidden=64)
    y2 = h2(x, tissue_ids=tissues_bt)
    assert y2.shape == (B, T)

    # 4) per_head
    h3 = build_nhpp_head(D, mode="per_head", num_tissues=K, tie_ln=True)
    y3 = h3(x, tissue_ids=tissues_bt)
    assert y3.shape == (B, T)

    print("NHPP heads sanity OK:",
          y0.min().item(), y1.mean().item(), y2.mean().item(), y3.max().item())
