#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
driverformer.losses ? loss functions (NHPP + optional rate losses)

Exports
-------
NHPP likelihood (bin-level / seg-weighted)
  - trapezoid_nhpp_loss
  - trapezoid_nhpp_loss_segment_weighted

Rate supervision (optional; if losses/rate.py exists)
  - rate_huber_loss
  - rate_mse_loss
  - poisson_nll_rate
"""

from .nhpp import (
    trapezoid_nhpp_loss,
    trapezoid_nhpp_loss_segment_weighted,
)

__all__ = [
    "trapezoid_nhpp_loss",
    "trapezoid_nhpp_loss_segment_weighted",
]

# Optional rate losses (present only if rate.py exists)
try:
    from .rate import (
        rate_huber_loss,
        rate_mse_loss,
        poisson_nll_rate,
    )
except Exception:
    pass
else:
    __all__ += [
        "rate_huber_loss",
        "rate_mse_loss",
        "poisson_nll_rate",
    ]
