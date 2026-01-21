# Optimization, sampling and quantization algorithms
# msip: mean shift interacting particles
# wfr-ips: interacting particles that follow the Wasserstein-Fisher-Rao gradient flow
# cbo: consensus-based optimization


# Ayoub Belhadji
# 05/12/2025

from . import eks
from .msip import msip
from .msip_ni import msip_ni
from .msip_greedy import msip_greedy
from .msip_geom_greedy import msip_geom_greedy
from .svgd import svgd
from .grad_aldi import grad_aldi


__all__ = [
    "msip",
    "msip_ni",
    "msip_greedy",
    "msip_geom_greedy",
    "svgd",
    "grad_aldi",
    "eks"
]
