import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import BatchPtType, BatchLogDensity

def eks_step(particles: BatchPtType, log_dens: BatchLogDensity) -> BatchPtType:
    return particles