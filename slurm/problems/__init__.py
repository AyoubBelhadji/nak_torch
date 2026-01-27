from dataclasses import dataclass
from typing import Optional

from torch import Tensor
import torch
from nak_torch.tools import GaussianModel
from nak_torch.tools.types import BatchPtType, BatchLogDensity
from .aristoff_bangerth import build_aristoff_bangerth
from . import joker

@dataclass
class Problem:
    model: GaussianModel | BatchLogDensity
    reference_samples: Optional[BatchPtType]

def aristoff_bangerth_logpdf():
    return Problem(build_aristoff_bangerth(), None)

def joker_logpdf():
    rng = torch.Generator()
    rng.manual_seed(0)
    samples = joker.sample(rng, 100000)
    return Problem(torch.compile(joker.logpdf), samples)