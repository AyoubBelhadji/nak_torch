from dataclasses import dataclass
from typing import Callable, Optional

import torch
from nak_torch.tools import GaussianModel
from nak_torch.tools.types import BatchPtType, BatchLogDensity
from . import joker, aristoff_bangerth

@dataclass
class Problem:
    model: GaussianModel | BatchLogDensity
    reference_samples: Optional[BatchPtType]
    prior_sample: Callable[[torch.Generator, int], BatchPtType]
    is_batched: bool = True

def aristoff_bangerth_logpdf():
    return Problem(aristoff_bangerth.build(), None, aristoff_bangerth.prior_sample)

def joker_logpdf():
    rng = torch.Generator()
    rng.manual_seed(0)
    samples = joker.sample(rng, 10000)
    return Problem(torch.compile(joker.logpdf), samples, joker.prior_sample, False)