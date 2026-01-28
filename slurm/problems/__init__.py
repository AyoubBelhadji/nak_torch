from dataclasses import dataclass
from typing import Callable, Optional
from functools import partial

import torch
from nak_torch.tools import GaussianModel
from nak_torch.tools.types import BatchPtType, BatchLogDensity
from . import joker, aristoff_bangerth, kfrflow_examples

def gaussian_sampler(d: int, rng: torch.Generator, n_samples: int):
    return torch.normal(0., 1., size=(n_samples, d), generator=rng)

@dataclass
class Problem:
    model: GaussianModel | BatchLogDensity
    reference_samples: Optional[BatchPtType]
    prior_sample: Callable[[torch.Generator, int], BatchPtType]

def aristoff_bangerth_logpdf():
    return Problem(aristoff_bangerth.build(), None, aristoff_bangerth.prior_sample)

def joker_logpdf():
    rng = torch.Generator()
    rng.manual_seed(0)
    samples = joker.sample(rng, 10000)
    return Problem(torch.compile(joker.logpdf), samples, joker.prior_sample)

def butterfly_logpdf():
    model = GaussianModel(
        kfrflow_examples.butterfly_model,
        1 / kfrflow_examples.butterfly_like_var,
        1., kfrflow_examples.butterfly_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2))

def spaceships_logpdf():
    model = GaussianModel(
            kfrflow_examples.spaceships_model,
        1 / kfrflow_examples.spaceships_like_var,
        1., kfrflow_examples.spaceships_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2))

def doughnut_logpdf():
    model = GaussianModel(
            kfrflow_examples.doughnut_model,
        1 / kfrflow_examples.doughnut_like_var,
        1., kfrflow_examples.doughnut_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2))