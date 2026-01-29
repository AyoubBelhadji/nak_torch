from dataclasses import dataclass
from typing import Callable, Optional
from functools import partial

import torch
from nak_torch.tools import GaussianModel
from nak_torch.tools.types import BatchPtType, BatchLogDensity, PtType
from nak_torch.functions import joker
from . import aristoff_bangerth, kfrflow_examples

__all__ = [
    "aristoff_bangerth_logpdf",
    "joker_logpdf",
    "butterfly_logpdf",
    "spaceships_logpdf",
]

def gaussian_sampler(d: int, rng: torch.Generator, n_samples: int):
    return torch.normal(0., 1., size=(n_samples, d), generator=rng)

@dataclass
class Problem:
    model: GaussianModel | BatchLogDensity
    reference_samples: Optional[BatchPtType]
    prior_sample: Callable[[torch.Generator, int], BatchPtType]
    true_mean: Optional[PtType] = None
    true_cov: Optional[torch.Tensor] = None

def aristoff_bangerth_logpdf(dim: int):
    assert dim == 64
    return Problem(aristoff_bangerth.build(), None, aristoff_bangerth.prior_sample)

def joker_logpdf(dim: int):
    assert dim == 2
    rng = torch.Generator(device=torch.get_default_device())
    rng.manual_seed(0)
    jokerdata = joker.JokerData().to(dtype=torch.get_default_dtype(), device=torch.get_default_device())
    samples = joker.JokerSampler(jokerdata)(rng, 10000)
    return Problem(torch.compile(lambda pts: joker.joker_logpdf_full(jokerdata, pts)), samples, joker.prior_sample)

def butterfly_logpdf(dim: int):
    assert dim == 2
    model = GaussianModel(
        kfrflow_examples.butterfly_model,
        1 / kfrflow_examples.butterfly_like_var,
        1., kfrflow_examples.butterfly_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2))

def spaceships_logpdf(dim: int):
    assert dim == 2
    model = GaussianModel(
            kfrflow_examples.spaceships_model,
        1 / kfrflow_examples.spaceships_like_var,
        1., kfrflow_examples.spaceships_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2))

def doughnut_logpdf(dim: int):
    model = GaussianModel(
            kfrflow_examples.doughnut_model,
        1 / kfrflow_examples.doughnut_like_var,
        1., kfrflow_examples.doughnut_obs, is_vectorized=True
    )
    return Problem(model, None, partial(gaussian_sampler, 2), torch.zeros(dim))

def funnel_logpdf(dim: int):
    @torch.compile
    def _logpdf_impl(pt: torch.Tensor, var1: float = 9.):
        d = pt.shape[-1]
        x0 = pt[...,0]
        term_0 = x0**2 / var1 + torch.log(var1*torch.ones(()))
        rest_terms = torch.sum(pt[...,1:]**2,-1) / torch.exp(x0)  + (d-1)*x0
        return -0.5 * (term_0 + rest_terms)
    N_samples = 10000
    samples_0 = 3. * torch.randn((N_samples,1))
    samples_rest = torch.randn((N_samples, dim-1)) * torch.exp(0.5 * samples_0)
    samples = torch.column_stack((samples_0, samples_rest))
    return Problem(_logpdf_impl, samples, partial(gaussian_sampler, dim))