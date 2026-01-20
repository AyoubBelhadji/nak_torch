import torch
from typing import Optional, Callable
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import BatchPtType, PtType
import warnings
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

ForwardModel = Callable[
    [Float[Tensor, " dim"]], Float[Tensor, " obs"]
]

BatchForwardModel = Callable[
    [Float[Tensor, "batch dim"]], Float[Tensor, "batch obs"]
]

def sym_sqrtm(A: Float[Tensor, "n n"]):
    e,v = torch.linalg.eigh(A)
    return torch.einsum("ij,j,kj->ik", v, e.sqrt_(), v)

@dataclass
class EKSModel:
    forward_model: BatchForwardModel
    likelihood_precision: float | Float[Tensor, "obs obs"]
    prior_precision: float | Float[Tensor, "dim dim"]
    true_obs: Float | Float[Tensor, " obs"]
    prior_mean: float | Float[Tensor, " dim"]
    def __init__(
            self,
            forward_model: ForwardModel | BatchForwardModel,
            likelihood_precision: float | Float[Tensor, "obs obs"] = 1.0,
            prior_precision: float | Float[Tensor, "dim dim"] = 1.0,
            true_obs: Float | Float[Tensor, " obs"] = torch.zeros(()),
            prior_mean: float | Float[Tensor, " dim"] = 0.0,
            is_vectorized: bool = False
    ):
        if is_vectorized:
            self.forward_model = forward_model
        else:
            self.forward_model = torch.vmap(forward_model)
        if prior_mean != 0.0:
            raise ValueError("Only support zero prior mean for now")
        self.likelihood_precision = likelihood_precision
        self.prior_precision = prior_precision
        self.true_obs = true_obs
        self.prior_mean = prior_mean


def build_eks_step(eks_model: EKSModel, dt: float):
    likelihood_precision = eks_model.likelihood_precision
    prior_precision = eks_model.prior_precision
    true_obs = eks_model.true_obs
    if isinstance(true_obs, Tensor):
        true_obs.reshape(1, -1)

    sqrt_2 = torch.sqrt(
        torch.tensor(2, dtype=true_obs.dtype, device=true_obs.device)
    )

    def eks_step(
            particles: BatchPtType,
            forecast_observations: Float[Tensor, "batch obs"]
    ) -> tuple[BatchPtType, Float[Tensor, "dim dim"]]:
        N_batch, dim = particles.shape
        particle_mean = particles.mean(0, True)
        forecast_obs_mean = forecast_observations.mean(0, True)
        obs_diff = forecast_observations - true_obs
        forecast_diff = forecast_observations - forecast_obs_mean
        prior_ens_diff = particles - particle_mean
        cov_prior = (prior_ens_diff.T @ prior_ens_diff) / N_batch

        if isinstance(likelihood_precision, float):
            likely_term = torch.einsum(
                "ko,jo,kd->jd",
                forecast_diff, obs_diff, particles
            )
            likely_term.mul_(dt * likelihood_precision / N_batch)
        else:
            likely_term = torch.einsum(
                "kp,pq,jq,kd->jd",
                forecast_diff, likelihood_precision, obs_diff, particles
            )
            likely_term.mul_(dt / N_batch)

        if isinstance(prior_precision, float):
            prior_term_premul = cov_prior.mul_(dt * prior_precision)
        elif isinstance(prior_precision, Tensor):
            prior_term_premul = torch.matmul(cov_prior, prior_precision).mul_(dt)
        else:
            raise ValueError()

        sqrt_prior_cov = sym_sqrtm(cov_prior)
        sqrt_prior_cov.mul_(sqrt_2)
        prior_term_premul.add_(torch.eye(dim))
        new_particles = torch.linalg.solve(prior_term_premul, particles - likely_term, left=False)
        return new_particles, sqrt_prior_cov

    return torch.compile(eks_step)

def eks(
    eks_model: EKSModel,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    noise=None,
    seed=None,
    device=None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    rng: Optional[torch.Generator] = None,
    **unused_kwargs
):
    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if rng is None:
        rng = torch.default_generator
    if seed is not None:
        rng.manual_seed(seed)

    particles: Tensor
    if init_particles is None:
        if bounds is None:
            particles = torch.randn((n_particles, dim), device=device)
        else:
            particles = (bounds[1] - bounds[0]) * torch.rand((n_particles, dim), device=device) + 1.5*bounds[0]
    else:
        particles = torch.as_tensor(init_particles, device=device)
    if keep_all:
        trajectories = torch.empty((n_steps, *particles.shape), dtype=particles.dtype)
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    eks_step = build_eks_step(eks_model, lr)
    noise_tens = torch.empty_like(particles)
    for idx in tqdm(range(n_steps)):
        forecast_obs = eks_model.forward_model(particles)
        with torch.no_grad():
            particles, noise_sqrt_cov = eks_step(particles, forecast_obs)
            noise_tens = torch.normal(mean=0., std=1., size=particles.shape, generator=rng, out=noise_tens)
            noise_samp = noise_tens @ noise_sqrt_cov
            particles = particles.add_(noise_samp)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    if keep_all:
        return trajectories, bounds
    else:
        return particles.unsqueeze_(0), bounds