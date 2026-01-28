from typing import Optional
import torch
from torch import Tensor
from nak_torch.functions import aristoff_bangerth as ab
from nak_torch import GaussianModel


def build(
        N: int = 32,
        N_obs: int = 13,
        sig_lik: float = 0.05,
        sig_pr: float = 2.0,
        z_hat: Optional[Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        use_compiled: bool = True,
        device: Optional[torch.device] = None
):
    z_hat = ab.z_hat_noisy
    z_hat = torch.as_tensor(z_hat, dtype=dtype, device=device)
    sig_lik_sq = sig_lik**2
    sig_pr_sq = sig_pr**2
    H_obs, *solve_args = ab.build_forward_solver_args(N, N_obs, dtype=dtype, device=device)
    def forward_model(log_theta: Tensor):
        theta = log_theta.exp()
        out = ab.forward_solver(theta, N, H_obs, *solve_args)
        return out @ H_obs.T
    if use_compiled:
        forward_model = torch.compile(forward_model)
    return GaussianModel(forward_model, 1 / sig_lik_sq, 1 / sig_pr_sq, z_hat, prior_mean = 0., is_vectorized=True)

def prior_sample(rng: torch.Generator, N_batch: int):
    return torch.normal(0., 2., size=(N_batch, 64), generator=rng, device=rng.device)