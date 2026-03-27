from functools import partial

from jaxtyping import Float
import matplotlib.pyplot as plt
import torch

from torch import Tensor

import nak_torch
from nak_torch.algorithms import msip
from nak_torch.algorithms.msip import MSIPQuadGradientInformed, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# %% Target distribution: mixture of 3 Gaussians

def post_log_dens(pt):
    means = [
        torch.tensor([6.2,  -6.0]),
        torch.tensor([-4.0,  5.0]),
        torch.tensor([ 7.0,  0.0]),
    ]
    precisions = [5.0, 5.0, 5.0]
    weights    = [1/3, 1/3, 1/3]

    log_probs = []
    for mean, prec, w in zip(means, precisions, weights):
        diff = pt - mean
        lp = torch.log(torch.tensor(w)) - 0.5 * prec * torch.linalg.norm(diff, dim=-1)**2
        log_probs.append(lp)

    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)


# %% Quadrature rules

def mc_quad_rule(batch_size: int, N_quad: int = 100, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad(batch_size: int, N_spherical: int = 5, N_radial: int = 3):
    pts, wts = spherical_MC_radial_Laguerre(batch_size, N_spherical, 2, N_radial)
    return pts, wts


# %% Shared parameters

torch.manual_seed(1023921)
n_steps, n_particles = 50, 25
lr_msip             = 5e-3
kernel_length_scale = 1.0
kernel_diag_infl    = 1e-8
gradient_decay      = 1.0
bounds              = (-1000., 1000.)

init_particles = torch.randn((n_particles, 2)) / 0.9 + torch.tensor([3.2, -5.0])


# %% Run all four combinations

# --- GradientInformed + MC ---
msip_gi_mc = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    mc_quad_rule,
    gradient_decay,
)
traj_gi_mc, wts_gi_mc = msip(
    msip_gi_mc, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles.clone(),
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=False, compile_step=False, verbose=True,
)

# --- GradientInformed + Spherical ---
msip_gi_sph = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    spherical_quad,
    gradient_decay,
)
traj_gi_sph, wts_gi_sph = msip(
    msip_gi_sph, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles.clone(),
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=False, compile_step=False, verbose=True,
)

# --- GradientFree + MC ---
msip_gf_mc = MSIPQuadGradientFree(
    post_log_dens_batch,
    partial(mc_quad_rule, N_quad=50),
)
traj_gf_mc, wts_gf_mc = msip(
    msip_gf_mc, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles.clone(),
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=False, compile_step=False, verbose=True,
)

# --- GradientFree + Spherical ---
msip_gf_sph = MSIPQuadGradientFree(
    post_log_dens_batch,
    partial(spherical_quad, N_spherical=5, N_radial=3),
)
traj_gf_sph, wts_gf_sph = msip(
    msip_gf_sph, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles.clone(),
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=False, compile_step=False, verbose=True,
)


# %% Plot: one figure per algorithm

Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

results = [
    (traj_gi_mc[-1],  wts_gi_mc[-1],  "GradInformed + MC"),
    (traj_gi_sph[-1], wts_gi_sph[-1], "GradInformed + Spherical"),
    (traj_gf_mc[-1],  wts_gf_mc[-1],  "GradFree + MC"),
    (traj_gf_sph[-1], wts_gf_sph[-1], "GradFree + Spherical"),
]

for pts, wts, title in results:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20)
    sc = ax.scatter(
        pts[:, 0].cpu(), pts[:, 1].cpu(),
        c=wts.cpu(), alpha=0.6,
    )
    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_aspect(1.0)
    fig.tight_layout()
    plt.show()