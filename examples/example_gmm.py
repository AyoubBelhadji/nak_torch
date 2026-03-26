# %%
from functools import partial

from jaxtyping import Float
import matplotlib.pyplot as plt
import torch

from torch import Tensor

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, kfrflow, msip_adapt
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre

from nak_torch.tools.kernel import sqexp_kernel_elem as kernel_elem, sqexp_kernel_matrix
from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)

# %%

def weighted_cov(pts: Tensor, wts: Tensor):
    mean = wts @ pts
    second_moment = torch.einsum("b,bi,bj", wts, pts, pts)
    return second_moment - mean.outer(mean)

def post_log_dens(pt):
    means = [
        torch.tensor([6.2,  -6.0]),
        torch.tensor([-4.0, 5.0]),
        torch.tensor([7.0,  0.0]),
    ]
    precisions = [5.0, 5.0, 5.0]   # one per component
    weights    = [1/3, 1/3, 1/3]   # must sum to 1

    log_probs = []
    for mean, prec, w in zip(means, precisions, weights):
        diff = pt - mean
        lp = torch.log(torch.tensor(w)) - 0.5 * prec * torch.linalg.norm(diff, dim=-1)**2
        log_probs.append(lp)

    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()

init_mean = torch.tensor([3.2,-5.0])

def post_log_like(pt):
    return post_log_dens(pt) + torch.sum((pt - init_mean)**2, dim=-1)

post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# %% Parameters that are common to all algorithms
n_steps, n_particles = 1000, 25
lr = 0.5

# %% Initialization
init_particles = torch.randn((n_particles, 2)) + init_mean

# %% EKS
# trajectories_eks = eks(
#     model, n_particles=n_particles,
#     n_steps=n_steps, dim=2, lr=lr,
#     init_particles=init_particles,
#     keep_all=False, compile_step=False,
#     verbose=True
# )

# %% KFR

# delta_ts = torch.ones(1000)/1000
#def imq(pt1,pt2,h):
#    return 1/torch.sqrt(1 + (torch.linalg.norm(pt1-pt2) / h)**2)

trajectories_kfr = kfrflow(
    post_log_like,
    n_particles,
    n_steps, 2,
    init_particles=init_particles,
    kernel_length_scale = 1e-2,
    kernel_diag_infl=1e-5,
    # bounds=(-10,10),
    # kernel_elem=imq,
    keep_all=False,
    compile_step=False,
    verbose = True
)


# %% GI-ALDI

trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr/3, init_particles=init_particles,
    keep_all=False, compile_step=False,
    verbose=True,
)

# %% GF-ALDI

# trajectories_gfaldi = gradfree_aldi(
#     model, n_particles, n_steps, dim=2,
#     lr=lr, init_particles=init_particles,
#     keep_all=True, compile_step=False,
#     verbose=True
# )

# %% CBS

trajectories_cbs = cbs(
    post_log_dens, n_particles, n_steps, inverse_temp=0.95, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True, compile_step=False,
    verbose=True
)

# %% F-MSIP

kernel_length_scale = 0.5
bounds = (-100., 100.)
gradient_decay = 1.0
lr_msip = 100e-2
kernel_diag_infl = 1e-8



msip_fredholm = MSIPFredholm(
    gradient_decay,
    post_log_dens_grad_val_batch
)

trajectories_msip, traj_wts_msip = msip(
    msip_fredholm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles[:n_particles],
    # use_quantile_length_scale=0.05,
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=True,
    compile_step=False,
    verbose=True
)

# %% F-MSIP-adapt
trajectories_msip_adapt, msip_adapt_aux = msip_adapt(
    msip_fredholm, n_particles, n_steps, dim=2,
    use_quantile_length_scale=0.1,
    lr0=10.0, init_particles=init_particles[:n_particles],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=True,
    compile_step=False,
    verbose=True,
    rtol = 1e-4, atol = 1e-2
)

# %%


def mc_quad_rule(batch_size: int, N_quad: int = 5, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad(batch_size: int, N_spherical: int = 5, N_radial: int = 3):
    pts, wts = spherical_MC_radial_Laguerre(
        batch_size, N_spherical, 2, N_radial)
    return pts, wts


# %%
# kernel_length_scale = 1e-3
# gradient_decay = 1.
msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch, mc_quad_rule,
    gradient_decay
)

trajectories_msip_qg, traj_wts_msip_qg = msip(
    msip_quadgrad, n_particles, n_steps, dim=2,
    lr=10., init_particles=init_particles[:n_particles],
    kernel_length_scale=kernel_length_scale,
    # is_log_density_batched=True,
    kernel_diag_infl=1e-8,
    bounds=(-1000, 1000),
    # gradient_decay=gradient_decay,
    keep_all=False,
    compile_step=False,
    verbose=True
)

# %%
# n_particles_msip = 500
# kernel_length_scale = 1e-2
msip_quadgf = MSIPQuadGradientFree(
    post_log_dens_batch, partial(mc_quad_rule, N_quad=100)
)

trajectories_msip_qgf, traj_wts_msip_qgf = msip(
    msip_quadgf, n_particles, n_steps, dim=2,
    lr=1., init_particles=init_particles[:n_particles],
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=1e-8,
    bounds=(-1000., 1000.),
    keep_all=False, compile_step=False,
    verbose=True
)


# %%
# pts_eks = trajectories_eks[-1]
#pts_kfr = particles_kfr
pts_galdi = trajectories_galdi[-1]
# pts_gfaldi = trajectories_gfaldi[-1]
pts_cbs = trajectories_cbs[-1]
idx_msip = -1
pts_msip = trajectories_msip[idx_msip]
wts_msip = traj_wts_msip[idx_msip]

pts_msip_adapt = trajectories_msip_adapt[idx_msip]
pts_msip_qg = trajectories_msip_qg[-1]
wts_msip_qg = traj_wts_msip_qg[-1]
wts_msip_qg = wts_msip_qg/wts_msip_qg.sum()
pts_msip_qgf = trajectories_msip_qgf[-1]
wts_msip_qgf = traj_wts_msip_qgf[-1]

Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)

which_plot = [(pts_msip, "MSIP"), (pts_msip_adapt, "MSIP Adapt"), (pts_galdi, "Grad-ALDI")]
fig, axs = plt.subplots(1, len(which_plot), sharex=True, sharey=True, figsize=(len(which_plot)*4, 4))
for (ax,(pts, label)) in zip(axs,which_plot):
    ax.contour(X, Y, post_log_dens(grid_pts).reshape(Ngrid, Ngrid), levels=20)
    ax.set_title(label)
    s = ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5)
    ax.set_aspect(1.0)
fig.tight_layout()
plt.show()

# %%
