# %%
import torch
from torch import Tensor
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip
import nak_torch
import matplotlib.pyplot as plt
from jaxtyping import Float
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from nak_torch.tools.kernel import sqexp_kernel_matrix

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)

# %%


def make_gaussian_post(
    forward_op: Float[Tensor, "obs dim"],
    mean_pr: Float[Tensor, " dim"],
    cov_pr: Float[Tensor, "dim dim"],
    mean_li: Float[Tensor, " obs"],
    cov_li: Float[Tensor, "obs obs"]
):
    forward_op = forward_op.T
    cov_post = torch.linalg.inv(
        forward_op.T @ torch.linalg.solve(cov_li,
                                          forward_op) + torch.linalg.inv(cov_pr)
    )
    mean_post = cov_post @ (forward_op.T @ torch.linalg.solve(cov_li,
                            mean_li) + torch.linalg.solve(cov_pr, mean_pr))
    return mean_post, cov_post

def weighted_cov(pts: Tensor, wts: Tensor):
    mean = wts @ pts
    second_moment = torch.einsum("b,bi,bj", wts, pts, pts)
    return second_moment - mean.outer(mean)

# %%
def kernel_optimal_weight_factory(
        pts: Tensor, log_dens_evals: Tensor,
        kernel_length_scale: float,
        kernel_diag_infl: float = 1e-6
) -> Tensor:
    n_particles = pts.shape[0]
    Kmat = sqexp_kernel_matrix(pts, kernel_length_scale)
    Kmat[
        torch.arange(n_particles), torch.arange(n_particles)
    ] += kernel_diag_infl
    wts = torch.linalg.solve(Kmat, log_dens_evals.exp())
    return wts


# %%
torch.manual_seed(1023921)
obs_op = torch.randn(2, 5)
forward_model = torch.compile(lambda particles: particles @ obs_op)
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])

model = nak_torch.GaussianModel(
    forward_model, likelihood_precision=0.53,
    prior_precision=0.12, true_obs=true_obs,
    is_vectorized=True
)


@torch.compile
def post_log_dens(pt):
    ll_term = model.likelihood_precision * \
        torch.linalg.norm(pt @ obs_op - model.true_obs, dim=-1)**2
    prior_term = model.prior_precision * torch.linalg.norm(pt, dim=-1)**2
    return -0.5 * (ll_term + prior_term).squeeze()


post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# %%
mean_pr, cov_pr = torch.zeros(2), torch.eye(2) / model.prior_precision
mean_li, cov_li = model.true_obs, torch.eye(
    len(model.true_obs)
) / model.likelihood_precision

mean_post, cov_post = make_gaussian_post(
    obs_op, mean_pr, cov_pr, mean_li, cov_li
)
vals, vecs = torch.linalg.eigh(cov_post)
cov_post_sqrt = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T
samps = torch.randn(10000, 2) @ cov_post_sqrt + mean_post

# %%
n_steps, n_particles = 1000, 500
lr = 0.1
init_particles = torch.randn((n_particles, 2))

# %%
trajectories_eks = eks(
    model, n_particles=n_particles,
    n_steps=n_steps, dim=2, lr=lr,
    init_particles=init_particles, keep_all=False,
)

# %%
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=False
)

# %%
trajectories_gfaldi = gradfree_aldi(
    model, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True
)

# %%
trajectories_cbs = cbs(
    post_log_dens, n_particles, n_steps, inverse_temp=0.95, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True
)

# %%
kernel_length_scale = 0.10
bounds = (-10., 10.)
gradient_decay = 1.0
n_particles_msip = 500
n_steps_msip = 100
lr_msip = 0.5
kernel_diag_infl = 1e-6
msip_fredholm = MSIPFredholm(
    kernel_length_scale**2, gradient_decay,
    post_log_dens_grad_val_batch
)

# %%
trajectories_msip, traj_wts_msip = msip(
    msip_fredholm, n_particles_msip, n_steps_msip, dim=2,
    lr=lr_msip, init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=False
)

# %%


def mc_quad_rule(batch_size: int, N_quad: int = 300, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim)).mul_(kernel_length_scale)
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch, mc_quad_rule,
    kernel_length_scale**2, gradient_decay
)

# %%
trajectories_msip_qg, traj_log_wts_msip_qg = msip(
    msip_quadgrad, n_particles_msip, n_steps_msip, dim=2,
    lr=1e-2, init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=False
)

# %%
from functools import partial
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
def spherical_quad(batch_size: int , N_spherical: int = 100, N_radial: int = 3):
    pts, wts = spherical_MC_radial_Laguerre((batch_size,), N_spherical, 2, N_radial)
    pts.mul_(kernel_length_scale)
    return pts, wts

msip_quadgf = MSIPQuadGradientFree(
    post_log_dens_batch, spherical_quad
)

# %%
trajectories_msip_qgf, traj_log_wts_msip_qgf = msip(
    msip_quadgf, n_particles_msip, n_steps_msip, dim=2,
    lr=1e-2, init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=False
)

# %%
pts_eks = trajectories_eks[-1]
pts_galdi = trajectories_galdi[-1]
pts_gfaldi = trajectories_gfaldi[-1]
pts_cbs = trajectories_cbs[-1]
pts_msip = trajectories_msip[-1]
wts_msip = kernel_optimal_weight_factory(
    pts_msip,
    post_log_dens_batch(pts_msip),
    kernel_length_scale
)
wts_msip /= wts_msip.sum()
pts_msip_qg = trajectories_msip_qg[-1]
log_v0_qg,_ = msip_quadgrad.get_v_evals(pts_msip_qg)
log_v0_qg.sub_(log_v0_qg.max())
wts_msip_qg = kernel_optimal_weight_factory(
    pts_msip_qg,
    log_v0_qg,
    kernel_length_scale
)
wts_msip_qg /= wts_msip_qg.sum()
pts_msip_qgf = trajectories_msip_qgf[-1]
log_v0_qgf,_ = msip_quadgf.get_v_evals(pts_msip_qgf)
log_v0_qgf.sub_(log_v0_qgf.max())
wts_msip_qgf = kernel_optimal_weight_factory(
    pts_msip_qgf,
    log_v0_qgf,
    kernel_length_scale
)
wts_msip_qgf /= wts_msip_qgf.sum()

Ngrid = 100
xgrid = torch.linspace(
    min(samps[:, 0].min(), -5), max(samps[:, 0].max(), 5), Ngrid
)
ygrid = torch.linspace(
    min(samps[:, 1].min(), -5), max(samps[:, 1].max(), 5), Ngrid
)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)

fig, ax = plt.subplots()
ax.contour(X, Y, post_log_dens(grid_pts).reshape(Ngrid, Ngrid), levels=10)
# ax.scatter(samps[:, 0], samps[:, 1], alpha=0.025, label="Truth")
# ax.scatter(pts_galdi[:, 0], pts_galdi[:, 1], alpha=0.2, label="Grad-ALDI")
# ax.scatter(pts_gfaldi[:, 0], pts_gfaldi[:, 1],
#            alpha=0.2, label="GradFree-ALDI")
# ax.scatter(pts_eks[:, 0], pts_eks[:, 1], alpha=0.1, label="EKS")
# ax.scatter(pts_cbs[:, 0], pts_cbs[:, 1], alpha=0.1, label="CBS")
# ax.scatter(pts_msip[:, 0], pts_msip[:, 1], c=wts_msip, alpha=0.15, label="MSIP")
s = ax.scatter(pts_msip_qg[:, 0], pts_msip_qg[:, 1],
               c = wts_msip_qg, alpha=0.15, label="MSIP-QuadGrad")
s = ax.scatter(pts_msip_qgf[:, 0], pts_msip_qgf[:, 1],
               c = wts_msip_qgf, alpha=0.15, label="MSIP-QuadGradFree")
plt.colorbar(s)
ax.set_aspect(1.0)
ax.legend()
plt.show()

# %%
print(f"""
Covariances---
Truth:
{cov_post}
EKS:
{pts_eks.T.cov()}
Grad-ALDI:
{pts_galdi.T.cov()}
GradFree-ALDI:
{pts_gfaldi.T.cov()}
MSIP:
{weighted_cov(pts_msip, wts_msip)}
MSIP-QuadGrad:
{weighted_cov(pts_msip_qg, wts_msip_qg)}
MSIP-QuadGradFree:
{weighted_cov(pts_msip_qgf, wts_msip_qgf)}
""")

# %%
