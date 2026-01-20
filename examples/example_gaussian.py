# %%
import math
import torch
from torch import Tensor
from nak_torch.functions import aristoff_bangerth as ab, build_aristoff_bangerth
from nak_torch.algorithms import msip, svgd, eks
from matplotlib import ticker
import gc
import matplotlib.pyplot as plt
from jaxtyping import Float
from nak_torch.tools.kernel import sqexp_kernel_matrix
from tqdm import tqdm
import pandas as pd

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

# %%
def make_gaussian_post(
    forward_op: Float[Tensor, "obs dim"],
    mean_pr: Float[Tensor, " dim"],
    cov_pr: Float[Tensor, "dim dim"],
    mean_li: Float[Tensor, " obs"],
    cov_li: Float[Tensor, "obs obs"]
):
    forward_op = forward_op.T
    # inner_op = forward_op @ cov_pr @ forward_op.T + cov_li
    # kalman_mat = cov_pr @ forward_op.T @ torch.linalg.solve(inner_op, forward_op @ cov_pr)
    cov_post = torch.linalg.inv(forward_op.T @ torch.linalg.solve(cov_li, forward_op) + torch.linalg.inv(cov_pr))
    mean_post = cov_post @ (forward_op.T @ torch.linalg.solve(cov_li, mean_li) + torch.linalg.solve(cov_pr, mean_pr))
    return mean_post, cov_post

# %%
obs_op = torch.randn(2, 5)
forward_model = torch.compile(lambda particles: particles @ obs_op)
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
eks_model = eks.EKSModel(
    forward_model, likelihood_precision = 0.1,
    prior_precision= 0.01 * torch.eye(2), true_obs=true_obs,
    is_vectorized=True
)
# log_th = torch.randn(500, 64, requires_grad=True, dtype=torch.float64)
# test_out = log_p(log_th)
# test_eval = torch.autograd.grad(test_out.sum(), log_th)

# %%
trajectories,_ = eks.eks(
    eks_model, n_particles=1000, n_steps=1000, dim=2, lr = 0.01
)

# %%
def post_log_dens(pts):
    ll_term = eks_model.likelihood_precision * torch.linalg.norm(pts @ obs_op - eks_model.true_obs, dim=1)**2
    prior_term = torch.einsum("bi,ij,bj->b", pts, eks_model.prior_precision, pts)
    return -0.5 * (ll_term + prior_term)

mean_pr, cov_pr = torch.zeros(2), torch.linalg.inv(eks_model.prior_precision)
mean_li, cov_li = eks_model.true_obs, torch.eye(len(eks_model.true_obs)) / eks_model.likelihood_precision

mean_post, cov_post = make_gaussian_post(obs_op, mean_pr, cov_pr, mean_li, cov_li)
vals, vecs  = torch.linalg.eigh(cov_post)
cov_post_sqrt = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T
samps = torch.randn(10000, 2) @ cov_post_sqrt + mean_post

# %%
fig,ax=plt.subplots()
pts = trajectories[-1] @ torch.sqrt(torch.linalg.inv(eks_model.prior_precision))
Ngrid = 100
xgrid = torch.linspace(min(samps[:,0].min(), -5), max(samps[:,0].max(), 5), Ngrid)
ygrid = torch.linspace(min(samps[:,1].min(), -5), max(samps[:,1].max(), 5), Ngrid)
X,Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
# ax.contour(X, Y, post_log_dens(grid_pts).reshape(Ngrid, Ngrid), levels=10)
# ax.scatter(samps[:,0], samps[:,1], alpha=0.05)
ax.scatter(pts[:,0], pts[:,1], alpha=0.15)
ax.set_aspect(1.0)
plt.show()

# %%
cov_post, pts.T.cov()

# %%
