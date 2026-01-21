# %%
import torch
from torch import Tensor
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi
import nak_torch
import matplotlib.pyplot as plt
from jaxtyping import Float

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
torch.manual_seed(1023921)
obs_op = torch.randn(2, 5)
forward_model = torch.compile(lambda particles: particles @ obs_op)
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])

model = nak_torch.GaussianModel(
    forward_model, likelihood_precision = 0.53,
    prior_precision= 0.12, true_obs=true_obs,
    is_vectorized=True
)

# %%
n_steps, n_particles = 1000, 500
lr = 0.1
init_particles = torch.randn((n_particles, 2))
trajectories_eks,_ = eks(
    model, n_particles=n_particles,
    n_steps=n_steps, dim=2, lr = lr,
    init_particles=init_particles, keep_all=False,
)

# %%
def post_log_dens(pts):
    ll_term = model.likelihood_precision * torch.linalg.norm(pts @ obs_op - model.true_obs, dim=1)**2
    prior_term = model.prior_precision * torch.linalg.norm(pts, dim=1)**2
    return -0.5 * (ll_term + prior_term)

# %%
trajectories_galdi,_ = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    is_density_vectorized=True,
    keep_all=False
)

# %%
trajectories_gfaldi,_ = gradfree_aldi(
    model, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True
)

# %%
pts_eks = trajectories_eks[-1]
pts_galdi = trajectories_galdi[-1]
pts_gfaldi = trajectories_gfaldi[-1]
mean_pr, cov_pr = torch.zeros(2), torch.eye(2) / model.prior_precision
mean_li, cov_li = model.true_obs, torch.eye(len(model.true_obs)) / model.likelihood_precision

mean_post, cov_post = make_gaussian_post(obs_op, mean_pr, cov_pr, mean_li, cov_li)
vals, vecs  = torch.linalg.eigh(cov_post)
cov_post_sqrt = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T
samps = torch.randn(10000, 2) @ cov_post_sqrt + mean_post

# %%
fig,ax=plt.subplots()
Ngrid = 100
xgrid = torch.linspace(min(samps[:,0].min(), -5), max(samps[:,0].max(), 5), Ngrid)
ygrid = torch.linspace(min(samps[:,1].min(), -5), max(samps[:,1].max(), 5), Ngrid)
X,Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
ax.contour(X, Y, post_log_dens(grid_pts).reshape(Ngrid, Ngrid), levels=10)
ax.scatter(samps[:,0], samps[:,1], alpha=0.025, label="Truth")
ax.scatter(pts_galdi[:,0], pts_galdi[:,1], alpha=0.2, label="Grad-ALDI")
ax.scatter(pts_gfaldi[:,0], pts_gfaldi[:,1], alpha=0.2, label="GradFree-ALDI")
ax.scatter(pts_eks[:,0], pts_eks[:,1], alpha=0.1, label="EKS")
ax.set_aspect(1.0)
ax.legend()
plt.show()

# %%
print("Covariances---\nTruth:\n{},\nEKS:\n{}\nGrad-ALDI:\n{}\nGradFree-ALDI:\n{}".format(
    cov_post, pts_eks.T.cov(), pts_galdi.T.cov(), pts_gfaldi.T.cov()
))

# %%
