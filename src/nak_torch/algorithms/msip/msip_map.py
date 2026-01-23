import torch
from typing import Optional, Callable
from nak_torch.tools import recursive_weighted_average_alpha_v
from nak_torch.tools.kernel import sqexp_kernel_matrix
from jaxtyping import Float
from nak_torch.tools.types import BatchLogDensity, BatchQuadratureRule, BatchLogDensityGradVal, BatchQuadrulePtType, KernelMatrixType, MultiBatchLogDensity, MultiBatchLogDensityGradVal, PtType, BatchPtType, BatchType
from abc import ABC, abstractmethod


MSIPEstimatorOutput = tuple[BatchType, BatchPtType]


def calculate_msip_map(
        out: PtType, i: int,
        particles: BatchPtType,
        estimators: MSIPEstimatorOutput,
        K_minus_one: KernelMatrixType
):
    log_v0_evals, v1_div_v0_minus_y = estimators
    alpha_i = K_minus_one[i, :]  # (N,)
    term_v0,_ = recursive_weighted_average_alpha_v(particles, alpha_i, log_v=log_v0_evals)
    term_v1, log_weight = recursive_weighted_average_alpha_v(v1_div_v0_minus_y, alpha_i, log_v=log_v0_evals)
    torch.add(term_v0, term_v1, out=out)
    return log_weight


class MSIPEstimator(ABC):
    @abstractmethod
    def get_v_evals(self, particles: BatchPtType) -> MSIPEstimatorOutput:
        """ Function that returns estimation of $(log(v_0), -y + v_1 / v_0)$ """
        pass


class MSIPFredholm(MSIPEstimator):
    sigma_sq: float
    # gamma > 0, constant decay on the gradient (i.e., multiplies grad)
    gradient_decay: float
    log_dens_grad_val: BatchLogDensityGradVal

    def __init__(
            self, sigma_sq: float,
            gradient_decay: float,
            log_dens_grad_val: BatchLogDensityGradVal
    ):
        self.sigma_sq, self.gradient_decay = sigma_sq, gradient_decay
        self.log_dens_grad_val = log_dens_grad_val

    def get_v_evals(self, particles):
        grads, vals = self.log_dens_grad_val(particles)
        ret_v1_ratio = grads.mul_(self.sigma_sq * self.gradient_decay)
        return vals, ret_v1_ratio


vmap_recursive_weighted_average_alpha_v = torch.vmap(
    recursive_weighted_average_alpha_v, in_dims=(0,0,0)
)


class MSIPQuadGradientFree(MSIPEstimator):
    quadrature: BatchQuadratureRule
    log_dens: MultiBatchLogDensity

    def __init__(
        self,
        log_dens: BatchLogDensity,
        quadrature: BatchQuadratureRule,
    ):
        self.quadrature = quadrature
        self.log_dens = log_dens

    def get_v_evals(self, particles):
        quad_pts, quad_wts = self.quadrature(particles.shape[0])
        particle_quad_pts = (
            particles.unsqueeze(1) + quad_pts
        )  # (N_part, N_quad, dim)
        log_dens_evals = self.log_dens(
            particle_quad_pts.reshape(-1, particles.shape[1])
        ).reshape(particle_quad_pts.shape[:-1])
        v1_ratio, log_v0 = vmap_recursive_weighted_average_alpha_v(
            particle_quad_pts, quad_wts, log_dens_evals
        )

        return log_v0, v1_ratio


class MSIPQuadGradientInformed(MSIPEstimator):
    quadrature: BatchQuadratureRule
    sigma_sq: float
    gradient_decay: float
    log_dens_grad_val: BatchLogDensityGradVal

    def __init__(
        self,
        log_dens_grad_val: BatchLogDensityGradVal,
        quadrature: BatchQuadratureRule,
        sigma_sq: float,
        gradient_decay: float,
    ):
        self.quadrature, self.sigma_sq, self.gradient_decay = quadrature, sigma_sq, gradient_decay
        self.log_dens_grad_val = log_dens_grad_val

    def get_v_evals(self, particles):
        quad_pts, quad_wts = self.quadrature(particles.shape[0])
        particle_quad_pts = (
            particles.unsqueeze(1) + quad_pts
        )  # (N_part, N_quad, dim)
        log_dens_grads, log_dens_evals = self.log_dens_grad_val(
            particle_quad_pts.reshape(-1, particles.shape[1])
        )
        log_dens_grads = log_dens_grads.reshape_as(particle_quad_pts)
        log_dens_evals = log_dens_evals.reshape(particle_quad_pts.shape[:-1])
        v1_integrand = particle_quad_pts.mul_(1 - self.gradient_decay).add_(
            log_dens_grads.mul_(self.gradient_decay * self.sigma_sq)
        )
        v1_ratio, log_v0 = vmap_recursive_weighted_average_alpha_v(
            v1_integrand, quad_wts, log_dens_evals
        )
        return log_v0, v1_ratio


def msip_map(
        estimators: MSIPEstimatorOutput,
        particles: torch.Tensor,
        kernel_length_scale: float,
        *_,
        kernel_diag_infl: float = 0.0,
        output_idx: Optional[int] = None,
        get_kernel_matrix: Callable[
            [torch.Tensor, float], torch.Tensor
        ] = sqexp_kernel_matrix,
        **__
) -> tuple[PtType, Float] | tuple[BatchPtType, BatchType]:
    """
    Compute the full MSIP map T(y) for all particles at once.
    Returns t_arr with shape (N, d).
    """
    n_particles = particles.shape[0]
    particles = particles.clone()

    with torch.no_grad():
        # (N, N)
        kernel_matrix = get_kernel_matrix(particles, kernel_length_scale)

        if kernel_diag_infl > 0:
            kernel_matrix[
                torch.arange(n_particles), torch.arange(n_particles)
            ] += kernel_diag_infl

        if kernel_diag_infl > 0.:  # We know it's invertible if diag is inflated
            K_inverse = torch.linalg.inv(kernel_matrix)
        else:
            K_inverse = torch.linalg.pinv(kernel_matrix)
        N, d = particles.shape

        if output_idx is None:
            t_arr = torch.empty_like(particles)
            log_wt = torch.empty(particles.shape[0], device=t_arr.device, dtype=t_arr.dtype)
            for idx in range(N):
                log_wt[idx] = calculate_msip_map(
                    t_arr[idx], idx,
                    particles,
                    estimators,
                    K_inverse
                )
        else:
            t_arr = torch.empty_like(particles[0])
            log_wt = calculate_msip_map(
                t_arr, output_idx,
                particles,
                estimators,
                K_inverse
            )

    return t_arr, log_wt
