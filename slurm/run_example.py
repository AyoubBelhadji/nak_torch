from dataclasses import dataclass
import math

from numpy.typing import ArrayLike
import torch
from torch import Tensor
from nak_torch.tools.types import BatchLogDensity, BatchType, BatchPtType, GaussianModel, KernelFunction, MatSelfKernelFunction, PtType
from nak_torch.algorithms.msip import estimators, MSIPFredholm, MSIPQuadGradientFree, MSIPQuadGradientInformed
from nak_torch.tools.kernel import stein_kernel_mat_factory
from . import problems
from typing import Any, Optional
from jaxtyping import Float
import nak_torch
import yaml

MeanType = Float[ArrayLike, " dim"]
CovType = Float[ArrayLike, "dim dim"]


@dataclass
class TestConfiguration:
    algorithm: str
    density: str
    prior: str
    lr: float
    n_particles: int
    dim: int
    T_steps: int
    kernel_length_scale: float
    device: Optional[str] = None
    msip_estimator: Optional[str] = None
    kernel_diag_infl: Optional[float] = None
    inner_quad: Optional[str] = None
    inner_quad_kwargs: dict[str, Any] = {}
    bounds: Optional[tuple[float, float]] = None
    test_length_scale: Optional[float] = None
    test_kernel: Optional[str] = None
    run_seed: Optional[int] = None


def is_kernel_alg(algorithm: str):
    return algorithm in ["msip", "svgd"]


alg_dict = nak_torch.algorithms.__dict__
inner_quad_dict = nak_torch.tools.quadrature.__dict__
msip_estimators = estimators.__all__
problem_dict: dict[str, tuple[GaussianModel,
                              BatchPtType | None]] = problems.__dict__
uses_gaussian_model = ["gradfree_aldi", "eks"]


def check_msip_config_valid(config: TestConfiguration):
    infl = config.kernel_diag_infl

    if infl is None:
        raise ValueError("Missing required field: kernel_diag_infl")
    if infl < 0:
        raise ValueError(f"Invalid kernel_diag_infl {infl}")

    est = config.msip_estimator
    if est is None:
        raise ValueError("Missing required field: msip_estimator")
    if est not in msip_estimators:
        raise ValueError(f"Invalid MSIP estimator {est}")
    if est != "MSIPFredholm":
        quad = config.inner_quad
        if quad is None:
            raise ValueError("Missing inner_quad")
        if quad not in inner_quad_dict:
            raise ValueError(f"Invalid inner_quad value {quad}")
        inner_quad_dict[quad](
            2, **config.inner_quad_kwargs)  # check quadrature


def check_config_valid(config: TestConfiguration):
    alg = config.algorithm
    if alg not in alg_dict:
        raise ValueError(f"Missing algorithm {alg}")
    if alg == 'msip':
        check_msip_config_valid(config)


def msip_estimator_factory(
    log_dens: BatchLogDensity,
    msip_estimator: str, inner_quad: str,
    gradient_decay: float,
    **inner_quad_kwargs
) -> estimators.MSIPEstimator:

    def grad_val_dens(pts: Tensor) -> tuple[BatchPtType, BatchType]:
        p = pts.clone().requires_grad_(True)
        out = log_dens(p)
        return torch.autograd.grad(out.sum(), p)[0], out.detach()
    if msip_estimator.lower() == 'MSIPFredholm':
        return MSIPFredholm(gradient_decay, grad_val_dens)
    else:
        quad_fcn = inner_quad_dict[inner_quad]

        def quad(batch_sz: int):
            return quad_fcn(batch_sz, **inner_quad_kwargs)
        if msip_estimator == "MSIPQuadGradientFree":
            return MSIPQuadGradientFree(log_dens, quad)
        elif msip_estimator == "MSIPQuadGradientInformed":
            return MSIPQuadGradientInformed(grad_val_dens, quad, gradient_decay)
        else:
            raise ValueError(f"Unexpected estimator name: {msip_estimator}")


def initialize_particles(
    prior: str,
    n_particles: int,
    dim: int,
    rng: torch.Generator,
    device: Optional[str]
):
    if prior.lower().startswith("normal"):
        proc = prior[7:-1].split(",")
        if len(proc) != 2:
            raise ValueError(
                "Currently only accepts prior normal(mu,sigma**2)"
            )
        mu_s, sigma_sq_s = proc
        mu, sigma_sq = float(mu_s), float(sigma_sq_s)
        return torch.normal(mean=mu, std=math.sqrt(sigma_sq), size=(n_particles, dim), generator=rng, device=device)
    else:
        raise ValueError("Currently only accepts prior normal(mu,sigma**2)")


def configuration_factory(fname: str) -> TestConfiguration:
    with open(fname, "r") as f:
        config_dict: dict = yaml.safe_load(f)
    config = TestConfiguration(**config_dict)
    check_config_valid(config)
    if config.test_length_scale is None:
        config.test_length_scale = config.kernel_length_scale
    if config.run_seed is None:
        config.run_seed = torch.default_generator.initial_seed()
    return config


def run_config(config: TestConfiguration) -> tuple[BatchPtType, BatchType | None]:
    alg_name = config.algorithm.lower()
    alg = alg_dict[alg_name]
    gauss_model, _ = problem_dict[config.density]
    prior = config.prior
    seed = config.run_seed
    if seed is None:
        raise ValueError("Invalid seed.")
    torch.set_default_device(config.device)
    rng = torch.Generator(config.device)
    rng = rng.manual_seed(seed)
    init_particles = initialize_particles(
        prior, config.n_particles, config.dim, rng, config.device
    )
    log_dens = nak_torch.tools.types.gaussian_log_dens_factory(gauss_model)
    kwargs = {
        'init_particles': init_particles,
        'keep_all': False,
        'rng': rng,
        'is_log_density_batched': True,
        **config.__dict__
    }
    if alg_name == 'msip':
        pts, wts = alg(gauss_model, **kwargs)
        wts = wts[-1] / wts[-1].sum()
        return pts[-1], wts
    else:  # Not weighted
        problem_input = gauss_model if alg_name in uses_gaussian_model else log_dens
        pts = alg(problem_input, **kwargs)
        return pts[-1], None


@dataclass
class TestOutput:
    points: Float[ArrayLike, "batch dim"]
    weights: Float[ArrayLike, " batch"] | None
    ksd: float
    mean: Float[ArrayLike, " dim"]
    cov: Float[ArrayLike, "dim dim"]
    sample_mmd: float | None


def get_stein_mat_fcn(log_dens: BatchLogDensity, kernel_elem: KernelFunction):
    def grad_log_dens(p: BatchPtType):
        p_ = p.clone().requires_grad_(True)
        out = log_dens(p_)
        return torch.autograd.grad(out, p_)[0]
    return stein_kernel_mat_factory(grad_log_dens, kernel_elem, is_grad_vectorized=True)


def get_moments(
    pts: BatchPtType, wts: Optional[BatchType]
) -> tuple[MeanType, CovType]:

    if wts is None:
        return pts.mean(0).cpu().numpy(), pts.T.cov().cpu().numpy()
    else:
        mean = wts @ pts
        cov = torch.einsum("b,bi,bj", wts, pts, pts)
        cov.div_(1 - torch.square(wts).sum())
        return mean.cpu().numpy(), cov.cpu().numpy()


def get_ksd(
    log_dens: BatchLogDensity,
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float
) -> float:
    stein_mat = get_stein_mat_fcn(log_dens, kernel_elem)
    stein_mat_eval = stein_mat(pts, test_kernel_length_scale, pts)
    ksd: Float
    if wts is None:
        ksd = torch.sqrt(stein_mat_eval.sum() / (pts.shape[0]**2))
    else:
        ksd = torch.sqrt((stein_mat_eval @ wts) @ wts)
    return ksd.item()


def get_mmd(
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float,
    ref_samples: Optional[BatchPtType],
    chunk_size: int = 1000
) -> Optional[float]:
    kernel_mat = nak_torch.tools.kernel.matricize_kernel_elem(kernel_elem)
    if ref_samples is None:
        return None
    N_ref = ref_samples.shape[0]
    N_chunks = (N_ref + chunk_size - 1) // chunk_size
    ref_mmd = get_self_mmd(
        test_kernel_length_scale, ref_samples,
        chunk_size, kernel_mat, N_ref, N_chunks
    )
    K_out = torch.empty(pts.shape[0])
    cross_mmd = torch.zeros(())
    for chunk_idx in range(N_chunks):
        min_idx = chunk_idx * chunk_size
        max_idx = min((chunk_idx + 1)*chunk_size, N_ref)
        samples_chunk = ref_samples[min_idx:max_idx]
        K_mat = kernel_mat(pts, test_kernel_length_scale, samples_chunk)
        if wts is None:
            cross_mmd += K_out.mean()
        else:
            K_out = torch.mean(K_mat, dim=1, out=K_out)
            cross_mmd += wts @ K_out
    pts_mmd: Float
    if wts is None:
        pts_mmd = kernel_mat(pts, test_kernel_length_scale).mean()
    else:
        K_mat = kernel_mat(pts, test_kernel_length_scale).mean()
        pts_mmd = (K_mat @ wts) @ wts
    return torch.sqrt(ref_mmd - 2*cross_mmd + pts_mmd).item()


def get_self_mmd(
    test_kernel_length_scale: float,
    ref_samples: BatchPtType, chunk_size: int,
    kernel_mat: MatSelfKernelFunction, N_ref: int,
    N_chunks: int
) -> Float:
    reference_offset = torch.zeros(())
    for chunk_idx in range(N_chunks):
        min_idx = chunk_idx * chunk_size
        max_idx = min((chunk_idx + 1)*chunk_size, N_ref)
        samples_chunk = ref_samples[min_idx:max_idx]
        K_mat = kernel_mat(
            samples_chunk, test_kernel_length_scale, samples_chunk
        )
        reference_offset += K_mat.mean()
        for comp_chunk_idx in range(chunk_idx+1, N_chunks):
            min_idx_comp = comp_chunk_idx * chunk_size
            max_idx_comp = min((comp_chunk_idx + 1)*chunk_size, N_ref)
            samples_comp = ref_samples[min_idx_comp:max_idx_comp]
            K_mat = kernel_mat(
                samples_chunk, test_kernel_length_scale, samples_comp)
            reference_offset += 2 * K_mat.mean()
    return reference_offset

def process_output(
    log_dens: BatchLogDensity,
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float,
    ref_samples: Optional[BatchPtType]
):
    ksd = get_ksd(log_dens, pts, wts, kernel_elem, test_kernel_length_scale)
    mmd = get_mmd(pts, wts, kernel_elem, test_kernel_length_scale, ref_samples)
    mean, cov = get_moments(pts, wts)
    pts_np = pts.cpu().numpy()
    if wts is not None:
        wts_np = wts.cpu().numpy()
    else:
        wts_np = None
    return TestOutput(pts_np, wts_np, ksd, mean, cov, mmd)
