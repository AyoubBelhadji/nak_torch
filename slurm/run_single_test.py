from dataclasses import dataclass, field
import math

from numpy.typing import ArrayLike
import torch
from torch import Tensor
from nak_torch.tools.types import BatchLogDensity, BatchType, BatchPtType, GaussianModel, KernelFunction, MatSelfKernelFunction
from nak_torch.algorithms.msip import estimators, MSIPFredholm, MSIPQuadGradientFree, MSIPQuadGradientInformed
from nak_torch.tools.kernel import stein_kernel_mat_factory
import problems
import pickle
import datetime
from typing import Any, Callable, Optional
from jaxtyping import Float
import nak_torch

MeanType = Float[ArrayLike, " dim"]
CovType = Float[ArrayLike, "dim dim"]


@dataclass
class TestConfiguration:
    algorithm: str
    problem: str
    prior: str
    lr: float
    n_particles: int
    dim: int
    n_steps: int
    kernel_length_scale: float
    device: Optional[str] = None
    msip_estimator: Optional[str] = None
    kernel_diag_infl: Optional[float] = None
    inner_quad: Optional[str] = None
    inner_quad_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    bounds: Optional[tuple[float, float]] = None
    test_length_scale: Optional[float] = None
    test_kernel: Optional[str] = None
    run_seed: Optional[int] = None
    gradient_decay: Optional[float] = None
    alg_kwargs: dict[str, Any] = field(default_factory=lambda: {})


alg_dict = nak_torch.algorithms.__dict__
inner_quad_dict = nak_torch.tools.quadrature.__dict__
msip_estimators = estimators.__all__
problem_dict: dict[str, Callable[[],problems.Problem]] = problems.__dict__
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
    problem = config.problem
    if problem != 'test' and problem not in problem_dict:
        raise ValueError(f"Unrecognized problem {problem}")
    if alg == 'msip':
        check_msip_config_valid(config)
    if alg == 'cbs':
        if 'inverse_temp' not in config.alg_kwargs:
            raise ValueError("Missing algorithm kwarg 'inverse_temp'")


def msip_estimator_factory(
    log_dens: BatchLogDensity,
    msip_estimator: Optional[str],
    inner_quad: Optional[str],
    gradient_decay: Optional[float],
    **inner_quad_kwargs
) -> estimators.MSIPEstimator:
    if msip_estimator is None:
        raise ValueError("Expected value for msip_estimator")
    def grad_val_dens(pts: Tensor) -> tuple[BatchPtType, BatchType]:
        p = pts.clone().requires_grad_(True)
        out = log_dens(p)
        return torch.autograd.grad(out.sum(), p)[0], out.detach()
    if msip_estimator.lower() == 'MSIPFredholm':
        if gradient_decay is None:
            raise ValueError(f"Estimator {msip_estimator} expects value gradient_decay")
        return MSIPFredholm(gradient_decay, grad_val_dens)
    else:
        if inner_quad is None:
            raise ValueError(f"Estimator {msip_estimator} expects value inner_quad")
        quad_fcn = inner_quad_dict[inner_quad]

        def quad(batch_sz: int):
            return quad_fcn(batch_sz, **inner_quad_kwargs)

        if msip_estimator == "MSIPQuadGradientFree":
            return MSIPQuadGradientFree(log_dens, quad)
        elif msip_estimator == "MSIPQuadGradientInformed":
            if gradient_decay is None:
                raise ValueError(f"Estimator {msip_estimator} expects value gradient_decay")
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
        proc = prior[7:-1].split(";")
        if len(proc) != 2:
            raise ValueError(
                "Currently only accepts prior normal(mu;sigma**2)"
            )
        mu_s, sigma_sq_s = proc
        mu, sigma_sq = float(mu_s), float(sigma_sq_s)
        return torch.normal(mean=mu, std=math.sqrt(sigma_sq), size=(n_particles, dim), generator=rng, device=device)
    else:
        raise ValueError("Currently only accepts prior normal(mu;sigma**2)")


def configuration_factory(config_dict: dict) -> TestConfiguration:
    alg_kwargs = {}
    for (key, value) in config_dict.items():
        if key not in TestConfiguration.__annotations__.keys():
            alg_kwargs[key] = value
    for key in alg_kwargs.keys():
        config_dict.pop(key)
    config = TestConfiguration(**config_dict, alg_kwargs=alg_kwargs)
    check_config_valid(config)
    if config.test_length_scale is None:
        config.test_length_scale = config.kernel_length_scale
    if config.run_seed is None:
        config.run_seed = torch.default_generator.initial_seed()
    return config


def run_config(config: TestConfiguration) -> tuple[BatchLogDensity, BatchPtType, BatchType | None]:
    alg_name = config.algorithm.lower()
    alg = alg_dict[alg_name]
    problem = problem_dict[config.problem + "_logpdf"]()
    model = problem.model
    if not isinstance(model, GaussianModel) and alg_name in uses_gaussian_model:
        raise ValueError(f"Invalid problem {config.problem} for algorithm {alg_name}")
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
    if isinstance(model, GaussianModel):
        log_dens = nak_torch.tools.types.gaussian_log_dens_factory(model)
    else:
        log_dens = model

    kwargs = {
        'init_particles': init_particles,
        'keep_all': False,
        'rng': rng,
        'is_log_density_batched': True,
        **config.__dict__,
        **config.alg_kwargs
    }
    if alg_name == 'msip':
        estimator = msip_estimator_factory(log_dens, config.msip_estimator, config.inner_quad, config.gradient_decay, **config.inner_quad_kwargs)
        pts, wts = alg(estimator, **kwargs)
        wts = wts[-1] / wts[-1].sum()
        return log_dens, pts[-1], wts
    else:  # Not weighted
        problem_input = model if alg_name in uses_gaussian_model else log_dens
        pts = alg(problem_input, **kwargs)
        return log_dens, pts[-1], None


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
        return torch.autograd.grad(out.sum(), p_)[0]
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
    print("Getting MMD...")
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

all_kernels = nak_torch.tools.kernel.kernel_elem_dict

def get_kernel_elem(test_kernel: str) -> KernelFunction:
    return all_kernels[test_kernel]

def run_single_test(configuration: dict):
    config = configuration_factory(configuration)
    assert config.test_length_scale is not None and config.test_kernel is not None
    if config.problem == "test":
        print(config)
        return
    log_dens, pts, wts = run_config(config)
    kernel_elem = get_kernel_elem(config.test_kernel)
    out = process_output(log_dens, pts, wts, kernel_elem, config.test_length_scale, None)
    experiment_run = {"config": config, "output": out}
    timestamp = datetime.datetime.now()
    save_name = f"data/{config.algorithm}_{timestamp}.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(experiment_run, f)