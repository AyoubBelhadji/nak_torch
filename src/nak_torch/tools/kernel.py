import torch
from typing import Optional, Callable
from jaxtyping import Float
from torch import Tensor
from .types import KernelFunction, MatSelfKernelFunction, PtType, BatchPtType, KernelMatrixType, GradLogDensity, BatchGradLogDensity

__all__ = [
    "sqexp_kernel_matrix",
    "sqexp_kernel_elem",
    "matricize_kernel_elem",
    "stein_kernel_mat_factory"
]

def sqexp_kernel_matrix(
        particles_leaf: BatchPtType,
        kernel_bandwidth: Float,
        p_leaf2: Optional[BatchPtType] = None
):
    if p_leaf2 is None:
        p_leaf2 = particles_leaf
    diff = particles_leaf.unsqueeze(1) - p_leaf2.unsqueeze(0)  # (N, N, d)

    return torch.exp(- (diff ** 2).sum(dim=-1) / (2 * kernel_bandwidth * kernel_bandwidth))


def sqexp_kernel_elem(x: Float[Tensor, " d"], y: Float[Tensor, " d"], kernel_bandwidth: float = 1.0) -> Float:
    torch._assert(x.shape == y.shape and y.ndim == 1, "Invalid input dimensions of x and y")
    ret = torch.exp(- (x - y).square_().sum() / (2 * kernel_bandwidth * kernel_bandwidth))
    return ret

def matricize_kernel_elem(kernel: KernelFunction, use_compiled: bool = True) -> MatSelfKernelFunction:
    r"""
    Vectorize elementwise kernel(pt1, pt2, length_scale) and return a function kernel_mat(pt, length_scale[, pt2=pt])
    """
    kernel_v = torch.vmap(
        torch.vmap(kernel, in_dims = (0, None, None), out_dims = 0),
        in_dims = (None, 0, None), out_dims=1
    )
    def kernel_self_v(pts, bandwidth, pts2 = None):
        return kernel_v(pts, pts if pts2 is None else pts2, bandwidth)
    return torch.compile(kernel_self_v) if use_compiled else kernel_self_v


# def build_kernel_diffs_for_stein_kernel(
#         kernel_fcn: KernelFunction,
# ) -> Callable[
#     [PtType, PtType, Float], tuple[Float, PtType, PtType, Float]
# ]:
#     kernel_grad = torch.func.grad_and_value(kernel_fcn, argnums=(0, 1))

#     def process_kernel_grad(x, y, bandwidth):
#         (grad1, grad2), aux = kernel_grad(x, y, bandwidth)
#         return grad1, (grad1, grad2, aux)

#     kernel_jac = torch.func.jacrev(process_kernel_grad, has_aux=True, argnums=1)

#     def process_kernel_jac(x, y, bandwidth):
#         jac, (grad1, grad2, ev) = kernel_jac(x, y, bandwidth)
#         return torch.trace(jac), grad1, grad2, ev

#     return process_kernel_jac

# def stein_kernel_mat_factory(
#         grad_log_p: GradLogDensity | BatchGradLogDensity,
#         kernel_fcn: KernelFunction,
#         is_grad_vectorized: bool = False,
#         use_compiled: bool = True
# ) -> MatSelfKernelFunction:
#     """
#     Build a stein kernel that works on two points.

#     Use `matricize_kernel_elem` for applying to sets of points

#     :param grad_log_p: Gradient of target log density. If only works on batch of inputs, use `is_grad_vectorized`
#     :type grad_log_p: GradLogDensity | BatchGradLogDensity
#     :param kernel_fcn: kernel(x,y,length_scale)->Float
#     :type kernel_fcn: KernelFunction
#     :param is_grad_vectorized: Whether `grad_log_p` only works on batches of vectors
#     :type is_grad_vectorized: bool
#     :return: stein_kernel(x, y, length_scale)->Float
#     :rtype: KernelFunction
#     """
#     kernel_diffs = build_kernel_diffs_for_stein_kernel(kernel_fcn)

#     def stein_kernel_elem(p1: PtType, grad_p1: PtType, p2: PtType, grad_p2: PtType, length_scale: Float) -> Float:
#         diffs_eval = kernel_diffs(p1, p2, length_scale)
#         trace_kernel, grad1_kernel, grad2_kernel, eval_kernel = diffs_eval
#         grad_term1 = grad1_kernel.dot(grad_p2)
#         grad_term2 = grad2_kernel.dot(grad_p1)
#         return trace_kernel + grad_term1 + grad_term2 + eval_kernel

#     if is_grad_vectorized:
#         kernel_v = torch.vmap(
#             torch.vmap(stein_kernel_elem, in_dims = (0, 0, None, None, None), out_dims = 0),
#             in_dims = (None, None, 0, 0, None), out_dims=1
#         )
#         def stein_kernel_mat(p1: BatchPtType, length_scale: Float, p2: Optional[BatchPtType] = None) -> KernelMatrixType:
#             grad_log_p_evals1 = grad_log_p(p1)
#             if p2 is None:
#                 p2 = p1
#                 grad_log_p_evals2 = grad_log_p_evals1
#             else:
#                 grad_log_p_evals2 = grad_log_p(p2)
#             return kernel_v(p1, grad_log_p_evals1, p2, grad_log_p_evals2, length_scale)
#         return torch.compile(stein_kernel_mat) if use_compiled else stein_kernel_mat
#     else:
#         def stein_kernel(p1: PtType, p2: PtType, length_scale: Float) -> Float:
#             grad_p1 = grad_log_p(p1)
#             grad_p2 = grad_log_p(p2)
#             return stein_kernel_elem(p1, grad_p1, p2, grad_p2, length_scale)
#         return matricize_kernel_elem(stein_kernel, use_compiled)

def stein_kernel_diffs_factory(kernel_fcn: KernelFunction) -> Callable[
    [BatchPtType, BatchPtType, float],
    tuple[KernelMatrixType, Float[Tensor, "batch batch d"], KernelMatrixType]
]:

    kernel_grad = torch.func.grad_and_value(kernel_fcn, argnums=0)

    def process_kernel_grad(x, y, length_scale):
        grad, aux = kernel_grad(x, y, length_scale)
        return grad, (grad, aux)

    kernel_jac = torch.func.jacrev(process_kernel_grad, has_aux=True, argnums=1)

    def process_kernel_jac(x, y, length_scale):
        jac, (grad, ev) = kernel_jac(x, y, length_scale)
        return torch.trace(jac), grad, ev

    kernel_diffs = torch.vmap(
        torch.vmap(process_kernel_jac, in_dims = (0, None, None), out_dims = 0),
        in_dims = (None, 0, None), out_dims=1
    )
    return kernel_diffs

def stein_kernel_mat_factory(
        grad_log_p: GradLogDensity | BatchGradLogDensity,
        kernel_fcn: KernelFunction,
        is_grad_vectorized: bool = False,
        use_compiled: bool = True
) -> MatSelfKernelFunction:
    kernel_diffs = stein_kernel_diffs_factory(kernel_fcn)
    grad_log_p_v = grad_log_p if is_grad_vectorized else torch.vmap(grad_log_p)
    def stein_kernel_mat(pts: BatchPtType, length_scale: float, pts2: Optional[BatchPtType] = None) -> KernelMatrixType:
        grad_log_p_eval1 = grad_log_p_v(pts)
        if pts2 is None:
            pts2 = pts
            grad_log_p_eval2 = grad_log_p_eval1
        else:
            grad_log_p_eval2 = grad_log_p_v(pts2)
        trace_kernel, grad1_kernel, eval_kernel = kernel_diffs(pts, pts2, length_scale)
        out = trace_kernel
        grad_term_1 = torch.einsum("ijd,jd->ij", grad1_kernel, grad_log_p_eval2)
        grad_term_2 = grad_term_1.T
        ev_term = torch.einsum("ij,id,jd->ij", eval_kernel, grad_log_p_eval1, grad_log_p_eval2)
        return trace_kernel + grad_term_1 + grad_term_2 + ev_term
    return torch.compile(stein_kernel_mat) if use_compiled else stein_kernel_mat