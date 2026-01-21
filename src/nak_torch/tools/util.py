import torch
from torch import Tensor
from jaxtyping import Float

def sym_sqrtm(A: Float[Tensor, "n n"]):
    e, v = torch.linalg.eigh(A)
    return torch.einsum("ij,j,kj->ik", v, e.sqrt_(), v)
