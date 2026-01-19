
from torch import Tensor
from jaxtyping import Float
from typing import Callable

KernelType = Callable[[
    Float[Tensor, " d"],
    Float[Tensor, " d"],
    float
], Float]

MatSelfKernelType = Callable[[
    Float[Tensor, "batch d"],
], Float[Tensor, "batch batch"]]

GradLogDensity = Callable[[
    Float[Tensor, " d"],
], Float]

VecGradLogDensity = Callable[[
    Float[Tensor, "batch d"],
], Float[Tensor, " batch"]]
