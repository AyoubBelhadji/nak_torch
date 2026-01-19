
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional

PtType = Float[Tensor, " d"]
VecPtType = Float[Tensor, "batch d"]
KernelMatrixType = Float[Tensor, "batch batch"]

KernelFunction = Callable[[PtType, PtType, float], Float]

MatSelfKernelFunction = Callable[[
    VecPtType, float, Optional[VecPtType]
], KernelMatrixType]

GradLogDensity = Callable[[PtType], Float]

VecGradLogDensity = Callable[[
    VecPtType,
], Float[Tensor, " batch"]]
