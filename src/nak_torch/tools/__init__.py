# Various tools for the package
# Ayoub Belhadji
# 05/12/2025


from .viz_tools.twodims_viz_tools import animate_trajectories_box
from . import kernel, types, quadrature
from .average import recursive_weighted_average_alpha_v
from .torchify import differentiable_density_factory
from .types import GaussianModel

__all__ = [
    "animate_trajectories_box",
    "kernel",
    "types",
    "recursive_weighted_average_alpha_v",
    "differentiable_density_factory",
    "GaussianModel",
    "quadrature"
]

