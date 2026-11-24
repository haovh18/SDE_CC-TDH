"""Paper-aligned, model-agnostic SDE and CC-TDH core."""

from .cctdh import CCTDHConfig, CCTDHGradients, combine_cctdh_gradients
from .rpu import ExampleState, PairState, build_rpu_manifest
from .sde import SDEWeights, compute_sde_objective

__all__ = [
    "CCTDHConfig",
    "CCTDHGradients",
    "ExampleState",
    "PairState",
    "SDEWeights",
    "build_rpu_manifest",
    "combine_cctdh_gradients",
    "compute_sde_objective",
]

