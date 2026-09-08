"""Model factory."""
from __future__ import annotations
from typing import Optional
from .data_structures import Instance
from .models.base import BaseModel
from .models.mip_v1 import V1Model as CPPModel
from .models.mip_rings import RingsModel
from .models.mip_edges import EdgesModel


def build_model(instance: Instance, model_type: str = "rings",
                verbose: bool = True, num_split_points: int = 1,
                wind_aware: bool = False) -> BaseModel:
    """Build the CPP model.

    ``num_split_points`` (K, only for the rings model) is the number of
    continuous split points allowed per ring: K=1 reproduces the original
    single full-loop behaviour, K>=2 partitions each ring into K arcs that
    can be covered in different operations.
    ``wind_aware`` (only for the rings model) replaces the pseudo-energy
    with real joules: per-heading drag energy on ring arcs, centroid-direction
    airspeed factors on inter-region legs and altitude-dependent density/wind
    on depot legs. Off by default (all previous results unaffected).
    """
    if model_type == "v1":
        return CPPModel(instance, verbose)
    elif model_type in ("rings", "v2"):
        return RingsModel(instance, verbose, num_split_points=num_split_points,
                          wind_aware=wind_aware)
    elif model_type == "edges":
        return EdgesModel(instance, verbose)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
