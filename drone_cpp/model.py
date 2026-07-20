"""Model factory."""
from __future__ import annotations
from typing import Optional
from .data_structures import Instance
from .models.base import BaseModel
from .models.mip_v1 import V1Model as CPPModel
from .models.mip_rings import RingsModel


def build_model(instance: Instance, model_type: str = "rings",
                verbose: bool = True) -> BaseModel:
    if model_type == "v1":
        return CPPModel(instance, verbose)
    elif model_type in ("rings", "v2"):
        return RingsModel(instance, verbose)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
