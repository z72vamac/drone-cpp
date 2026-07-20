from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict
from ..data_structures import Instance, Solution


class BaseModel(ABC):
    def __init__(self, instance: Instance, verbose: bool = True):
        self.inst = instance

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def optimize(self, time_limit: float = 3600.) -> Optional[Solution]:
        ...

    def variable_summary(self) -> Dict[str, int]:
        return {}
