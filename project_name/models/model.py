from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor
from typing import Callable, Tuple, Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class NN(nn.Module, ABC):
    def __init__(self):
        super(NN, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, x: Tensor, y: Tensor, optimizer: Optimizer,loss: Callable[[Tensor, Tensor], Tensor]) -> float:
        pass

    @abstractmethod
    def evaluate(self, data: DataLoader, loss: Callable[[Tensor, Tensor], Tensor]) -> Tuple[float, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
