from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor
from typing import Callable, Tuple, Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch

class NN(nn.Module, ABC):
    def __init__(self):
        super(NN, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, x: Tensor, y: Tensor, optimizer: Optimizer,loss: Callable[[Tensor, Tensor], Tensor]) -> float:
        pass

    from sklearn.metrics import f1_score

    def evaluate(self, dataloader, loss_fn):
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                device = next(self.parameters()).device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = self(x_batch)
                loss = loss_fn(output, y_batch)
                total_loss += loss.item()

                preds = output.argmax(dim=1).cpu().numpy()
                labels = y_batch.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        f1 = f1_score(all_labels, all_preds, average="macro")

        return total_loss / len(dataloader), f1


    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
