from model import NN
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.optim import Optimizer
from typing import Callable, Tuple
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing

class RnnClassifier(NN):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.3):
        super(RnnClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor):
        output, h_n = self.rnn(x)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits
    
    def train_step(self, x: Tensor, y: Tensor, optimizer: Optimizer, loss: Callable[[Tensor, Tensor], Tensor]) -> Tuple[float, float]:
        self.train()
        optimizer.zero_grad()
        y_pred = self.forward(x)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        return l.item()

    def get_model_args(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

