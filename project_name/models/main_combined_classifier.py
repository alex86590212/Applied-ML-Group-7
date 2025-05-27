from .model import NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Tuple, Sequence
from sklearn.metrics import f1_score

class CombinedClassifier(NN):
    def __init__(self,
             input_rnn_dim: int,
             hidden_rnn_dim: int,
             no_channels: int = 3,
             no_classes: int = 8,
             conv_channels: Sequence[int] = [16, 32, 64],
             kernel_size: int = 3,
             pool_size: int = 2,
             hidden_dim: int = 256,
             dropout: float = 0.3,
             input_cnn_h: int = 128,
             input_cnn_w: int = 300,
             num_rnn_layers: int = 1):

        super(CombinedClassifier, self).__init__()
        self.no_channels = no_channels
        self.no_classes = no_classes
        self.input_cnn_h = input_cnn_h
        self.input_cnn_w = input_cnn_w
        self.input_rnn_dim = input_rnn_dim
        self.hidden_rnn_dim = hidden_rnn_dim
        self.num_rnn_layers = num_rnn_layers

        self.cnn = nn.ModuleList()
        current_channel = no_channels
        for out_ch in conv_channels:
            self.cnn.append(nn.Sequential(
                nn.Conv2d(current_channel, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(pool_size))
            )
            current_channel = out_ch
        dummy = torch.zeros(1,no_channels, input_cnn_h, input_cnn_w)
        with torch.no_grad():
            x = dummy
            for block in self.cnn:
                x = block(x)
        flat_cnn_size = x.numel()

        self.rnn = nn.GRU(input_rnn_dim, hidden_rnn_dim, num_rnn_layers, batch_first=True, dropout=dropout if num_rnn_layers > 1 else 0)

        self.classifier = nn.Sequential(
                nn.Linear(flat_cnn_size + hidden_rnn_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, no_classes)
            )

    def forward(self, x_spec: Tensor, x_manual: Tensor) -> Tensor:
        for block in self.cnn:
            x_spec = block(x_spec)
        x_spec = x_spec.view(x_spec.size(0), -1)  

        rnn_out, _ = self.rnn(x_manual)
        x_manual = rnn_out[:, -1, :]  

        combined = torch.cat((x_spec, x_manual), dim=1)
        return self.classifier(combined)

    def train_step(self, x_spec: Tensor, x_manual: Tensor, y: Tensor, optimizer: Optimizer, loss_fn: Callable) -> float:
        self.train()
        optimizer.zero_grad()
        y_pred = self.forward(x_spec, x_manual)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, dataloader, loss_fn):
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_spec_batch, x_manual_batch, y_batch in dataloader:
                device = next(self.parameters()).device
                x_spec_batch = x_spec_batch.to(device)
                x_manual_batch = x_manual_batch.to(device)
                y_batch = y_batch.to(device)

                output = self(x_spec_batch, x_manual_batch)
                loss = loss_fn(output, y_batch)
                total_loss += loss.item()

                preds = output.argmax(dim=1).cpu().numpy()
                labels = y_batch.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        f1 = f1_score(all_labels, all_preds, average="macro")
        return total_loss / len(dataloader), f1

    def get_model_args(self):
        return {
            "no_channels": self.no_channels,
            "no_classes": self.no_classes,
            "input_cnn_h": self.input_cnn_h,
            "input_cnn_w": self.input_cnn_w,
            "input_rnn_dim": self.input_rnn_dim,
            "hidden_rnn_dim": self.hidden_rnn_dim
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
