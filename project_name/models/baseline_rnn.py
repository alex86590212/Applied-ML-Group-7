from model import NN
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.optim import Optimizer
from typing import Callable, Tuple
from torch.utils.data import DataLoader, TensorDataset
import torch
from data.preprocessing import Preprocessing
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class RnnClassifier(NN):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.3):
        super(RnnClassifier, self)._init_()
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
    
    def evaluate(self, data: DataLoader, loss: Callable[[Tensor, Tensor], Tensor]) -> Tuple[float, float]:
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in data:
                y_pred = self.forward(x_batch)
                total_loss += loss(y_pred, y_batch).item() * x_batch.size(0)
                total_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
                total_samples += x_batch.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

if __name__ == "_main_":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)
    train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
    train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(train_spec, train_manual)

    valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
    valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(valid_spec, valid_manual)

    input_dim = X_manual_train.shape[2]
    hidden_dim = 64
    output_dim = len(np.unique(y_manual_train))
    model = RnnClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=2)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    x_train_tensor = torch.tensor(X_manual_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_manual_train, dtype=torch.long)
    x_valid_tensor = torch.tensor(X_manual_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_manual_valid, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=32)

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            loss = model.train_step(x_batch, y_batch, optimizer, loss_fn)
            total_loss += loss

        val_loss, val_acc = model.evaluate(valid_loader, loss_fn)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
