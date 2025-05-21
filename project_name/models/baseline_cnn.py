from model import NN
from typing import Sequence, Tuple, Callable
import torch.nn as nn
import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project_name.data.preprocessing import Preprocessing

class CNN(NN):
    def __init__(self, no_channels :int = 3,
                  no_classes : int = 8 ,
                  conv_channels: Sequence[int] = [16, 32, 64],
                  kernel_size: int = 3,
                  pool_size: int = 2,
                  hidden_dim: int = 256,
                  dropout: float = 0.3,
                  input_h : int = 128,
                  input_w : int = 300
        ):
         super(CNN, self).__init__()
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
         dummy = torch.zeros(1,no_channels, input_h, input_w)
         with torch.no_grad():
               x = dummy
               for block in self.cnn:
                   x = block(x)
         flat_size = x.numel()
         self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear(flat_size, hidden_dim),
              nn.ReLU(),
              nn.Dropout(dropout),
              nn.Linear(hidden_dim, no_classes)
              )
         

    def forward(self, x : Tensor) -> Tensor:
         for block in self.cnn:
              x = block(x)
         logits = self.classifier(x)
         return logits
    
    def train_step(self, x, y, optimizer, loss):
         self.train()
         optimizer.zero_grad()
         pred = self.forward(x)
         l = loss(pred, y)
         l.backward()
         optimizer.step()
         return l.item()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    p = Preprocessing(0.7, 0.15, 0.15, 48000)

    train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
    train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
    X_spec_train, X_manual_train, y_spec_train, y_manual_train = p.load_dual_inputs(train_spec, train_manual)

    valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
    valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
    X_spec_valid, X_manual_valid, y_spec_valid, y_manual_valid = p.load_dual_inputs(valid_spec, valid_manual)

    
    input_channels = X_spec_train.shape[1] 
    input_h = X_spec_train.shape[2]        
    input_w = X_spec_train.shape[3]         
    output_dim = len(np.unique(y_spec_train))  

 
    model = CNN(no_channels=input_channels, 
                no_classes=output_dim, 
                input_h=input_h, 
                input_w=input_w)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    
    x_train_tensor = torch.tensor(X_spec_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_spec_train, dtype=torch.long)
    x_valid_tensor = torch.tensor(X_spec_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_spec_valid, dtype=torch.long)

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