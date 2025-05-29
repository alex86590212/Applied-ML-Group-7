from .model import NN
from typing import Sequence, Tuple, Callable
import torch.nn as nn
import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import numpy as np


class CNN(NN):
     def __init__(self, no_channels :int = 3,
                    no_classes : int = 8 ,
                    conv_channels: Sequence[int] = [16, 32, 64],
                    kernel_size: int = 3,
                    pool_size: int = 2,
                    hidden_dim: int = 256,
                    dropout: float = 0.3,
                    input_h : int = 128,
                    input_w : int = 300):
          super(CNN, self).__init__()
          self.no_channels = no_channels
          self.no_classes = no_classes
          self.input_h = input_h
          self.input_w = input_w
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
     
     def get_model_args(self):
          return {
               "no_channels": self.no_channels,
               "no_classes": self.no_classes,
               "input_h": self.input_h,
               "input_w": self.input_w
          }

     def save(self, path: str) -> None:
          torch.save(self.state_dict(), path)

     def load(self, path: str) -> None:
          self.load_state_dict(torch.load(path))

