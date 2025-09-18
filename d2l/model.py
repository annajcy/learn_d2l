from numpy.random import f
import torch
import numpy as np
from typing import Generator, List, Any
from matplotlib import pyplot as plt
from torch.nn.modules import loss
from d2l.plot import plot
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, optimizer: Any) -> None:
        self.optimizer: Any = optimizer
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
    
    def test(self, test_data_loader: Any) -> Any:
        total_loss = 0.0
        total_samples = 0
        for (X, y) in test_data_loader:
            y_hat = self.forward(X)
            batch_loss = self.loss(y_hat, y)  # 这已经是平均损失
            batch_size = y.shape[0]
            total_loss += batch_loss.item() * batch_size  # 转换为总损失
            total_samples += batch_size
        return total_loss / total_samples
    
    def train_epoch(self, train_data_loader: Any) -> List[float]:
        batch_loss = []
        for (X, y) in train_data_loader:
            y_hat = self.forward(X)
            loss = self.loss(y_hat, y)
            loss.backward()
            batch_loss.append(loss.item())
            self.optimizer.step()
            self.optimizer.zero_grad()
        return batch_loss
        
    def train(self,
              train_data_loaders: Generator[Any, None, None]) -> List[List[float]]:
        all_epoch_loss = []
        for train_data_loader in train_data_loaders:
            epoch_loss = self.train_epoch(train_data_loader)
            all_epoch_loss.append(epoch_loss)
        return all_epoch_loss
    
    def plot_loss(self, all_epoch_loss: List[List[float]]) -> None:
        num_epochs = len(all_epoch_loss)
        num_batch = len(all_epoch_loss[0])
        x = np.arange(1, num_epochs + 1, 1 / num_batch)
        y = np.array([[batch_loss for batch_loss in epoch_loss] for epoch_loss in all_epoch_loss]).flatten()
        fig, ax = plt.subplots()
        plot(ax, (x, [y]), ('epoch', 'loss'), ((1, num_epochs), (0, max(y))), legend=['loss'])
        
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass