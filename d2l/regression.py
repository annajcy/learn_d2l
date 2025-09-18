import torch
from torch import nn
import numpy as np
from typing import Tuple,  Generator, List, Any
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from d2l.plot import plot
from abc import ABC, abstractmethod
from .optimizer import SGD
from .dataset import Dataset
from .model import Model

class SyntheticRegressionDataset(Dataset):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        super().__init__()
        self.w: torch.Tensor = w
        self.b: torch.Tensor = b
        self.num_features: int = len(w)
        self.noise_std: float = noise_std
        self.num_train: int = num_train
        self.num_test: int = num_test
        self.n: int = self.num_test + self.num_train
        self.rng: torch.Generator = rng
        
        self.X: torch.Tensor
        self.y: torch.Tensor

        self.generate()

    def generate(self) -> None:
        self.X = torch.randn((self.n, len(self.w)), generator=self.rng)
        self.noise = torch.normal(0, self.noise_std, (self.n, 1), generator=self.rng)
        self.y = self.X @ self.w.reshape((-1, 1)) + self.b + self.noise

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[:self.num_train, :], self.y[:self.num_train]
    
    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[self.num_train:, :], self.y[self.num_train:]

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X, self.y
    
    def get_train_data_batch_sampled(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(self.num_train, generator=self.rng)[:batch_size]
        return self.X[indices], self.y[indices]

    @abstractmethod
    def get_train_dataloader(self, batch_size: int) -> Any:
        pass
    
    @abstractmethod
    def get_test_dataloader(self, batch_size: int) -> Any:
        pass

class SyntheticRegressionDataTorch(SyntheticRegressionDataset):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        super().__init__(w, b, noise_std, num_train, num_test, rng)
        
    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self.X[:self.num_train, :], self.y[:self.num_train])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=self.rng)
    
    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self.X[self.num_train:, :], self.y[self.num_train:])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=self.rng)

class SyntheticRegressionDataScratch(SyntheticRegressionDataset):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        super().__init__(w, b, noise_std, num_train, num_test, rng)
    
    def get_train_dataloader(self, batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        indices = torch.randperm(self.num_train, generator=self.rng)
        for i in range(0, self.num_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
            
    def get_test_dataloader(self, batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        indices = torch.arange(self.num_test)
        for i in range(0, self.num_test, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.X[self.num_train + batch_indices], self.y[self.num_train + batch_indices]

class LinearRegressionScratch(Model):
    def __init__(self, 
                 num_features: int, 
                 lr: float, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        
        self.num_features: int = num_features
        self.lr: float = lr
        self.rng: torch.Generator = rng
        self.w: torch.Tensor = torch.normal(0, 0.01, (num_features, 1), generator=self.rng).requires_grad_(True)
        self.b: torch.Tensor = torch.zeros(1).requires_grad_(True)
        super().__init__(SGD([self.w, self.b], self.lr))
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        loss = (y_hat - y) ** 2 / 2
        return loss.mean()

class LinearRegressionTorch(Model):
    def __init__(self, 
                 num_features,
                 lr: float, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_features: int = num_features
        self.lr: float = lr
        self.rng: torch.Generator = rng
        self.net = nn.Linear(num_features, 1)
        
        torch.nn.init.normal_(self.net.weight, 0, 0.01, generator=self.rng)
        torch.nn.init.zeros_(self.net.bias)
        
        self.loss_fn = nn.MSELoss()
        super().__init__(torch.optim.SGD(self.net.parameters(), lr=self.lr))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

class LinearRegressionTorchL2(LinearRegressionTorch):
    def __init__(self, 
                 num_features: int,
                 lr: float, 
                 weight_decay: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, lr, rng)
        self.weight_decay = weight_decay

    def l2_penalty(self, weight: torch.Tensor) -> torch.Tensor:
        return 0.5 * weight.pow(2).sum()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l2_reg = self.l2_penalty(self.net.weight)
        return self.loss_fn(y_hat, y) + self.weight_decay * l2_reg
    
class LinearRegressionTorchL2Optim(LinearRegressionTorch):
    def __init__(self, 
                 num_features: int,
                 lr: float, 
                 weight_decay: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, lr, rng)
        
        self.optimizer = torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': weight_decay},
            {'params': self.net.bias,   'weight_decay': 0.0}
        ], lr=self.lr)
        
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)
