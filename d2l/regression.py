import torch
from torch import nn
import numpy as np
from typing import Tuple,  Generator, List, Any
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from d2l.plot import plot
from abc import ABC, abstractmethod
from .optimizer import SGD

class SyntheticRegressionDataBase(ABC):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
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
    
    def get_train_data_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(self.num_train, generator=self.rng)[:batch_size]
        return self.X[indices], self.y[indices]
    
    def get_train_data_loaders(self, batch_size: int, epochs: int) -> Generator[List[Any], None, None]:
        for _ in range(epochs):
            yield self.get_train_data_loader(batch_size)

    @abstractmethod
    def get_train_data_loader(self, batch_size: int) -> Any:
        pass

class SyntheticRegressionDataTorch(SyntheticRegressionDataBase):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        super().__init__(w, b, noise_std, num_train, num_test, rng)
        
    def get_train_data_loader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self.X[:self.num_train, :], self.y[:self.num_train])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=self.rng)

class SyntheticRegressionDataScratch(SyntheticRegressionDataBase):
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        super().__init__(w, b, noise_std, num_train, num_test, rng)

    
    def get_train_data_loader(self, batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        indices = torch.randperm(self.num_train, generator=self.rng)
        for i in range(0, self.num_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

class LinearRegressionBase(ABC):
    def __init__(self,
                 num_features: int, 
                 lr: float, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.rng: torch.Generator = rng
        self.lr: float = lr
        self.num_features: int = num_features
        self.optim: Any
        
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    def test(self, test_data: Any) -> float:
        X_test, y_test = test_data
        with torch.no_grad():
            y_hat = self.forward(X_test)
            loss = self.loss(y_hat, y_test)
        return loss.item()
    
    def _train_epoch(self, train_data_loader: Any) -> List[float]:
        batch_loss = []
        for (X, y) in train_data_loader:
            y_hat = self.forward(X)
            loss = self.loss(y_hat, y)
            loss.backward()
            batch_loss.append(loss.item())
            self.optim.step()
            self.optim.zero_grad()
        return batch_loss
        
    def train(self,
              train_data_loaders: Any) -> List[List[float]]:
        all_epoch_loss = []
        for train_data_loader in train_data_loaders:
            epoch_loss = self._train_epoch(train_data_loader)
            all_epoch_loss.append(epoch_loss)
        return all_epoch_loss
    
    def plot_loss(self, all_epoch_loss: List[List[float]]) -> None:
        num_epochs = len(all_epoch_loss)
        num_batch = len(all_epoch_loss[0])
        x = np.arange(1, num_epochs + 1, 1 / num_batch)
        y = np.array([[batch_loss for batch_loss in epoch_loss] for epoch_loss in all_epoch_loss]).flatten()
        fig, ax = plt.subplots()
        plot(ax, (x, [y]), ('epoch', 'loss'), ((1, num_epochs), (0, max(y))), legend=['loss'])

class LinearRegressionScratch(LinearRegressionBase):
    def __init__(self, 
                 num_features: int, 
                 lr: float, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, lr, rng)
        self.w: torch.Tensor = torch.normal(0, 0.01, (num_features, 1), generator=rng).requires_grad_(True)
        self.b: torch.Tensor = torch.zeros(1).requires_grad_(True)
        self.optim: SGD = SGD([self.w, self.b], self.lr)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        loss = (y_hat - y) ** 2 / 2
        return loss.mean()

class LinearRegressionTorch(LinearRegressionBase):
    def __init__(self, 
                 num_features,
                 lr: float, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, lr, rng)
        self.net = nn.Linear(num_features, 1)
        self.net.weight.data.normal_(0, 0.01, generator=rng)
        self.net.bias.data.fill_(0)
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

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
    
class LinearRegressionTorchWD(LinearRegressionTorch):
    def __init__(self, 
                 num_features: int,
                 lr: float, 
                 weight_decay: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, lr, rng)
        # 重新配置优化器：只对 weight 衰减，不对 bias
        self.optim = torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': weight_decay},
            {'params': self.net.bias,   'weight_decay': 0.0}
        ], lr=self.lr)
        self.mse = nn.MSELoss()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mse(y_hat, y)  # 不需要手动加 L2
