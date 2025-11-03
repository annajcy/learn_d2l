import torch
from abc import ABC, abstractmethod
from typing import Iterable

class Model(ABC):
    def __init__(self) -> None:
        pass
    
    def train_mode(self) -> None:
        pass

    def eval_mode(self) -> None:
        pass
    
    def to_device(self, device: torch.device) -> None:
        for param in self.parameters():
            param.to(device)

    @abstractmethod
    def parameters(self) -> Iterable[torch.Tensor]:
        pass

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass
    
class ModelTorch(Model):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net: torch.nn.Module = net
        
    def train_mode(self) -> None:
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def parameters(self) -> Iterable[torch.Tensor]:
        return self.net.parameters()

    def to_device(self, device: torch.device) -> None:
        self.net.to(device)