import torch
from abc import ABC, abstractmethod
from typing import Iterable

class Model(ABC):
    def __init__(self) -> None:
        self.is_training: bool = False
    
    def train_mode(self) -> None:
        self.is_training = True

    def eval_mode(self) -> None:
        self.is_training = False

    def to_device(self, device: torch.device) -> None:
        for param in self.parameters():
            param.to(device)
            
    def save(self, path: str) -> None:
        print("Warning: save method not implemented.")
        pass

    def load(self, path: str) -> None:
        print("Warning: load method not implemented.")
        pass

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
        super().train_mode()
        self.net.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.net.eval()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def parameters(self) -> Iterable[torch.Tensor]:
        return self.net.parameters()

    def to_device(self, device: torch.device) -> None:
        self.net.to(device)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))
        
    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass