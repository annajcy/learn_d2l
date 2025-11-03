import torch
from torch import nn
from typing import List
from d2l.base.model import Model, ModelTorch

class LinearRegression(Model):
    def __init__(self, 
                 num_features: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.rng: torch.Generator = rng
        self.w: torch.Tensor = torch.normal(0, 0.01, (num_features, 1), generator=self.rng).requires_grad_(True)
        self.b: torch.Tensor = torch.zeros(1).requires_grad_(True)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        loss = (y_hat - y) ** 2 / 2
        return loss.mean()

    def parameters(self) -> List[torch.Tensor]:
        return [self.w, self.b]
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
    
class LinearRegressionL2(LinearRegression):
    def __init__(self, 
                 num_features: int, 
                 weight_decay: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, rng)
        self.weight_decay: float = weight_decay

    def l2_penalty(self, weight: torch.Tensor) -> torch.Tensor:
        return 0.5 * weight.pow(2).sum()

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        l2_penalty = self.l2_penalty(self.w)
        loss = (y_hat - y) ** 2 / 2 + self.weight_decay * l2_penalty
        return loss.mean()

class LinearRegressionTorch(ModelTorch):
    def __init__(self, 
                 num_features: int,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_features: int = num_features
        self.rng: torch.Generator = rng
        
        linear = nn.Linear(num_features, 1)
        torch.nn.init.normal_(linear.weight, 0, 0.01, generator=self.rng)
        torch.nn.init.zeros_(linear.bias)
        
        net = nn.Sequential(
            linear
        )

        super().__init__(net)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:   
        return nn.functional.mse_loss(y_hat, y)