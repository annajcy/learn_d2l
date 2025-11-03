from typing import Iterable, List
import torch
from d2l.base.model import Model, ModelTorch

class SoftmaxClassifier(Model):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.rng = rng

        self.W = torch.normal(0, 0.01, (num_features, num_outputs), generator=rng).requires_grad_(True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
        
    @classmethod
    def softmax(cls, X: torch.Tensor) -> torch.Tensor:
        X_max = X.max(dim=1, keepdim=True).values
        X_exp = torch.exp(X - X_max)
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = self.softmax(y_hat) 
        correct_probs = probs[range(len(y_hat)), y] 
        return -torch.log(correct_probs).mean()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        return X @ self.W + self.b
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)

    def parameters(self) -> List[torch.Tensor]:
        return [self.W, self.b]
    
class SoftmaxClassifierLogSumExp(SoftmaxClassifier):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, num_outputs, rng)
        
    @classmethod    
    def log_sum_exp(cls, X: torch.Tensor) -> torch.Tensor:
        c = X.max(dim=1, keepdim=True).values
        return torch.log(torch.exp(X - c).sum(dim=1)) + c.squeeze(dim=1)
        
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_sum_exp = self.log_sum_exp(y_hat)
        correct_prob = y_hat[range(len(y_hat)), y]
        return (log_sum_exp - correct_prob).mean()

class SoftmaxClassifierTorch(ModelTorch):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.rng = rng

        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(num_features, num_outputs)
        torch.nn.init.normal_(linear.weight, 0, 0.01, generator=self.rng)  
        torch.nn.init.zeros_(linear.bias)
        net = torch.nn.Sequential(flatten, linear)
        super().__init__(net)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

class MLPClassifierTorch(ModelTorch):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.rng = rng

        layers: List[torch.nn.Module] = [torch.nn.Flatten()]  # Add flatten layer for image input
        input_size = num_features
        for hidden_size in num_hiddens:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, num_outputs))
        net = torch.nn.Sequential(*layers)

        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.01, generator=self.rng)
                torch.nn.init.zeros_(layer.bias)
        super().__init__(net)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
