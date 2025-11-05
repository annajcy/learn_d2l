from typing import List, Tuple
import torch
import torch.nn as nn
from d2l.base.model import Model, ModelTorch
import d2l.base.function as d2l_F

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
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = d2l_F.softmax(y_hat) 
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
        
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_sum_exp = d2l_F.log_sum_exp(y_hat)
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

class MLPClassifier(Model):
    def __init__(self,
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.rng = rng

        self.params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_sizes = [num_features] + num_hiddens + [num_outputs]
        for i in range(len(layer_sizes) - 1):
            d, h = layer_sizes[i], layer_sizes[i + 1]
            W = torch.normal(0, 0.01, (d, h), generator=rng).requires_grad_(True)
            b = torch.zeros(h, requires_grad=True)
            self.params.append((W, b))
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        for i, (W, b) in enumerate(self.params):
            X = X @ W + b
            if i != len(self.params) - 1:
                X = d2l_F.relu(X)
        return X
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = d2l_F.softmax(y_hat) 
        correct_probs = probs[range(len(y_hat)), y] 
        return -torch.log(correct_probs).mean()
    
    def parameters(self) -> List[torch.Tensor]:
        return [param for W_b in self.params for param in W_b]
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
        
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

        super().__init__(self.make_net())
        
    def make_net(self) -> torch.nn.Module:
        layers: List[torch.nn.Module] = [torch.nn.Flatten()]  # Add flatten layer for image input
        input_size = self.num_features
        for hidden_size in self.num_hiddens:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, self.num_outputs))
        net = torch.nn.Sequential(*layers)

        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.01, generator=self.rng)
                torch.nn.init.zeros_(layer.bias)
        return net
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
    
class MLPClassifierDropout(MLPClassifier):
    def __init__(self,
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 dropouts: List[float],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.dropouts = dropouts
        super().__init__(num_features, num_outputs, num_hiddens, rng)
        
    def dropout(self, X: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if drop_prob <= 0.0 or drop_prob >= 1.0:
            return X
        mask = (torch.rand(X.shape, generator=self.rng) > drop_prob).float()
        return X * mask / (1.0 - drop_prob)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        for i, (W, b) in enumerate(self.params):
            X = X @ W + b
            if i != len(self.params) - 1:
                X = d2l_F.relu(X)
                if self.is_training:
                    X = self.dropout(X, self.dropouts[i])
        return X
    
class MLPClassifierDropoutTorch(MLPClassifierTorch):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 dropouts: List[float],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.dropouts = dropouts
        super().__init__(num_features=num_features,
                      num_outputs=num_outputs,
                      num_hiddens=num_hiddens,
                      rng=rng) 

    def make_net(self) -> torch.nn.Module:
        layers: List[torch.nn.Module] = [torch.nn.Flatten()]  # Add flatten layer for image input
        input_size = self.num_features
        for i, hidden_size in enumerate(self.num_hiddens):
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            if self.dropouts[i] > 0.0:
                layers.append(torch.nn.Dropout(p=self.dropouts[i]))
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, self.num_outputs))
        net = torch.nn.Sequential(*layers)

        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.01, generator=self.rng)
                torch.nn.init.zeros_(layer.bias)
        return net
    
class LeNetClassifierTorch(ModelTorch):
    def __init__(self, 
                num_features: Tuple[int, ...],
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.num_features = num_features
        self.rng = rng
        
        super().__init__(self.make_net())

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
    def init(self, X_shape: Tuple[int, ...]) -> None:
        X = torch.randn(*X_shape, generator=self.rng)
        self.forward(X)
        assert isinstance(self.net, torch.nn.Sequential), "init only supports Sequential models."
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=1, generator=self.rng)

    def layer_summary(self, X_shape: Tuple[int, ...]) -> None:
        X = torch.randn(*X_shape, generator=self.rng)
        assert isinstance(self.net, torch.nn.Sequential), "layer_summary only supports Sequential models."
        for layer in self.net:
            print(f'{layer.__class__.__name__:>20}  input shape: {X.shape}')
            X = layer(X)
            print(f'{layer.__class__.__name__:>20}  output shape: {X.shape}')
        print(f'{"Total":>20}  output shape: {X.shape}')
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
