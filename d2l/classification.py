import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .optimizer import SGD
from .plot import plot
from .dataset import Dataset
from .model import Model
import numpy as np

from typing import Generator, Tuple, List, Any, cast
import matplotlib.pyplot as plt

class FashionMNIST(Dataset):
    def __init__(self, 
                 resize: Tuple[int, int]=(28, 28), 
                 root: str='../data/FashionMINIST') -> None:
        self.resize = resize
        self.root = root
        
        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True
        )
        
        self.test = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True
        )
        
        self.text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat', 
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        
    def get_text_labels(self, labels: List[int]) -> List[str]:
        return [self.text_labels[int(i)] for i in labels]
    
    def get_train_dataloader(self, batch_size: int=64) -> DataLoader:
        return DataLoader(
            self.train, batch_size=batch_size, shuffle=True
        )

    def get_test_dataloader(self, batch_size: int=64) -> DataLoader:
        return DataLoader(
            self.test, batch_size=batch_size, shuffle=False
        )
    
class SoftmaxClassifier(Model):
    def __init__(self, optimizer: Any = None) -> None:
        
 
        super().__init__(optimizer)
        
    def accuracy(self, test_data_loader: DataLoader) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for (X, y) in test_data_loader:
                y_hat = self.forward(X)
                correct += (y_hat.argmax(dim=1) == y).sum().item()
                total += y.shape[0]
        return correct / total

class SoftmaxClassifierScratch(SoftmaxClassifier):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 lr: float = 0.1, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lr = lr
        self.rng = rng
        
        self.W = torch.normal(0, 0.01, (num_features, num_outputs), generator=rng).requires_grad_(True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
        super().__init__(SGD([self.W, self.b], lr))
    
    def softmax(self, X: torch.Tensor) -> torch.Tensor:
        X_exp = torch.exp(X)
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition 
    
    def cross_entropy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return - torch.log(y_hat[torch.arange(len(y_hat)), y])
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(y_hat, y).mean()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape((-1, self.num_features))
        return self.softmax(X @ self.W + self.b)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return super().predict(X).argmax(dim=1)
    
class SoftmaxClassifierTorch(SoftmaxClassifier):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 lr: float = 0.1, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lr = lr
        self.rng = rng

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_outputs)
        )
        
        torch.nn.init.normal_(self.net[1].weight, 0, 0.01, generator=self.rng)  # type: ignore
        torch.nn.init.zeros_(self.net[1].bias)  # type: ignore
        super().__init__(torch.optim.SGD(self.net.parameters(), lr=self.lr))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return super().predict(X).softmax(dim=1).argmax(dim=1)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

