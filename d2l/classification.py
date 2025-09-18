import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .optimizer import SGD

from typing import Generator, Tuple, List

class FashionMNIST():
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
        
    def get_train_dataloaders(self, batch_size: int=64, epochs: int=10) -> Generator[DataLoader, None, None]:
        for _ in range(epochs):
            yield self.get_train_dataloader(batch_size)

    def get_test_dataloader(self, batch_size: int=64) -> DataLoader:
        return DataLoader(
            self.test, batch_size=batch_size, shuffle=False
        )
        
class SoftmaxClassifierScratch():
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 lr: float = 0.1, 
                 sigma: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lr = lr
        self.sigma = sigma
        self.rng = rng
        
        self.W = torch.normal(
            0, sigma, size=(num_features, num_outputs), generator=rng
        ).requires_grad_(True)
        
        self.b = torch.zeros(num_outputs).requires_grad_(True)
        
        self.optim = SGD([self.W, self.b], lr=self.lr)
        
    def parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.W, self.b)
    
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
    
    def _train_epoch(self, train_data_loader: DataLoader) -> List[float]:
        batch_loss = []
        for (X, y) in train_data_loader:
            y_hat = self.forward(X)
            loss = self.loss(y_hat, y)
            batch_loss.append(loss.item())
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return batch_loss
    
    def train(self, 
              train_dataloaders: Generator[DataLoader, None, None]) -> List[List[float]]:
        all_epoch_loss = []
        for train_data_loader in train_dataloaders:
            epoch_loss = self._train_epoch(train_data_loader)
            all_epoch_loss.append(epoch_loss)
        return all_epoch_loss

    
    