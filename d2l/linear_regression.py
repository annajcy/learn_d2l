import random
import torch

from typing import List, Tuple, Dict, Any, Generator

class SyntheticRegressionData:
    def __init__(self, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float = 0.01, 
                 num_train: int = 1000, 
                 num_test: int = 100,
                 rng: torch.Generator = torch.Generator().manual_seed(0)) -> None:
        
        self.w: torch.Tensor = w
        self.b: torch.Tensor = b
        self.noise_std: float = noise_std
        self.num_train: int = num_train
        self.num_test: int = num_test
        self.n: int = self.num_test + self.num_train
        self.rng: torch.Generator = rng
        
        self.X: torch.Tensor
        self.y: torch.Tensor

        self.generate()

    def generate(self) -> None:
        self.X: torch.Tensor = torch.randn((self.n, len(self.w)), generator=self.rng)
        self.noise: torch.Tensor = torch.normal(0, self.noise_std, (self.n, 1), generator=self.rng)
        self.y: torch.Tensor = self.X @ self.w.reshape((-1, 1)) + self.b + self.noise
    
    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[:self.num_train, :], self.y[:self.num_train]
    
    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[self.num_train:, :], self.y[self.num_train:]

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X, self.y
    
    def get_train_data_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(self.num_train, generator=self.rng)[:batch_size]
        return self.X[indices], self.y[indices]
    
    def get_train_data_batch_generator(self, batch_size: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        indices = torch.randperm(self.num_train, generator=self.rng)
        for i in range(0, self.num_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
            
    def get_train_data_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(self.X[:self.num_train, :], self.y[:self.num_train])
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_test_data_loader(self) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(self.X[self.num_train:, :], self.y[self.num_train:])
        return torch.utils.data.DataLoader(dataset, batch_size=self.num_test, shuffle=False)
