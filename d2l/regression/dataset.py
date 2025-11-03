import torch
from typing import Tuple, Generator, Any
from torch.utils.data import DataLoader, TensorDataset
from abc import abstractmethod
from d2l.base.dataset import Dataset

class SyntheticRegressionDatasetBase(Dataset):
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
        
        self.X, self.y, self.noise = self.generate(
            self.n, 
            self.w, 
            self.b, 
            self.noise_std, 
            self.rng
        )

    @classmethod
    def generate(cls, 
                 n: int, 
                 w: torch.Tensor, 
                 b: torch.Tensor, 
                 noise_std: float, 
                 rng: torch.Generator
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = torch.randn((n, len(w)), generator=rng)
        noise = torch.normal(0, noise_std, (n, 1), generator=rng)
        y = X @ w.reshape((-1, 1)) + b + noise
        return X, y, noise

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

class SyntheticRegressionDataset(SyntheticRegressionDatasetBase):
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

class SyntheticRegressionDatasetTorch(SyntheticRegressionDatasetBase):
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
