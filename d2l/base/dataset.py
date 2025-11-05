from typing import  Generator, Any
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_train_dataloader(self, batch_size: int) -> Any:
        pass
    
    @abstractmethod
    def get_test_dataloader(self, batch_size: int) -> Any:
        pass

    def get_train_dataloader_epochs(self, batch_size: int, num_epochs: int) -> Generator[Any, None, None]:
        for _ in range(num_epochs):
            yield self.get_train_dataloader(batch_size)