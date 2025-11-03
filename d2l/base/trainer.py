from typing import Any, Generator, List
from d2l.base.model import Model

class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: Any
    ) -> None:
        self.model = model
        self.optimizer = optimizer

    def _train_single_epoch(self, train_data_loader: Any) -> List[float]:
        self.model.train_mode()
        batch_losses: List[float] = []
        for X, y in train_data_loader:
            y_hat = self.model.forward(X)
            loss = self.model.loss(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            self.optimizer.step()
            self.optimizer.zero_grad()
        return batch_losses

    def train(self, train_data_loaders: Generator[Any, None, None]) -> List[List[float]]:
        all_epoch_losses: List[List[float]] = []
        for train_data_loader in train_data_loaders:
            epoch_loss = self._train_single_epoch(train_data_loader)
            all_epoch_losses.append(epoch_loss)
        return all_epoch_losses