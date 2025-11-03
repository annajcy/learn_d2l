from typing import Any, Callable, Generator, List, Optional, TypeVar
from d2l.base.model import Model

TrainerType = TypeVar('TrainerType', bound='Trainer')

class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: Any,
        on_train_epoch_end: Optional[Callable[[Any, int, List[float]], Any]] = None
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.on_train_epoch_end = on_train_epoch_end or (lambda model, epoch_id, batch_losses: None)

    def _train_single_epoch(self, train_data_loader: Any) -> Any:
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

    def train(self, train_data_loaders: Generator[Any, None, None]) -> Any:
        epoch_losses: List[List[float]] = []
        for epoch_id, train_data_loader in enumerate(train_data_loaders):
            batch_losses = self._train_single_epoch(train_data_loader)
            epoch_losses.append(batch_losses)
            self.on_train_epoch_end(self.model, epoch_id, batch_losses)
        return epoch_losses