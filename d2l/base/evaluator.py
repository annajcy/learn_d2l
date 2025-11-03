from typing import Any, TypeVar

import torch

from d2l.base.model import Model

EvaluatorType = TypeVar('EvaluatorType', bound='Evaluator')

class Evaluator:
    def __init__(self, model: Model) -> None:
        self.model = model

    def loss(self, test_data_loader: Any) -> float:
        self.model.eval_mode()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in test_data_loader:
                y_hat = self.model.forward(X)
                batch_loss = self.model.loss(y_hat, y)
                batch_size = y.shape[0]
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise ValueError("Test dataloader produced zero samples.")
        return total_loss / total_samples

class ClassificationEvaluator(Evaluator):
    def accuracy(self, test_data_loader: Any) -> float:
        self.model.eval_mode()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_data_loader:
                y_hat = self.model.forward(X)
                correct += (y_hat.argmax(dim=1) == y).sum().item()
                total += y.shape[0]
        if total == 0:
            raise ValueError("Test dataloader produced zero samples.")
        return correct / total
