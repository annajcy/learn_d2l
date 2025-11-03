from typing import List
import torch

class SGDOptimizer:
    def __init__(self, params: List[torch.Tensor], lr: float) -> None:
        self.params: List[torch.Tensor] = params
        self.lr: float = lr
        
    def step(self) -> None:
        for param in self.params:
            assert param.grad is not None
            param.data -= self.lr * param.grad
            
    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()