import torch

# softmax and log_sum_exp functions
def softmax(X: torch.Tensor) -> torch.Tensor:
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def log_sum_exp(X: torch.Tensor) -> torch.Tensor:
    c = X.max(dim=1, keepdim=True).values
    return torch.log(torch.exp(X - c).sum(dim=1)) + c.squeeze(dim=1)
    
# activation functions 
def relu(X: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(X)
    return torch.max(X, zero)

def p_relu(X: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    zero = torch.zeros_like(X)
    return torch.max(zero, X) + alpha * torch.min(zero, X)

def sigmoid(X: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-X))

def tanh(X: torch.Tensor) -> torch.Tensor:
    return (1 - torch.exp(-2 * X)) / (1 + torch.exp(-2 * X))

def gelu(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * X * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (X + 0.044715 * torch.pow(X, 3))))

def swish(X: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return X * sigmoid(beta * X)
