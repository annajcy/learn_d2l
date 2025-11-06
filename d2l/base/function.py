from typing import Tuple
import torch
import torch.nn as nn

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


# convolution operation
def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return sum(corr2d(x, k) for x, k in zip(X, K)) # type: ignore

def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)  # type: ignore

def corr2d_multi_in_out_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

def comp_conv2d(conv2d: nn.Module, X: torch.Tensor) -> torch.Tensor:
    # X = X.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # Remove batch and channel dimensions

def max_pool2d(X: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1), dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = X[i: i + p_h, j: j + p_w].max()
    return Y

def avg_pool2d(X: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1), dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

def batch_norm_1d(
        X: torch.Tensor, 
        gamma: torch.Tensor,
        beta: torch.Tensor,
        moving_mean: torch.Tensor,
        moving_var: torch.Tensor,
        momentum: float,
        eps: float = 1e-5,
        is_training: bool = True) -> torch.Tensor:
    if is_training:
        batch_mean = X.mean(dim=0, keepdim=True)
        batch_var = X.var(dim=0, unbiased=False, keepdim=True)
        X_hat = (X - batch_mean) / torch.sqrt(batch_var + eps)
        with torch.no_grad():
            moving_mean.mul_(momentum).add_((1.0 - momentum) * batch_mean)
            moving_var.mul_(momentum).add_((1.0 - momentum) * batch_var)
    else:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    Y = gamma * X_hat + beta
    return Y

def batch_norm_2d(
        X: torch.Tensor, 
        gamma: torch.Tensor,
        beta: torch.Tensor,
        moving_mean: torch.Tensor,
        moving_var: torch.Tensor,
        momentum: float,
        eps: float = 1e-5,
        is_training: bool = True) -> torch.Tensor:
    if is_training:
        batch_mean = X.mean(dim=(0, 2, 3), keepdim=True)
        batch_var = X.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        X_hat = (X - batch_mean) / torch.sqrt(batch_var + eps)
        with torch.no_grad():
            moving_mean.mul_(momentum).add_((1.0 - momentum) * batch_mean)
            moving_var.mul_(momentum).add_((1.0 - momentum) * batch_var)
    else:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    Y = gamma * X_hat + beta
    return Y

def layer_norm(
        X: torch.Tensor, 
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float = 1e-5) -> torch.Tensor:
    mean = X.mean(dim=1, keepdim=True)
    var = X.var(dim=1, keepdim=True, unbiased=False)
    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = gamma * X_hat + beta
    return Y