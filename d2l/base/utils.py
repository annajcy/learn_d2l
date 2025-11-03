import torch

def cpu() -> torch.device:  
    return torch.device('cpu')

def gpu(i=0) -> torch.device: 
    return torch.device(f'cuda:{i}')

def mps() -> torch.device:  
    return torch.device('mps')

def num_gpus() -> int:  
    return torch.cuda.device_count()

def try_gpu(i=0) -> torch.device:  
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus() -> list[torch.device]: 
    return [gpu(i) for i in range(num_gpus())]
