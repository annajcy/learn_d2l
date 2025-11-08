from typing import List, Optional, Any, Dict
import torch
from d2l.base.dataset import Dataset

import re
import collections
import urllib.request
import os
from pathlib import Path

class Vocab:
    def __init__(self, 
                 tokens: List[str],
                 reserved_tokens: Optional[List[str]] = None,
                 min_freq: int = 0) -> None:
        if reserved_tokens is None:
            reserved_tokens = []
        self.counter = collections.Counter(tokens)
        self.token_freqs = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        self.unique_tokens = ['<unk>'] + reserved_tokens + [token for (token, freq) in self.token_freqs if freq >= min_freq and token not in reserved_tokens]
        self.idx_to_token : List[str] = self.unique_tokens
        self.token_to_idx : Dict[str, int] = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.unk_idx = 0
        self.unk_token = '<unk>'
        
    def size(self) -> int:
        return len(self.idx_to_token)

    def indices_to_token(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token[index] for index in indices]
    
    def token_to_indices(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        
class TimeMachineDataset(Dataset):
    def __init__(self, root="./data", num_steps: int = 35, train_data_scale: float = 0.7, is_word_split_level: bool = False) -> None:
        self.root = root
        self.num_steps = num_steps
        self.is_word_split_level = is_word_split_level
        self.downloaded_text = self.download(root)
        self.processed_text = self.preprocess(self.downloaded_text)
        self.tokens = self.tokenize(self.processed_text)
        self.vocab = Vocab(self.tokens)
        self.corpus = self.vocab.token_to_indices(self.tokens)
        self.num_samples = len(self.corpus) // (self.num_steps + 1)
        self.X_data = torch.zeros((self.num_samples, self.num_steps), dtype=torch.long)
        self.Y_data = torch.zeros((self.num_samples, self.num_steps), dtype=torch.long)
        for i in range(self.num_samples):
            start_idx = i * (self.num_steps + 1)
            self.X_data[i] = torch.tensor(self.corpus[start_idx : start_idx + self.num_steps], dtype=torch.long)
            self.Y_data[i] = torch.tensor(self.corpus[start_idx + 1 : start_idx + self.num_steps + 1], dtype=torch.long)
            
        self.train_size = int(self.num_samples * train_data_scale)
        self.test_size = self.num_samples - self.train_size
        self.X_train = self.X_data[:self.train_size]
        self.Y_train = self.Y_data[:self.train_size]
        self.X_test = self.X_data[self.train_size:]
        self.Y_test = self.Y_data[self.train_size:]

    @classmethod
    def download(cls, root="./data") -> str:
        url = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
        root = Path(root)
        filename = root / 'timemachine.txt'
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(filename):
            print(f'Downloading {url} to {filename}...')
            urllib.request.urlretrieve(url, filename)
        return filename.read_text(encoding='utf-8')

    @classmethod
    def preprocess(cls, text: str) -> str:
        return re.sub(r"[^A-Za-z]+", " ", text).lower()

    def tokenize(self, text: str) -> List[str]:
        if self.is_word_split_level:
            return text.split()
        else:
            return list(text)
    
    def get_train_dataloader(self, batch_size: int) -> Any:
        dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def get_test_dataloader(self, batch_size: int) -> Any:
        dataset = torch.utils.data.TensorDataset(self.X_test, self.Y_test)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
    


    