from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
import torchvision # type: ignore
from torchvision import transforms
from d2l.base.dataset import Dataset
import torch

class FashionMNISTDataset(Dataset):
    def __init__(self, 
                 transform: Optional[transforms.Compose] = None,
                 resize: Tuple[int, int]=(28, 28), 
                 root: str='../data/FashionMNIST') -> None:
        self.resize = resize
        self.root = root
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat', 
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]

        self.train, self.test = self.generate(self.root, self.transform)
        self.train_size = len(self.train)
        self.test_size = len(self.test)

    @classmethod
    def generate(cls, root: str, transform: transforms.Compose) -> Tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
        train = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=transform, download=True
        )
        test = torchvision.datasets.FashionMNIST(
            root=root, train=False, transform=transform, download=True
        )
        return train, test
        
    def get_text_labels(self, labels: List[int]) -> List[str]:
        return [self.text_labels[int(i)] for i in labels]
    
    def get_train_dataloader(self, batch_size: int=64) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.train, batch_size=batch_size, shuffle=True
        )

    def get_test_dataloader(self, batch_size: int=64) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.test, batch_size=batch_size, shuffle=False
        )