import torch
from torch._C import dtype
from torch.tensor import Tensor
from torch.utils.data import Dataset, data

from typing import Tuple

class CharDataset(Dataset):
    def __init__(self, data, seq_len = 128):
        chars = sorted(set(data))

        self.data = data
        self.data_size = len(data)
        self.char_size = len(chars)
        self.seq_len = seq_len

        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for i, s in enumerate(chars)}

        print(f'Data has {self.data_size} chars, {self.char_size} unique vocabs total')

    def __len__(self) -> int:
        return self.data_size - self.seq_len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        chunk = self.data[idx:idx + self.seq_len + 1]
        idc = [self.stoi[s] for s in chunk]
        
        x = torch.tensor(idc[:self.seq_len], dtype = torch.long)
        y = torch.tensor(idc[1:], dtype = torch.long)

        return x, y