import torch
from torch._C import dtype
from torch.tensor import Tensor
from torch.utils.data import Dataset

from typing import List, Tuple

class CharDataset(Dataset):
    """
    Takes a giant str of read text file as input, operates on CHARACTER level, so vocab will just be 
    whatever letters and symbols found in text, not full words. 
    """
    def __init__(self, data: str, seq_len=128):
        chars = sorted(set(data))

        self.data = data
        self.data_size = len(data)
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for i, s in enumerate(chars)}

        print(f'Data has {self.data_size} chars, {self.vocab_size} unique vocabs total')

    def __len__(self) -> int:
        return self.data_size - self.seq_len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        start = idx
        end = idx + self.seq_len + 1
        chunk = self.data[start:end]
        idc: List[int] = [self.stoi[s] for s in chunk] # (seq_len + 1) ints
        
        x = torch.tensor(idc[:self.seq_len], dtype=torch.long) # seq_len 
        y = torch.tensor(idc[1:], dtype=torch.long) # seq_len shifted over by 1 as target

        return x, y