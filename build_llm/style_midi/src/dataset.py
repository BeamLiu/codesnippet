import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import os

from tokenizer import REMITokenizer

class MIDIDataset(Dataset):
    """
    Dataset for StyleMIDI.
    Loads tokenized MIDI sequences and yields sliding window segments for next-token prediction.
    """
    def __init__(self, data_dir: str, seq_len: int = 1024, stride: int = 512, split: str = "train", is_smoke_test: bool = False, vocab_size: int = 420):
        self.seq_len = seq_len
        self.stride = stride
        self.samples: List[torch.Tensor] = []
        
        if is_smoke_test:
            self._generate_smoke_test_data(vocab_size)
            return
            
        # Real data logic could go here. Expecting .pt files containing lists of tensor tokens
        # e.g., tokens = torch.load(file_path)
        # For now, if no actual data is loaded, we warn the user
        valid_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')] if os.path.exists(data_dir) else []
        if not valid_files and not is_smoke_test:
            print(f"Warning: No .pt files found in {data_dir}. Use --smoke-test or generate data first.")
            return

        for f in valid_files:
            tokens = torch.load(os.path.join(data_dir, f))
            # Slice into windows
            for i in range(0, len(tokens) - self.seq_len, self.stride):
                window = tokens[i : i + self.seq_len + 1] # +1 for target
                if len(window) == self.seq_len + 1:
                    self.samples.append(window)

    def _generate_smoke_test_data(self, vocab_size: int):
        print("Using smoke test data (random tokens).")
        # generate 100 random sequences of length seq_len + 1
        for _ in range(100):
            seq = torch.randint(0, vocab_size, (self.seq_len + 1,), dtype=torch.long)
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.samples[idx]
        x = window[:-1]
        y = window[1:]
        return x, y

def get_dataloader(data_dir: str, batch_size: int, seq_len: int, stride: int = 512, split: str = "train", is_smoke_test: bool = False, vocab_size: int = 420) -> DataLoader:
    dataset = MIDIDataset(
        data_dir=data_dir, 
        seq_len=seq_len, 
        stride=stride, 
        split=split, 
        is_smoke_test=is_smoke_test,
        vocab_size=vocab_size
    )
    # Using small dataloader workers for simple test
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)

