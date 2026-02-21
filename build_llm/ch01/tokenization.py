#download the txt file
import requests
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class DataDownloader():
    def __init__(self):
        self.file_name = "the-verdict.txt"

    #download the txt file
    def download_txt(self):
        if os.path.exists(self.file_name):
            print("the-verdict.txt already exists")
            return
        
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url)
        with open(self.file_name, "w") as f:
            f.write(response.text)
        print(f"Downloaded the txt file with size: {len(response.text)} characters")

    def print_text_info(self):
        with open(self.file_name, "r") as f:
            text = f.read()
        print("Total number of character:", len(text))
        print(text[:99])

    def create_vocabulary(self):
        with open(self.file_name, "r") as f:
            text = f.read()
            result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item for item in result if item.strip()]
            preprocessed.extend(["<|endoftext|>", "<|unk|>"])
            all_words = sorted(set(preprocessed))
            print(f'Vocabulary size: {len(all_words)}')
            print(all_words[:30])
            vocab = {token: i for i, token in enumerate(all_words)}
            return vocab
            

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}    

    def encode(self, text: str):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def simple_tokenzier_sample(vocab: dict[str, int]):
    tokenizer = SimpleTokenizerV1(vocab)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    encoded = tokenizer.encode(text)
    print("SimpleTokenizer Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("SimpleTokenizer Decoded:", decoded)

def bpe_tokenizer_sample():
    tokenizer = tiktoken.get_encoding("gpt2")
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    
    # tiktoken requires allowed_special to parse <|endoftext|>
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("BPE Encoded:", encoded)
    
    # print the subwords
    bpe_subwords = [tokenizer.decode([token_id]) for token_id in encoded]
    print("BPE Subwords:", bpe_subwords)
    
    decoded = tokenizer.decode(encoded)
    print("BPE Decoded:", decoded)


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
    

def tokenizer_sample(batch_size=8, max_length=4):
    data_downloader = DataDownloader()
    data_downloader.download_txt()
    data_downloader.print_text_info()
    vocab = data_downloader.create_vocabulary()
    print("-" * 50)
    simple_tokenzier_sample(vocab)
    print("-" * 50)
    bpe_tokenizer_sample()
    print("-" * 50)
    with open(data_downloader.file_name, "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    return inputs, targets

if __name__ == "__main__":
    tokenizer_sample()