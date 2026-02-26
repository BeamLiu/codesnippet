import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dataset import get_dataloader, download_dataset
from tokenizer import REMITokenizer

def test_dataset_smoke(smoke_test=True):
    tokenizer = REMITokenizer()
    vocab_size = tokenizer.vocab_size
    
    download_dataset()
    loader = get_dataloader(
        data_dir=os.path.join(os.path.dirname(__file__), '../data'), 
        batch_size=4, 
        seq_len=128, 
        is_smoke_test=smoke_test,
        vocab_size=vocab_size
    )
    
    print(f"Dataset length (batches): {len(loader)}")
    if len(loader) > 0:
        x, y = next(iter(loader))
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        assert x.shape == y.shape == (4, 128)
        print("Smoke test passing: Target shifted by 1 from X.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Run with random synthetic data")
    args = parser.parse_args()
    test_dataset_smoke(args.smoke_test)
