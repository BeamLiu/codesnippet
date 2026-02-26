import sys
import os
import torch
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import StyleMIDIModel, ModelConfig

def test_model_smoke():
    config = ModelConfig()
    model = StyleMIDIModel(config)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f} M")
    
    # Smoke test input
    idx = torch.randint(0, config.vocab_size, (2, 128))
    targets = torch.randint(0, config.vocab_size, (2, 128))
    
    logits, loss, _ = model(idx, targets=targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    assert logits.shape == (2, 128, config.vocab_size)
    print("Transformer model smoke test passing.")

if __name__ == "__main__":
    test_model_smoke()
