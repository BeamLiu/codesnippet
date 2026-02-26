import sys
import os
import torch
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import StyleMIDIModel, ModelConfig
from tokenizer import REMITokenizer
from generate import generate_music

def test_generate_smoke(ckpt_path=""):
    tokenizer = REMITokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = StyleMIDIModel(config)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint provided. Output will be noise.")
        
    device = "cpu"
    model = model.to(device)
    
    conds = {
        "COMPOSER": "beethoven",
        "MOOD": "energetic",
        "TEMPO": "allegro",
        "KEY": "C_major"
    }
    
    print(f"Generating music with conditions: {conds}")
    midi_path = generate_music(
        model=model,
        tokenizer=tokenizer,
        conditions=conds,
        max_new_tokens=100, 
        temperature=1.0,
        top_p=0.9,
        device=device
    )
    print(f"Saved to: {midi_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to model checkpoint")
    args = parser.parse_args()
    
    test_generate_smoke(args.ckpt)
