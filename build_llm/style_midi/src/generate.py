import torch
import torch.nn.functional as F
from typing import Dict, Optional
import os

from model import StyleMIDIModel, ModelConfig
from tokenizer import REMITokenizer

@torch.no_grad()
def generate_music(
    model: StyleMIDIModel,
    tokenizer: REMITokenizer,
    conditions: Dict[str, str],
    max_duration: float = 30.0,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu"
) -> str:
    """Generate MIDI sequence based on conditions."""
    model.eval()
    
    # 1. Encode conditions into prefix tokens
    input_ids = [tokenizer.bos_token_id]
    if conditions:
        for k, v in conditions.items():
            tok_str = f"[{k}:{v}]"
            if tok_str in tokenizer.vocab:
                input_ids.append(tokenizer.vocab[tok_str])
        
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # print initial condition prefix
    print(f"Condition prefix tokens: {x.tolist()[0]}")
    
    # 2. Autoregressive Generation with KV Cache
    kv_caches = None
    start_pos = 0
    
    current_bar = 0
    current_time_sec = 0.0
    sec_per_bar = 4.0 * (60.0 / 120.0)  # default tempo 120 bpm, 4/4 time
    
    for _ in range(20000):  # hard limit to prevent infinite loops
        # We only pass the new token(s) and the current kv_caches
        logits, _, kv_caches = model(x, kv_caches=kv_caches, start_pos=start_pos)
        
        # next token is the last prediction
        next_token_logits = logits[0, -1, :]
        
        # Top-p (nucleus) sampling
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # top-p filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy Decode
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        # Append to generated sequence
        input_ids.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            print("Hit EOS token, stopping generation.")
            break
            
        token_str = tokenizer.idx_to_str.get(next_token.item(), "")
        if token_str == "Bar_None":
            current_bar += 1
            current_time_sec = current_bar * sec_per_bar
        elif token_str.startswith("Position_"):
            pos = int(token_str.split("_")[1])
            current_time_sec = current_bar * sec_per_bar + (pos / tokenizer.positions_per_bar) * sec_per_bar
            
        if current_time_sec >= max_duration:
            print(f"Reached max duration {max_duration}s, stopping generation.")
            break
            
        # prepare for next iteration
        x = next_token.unsqueeze(0)
        start_pos += logits.shape[1]
        
    print(f"Generated {len(input_ids)} tokens.")
    
    # 3. Decode Tokens to MIDI
    output_pm = tokenizer.decode(input_ids)
    
    import time
    # Save MIDI
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../samples')), exist_ok=True)
    timestamp = int(time.time())
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../samples/generated_{conditions.get('COMPOSER', 'unknown')}_{timestamp}.mid"))
    output_pm.write(out_path)
    
    return out_path


