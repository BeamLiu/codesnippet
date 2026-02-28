import os
import math
import argparse
import time
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import StyleMIDIModel, ModelConfig
from dataset import get_dataloader
from tokenizer import REMITokenizer

def get_lr(it: int, warmup_steps: int = 750, max_steps: int = 7500, max_lr: float = 6e-4, min_lr: float = 3e-5):
    """Cosine learning rate decay with warmup."""
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run with random synthetic data without actual dataset files")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if available")
    parser.add_argument("--steps", type=int, default=7500, help="Total training steps (approx 20 epochs for 1200 MIDIs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--data-dir", type=str, default="./data/maestro/tokens", help="Directory with processed token sequences")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=500, help="Steps between checkpoints")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up tokenizer
    tokenizer = REMITokenizer()
    vocab_size = tokenizer.vocab_size

    # Dataloader
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        is_smoke_test=args.smoke_test
    )
    
    if len(dataloader) == 0:
        print("DataLoader is empty. Generate data or use --smoke-test.")
        return

    # Model definition
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 2048
    )
    model = StyleMIDIModel(config).to(device)
    print(f"Model params: {model.get_num_params() / 1e6:.2f}M")

    # Optimizer config to properly separate weight decay
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optim_groups, lr=0.0, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Tensorboard logger
    writer = SummaryWriter(log_dir=f"../logs/stylemidi_{int(time.time())}")

    # Training Loop
    step = 0
    if args.resume:
        import glob
        import re
        ckpts = glob.glob(os.path.join(args.ckpt_dir, "model_step_*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda p: int(re.search(r'model_step_(\d+)\.pt', p).group(1)))
            print(f"Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            step = checkpoint['step']
            print(f"Resumed training from step {step}")
        else:
            print(f"No checkpoints found in {args.ckpt_dir}, started from scratch.")

    model.train()
    
    # Infinite loop over dataloader for step-based training
    data_iter = iter(dataloader)
    
    pbar = tqdm(total=args.steps if not args.smoke_test else min(100, args.steps), desc="Training")
    
    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            #End of dataset, restarting.
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        # update LR
        lr = get_lr(step, max_steps=args.steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        device_type_autocast = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type_autocast, enabled=(device.type == 'cuda'), dtype=torch.float16 if device_type_autocast == 'cuda' else torch.bfloat16):
            logits, loss, _ = model(x, targets=y)
            
        scaler.scale(loss).backward()
        
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", lr, step)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
        pbar.update(1)
        step += 1
        
        if step % args.save_every == 0 and not args.smoke_test:
            ckpt_path = os.path.join(args.ckpt_dir, f"model_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config
            }, ckpt_path)
            
        if args.smoke_test and step >= 100:
            print("Smoke test reached 100 steps. Exiting successfully.")
            break
            
    pbar.close()
    writer.close()

if __name__ == "__main__":
    train()
