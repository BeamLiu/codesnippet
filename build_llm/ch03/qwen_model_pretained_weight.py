import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")
    return device

def download_model(model_name):
    repo_id = f"Qwen/Qwen3-{model_name}-Base"
    local_dir = "models/" + Path(repo_id).parts[-1]
    
    # If local directory already exists, use it to avoid re-downloading due to network issues
    if Path(local_dir).exists():
        print(f"Found local directory: {local_dir}")
        return local_dir
        
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return repo_dir

def main():
    torch.manual_seed(123)
    
    model_id = download_model("0.6B")
    device = choose_device()
    
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True
    ).to(device)

    print(model)
    
    prompt = "Every effort moves you"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating text...")
    # same as gpt_model_pretrained_weight.py: max_new_tokens=25, top_k=50, temperature=1.5
    outputs = model.generate(
        **inputs,
        max_new_tokens=25,
        top_k=50,
        temperature=1.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 50)
    print("Output text:\n", output_text)

if __name__ == "__main__":
    main()
