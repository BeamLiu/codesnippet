import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ch03.qwen_model_pretained_weight import download_model, choose_device

def predict(text, model, tokenizer, device):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=120, 
        padding="max_length"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Label mapping: ham = 0, spam = 1
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_class == 1 else "ham"

def main():
    device = choose_device()
    model_id = download_model("0.6B")
    lora_path = "models/qwen_0.6b_lora_sms" # The path where the model was saved
    
    if not os.path.exists(lora_path):
        print(f"Error: LoRA path '{lora_path}' not found. Please run the fine-tuning script first to save the model.")
        return
    
    print("-" * 50)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2, 
        trust_remote_code=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print("-" * 50)
    print(f"Applying LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path).to(device)
    model.eval()
    
    test_cases = [
        "Congratulations! You've won a $1,000 Walmart gift card. Call 09061790121 now to claim your prize.",
        "Hey, what time are we meeting for dinner today?",
        "URGENT: Your bank account has been locked. Click the link below to verify your identity.",
        "Can you send over the report when you have a minute?",
        "Win £1000 cash! Text WIN to 89999 for your chance. T&Cs apply.",
        "Sounds good, I'll see you at 6!"
    ]
    
    print("-" * 50)
    print("Running Inference Test Cases:")
    for i, text in enumerate(test_cases, 1):
        prediction = predict(text, model, tokenizer, device)
        print(f"Test case {i}:\n\"{text}\"")
        print(f"Prediction: ======> {prediction.upper()} <======")
        print("-" * 40)

if __name__ == "__main__":
    main()
