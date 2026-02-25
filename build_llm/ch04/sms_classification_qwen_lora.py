import os
import pandas as pd
import torch
import sys
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ch03.qwen_model_pretained_weight import download_model, choose_device
from ch04.sms_classification_gpt import download_and_unzip_spam_data, create_balanced_dataset, random_split

class HFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=120):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def main():
    torch.manual_seed(123)
    
    # 1. Download and load Data
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.8, 0.1)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")

    # 2. Get Model and Tokenizer from Hugging Face
    device = choose_device()
    model_id = download_model("0.6B")
    
    print("-" * 50)
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2, 
        trust_remote_code=True
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    print("-" * 50)
    model.print_trainable_parameters()

    # Dataset Preparation
    max_len = 120
    train_dataset = HFDataset(train_df["Text"], train_df["Label"], tokenizer, max_len)
    validation_dataset = HFDataset(validation_df["Text"], validation_df["Label"], tokenizer, max_len)
    test_dataset = HFDataset(test_df["Text"], test_df["Label"], tokenizer, max_len)

    # Demo output
    inputs = tokenizer("Do you have time", return_tensors="pt").to(device)
    print("-" * 50)
    print("Inputs:", inputs["input_ids"])
    print("Inputs dimensions:", inputs["input_ids"].shape)
    with torch.no_grad():
        outputs = model(**inputs)
        print("Outputs (logits):", outputs.logits)
        print("Outputs dimensions:", outputs.logits.shape)

    # 3. Fine-tuning the model using Hugging Face Trainer
    print("-" * 50)
    print("Fine-tuning the model using Hugging Face Trainer...")
    
    training_args = TrainingArguments(
        output_dir="./qwen_results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.1,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=(device.type == "cpu"),
        bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device.type == "cuda" and not torch.cuda.is_bf16_supported())
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("-" * 50)
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")

    print("-" * 50)
    save_path = "models/qwen_0.6b_lora_sms"
    print(f"Saving model adapters and tokenizer to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
