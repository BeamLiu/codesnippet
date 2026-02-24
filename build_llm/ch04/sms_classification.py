import urllib.request
import zipfile
import os
import pandas as pd
import torch
import tiktoken
import sys
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ch03.gpt_model_pretrained_weight import load_gpt2_small_weights, GPT_CONFIG_124M, choose_device

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    print(f"Downloading {url} to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {zip_path} to {extracted_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    print("Download and extraction complete.")
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]], ignore_index=True)
    return balanced_df

def random_split(df, train_frac, validation_frac):
    #Shuffles the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    #split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=120):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.max_len = max_len
        self.encoded_texts = [tokenizer.encode(text) for text in self.texts]
        # truncate texts that are longer than max_len
        self.encoded_texts = [
            encoded_text[:self.max_len]
            for encoded_text in self.encoded_texts
        ]
        self.tokenizer = tokenizer
        self.encoded_texts = [
            encoded_text + [tokenizer.eot_token] * (self.max_len - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.labels[idx]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

def create_data_loader(df, tokenizer, batch_size, max_len, shuffle=True):
    dataset = SMSDataset(df["Text"], df["Label"], tokenizer, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader

def update_last_n_layers(gpt, n=2):
    for param in gpt.parameters():
        param.requires_grad = False
    # unfreeze the last n-1 transformer blocks
    for i in range(n-1):
        gpt.trf_blocks[-1-i].requires_grad_(True)
    # replace the last layer with a new linear layer
    gpt.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"],
        out_features=2
    )

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    model.train()
    return correct_predictions / num_examples


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def main():
    torch.manual_seed(123)
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print("-" * 50)
    print(df)
    print("-" * 50)
    print(df["Label"].value_counts())
    print("-" * 50)
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    print("-" * 50)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.8, 0.1)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    # hardcoded here, actully it will be the longest encoded length of all texts
    max_len = 120

    batch_size = 8
    train_dataloader = create_data_loader(train_df, tokenizer, batch_size, max_len)
    validation_dataloader = create_data_loader(validation_df, tokenizer, batch_size, max_len)
    test_dataloader = create_data_loader(test_df, tokenizer, batch_size, max_len)

    gpt = load_gpt2_small_weights()
    update_last_n_layers(gpt, n=2)
    gpt.to(choose_device())

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0).to(choose_device())
    print("-" * 50)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)
    with torch.no_grad():
        outputs = gpt(inputs)
        # outputs is the logits for each token in the sequence, we may need the last token's logits for classification
        print("Outputs:", outputs)
        print("Outputs dimensions:", outputs.shape)

    print("-" * 50)
    print("Fine-tuning the model...")
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model=gpt,
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        optimizer=torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1),
        device=choose_device(),
        num_epochs=5,
        eval_freq=100,
        eval_iter=5
    )

if __name__ == "__main__":
    main()