import os
import sys
import pandas as pd
import torch
import json
import pretty_midi
import kagglehub
from tqdm import tqdm

# Add parent directory to sys.path to import src.tokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer import REMITokenizer
from scripts.extract_features import estimate_key, calculate_audio_metrics
from scripts.analyze_conditions import plot_condition_distributions

def download_dataset():
    output_dir = "./data/maestro"
    if not os.path.exists(output_dir):
        try:
            # Users may have specified output_dir in earlier snippets, 
            # preserving the original logic intention while preventing errors.
            path = kagglehub.dataset_download("kritanjalijain/maestropianomidi")
            print(f"Downloaded to cache: {path}")
            print("Please ensure the dataset is moved/extracted to ./data/maestro/maestro-v3.0.0")
        except TypeError:
            # fallback if output_dir is supported
            kagglehub.dataset_download("kritanjalijain/maestropianomidi", output_dir=output_dir)
    else:
        print("Dataset directory already exists!")


def process_dataset(encode_tokens=True):
    dataset_dir = "./data/maestro/maestro-v3.0.0"
    aug_csv_path = os.path.join(dataset_dir, "maestro-v3.0.0_augmented.csv")
    csv_path = os.path.join(dataset_dir, "maestro-v3.0.0.csv")
    output_dir = "./data/maestro/tokens"
    enum_json_path = os.path.join(output_dir, "enums.json")
    
    # Use augmented CSV if available, else original
    if os.path.exists(aug_csv_path):
        target_csv = aug_csv_path
        print(f"Loading augmented CSV metadata from {target_csv}...")
    elif os.path.exists(csv_path):
        target_csv = csv_path
        print(f"Loading original CSV metadata from {target_csv}...")
    else:
        print(f"Error: CSV metadata not found at {csv_path} or {aug_csv_path}")
        print("Please ensure the dataset is extracted correctly to ./data/maestro/maestro-v3.0.0")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    try:
        df = pd.read_csv(target_csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
        
    # Read enums if available, otherwise generate them on the fly
    valid_composers = []
    if os.path.exists(enum_json_path):
        with open(enum_json_path, 'r', encoding='utf-8') as f:
            enums_dict = json.load(f)
            valid_composers = enums_dict.get("COMPOSER", [])
            
    if not valid_composers:
        unique_composers = df['canonical_composer'].unique()
        valid_composers = sorted(list(set([str(c).lower().strip() for c in unique_composers])))
        # Create continuous buckets 0.0, 0.1, ... 1.0 (as strings)
        continuous_buckets = [f"{i/10:.1f}" for i in range(11)]
        keys_enum = [f"{k}_{mode}" for k in ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"] for mode in ["major", "minor"]]
        
        enums_dict = {
            "COMPOSER": valid_composers,
            "VELOCITY": continuous_buckets,    # Formerly MOOD
            "TEMPO": continuous_buckets,
            "DENSITY": continuous_buckets,
            "KEY": keys_enum
        }
        with open(enum_json_path, 'w', encoding='utf-8') as f:
            json.dump(enums_dict, f, indent=4, ensure_ascii=False)
            
    if 'velocity' not in df.columns: df['velocity'] = None
    if 'density' not in df.columns: df['density'] = None
    if 'tempo' not in df.columns: df['tempo'] = None
    if 'key' not in df.columns: df['key'] = None
    
    # Initialize the tokenizer
    tokenizer = REMITokenizer()
    print(f"Starting tokenization and feature extraction for {len(df)} MIDI files...")
    
    needs_save = False
    all_extracted_conditions = []
    
    # Iterate through metadata to encode MIDI files
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIDI files"):
        midi_filename = row['midi_filename']
        midi_path = os.path.join(dataset_dir, midi_filename)
        
        if not os.path.exists(midi_path):
            continue
            
        # 1. Edge-Compute Features if missing
        if pd.isna(row.get('tempo')) or pd.isna(row.get('velocity')) or pd.isna(row.get('density')) or pd.isna(row.get('key')):
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)
                metrics = calculate_audio_metrics(pm)
                df.at[idx, 'velocity'] = row['velocity'] = metrics['velocity']
                df.at[idx, 'density'] = row['density'] = metrics['density']
                df.at[idx, 'tempo'] = row['tempo'] = metrics['tempo']
                df.at[idx, 'key'] = row['key'] = estimate_key(pm)
                needs_save = True
            except Exception as e:
                pass # Skip if reading fails
                
        # Extract Conditions for Tokenizer
        conditions = {}
        composer_full = str(row.get('canonical_composer', '')).lower().strip()
        
        # Match composer directly
        if composer_full in valid_composers:
             conditions["COMPOSER"] = composer_full
        else:
            for comp in valid_composers:
                if comp in composer_full:
                    conditions["COMPOSER"] = comp
                    break
        
        if 'velocity' in row and pd.notna(row['velocity']):
            conditions["VELOCITY"] = str(row['velocity'])
        if 'density' in row and pd.notna(row['density']):
            conditions["DENSITY"] = str(row['density'])
        if 'tempo' in row and pd.notna(row['tempo']):
            conditions["TEMPO"] = str(row['tempo'])
        if 'key' in row and pd.notna(row['key']):
            conditions["KEY"] = row['key']
            
        all_extracted_conditions.append(conditions)
            
        # Tokenize and Save
        if encode_tokens:
            # Encode MIDI file into token IDs
            tokens = tokenizer.encode(midi_path, conditions=conditions)
            if tokens and len(tokens) > 0:
                # Save as Pytorch Tensor
                tensor_data = torch.tensor(tokens, dtype=torch.long)
                
                # Create a simple flat filename based on the midi filename hierarchy
                save_name = str(midi_filename).replace("/", "_").replace("\\", "_").replace(".midi", ".pt").replace(".mid", ".pt")
                save_path = os.path.join(output_dir, save_name)
                
                torch.save(tensor_data, save_path)

    # Save augmented state
    if needs_save:
        df.to_csv(aug_csv_path, index=False)
        print(f"Saved computed augmented data to {aug_csv_path}")

    # Analyze and plot condition distributions
    print(f"\nAnalyzing condition distributions for {len(all_extracted_conditions)} items...")
    plot_condition_distributions(all_extracted_conditions, output_dir)


if __name__ == "__main__":
    print("Data Preparation Script:")
    print("1. Download MAESTRO from Kaggle")
    print("2. After extracting, ensure it is placed in the data/maestro/maestro-v3.0.0 directory.")
    print("3. Extracts any missing Tempo/Mood/Key inline from MIDI files.")
    print("4. Uses src/tokenizer.py to encode each MIDI file and extract tokens.")
    print("5. Uses torch.save(tokens) to save as PyTorch Tensor format for Dataset consumption.")

    print("\n--- Checking and Downloading Dataset ---")
    download_dataset()
    
    print("\n--- Starting Dataset Processing and Token Conversion ---")
    process_dataset()
    print("\nData processing complete! Token files have been saved to ./data/maestro/tokens")