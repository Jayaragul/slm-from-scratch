import os
import numpy as np
import tiktoken
from tqdm import tqdm

train_file = "TinyStoriesV2-GPT4-train.txt"
valid_file = "TinyStoriesV2-GPT4-val.txt"

enc = tiktoken.get_encoding("gpt2")

def pretokenize_to_bin(filename, out_filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    
    ids = enc.encode_ordinary(data)
    ids = np.array(ids, dtype=np.uint16)
    
    ids.tofile(out_filename)
    print(f"Saved {len(ids)} tokens to {out_filename}")

print("Pre-processing local data into binary format...")
pretokenize_to_bin(train_file, 'train.bin')
pretokenize_to_bin(valid_file, 'val.bin')