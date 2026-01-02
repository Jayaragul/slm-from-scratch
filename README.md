# TINY Stories Dataset - GPT Training Project

This project implements a minimal GPT model for training on the TINY Stories dataset.

## Project Structure

- **train.py** - Main training script that trains the GPT model
- **test.py** - Inference and testing script for the trained model
- **tockenizer.py** - Data preprocessing script that converts text data to binary format using GPT-2 tokenizer
- **tinystories_28M_final.pt** - Pre-trained model checkpoint

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tiktoken (OpenAI's tokenizer)
- numpy
- tqdm

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Convert your text data to binary format:
```bash
python tockenizer.py
```

The script expects:
- `TinyStoriesV2-GPT4-train.txt` - Training data
- `TinyStoriesV2-GPT4-val.txt` - Validation data

Output:
- `train.bin` - Binary training data
- `val.bin` - Binary validation data

### 2. Training

Train the model:
```bash
python train.py
```

Configuration:
- Batch size: 32
- Learning rate: 3e-4
- Max steps: 200,000
- Model: 8 layers, 8 attention heads, 336 embeddings

### 3. Testing/Inference

Test the trained model:
```bash
python test.py
```

## Model Architecture

- **Vocab Size**: 50,257 (GPT-2)
- **Block Size**: 256
- **Layers**: 8
- **Attention Heads**: 8
- **Embedding Dimension**: 336

## Device Support

The project automatically detects and uses:
- **GPU (CUDA)** - if available
- **CPU** - fallback if CUDA not available

## Notes

- The model uses causal self-attention to prevent the attention from looking at future tokens
- Supports mixed precision training with proper gradient clipping
- Includes learning rate warmup for better convergence

## References

- **Attention is All You Need** - Vaswani et al., 2017
  - https://arxiv.org/abs/1706.03762
  - Foundational paper introducing the Transformer architecture

- **TinyStories: How Small Can Language Models Be and Still Speak Coherently?** - Zhang et al., 2023
  - https://arxiv.org/abs/2305.07759
  - Demonstrates effective language model training on small datasets
