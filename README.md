# SLM from scratch — a 27.8M-parameter GPT trained on TinyStories

A small language model built from the ground up in PyTorch: the transformer, the attention
mechanism, the tokenizer pipeline and the training loop are all written by hand. No
`transformers`, no `nanoGPT` fork, no trainer abstraction.

The goal was not to beat a benchmark — it was to understand every tensor that moves. At this
scale nothing hides: if the causal mask is wrong, or the learning-rate schedule is wrong, the
loss tells you immediately.

**[⬇ Download the trained weights (108 MB)](../../releases/tag/model)** — run inference without
training anything.

---

## The model

| | |
|---|---|
| **Parameters** | **27,846,000** (27.8M), with tied input/output embeddings |
| Architecture | Decoder-only transformer (GPT-style) |
| Layers | 8 |
| Attention heads | 8 (head dimension 42) |
| Embedding dimension | 336 |
| Context length | 256 tokens |
| Feed-forward | 4× expansion (1344), GELU |
| Vocabulary | 50,257 — GPT-2 BPE via `tiktoken` |
| Dropout | 0.1 |

Where the parameters live:

```
token + position embeddings   16,972,368   (61%)
8 transformer blocks          10,872,960   (39%)
final layer norm                     672
───────────────────────────────────────────
total (unique, tied)          27,846,000
```

At this scale the embedding table dominates — a direct consequence of borrowing GPT-2's
50k vocabulary for a dataset with a small effective vocabulary. Tying `lm_head.weight` to
`wte.weight` saves a further 16.9M parameters that would otherwise sit in the output projection.

## Implementation notes

**Causal attention is written out, not called.** `CausalSelfAttention` computes
`softmax(QKᵀ/√d_k)` explicitly and applies a lower-triangular mask registered as a buffer,
rather than delegating to `F.scaled_dot_product_attention`. Slower, but the masking is visible
and debuggable — which was the point.

**Pre-norm residual blocks.** LayerNorm is applied *before* attention and MLP
(`x = x + attn(ln1(x))`), which keeps gradients well-behaved at depth without needing careful
residual scaling.

**Data is memory-mapped, not loaded.** `tokenizer.py` pre-encodes the corpus once into a flat
`uint16` binary; training then `np.memmap`s it and slices random windows. The dataset never
enters RAM, and batch construction is a pointer offset instead of a tokenization pass.

**Training stability.** AdamW (β = 0.9/0.95, weight decay 0.1), gradient clipping at 1.0,
2,000 steps of linear warmup followed by cosine decay to zero, and mixed-precision autocast
with a gradient scaler on CUDA. Each step sees 32 × 256 = 8,192 tokens.

## Running it

```bash
pip install -r requirements.txt
```

**Inference** — download the checkpoint from the
[release](../../releases/tag/model) into the project root, then:

```bash
python test.py
```

**Training from zero** — get `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-val.txt`
from the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), then:

```bash
python tokenizer.py   # encodes the corpus -> train.bin / val.bin
python train.py       # trains and writes tinystories_28M_final.pt
```

Training runs on CUDA if available and falls back to CPU. Validation loss is estimated every
2,000 steps over 200 batches.

## Files

| File | Purpose |
|---|---|
| `train.py` | Model definition (attention, blocks, GPT) and the training loop |
| `tokenizer.py` | One-off corpus encoding into memory-mappable `uint16` binaries |
| `test.py` | Loads a checkpoint and generates text |

Checkpoints and `.bin` files are gitignored — the trained weights live in the
[release](../../releases/tag/model) instead.

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) — the
  transformer architecture implemented here.
- Eldan & Li, [*TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?*](https://arxiv.org/abs/2305.07759) (2023) — the dataset, and the result that models
  in this size class can still produce fluent text.

## Licence

MIT
