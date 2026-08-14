import math

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

class GPTConfig:
    vocab_size = 50257
    block_size = 256
    n_layer = 8
    n_head = 8
    n_embd = 336
    dropout = 0.1

batch_size = 32
learning_rate = 3e-4
weight_decay = 0.1
betas = (0.9, 0.95)
grad_clip = 1.0

max_steps = 200_000
warmup_steps = 2_000
eval_interval = 2_000
eval_iters = 200

train_bin = "train.bin"
val_bin = "val.bin"

enc = tiktoken.get_encoding("gpt2")

class BinDataset:
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+self.block_size+1].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, self.dropout, self.training)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        _B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # reshape, not view: targets may be a non-contiguous slice
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

model = GPT(GPTConfig()).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=betas,
    weight_decay=weight_decay
)

scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

train_data = BinDataset(train_bin, 256)
val_data = BinDataset(val_bin, 256)

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = val_data.get_batch(batch_size)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
model.train()
for step in range(max_steps):

    for param_group in optimizer.param_groups:
        param_group["lr"] = get_lr(step)

    xb, yb = train_data.get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        _, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        print(f"step {step} | train loss {loss.item():.4f}")

    if step % eval_interval == 0 and step > 0:
        val_loss = estimate_loss()
        print(f"🔍 validation loss: {val_loss:.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": vars(GPTConfig()),
    }, 
    "tinystories_28M_final.pt"
)

print("Training complete. Final model saved.")