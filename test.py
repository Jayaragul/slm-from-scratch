import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class GPTConfig:
    vocab_size = 50257
    block_size = 256
    n_layer = 8
    n_head = 8
    n_embd = 336
    dropout = 0.0  

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

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
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
    
ckpt = torch.load("tinystories_28M_final.pt", map_location=device)
model = GPT(GPTConfig()).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded successfully.")
enc = tiktoken.get_encoding("gpt2")
@torch.no_grad()
@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=120,
    temperature=0.4,
    top_k=50
):
    model.eval()
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, :].to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -GPTConfig.block_size:]
        logits = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        # top-k sampling
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == enc.eot_token:
            break

        idx = torch.cat([idx, next_token], dim=1)

    return enc.decode(idx[0].tolist())


# ----------------------------
# TEST
# ----------------------------
print("\n--- SAMPLE OUTPUT ---\n")
print(generate("Hi, how are you"))
