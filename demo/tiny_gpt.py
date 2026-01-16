import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================
# 1. 数据 & tokenizer
# =============================
text = "世界很大，我想去看看"

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return torch.tensor([char_to_id[ch] for ch in s], dtype=torch.long)

def decode(ids):
    return "".join(id_to_char[i] for i in ids)

data = encode(text)

# =============================
# 2. GPT 配置
# =============================
embed_dim = 32
num_heads = 4
num_layers = 2
block_size = 32

# =============================
# 3. Self-Attention
# =============================
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# =============================
# 4. Transformer Block
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# =============================
# 5. tiny-GPT
# =============================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T)

        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        return self.head(x)

    def generate(self, idx, steps=10):
        for _ in range(steps):
            logits = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =============================
# 6. 训练
# =============================
model = TinyGPT()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for step in range(500):
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

# =============================
# 7. 测试生成
# =============================
start = encode("我").unsqueeze(0)
out = model.generate(start, 12)
print("tiny-GPT 生成结果:", decode(out[0].tolist()))
