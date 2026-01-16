import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# =============================
# 1. 训练文本
# =============================
text = "你好你好世界你好你好世界你好你好世界你好你好世界"

# =============================
# 2. tokenizer
# =============================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    return "".join([itos[i] for i in ids])

data = encode(text)

# =============================
# 3. 训练参数（真实 GPT 风格）
# =============================
batch_size = 4      # 一次几句话
block_size = 8      # 每句话多长
embed_dim = 32
num_heads = 4
num_layers = 2
max_iters = 800
lr = 1e-2

# =============================
# 4. 取 batch 的函数（核心）
# =============================
def get_batch():
    xs, ys = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(data) - block_size - 1)
        x = data[start:start + block_size]
        y = data[start + 1:start + block_size + 1]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# =============================
# 5. Self-Attention
# =============================
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

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
        return self.proj(out)

# =============================
# 6. Transformer Block
# =============================
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# =============================
# 7. tiny-GPT（工程版）
# =============================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )

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

    @torch.no_grad()
    def generate(self, idx, steps=20):
        for _ in range(steps):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =============================
# 8. 训练
# =============================
model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for step in range(max_iters):
    xb, yb = get_batch()
    logits = model(xb)
    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# =============================
# 9. 测试生成
# =============================
start = encode("你").unsqueeze(0)
out = model.generate(start, 20)
print("\n生成结果：", decode(out[0].tolist()))
