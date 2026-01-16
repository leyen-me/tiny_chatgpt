import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os

# =============================
# 1. 读取文本（A2）
# =============================
text = ""
for path in glob.glob("data/*.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text += f.read() + "\n"

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
# 2. 参数
# =============================
batch_size = 8
block_size = min(16, len(data) - 1)
embed_dim = 64
num_heads = 4
num_layers = 2
max_iters = 10000
lr = 1e-3
model_path = "tiny_chatgpt.pth"

# =============================
# 3. batch
# =============================
def get_batch():
    xs, ys = [], []
    for _ in range(batch_size):
        i = random.randint(0, len(data) - block_size - 1)
        xs.append(data[i:i+block_size])
        ys.append(data[i+1:i+block_size+1])
    return torch.stack(xs), torch.stack(ys)

# =============================
# 4. GPT 模型
# =============================
class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        mask = torch.tril(torch.ones(T, T))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, steps=50):
        for _ in range(steps):
            idx = idx[:, -block_size:]
            logits = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =============================
# 5. 训练 or 加载（A1）
# =============================
model = TinyGPT()
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("✅ 已加载已训练模型")
else:
    print("🚀 开始训练模型")
    for step in range(max_iters):
        xb, yb = get_batch()
        logits = model(xb)
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"step {step}, loss {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print("💾 模型已保存")

# =============================
# 6. 命令行聊天（A3）
# =============================
print("\n💬 进入聊天模式（输入 q 退出）")

while True:
    user = input("你：")
    if user == "q":
        break
    idx = encode(user).unsqueeze(0)
    out = model.generate(idx, 40)
    print("GPT：", decode(out[0].tolist()))
