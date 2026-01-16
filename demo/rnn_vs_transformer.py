import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================
# 数据 & tokenizer
# =============================
text = "你好你好世界你好你好世界"

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
# RNN 模型
# =============================
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits

    def generate(self, idx, steps=10):
        for _ in range(steps):
            logits = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =============================
# Transformer（极简 Attention）
# =============================
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1)))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        logits = self.fc(out)
        return logits

    def generate(self, idx, steps=10):
        for _ in range(steps):
            logits = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =============================
# 训练函数
# =============================
def train(model, steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)

        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

# =============================
# 开始实验
# =============================
rnn = RNNLanguageModel(vocab_size)
transformer = TinyTransformer(vocab_size)

train(rnn)
train(transformer)

start = encode("你").unsqueeze(0)

print("RNN 生成结果:        ", decode(rnn.generate(start, 12)[0].tolist()))
print("Transformer 生成结果:", decode(transformer.generate(start, 12)[0].tolist()))
