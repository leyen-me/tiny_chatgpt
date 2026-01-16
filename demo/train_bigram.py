import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# 1. 数据（我们的小语料）
# =============================
text = "你好你好世界你好你好世界"

# =============================
# 2. 字符级 tokenizer
# =============================
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
# 3. Bigram 模型
# =============================
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 一个表：看到当前字 → 猜下一个字
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        # x: (B, T)
        logits = self.table(x)  # (B, T, vocab)
        return logits

    def generate(self, idx, steps=10):
        for _ in range(steps):
            logits = self(idx)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# =============================
# 4. 训练
# =============================
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for step in range(300):
    x = data[:-1].unsqueeze(0)   # 输入
    y = data[1:].unsqueeze(0)    # 正确答案

    logits = model(x)
    loss = loss_fn(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

# =============================
# 5. 测试生成
# =============================
start = encode("你").unsqueeze(0)
out = model.generate(start, steps=10)
print("生成结果:", decode(out[0].tolist()))
