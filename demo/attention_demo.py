import torch
import torch.nn.functional as F
import math

# =============================
# 一句话（字符级）
# =============================
sentence = "你好世界"
chars = list(sentence)

print("句子:", chars)

# =============================
# 给每个字一个“向量”（随便的）
# =============================
torch.manual_seed(42)
embed_dim = 4

embeddings = torch.randn(len(chars), embed_dim)

# =============================
# Self-Attention（最小版）
# =============================
Q = embeddings
K = embeddings
V = embeddings

# Attention 分数
scores = Q @ K.T / math.sqrt(embed_dim)

# 因果遮挡（不能看未来）
mask = torch.tril(torch.ones(len(chars), len(chars)))
scores = scores.masked_fill(mask == 0, float("-inf"))

# Attention 权重
attn = F.softmax(scores, dim=-1)

# 输出
output = attn @ V

# =============================
# 打印 Attention Map
# =============================
print("\nAttention Map（每一行表示：当前字在看谁）\n")

for i, row in enumerate(attn):
    print(f"当前字: {chars[i]}")
    for j, weight in enumerate(row):
        if weight > 0:
            print(f"  看 {chars[j]} 的程度: {weight.item():.2f}")
    print()
