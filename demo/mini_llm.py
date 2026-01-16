import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# 基础配置
# =============================
VOCAB_SIZE = 100     # 词表大小（toy）
EMBED_DIM = 64       # embedding 维度
NUM_HEADS = 4        # 注意力头数
NUM_LAYERS = 2       # Transformer 层数
MAX_SEQ_LEN = 32     # 最大上下文长度


# =============================
# 位置编码（Sinusoidal）
# =============================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        return x + self.pe[: x.size(1)]


# =============================
# 多头自注意力
# =============================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask（防止看到未来 token）
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# =============================
# Transformer Block
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# =============================
# Mini LLM
# =============================
class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = PositionalEncoding(EMBED_DIM, MAX_SEQ_LEN)

        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)]
        )

        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx):
        # idx: (B, T)
        x = self.token_emb(idx)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# =============================
# 测试
# =============================
if __name__ == "__main__":
    model = MiniLLM()
    x = torch.randint(0, VOCAB_SIZE, (1, 10))
    out = model(x)
    print("logits shape:", out.shape)

    gen = model.generate(x, max_new_tokens=5)
    print("generated:", gen)
