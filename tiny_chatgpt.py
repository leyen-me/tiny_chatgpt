# =============================
# 导入必要的库
# =============================
import glob  # 用于查找匹配特定模式的文件路径（如 "data/*.txt"）
import torch  # PyTorch 核心库，用于张量操作和深度学习
import torch.nn as nn  # PyTorch 神经网络模块，包含各种层（Linear, Embedding等）
import torch.nn.functional as F  # PyTorch 函数式接口，包含激活函数、损失函数等
import math  # 数学库，用于计算（如 sqrt 用于注意力机制中的缩放）
import random  # 随机数库，用于随机选择训练样本
import os  # 操作系统接口，用于检查文件是否存在

# =============================
# 1. 读取文本数据并构建词汇表（A2）
# =============================
# 这个部分的目标是：从文件中读取文本，然后将文本转换成数字，方便模型处理

# 初始化一个空字符串，用于存储所有读取的文本
text = ""

# 使用 glob.glob() 查找 data 目录下所有 .txt 文件
# 例如：如果 data 目录下有 data01.txt 和 data02.txt，这里会找到这两个文件
for path in glob.glob("data/*.txt"):
    # 以 UTF-8 编码打开文件并读取内容
    # 每次读取后添加一个换行符，确保不同文件的内容之间有分隔
    with open(path, "r", encoding="utf-8") as f:
        text += f.read() + "\n"

# 从文本中提取所有唯一的字符，并排序
# set(text) 获取所有不重复的字符，list() 转为列表，sorted() 排序
# 例如：如果文本是 "hello"，chars 就是 ['e', 'h', 'l', 'o']
chars = sorted(list(set(text)))

# 词汇表大小：有多少个不同的字符
# 这个数字决定了模型需要预测多少个不同的字符
vocab_size = len(chars)

# stoi: string to index（字符串到索引的映射）
# 将每个字符映射到一个数字，例如：{'h': 0, 'e': 1, 'l': 2, 'o': 3}
# 这样我们就可以用数字来表示字符了
stoi = {ch: i for i, ch in enumerate(chars)}

# itos: index to string（索引到字符串的映射）
# 与 stoi 相反，将数字映射回字符，例如：{0: 'h', 1: 'e', 2: 'l', 3: 'o'}
# 这样我们就可以把模型输出的数字转换回文字了
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """
    将字符串编码成数字序列（张量）
    
    参数:
        s: 输入的字符串，例如 "hello"
    
    返回:
        一个 PyTorch 张量，包含每个字符对应的数字
        例如："hello" -> tensor([0, 1, 2, 2, 3])
    
    工作原理:
        遍历字符串中的每个字符，查找它在 stoi 字典中对应的数字
    """
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    """
    将数字序列（张量或列表）解码成字符串
    
    参数:
        ids: 数字序列，可以是张量或列表，例如 [0, 1, 2, 2, 3]
    
    返回:
        对应的字符串，例如 tensor([0, 1, 2, 2, 3]) -> "hello"
    
    工作原理:
        遍历数字序列，查找每个数字在 itos 字典中对应的字符，然后拼接起来
    """
    return "".join([itos[i] for i in ids])

# 将整个文本转换成数字序列
# 例如：如果文本是 "hello world"，data 就是一个包含所有字符对应数字的张量
data = encode(text)

# 将数据分成训练集和验证集
# 90% 用于训练，10% 用于验证
# split 是分割点的索引位置
split = int(0.9 * len(data))

# 训练集：前 90% 的数据，用于训练模型
train_data = data[:split]

# 验证集：后 10% 的数据，用于评估模型性能（不参与训练）
val_data = data[split:]

# =============================
# 2. 模型和训练的超参数设置
# =============================
# 这些参数控制模型的大小、训练过程等，可以根据需要调整

# batch_size: 批次大小，每次训练时同时处理的样本数量
# 越大训练越快，但需要更多内存；越小训练越慢，但内存占用更少
batch_size = 16

# block_size: 上下文长度，模型能看到的前面多少个字符
# 例如 block_size=64 意味着模型在预测下一个字符时，最多能看到前面 64 个字符
# min(64, len(data) - 1) 确保不会超过数据长度
block_size = min(64, len(data) - 1)

# embed_dim: 词嵌入维度，每个字符用多少维的向量表示
# 更大的维度可以表示更丰富的信息，但计算量也更大
embed_dim = 128

# num_heads: 多头注意力的头数
# 注意力机制可以让模型关注文本的不同方面，多头意味着可以同时关注多个方面
# 例如：一个头关注语法，另一个头关注语义
num_heads = 4

# num_layers: Transformer 块的层数
# 每一层都会对输入进行更复杂的处理，层数越多模型越强大，但也越慢
num_layers = 4

# dropout: 随机失活率，防止模型过拟合
# 训练时会随机将一些神经元置零，让模型不过度依赖某些特征
# 0.1 表示 10% 的神经元会被随机置零
dropout = 0.1

# max_iters: 最大训练迭代次数
# 模型会训练这么多次，每次处理一个批次的数据
max_iters = 4500

# lr: 学习率，控制模型参数更新的步长
# 太大可能导致训练不稳定，太小训练太慢
# 3e-4 表示 0.0003
lr = 3e-4

# weight_decay: 权重衰减（L2 正则化），防止模型过拟合
# 会让模型参数倾向于更小的值，提高泛化能力
weight_decay = 0.01

# eval_interval: 评估间隔，每训练多少次就评估一次模型性能
# 500 表示每训练 500 次就打印一次训练和验证损失
eval_interval = 500

# eval_iters: 评估时使用的迭代次数
# 评估时会随机采样多个批次来计算平均损失，这个数字控制采样多少个批次
eval_iters = 100

# model_path: 模型保存路径
# 训练完成后模型会保存到这个文件，下次可以直接加载使用
model_path = "tiny_chatgpt.pth"

# device: 计算设备，优先使用 GPU（如果可用），否则使用 CPU
# GPU 训练速度快很多，但需要安装 CUDA 和对应的 PyTorch 版本
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 3. 批次数据生成和损失评估函数
# =============================

def get_batch(split_name="train"):
    """
    生成一个训练批次的数据
    
    参数:
        split_name: "train" 或 "val"，指定使用训练集还是验证集
    
    返回:
        xs: 输入序列，形状为 [batch_size, block_size]
           例如：如果 batch_size=2, block_size=5，可能是：
           [[1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14]]
        ys: 目标序列（标签），形状为 [batch_size, block_size]
           这是 xs 向右移动一位的结果，因为我们要预测下一个字符
           例如：如果 xs 是 [[1, 2, 3, 4, 5]]，ys 就是 [[2, 3, 4, 5, 6]]
    
    工作原理:
        1. 从训练集或验证集中随机选择 batch_size 个位置
        2. 每个位置取 block_size 个字符作为输入（xs）
        3. 对应的下一个字符序列作为目标（ys）
        4. 将所有样本堆叠成批次，并移动到指定设备（CPU/GPU）
    """
    # xs 存储输入序列，ys 存储目标序列（标签）
    xs, ys = [], []
    
    # 根据 split_name 选择使用训练集还是验证集
    src = train_data if split_name == "train" else val_data
    
    # 生成 batch_size 个样本
    for _ in range(batch_size):
        # 随机选择一个起始位置
        # len(src) - block_size - 1 确保不会越界
        # 例如：如果数据长度是 100，block_size 是 10，i 的范围是 0 到 89
        i = random.randint(0, len(src) - block_size - 1)
        
        # 输入序列：从位置 i 开始，取 block_size 个字符
        # 例如：src[i:i+block_size] 表示从索引 i 到 i+block_size-1
        xs.append(src[i:i+block_size])
        
        # 目标序列：从位置 i+1 开始，取 block_size 个字符
        # 这就是输入序列的下一个字符，模型要学习预测这个
        # 例如：如果输入是 [1,2,3,4,5]，目标就是 [2,3,4,5,6]
        ys.append(src[i+1:i+block_size+1])
    
    # torch.stack() 将列表中的多个张量堆叠成一个批次
    # .to(device) 将数据移动到 GPU 或 CPU
    return torch.stack(xs).to(device), torch.stack(ys).to(device)

@torch.no_grad()  # 这个装饰器表示在评估时不计算梯度，节省内存和加速
def estimate_loss(model):
    """
    评估模型在训练集和验证集上的平均损失
    
    参数:
        model: 要评估的模型
    
    返回:
        一个字典，包含训练集和验证集的平均损失
        例如：{'train': 2.345, 'val': 2.567}
    
    工作原理:
        1. 将模型设置为评估模式（关闭 dropout 等）
        2. 对训练集和验证集分别采样多个批次
        3. 计算每个批次的损失并求平均
        4. 将模型恢复为训练模式
    """
    # 设置为评估模式，这会关闭 dropout、batch normalization 的训练行为等
    model.eval()
    
    # 存储结果的字典
    out = {}
    
    # 分别评估训练集和验证集
    for split_name in ["train", "val"]:
        # 存储所有批次的损失值
        losses = []
        
        # 采样 eval_iters 个批次来计算平均损失
        # 采样多个批次可以更准确地估计损失
        for _ in range(eval_iters):
            # 获取一个批次的数据
            xb, yb = get_batch(split_name)
            
            # 模型前向传播，得到预测结果
            # logits 的形状是 [batch_size, block_size, vocab_size]
            # 表示每个位置对每个字符的预测分数
            logits = model(xb)
            
            # 计算损失
            # logits.view(-1, vocab_size) 将 logits 重塑为 [batch_size*block_size, vocab_size]
            # yb.view(-1) 将目标重塑为 [batch_size*block_size]
            # 这样每一行对应一个字符位置的预测和真实值
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            
            # 将损失值添加到列表中（.item() 将张量转为 Python 数字）
            losses.append(loss.item())
        
        # 计算平均损失
        out[split_name] = sum(losses) / len(losses)
    
    # 恢复训练模式，这样后续训练时 dropout 等会正常工作
    model.train()
    
    return out

# =============================
# 4. GPT 模型定义
# =============================
# GPT 模型的核心组件：自注意力机制（Self-Attention）和 Transformer 块

class SelfAttention(nn.Module):
    """
    自注意力机制（Self-Attention）
    
    这是 Transformer 的核心组件，让模型能够关注输入序列中不同位置的信息。
    简单理解：当模型处理一个词时，它可以"看"到序列中其他所有词，并决定哪些词更重要。
    
    例如：处理 "我喜欢吃苹果" 这句话时，处理"苹果"这个词时，
    模型会关注"吃"这个词，因为"吃"和"苹果"有很强的关联。
    """
    
    def __init__(self, dim, heads):
        """
        初始化自注意力层
        
        参数:
            dim: 输入维度（通常是 embed_dim）
            heads: 多头注意力的头数
        """
        super().__init__()
        
        # 头数：多头注意力可以让模型同时关注不同类型的信息
        self.h = heads
        
        # 每个头的维度：总维度除以头数
        # 例如：如果 dim=128, heads=4，那么每个头的维度 d=32
        self.d = dim // heads
        
        # QKV 线性层：将输入映射为 Query（查询）、Key（键）、Value（值）
        # 这三个是注意力机制的核心概念：
        # - Query: 当前要查询的信息（"我想知道什么"）
        # - Key: 每个位置的关键信息（"我有什么信息"）
        # - Value: 每个位置的实际内容（"我的实际值是什么"）
        # dim * 3 是因为要同时生成 Q、K、V 三个矩阵
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 输出投影层：将多头注意力的结果映射回原始维度
        self.proj = nn.Linear(dim, dim)
        
        # 注意力 dropout：在注意力权重上应用 dropout
        self.attn_drop = nn.Dropout(dropout)
        
        # 残差连接 dropout：在输出上应用 dropout
        self.resid_drop = nn.Dropout(dropout)
        
        # 因果掩码（Causal Mask）：确保模型只能看到当前位置之前的信息
        # torch.tril() 生成下三角矩阵，上三角部分为 0
        # 例如：[[1,0,0],
        #       [1,1,0],
        #       [1,1,1]]
        # 这样在预测第 3 个字符时，只能看到前 3 个字符，看不到后面的
        # register_buffer 表示这是模型的缓冲区（不是可训练参数），但会随模型一起保存
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        前向传播：计算自注意力
        
        参数:
            x: 输入张量，形状为 [batch_size, sequence_length, embed_dim]
        
        返回:
            经过自注意力处理后的输出，形状与输入相同
        """
        # B: batch_size（批次大小）
        # T: sequence_length（序列长度，即 block_size）
        # C: embed_dim（嵌入维度）
        B, T, C = x.shape
        
        # 通过 QKV 层得到 Q、K、V 三个矩阵
        # chunk(3, dim=-1) 将输出分成 3 份，每份的维度是 C
        # 例如：如果输入是 [B, T, 128]，输出是 [B, T, 384]，分成 3 份就是 3 个 [B, T, 128]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # 重塑为多头形式
        # view(B, T, self.h, self.d) 将 [B, T, C] 重塑为 [B, T, heads, d]
        # transpose(1, 2) 交换维度，得到 [B, heads, T, d]
        # 这样每个头可以独立计算注意力
        q = q.view(B, T, self.h, self.d).transpose(1, 2)  # [B, heads, T, d]
        k = k.view(B, T, self.h, self.d).transpose(1, 2)  # [B, heads, T, d]
        v = v.view(B, T, self.h, self.d).transpose(1, 2)  # [B, heads, T, d]

        # 计算注意力分数：Q @ K^T
        # @ 是矩阵乘法，k.transpose(-2, -1) 转置最后两个维度
        # 结果形状：[B, heads, T, T]，表示每个位置对其他位置的注意力分数
        # 除以 sqrt(d) 进行缩放，防止分数过大导致 softmax 饱和
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        
        # 应用因果掩码：将未来位置的注意力分数设为负无穷
        # masked_fill 将 mask 中为 0 的位置（上三角部分）填充为 -inf
        # 这样经过 softmax 后，这些位置的权重会变成 0
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        
        # Softmax 归一化：将注意力分数转换为概率分布（权重和为 1）
        # 这样每一行的权重加起来等于 1，表示每个位置对其他位置的关注程度
        att = F.softmax(att, dim=-1)
        
        # 应用 dropout：随机将一些注意力权重置零，防止过拟合
        att = self.attn_drop(att)

        # 加权求和：用注意力权重对 Value 进行加权
        # 结果形状：[B, heads, T, d]
        out = att @ v
        
        # 合并多头：将多个头的结果拼接起来
        # transpose(1, 2) 交换维度：[B, heads, T, d] -> [B, T, heads, d]
        # contiguous() 确保内存连续（某些操作需要）
        # view(B, T, C) 重塑为 [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影和 dropout：将结果映射回原始维度并应用 dropout
        return self.resid_drop(self.proj(out))

class Block(nn.Module):
    """
    Transformer 块（Block）
    
    这是 GPT 模型的基本构建单元，包含：
    1. 自注意力层：让模型关注序列中不同位置的信息
    2. 前馈神经网络：对每个位置进行非线性变换
    3. 残差连接：让信息可以直接传递，帮助训练深层网络
    4. 层归一化：稳定训练过程
    
    多个这样的块堆叠起来就构成了完整的 GPT 模型。
    """
    
    def __init__(self, dim, heads):
        """
        初始化 Transformer 块
        
        参数:
            dim: 输入维度（通常是 embed_dim）
            heads: 多头注意力的头数
        """
        super().__init__()
        
        # 第一个层归一化：在自注意力之前使用
        # LayerNorm 对每个样本的特征进行归一化，稳定训练
        self.ln1 = nn.LayerNorm(dim)
        
        # 自注意力层：核心组件
        self.attn = SelfAttention(dim, heads)
        
        # 第二个层归一化：在前馈网络之前使用
        self.ln2 = nn.LayerNorm(dim)
        
        # 前馈神经网络（Feed-Forward Network）
        # 这是一个两层的全连接网络，对每个位置独立处理
        # 结构：输入 -> 扩展（4倍）-> 激活 -> 压缩回原维度 -> dropout
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 第一层：扩展到 4 倍维度
            nn.GELU(),                 # GELU 激活函数（比 ReLU 更平滑）
            nn.Linear(dim * 4, dim),  # 第二层：压缩回原始维度
            nn.Dropout(dropout)        # Dropout 防止过拟合
        )

    def forward(self, x):
        """
        前向传播：执行 Transformer 块的计算
        
        参数:
            x: 输入张量，形状为 [batch_size, sequence_length, embed_dim]
        
        返回:
            处理后的输出，形状与输入相同
        
        注意：
            这里使用了残差连接（residual connection）
            x = x + ... 表示将输入直接加到输出上
            这样可以让梯度更容易传播，帮助训练深层网络
        """
        # 第一个残差连接：自注意力 + 残差
        # 先对输入进行层归一化，然后通过自注意力，最后加上原始输入
        # 这允许信息"跳过"注意力层直接传递
        x = x + self.attn(self.ln1(x))
        
        # 第二个残差连接：前馈网络 + 残差
        # 先对输入进行层归一化，然后通过前馈网络，最后加上原始输入
        x = x + self.ff(self.ln2(x))
        
        return x

class TinyGPT(nn.Module):
    """
    完整的 GPT 模型
    
    这是整个语言模型的主体，包含：
    1. 词嵌入层：将字符 ID 转换为向量
    2. 位置嵌入层：为每个位置添加位置信息
    3. 多个 Transformer 块：堆叠的注意力层
    4. 输出层：将隐藏状态映射为词汇表上的概率分布
    """
    
    def __init__(self):
        """
        初始化 GPT 模型
        
        创建所有必要的层和组件
        """
        super().__init__()
        
        # 词嵌入层（Token Embedding）
        # 将每个字符的 ID（0 到 vocab_size-1）映射为一个 embed_dim 维的向量
        # 例如：字符 'h' 的 ID 是 0，会被映射为一个 128 维的向量
        self.tok = nn.Embedding(vocab_size, embed_dim)
        
        # 位置嵌入层（Position Embedding）
        # 为序列中的每个位置（0 到 block_size-1）学习一个 embed_dim 维的向量
        # 这样模型就知道每个字符在序列中的位置
        # 例如：第 0 个位置和第 5 个位置会有不同的位置向量
        self.pos = nn.Embedding(block_size, embed_dim)
        
        # Transformer 块列表：堆叠多个 Block
        # ModuleList 是 PyTorch 的特殊容器，用于存储多个模块
        # 这里创建 num_layers 个 Block，每个 Block 的结构相同
        # 例如：如果 num_layers=4，就有 4 个 Block 堆叠在一起
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        # 最终的层归一化：在所有 Block 之后使用
        self.ln = nn.LayerNorm(embed_dim)
        
        # 输出头（Output Head）：将隐藏状态映射为词汇表上的分数
        # Linear(embed_dim, vocab_size) 将每个位置的 embed_dim 维向量
        # 映射为 vocab_size 维的向量，表示对每个字符的预测分数
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        前向传播：给定输入序列，预测每个位置的下一个字符
        
        参数:
            idx: 输入序列的字符 ID，形状为 [batch_size, sequence_length]
                 例如：[[1, 2, 3, 4, 5]] 表示一个批次，包含 5 个字符
        
        返回:
            logits: 每个位置对每个字符的预测分数
                    形状为 [batch_size, sequence_length, vocab_size]
                    例如：logits[0, 0, :] 表示第一个位置对所有字符的预测分数
        """
        # B: batch_size（批次大小）
        # T: sequence_length（序列长度）
        B, T = idx.shape
        
        # 生成位置索引：0, 1, 2, ..., T-1
        # 这些索引用于查找位置嵌入
        pos = torch.arange(T, device=idx.device)
        
        # 词嵌入 + 位置嵌入
        # self.tok(idx) 将字符 ID 转换为向量，形状 [B, T, embed_dim]
        # self.pos(pos) 将位置索引转换为向量，形状 [T, embed_dim]（会自动广播到 [B, T, embed_dim]）
        # 相加得到最终的嵌入表示，这样每个字符既有语义信息又有位置信息
        x = self.tok(idx) + self.pos(pos)
        
        # 通过所有 Transformer 块
        # 每个块都会对输入进行变换，提取更高级的特征
        for b in self.blocks:
            x = b(x)
        
        # 最终的层归一化
        x = self.ln(x)
        
        # 输出层：将隐藏状态映射为词汇表上的分数
        # 返回形状：[B, T, vocab_size]
        return self.head(x)

    @torch.no_grad()  # 生成时不需要计算梯度，节省内存和加速
    def generate(self, idx, steps=50, temperature=1.0, top_k=50):
        """
        生成文本：根据给定的起始序列，逐步生成后续字符
        
        参数:
            idx: 起始序列的字符 ID，形状为 [batch_size, sequence_length]
                 例如：[[1, 2, 3]] 表示从 "abc" 开始生成
            steps: 要生成多少个字符
            temperature: 温度参数，控制生成的随机性
                         - temperature=1.0：正常随机性
                         - temperature<1.0：更确定（倾向于高概率字符）
                         - temperature>1.0：更随机（概率分布更平滑）
            top_k: Top-K 采样，只从概率最高的 k 个字符中选择
                   None 表示不使用 Top-K（考虑所有字符）
        
        返回:
            完整的序列（包括输入的起始序列和生成的新字符）
            形状为 [batch_size, sequence_length + steps]
        
        工作原理（自回归生成）:
            1. 输入起始序列，预测下一个字符
            2. 将预测的字符添加到序列末尾
            3. 重复步骤 1-2，直到生成足够多的字符
        """
        # 设置模型为评估模式（虽然已经有 @torch.no_grad()，但显式设置更安全）
        self.eval()
        
        # 逐步生成 steps 个字符
        for _ in range(steps):
            # 如果序列长度超过 block_size，只保留最后 block_size 个字符
            # 因为模型只能处理 block_size 长度的序列
            # idx[:, -block_size:] 表示取最后 block_size 列
            idx = idx[:, -block_size:]
            
            # 前向传播，得到所有位置的预测分数
            # logits 形状：[B, T, vocab_size]
            logits = self(idx)
            
            # 只取最后一个位置的预测分数（因为我们只关心下一个字符）
            # logits[:, -1, :] 形状：[B, vocab_size]
            # 除以 temperature 进行温度缩放
            # max(temperature, 1e-6) 防止除零
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # Top-K 采样：只保留概率最高的 k 个字符
            if top_k is not None:
                # torch.topk 找到 top_k 个最大值
                # v 是这些最大值，形状 [B, top_k]
                v, _ = torch.topk(logits, top_k)
                
                # 将不在 top_k 中的字符的分数设为负无穷
                # logits < v[:, [-1]] 找出小于第 k 大值的所有位置
                # 这些位置的分数会被设为 -inf，经过 softmax 后概率为 0
                logits[logits < v[:, [-1]]] = -float("inf")
            
            # Softmax：将分数转换为概率分布
            # 每一行的概率加起来等于 1
            probs = F.softmax(logits, dim=-1)
            
            # 根据概率分布采样下一个字符
            # torch.multinomial 根据概率分布随机采样
            # 例如：如果 probs[0] = [0.1, 0.3, 0.6]，那么更可能采样到索引 2
            # 结果形状：[B, 1]
            next_id = torch.multinomial(probs, 1)
            
            # 将新生成的字符添加到序列末尾
            # torch.cat([idx, next_id], dim=1) 在序列维度上拼接
            # 例如：如果 idx 是 [[1,2,3]]，next_id 是 [[4]]，结果就是 [[1,2,3,4]]
            idx = torch.cat([idx, next_id], dim=1)
        
        return idx

# =============================
# 5. 训练模型或加载已训练的模型（A1）
# =============================
# 这部分会检查是否存在已训练的模型：
# - 如果存在，直接加载使用
# - 如果不存在，从头开始训练

# 创建模型实例并移动到指定设备（CPU 或 GPU）
model = TinyGPT().to(device)

# 创建优化器：AdamW 是一种改进的 Adam 优化算法
# 参数说明：
# - model.parameters(): 模型的所有可训练参数
# - lr: 学习率，控制参数更新的步长
# - weight_decay: 权重衰减（L2 正则化），防止过拟合
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 损失函数：交叉熵损失（Cross Entropy Loss）
# 这是分类任务常用的损失函数，衡量预测概率分布和真实标签之间的差异
# 例如：如果真实字符是 'h'（ID=0），但模型预测 'h' 的概率很低，损失就会很大
loss_fn = nn.CrossEntropyLoss()

# 检查模型文件是否存在
if os.path.exists(model_path):
    # 如果模型文件存在，加载已训练的模型参数
    # load_state_dict 将保存的参数加载到模型中
    # map_location=device 确保参数加载到正确的设备（CPU 或 GPU）
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ 已加载已训练模型")
else:
    # 如果模型文件不存在，从头开始训练
    print("🚀 开始训练模型")
    
    # 训练循环：重复 max_iters 次
    for step in range(max_iters):
        # 获取一个批次的训练数据
        # xb: 输入序列，形状 [batch_size, block_size]
        # yb: 目标序列（下一个字符），形状 [batch_size, block_size]
        xb, yb = get_batch("train")
        
        # 前向传播：模型根据输入预测下一个字符
        # logits 形状：[batch_size, block_size, vocab_size]
        logits = model(xb)
        
        # 计算损失
        # logits.view(-1, vocab_size) 将预测结果重塑为 [batch_size*block_size, vocab_size]
        # yb.view(-1) 将目标重塑为 [batch_size*block_size]
        # 这样每一行对应一个字符位置的预测和真实值
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        
        # 反向传播：计算梯度
        # zero_grad() 清零之前的梯度（因为 PyTorch 会累积梯度）
        opt.zero_grad()
        
        # backward() 计算损失对每个参数的梯度
        # 这些梯度告诉我们如何调整参数来减少损失
        loss.backward()
        
        # 更新参数：根据梯度调整模型参数
        # step() 使用优化器算法（AdamW）来更新参数
        opt.step()
        
        # 每隔 eval_interval 步评估一次模型性能
        if step % eval_interval == 0:
            # 计算训练集和验证集的平均损失
            losses = estimate_loss(model)
            
            # 打印当前训练进度和损失值
            # train loss: 训练集上的损失（越小越好）
            # val loss: 验证集上的损失（越小越好，如果比 train loss 大很多，可能过拟合）
            print(f"step {step}, train {losses['train']:.4f}, val {losses['val']:.4f}")

    # 训练完成后，保存模型参数到文件
    # state_dict() 获取模型的所有参数
    # 保存后下次可以直接加载，不需要重新训练
    torch.save(model.state_dict(), model_path)
    print("💾 模型已保存")

# =============================
# 6. 命令行聊天界面（A3）
# =============================
# 这是一个简单的交互式聊天界面，用户可以输入文本，模型会生成回复

print("\n💬 进入聊天模式（输入 q 退出）")

# 无限循环，持续接收用户输入并生成回复
while True:
    # 获取用户输入的文本
    user = input("你：")
    
    # 如果用户输入 "q"，退出聊天模式
    if user == "q":
        break
    
    # 将用户输入的文本编码为数字序列
    # encode(user) 将字符串转换为数字列表，例如 "hello" -> [0, 1, 2, 2, 3]
    # unsqueeze(0) 添加一个批次维度，形状从 [T] 变为 [1, T]
    # 因为模型期望输入是 [batch_size, sequence_length] 的形状
    # .to(device) 将数据移动到 GPU 或 CPU
    idx = encode(user).unsqueeze(0).to(device)
    
    # 使用模型生成回复
    # 参数说明：
    # - idx: 起始序列（用户的输入）
    # - 80: 生成 80 个字符
    # - temperature=0.9: 温度参数，稍微降低随机性，让生成更稳定
    # - top_k=40: Top-K 采样，只从概率最高的 40 个字符中选择
    # 返回的 out 形状是 [1, sequence_length + 80]，包含输入和生成的内容
    out = model.generate(idx, 80, temperature=0.9, top_k=40)
    
    # 将生成的数字序列解码为文本并打印
    # out[0] 取第一个批次（因为我们只有一个输入）
    # .tolist() 将张量转换为 Python 列表
    # decode() 将数字列表转换为字符串
    print("GPT：", decode(out[0].tolist()))
