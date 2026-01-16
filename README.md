# Tiny ChatGPT

一个基于 Transformer 架构的小型 GPT 模型实现，支持文本生成和交互式聊天。本项目使用 PyTorch 实现，采用字符级别的 tokenization，适合学习和实验。

## ✨ 功能特性

- 🤖 **完整的 GPT 架构**：实现自注意力机制、Transformer 块和位置编码
- 📚 **自动数据加载**：从 `data/` 目录下的文本文件自动读取训练数据
- 💾 **模型持久化**：自动保存和加载训练好的模型
- 💬 **交互式聊天**：命令行界面，支持实时对话
- 🎯 **Top-k 采样**：支持温度控制和 Top-k 采样，生成更自然的文本

## 📋 项目结构

```
demo-mini-llm/
├── tiny_chatgpt.py      # 主程序文件
├── tiny_chatgpt.pth     # 训练好的模型权重
├── data/                # 训练数据目录
│   ├── data01.txt
│   └── data02.txt
└── README.md
```

## 🔧 环境要求

- Python 3.7+
- PyTorch 1.8+

安装依赖：

```bash
pip install torch
```

## 🚀 使用方法

### 1. 准备训练数据

将你的文本文件放在 `data/` 目录下（支持多个 `.txt` 文件）：

```bash
mkdir -p data
# 将你的文本文件放入 data/ 目录
```

### 2. 运行程序

```bash
python3 tiny_chatgpt.py
```

程序会自动执行以下步骤：

1. **数据加载**：读取 `data/` 目录下的所有 `.txt` 文件
2. **模型初始化**：创建 TinyGPT 模型
3. **训练或加载**：
   - 如果存在 `tiny_chatgpt.pth`，则加载已训练的模型
   - 否则开始训练模型（默认 4500 步）
4. **进入聊天模式**：训练完成后自动进入交互式聊天界面

### 3. 聊天示例

```
💬 进入聊天模式（输入 q 退出）
你：偌大一个村庄
GPT： 有一条老得走路不稳、耳聋眼花，陪伴他二十多年的老黄狗和一群鸡鸭鹅，以及满树聒噪的鸟。

老黄有二男一女三个孩子，都不在身边，远在北
你：我有个朋友对父母很孝顺
GPT： 亲关心备至，唯独一条，就是阻止母亲重新组建家庭。他的理由就是觉得面子上过不去。我说，人都是越老越孤独，物质需求肉眼可见地下降，吃穿
你：q
```

## 🏗️ 模型架构

### TinyGPT 模型结构

- **嵌入层**：字符嵌入 + 位置嵌入
- **Transformer 块**：4 层，每层包含：
  - 多头自注意力机制（4 个头）
  - 前馈神经网络（4 倍扩展）
  - Layer Normalization
  - Dropout（0.1）
- **输出层**：线性层映射到词汇表大小

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `embed_dim` | 128 | 嵌入维度 |
| `num_heads` | 4 | 注意力头数 |
| `num_layers` | 4 | Transformer 层数 |
| `block_size` | 64 | 上下文窗口大小 |
| `batch_size` | 16 | 批次大小 |
| `dropout` | 0.1 | Dropout 比率 |
| `learning_rate` | 3e-4 | 学习率 |
| `max_iters` | 4500 | 最大训练步数 |

## 📊 训练结果

训练过程中的损失变化：

```
step 0,    train 6.5717, val 6.6345
step 500,  train 0.7082, val 6.7230
step 1000, train 0.1145, val 7.8184
step 1500, train 0.0669, val 8.3761
step 2000, train 0.0532, val 8.8660
step 2500, train 0.0490, val 9.0130
step 3000, train 0.0453, val 9.3524
step 3500, train 0.0426, val 9.5855
step 4000, train 0.0415, val 9.6446
```

**观察**：
- 训练损失从 6.57 降至 0.04，模型成功学习
- 验证损失先上升后趋于稳定，可能存在轻微过拟合
- 模型在训练数据上表现良好，能够生成连贯的文本

## 🎛️ 生成参数

在 `generate()` 方法中可以调整以下参数：

- `steps`：生成的最大 token 数量（默认 80）
- `temperature`：温度参数，控制随机性（默认 0.9）
  - 较低值（0.1-0.5）：更确定性的输出
  - 较高值（1.0-2.0）：更随机的输出
- `top_k`：Top-k 采样，限制候选 token（默认 40）

## 🔍 代码说明

### 核心组件

1. **SelfAttention**：实现多头自注意力机制，包含因果掩码
2. **Block**：Transformer 块，包含注意力层和前馈层
3. **TinyGPT**：完整的 GPT 模型，包含嵌入、Transformer 块和输出层

### 数据流程

1. 文本 → 字符 tokenization → 整数序列
2. 随机采样 → 批次数据
3. 模型前向传播 → 预测下一个字符
4. 生成时使用 Top-k 采样 → 生成文本

## 📝 注意事项

- 模型使用字符级别的 tokenization，适合中文文本
- 训练数据量较小可能导致过拟合，建议增加数据量
- 模型会自动检测 CUDA 设备，如果有 GPU 会自动使用
- 首次运行会进行训练，训练时间取决于数据量和硬件配置

## 🛠️ 自定义配置

可以在代码中修改以下参数来调整模型：

```python
# 模型参数
embed_dim = 128      # 嵌入维度
num_heads = 4        # 注意力头数
num_layers = 4       # Transformer 层数
block_size = 64      # 上下文长度

# 训练参数
batch_size = 16      # 批次大小
max_iters = 4500     # 训练步数
lr = 3e-4            # 学习率
dropout = 0.1        # Dropout 比率
```

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

本项目参考了 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 实现思路。
