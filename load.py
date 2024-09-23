from datetime import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

print(torch.cuda.is_available())

start_time = time.time()

# 超参数
batch_size = 64  # 批大小 序列并行大小
block_size = 256  # 最大内容长度
max_iters = 5000  # 最大训练迭代次数
eval_interval = 500  # 评估间隔
learning_rate = 3e-4  # 学习率
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200  # 评估次数
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)
with open('pretrain/dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 统计字符数，作为我们的词本
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 实现一个encode、decode
# mapping 映射，最简单的tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder 输入字符串，转化成数字列表
decode = lambda l: ''.join([itos[i] for i in l])  # decoder 输入数字列表，转化成字符串

# 数据集划分
data = torch.tensor(encode(text), dtype=torch.long)
# 训练集和测试集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# 数据加载
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# 评估损失
@torch.no_grad()  # 不进行反向传播
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # Q K V
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    # 前向
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute attention scores ("affinities") 计算注意力分数 亲和力
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,C) @ (B,C,T) -->(B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接后的处理
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        # return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity 一个简单的线性层，后跟一个非线性"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block:  communication followed by computation 通信后计算"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension 嵌入维度 , n_head: the number of heads we'd like 我们想要的头数
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # add & norm 预规范，层规范
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # 位置
        # self.sa_head = Head(n_embd)  # 自注意力
        # self.sa_head = MultiHeadAttention(4, n_embd // 4)  # 自注意力
        # self.ffwd = FeedForward(n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size)

    # 前向
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 将输入的idx从 4x8 变成 4x8x65
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) B: Batch size 这里是4；T: context length 这里是8；C：n_emb 32
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # apply one head of self-attention (B,T,C)
        # x = self.sa_head(x)
        # x = self.ffwd(x)

        # block
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B,T,C) B: Batch size 这里是4；T: context length 这里是8；C：vocab_size 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # 交叉熵
            loss = F.cross_entropy(logits, targets)  # 这里不改变张量结构会报错

        return logits, loss

    def generate(self, idx, max_new_token):
        # idx 是当前上下文中大小为 (B,T) 的索引数组
        for _ in range(max_new_token):
            # crop idx to the last block_size tokens 将 IDX 裁剪为最后 block_size 个令牌
            idx_cond = idx[:, -block_size:]
            # 获取预测值
            logits, loss = self(idx_cond)
            # 专注于最后一步
            logits = logits[:, -1, :]  # 变成(B,C)
            # 加入softmax函数得到概率，dim=-1取最后一个维度来计算也就是C，65
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution 从分布中抽样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# 打印模型的参数数量
print(sum(p.numel() for p in m.parameters()), "参数")


# 加载模型
model = torch.load('D:/work/models/nano/nano-pretrain.model')

# generate from the model \从模型生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_token=500)[0].tolist()))


