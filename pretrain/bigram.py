import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.cuda.is_available())

# 超参数
batch_size = 32  # 批大小 序列并行大小
block_size = 8  # 最大内容长度
max_iters = 3000  # 最大训练迭代次数
eval_interval = 300  # 评估间隔
learning_rate = 1e-3  # 学习率
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200  # 评估次数

torch.manual_seed(1337)
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
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
@torch.no_grad() # 不进行反向传播
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


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 构造函数 创建一个词表大小为 65x65
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # 前向
    def forward(self, idx, targets=None):
        # 将输入的idx从 4x8 变成 4x8x65
        logits = self.token_embedding_table(idx)  # (B,T,C) B: Batch size 这里是4；T: context length 这里是8；C：vocab_size 65

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
            # 获取预测值
            logits, loss = self(idx)
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

# 优化
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets \每隔一段时间评估 train 和 val 集的损失
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}:train loss: {losses['train']:.4f} , val loss: {losses['val']:.4f}")

    # sample to a batch of data \样本转换为一批数据
    xb, yb = get_batch('train')

    # evaluate the loss \评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model \从模型生成
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_token=500)[0].tolist()))
