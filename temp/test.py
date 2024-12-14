import torch
import torch.nn as nn

# 创建一个嵌入层
# num_embeddings 表示嵌入矩阵的大小（即词汇表的大小）
# embedding_dim 表示每个嵌入向量的维度
embedding = nn.Embedding(32, 5)

# 假设我们有一个包含词汇索引的张量
input = torch.tensor([[1, 2, 5, 31], [3, 5, 7, 23]], dtype=torch.long)

# 通过嵌入层获取嵌入向量
embedded = embedding(input)

print(embedded.shape)
print(embedded)

B, T, C = embedded.shape
embedded_view = embedded.view(B * T, C)
print(embedded_view.shape)
print(embedded_view)

embedded_view1 = embedded.view(B * C, T)
print(embedded_view1.shape)
print(embedded_view1)

embedded_view2 = embedded.view(T * C, B)
print(embedded_view2.shape)
print(embedded_view2)

multinomial = torch.multinomial(torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]), num_samples=1)
print(multinomial)
