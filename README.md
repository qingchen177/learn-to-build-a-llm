# learn-to-build-a-llm

自己学习从零开始构建一个大语言模型的相关记录，比较杂

| 目录           | 介绍                                                         |
| -------------- | ------------------------------------------------------------ |
| AndrejKarpathy | 跟着大佬学，此目录下每个子文件夹存放对应其某个视频的学习记录 |
| minimind       | 学习大佬的开源项目：https://github.com/jingyaogong/minimind，此目录存放应该项目的学习和问题记录 |

## Andrej Karpathy

| 子文件夹  | 介绍                                                         |
| --------- | ------------------------------------------------------------ |
| SimpleGPT | 如何从零开始用python代码构建一个最小的nanoGPT（基于莎士比亚全文本训练集，对标GPT2） |

### SimpleGPT

#### 视频地址

- YouTube：https://www.youtube.com/watch?v=kCc8FmEb1nY&t=31s
- b站有中英字幕 ：https://www.bilibili.com/video/BV1CP41147Cw

### 目录结构

| file                     | what is it                                                                    |
|:-------------------------|:------------------------------------------------------------------------------|
| dataset/input.txt        | 莎士比亚文本                                                                        |
| bigram.py                | 实现一个最简单的GPT                                                                   |
| bigram_self_attention.py | 构建Transformer，加入self-attention多头自注意、残差链接、Dropout、前馈层FeedForward、层规范layer norm |
| 最简单的模型实现.ipynb           | 如何一步一步实现的notebook文件                                                           |
