<div align="center">

# learn-to-build-a-llm
自己学习从零开始构建一个大语言模型的相关记录，比较杂
</div>

---

# 目录

- <a href="#Andrej Karpathy">Andrej Karpathy</a>：跟着大佬学，此目录下每个子文件夹存放对应其某个视频的学习记录
- <a href="#minimind">minimind</a>：学习大佬的开源项目：https://github.com/jingyaogong/minimind
此目录存放应该项目的学习和问题记录

---

## Andrej Karpathy

| 子文件夹  | 介绍                                                         |
| --------- | ------------------------------------------------------------ |
| SimpleGPT | 如何从零开始用python代码构建一个最小的nanoGPT（基于莎士比亚全文本训练集，对标GPT2） |

- ### SimpleGPT

  #### 视频地址

  - YouTube：https://www.youtube.com/watch?v=kCc8FmEb1nY&t=31s
  - b站有中英字幕 ：https://www.bilibili.com/video/BV1CP41147Cw

  ### 目录结构

  | file                     | what is it                                                   |
  | :----------------------- | :----------------------------------------------------------- |
  | dataset/input.txt        | 莎士比亚文本                                                 |
  | bigram.py                | 实现一个最简单的GPT                                          |
  | bigram_self_attention.py | 构建Transformer，加入self-attention多头自注意、残差链接、Dropout、前馈层FeedForward、层规范layer norm |
  | 最简单的模型实现.ipynb   | 如何一步一步实现的notebook文件                               |

---

## minimind

| 子文件夹 | 介绍                 |
| -------- | -------------------- |
| model    | 训练后的模型存放位置 |
| train    | 训练代码存放位置     |

- model：

  | file/folder        | sub_file              | what is it                  |
  | ------------------ | --------------------- | --------------------------- |
  | minimind_tokenizer |                       | 分词器（tokenizer）存放目录 |
  |                    | merges.txt            | -                           |
  |                    | tokenizer.json        | -                           |
  |                    | tokenizer_config.json | 分词器配置文件              |
  |                    | vocab.json            | 分词器词表                  |

- train：

  | file/folder | sub_file                    | what is it                                                   |
  | ----------- | --------------------------- | ------------------------------------------------------------ |
  | 1-tokenizer |                             | 训练分词器代码存放目录                                       |
  |             | train_tokenizer.py          | 训练tokenizer代码                                            |
  |             | train_tokenizer_steps.ipynb | jupyter notebook一步步操作（推荐一开始用这个可以自己边跑边把不懂的直接打印出来看） |

  

