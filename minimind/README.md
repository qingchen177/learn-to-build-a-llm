## 环境

- 操作系统：Ubuntu22.04
- 内存：64GB
- 显卡：3090 24GB

Python 环境
拿的老环境接着用，后面缺什么装什么
```shell
conda create --name pytorch
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 数据
参考[minimind](https://github.com/jingyaogong/minimind)的`README.md`去下载就好了
我全部放在`/home/li/datasets/`里面

> 20241213：等全部流程做完会把数据集打包上传到百度云或者夸克