import torch
import numpy as np
from train import train, init_network
from utils import build_dataset, build_iterator
# 模型TextRNN
import TextRNN as model

if __name__ == '__main__':
    # TextRNN初始化
    # dataset--数据集根路径
    config = model.Config(dataset='Train/News')

    # 创建随机种子，保证每次结果一样
    np.random.seed(1)  # 生成随机种子
    torch.manual_seed(1)  # 保证每次训练结果一样
    torch.cuda.manual_seed_all(1)  # 和上面一个意思，开启cuda 的情况下保证结果一致
    torch.backends.cudnn.deterministic = True  # 需要开启这个功能才能保证一致

    print("开始构建..")
    # 词表，训练数据，验证数据，测试数据(词向量格式)
    vocab, train_data, dev_data, test_data = build_dataset(config)
    # 使用Dataset构建数据
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    print("构建完成..")

    print("开始训练..")
    # 变成词表的大小（依据自己的词表）
    config.n_vocab = len(vocab)
    # 根据config进行Textmodel初始化
    Textmodel = model.Model(config).to(config.device)
    init_network(Textmodel)  # 初始化函数
    print(Textmodel.parameters)  # 打印结构
    train(config, Textmodel, train_iter, dev_iter, test_iter)
    print("训练完成..")
