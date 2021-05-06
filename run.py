import time
import torch
import numpy as np
from train import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    dataset = 'News'  # 数据集
    # 词向量:搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz
    embedding = 'embedding_SougouNews.npz'
    # 模型  TextCNN, TextRNN,
    model_name = 'TextRNN'
    # 导入响应配置
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 创建随机种子，保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("开始构建..")

    # 构建词向量，将训练数据，验证数据，测试数据转换为词向量格式
    vocab, train_data, dev_data, test_data = build_dataset(config)
    # 使用Dataset构建数据
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # 打印出构建时间
    time_dif = get_time_dif(start_time)
    print("构建时间：", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
