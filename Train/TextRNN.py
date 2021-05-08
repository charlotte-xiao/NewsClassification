# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np

# TextRNN训练模型


class Config(object):

    """配置参数"""

    def __init__(self, dataset):
        # 训练集
        self.train_path = dataset + "/data/train.xls"
        # 验证集
        self.dev_path = dataset + "/data/test.xls"
        # 测试集
        self.test_path = dataset + "/data/test.xls"
        # 类别名单
        self.class_list = [
            x.strip()
            for x in open(
                dataset + "/data/class.txt", encoding="utf-8"
            ).readlines()
        ]
        # 词表
        self.vocab_path = dataset + "/data/vocab.pkl"
        # 模型训练结果保存路径
        self.save_path = dataset + "/save/TextRNN.ckpt"
        # 预训练词向量
        self.embedding_pretrained = torch.tensor(  # 向量格式是tensor
            np.load(dataset + "/data/embedding.npz")["embeddings"].astype(
                "float32"
            )
        )
        # 设备
        self.device = torch.device(
            # 如果有显卡，使用gpu训练；如果没有，使用cpu训练
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数 即9个分类（之后应该十个分类，第十个是其他）
        self.num_epochs = 20  # epoch数 一个轮回就是一个epoch
        self.batch_size = 128  # mini-batch大小 同时训练多少条数据
        self.pad_size = 32  # 每句话处理成的长度(短填长切) 提取每一个文段的前32个字，少于32自动填充
        self.learning_rate = 1e-3  # 学习率
        # 字向量维度, 这里使用了预训练词向量
        # 一个将离散变量转为连续向量表示的一个方式，类似于稀疏表节省空间
        self.embed = self.embedding_pretrained.size(1)
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数


"""模型"""


class Model(nn.Module):  # 进行初始化,使用现成的pytorch库
    def __init__(self, config):
        super(Model, self).__init__()
        # 预训练词向量,使用现成的搜狗或者腾讯的
        self.embedding = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=False
        )
        # 构建lstm层
        self.lstm = nn.LSTM(
            config.embed,  # 词向量
            config.hidden_size,  # lstm隐藏层大小
            config.num_layers,  # lstm层数
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout,
        )
        # bidirectional=True所以特征值为hidden_size*2<-->10分类
        # 全连接层：把一个矩阵映射到九个分类，变成线性，*2的原因是因为bidirectional=True设置的lstm结构是双向的，输出结果翻倍
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    # 前向传播
    def forward(self, x):
        x, _ = x

        out = self.embedding(x)  # 转换为词向量
        out, _ = self.lstm(out)  # 经过lstm得到输出结果和隐层状态
        out = self.fc(out[:, -1, :])  # 经过全连接层次得到结果
        return out
