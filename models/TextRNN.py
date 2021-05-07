# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""

    def __init__(self, dataset, embedding):
        # 模型名
        self.model_name = "TextRNN"
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
        self.vocab_path = dataset + "/data/vocab.pkl"#词表
        # 模型训练结果和日志路径
        self.save_path = dataset + "/save/" + self.model_name + ".ckpt"#保存路径
        self.log_path = dataset + "/log/" + self.model_name#训练日志
        # 预训练词向量
        self.embedding_pretrained = torch.tensor(#向量格式是tensor
            np.load(dataset + "/data/" + embedding)["embeddings"].astype(
                "float32"
            )
        )
        # 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"#如果有显卡，使用显卡的cuda核心跑；如果没有，使用cpu训练
        )

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数 就是9个分类（之后应该十个分类，第十个是其他）
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数 一个轮回就是一个epoch
        self.batch_size = 128  # mini-batch大小 同时训练多少条数据
        self.pad_size = 32  # 每句话处理成的长度(短填长切) 提取每一个文段的前32个字，少于32自动填充
        self.learning_rate = 1e-3  # 学习率
        # 字向量维度, 这里使用了预训练词向量，否则统一设置（可以为300）
        self.embed = self.embedding_pretrained.size(1) #一个将离散变量转为连续向量表示的一个方式，类似于稀疏表节省空间

        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数


"""模型"""


class Model(nn.Module):#进行初始化#现成的pytorch库
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False#预训练词向量，如果不为空，那么使用我们自己的
            )
        else:#如果是空的，使用pytorch提供的
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed, padding_idx=config.n_vocab - 1
            )
        self.lstm = nn.LSTM(#构建lstm层
            config.embed,#构建词向量
            config.hidden_size,#隐藏层的大小
            config.num_layers,#隐藏层的层数
            bidirectional=True,#默认参数们
            batch_first=True,
            dropout=config.dropout,
        )
        # bidirectional=True所以特征值为hidden_size*2<-->10分类
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)#全联接层 把一个矩阵映射到九个分类，变成线性的
#*2的原因是因为这个是双向的，所以乘以2
    def forward(self, x):#前向传播
        x, _ = x
        out = self.embedding(#构建词向量
            x
        )  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)  # 经过lstm得到输出结果和隐层状态
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
