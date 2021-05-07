import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif

# 对LSTM神经网络进行权重初始化，默认xavier


def init_network(model, method="xavier", exclude="embedding", seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass

# 模型训练（配置，模型，训练集，验证集合，测试集合）


def train(config, model, train_iter, dev_iter, test_iter):
    # 记录开始时间
    start_time = time.time()
    # 开始训练
    model.train()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # 定义已经训练了多少个batch
    total_batch = 0
    # 初始化损失率为无穷大
    dev_best_loss = float("inf")
    last_improve = 0  # 记录上次验证集loss下降出的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):  # 训练完一个循环之后为一个epoch
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        # 把输入和数据从集合中取出来
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)  # 前向传播
            model.zero_grad()  # 梯度清0
            loss = F.cross_entropy(outputs, labels)  # 计算损失率
            loss.backward()  # 用损失率进行反向传播
            optimizer.step()  # 加入优化器，迭代次数越多，优化会变得越来越慢
            if total_batch % 100 == 0:  # 训练完100个batch之后进行验证
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()  # 取最大概率的为预测值
                train_acc = metrics.accuracy_score(true, predic)  # 训练的准确率
                # 验证集合的准确率和损失率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 如果损失率比最好的一次损失率还要小的话，对模型进行一次保存
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
                # 否则的话表示成空（小星星表示没有提高）
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "Batch数: {0:>6},  训练集损失率: {1:>5.2},  训练集准确率: {2:>6.2%},\
                       验证集损失率: {3:>5.2},  验证集损失率: {4:>6.2%},  消耗时间: {5} {6}"
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        dev_loss,
                        dev_acc,
                        time_dif,
                        improve,
                    )
                )
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集损失率超过1000batch没下降，结束训练
                print("验证集loss超过1000batch没下降，结束训练...")
                flag = True
                break
        if flag:
            break
    # 使用验证集检验
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_confusion = evaluate(
        config, model, test_iter, test=True
    )
    msg = "验证集损失率: {0:>5.2},  验证集准确率: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("混淆矩阵：")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("消耗时间:", time_dif)


def evaluate(config, model, data_iter, test=False):  # 验证函数
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    # 最后一次验证的时候,会生成混淆矩阵
    if test:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), confusion
    return acc, loss_total / len(data_iter)
