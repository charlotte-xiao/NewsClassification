import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif

# 权重初始化，默认xavier


def init_network(model, method="xavier", exclude="embedding", seed=123):#初始化LSTM神经网络的
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


def train(config, model, train_iter, dev_iter, test_iter):#传入的config，模型，三个集合
    start_time = time.time()#开始时间
    model.train()#开始训练
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)#优化器
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float("inf")  # 初始化为无穷大#最开始的损失率是无穷大的
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):#训练完一个循环之后为一个epoch
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):#把trains和labels从集合中取出来
            # print (trains[0].shape)
            outputs = model(trains)#把train输入
            model.zero_grad()#把梯度置为0
            loss = F.cross_entropy(outputs, labels)#计算损失率，自己本来的函数
            loss.backward()#把损失率进行一个反向传播
            optimizer.step()#加入优化器，迭代次数越多，优化会变得越来越慢
            if total_batch % 100 == 0:#训练完100个batch之后进行验证
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()#预测值
                train_acc = metrics.accuracy_score(true, predic)#训练的准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)#调用
                if dev_loss < dev_best_loss:#如果损失率比最好的一次损失率还要小的话对模型进行一次保存
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)#如果损失率比最好的一次损失率还要小的话对模型进行一次保存
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""#否则的话表示成空（小星星表示没有提高）
                time_dif = get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},\
                      Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(
                    msg.format(#迭代次数等信息使用total_batch表示
                        total_batch,
                        loss.item(),#各种信息
                        train_acc,
                        dev_loss,
                        dev_acc,
                        time_dif,
                        improve,
                    )
                )
                model.train()#进行完一个dataset（batchset）（一个dataset表示一个batch）（每次训练128个）之后进行一次训练
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过10000batch没下降，结束训练
                print("验证集loss超过10000batch没下降，结束训练...")#如果超过1w个还没下降，就结束训练
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)#进行验证


def test(config, model, test_iter):#进行验证
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True
    )#获得他的准确率以及他的损失率还有text report
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):#最后一次使用一个测试集来测试准确率
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
    # 打印出回交矩阵
    if test:#如果是没问题的，使用回交矩阵表示出来
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_list, digits=4
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
