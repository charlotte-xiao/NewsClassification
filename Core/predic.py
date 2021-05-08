import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif


def evaluate_multi(config, model, predic_data, flag=True):  # 预测函数
    start_time = time.time()
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        # for texts, labels in predic_data:用迭代器
        for datas in predic_data:
            texts = (
                torch.LongTensor([datas[0]]).to(config.device),
                torch.LongTensor([datas[2]]).to(config.device)
                )
            labels = torch.LongTensor([datas[1]]).to(config.device)
            content, label, title = datas[3]
            # print("实际值："+label)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # print("预测值"+config.class_list[predic.tolist()[0]])
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    # 最后一次验证的时候,会生成混淆矩阵
    if flag:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print("混淆矩阵：")
        print(confusion)
    msg = "验证集损失率: {0:>5.2},  验证集准确率: {1:>6.2%}"
    print(msg.format(loss_total, acc))
    time_dif = get_time_dif(start_time)
    print("消耗时间:", time_dif)


def evaluate_single(config, model, data):  # 预测函数
    model.eval()
    with torch.no_grad():
        texts = (
            torch.LongTensor([data[0]]).to(config.device),
            torch.LongTensor([data[2]]).to(config.device)
            )
        labels = torch.LongTensor([data[1]]).to(config.device)
        content, label, title = data[3]
        print("实际值："+label)
        outputs = model(texts)
        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        print("预测值:"+config.class_list[predic.tolist()[0]])


