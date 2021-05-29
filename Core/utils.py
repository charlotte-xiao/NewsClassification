# coding: UTF-8
import torch
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import xlrd

# 未知字，padding符号
UNK, PAD = "<UNK>", "<PAD>"

# 文本数据处理方法定义(单行格式)


def load_single_dataset(config, data, pad_size=32):
    # 打开词表
    vocab = pkl.load(open(config.vocab_path, "rb"))
    content, label, title = data
    # 存储每一行内容
    words_line = []
    # 拼接title和content
    token = (lambda x: [y for y in x])(title + content)
    seq_len = len(token)
    # pad_size最长为32
    # 如果不足32位，扩展为32位
    if len(token) < pad_size:
        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
    # 否则进行截断为32位
    else:
        token = token[:pad_size]
        seq_len = pad_size
    # 将词转换为词表中对应的索引，构成1（line）*32（list）的矩阵
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    return (words_line, -1, seq_len)


# 文本数据处理方法定义(表格式)


def load_multi_dataset(config, path, method=""):
    # 打开词表
    vocab = pkl.load(open(config.vocab_path, "rb"))
    contents = []
    # pad_size默认为32
    pad_size = 32
    # excel处理格式
    if path.endswith("xlsx") or path.endswith("xls"):
        workbook = xlrd.open_workbook(path)
        # 读取第一个表
        table = workbook.sheet_by_index(0)
        for row_num in range(table.nrows):
            if row_num == 0:
                continue
            # 内容,标签,标题
            content, label, title = table.row_values(row_num)
            # 存储每一行内容
            words_line = []
            # 拼接title和content
            token = (lambda x: [y for y in x])(title + content)
            seq_len = len(token)
            # pad_size最长为32
            # 如果不足32位，扩展为32位
            if len(token) < pad_size:
                token.extend([vocab.get(PAD)] * (pad_size - len(token)))
            # 否则进行截断为32位
            else:
                token = token[:pad_size]
                seq_len = pad_size
            # 将词转换为词表中对应的索引，构成1（line）*32（list）的矩阵
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            if method == "predict":
                contents.append(words_line, -1, seq_len, [content, label, title])
            else:
                contents.append(
                    # 词内容，结果对应的索引，词长度
                    (words_line, config.class_list.index(label), seq_len, [content, label, title])
                )
    # cls处理格式
    else:
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                # 内容和标签用,分隔(半角)
                content, label, title = table.row_values(row_num)
                # 存储每一行内容
                words_line = []
                # 拼接title和content
                token = (lambda x: [y for y in x])(title + content)
                seq_len = len(token)
                # pad_size最长为32
                # 如果不足32位，扩展为32位
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                # 否则进行截断为32位
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
                # 将词转换为词表中对应的索引，构成1（line）*32（list）的矩阵
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                if method == "predict":
                    contents.append(words_line, -1, seq_len, [content, label, title])
                else:
                    contents.append(
                        # 词内容，结果对应的索引，词长度
                        (words_line, config.class_list.index(label), seq_len, [content, label, title])
                    )
    return contents


def get_time_dif(start_time):  # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
