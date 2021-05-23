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
    if label == "":
        return (words_line, -1, seq_len, [content, label, title])
    return (words_line,
            config.class_list.index(label),
            seq_len, [content, label, title]
            )

# 文本数据处理方法定义(表格式)


def load_multi_dataset(config, path, pad_size=32):
    # 打开词表
    vocab = pkl.load(open(config.vocab_path, "rb"))
    contents = []
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
            contents.append(
                # 词内容，结果对应的索引，词长度
                (words_line, config.class_list.index(label),
                 seq_len, [content, label, title])
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
                contents.append(
                    # 词内容，结果对应的索引，词长度
                    (words_line, config.class_list.index(label), seq_len)
                )
    return contents


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[
                self.index * self.batch_size: len(self.batches)
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index
                * self.batch_size: (self.index + 1)
                * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):  # 构建dataset，按照一个batch数据大小进行训练
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):  # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
