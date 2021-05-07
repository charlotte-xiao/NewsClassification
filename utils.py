# coding: UTF-8
import torch
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import xlrd

UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号


def build_dataset(config):  # 语料表是按照字构建
    # char-level
    vocab = pkl.load(open(config.vocab_path, "rb"))
    print(f"Vocab size: {len(vocab)}")

    # 文本数据处理方法定义（需要定义不同格式的数据处理方法！！！）
    def load_dataset(path, pad_size=32):
        contents = []
        if path.endswith("xlsx") or path.endswith("xls"):
            workbook = xlrd.open_workbook(path)
            table = workbook.sheet_by_index(0) #读取第一个表
            for row_num in range(table.nrows):
                if row_num == 0:
                    continue
                # 内容,标签,标题
                content, label, title = table.row_values(row_num)#分别读取内容，标签，标题
                words_line = [] #先滞为0 ，之后使用索引表示
                token = (lambda x: [y for y in x])(title + content) #把title和content拼接在一起
                seq_len = len(token)
                # pad_size最长为32
               # if pad_size:
                if len(token) < pad_size:#如果不足32位，扩展为32位
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:#否则进行截断操作32位
                    token = token[:pad_size]
                    seq_len = pad_size
                # word to id
                for word in token:#找之前词表中的索引，构成1（line）*32（list）的矩阵
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append(
                    (words_line, config.class_list.index(label), seq_len)#通过之前的9分类，得出对应的索引，最大32
                )
        else:
            with open(path, "r", encoding="UTF-8") as f:
                for line in tqdm(f):
                    lin = line.strip()
                    if not lin:
                        continue
                    # 内容和标签用,分隔(半角)
                    content, label = lin.split(",")#使用逗号分隔词
                    words_line = []
                    token = (lambda x: [y for y in x])(content)
                    seq_len = len(token)
                    # pad_size最长为32
                    if pad_size:
                        if len(token) < pad_size:
                            token.extend(
                                [vocab.get(PAD)] * (pad_size - len(token))
                            )
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                    # word to id
                    for word in token:
                        words_line.append(vocab.get(word, vocab.get(UNK)))
                    contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)#导入训练集
    dev = load_dataset(config.dev_path, config.pad_size)#导入验证集/工作集
    test = load_dataset(config.test_path, config.pad_size)#导入测试集
    return vocab, train, dev, test


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


def build_iterator(dataset, config):#构建dataset，在pytorch中的每次迭代的小单元
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
