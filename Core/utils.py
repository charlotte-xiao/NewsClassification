# coding: UTF-8
import pickle as pkl
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
    return (words_line, seq_len)


# 文本数据处理方法定义(表格式)


def load_multi_dataset(config, path):
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
            # 编号, 标签, 标题, 内容
            index, label, title, content = table.row_values(row_num)
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
            contents.append((words_line, seq_len, [index, title, content]))
    return contents
