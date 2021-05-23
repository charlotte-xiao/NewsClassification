import torch
from predic import evaluate_single, evaluate_multi
from utils import load_single_dataset, load_multi_dataset
# 模型TextRNN
import TextRNN as model


class run(object):
    def __init__(self):
        # TextRNN初始化
        # dataset--数据集根路径
        self.config = model.Config(dataset='Core/News')
        # 根据config进行Textmodel初始化
        self.Textmodel = model.Model(self.config).to(self.config.device)
        # 加载训练好的模型
        self.Textmodel.load_state_dict(torch.load(self.config.tain_path))

    def single(self, title, content, label="财经"):
        predic_data = load_single_dataset(self.config, [content, label, title])
        return evaluate_single(self.config, self.Textmodel, predic_data)

    def multi(self, path="Core/News/predic/test.xls"):
        # 词表，训练数据，验证数据，测试数据(词向量格式)
        predic_data = load_multi_dataset(self.config, path)
        evaluate_multi(self.config, self.Textmodel, predic_data)

