# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert+rnn'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/saved_log/' + self.model_name + '.log'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                         # 类别数
        self.num_epoch = 100                                             # epoch数
        self.batch_size = 16                                             # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        self.copper = 0


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm1 = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc_rnn1 = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.activate1 = nn.Softmax(dim = 1)

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]
        encoder_out, _= self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False) 
        out1, _ = self.lstm1(encoder_out)
        out1 = self.dropout1(out1)
        out1 = self.fc_rnn1(out1[:, -1, :])  # 句子最后时刻的 hidden state
        out1 = self.activate1(out1)

        return out1
