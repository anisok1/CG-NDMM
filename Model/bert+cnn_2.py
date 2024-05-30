# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert+cnn*2'
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
        self.copper = 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc_cnn1 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.activate1 = nn.Softmax(dim = 1)

        self.dropout2 = nn.Dropout(config.dropout)
        self.fc_cnn2 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.activate2 = nn.Softmax(dim = 1)



    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
        
    def cheng(self, x, y):
        a = torch.split(x, 1, dim = 1)
        b = torch.split(y, 1, dim = 1)
        la00 = a[0]*b[0]
        la01 = a[0]*b[1]
        la10 = a[1]*b[0]
        la11 = a[1]*b[1]
        result = torch.cat([la00, la01, la10, la11], 1)
        return result

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]
        encoder_out, _= self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False) 
        out1 = encoder_out.unsqueeze(1)
        out1 = torch.cat([self.conv_and_pool(out1, conv) for conv in self.convs1], 1)
        out1 = self.dropout1(out1)
        out1 = self.fc_cnn1(out1)
        out1 = self.activate1(out1)

        out2 = encoder_out.unsqueeze(1)
        out2 = torch.cat([self.conv_and_pool(out2, conv) for conv in self.convs1], 1)
        out2 = self.dropout2(out2)
        out2 = self.fc_cnn2(out2)
        out2 = self.activate2(out2)

        out3 = self.cheng(out1, out2)
        return out1, out2, out3
