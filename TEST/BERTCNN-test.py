import logging
from pyexpat import model
from unittest import result
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import *
from typing import Dict, Iterable, Callable
from torch import NoneType, Tensor
from sklearn import metrics
from loguru import logger
from sklearn.metrics import precision_score, recall_score, accuracy_score

best_model_path = '/home/hhl/yhq/model/bert+cnn+movie+na_steg.pth'
datapath = "/home/hhl/yhq/linguistic_steganalysis/MOVIE/data/"
bert_path = '/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased/'
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("---------------------------------------------pre-process data-------------------------------------------------")

pad_size = 32  # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)

def dataloader(path, tp, input_ids, input_types, input_masks, label):
    with open(path + tp + ".txt", encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split("|||")
            x1 = tokenizer.tokenize(tmp[3])
            tokens = ["[CLS]"] + x1 + ["[SEP]"]
            # 得到input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)
            # 短则补齐，长则切断
            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)
            #         print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == pad_size
            label.append([int(tmp[0])])


test_input_ids = []  # input char ids
test_input_types = []  # segment ids
test_input_masks = []  # attention mask
test_label = []  # 标签


dataloader(datapath, "test", test_input_ids, test_input_types, test_input_masks, test_label)



input_ids_test = np.array(test_input_ids)
input_types_test = np.array(test_input_types)
input_masks_test = np.array(test_input_masks)
y_test = np.array(test_label)
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

print("----------------------------------------------data processe finised----------------------------------------------------")


print("-------------------------------------------define data loader--------------------------------------------------")
BATCH_SIZE = 32

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_types_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


print("-------------------------------------------define BERTmodel--------------------------------------------------")

class BERTCNNModel(nn.Module):
    def __init__(self):
        super(BERTCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(0.1)
        self.fc_cnn = nn.Linear(256 * len((2, 3, 4)), 2)
        self.activate1 = nn.Softmax(dim = 1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]
        encoder_out, _= self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False) 
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        out = self.activate1(out)
        return out

"----------------------------------------------定义训练,评估和测试函数----------------------------------------------"



def test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
    model.eval()
    acc = 0
    label_pre = []
    label = []
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3 , y= x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        pred.numpy()
        y = y.cpu()
        y.numpy()
        label_pre = np.append(label_pre, pred)
        label = np.append(label, y)
    acc = accuracy_score(label, label_pre)
    pre = precision_score(label, label_pre, average='macro')
    recall = recall_score(label, label_pre, average='macro')    
    mat = metrics.confusion_matrix(label, label_pre) 
    return acc, pre, recall ,mat

print("-------------------------------------------------begin test-------------------------------------------------")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTCNNModel().to(DEVICE)
model.load_state_dict(torch.load(best_model_path))
print(model)
acc, pre, recall, mat = test(model, DEVICE, test_loader)
print("acc is {}, pre is {}, recall is {}".format(acc, pre, recall))
print("mat is ")
print(mat)

