# coding: UTF-8
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
from loguru import logger
from sklearn import metrics

logger.add("/home/hhl/yhq/log/BERT-4class-all-runtime-11.log")
datapath = "/home/hhl/yhq/linguistic_steganalysis/MOVIE/data/"
bert_path = "/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased/"
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("---------------------------------------------pre-process data-------------------------------------------------")
input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []  # 标签
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
            label.append([int(tmp[2])])

t_input_ids = []  # input char ids
t_input_types = []  # segment ids
t_input_masks = []  # attention mask
t_label = []  # 标签
test_input_ids = []  # input char ids
test_input_types = []  # segment ids
test_input_masks = []  # attention mask
test_label = []  # 标签


dataloader(datapath, "train", t_input_ids, t_input_types, t_input_masks, t_label)
dataloader(datapath, "dev", test_input_ids, test_input_types, test_input_masks, test_label)


input_ids_train = np.array(t_input_ids)
input_types_train = np.array(t_input_types)
input_masks_train = np.array(t_input_masks)
y_train = np.array(t_label)
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array(test_input_ids)
input_types_test = np.array(test_input_types)
input_masks_test = np.array(test_input_masks)
y_test = np.array(test_label)
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

print("-------------------------------------------define data loader--------------------------------------------------")
BATCH_SIZE = 32
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_types_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

"define BERT"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, 4)  # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False)  # 控制是否输出所有encoder层的结果
        out = F.dropout(pooled, p=0.5, training=self.training) # BERT的第一个和第二个输出取哪个来做分类
        out = self.fc(out)
        return out

print("----------------------------------------------实例化BERT模型-----------------------------------------------------")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)
print(model)

"定义优化器"

param_optimizer = list(model.named_parameters())  # 模型参数名字列表
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

NUM_EPOCHS = 100
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_loader) * NUM_EPOCHS
                     )

# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)   # 简单起见，可用这一行代码完事

"定义训练和测试函数"


def train(model, device, train_loader, optimizer, epoch):  # 训练模型
    model.train()
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1, x2, x3])  # 得到预测结果
        model.zero_grad()  # 梯度清零
        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss，交叉熵损失函数
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)] train Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader),
                                                                          loss.item()))  # 记得为loss.item()


def test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0
    acc = 0
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
    test_loss /= len(test_loader)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

"开始训练和测试"

best_acc = 0.0
PATH = '/home/hhl/yhq/model/best-bert-4class-all-pad32.pth'  # 定义模型保存路径
for epoch in range(1, NUM_EPOCHS+1):  # 3个epoch
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)  # 保存最优模型
    logger.info("acc is: {:.4f}, best acc is {:.4f}".format(acc, best_acc))









