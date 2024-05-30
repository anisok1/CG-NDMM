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
from sklearn.metrics import precision_score, recall_score, accuracy_score

logger.add("/home/hhl/yhq/bert+cnn+movie+na_gen.log")
datapath = "/home/hhl/yhq/linguistic_steganalysis/MOVIE/data/"
bert_path ='/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased/'
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
            label.append([int(tmp[0])])

t_input_ids = []  # input char ids
t_input_types = []  # segment ids
t_input_masks = []  # attention mask
t_label = []  # 标签
eval_input_ids = []  # input char ids
eval_input_types = []  # segment ids
eval_input_masks = []  # attention mask
eval_label = [] 


dataloader(datapath, "train", t_input_ids, t_input_types, t_input_masks, t_label)
dataloader(datapath, "dev", eval_input_ids, eval_input_types, eval_input_masks, eval_label)


input_ids_train = np.array(t_input_ids)
input_types_train = np.array(t_input_types)
input_masks_train = np.array(t_input_masks)
y_train = np.array(t_label)
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_eval = np.array(eval_input_ids)
input_types_eval = np.array(eval_input_types)
input_masks_eval = np.array(eval_input_masks)
y_eval = np.array(eval_label)
print(input_ids_eval.shape, input_types_eval.shape, input_masks_eval.shape, y_eval.shape)

print("-------------------------------------------define data loader--------------------------------------------------")
BATCH_SIZE = 16
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

eval_data = TensorDataset(torch.LongTensor(input_ids_eval),
                          torch.LongTensor(input_types_eval),
                          torch.LongTensor(input_masks_eval),
                          torch.LongTensor(y_eval))
eval_sampler = SequentialSampler(eval_data)
eval_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)

"define BERT"


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

print("----------------------------------------------实例化BERT模型-----------------------------------------------------")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTCNNModel().to(DEVICE)
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
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)], train Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader),
                                                                          loss.item()))  # 记得为loss.item()


def test(model, device, test_loader, epoch):  # 测试模型, 得到测试集评估结果
    model.eval()
    label_pre = []
    label = []
    acc = 0
    eval_loss = 0
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])
        eval_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        y = y.cpu()
        pred.numpy()
        y.numpy()
        label_pre = np.append(label_pre, pred)
        label = np.append(label, y)
    acc = accuracy_score(label, label_pre)
    pre = precision_score(label, label_pre, average='macro')
    recall = recall_score(label, label_pre, average='macro') 
    eval_loss /= len(eval_loader)
    mat = metrics.confusion_matrix(label, label_pre)
    logger.info("--------------------------------------------Train Epoch: {}-----------------------------------------------------".format(epoch))
    logger.info('Eval set: Average loss: {:.4f}, Accuracy: ({:.2f}), Pre: ({:.2f}), Recall: ({:.2f})'.format(
        eval_loss, acc, pre, recall
        ))
    logger.info(mat)
    return acc


"开始训练和测试"

tmp = 0
best_acc = 0
PATH = '/home/hhl/yhq/model/bert+cnn+movie+na_steg.pth'  # 定义模型保存路径
for epoch in range(1, NUM_EPOCHS+1): 
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, eval_loader, epoch)
    if best_acc < acc:
        best_acc = acc
        tmp = epoch
        torch.save(model.state_dict(), PATH)  # 保存最优模型
    logger.info("acc is: {:.4f}, best acc is {:.4f}, best model get in {}".format(acc, best_acc, tmp))









