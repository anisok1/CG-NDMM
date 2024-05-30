from cProfile import label
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
from torch import Tensor
from sklearn import metrics
from loguru import logger
from sklearn.metrics import precision_score, recall_score, accuracy_score


logger.add("/home/hhl/yhq/log/CNN-runtime-all-2.log")
datapath = "/home/hhl/yhq/data/4-class-data/"
bert_path = "/home/hhl/yhq/bert/bert-base-uncased/"
best_bert_path = "/home/hhl/yhq/model/best-bert-4class.pth"
tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("---------------------------------------------pre-process data-------------------------------------------------")

pad_size = 38  # 也称为 max_len (前期统计分析，文本长度平均值为58)

def dataloader(path, tp, input_ids, input_types, input_masks, label):
    with open(path + tp + ".txt", encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split("|||")
            x1 = tokenizer.tokenize(tmp[1])
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
eval_label = []  # 标签


dataloader(datapath, "train", t_input_ids, t_input_types, t_input_masks, t_label)
# dataloader(datapath, "eval_copy", t_input_ids, t_input_types, t_input_masks, t_label)
dataloader(datapath, "eval", eval_input_ids, eval_input_types, eval_input_masks, eval_label)


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



print("----------------------------------------------data processe finised----------------------------------------------------")


print("-----------------------------------------------define data loader------------------------------------------------------")
BATCH_SIZE = 32
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




print("-------------------------------------------define BERTmodel--------------------------------------------------")

class BModel(nn.Module):
    def __init__(self):
        super(BModel, self).__init__()
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
        out = F.dropout(pooled, p=0.5, training=self.training)
        out = self.fc(out)
        out = F.softmax(out, dim = 1)
        return out

print("-------------------------------------------define BERT feature extractor--------------------------------------------------")







t_features = []
e_features = []
t_label = []
e_label = []
def train_hook(module, input1, output):
    t_features.append(input1[0].cpu())
    return None
def eval_hook(module, input2, output):
    e_features.append(input2[0].cpu())
    return None



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature_net = BModel().to(DEVICE)
feature_net.load_state_dict(torch.load(best_bert_path))  #加载最优bert模型
feature_net.bert.pooler.register_forward_hook(train_hook)



eval_feature_net = BModel().to(DEVICE)
eval_feature_net.load_state_dict(torch.load(best_bert_path))  #加载最优bert模型
eval_feature_net.bert.pooler.register_forward_hook(eval_hook)



def BERT_feature_extractor(feature_net, data_loader, label): 
    for batch_idx, (x1,x2,x3, y) in tqdm(enumerate(data_loader)):
        x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
        label.append(y.cpu())
        with torch.no_grad():
            feature_net([x1,x2,x3])
    return None

BERT_feature_extractor(feature_net, train_loader, t_label)
BERT_feature_extractor(eval_feature_net, eval_loader, e_label)
print(len(t_features), len(t_label))
print(len(e_features), len(e_label))
print("---------------------------------------feature extraction complete---------------------------------------------")




print("---------------------------------------------------defien CNNmodle-----------------------------------------")

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(0.8)
        self.fc_cnn = nn.Linear(256 * len((2, 3, 4)), 4)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        out = F.softmax(out, dim = 1)
        return out

print("----------------------------------------------实例化CNN模型-----------------------------------------------------")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(DEVICE)
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


"----------------------------------------------定义训练,评估和测试函数----------------------------------------------"



def train(model, device, features, lable, optimizer, epoch):  # 训练模型
    model.train()
    for batch_idx in range(len(features)):
        feature = features[batch_idx].to(device)
        y = lable[batch_idx].to(device)
        y_pred = model(feature)# 得到预测结果
        model.zero_grad()
        loss = F.cross_entropy(y_pred, y.squeeze()) # 得到loss，交叉熵损失函数
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 100 == 0: # 打印loss
            logger.info('-------------------------------------------print loss----------------------------------------------------')
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)] train Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(features),
                                                                            len(features),
                                                                            100. * batch_idx / len(features),
                                                                            loss.item()))  # 记得为loss.item()
            logger.info('---------------------------------------------------------------------------------------------------------')                                                            
    return None




def eval(model, device, features, label, epoch):  # 测试模型, 得到测试集评估结果
    model.eval()
    eval_loss = 0.0
    acc = 0
    recall = 0
    pre = 0
    lb_pre = []
    lb = []
    for batch_idx in range(len(features)):
        feature = features[batch_idx].to(device)
        y = label[batch_idx].to(device)
        with torch.no_grad():
           y_ = model(feature)
        eval_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        pred.numpy()
        y = y.cpu()
        y.numpy()
        lb_pre = np.append(lb_pre, pred)
        lb = np.append(lb, y)
    acc = accuracy_score(lb, lb_pre)
    pre = precision_score(lb, lb_pre, average='macro')
    recall = recall_score(lb, lb_pre, average='macro') 
    eval_loss /= len(features)
    mat = metrics.confusion_matrix(lb, lb_pre)
    logger.info("--------------------------------------------Train Epoch: {}-----------------------------------------------------".format(epoch))
    logger.info('Eval set: Average loss: {:.4f}, Accuracy: ({:.2f}), Pre: ({:.2f}), Recall: ({:.2f})'.format(
        eval_loss, acc, pre, recall
        ))
    logger.info(mat)
    return acc



best_acc = 0.0
PATH = "/home/hhl/yhq/model/best-CNN-all-4bert-1.ckpt"


print("-------------------------------------------------bert feature extract finished-------------------------------------------------")


for epoch in range(1000):  # 3个epoch    
    temp = 0
    train(model, DEVICE, t_features, t_label, optimizer, epoch)
    acc = eval(model, DEVICE, e_features, e_label, epoch)
    if best_acc < acc:
        best_acc = acc
        tmp = epoch
        torch.save(model.state_dict(), PATH)  # 保存最优模型
    logger.info(" acc is: {:.4f}, best acc is {:.4f}, best acc get in Epoch {}".format(acc, best_acc, tmp))

