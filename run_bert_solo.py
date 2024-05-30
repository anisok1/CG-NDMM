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
import torch.optim as opt
import random
from itertools import accumulate
from importlib import import_module



def build_dataset(config):
    t_input_ids = []  # input char ids
    t_input_types = []  # segment ids
    t_input_masks = []  # attention mask
    t_label1 = [] 
    t_label2 = [] 
    t_label3 = []   # 标签
    eval_input_ids = []  # input char ids
    eval_input_types = []  # segment ids
    eval_input_masks = []  # attention mask
    eval_label1 = [] 
    eval_label2 = [] 
    eval_label3 = []
    test_input_ids = []  # input char ids
    test_input_types = []  # segment ids
    test_input_masks = []  # attention mask
    test_label1 = [] 
    test_label2 = [] 
    test_label3 = []   # 标签

    tokenizer = BertTokenizer(vocab_file='/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased/vocab.txt')
    def dataloader(path, input_ids, input_types, input_masks, label1, label2, label3, pad_size):
        with open(path, encoding='utf-8') as f:
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
                label1.append([int(tmp[0])])  #自然、生成
                label2.append([int(tmp[1])])  #隐写、非隐写
                label3.append([int(tmp[2])])  #4分类
    dataloader(config.train_path, t_input_ids, t_input_types, t_input_masks, t_label1, t_label2, t_label3, config.pad_size)
    dataloader(config.dev_path, eval_input_ids, eval_input_types, eval_input_masks, eval_label1, eval_label2, eval_label3, config.pad_size)
    dataloader(config.test_path, test_input_ids, test_input_types, test_input_masks, test_label1, test_label2, test_label3, config.pad_size)
    train = []
    eval = []
    test = []
    train.append(t_input_ids)
    train.append(t_input_types)
    train.append(t_input_masks)
    train.append(t_label1)
    train.append(t_label2)
    train.append(t_label3)
    eval.append(eval_input_ids)
    eval.append(eval_input_types)
    eval.append(eval_input_masks)
    eval.append(eval_label1)
    eval.append(eval_label2)
    eval.append(eval_label3)
    test.append(test_input_ids)
    test.append(test_input_types)
    test.append(test_input_masks)
    test.append(test_label1)
    test.append(test_label2)
    test.append(test_label3)
    return train, eval, test

def build_iterator(dataset, config):
    input_ids = np.array(dataset[0])
    input_types= np.array(dataset[1])
    input_masks = np.array(dataset[2])
    y1 = np.array(dataset[3])
    y2 = np.array(dataset[4])
    y3 = np.array(dataset[5])
    print(input_ids.shape, input_types.shape, input_masks.shape, y1.shape, y2.shape, y3.shape)
    data = TensorDataset(torch.LongTensor(input_ids),
                           torch.LongTensor(input_types),
                           torch.LongTensor(input_masks),
                           torch.LongTensor(y1), torch.LongTensor(y2), torch.LongTensor(y3))
    data_sampler = RandomSampler(data)
    data_iter = DataLoader(data, sampler=data_sampler, batch_size=config.batch_size)
    return data_iter

def train(config, model, device, features, label1, label2, label3, optimizer, epoch, train_loader):  # 训练模型
    model.train()
    for batch_idx in range(len(features)):
        feature = features[batch_idx].to(device)
        y1 = label1[batch_idx].to(device) #自然和生成标签
        y2 = label2[batch_idx].to(device) #隐写和非隐写标签
        y3 = label3[batch_idx].to(device) #4类精准标签
        if config.copper:
                out1, out2, out3 = model(feature)
                model.zero_grad()
                loss1 = F.cross_entropy(out1, y1.squeeze())
                loss2 = F.cross_entropy(out3, y3.squeeze())
                loss = loss1 + loss2
        else:
            out2 = model(feature)
            model.zero_grad()
            loss = F.cross_entropy(out2, y2.squeeze())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)], train Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(features),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader),
                                                                          loss.item()))  # 记得为loss.item()

def evaluate(config, model, device, features, label1, label2, label3, epoch):  # 测试模型, 得到测试集评估结果
    model.eval()
    eval_loss = 0.0
    acc = 0
    recall = 0
    pre = 0
    lb_pre = []
    lb = []
    for batch_idx in range(len(features)):
        feature = features[batch_idx].to(device)
        y1 = label1[batch_idx].to(device) #自然和生成标签
        y2 = label2[batch_idx].to(device) #隐写和非隐写标签
        y3 = label3[batch_idx].to(device) #4类精准标签
        if config.copper:
            with torch.no_grad():
                out1, out2, out3 = model(feature)
        else:
            with torch.no_grad():
                out2 = model(feature)
        eval_loss += F.cross_entropy(out2, y2.squeeze())
        pred = out2.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        pred.numpy()
        y = y2.cpu()
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

def test(config, model, device, features, label1, label2, label3):  # 测试模型, 得到测试集评估结果
    model.eval()
    acc = 0
    recall = 0
    pre = 0
    label_pre = []
    label = []
    for batch_idx in range(len(features)):
        feature = features[batch_idx].to(device)
        y1 = label1[batch_idx].to(device) #自然和生成标签
        y2 = label2[batch_idx].to(device) #隐写和非隐写标签
        y3 = label3[batch_idx].to(device) #4类精准标签
        if config.copper:
            with torch.no_grad():
                out1, out2, out3 = model(feature)
        else:
            with torch.no_grad():
                out2 = model(feature)
        pred = out2.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        pred.numpy()
        y = y2.cpu()
        y.numpy()
        label_pre = np.append(label_pre, pred)
        label = np.append(label, y)
    acc = accuracy_score(label, label_pre)
    pre = precision_score(label, label_pre, average='macro')
    recall = recall_score(label, label_pre, average='macro')    
    mat = metrics.confusion_matrix(label, label_pre) 
    return acc, pre, recall ,mat

bert_path = "/home/hhl/yhq/linguistic_steganalysis/bert-base-uncased/"
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



if __name__ == '__main__':
    dataset = '/home/hhl/yhq/linguistic_steganalysis/MOVIE'  # 数据集
    model_name = 'rnn*2'
    best_bert_path = '/home/hhl/yhq/model/best-bert-4class-all-pad32.pth' 

    x = import_module('Model.' + model_name)
    config = x.Config(dataset)
    logger.add(config.log_path)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    train_features = []
    train_label1 = []
    train_label2 = []
    train_label3 = []
    dev_features = []
    dev_label1 = []
    dev_label2 = []
    dev_label3 = []
    test_features = []
    test_label1 = []
    test_label2 = []
    test_label3 = []
    def train_hook(module, input1, output):
        train_features.append(input1[0].cpu())
        return None
    def eval_hook(module, input2, output):
        dev_features.append(input2[0].cpu())
        return None
    def test_hook(module, input3, output):
        test_features.append(input3[0].cpu())
        return None

    

    feature_net = BModel().to(config.device)
    feature_net.load_state_dict(torch.load(best_bert_path))  #加载最优bert模型
    feature_net.bert.pooler.register_forward_hook(train_hook)

    eval_feature_net = BModel().to(config.device)
    eval_feature_net.load_state_dict(torch.load(best_bert_path))  #加载最优bert模型
    eval_feature_net.bert.pooler.register_forward_hook(eval_hook)

    test_feature_net = BModel().to(config.device)
    test_feature_net.load_state_dict(torch.load(best_bert_path))  #加载最优bert模型
    test_feature_net.bert.pooler.register_forward_hook(test_hook)

    def BERT_feature_extractor(feature_net, data_loader, label1, label2, label3): 
        for batch_idx, (x1,x2,x3, y1, y2, y3) in tqdm(enumerate(data_loader)):
            x1, x2, x3, y1, y2, y3= x1.to(config.device), x2.to(config.device), x3.to(config.device), y1.to(config.device), y2.to(config.device), y3.to(config.device)
            label1.append(y1.cpu())
            label2.append(y2.cpu())
            label3.append(y3.cpu())
            with torch.no_grad():
                feature_net([x1,x2,x3])
        return None
    BERT_feature_extractor(feature_net, train_iter, train_label1, train_label2, train_label3)
    BERT_feature_extractor(eval_feature_net, dev_iter, dev_label1, dev_label2, dev_label3)
    BERT_feature_extractor(test_feature_net, test_iter, test_label1, test_label2, test_label3)

    model = x.Model(config).to(config.device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    NUM_EPOCHS = 100
    best_acc = 0
    for epoch in range(1, NUM_EPOCHS+1): 
        train(config, model, config.device, train_features, train_label1, train_label2, train_label3, optimizer, epoch, train_iter)
        acc =  evaluate(config, model, config.device, dev_features, dev_label1, dev_label2, dev_label3, epoch)
        if best_acc < acc:
            best_acc = acc
            tmp = epoch
            torch.save(model.state_dict(), config.save_path)  # 保存最优模型
        logger.info("acc is: {:.4f}, best acc is {:.4f}, best model get in {}".format(acc, best_acc, tmp))
    model.load_state_dict(torch.load(config.save_path))
    acc, pre, recall, mat = test(config, model, config.device, test_features, test_label1, test_label2, test_label3)
    print("acc is {}, pre is {}, recall is {}".format(acc, pre, recall))
    print("mat is ")
    print(mat)

