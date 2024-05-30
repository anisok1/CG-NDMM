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

def train(config, model, device, train_iter, optimizer, epoch):  # 训练模型
    model.train()
    for batch_idx, (x1, x2, x3, y1, y2, y3) in enumerate(train_iter):
        x1, x2, x3, y1, y2, y3 = x1.to(device), x2.to(device), x3.to(device), y1.to(device),y2.to(device),y3.to(device)
        if config.copper:
                out1, out2, out3 = model([x1, x2, x3])
                model.zero_grad()
                loss1 = F.cross_entropy(out1, y1.squeeze())
                loss2 = F.cross_entropy(out3, y3.squeeze())
                loss3 = F.cross_entropy(out2, y2.squeeze())
                loss = loss1 + loss2 +loss3
        else:
            out = model([x1, x2, x3])
            model.zero_grad()
            loss = F.cross_entropy(out, y2.squeeze())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)], train Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                          len(train_iter.dataset),
                                                                          100. * batch_idx / len(train_iter),
                                                                          loss.item()))  # 记得为loss.item(

def evaluate(config, model, device, eval_iter, epoch):  # 测试模型, 得到测试集评估结果
    model.eval()
    label_pre = []
    label = []
    acc = 0
    eval_loss = 0
    for batch_idx, (x1, x2, x3, y1, y2, y3) in enumerate(eval_iter):
        x1, x2, x3, y1, y2, y3 = x1.to(device), x2.to(device), x3.to(device), y1.to(device),y2.to(device),y3.to(device)
        if config.copper:
            with torch.no_grad():
                out1, out2, out3 = model([x1, x2, x3])
        else:
            with torch.no_grad():
                out2 = model([x1, x2, x3])
        eval_loss += F.cross_entropy(out2, y2.squeeze())
        pred = out2.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        pred = pred.cpu()
        y = y2.cpu()
        pred.numpy()
        y.numpy()
        label_pre = np.append(label_pre, pred)
        label = np.append(label, y)
    acc = accuracy_score(label, label_pre)
    pre = precision_score(label, label_pre, average='macro')
    recall = recall_score(label, label_pre, average='macro') 
    eval_loss /= len(eval_iter)
    mat = metrics.confusion_matrix(label, label_pre)
    logger.info("--------------------------------------------Train Epoch: {}-----------------------------------------------------".format(epoch))
    logger.info('Eval set: Average loss: {:.4f}, Accuracy: ({:.2f}), Pre: ({:.2f}), Recall: ({:.2f})'.format(
        eval_loss, acc, pre, recall
        ))
    logger.info(mat)
    return acc

def test(config, model, device, test_iter):  # 测试模型, 得到测试集评估结果
    model.eval()
    acc = 0
    label_pre = []
    label = []
    for batch_idx, (x1, x2, x3, y1, y2, y3) in enumerate(test_iter):
        x1, x2, x3 , y1, y2, y3 = x1.to(device), x2.to(device), x3.to(device), y1.to(device), y2.to(device), y3.to(device)
        if config.copper:
            with torch.no_grad():
                out1, out2, out3 = model([x1, x2, x3])
        else:
            with torch.no_grad():
                out2 = model([x1, x2, x3])
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


if __name__ == '__main__':
    dataset = '/home/hhl/yhq/linguistic_steganalysis/MOVIE'  # 数据集
    model_name = 'bert+rnn*2'

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

    model = x.Model(config).to(config.device)
    print(model)

    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    NUM_EPOCHS = 100
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=2e-5,
                        warmup=0.05,
                        t_total=len(train_iter) * NUM_EPOCHS
                        )
    best_acc = 0
    for epoch in range(1, NUM_EPOCHS+1): 
        train(config, model, config.device, train_iter, optimizer, epoch)
        acc =  evaluate(config, model, config.device, dev_iter, epoch)
        if best_acc < acc:
            best_acc = acc
            tmp = epoch
            torch.save(model.state_dict(), config.save_path)  # 保存最优模型
        logger.info("acc is: {:.4f}, best acc is {:.4f}, best model get in {}".format(acc, best_acc, tmp))
    model.load_state_dict(torch.load(config.save_path))
    acc, pre, recall, mat = test(config, model, config.device, test_iter)
    logger.info("acc is {}, pre is {}, recall is {}".format(acc, pre, recall))
    logger.info("mat is ")
    logger.info(mat)

