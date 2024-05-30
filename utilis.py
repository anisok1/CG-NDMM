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




