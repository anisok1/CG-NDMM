# coding: UTF-8
from operator import mod
import numpy as np
from importlib import import_module
from utilis import build_dataset, build_iterator
from train import train
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os







if __name__ == '__main__':
    dataset = '/home//linguistic_steganalysis/MOVIE'  # 数据集

    model_name = 'bert+rnn*2'
    x = import_module('Model.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  #

    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)


    # train
    model = x.Model(config).to(config.device)
    print(model)
    train(config, model, train_iter, dev_iter, test_iter, config.device)
