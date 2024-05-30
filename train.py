# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam


def train(config, model, train_iter, dev_iter, test_iter, device):
    model.train()
    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=2e-5,
                        warmup=0.05,
                        t_total=len(train_iter) * config.num_epoch
                        )
    model.train()
    for epoch in range(config.num_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epoch))
        for batch_idx, (x1, x2, x3, y1, y2, y3) in enumerate(train_iter):
            x1, x2, x3, y1, y2, y3 = x1.to(device), x2.to(device), x3.to(device), y1.to(device),y2.to(device),y3.to(device)
            if config.copper:
                out1, out2, out3 = model([x1, x2, x3])
                loss1 = F.cross_entropy(out1, y1.squeeze())
                loss2 = F.cross_entropy(out3, y3.squeeze())
                loss = loss1 + loss2
            else:
                out2 = model([x1, x2, x3])
                loss = F.cross_entropy(out2, y2.squeeze())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            dev_best_loss = float('inf')
            if (batch_idx + 1) % 100 == 0:  # 打印loss
                true = y2.cpu()
                pred = out2.max(-1, keepdim=True)[1]
                pred = pred.cpu()
                train_acc = metrics.accuracy_score(true, pred)
                dev_acc, dev_loss = evaluate(config, model, dev_iter, config.device)
                if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = 'have improve'
                        save_epoch = epoch
                else:
                    improve = 'no improve'
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, save_epoch{5}, {6}'
                print(msg.format(batch_idx, loss.item(), train_acc, dev_loss, dev_acc, save_epoch, improve))
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, config.device, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)



def evaluate(config, model, eval_iter, device, test=False):
    model.eval()
    label_pre = []
    label = []
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
    acc = metrics.accuracy_score(label, label_pre)
    # pre = metrics.precision_score(label, label_pre, average='macro')
    # recall = metrics.recall_score(label, label_pre, average='macro') 
    # mat = metrics.confusion_matrix(label, label_pre)
    if test:
        report = metrics.classification_report(label, label_pre, target_names=['cover', 'steg'], digits=4)
        confusion = metrics.confusion_matrix(label, label_pre)
        return acc, eval_loss, report, confusion
    return acc, eval_loss
    







